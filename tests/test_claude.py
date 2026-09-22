"""Phase 2 tests: Claude generation + local HF embeddings in app/rag_chain.py.

Run: python3 -m unittest discover -s tests -v   (from project root)
Third-party deps are stubbed via tests/_stubs.py; no network, no PDF load.
"""

import glob
import importlib
import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _stubs import install_stubs  # noqa: E402

install_stubs()

from app import rag_chain  # noqa: E402

REFUSAL_ANSWER = (
    "Je ne peux pas répondre à cette question. Consultez un professionnel du droit."
)


class FakeDoc:
    def __init__(self, text, page=None):
        self.page_content = text
        self.metadata = {} if page is None else {"page": page}


def block(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def fake_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def fake_claude(response):
    client = MagicMock(name="Anthropic()")
    client.beta.messages.create.return_value = response
    return client


# ---------- constants ----------
class TestConstants(unittest.TestCase):
    def test_values(self):
        self.assertEqual(rag_chain.PDF_PATH, "Burundi_Code_2017_penal.pdf")
        self.assertEqual(rag_chain.PERSIST_DIR, "rag")
        self.assertEqual(
            rag_chain.EMBEDDING_MODEL,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.assertEqual(rag_chain.CLAUDE_MODEL, "claude-opus-5-5")
        self.assertEqual(rag_chain.CLAUDE_EFFORT, "medium")
        self.assertEqual(rag_chain.MAX_TOKENS, 16000)
        self.assertEqual(rag_chain.IN_SCOPE_THRESHOLD, 0.5)
        self.assertEqual(rag_chain.GROUNDED_THRESHOLD, 0.7)
        self.assertEqual(rag_chain.REFUSAL_ANSWER, REFUSAL_ANSWER)

    def test_system_prompt(self):
        self.assertIsInstance(rag_chain.SYSTEM_PROMPT, str)
        self.assertNotIn("{context}", rag_chain.SYSTEM_PROMPT)
        self.assertIn("<extraits>", rag_chain.SYSTEM_PROMPT)

    def test_old_chain_removed(self):
        self.assertFalse(hasattr(rag_chain, "question_answer_chain"))
        self.assertFalse(hasattr(rag_chain, "llm"))


# ---------- get_claude ----------
class TestGetClaude(unittest.TestCase):
    def test_no_client_at_import(self):
        self.assertIsNone(rag_chain._claude)

    def test_lazy_and_cached(self):
        sentinel = MagicMock(name="client-instance")
        ctor = MagicMock(return_value=sentinel)
        with (
            patch.object(rag_chain, "_claude", None),
            patch.object(rag_chain.anthropic, "Anthropic", ctor),
        ):
            ctor.assert_not_called()
            c1 = rag_chain.get_claude()
            c2 = rag_chain.get_claude()
        self.assertIs(c1, sentinel)
        self.assertIs(c2, sentinel)
        self.assertEqual(ctor.call_count, 1)
        ctor.assert_called_once_with()


# ---------- format_context ----------
class TestFormatContext(unittest.TestCase):
    def test_pages_and_tags(self):
        docs = [FakeDoc("Art. 1 texte", page=12), FakeDoc("Art. 2 texte", page=40)]
        self.assertEqual(
            rag_chain.format_context(docs),
            "<extraits>\n[page 12]\nArt. 1 texte\n\n[page 40]\nArt. 2 texte\n</extraits>",
        )

    def test_missing_page_is_none(self):
        self.assertEqual(
            rag_chain.format_context([FakeDoc("x")]),
            "<extraits>\n[page None]\nx\n</extraits>",
        )

    def test_empty(self):
        self.assertEqual(rag_chain.format_context([]), "<extraits>\n\n</extraits>")


# ---------- generate ----------
class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.docs = [FakeDoc("Art. 1 texte", page=12), FakeDoc("Art. 2 texte", page=40)]

    def _run(self, response):
        client = fake_claude(response)
        with patch.object(rag_chain, "get_claude", return_value=client):
            out = rag_chain.generate("Quelle peine ?", self.docs)
        return out, client

    def test_request_kwargs(self):
        _, client = self._run(fake_response([block("text", text="ok")]))
        client.beta.messages.create.assert_called_once()
        args, kwargs = client.beta.messages.create.call_args
        self.assertEqual(args, ())
        self.assertEqual(
            kwargs,
            {
                "model": "claude-opus-5-5",
                "max_tokens": 16000,
                "system": rag_chain.SYSTEM_PROMPT,
                "output_config": {"effort": "medium"},
                "betas": ["server-side-fallback-2026-07-01"],
                "extra_body": {"fallbacks": "default"},
                "messages": [
                    {
                        "role": "user",
                        "content": rag_chain.format_context(self.docs)
                        + "\n\nQuestion : Quelle peine ?",
                    }
                ],
            },
        )
        # Opus 5.5 returns 400 for these.
        for forbidden in ("temperature", "top_p", "top_k", "thinking"):
            self.assertNotIn(forbidden, kwargs)
        # Only the non-beta messages API must not be used.
        client.messages.create.assert_not_called()

    def test_returns_text(self):
        out, _ = self._run(fake_response([block("text", text="Réponse.")]))
        self.assertEqual(out, "Réponse.")

    def test_skips_thinking_blocks(self):
        content = [
            block("thinking", thinking="raisonnement", signature="sig"),
            block("text", text="Première partie. "),
            block("redacted_thinking", data="xxx"),
            block("text", text="Seconde partie."),
        ]
        out, _ = self._run(fake_response(content))
        self.assertEqual(out, "Première partie. Seconde partie.")
        self.assertNotIn("raisonnement", out)

    def test_refusal(self):
        out, _ = self._run(
            fake_response([block("text", text="partiel")], stop_reason="refusal")
        )
        self.assertEqual(out, REFUSAL_ANSWER)

    def test_refusal_with_empty_content(self):
        out, _ = self._run(fake_response([], stop_reason="refusal"))
        self.assertEqual(out, REFUSAL_ANSWER)

    def test_empty_text_falls_back_to_refusal(self):
        out, _ = self._run(
            fake_response([block("thinking", thinking="…")], stop_reason="max_tokens")
        )
        self.assertEqual(out, REFUSAL_ANSWER)

    def test_answer_question_uses_generate(self):
        docs = [FakeDoc("Art. 1", page=3)]
        gen = MagicMock(return_value="Réponse brute.")
        retriever = MagicMock()
        retriever.invoke.return_value = docs
        with (
            patch.object(
                rag_chain.jev, "triage", MagicMock(return_value=(0.9, "faible"))
            ),
            patch.object(rag_chain.jev, "rerank", MagicMock(return_value=docs)),
            patch.object(rag_chain.jev, "grounded", MagicMock(return_value=0.9)),
            patch.object(rag_chain, "retriever", retriever),
            patch.object(rag_chain, "generate", gen),
        ):
            out = rag_chain.answer_question("Q?")
        gen.assert_called_once_with("Q?", docs)
        self.assertEqual(out["answer"], "Réponse brute.")
        self.assertEqual(out["sources"], [3])


# ---------- vector store build at import ----------
class TestVectorStoreBuild(unittest.TestCase):
    """Module-level code is re-run with importlib.reload under patched stubs.

    reload mutates the module object in place, so a final plain reload in
    tearDown restores default stub-backed state for the other test modules.
    """

    def tearDown(self):
        importlib.reload(rag_chain)

    def _reload(self, ids):
        chroma = MagicMock(name="Chroma")
        chroma.return_value.get.return_value = {"ids": ids}
        splitter = MagicMock(name="RecursiveCharacterTextSplitter")
        loader = MagicMock(name="PyPDFLoader")
        embeddings = MagicMock(name="HuggingFaceEmbeddings")
        with (
            patch.object(sys.modules["langchain_chroma"], "Chroma", chroma),
            patch.object(
                sys.modules["langchain_text_splitters"],
                "RecursiveCharacterTextSplitter",
                splitter,
            ),
            patch.object(
                sys.modules["langchain_community.document_loaders"],
                "PyPDFLoader",
                loader,
            ),
            patch.object(
                sys.modules["langchain_huggingface"],
                "HuggingFaceEmbeddings",
                embeddings,
            ),
        ):
            importlib.reload(rag_chain)
        return chroma, splitter, loader, embeddings

    def test_empty_store_is_built(self):
        chroma, splitter, loader, embeddings = self._reload(ids=[])
        embeddings.assert_called_once_with(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        chroma.assert_called_once_with(
            persist_directory="rag", embedding_function=embeddings.return_value
        )
        store = chroma.return_value
        store.get.assert_called_once_with(limit=1)
        loader.assert_called_once_with("Burundi_Code_2017_penal.pdf")
        loader.return_value.load.assert_called_once_with()
        splitter.assert_called_once_with(chunk_size=600, chunk_overlap=100)
        splitter.return_value.split_documents.assert_called_once_with(
            loader.return_value.load.return_value
        )
        store.add_documents.assert_called_once_with(
            splitter.return_value.split_documents.return_value
        )
        store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 10}
        )
        self.assertIs(rag_chain.retriever, store.as_retriever.return_value)
        self.assertIsNone(rag_chain._claude)

    def test_existing_store_is_reused(self):
        chroma, splitter, loader, _ = self._reload(ids=["id-1"])
        store = chroma.return_value
        store.add_documents.assert_not_called()
        store.from_documents.assert_not_called()
        chroma.from_documents.assert_not_called()
        loader.assert_not_called()
        splitter.assert_not_called()
        store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 10}
        )
        self.assertIs(rag_chain.retriever, store.as_retriever.return_value)


# ---------- static checks ----------
def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestStatic(unittest.TestCase):
    def test_env_example(self):
        lines = [s.strip() for s in _read(".env.example").splitlines() if s.strip()]
        self.assertEqual(
            lines, ["ANTHROPIC_API_KEY=", "TYPESAFE_API_KEY=", "HF_TOKEN="]
        )

    def test_requirements(self):
        reqs = [
            re.split(r"[<>=!~\[;\s]", line.strip(), maxsplit=1)[0]
            .lower()
            .replace("_", "-")
            for line in _read("requirements.txt").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertIn("anthropic", reqs)
        self.assertIn("langchain-huggingface", reqs)
        self.assertNotIn("langchain-google-genai", reqs)

    def test_no_google_or_gemini_in_app(self):
        files = sorted(glob.glob(os.path.join(_ROOT, "app", "*.py")))
        self.assertTrue(files)
        for path in files:
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    self.assertIsNone(
                        re.search(r"google|gemini", line, re.I),
                        f"{os.path.relpath(path, _ROOT)}:{n}: {line.strip()}",
                    )

    def test_rag_chain_imports(self):
        src = _read("app", "rag_chain.py")
        self.assertRegex(src, r"(?m)^import anthropic$")
        self.assertIn("from langchain_huggingface import HuggingFaceEmbeddings", src)
        for gone in ("langchain.chains", "langchain_core.prompts", "langchain_google"):
            self.assertNotIn(gone, src)


if __name__ == "__main__":
    unittest.main()
