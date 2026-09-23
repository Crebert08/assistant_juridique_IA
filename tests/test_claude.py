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
        self.assertEqual(rag_chain.DEFAULT_PROVIDER, "gemini")
        self.assertEqual(rag_chain.LLM["provider"], "gemini")
        self.assertEqual(rag_chain.LLM["model"], "gemini-3.8-flash")
        self.assertEqual(rag_chain.LLM["effort"], "medium")
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


# ---------- LLM provider config ----------
class TestLoadLlmConfig(unittest.TestCase):
    def test_defaults_to_gemini_flash(self):
        for env in ({}, {"LLM_PROVIDER": ""}):
            cfg = rag_chain.load_llm_config(env)
            self.assertEqual(cfg["provider"], "gemini")
            self.assertEqual(cfg["model"], "gemini-3.8-flash")
            self.assertEqual(cfg["effort"], "medium")
            self.assertEqual(cfg["api_key_env"], "GEMINI_API_KEY")

    def test_anthropic_defaults_to_opus(self):
        cfg = rag_chain.load_llm_config({"LLM_PROVIDER": "anthropic"})
        self.assertEqual(cfg["provider"], "anthropic")
        self.assertEqual(cfg["model"], "claude-opus-5-5")
        self.assertEqual(cfg["effort"], "medium")
        self.assertIsNone(cfg["base_url"])
        self.assertEqual(cfg["api_key_env"], "ANTHROPIC_API_KEY")

    def test_deepseek_defaults_to_v4_pro(self):
        cfg = rag_chain.load_llm_config({"LLM_PROVIDER": "DeepSeek "})
        self.assertEqual(cfg["provider"], "deepseek")
        self.assertEqual(cfg["model"], "deepseek-v4-pro")
        self.assertEqual(cfg["effort"], "high")
        self.assertEqual(cfg["base_url"], "https://api.deepseek.com/anthropic")
        self.assertEqual(cfg["api_key_env"], "DEEPSEEK_API_KEY")

    def test_overrides(self):
        cfg = rag_chain.load_llm_config(
            {"LLM_PROVIDER": "deepseek", "LLM_MODEL": "deepseek-flash", "LLM_EFFORT": "max"}
        )
        self.assertEqual((cfg["model"], cfg["effort"]), ("deepseek-flash", "max"))
        cfg = rag_chain.load_llm_config(
            {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-sonnet-5"}
        )
        self.assertEqual(cfg["model"], "claude-sonnet-5")

    def test_empty_values_use_defaults(self):
        # .env.example ships LLM_MODEL= and LLM_EFFORT= empty.
        cfg = rag_chain.load_llm_config(
            {"LLM_PROVIDER": "deepseek", "LLM_MODEL": "", "LLM_EFFORT": " "}
        )
        self.assertEqual((cfg["model"], cfg["effort"]), ("deepseek-v4-pro", "high"))

    def test_unknown_provider_fails(self):
        with self.assertRaises(ValueError):
            rag_chain.load_llm_config({"LLM_PROVIDER": "openai"})

    def test_unknown_deepseek_model_fails(self):
        # DeepSeek would silently serve deepseek-flash for a typo.
        with self.assertRaises(ValueError):
            rag_chain.load_llm_config(
                {"LLM_PROVIDER": "deepseek", "LLM_MODEL": "deepseek-r1"}
            )


class TestLlmClient(unittest.TestCase):
    def _client(self, provider, env):
        ctor = MagicMock(return_value=MagicMock(name="client-instance"))
        with (
            patch.object(rag_chain, "_llm_client", None),
            patch.object(rag_chain, "LLM", rag_chain.load_llm_config(provider)),
            patch.object(rag_chain.anthropic, "Anthropic", ctor),
            patch.dict(os.environ, env, clear=True),
        ):
            c1 = rag_chain.get_llm_client()
            c2 = rag_chain.get_llm_client()
        self.assertIs(c1, c2)
        return ctor

    def test_no_client_at_import(self):
        self.assertIsNone(rag_chain._llm_client)

    def test_anthropic_lazy_and_cached(self):
        ctor = self._client(
            {"LLM_PROVIDER": "anthropic"}, {"ANTHROPIC_API_KEY": "sk-ant-x"}
        )
        ctor.assert_called_once_with(api_key="sk-ant-x", base_url=None)

    def test_deepseek_uses_its_own_key_and_url(self):
        ctor = self._client(
            {"LLM_PROVIDER": "deepseek"},
            {"ANTHROPIC_API_KEY": "sk-ant-x", "DEEPSEEK_API_KEY": "sk-ds-y"},
        )
        ctor.assert_called_once_with(
            api_key="sk-ds-y", base_url="https://api.deepseek.com/anthropic"
        )

    def test_gemini_client_uses_gemini_key(self):
        genai = sys.modules["google.genai"]
        genai.Client.reset_mock()
        with (
            patch.object(rag_chain, "_llm_client", None),
            patch.object(rag_chain, "LLM", rag_chain.load_llm_config({})),
            patch.dict(os.environ, {"GEMINI_API_KEY": "g-key"}, clear=True),
        ):
            c1 = rag_chain.get_llm_client()
            c2 = rag_chain.get_llm_client()
        self.assertIs(c1, c2)
        genai.Client.assert_called_once_with(api_key="g-key")

    def test_gemini_without_key_fails(self):
        with (
            patch.object(rag_chain, "_llm_client", None),
            patch.object(rag_chain, "LLM", rag_chain.load_llm_config({})),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-x"}, clear=True),
        ):
            with self.assertRaises(RuntimeError):
                rag_chain.get_llm_client()

    def test_deepseek_without_key_never_falls_back_to_anthropic_key(self):
        with self.assertRaises(RuntimeError):
            self._client({"LLM_PROVIDER": "deepseek"}, {"ANTHROPIC_API_KEY": "sk-ant-x"})


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
        # These tests cover the Anthropic path; DeepSeek has test_deepseek_request.
        client = fake_claude(response)
        cfg = rag_chain.load_llm_config({"LLM_PROVIDER": "anthropic"})
        with (
            patch.object(rag_chain, "LLM", cfg),
            patch.object(rag_chain, "get_llm_client", return_value=client),
        ):
            out = rag_chain.generate("Quelle peine ?", self.docs)
        return out, client

    def _run_gemini(self, output_text):
        client = MagicMock(name="gemini-client")
        client.interactions.create.return_value = SimpleNamespace(
            output_text=output_text
        )
        with (
            patch.object(rag_chain, "LLM", rag_chain.load_llm_config({})),
            patch.object(rag_chain, "get_llm_client", return_value=client),
        ):
            out = rag_chain.generate("Quelle peine ?", self.docs)
        return out, client

    def test_gemini_request(self):
        out, client = self._run_gemini("Réponse Gemini.")
        self.assertEqual(out, "Réponse Gemini.")
        client.interactions.create.assert_called_once_with(
            model="gemini-3.8-flash",
            input=rag_chain.format_context(self.docs) + "\n\nQuestion : Quelle peine ?",
            system_instruction=rag_chain.SYSTEM_PROMPT,
            generation_config={"thinking_level": "medium", "max_output_tokens": 16000},
            store=False,
        )

    def test_gemini_empty_output_is_refusal(self):
        for empty in ("", None):
            out, _ = self._run_gemini(empty)
            self.assertEqual(out, REFUSAL_ANSWER)

    def test_deepseek_request(self):
        client = MagicMock(name="deepseek-client")
        client.messages.create.return_value = fake_response(
            [block("thinking", thinking="…"), block("text", text="Réponse DS.")]
        )
        cfg = rag_chain.load_llm_config({"LLM_PROVIDER": "deepseek"})
        with (
            patch.object(rag_chain, "LLM", cfg),
            patch.object(rag_chain, "get_llm_client", return_value=client),
        ):
            out = rag_chain.generate("Quelle peine ?", self.docs)
        self.assertEqual(out, "Réponse DS.")
        client.beta.messages.create.assert_not_called()
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["output_config"], {"effort": "high"})
        # Anthropic-only settings must not be sent to DeepSeek.
        for key in ("betas", "extra_body", "temperature", "thinking"):
            self.assertNotIn(key, kwargs)

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
        self.assertIsNone(rag_chain._llm_client)

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
        names = [
            s.split("=", 1)[0].strip()
            for s in _read(".env.example").splitlines()
            if s.strip() and not s.lstrip().startswith("#")
        ]
        self.assertEqual(
            names,
            [
                "LLM_PROVIDER",
                "LLM_MODEL",
                "LLM_EFFORT",
                "GEMINI_API_KEY",
                "DEEPSEEK_API_KEY",
                "ANTHROPIC_API_KEY",
                "TYPESAFE_API_KEY",
                "HF_TOKEN",
            ],
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
        self.assertIn("google-genai", reqs)
        self.assertIn("langchain-huggingface", reqs)
        self.assertNotIn("langchain-google-genai", reqs)
        # The Interactions API needs google-genai >= 2.3.0.
        self.assertIn("google-genai>=2.3.0", _read("requirements.txt"))

    def test_no_old_langchain_gemini_in_app(self):
        # Gemini now goes through google-genai; the old LangChain wrapper and
        # its Google embeddings must not come back.
        files = sorted(glob.glob(os.path.join(_ROOT, "app", "*.py")))
        self.assertTrue(files)
        for path in files:
            with open(path, encoding="utf-8") as f:
                for n, line in enumerate(f, 1):
                    self.assertIsNone(
                        re.search(r"langchain_google|GoogleGenerativeAI|embedding-001", line),
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
