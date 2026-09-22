"""Phase 1 tests: app/jev.py and app/rag_chain.answer_question.

Run: python3 -m unittest discover -s tests -v   (from project root)
Third-party deps are stubbed via tests/_stubs.py; no network, no PDF load.
"""

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

from app import (
    jev,  # noqa: E402
    rag_chain,  # noqa: E402
)

REFUSAL = "Je ne peux répondre qu'aux questions sur le Code pénal du Burundi."
WARNING_MARK = "Réponse partiellement vérifiée"
LAWYER_MARK = "consultez un avocat"


# ---------- fakes ----------
class FakeDoc:
    def __init__(self, text, page=None):
        self.page_content = text
        self.metadata = {} if page is None else {"page": page}

    def __repr__(self):
        return f"FakeDoc({self.page_content!r})"


def noul_result(**nouls):
    return SimpleNamespace(
        nouls={k: SimpleNamespace(noul=v) for k, v in nouls.items()}, scores={}
    )


def triage_result(in_scope, sensitivity):
    return SimpleNamespace(
        nouls={"in_scope": SimpleNamespace(noul=in_scope)},
        scores={"sensitivity": SimpleNamespace(score=sensitivity, confidence=0.9)},
    )


def relevance_result(score, confidence):
    return SimpleNamespace(
        nouls={},
        scores={"relevance": SimpleNamespace(score=score, confidence=confidence)},
    )


def extract_state(args, kwargs):
    """Find the state dict passed to client.system_one, however it's passed."""
    if isinstance(kwargs.get("state"), dict):
        return kwargs["state"]
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, dict):
            return v
    raise AssertionError(f"no state dict in call args={args} kwargs={kwargs}")


def fake_client(side_effect=None, return_value=None):
    client = MagicMock(name="TypeSafeClient()")
    if side_effect is not None:
        client.system_one.side_effect = side_effect
    else:
        client.system_one.return_value = return_value
    return client


# ---------- app/jev.py ----------
class TestJevModule(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(jev.SENSITIVITY_LEVELS, ["faible", "moyen", "élevé"])
        self.assertEqual(jev.RELEVANCE_LEVELS, ["non", "partiellement", "oui"])

    def test_no_client_at_import(self):
        # _client must be None until get_client() is first called.
        self.assertIsNone(jev._client)
        self.assertTrue(callable(jev.get_client))

    def test_get_client_is_lazy_and_cached(self):
        sentinel = MagicMock(name="client-instance")
        with (
            patch.object(jev, "_client", None),
            patch.object(
                jev, "TypeSafeClient", return_value=sentinel, create=True
            ) as ctor,
        ):
            # If jev references typesafe_sdk.TypeSafeClient instead, patch that too.
            with patch.object(sys.modules["typesafe_sdk"], "TypeSafeClient", ctor):
                c1 = jev.get_client()
                c2 = jev.get_client()
        self.assertIs(c1, sentinel)
        self.assertIs(c2, sentinel)
        self.assertEqual(ctor.call_count, 1)


class TestTriage(unittest.TestCase):
    def _sensitivity(self, score):
        client = fake_client(return_value=triage_result(0.9, score))
        with patch.object(jev, "get_client", return_value=client):
            return jev.triage("Q")[1]

    def test_score_maps_to_nearest_level(self):
        # Score is a probability-weighted level index (0..2), e.g. 0.24 from a real call.
        self.assertEqual(self._sensitivity(0.24), "faible")
        self.assertEqual(self._sensitivity(1.2), "moyen")
        self.assertEqual(self._sensitivity(1.6), "élevé")

    def test_sends_app_context_with_question(self):
        # Without the Burundi context, real Jev scored "vol simple" 0.28 (refused).
        client = fake_client(return_value=triage_result(0.9, 0.0))
        with patch.object(jev, "get_client", return_value=client):
            jev.triage("Quelle est la peine pour le vol simple ?")
        state = extract_state(*client.system_one.call_args)
        self.assertEqual(state["contexte"], jev.APP_CONTEXT)
        self.assertEqual(state["question"], "Quelle est la peine pour le vol simple ?")

    def test_score_out_of_range_is_clamped(self):
        self.assertEqual(self._sensitivity(-0.7), "faible")
        self.assertEqual(self._sensitivity(2.9), "élevé")

    def test_returns_noul_and_score(self):
        client = fake_client(return_value=triage_result(0.83, 2.0))
        with patch.object(jev, "get_client", return_value=client):
            in_scope, sensitivity = jev.triage("Quelle est la peine pour vol ?")
        self.assertEqual(in_scope, 0.83)
        self.assertEqual(sensitivity, "élevé")
        self.assertEqual(client.system_one.call_count, 1)
        # Contract doesn't fix triage's state shape: accept raw str or {"question": ...}.
        args, kwargs = client.system_one.call_args
        passed = list(args) + list(kwargs.values())
        self.assertTrue(
            "Quelle est la peine pour vol ?" in passed
            or any(
                isinstance(v, dict)
                and v.get("question") == "Quelle est la peine pour vol ?"
                for v in passed
            ),
            f"question not passed to system_one: {client.system_one.call_args}",
        )


class TestRerank(unittest.TestCase):
    def _run(self, table, docs, keep):
        """table: page_content -> (score, confidence)."""

        def side_effect(*args, **kwargs):
            state = extract_state(args, kwargs)
            self.assertEqual(state.get("question"), "Q")
            return relevance_result(*table[state["extrait"]])

        client = fake_client(side_effect=side_effect)
        with patch.object(jev, "get_client", return_value=client):
            out = jev.rerank("Q", docs, keep=keep)
        return out, client

    def test_orders_by_score(self):
        # Score is a 0..2 level index; higher confidence must not beat a higher score.
        docs = [FakeDoc("a"), FakeDoc("b"), FakeDoc("c")]
        table = {"a": (0.1, 0.99), "b": (1.2, 0.99), "c": (1.9, 0.10)}
        out, client = self._run(table, docs, keep=3)
        self.assertEqual([d.page_content for d in out], ["c", "b", "a"])
        self.assertEqual(client.system_one.call_count, 3)

    def test_confidence_tie_break(self):
        docs = [FakeDoc("low"), FakeDoc("high"), FakeDoc("mid")]
        table = {"low": (2.0, 0.2), "high": (2.0, 0.9), "mid": (2.0, 0.5)}
        out, _ = self._run(table, docs, keep=3)
        self.assertEqual([d.page_content for d in out], ["high", "mid", "low"])

    def test_keeps_only_keep(self):
        docs = [FakeDoc(str(i)) for i in range(6)]
        table = {
            "0": (0.2, 0.9),
            "1": (1.8, 0.7),
            "2": (1.1, 0.8),
            "3": (1.9, 0.95),
            "4": (0.0, 0.1),
            "5": (1.0, 0.3),
        }
        out, client = self._run(table, docs, keep=2)
        self.assertEqual([d.page_content for d in out], ["3", "1"])
        self.assertEqual(client.system_one.call_count, 6)

    def test_default_keep_is_4(self):
        docs = [FakeDoc(str(i)) for i in range(10)]
        table = {str(i): (i / 5, 0.5) for i in range(10)}
        client = fake_client(
            side_effect=lambda *a, **k: relevance_result(
                *table[extract_state(a, k)["extrait"]]
            )
        )
        with patch.object(jev, "get_client", return_value=client):
            out = jev.rerank("Q", docs)
        self.assertEqual([d.page_content for d in out], ["9", "8", "7", "6"])

    def test_returns_doc_objects(self):
        docs = [FakeDoc("x", page=3)]
        out, _ = self._run({"x": (2.0, 0.5)}, docs, keep=4)
        self.assertIs(out[0], docs[0])

    def test_empty_docs_no_client_call(self):
        get_client = MagicMock(name="get_client")
        with patch.object(jev, "get_client", get_client):
            out = jev.rerank("Q", [], keep=4)
        self.assertEqual(out, [])
        get_client.return_value.system_one.assert_not_called()


class TestGrounded(unittest.TestCase):
    def test_joins_articles_and_returns_noul(self):
        client = fake_client(return_value=noul_result(grounded=0.42))
        docs = [FakeDoc("Article 1"), FakeDoc("Article 2"), FakeDoc("Article 3")]
        with patch.object(jev, "get_client", return_value=client):
            g = jev.grounded("Q", docs, "La réponse")
        self.assertEqual(g, 0.42)
        self.assertEqual(client.system_one.call_count, 1)
        state = extract_state(*client.system_one.call_args)
        self.assertEqual(state["articles"], "Article 1\n\nArticle 2\n\nArticle 3")
        self.assertEqual(state["question"], "Q")
        self.assertEqual(state["reponse"], "La réponse")


# ---------- app/rag_chain.py ----------
class TestAnswerQuestion(unittest.TestCase):
    def setUp(self):
        self.docs = [FakeDoc("Art. 1 ...", page=12), FakeDoc("Art. 2 ...", page=40)]
        self.retriever = MagicMock(name="retriever")
        self.retriever.invoke.return_value = self.docs + [FakeDoc("noise", page=99)]
        self.generate = MagicMock(name="generate", return_value="Réponse brute.")

    def _call(self, triage, grounded, rerank=None):
        rerank = rerank or MagicMock(return_value=self.docs)
        grounded_mock = MagicMock(return_value=grounded)
        with (
            patch.object(rag_chain.jev, "triage", MagicMock(return_value=triage)),
            patch.object(rag_chain.jev, "rerank", rerank),
            patch.object(rag_chain.jev, "grounded", grounded_mock),
            patch.object(rag_chain, "retriever", self.retriever),
            patch.object(rag_chain, "generate", self.generate),
        ):
            out = rag_chain.answer_question("Q?")
        return out, rerank, grounded_mock

    def test_thresholds(self):
        self.assertEqual(rag_chain.IN_SCOPE_THRESHOLD, 0.5)
        self.assertEqual(rag_chain.GROUNDED_THRESHOLD, 0.7)

    def test_out_of_scope(self):
        out, rerank, grounded = self._call(triage=(0.2, "faible"), grounded=1.0)
        self.assertEqual(out["answer"], REFUSAL)
        self.assertIsNone(out["grounded"])
        self.assertEqual(out["sources"], [])
        self.assertEqual(out["in_scope"], 0.2)
        self.assertIn("sensitivity", out)
        self.retriever.invoke.assert_not_called()
        self.generate.assert_not_called()
        rerank.assert_not_called()
        grounded.assert_not_called()

    def test_in_scope_grounded_low_sensitivity(self):
        out, rerank, grounded = self._call(triage=(0.9, "faible"), grounded=0.95)
        self.assertEqual(out["answer"], "Réponse brute.")
        self.assertEqual(out["sources"], [12, 40])
        self.assertEqual(out["grounded"], 0.95)
        self.assertEqual(out["in_scope"], 0.9)
        self.assertEqual(out["sensitivity"], "faible")
        self.assertEqual(
            set(out), {"answer", "in_scope", "grounded", "sensitivity", "sources"}
        )
        self.retriever.invoke.assert_called_once_with("Q?")
        # rerank receives retrieved docs; generate gets reranked docs
        self.assertEqual(rerank.call_args[0][0], "Q?")
        self.assertEqual(
            list(rerank.call_args[0][1]), self.retriever.invoke.return_value
        )
        self.generate.assert_called_once_with("Q?", self.docs)
        g_args = grounded.call_args[0]
        self.assertEqual(g_args[0], "Q?")
        self.assertEqual(list(g_args[1]), self.docs)
        self.assertEqual(g_args[2], "Réponse brute.")

    def test_threshold_boundary_in_scope_exactly_05(self):
        out, _, _ = self._call(triage=(0.5, "faible"), grounded=0.95)
        self.assertNotEqual(out["answer"], REFUSAL)

    def test_low_grounding_appends_warning(self):
        out, _, _ = self._call(triage=(0.9, "moyen"), grounded=0.3)
        self.assertTrue(out["answer"].startswith("Réponse brute."))
        self.assertIn(WARNING_MARK, out["answer"])
        self.assertIn("⚠️", out["answer"])
        self.assertNotIn(LAWYER_MARK, out["answer"].lower())

    def test_high_sensitivity_appends_lawyer_line(self):
        out, _, _ = self._call(triage=(0.9, "élevé"), grounded=0.95)
        self.assertTrue(out["answer"].startswith("Réponse brute."))
        self.assertIn(LAWYER_MARK, out["answer"].lower())
        self.assertNotIn(WARNING_MARK, out["answer"])

    def test_both_warnings(self):
        out, _, _ = self._call(triage=(0.9, "élevé"), grounded=0.1)
        self.assertIn(WARNING_MARK, out["answer"])
        self.assertIn(LAWYER_MARK, out["answer"].lower())

    def test_empty_retrieval_skips_llm_and_grounding(self):
        self.retriever.invoke.return_value = []
        out, _, grounded = self._call(
            triage=(0.9, "faible"), grounded=0.9, rerank=MagicMock(return_value=[])
        )
        self.assertEqual(
            out["answer"], "Je ne trouve pas cette information dans le Code pénal."
        )
        self.assertIsNone(out["grounded"])
        self.assertEqual(out["sources"], [])
        self.generate.assert_not_called()
        grounded.assert_not_called()

    def test_claude_refusal_skips_grounding_and_warnings(self):
        self.generate.return_value = rag_chain.REFUSAL_ANSWER
        out, _, grounded = self._call(triage=(0.9, "élevé"), grounded=0.1)
        self.assertEqual(out["answer"], rag_chain.REFUSAL_ANSWER)
        self.assertIsNone(out["grounded"])
        self.assertEqual(out["sources"], [])
        grounded.assert_not_called()

    def test_missing_page_metadata_gives_none(self):
        docs = [FakeDoc("x")]
        out, _, _ = self._call(
            triage=(0.9, "faible"), grounded=0.9, rerank=MagicMock(return_value=docs)
        )
        self.assertEqual(out["sources"], [None])


# ---------- app/main.py (static check; fastapi not installed) ----------
class TestMainStatic(unittest.TestCase):
    def test_query_response_fields(self):
        with open(os.path.join(_ROOT, "app", "main.py"), encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"class QueryResponse\(BaseModel\):\n((?:[ \t]+.*\n?)+)", src)
        self.assertIsNotNone(m, "QueryResponse class not found")
        body = m.group(1)
        fields = dict(re.findall(r"^\s+(\w+)\s*:\s*(.+?)\s*(?:=.*)?$", body, re.M))
        self.assertEqual(
            set(fields), {"answer", "in_scope", "grounded", "sensitivity", "sources"}
        )
        self.assertIn("Optional", fields["grounded"])
        self.assertIn("float", fields["grounded"])
        self.assertIn("str", fields["answer"])
        self.assertIn("float", fields["in_scope"])


if __name__ == "__main__":
    unittest.main()
