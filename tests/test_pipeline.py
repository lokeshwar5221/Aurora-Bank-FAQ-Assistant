"""Run with:  python -m pytest -q      (or)      python tests/test_pipeline.py

Uses the TF-IDF embedder so the tests need no model download or API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import POLICY_DIR  # noqa: E402
from rag.generator import (NOT_FOUND_TOKEN, build_messages, smalltalk_reply,  # noqa: E402
                           unverified_numbers)
from rag.ingest import Document, chunk_document  # noqa: E402
from rag.pipeline import KnowledgeBase  # noqa: E402
from rag.retriever import contextualize_query  # noqa: E402

_KB = None


def kb() -> KnowledgeBase:
    global _KB
    if _KB is None:
        _KB = KnowledgeBase([POLICY_DIR], embedder_preference="tfidf").build()
    return _KB


# ---- ingestion -------------------------------------------------------------
def test_all_policy_documents_loaded():
    assert kb().stats()["documents"] >= 10
    assert kb().stats()["chunks"] > 50


def test_chunks_keep_source_metadata():
    for c in kb().chunks:
        assert c.doc_title and c.section and c.source and c.text.strip()


def test_long_section_is_split_with_size_limit():
    text = "# Title\n\n## Big\n" + "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    doc = Document("d1", "Title", "t.md", text)
    chunks = chunk_document(doc, size=500, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 600 for c in chunks)
    assert all(c.section == "Big" for c in chunks)


# ---- retrieval -------------------------------------------------------------
IN_SCOPE = [
    ("What is the minimum balance for a regular savings account?", "Savings Account Policy"),
    ("What are the FD interest rates for senior citizens?", "Fixed Deposit"),
    ("What are the eligibility criteria for a personal loan?", "Personal Loan"),
    ("What is the maximum tenure of a home loan?", "Home Loan"),
    ("How do I report a fraudulent transaction?", "Digital Banking"),
    ("What is the late payment fee on credit cards?", "Credit Card"),
    ("What documents are needed to open an account?", "KYC"),
    ("What are the NEFT and RTGS charges?", "Schedule of Fees"),
]

OUT_OF_SCOPE = [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "Tell me a joke",
    "What is the weather in Mumbai today?",
]


def test_in_scope_questions_find_the_right_document():
    for question, expected in IN_SCOPE:
        res = kb().retriever.retrieve(question, top_k=4)
        assert res.answerable, f"should be answerable: {question}"
        titles = " | ".join(h.chunk.doc_title for h in res.hits)
        assert expected in titles, f"{question!r} -> {titles}"


def test_out_of_scope_questions_are_rejected_by_the_similarity_gate():
    for question in OUT_OF_SCOPE:
        res = kb().retriever.retrieve(question, top_k=4)
        assert not res.answerable, f"should NOT be answerable: {question} (score {res.best_score:.2f})"


def test_followup_question_uses_previous_topic():
    history = [{"role": "user", "content": "What is the home loan interest rate?"}]
    assert "home loan" in contextualize_query("and for women?", history)
    assert contextualize_query("What is the minimum balance for a regular savings account?", history) \
        == "What is the minimum balance for a regular savings account?"


# ---- generator helpers -------------------------------------------------------
def test_prompt_contains_rules_context_and_question():
    res = kb().retriever.retrieve("home loan processing fee", top_k=3)
    msgs = build_messages("home loan processing fee", res.hits, [])
    assert NOT_FOUND_TOKEN in msgs[0]["content"]
    assert "[1] Source:" in msgs[1]["content"]
    assert "home loan processing fee" in msgs[1]["content"]


def test_number_check_flags_invented_figures():
    res = kb().retriever.retrieve("home loan processing fee", top_k=3)
    ok = "The processing fee is 0.50% of the loan amount, with a maximum of ₹15,000 [2]."
    bad = "The processing fee is 3.75% of the loan amount, with a maximum of ₹99,000 [2]."
    assert unverified_numbers(ok, res.hits) == []
    assert set(unverified_numbers(bad, res.hits)) == {"3.75", "99,000"}


def test_smalltalk():
    assert smalltalk_reply("hello") is not None
    assert smalltalk_reply("thanks!") is not None
    assert smalltalk_reply("What is the FD rate?") is None



class _FakeLLM:
    """Stands in for the Groq client: streams the given pieces."""
    def __init__(self, *pieces):
        self.pieces = pieces

    def stream(self, messages, **kwargs):
        yield from self.pieces


def _run(llm):
    from rag.generator import stream_answer
    res = kb().retriever.retrieve("home loan processing fee", top_k=3)
    return "".join(stream_answer(llm, "home loan processing fee", res.hits, []))


def test_refusal_token_is_swallowed_even_when_split_across_chunks():
    assert _run(_FakeLLM("[[NOT_", "IN_POL", "ICY]]")) == ""
    assert _run(_FakeLLM("  [[NOT_IN_POLICY]]")) == ""


def test_normal_answer_streams_through_unchanged():
    assert _run(_FakeLLM("The fee is ", "0.50% [1].")) == "The fee is 0.50% [1]."
    assert _run(_FakeLLM("Yes.")) == "Yes."
    assert _run(_FakeLLM("[", "1] is cited")) == "[1] is cited"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
