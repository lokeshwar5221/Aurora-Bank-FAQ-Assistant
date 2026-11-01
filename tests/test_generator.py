from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag.generator import LLMClient, LLMError, SYSTEM_PROMPT, build_messages, stream_answer
from rag.ingest import Chunk
from rag.retriever import Hit


def _event(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


class _Completions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _llm(responses):
    client = object.__new__(LLMClient)
    client.model = "test-model"
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions(responses))
    )
    return client


def test_stream_does_not_retry_after_emitting_partial_text(monkeypatch):
    def interrupted_response():
        yield _event("Start. ")
        raise RuntimeError("connection dropped")

    llm = _llm([interrupted_response(), [_event("Start. Done.")]])
    monkeypatch.setattr("rag.generator.time.sleep", lambda _: None)

    with pytest.raises(LLMError, match="after sending text"):
        list(llm.stream([], attempts=3))

    assert llm.client.chat.completions.calls == 1


def test_stream_retries_when_failure_happens_before_first_token(monkeypatch):
    llm = _llm([RuntimeError("connection dropped"), [_event("Complete answer.")]])
    monkeypatch.setattr("rag.generator.time.sleep", lambda _: None)

    assert list(llm.stream([], attempts=2)) == ["Complete answer."]
    assert llm.client.chat.completions.calls == 2


def test_empty_stream_returns_no_text():
    llm = _llm([[]])

    assert list(llm.stream([], attempts=1)) == []


def test_prompt_rejects_personal_account_and_approval_claims():
    assert "personal balance" in SYSTEM_PROMPT.lower()
    assert "approval decision" in SYSTEM_PROMPT.lower()


def test_personal_balance_and_approval_questions_can_be_refused():
    hit = Hit(
        Chunk("chunk", "policy", "Policy", "policy.md", "Eligibility", "Published criteria only."),
        score=0.9,
        rank=1,
    )

    class RefusingLLM:
        def stream(self, messages, **kwargs):
            yield "[[NOT_IN_POLICY]]"

    for question in ("What is my account balance?", "Will my loan be approved?"):
        messages = build_messages(question, [hit], [])
        assert "personal balance" in messages[0]["content"].lower()
        assert list(stream_answer(RefusingLLM(), question, [hit], [])) == []
