"""Module 3 - Retriever.

Given a customer question, finds the most relevant policy chunks.

Hallucination mitigation, layer 1 - the similarity gate:
    If even the best chunk scores below `min_score`, the question is treated
    as NOT answerable from the documents and the LLM is never called.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embedder import BaseEmbedder
from .ingest import Chunk
from .vectorstore import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit] = field(default_factory=list)       # relevant chunks (>= threshold)
    closest: list[Hit] = field(default_factory=list)    # best chunks regardless of threshold
    best_score: float = 0.0
    min_score: float = 0.0

    @property
    def answerable(self) -> bool:
        return bool(self.hits)


_FOLLOW_UP_RE = re.compile(
    r"^\s*(?:(?:and|also|then|what about|how about|what if|in that case|"
    r"compared with that)\b|(?:it|they|them|this|that|those|these|there|same)\b)",
    re.I,
)
_CONDITION_FOLLOW_UP_RE = re.compile(
    r"^\s*for\s+(?:women|men|senior citizens|students|retired people|"
    r"existing customers|new customers)\b",
    re.I,
)
_TOPIC_PATTERNS = (
    re.compile(r"\bhome loans?\b", re.I),
    re.compile(r"\bpersonal loans?\b", re.I),
    re.compile(r"\b(?:education|vehicle) loans?\b", re.I),
    re.compile(r"\b(?:savings|current) accounts?\b", re.I),
    re.compile(r"\b(?:fixed deposits?|fds?)\b", re.I),
    re.compile(r"\bcredit cards?\b", re.I),
    re.compile(r"\batms?\b", re.I),
    re.compile(r"\b(?:upi|neft|rtgs|imps)\b", re.I),
    re.compile(r"\bkyc\b", re.I),
    re.compile(r"\bfraud(?:ulent)? transactions?\b", re.I),
)


def contextualize_query(question: str, history: list[dict]) -> str:
    """Attach prior context only when the wording signals a follow-up."""
    is_follow_up = bool(_FOLLOW_UP_RE.match(question) or _CONDITION_FOLLOW_UP_RE.match(question))
    if is_follow_up:
        for msg in reversed(history):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                previous = msg["content"]
                new_topics = [pattern for pattern in _TOPIC_PATTERNS if pattern.search(question)]
                previous_topics = [pattern for pattern in _TOPIC_PATTERNS if pattern.search(previous)]
                if new_topics and not any(topic in previous_topics for topic in new_topics):
                    return question
                return f"{previous} {question}"
    return question


class Retriever:
    def __init__(self, store: VectorStore, embedder: BaseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 4, min_score: float | None = None) -> RetrievalResult:
        min_score = self.embedder.default_min_score if min_score is None else min_score
        if len(self.store) == 0:
            return RetrievalResult(query=query, best_score=0.0, min_score=min_score)
        qvec = self.embedder.encode_query(query)
        scored = self.store.search(qvec, k=top_k)
        closest = [Hit(c, s, i) for i, (c, s) in enumerate(scored, start=1)]
        hits = [h for h in closest if h.score >= min_score]
        for i, h in enumerate(hits, start=1):   # re-number so citations are [1], [2], ...
            h.rank = i
        return RetrievalResult(
            query=query,
            hits=hits,
            closest=closest,
            best_score=closest[0].score if closest else 0.0,
            min_score=min_score,
        )
