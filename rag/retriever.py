"""Module 3 - Retriever.

Given a customer question, finds the most relevant policy chunks.

Hallucination mitigation, layer 1 - the similarity gate:
    If even the best chunk scores below `min_score`, the question is treated
    as NOT answerable from the documents and the LLM is never called.
"""
from __future__ import annotations

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


def contextualize_query(question: str, history: list[dict]) -> str:
    """Make short follow-ups ("and for senior citizens?") searchable.

    A very short question is combined with the previous user question so the
    retriever still knows the topic.
    """
    if len(question.split()) <= 5:
        for msg in reversed(history):
            if msg["role"] == "user":
                return f"{msg['content']} {question}"
    return question


class Retriever:
    def __init__(self, store: VectorStore, embedder: BaseEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 4, min_score: float | None = None) -> RetrievalResult:
        min_score = self.embedder.default_min_score if min_score is None else min_score
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
