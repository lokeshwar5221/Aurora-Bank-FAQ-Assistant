"""KnowledgeBase = ingestion + embedding + indexing, built from many documents."""
from __future__ import annotations

import time
from collections import Counter
from datetime import date
from pathlib import Path

from .config import STORAGE_DIR, get_embedder_preference
from .embedder import BaseEmbedder, get_embedder
from .ingest import Chunk, Document, build_chunks, load_documents
from .retriever import Retriever
from .vectorstore import VectorStore, fingerprint


def _policy_key(document: Document) -> str:
    return (document.policy_id or document.title).strip().casefold()


def _active_policy_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """Keep only an unambiguous active revision for each policy family."""
    groups: dict[str, list[Document]] = {}
    for document in documents:
        groups.setdefault(_policy_key(document), []).append(document)

    active: list[Document] = []
    conflicts: list[str] = []
    for key, group in groups.items():
        distinct: dict[tuple[str, str | None, str | None, tuple[str, ...]], Document] = {}
        for document in group:
            identity = (
                document.text, document.version, document.effective_date,
                tuple(document.supersedes),
            )
            distinct.setdefault(identity, document)
        group = list(distinct.values())
        if len(group) == 1:
            active.extend(group)
            continue

        superseded: set[str] = set()
        for document in group:
            for target in document.supersedes or []:
                for candidate in group:
                    if candidate is not document and target.casefold() in {
                        (candidate.version or "").casefold(),
                        candidate.source.casefold(),
                        candidate.doc_id.casefold(),
                    }:
                        superseded.add(candidate.doc_id)
        candidates = [document for document in group if document.doc_id not in superseded]
        if len(candidates) == 1:
            active.extend(candidates)
            continue

        dated: list[tuple[date, Document]] = []
        try:
            dated = [(date.fromisoformat(document.effective_date or ""), document)
                     for document in candidates]
        except ValueError:
            dated = []
        if len(dated) == len(candidates) and dated:
            latest_date = max(effective for effective, _ in dated)
            latest = [document for effective, document in dated if effective == latest_date]
            if len(latest) == 1:
                active.extend(latest)
                continue

        sources = ", ".join(sorted(document.source for document in group))
        conflicts.append(f"{key}: {sources}")
    return active, conflicts


class KnowledgeBase:
    def __init__(self, folders: list[Path], embedder_preference: str | None = None) -> None:
        self.folders = [Path(f) for f in folders]
        self.embedder_preference = embedder_preference or get_embedder_preference()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.ingestion_errors: list[str] = []
        self.embedder: BaseEmbedder | None = None
        self.store: VectorStore | None = None
        self.fallback_reason: str | None = None
        self.policy_conflicts: list[str] = []
        self.loaded_from_cache = False
        self.build_seconds = 0.0

    # ---- build ------------------------------------------------------------
    def build(self, persist: bool = True) -> "KnowledgeBase":
        t0 = time.perf_counter()
        self.documents = load_documents(self.folders, self.ingestion_errors)  # 1. ingest
        active_documents, self.policy_conflicts = _active_policy_documents(self.documents)
        self.chunks = build_chunks(active_documents)                  # 2. chunk
        self.embedder, self.fallback_reason = get_embedder(self.embedder_preference)  # 3. embed

        texts = [c.embed_text for c in self.chunks]
        fp = fingerprint(self.chunks, self.embedder.name)
        store = VectorStore.load(STORAGE_DIR, fp) if persist and self.embedder.persistable else None
        if store is not None:
            self.loaded_from_cache = True
        else:                                                         # 4. index
            if texts:
                self.embedder.fit(texts)
                vectors = self.embedder.encode_documents(texts)
            else:
                import numpy as np

                vectors = np.zeros((0, 1), dtype="float32")
            store = VectorStore(vectors, self.chunks)
            if persist and self.embedder.persistable and texts:
                store.save(STORAGE_DIR, fp)
        self.store = store
        self.build_seconds = time.perf_counter() - t0
        return self

    # ---- helpers ----------------------------------------------------------
    @property
    def retriever(self) -> Retriever:
        assert self.store is not None and self.embedder is not None, "call build() first"
        return Retriever(self.store, self.embedder)

    def chunks_per_document(self) -> Counter:
        return Counter(c.doc_id for c in self.chunks)

    def stats(self) -> dict:
        n = len(self.chunks)
        return {
            "documents": len(self.documents),
            "chunks": n,
            "avg_chunk_chars": int(sum(len(c.text) for c in self.chunks) / n) if n else 0,
            "embedder": self.embedder.label if self.embedder else "-",
            "embedder_name": self.embedder.name if self.embedder else "-",
            "cached": self.loaded_from_cache,
            "build_seconds": round(self.build_seconds, 2),
            "fallback_reason": self.fallback_reason,
        }
