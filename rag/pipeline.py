"""KnowledgeBase = ingestion + embedding + indexing, built from many documents."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .config import STORAGE_DIR, get_embedder_preference
from .embedder import BaseEmbedder, get_embedder
from .ingest import Chunk, Document, build_chunks, load_documents
from .retriever import Retriever
from .vectorstore import VectorStore, fingerprint


class KnowledgeBase:
    def __init__(self, folders: list[Path], embedder_preference: str | None = None) -> None:
        self.folders = [Path(f) for f in folders]
        self.embedder_preference = embedder_preference or get_embedder_preference()
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.embedder: BaseEmbedder | None = None
        self.store: VectorStore | None = None
        self.fallback_reason: str | None = None
        self.loaded_from_cache = False
        self.build_seconds = 0.0

    # ---- build ------------------------------------------------------------
    def build(self) -> "KnowledgeBase":
        t0 = time.perf_counter()
        self.documents = load_documents(self.folders)                 # 1. ingest
        self.chunks = build_chunks(self.documents)                    # 2. chunk
        self.embedder, self.fallback_reason = get_embedder(self.embedder_preference)  # 3. embed

        texts = [c.embed_text for c in self.chunks]
        fp = fingerprint(self.chunks, self.embedder.name)
        store = VectorStore.load(STORAGE_DIR, fp) if self.embedder.persistable else None
        if store is not None:
            self.loaded_from_cache = True
        else:                                                         # 4. index
            self.embedder.fit(texts)
            vectors = self.embedder.encode_documents(texts) if texts else None
            if vectors is None:
                import numpy as np

                vectors = np.zeros((0, 1), dtype="float32")
            store = VectorStore(vectors, self.chunks)
            if self.embedder.persistable and texts:
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
