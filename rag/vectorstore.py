"""Module 2b - Indexing (a small, transparent vector database).

Stores one vector per chunk and answers "which chunks are closest to this
query vector?" using cosine similarity (a dot product, because vectors are
normalised). The index is saved to disk together with a fingerprint of the
documents, so it is only rebuilt when the policy documents change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .ingest import Chunk


def fingerprint(chunks: list[Chunk], embedder_name: str) -> str:
    h = hashlib.sha256(embedder_name.encode("utf-8"))
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(c.embed_text.encode("utf-8"))
    return h.hexdigest()


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        assert len(vectors) == len(chunks), "one vector per chunk"
        self.vectors = vectors
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- similarity search ------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if len(self) == 0:
            return []
        scores = self.vectors @ query_vec            # cosine similarity
        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]    # k best (unordered)
        top = top[np.argsort(-scores[top])]          # sort them best-first
        return [(self.chunks[i], float(scores[i])) for i in top]

    # ---- persistence ------------------------------------------------------
    def save(self, folder: Path, fp: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"fingerprint": fp, "chunks": [c.to_dict() for c in self.chunks]}
        (folder / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path, fp: str) -> "VectorStore | None":
        vec_path, meta_path = folder / "vectors.npy", folder / "chunks.json"
        if not (vec_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != fp:
                return None
            chunks = [Chunk(**c) for c in meta["chunks"]]
            return cls(np.load(vec_path), chunks)
        except Exception:
            return None
