"""Module 2a - Embedding.

Turns text into vectors so that "similar meaning" becomes "nearby in space".

Two interchangeable embedders:
  * FastEmbedder  - semantic sentence embeddings (all-MiniLM-L6-v2, ONNX, no
                    PyTorch needed). Understands paraphrases such as
                    "min balance" ~ "minimum monthly average balance".
  * TfidfEmbedder - keyword-based fallback (scikit-learn). Used automatically
                    if the embedding model cannot be downloaded (e.g. offline).

All vectors are L2-normalised, so cosine similarity == dot product.
"""
from __future__ import annotations

import re

import numpy as np

from .config import MODEL_CACHE_DIR

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(token: str) -> str:
    """Very light plural stemming so 'lockers' matches 'locker', 'charges' matches 'charge'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class BaseEmbedder:
    name = "base"
    label = "base"
    default_min_score = 0.30     # similarity below this => "not in the documents"
    persistable = False          # can the index be cached on disk safely?

    def fit(self, texts: list[str]) -> None:  # only TF-IDF needs this
        pass

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_documents([text])[0]


class FastEmbedder(BaseEmbedder):
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    name = "fastembed:all-MiniLM-L6-v2"
    label = "MiniLM-L6 semantic embeddings"
    default_min_score = 0.30
    persistable = True

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=self.MODEL, cache_dir=str(MODEL_CACHE_DIR))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return _normalise(vectors)


class TfidfEmbedder(BaseEmbedder):
    name = "tfidf"
    label = "TF-IDF keyword vectors (fallback)"
    default_min_score = 0.12
    persistable = False

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return _normalise(mat)


def get_embedder(preference: str = "auto") -> tuple[BaseEmbedder, str | None]:
    """Return (embedder, fallback_reason). fallback_reason is None if all went well."""
    preference = (preference or "auto").lower()
    if preference == "tfidf":
        return TfidfEmbedder(), None
    try:
        return FastEmbedder(), None
    except Exception as exc:  # missing package, no internet, corrupted cache...
        reason = f"{type(exc).__name__}: {exc}"
        if preference == "fastembed":
            raise
        return TfidfEmbedder(), reason
