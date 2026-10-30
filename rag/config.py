"""Central configuration.

Secrets are read from the environment (.env file, local) or from
st.secrets (Streamlit Community Cloud) - the same approach used in the
reference GENAI_PROJECT app.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "policies"      # bundled bank policy documents
UPLOAD_DIR = BASE_DIR / "data" / "uploads"       # documents added from the UI
STORAGE_DIR = BASE_DIR / "storage"               # persisted vector index
MODEL_CACHE_DIR = STORAGE_DIR / "models"         # downloaded embedding model

for _d in (POLICY_DIR, UPLOAD_DIR, STORAGE_DIR, MODEL_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BANK_NAME = "Aurora National Bank"
ASSISTANT_NAME = "Aria"
SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf")

# ---- Chunking -------------------------------------------------------------
CHUNK_SIZE = 1100        # max characters per chunk
CHUNK_OVERLAP = 150      # characters carried over between chunks of one section

# ---- Retrieval ------------------------------------------------------------
DEFAULT_TOP_K = 4


def get_config_value(key: str, default: str | None = None) -> str | None:
    """Read from environment (.env / local) or st.secrets (Streamlit Cloud)."""
    value = os.getenv(key)
    if value:
        return value
    try:  # imported lazily so the engine also works without Streamlit
        import streamlit as st

        return st.secrets[key]
    except Exception:
        return default


def get_llm_settings() -> tuple[str | None, str]:
    api_key = get_config_value("GROQ_API_KEY")
    model = get_config_value("MODEL_NAME", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    return api_key, model


def get_embedder_preference() -> str:
    """'auto' (default), 'fastembed' or 'tfidf'."""
    return (get_config_value("EMBEDDER", "auto") or "auto").lower()
