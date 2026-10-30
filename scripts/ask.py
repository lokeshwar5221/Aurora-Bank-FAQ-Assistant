"""Command-line demo of the RAG pipeline (no Streamlit needed).

    python scripts/ask.py "What is the minimum balance for a savings account?"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import POLICY_DIR, UPLOAD_DIR, get_llm_settings  # noqa: E402
from rag.generator import (NOT_FOUND_MESSAGE, LLMClient, stream_answer,  # noqa: E402
                           unverified_numbers)
from rag.pipeline import KnowledgeBase  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "What is the minimum balance for a regular savings account?"
    kb = KnowledgeBase([POLICY_DIR, UPLOAD_DIR]).build()
    print(f"Knowledge base: {kb.stats()}\n")
    result = kb.retriever.retrieve(question, top_k=4)
    print(f"Question: {question}")
    print(f"Best similarity: {result.best_score:.2f} (threshold {result.min_score:.2f})\n")
    for h in result.closest:
        flag = "USED " if h in result.hits else "below"
        print(f"  [{flag}] {h.score:.2f}  {h.chunk.label}")
    if not result.answerable:
        print("\n" + NOT_FOUND_MESSAGE)
        return
    api_key, model = get_llm_settings()
    if not api_key:
        print("\n(No GROQ_API_KEY set - showing the best passage instead)\n")
        print(result.hits[0].chunk.text)
        return
    print("\nAnswer:\n")
    text = "".join(stream_answer(LLMClient(api_key, model), question, result.hits, []))
    print(text or NOT_FOUND_MESSAGE)
    if text and (bad := unverified_numbers(text, result.hits)):
        print(f"\nWarning: could not verify figures in the sources: {bad}")


if __name__ == "__main__":
    main()
