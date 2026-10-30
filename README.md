# 🏦 Personal Finance and Banking FAQ Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot that answers customer questions about account types, interest rates, loan eligibility, cards and fees **strictly from a bank's policy documents**, and clearly says when a question **cannot** be answered from them.

Built with **Python, Streamlit, Groq (OpenAI-compatible API), sentence embeddings and a vector index**. It uses the same stack and configuration style (`.env` / `st.secrets`, Groq client, retries, prompt templates) as the reference `GENAI_PROJECT` app.

> **Note:** Aurora National Bank is a **fictional bank** and the 10 policy documents in `data/policies/` are sample data written for this academic project.

---

## Features mapped to the assignment

| Required module | File | What it does |
|---|---|---|
| Policy Document Ingestion | `rag/ingest.py` | Loads `.md`, `.txt`, `.pdf`; splits on headings into topic-focused passages that remember document, section and page |
| Embedding and Indexing | `rag/embedder.py`, `rag/vectorstore.py` | MiniLM sentence embeddings (TF-IDF fallback); cosine-similarity vector index saved on disk and rebuilt only when documents change |
| Retriever | `rag/retriever.py` | Top-k similarity search with a relevance threshold; follow-up questions are combined with the previous topic |
| Answer Generator | `rag/generator.py` | Prompt templates, Groq LLM streaming, refusal handling, figure verification |
| Chat Interface | `app.py`, `assets/style.css` | Streamlit chat with topic tabs of ready-made questions, streaming answers, numbered source slips, knowledge-base browser, document upload |

| Course concept | Where |
|---|---|
| Vector databases and similarity search | `VectorStore.search()` (cosine similarity), `Retriever` |
| Knowledge base from multiple documents | 10 policy documents, `KnowledgeBase` in `rag/pipeline.py`, upload in the UI |
| Prompt templates for consistent tone | `SYSTEM_PROMPT` and `USER_TEMPLATE` in `rag/generator.py` |
| Hallucination mitigation by grounding | Four layers, below |

### Four layers against hallucination
1. **Similarity gate**: if the best passage scores below the threshold, the LLM is never called and the app answers "not covered in the policy documents".
2. **Grounded prompt**: the model may use only the numbered passages; guessing, estimating and outside knowledge are forbidden; every fact must cite `[n]`.
3. **Refusal token**: if the passages lack the answer, the model must output `[[NOT_IN_POLICY]]`, which the app converts into a clear "cannot answer" message.
4. **Figure check**: every number in the answer is compared with the retrieved text; unmatched figures show a "please double-check" warning.

---

## Project structure

```
aurora-bank-faq-assistant/
├── app.py                  # Streamlit chat interface (run this)
├── assets/style.css        # UI styling
├── rag/
│   ├── config.py           # settings, .env / st.secrets loading
│   ├── ingest.py           # document loading + chunking
│   ├── embedder.py         # embeddings (MiniLM, TF-IDF fallback)
│   ├── vectorstore.py      # vector index + similarity search + persistence
│   ├── retriever.py        # top-k retrieval + threshold gate
│   ├── generator.py        # prompts, LLM client, guardrails
│   └── pipeline.py         # KnowledgeBase: ingest -> embed -> index
├── data/policies/          # 10 bank policy documents (the knowledge base)
├── data/uploads/           # documents added from the UI
├── tests/test_pipeline.py  # 11 automated tests
├── scripts/ask.py          # command-line demo
├── requirements.txt
├── .env.example
└── .streamlit/             # theme + secrets example
```

---

## Setup (about 5 minutes)

1. **Install Python 3.10+**, then open a terminal in this folder.
2. **Create a virtual environment and install packages**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```
3. **Add your Groq API key** (free at https://console.groq.com/keys): copy `.env.example` to `.env` and paste the key.
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
4. **Run**
   ```bash
   streamlit run app.py
   ```
   The first start downloads the small embedding model (~90 MB) and builds the index; later starts are quick.

**No API key?** The app still works in *excerpt mode*: it retrieves and shows the exact policy passages without AI-written answers.

### Deploy free on Streamlit Community Cloud
Push this folder to GitHub, create an app pointing at `app.py`, and in **Advanced settings > Secrets** paste:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

---

## Try these questions

**Should be answered (with sources):**
- What is the minimum balance for a regular savings account?
- What are the FD interest rates for senior citizens?
- What are the eligibility criteria for a personal loan?
- What is the home loan processing fee?
- How do I report a fraudulent transaction?
- What is the late payment fee on credit cards?
- *Follow-up:* "and for women borrowers?" after a home loan question

**Should be declined (not in the documents):**
- Who won the cricket world cup?
- Can I buy bitcoin through the bank?
- What is the interest rate at another bank?

---

## Tuning and testing

- **Relevance threshold** (sidebar > Search settings): raise it if the bot answers unrelated questions, lower it if it refuses valid ones. Defaults: `0.30` for MiniLM, `0.12` for TF-IDF.
- **Run the tests** (no API key or internet needed):
  ```bash
  python tests/test_pipeline.py
  ```
- **Command-line demo:** `python scripts/ask.py "What is the NEFT charge?"` shows every retrieved passage with its similarity score. This is useful to explain retrieval in a viva.
- **Add your own documents:** drop `.md`, `.txt` or `.pdf` files into `data/policies/` (or upload in the *Knowledge base* page). The index rebuilds automatically.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No AI key, excerpt mode" in the sidebar | Create `.env` with `GROQ_API_KEY`, then restart |
| Warning that TF-IDF fallback is in use | The embedding model could not download. Check internet, run `pip install fastembed`, restart |
| `TypeError ... use_container_width` | Install the pinned version: `pip install "streamlit>=1.40,<1.50"` |
| Model name error from Groq | Set `MODEL_NAME` in `.env` to a model available in your Groq console |
| Answers seem to refuse too much | Lower the relevance threshold in the sidebar |

---

## Possible extensions
Hybrid keyword + semantic search, a re-ranker, chat feedback buttons, multilingual support (Telugu / Hindi), an EMI calculator tab, and ChromaDB or FAISS in place of the built-in vector index.
