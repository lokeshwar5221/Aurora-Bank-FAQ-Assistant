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
| Retriever | `rag/retriever.py` | Top-k similarity search with a relevance threshold; only explicit follow-up wording reuses the previous topic |
| Answer Generator | `rag/generator.py` | Prompt templates, Groq LLM streaming, refusal handling, citation-scoped figure checks |
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
4. **Citation-scoped checks**: numeric claims are checked against the passage(s) they cite, including product, conditions and units; missing or invalid citations are flagged. These checks are a warning, not proof that all factual claims are correct.

### Policy versions

Markdown policies may start with this optional metadata block:

```text
---
policy_id: home-loans
version: 2.1
effective_date: 2026-03-01
supersedes: 2.0, previous-policy.md
---
```

Documents are grouped by `policy_id` (or title when no ID is supplied). An explicit `supersedes` entry may identify an older document by version, filename or document ID. Otherwise the unique latest ISO `effective_date` is used. If a group has no unique active version, the whole group is excluded from retrieval and reported on the knowledge-base page rather than asking the model to choose. Version and effective-date metadata stay attached to retrieved passages and citations.

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
├── scripts/
│   ├── ask.py              # command-line demo
│   └── evaluate_live.py    # opt-in live evaluation runner
├── tests/                  # offline ingestion, retrieval, streaming and verification tests
│   └── evaluations/        # recorded live semantic-embedder/LLM evaluation
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
- **Run the offline tests** (no API key or internet needed):
  ```bash
  python tests/test_pipeline.py
  # or: python -m pytest -q
  ```
- **Command-line demo:** `python scripts/ask.py "What is the NEFT charge?"` shows every retrieved passage with its similarity score. This is useful to explain retrieval in a viva.
- **Add your own documents:** drop `.md`, `.txt` or `.pdf` files into `data/policies/` (or upload in the *Knowledge base* page). The index rebuilds automatically.

The offline suite uses TF-IDF and simulated LLM streams. A separate 10-case evaluation using the semantic embedder and Groq is recorded in `tests/evaluations/live_results.json` (6 passed, 4 failed); the failed cases expose missing citations and an unsupported condition paraphrase. To collect fresh outputs and judgments:

```bash
python scripts/evaluate_live.py --confirm-live
```

This sends questions and retrieved passages from the bundled fictional policies to Groq and may consume API quota. The report records expected answers, retrieved passages, actual model output and pass/fail judgments. To recompute judgments from saved output without API calls:

```bash
python scripts/evaluate_live.py --rejudge tests/evaluations/live_results.json
```

The local test environment reported Python 3.13.13, pytest 8.4.2, Streamlit 1.49.1, NumPy 2.5.4, scikit-learn 1.9.1, FastEmbed 0.9.1, pypdf 6.21.4 and OpenAI SDK 3.41.0. These are recorded environment versions, not a fully locked dependency set.

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
