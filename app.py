# Run with:  streamlit run app.py
"""Module 5 - Chat Interface.
 
Personal Finance and Banking FAQ Assistant (RAG).
Answers strictly from the bank's policy documents and says clearly when a
question cannot be answered from them.
"""
from __future__ import annotations
 
import html
import re
from pathlib import Path
 
import streamlit as st
 
from rag.config import (ASSISTANT_NAME, BANK_NAME, DEFAULT_TOP_K, POLICY_DIR,
                        SUPPORTED_EXTENSIONS, UPLOAD_DIR, get_llm_settings)
from rag.generator import (NOT_FOUND_MESSAGE, LLMClient, LLMError, extractive_answer,
                           smalltalk_reply, stream_answer, unverified_numbers)
from rag.pipeline import KnowledgeBase
from rag.retriever import Hit, contextualize_query
 
# ============================================================
# 1. Page config + styling
# ============================================================
st.set_page_config(page_title=f"{ASSISTANT_NAME} | {BANK_NAME} FAQ Assistant",
                   page_icon="🏦", layout="wide", initial_sidebar_state="expanded")
 
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
 
NAV_CHAT, NAV_KB, NAV_HOW = "💬  Ask Aria", "📚  Knowledge base", "🧭  How it works"
 
TOPICS = {
    "🏦 Accounts": [
        "What is the minimum balance for a regular savings account?",
        "What is the interest rate on savings accounts?",
        "What documents are needed to open an account?",
    ],
    "💰 Deposits": [
        "What are the fixed deposit interest rates for senior citizens?",
        "What is the penalty for premature FD withdrawal?",
        "When is TDS deducted on FD interest?",
    ],
    "🏠 Loans": [
        "What are the eligibility criteria for a personal loan?",
        "What is the home loan interest rate and maximum tenure?",
        "What is the loan-to-value ratio for home loans?",
    ],
    "💳 Cards": [
        "What is the late payment fee on credit cards?",
        "What is the interest rate on unpaid credit card dues?",
        "What should I do if my card is lost?",
    ],
    "🧾 Fees": [
        "What are the charges for IMPS, NEFT and RTGS?",
        "How many free ATM transactions do I get?",
        "What is the annual rent for a safe deposit locker?",
    ],
    "🛡️ Safety": [
        "How do I report a fraudulent transaction?",
        "What are the UPI transaction limits?",
        "How do I file a complaint with the bank?",
    ],
}
 
# ============================================================
# 2. Cached resources (built once, reused across reruns)
# ============================================================
@st.cache_resource(show_spinner="Reading policy documents and building the search index... "
                                "(the first run also downloads the embedding model)")
def load_kb(version: int) -> KnowledgeBase:
    return KnowledgeBase([POLICY_DIR, UPLOAD_DIR]).build()
 
 
@st.cache_resource(show_spinner=False)
def load_llm(api_key: str, model: str) -> LLMClient:
    return LLMClient(api_key, model)
 
 
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("kb_version", 0)
 
kb = load_kb(ss.kb_version)
api_key, model_name = get_llm_settings()
llm: LLMClient | None = None
llm_error: str | None = None
if api_key:
    try:
        llm = load_llm(api_key, model_name)
    except Exception as exc:  # e.g. openai package missing
        llm_error = str(exc)
 
ss.setdefault("top_k", DEFAULT_TOP_K)
ss.setdefault("min_score", float(kb.embedder.default_min_score))
 
# ============================================================
# 3. Small rendering helpers
# ============================================================
def esc(text: str) -> str:
    return html.escape(str(text), quote=True)
 
 
def hit_to_dict(h: Hit) -> dict:
    c = h.chunk
    return {"rank": h.rank, "title": c.doc_title, "section": c.section, "page": c.page,
            "label": c.label, "text": c.text, "score": round(h.score, 3)}
 
 
def slips_html(items: list[dict], weak: bool = False) -> str:
    out = []
    for it in items:
        pct = max(2, min(100, int(round(it["score"] * 100))))
        page = f" (page {it['page']})" if it.get("page") else ""
        cls = "slip weak" if weak else "slip"
        out.append(
            f'<div class="{cls}"><div class="slip-no">{it["rank"]}</div><div class="slip-main">'
            f'<div class="slip-title">{esc(it["title"])}</div>'
            f'<div class="slip-sec">{esc(it["section"])}{esc(page)}</div>'
            f'<div class="meter"><i style="width:{pct}%"></i></div>'
            f'<div class="slip-score">Match {pct}%</div></div></div>'
        )
    return '<div class="slips">' + "".join(out) + "</div>"
 
 
def render_sources(items: list[dict], label: str, expander_label: str, weak: bool = False) -> None:
    if not items:
        return
    st.markdown(f'<div class="slips-label">{esc(label)}</div>' + slips_html(items, weak),
                unsafe_allow_html=True)
    with st.expander(expander_label):
        for it in items:
            page = f" (page {it['page']})" if it.get("page") else ""
            st.markdown(f"**[{it['rank']}] {it['title']}** - {it['section']}{page}")
            st.markdown(it["text"])
            st.divider()
 
 
def not_found_card(extra: str = "") -> str:
    body = esc(NOT_FOUND_MESSAGE).replace("\n\n", "</p><p>")
    body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
    return (f'<div class="notfound"><h4>Not covered in the policy documents</h4><p>{body}</p>{extra}</div>')
 
 
def render_body(m: dict, include_text: bool = True) -> None:
    """Render one assistant message (text + badges + sources)."""
    status = m["status"]
    if status == "smalltalk":
        if include_text:
            st.markdown(m["content"])
        return
    if status == "error":
        st.error(m["content"])
        return
    if status == "not_found":
        st.markdown(not_found_card(), unsafe_allow_html=True)
        render_sources(m.get("closest", []),
                       "Closest passages found (not relevant enough to answer from)",
                       "See the closest passages", weak=True)
        return
    if status == "extractive":
        if include_text:
            st.markdown('<div class="notice info"><b>Excerpt mode.</b> No AI key is configured, so here are the '
                        'most relevant policy passages word for word.</div>', unsafe_allow_html=True)
            st.markdown(m["content"])
        render_sources(m["sources"], "Matching passages", "Read the exact policy text")
        return
    # status == "answered"
    if include_text:
        st.markdown(m["content"])
    n = len(m["sources"])
    best = int(round(m["sources"][0]["score"] * 100))
    st.markdown(f'<div class="stamps"><span class="badge ok">✓ Grounded in {n} policy passage{"s" if n != 1 else ""}</span>'
                f'<span class="badge info">Best match {best}%</span></div>', unsafe_allow_html=True)
    if m.get("warnings"):
        figs = ", ".join(m["warnings"])
        st.markdown(f'<div class="notice warn"><b>Please double-check:</b> the figure(s) {esc(figs)} could not be '
                    f'matched to the retrieved policy text. Compare with the sources below.</div>',
                    unsafe_allow_html=True)
    render_sources(m["sources"], "Sources used in this answer", "Read the exact policy text")
 
 
def render_message(m: dict) -> None:
    if m["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(m["content"])
    else:
        with st.chat_message("assistant", avatar="🏦"):
            render_body(m)
 
 
# ============================================================
# 4. The RAG answer flow for one question
# ============================================================
def answer_question(question: str, history: list[dict]) -> dict:
    """Runs inside an assistant chat bubble; renders live and returns the message dict."""
    small = smalltalk_reply(question)
    if small:
        st.markdown(small)
        return {"role": "assistant", "status": "smalltalk", "content": small}
 
    with st.spinner("Searching the policy documents..."):
        query = contextualize_query(question, history)
        result = kb.retriever.retrieve(query, top_k=ss.top_k, min_score=ss.min_score)
 
    # Layer 1: similarity gate - nothing relevant => never call the LLM
    if not result.answerable:
        msg = {"role": "assistant", "status": "not_found", "content": NOT_FOUND_MESSAGE,
               "closest": [hit_to_dict(h) for h in result.closest[:3]]}
        render_body(msg)
        return msg
 
    sources = [hit_to_dict(h) for h in result.hits]
 
    if llm is None:  # no API key: show the passages themselves
        msg = {"role": "assistant", "status": "extractive", "sources": sources,
               "content": extractive_answer(result.hits)}
        render_body(msg)
        return msg
 
    placeholder = st.empty()
    placeholder.markdown('<div class="typing"><span></span><span></span><span></span></div>',
                         unsafe_allow_html=True)
    text = ""
    try:
        for delta in stream_answer(llm, question, result.hits, history):
            text += delta
            placeholder.markdown(text + " ▌")
    except LLMError as exc:
        placeholder.empty()
        msg = {"role": "assistant", "status": "error", "content": str(exc)}
        render_body(msg)
        return msg
 
    # Layer 3: the model itself said the passages do not contain the answer
    if not text.strip():
        placeholder.empty()
        msg = {"role": "assistant", "status": "not_found", "content": NOT_FOUND_MESSAGE,
               "closest": [hit_to_dict(h) for h in result.closest[:3]]}
        render_body(msg)
        return msg
 
    placeholder.markdown(text)
    msg = {"role": "assistant", "status": "answered", "content": text, "sources": sources,
           "warnings": unverified_numbers(text, result.hits)}   # Layer 4: number check
    render_body(msg, include_text=False)
    return msg
 
 
def handle_question(question: str) -> None:
    history = list(ss.messages)
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🏦"):
        reply = answer_question(question, history)
    ss.messages.append(reply)
 
 
# ============================================================
# 5. Pages
# ============================================================
def render_hero() -> None:
    s = kb.stats()
    st.markdown(
        '<div class="hero"><div class="hero-text"><h1>Straight answers from the bank&#39;s own policy documents.</h1>'
        f'<p>Ask about accounts, interest rates, loan eligibility, cards and fees. {esc(ASSISTANT_NAME)} answers only '
        'from the documents, and tells you plainly when something isn&#39;t covered.</p></div>'
        '<div class="sample"><div class="sample-tag">✓ Example of an answer</div>'
        '<div class="sample-q">What is the minimum balance for a Regular Savings account?</div>'
        '<div class="sample-a">It depends on your branch location: ₹5,000 in metro and urban areas, ₹2,500 in semi-urban '
        'areas and ₹1,000 in rural areas [1].</div>'
        '<div class="sample-src"><b>1</b>Savings Account Policy, Minimum balance requirements</div></div></div>'
        '<div class="flow">'
        '<div class="flow-step"><b>1</b><div><h5>Ask in your own words</h5><p>No forms and no banking jargon needed.</p></div></div>'
        f'<div class="flow-step"><b>2</b><div><h5>{esc(ASSISTANT_NAME)} searches {s["documents"]} documents</h5>'
        f'<p>She checks {s["chunks"]} passages for the closest match.</p></div></div>'
        '<div class="flow-step"><b>3</b><div><h5>Get an answer you can verify</h5>'
        '<p>Every answer shows its sources, or says &ldquo;not covered&rdquo;.</p></div></div></div>',
        unsafe_allow_html=True)
 
 
def page_chat() -> None:
    queued = ss.pop("queued_question", None)
    typed = st.chat_input("Ask about accounts, loans, deposits, cards or fees...")
    question = typed or queued
 
    render_hero()
 
    if not ss.messages and not question:
        st.markdown('<div class="welcome-title">What would you like to know?</div>'
                    '<div class="welcome-sub">Pick a topic, tap a question, or type your own below.</div>',
                    unsafe_allow_html=True)
        for tab, (topic, questions) in zip(st.tabs(list(TOPICS)), TOPICS.items()):
            with tab:
                for j, text in enumerate(questions):
                    if st.button(text, key=f"q_{topic}_{j}", use_container_width=True):
                        ss.queued_question = text
                        st.rerun()
        st.markdown('<div class="welcome-sub" style="margin-top:10px">Want to test the guardrail? Ask something '
                    'unrelated, like <i>&ldquo;Who won the cricket world cup?&rdquo;</i> and watch '
                    f'{esc(ASSISTANT_NAME)} decline instead of guessing.</div>', unsafe_allow_html=True)
 
    for m in ss.messages:
        render_message(m)
 
    if question:
        handle_question(question)
 
 
def page_knowledge_base() -> None:
    s = kb.stats()
    st.markdown('<h2 class="page-title">Knowledge base</h2><p class="page-sub">Every answer comes from the documents '
                'listed here. Add your own PDF, Markdown or text files and Aria can answer from them too.</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="tiles"><div class="tile"><b>{s["documents"]}</b><span>documents</span></div>'
        f'<div class="tile"><b>{s["chunks"]}</b><span>indexed passages</span></div>'
        f'<div class="tile"><b>{s["avg_chunk_chars"]}</b><span>avg. characters per passage</span></div>'
        f'<div class="tile"><b>{s["build_seconds"]}s</b><span>{"loaded from saved index" if s["cached"] else "index build time"}</span></div></div>',
        unsafe_allow_html=True)
    st.caption(f"Embedding method: {s['embedder']}")
    if s["fallback_reason"]:
        st.warning("The semantic embedding model could not be loaded, so the keyword (TF-IDF) fallback is in use. "
                   "Check your internet connection and `pip install fastembed`. "
                   f"Details: {s['fallback_reason']}")
 
    per_doc = kb.chunks_per_document()
    rows = [{"Document": d.title, "File": d.source, "Type": d.kind.upper(), "Passages": per_doc.get(d.doc_id, 0)}
            for d in kb.documents]
    st.dataframe(rows, use_container_width=True, hide_index=True)
 
    st.markdown("### Browse the indexed passages")
    if kb.documents:
        choice = st.selectbox("Document", [d.doc_id for d in kb.documents], label_visibility="collapsed",
                              format_func=lambda i: next(d.title for d in kb.documents if d.doc_id == i))
        for c in [c for c in kb.chunks if c.doc_id == choice]:
            with st.expander(c.section + (f"  (page {c.page})" if c.page else "")):
                st.markdown(c.text)
                st.caption(f"Passage ID {c.chunk_id} | {len(c.text)} characters")
 
    st.markdown("### Add documents")
    files = st.file_uploader("Upload policy documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                             accept_multiple_files=True, label_visibility="collapsed")
    c1, c2 = st.columns(2)
    if c1.button("Add to knowledge base", type="primary", use_container_width=True, disabled=not files):
        for f in files:
            (UPLOAD_DIR / Path(f.name).name).write_bytes(f.getvalue())
        load_kb.clear()
        ss.kb_version += 1
        st.toast(f"Added {len(files)} document(s). Rebuilding the index...", icon="✅")
        st.rerun()
    uploaded = [p for p in UPLOAD_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    if c2.button(f"Remove {len(uploaded)} uploaded document(s)", use_container_width=True, disabled=not uploaded):
        for p in uploaded:
            p.unlink()
        load_kb.clear()
        ss.kb_version += 1
        st.rerun()
 
 
def page_how_it_works() -> None:
    st.markdown('<h2 class="page-title">How it works</h2><p class="page-sub">A Retrieval-Augmented Generation (RAG) '
                'pipeline: find the right policy text first, then let the AI write an answer from that text only.</p>',
                unsafe_allow_html=True)
    steps = [
        ("Policy document ingestion", "Policy files (<code>.md</code>, <code>.txt</code>, <code>.pdf</code>) are loaded "
         "and split into passages along their headings, so each passage covers one topic. <code>rag/ingest.py</code>"),
        ("Embedding and indexing", "Each passage becomes a vector (a list of numbers capturing its meaning) and is stored in "
         "a vector index saved on disk. <code>rag/embedder.py</code>, <code>rag/vectorstore.py</code>"),
        ("Retriever", "Your question is embedded the same way. Cosine similarity finds the closest passages. If none is close "
         "enough, the question is treated as not covered. <code>rag/retriever.py</code>"),
        ("Answer generator", "A prompt template gives the AI the passages plus strict rules: use only this text, cite "
         "[1], [2], keep a consistent tone. <code>rag/generator.py</code>"),
        ("Chat interface", "This Streamlit app streams the answer and shows the source passages next to it, so every "
         "answer can be checked. <code>app.py</code>"),
    ]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step"><div class="step-n">{i}</div><div><h4>{t}</h4><p>{d}</p></div></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)
 
    st.markdown("### Four layers that prevent made-up answers")
    guards = [
        ("Similarity gate", "If the best passage scores below the threshold, the AI is never called and Aria says the "
         "question is not covered."),
        ("Grounded prompt", "The AI may use only the numbered passages. Outside knowledge, guesses and estimates are forbidden."),
        ("Refusal token", "If the passages don't contain the answer, the AI must reply with a fixed signal, which the app "
         "turns into a clear &ldquo;not covered&rdquo; message."),
        ("Figure check", "Every number in an answer is compared with the retrieved text. Unmatched figures trigger a "
         "warning to verify."),
    ]
    st.markdown('<div class="guards">' + "".join(
        f'<div class="guard"><b>{t}</b><p>{d}</p></div>' for t, d in guards) + "</div>", unsafe_allow_html=True)
 
    st.markdown("### Course concepts used")
    st.dataframe([
        {"Concept": "Vector databases and similarity search", "Where": "rag/vectorstore.py (cosine similarity index), rag/retriever.py"},
        {"Concept": "Knowledge base from multiple documents", "Where": "rag/ingest.py, rag/pipeline.py, data/policies/ (10 documents)"},
        {"Concept": "Prompt templates for consistent tone", "Where": "rag/generator.py (SYSTEM_PROMPT, USER_TEMPLATE)"},
        {"Concept": "Hallucination mitigation by grounding", "Where": "Similarity gate, grounded prompt, refusal token, figure check"},
    ], use_container_width=True, hide_index=True)
 
 
# ============================================================
# 6. Sidebar + routing
# ============================================================
with st.sidebar:
    stt = kb.stats()
    if llm:
        ai_pill = f'<div class="pill"><span class="dot ok"></span>AI model: {esc(model_name)}</div>'
    else:
        why = "AI key invalid" if llm_error else "No AI key, excerpt mode"
        ai_pill = f'<div class="pill"><span class="dot warn"></span>{why}</div>'
    # One single HTML block (brand + status pills) so the pieces can never overlap.
    st.markdown(
        f'<div class="side-head"><div class="brand"><div class="brand-name">{esc(BANK_NAME)}</div>'
        f'<div class="brand-sub">{esc(ASSISTANT_NAME)}, your FAQ assistant</div></div>'
        f'{ai_pill}'
        f'<div class="pill"><span class="dot ok"></span>{stt["documents"]} documents | {stt["chunks"]} passages</div></div>',
        unsafe_allow_html=True)
 
    with st.expander("Search settings", expanded=False):
        st.slider("Passages to retrieve", 1, 8, key="top_k",
                  help="How many of the closest passages are given to the AI.")
        st.slider("Relevance threshold", 0.0, 0.9, step=0.01, key="min_score",
                  help="Higher = stricter. Questions whose best passage scores below this are answered with 'not covered'.")
 
    if ss.messages:
        transcript = "\n\n".join(
            f"{'You' if m['role'] == 'user' else ASSISTANT_NAME}: {m['content']}" for m in ss.messages)
        st.download_button("Download chat", transcript, file_name="aria_chat.txt", use_container_width=True)
        if st.button("Clear chat", use_container_width=True):
            ss.messages = []
            st.rerun()
 
    st.markdown('<div class="side-note">Aurora National Bank is a fictional bank and the policy documents are sample '
                'data created for an academic project. Not financial advice.</div>', unsafe_allow_html=True)
 
# Navigation lives in the main area so it is always visible, even if the sidebar is collapsed.
page = st.radio("Navigate", [NAV_CHAT, NAV_KB, NAV_HOW], horizontal=True,
                label_visibility="collapsed", key="page")
 
if page == NAV_CHAT:
    page_chat()
elif page == NAV_KB:
    page_knowledge_base()
else:
    page_how_it_works()
