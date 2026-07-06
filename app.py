"""Streamlit demo — ask the agent a question and watch it reason live.

Thin UI over the SAME compiled LangGraph agent used by the eval (imported from
`eval.py`; the agent, chunking, stores and prompts are unchanged). As the graph runs,
each node is streamed to the page — RETRIEVE → GRADE_DOCS → ROUTE → (REWRITE / WEB) →
GENERATE → SELF-CHECK — so a reviewer can see the agent route, rewrite, compute and
self-verify in real time.

Run:  streamlit run app.py
Keys are read from .env by the imported module (never hardcoded).
"""
import streamlit as st

st.set_page_config(page_title="Agentic RAG — HSBC Pillar 3", page_icon="🏦", layout="wide")


@st.cache_resource(show_spinner="Loading agent + vector store…")
def load_agent():
    """Import the compiled agent once per process. eval.py asserts the API keys and
    opens the persisted Chroma collections at import time."""
    import eval as agent
    return agent


try:
    agent = load_agent()
except AssertionError as e:
    st.error(f"Missing API key: {e}. Copy `.env.example` → `.env` and fill in your keys.")
    st.stop()
except Exception as e:  # noqa: BLE001 — surface any load failure to the UI
    st.error(f"Could not load the agent: {type(e).__name__}: {e}")
    st.stop()

STORE_N = agent.structured_store._collection.count()

# ── Trace rendering ───────────────────────────────────────────────────────────
NODE_STYLE = {
    "retrieve":        ("🔍", "RETRIEVE"),
    "grade_documents": ("🧮", "GRADE_DOCS"),
    "route_question":  ("🧭", "ROUTE"),
    "rewrite_query":   ("✏️", "REWRITE"),
    "web_search":      ("🌐", "WEB_SEARCH"),
    "generate":        ("✨", "GENERATE"),
    "grade_answer":    ("✅", "SELF-CHECK"),
}


def _pages(docs):
    return [d.metadata.get("source_page") for d in docs]


def trace_detail(node, delta, working_q):
    """One-line human summary of what a node just did, from its streamed delta."""
    d = delta or {}
    if node == "retrieve":
        return f"query `{working_q}` → pages {_pages(d.get('candidates', []))}"
    if node == "grade_documents":
        kept = d.get("documents", [])
        return f"kept {len(kept)} relevant chunk(s) — pages {_pages(kept)}"
    if node == "route_question":
        dec = d.get("decision", "")
        lane = "web (external)" if dec == "web_search" else "in-domain"
        return f"router → **{lane}**, next: `{dec}`"
    if node == "rewrite_query":
        return (f"rewrite {d.get('retrieval_rewrites')}/{agent.MAX_REWRITES} → "
                f"new query `{d.get('question')}`")
    if node == "web_search":
        return f"Tavily → {len(d.get('web_results',''))} chars of external context"
    if node == "generate":
        snippet = (d.get("generation", "") or "").replace("\n", " ")[:160]
        return f"draft attempt {d.get('generation_retries')} → “{snippet}…”"
    if node == "grade_answer":
        return f"grounded=**{d.get('grounded')}**  useful=**{d.get('useful')}**"
    return str(d)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        "Agentic RAG over **HSBC's Pillar 3 Disclosures (31 Dec 2025)**. "
        "The agent grades retrieved chunks, routes in-domain vs web, rewrites queries, "
        "computes across cells, and self-checks its answer — bounded by "
        f"`MAX_REWRITES={agent.MAX_REWRITES}` / `MAX_RETRIES={agent.MAX_RETRIES}`."
    )
    st.caption(f"Store `pillar3_structured`: {STORE_N} vectors · "
               f"gen `{agent.GEN_MODEL}` · embed `gemini-embedding-001` · top-k {agent.TOP_K}")

    st.subheader("Try an example")
    st.caption("In-domain (stays internal):")
    examples_internal = [
        "What is the combined RWA for market risk and counterparty credit risk at 31 Dec 2025?",
        "Is HSBC's reported CET1 ratio on a transitional or end-point basis?",
        "What total capital charge does HSBC apply as a percentage of RWAs, and under which regulation?",
        "What is HSBC Holdings' total RWA, and in what currency?",
    ]
    for q in examples_internal:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["question"] = q
    st.caption("External (should route to web):")
    web_q = "How does HSBC's CET1 ratio compare to Barclays' latest reported CET1 ratio?"
    if st.button(web_q, key="ex_web", use_container_width=True):
        st.session_state["question"] = web_q

    st.divider()
    st.caption("Answer *wording* varies run-to-run (live gpt-4o-mini); the eval scores "
               "are deterministic. Baseline 3/10 → agentic 10/10 — see docs/eval-results.md.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏦 Agentic RAG — watch it reason")
st.markdown(
    "Ask a capital / leverage / liquidity question about HSBC's Pillar 3 filing and "
    "watch the agent **route → rewrite → compute → self-check** live."
)

question = st.text_input(
    "Your question", key="question",
    placeholder="e.g. What were HSBC's LCR and NSFR at 31 Dec 2025, and the HQLA amount?",
)
go = st.button("Ask the agent", type="primary")

if go and question.strip():
    init = {
        "question": question, "original_question": question, "candidates": [],
        "documents": [], "web_results": "", "decision": "", "generation": "",
        "retrieval_rewrites": 0, "generation_retries": 0, "grounded": False, "useful": False,
    }
    final = dict(init)
    path = []

    st.subheader("Reasoning trace")
    trace_box = st.container()
    try:
        with st.spinner("Agent running…"):
            for upd in agent.app.stream(init, {"recursion_limit": 40}, stream_mode="updates"):
                for node, delta in upd.items():
                    path.append(node)
                    icon, label = NODE_STYLE.get(node, ("•", node.upper()))
                    # `retrieve` uses the working question as it stands *now*
                    detail = trace_detail(node, delta, final.get("question"))
                    trace_box.markdown(f"{icon} **{label}** — {detail}")
                    if delta:
                        final.update(delta)
    except Exception as e:  # noqa: BLE001
        st.error(f"Run failed: {type(e).__name__}: {e}")
        st.stop()

    # ── Result + control-flow proxies ────────────────────────────────────────
    web = "web_search" in path
    grounded_retries = sum(1 for i in range(len(path) - 1)
                           if path[i] == "grade_answer" and path[i + 1] == "generate")

    st.subheader("Answer")
    st.success(final["generation"] or "(no answer generated)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", "web" if web else "in-domain")
    c2.metric("Rewrites", f"{final['retrieval_rewrites']}/{agent.MAX_REWRITES}")
    c3.metric("Retries", f"{grounded_retries}/{agent.MAX_RETRIES}")
    c4.metric("Generate calls", final["generation_retries"])
    c5.metric("Nodes fired", len(path))

    docs = final.get("documents") or []
    if docs:
        st.caption("Grounded on internal pages: "
                   + ", ".join(str(p) for p in sorted({p for p in _pages(docs) if p is not None})))
    if web:
        st.caption("⚠️ Used external web context (labelled in the answer).")
    st.caption(f"Self-check: grounded={final['grounded']} · useful={final['useful']}. "
               "Wording varies per run; eval scoring is deterministic.")

elif go:
    st.warning("Type a question first.")
