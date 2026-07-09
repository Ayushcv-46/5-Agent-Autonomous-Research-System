# ui/components.py
# UI components: NODE_LABELS, render_ragas_metrics, render_assistant_extras.

from typing import Dict, Any, Optional

import streamlit as st

from evaluation.ragas_eval import RAGAS_METRIC_NAMES
from utils.helpers import safe_filename_from


NODE_LABELS = {
    "planner": "Planner Agent — breaking the topic into sub-questions",
    "searcher": "Searcher Agent — searching the web",
    "reader": "Reader Agent — extracting grounded answers (+ RAGAS eval)",
    "writer": "Writer Agent — drafting the report",
    "critic": "Critic Agent — evaluating quality",
    "insufficient_evidence": "Insufficient evidence found — skipping Writer and Critic",
    "writer_failure": "Writer failed — LLM backend unavailable",
    "finalize": "Finalizing report",
}

def render_ragas_metrics(ragas_scores: Optional[Dict[str, Any]]):
    """NEW DAY 23: 4-column RAGAS metric row (or an error caption)."""
    if not ragas_scores:
        return
    if "error" in ragas_scores:
        st.caption(f"RAGAS evaluation unavailable: {ragas_scores['error']}")
        return
    st.caption("RAGAS — Reader Agent retrieval quality (0.0 – 1.0)")
    cols = st.columns(4)
    for i, name in enumerate(RAGAS_METRIC_NAMES):
        val = ragas_scores.get(name)
        cols[i].metric(
            label=name.replace("_", " ").title(),
            value=f"{val:.2f}" if val is not None else "—",
        )

def render_assistant_extras(entry: Dict[str, Any], key_prefix: str):
    score = entry.get("score")
    verdict = entry.get("verdict")
    revision_count = entry.get("revision_count", 0)
    sources = entry.get("sources", [])
    report = entry.get("content", "")
    topic = entry.get("topic", "report")

    if score is not None:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Quality Score", f"{score}/10", verdict)
        with col2:
            if revision_count > 0:
                st.caption(f"Revised {revision_count} time{'s' if revision_count != 1 else ''} before this result.")
            else:
                st.caption("Passed on the first draft — no revisions needed.")

    render_ragas_metrics(entry.get("ragas_scores"))   # NEW DAY 23

    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for url in sources:
                st.markdown(f"- {url}")

    st.download_button(
        label="Download report (.md)",
        data=report,
        file_name=f"{safe_filename_from(topic)}.md",
        mime="text/markdown",
        key=f"dl_{key_prefix}",
    )
