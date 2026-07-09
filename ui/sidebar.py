# ui/sidebar.py
# Sidebar: RAGAS toggle + Research History rendering.

import streamlit as st

from evaluation.ragas_eval import RAGAS_AVAILABLE, RAGAS_METRIC_NAMES
from utils.helpers import safe_filename_from


def render_sidebar():
    """Render the sidebar and return (run_ragas_toggle, RAGAS_AVAILABLE)."""
    with st.sidebar:
        st.header("Settings")
        # NEW DAY 23: RAGAS adds 1-3 minutes per run, so it's opt-out-able.
        run_ragas_toggle = st.toggle(
            "Run RAGAS evaluation",
            value=True,
            help="Scores the Reader Agent on faithfulness, answer relevancy, "
                 "context precision, and context recall. Adds 1-3 min per run.",
            disabled=not RAGAS_AVAILABLE,
        )
        if not RAGAS_AVAILABLE:
            st.caption('Install: pip install ragas datasets langchain-huggingface "langchain-community<0.4"')

        st.header("Research History")
        if not st.session_state.research_history:
            st.caption("Completed research runs will appear here.")
        else:
            for i, record in enumerate(reversed(st.session_state.research_history)):
                score_label = f"{record['score']}/10" if record.get("score") is not None else "—"
                r = record.get("ragas_scores") or {}
                faith = r.get("faithfulness")
                ragas_label = f"  ·  F {faith:.2f}" if isinstance(faith, float) else ""
                with st.expander(f"{record['topic'][:45]}  ·  {score_label}{ragas_label}"):
                    st.caption(f"Verdict: {record.get('verdict', 'N/A')}  ·  Revisions: {record.get('revision_count', 0)}")
                    if r and "error" not in r:
                        parts = []
                        for name in RAGAS_METRIC_NAMES:
                            v = r.get(name)
                            parts.append(f"{name.replace('_', ' ')}: {v:.2f}" if v is not None else f"{name}: —")
                        st.caption("RAGAS — " + "  |  ".join(parts))
                    st.markdown(record["report"])
                    if record.get("sources"):
                        st.caption(f"{len(record['sources'])} source(s) cited.")
                    st.download_button(
                        label="Download",
                        data=record["report"],
                        file_name=f"{safe_filename_from(record['topic'])}.md",
                        mime="text/markdown",
                        key=f"dl_hist_{i}",
                    )

    return run_ragas_toggle, RAGAS_AVAILABLE
