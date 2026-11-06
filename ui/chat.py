# ui/chat.py
# Main chat loop: message rendering, topic input, graph streaming.

import time
from typing import List

import streamlit as st

from schemas.state import AutoResearchState
from graph.builder import build_graph
from utils.helpers import extract_unique_sources
from ui.components import NODE_LABELS, render_assistant_extras


def render_chat(run_ragas_toggle: bool):
    """Render the chat history and handle new topic submissions."""

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "score" in msg:
                render_assistant_extras(msg, key_prefix=f"hist_{idx}")

    topic = st.chat_input("Enter a research topic...")

    if topic:
        st.session_state.messages.append({"role": "user", "content": topic})
        with st.chat_message("user"):
            st.markdown(topic)

        initial_state: AutoResearchState = {
            "topic": topic,
            "sub_questions": [],
            "search_results": {},
            "extracted_points": [],
            "reader_failed": False,
            "writer_failed": False,
            "critic_failed": False,
            "best_report": "",
            "best_score": -1,
            "revision_count": 0,
            "critique": None,
            "run_ragas": bool(run_ragas_toggle),   # NEW DAY 23
            "ragas_scores": None,                  # NEW DAY 23
            "stage_timings": {},
            "planner_fallback_used": False,
            "citation_warnings": [],
        }

        app = build_graph()
        running_state = dict(initial_state)

        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            progress_lines: List[str] = []
            final_report = "Something went wrong before a report could be generated."
            run_started = time.perf_counter()

            try:
                for step in app.stream(initial_state):
                    for node_name, update in step.items():
                        running_state.update(update)
                        line = NODE_LABELS.get(node_name, node_name)

                        if node_name == "planner" and update.get("sub_questions"):
                            sub_qs = "; ".join(update["sub_questions"])
                            line += f" — {sub_qs}"
                            if update.get("planner_fallback_used"):
                                line += " (fallback plan used)"
                        if node_name == "critic" and update.get("critique"):
                            c = update["critique"]
                            line += f" — score {c.get('quality_score')}/10, verdict {c.get('verdict')}"
                        # NEW DAY 23: show faithfulness inline in the trace
                        if node_name == "reader":
                            r = update.get("ragas_scores") or {}
                            f = r.get("faithfulness")
                            if isinstance(f, float):
                                line += f" — RAGAS faithfulness {f:.2f}"
                            t = update.get("stage_timings") or {}
                            if t:
                                line += (
                                    f" — fetch wall {t.get('page_fetch_wall_seconds', 0):.1f}s, "
                                    f"index {t.get('chunk_and_index_seconds', 0):.1f}s, "
                                    f"retrieve {t.get('question_retrieval_seconds', 0):.1f}s, "
                                    f"RAGAS {t.get('ragas_seconds', 0):.1f}s"
                                )

                        node_time = (update.get("stage_timings") or {}).get(
                            {
                                "planner": "planner_seconds",
                                "searcher": "searcher_seconds",
                                "reader": "reader_seconds",
                                "writer": "writer_seconds",
                                "critic": "critic_seconds",
                            }.get(node_name, "")
                        )
                        if isinstance(node_time, (int, float)):
                            line += f" ({node_time:.1f}s)"

                        progress_lines.append(line)
                        progress_placeholder.markdown(
                            "\n".join(f"*{l}*" for l in progress_lines)
                        )

                final_report = running_state.get("final_report", "No report generated.")
            except Exception as e:
                final_report = f"Something went wrong: {e}"

            progress_placeholder.empty()
            st.markdown(final_report)
            if progress_lines:
                with st.expander("Agent trace"):
                    st.markdown("\n".join(f"- {l}" for l in progress_lines))

            critique = running_state.get("critique") or {}
            total_seconds = round(time.perf_counter() - run_started, 3)
            stage_timings = dict(running_state.get("stage_timings", {}))
            stage_timings["total_seconds"] = total_seconds
            record = {
                "topic": topic,
                "content": final_report,
                "report": final_report,
                "score": critique.get("quality_score"),
                "verdict": critique.get("verdict"),
                "revision_count": running_state.get("revision_count", 0),
                "sources": extract_unique_sources(running_state.get("extracted_points", [])),
                "ragas_scores": running_state.get("ragas_scores"),   # NEW DAY 23
                "stage_timings": stage_timings,
                "total_seconds": total_seconds,
                "citation_warnings": running_state.get("citation_warnings", []),
            }

            render_assistant_extras(record, key_prefix=f"new_{len(st.session_state.messages)}")

        st.session_state.messages.append({"role": "assistant", **record})
        st.session_state.research_history.append(record)
