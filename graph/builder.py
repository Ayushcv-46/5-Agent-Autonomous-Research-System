# graph/builder.py
# Builds and compiles the LangGraph state machine.

import time

from langgraph.graph import StateGraph, END

from schemas.state import AutoResearchState
from agents.planner import planner_node
from agents.searcher import search_node
from agents.reader import reader_node
from agents.writer import writer_node
from agents.critic import critic_node
from graph.terminal_nodes import input_error_node, insufficient_evidence_node, writer_failure_node, finalize_node
from graph.routing import should_revise, route_after_planner, route_after_reader, route_after_writer


def _timed_node(node_name, node):
    def run(state):
        started = time.perf_counter()
        try:
            update = dict(node(state))
        except Exception as exc:
            update = dict(state)
            errors = dict(state.get("node_errors", {}))
            errors[node_name] = f"{type(exc).__name__}: {exc}"
            update["node_errors"] = errors
            print(f"[GRAPH][{node_name.upper()}][ERROR] {errors[node_name]}")
            if node_name == "reader_seconds":
                update.update({"reader_failed": True, "reader_error": errors[node_name], "extracted_points": []})
            elif node_name == "writer_seconds":
                update["writer_failed"] = True
            elif node_name == "critic_seconds":
                update["critic_failed"] = True
                update["critique"] = {
                    "quality_score": None,
                    "issues": ["Critic crashed; report is unverified."],
                    "verdict": "UNVERIFIED",
                }
                update["best_report"] = state.get("best_report") or state.get("draft_report", "")
            elif node_name == "searcher_seconds":
                update["search_results"] = {q: [] for q in state.get("sub_questions", [])}
        timings = dict(state.get("stage_timings", {}))
        timings.update(update.get("stage_timings", {}))
        timings[node_name] = round(
            timings.get(node_name, 0.0) + time.perf_counter() - started,
            3,
        )
        update["stage_timings"] = timings
        return update

    return run


def build_graph():
    graph = StateGraph(AutoResearchState)
    graph.add_node("planner", _timed_node("planner_seconds", planner_node))
    graph.add_node("input_error", input_error_node)
    graph.add_node("searcher", _timed_node("searcher_seconds", search_node))
    graph.add_node("reader", _timed_node("reader_seconds", reader_node))
    graph.add_node("insufficient_evidence", insufficient_evidence_node)
    graph.add_node("writer", _timed_node("writer_seconds", writer_node))
    graph.add_node("writer_failure", writer_failure_node)
    graph.add_node("critic", _timed_node("critic_seconds", critic_node))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"input_error": "input_error", "continue": "searcher"},
    )
    graph.add_edge("input_error", "finalize")
    graph.add_edge("searcher", "reader")
    graph.add_conditional_edges("reader", route_after_reader, {"insufficient": "insufficient_evidence", "write": "writer"})
    graph.add_edge("insufficient_evidence", "finalize")
    graph.add_conditional_edges("writer", route_after_writer, {"failed": "writer_failure", "critique": "critic"})
    graph.add_edge("writer_failure", "finalize")
    graph.add_conditional_edges("critic", should_revise, {"revise": "writer", "pass": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()
