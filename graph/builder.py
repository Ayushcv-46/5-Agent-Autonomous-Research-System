# graph/builder.py
# Builds and compiles the LangGraph state machine.

from langgraph.graph import StateGraph, END

from schemas.state import AutoResearchState
from agents.planner import planner_node
from agents.searcher import search_node
from agents.reader import reader_node
from agents.writer import writer_node
from agents.critic import critic_node
from graph.terminal_nodes import insufficient_evidence_node, writer_failure_node, finalize_node
from graph.routing import should_revise, route_after_reader, route_after_writer


def build_graph():
    graph = StateGraph(AutoResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("insufficient_evidence", insufficient_evidence_node)
    graph.add_node("writer", writer_node)
    graph.add_node("writer_failure", writer_failure_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "reader")
    graph.add_conditional_edges("reader", route_after_reader, {"insufficient": "insufficient_evidence", "write": "writer"})
    graph.add_edge("insufficient_evidence", "finalize")
    graph.add_conditional_edges("writer", route_after_writer, {"failed": "writer_failure", "critique": "critic"})
    graph.add_edge("writer_failure", "finalize")
    graph.add_conditional_edges("critic", should_revise, {"revise": "writer", "pass": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()
