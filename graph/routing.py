# graph/routing.py
# Routing functions for conditional edges in the LangGraph.

from typing import Literal

from schemas.state import AutoResearchState
from config.settings import MAX_REVISIONS


def should_revise(state: AutoResearchState) -> Literal["revise", "pass"]:
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)
    if critique is None:
        return "pass"
    if critique["verdict"] == "REVISE" and revision_count < MAX_REVISIONS:
        return "revise"
    return "pass"

def route_after_reader(state: AutoResearchState) -> Literal["insufficient", "write"]:
    return "insufficient" if state.get("reader_failed", False) else "write"

def route_after_writer(state: AutoResearchState) -> Literal["failed", "critique"]:
    return "failed" if state.get("writer_failed", False) else "critique"
