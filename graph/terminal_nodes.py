# graph/terminal_nodes.py
# Non-agent terminal nodes: insufficient evidence, writer failure, finalize.

from schemas.state import AutoResearchState


def input_error_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    new_state["final_report"] = state.get("input_error") or "Enter a research topic to begin."
    return new_state


def insufficient_evidence_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    new_state["final_report"] = "Insufficient evidence retrieved — cannot generate a grounded report"
    return new_state

def writer_failure_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    prior_report = state.get("best_report") or state.get("draft_report")
    if prior_report:
        new_state["final_report"] = (
            "Writer could not complete the latest revision. Showing the best "
            "usable draft produced so far; it may not include all requested fixes.\n\n"
            + prior_report
        )
    else:
        new_state["final_report"] = (
            "Report generation failed because the LLM backend was unavailable "
            "after multiple attempts. Please retry."
        )
    return new_state

def finalize_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    if "final_report" not in new_state:
        report = state.get("best_report") or state.get("draft_report") or (
            "No usable report was generated. Check the agent trace and retry."
        )
        if state.get("critic_failed"):
            report = (
                "⚠️ Note: automated quality review could not be completed. "
                "This report has not been critic-verified.\n\n" + report
            )
        new_state["final_report"] = report
    return new_state
