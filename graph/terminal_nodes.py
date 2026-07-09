# graph/terminal_nodes.py
# Non-agent terminal nodes: insufficient evidence, writer failure, finalize.

from schemas.state import AutoResearchState


def insufficient_evidence_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    new_state["final_report"] = "Insufficient evidence retrieved — cannot generate a grounded report"
    return new_state

def writer_failure_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    new_state["final_report"] = (
        "Report generation failed — the LLM backend was unavailable after "
        "multiple attempts. Please retry."
    )
    return new_state

def finalize_node(state: AutoResearchState) -> AutoResearchState:
    new_state = dict(state)
    if "final_report" not in new_state:
        report = state.get("best_report", state["draft_report"])
        if state.get("critic_failed"):
            report = (
                "⚠️ Note: automated quality review could not be completed (API timeout). "
                "This report has not been critic-verified.\n\n" + report
            )
        new_state["final_report"] = report
    return new_state
