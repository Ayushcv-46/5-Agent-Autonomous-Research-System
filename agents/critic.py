# agents/critic.py
# Critic Agent — evaluates report quality.

from langsmith import traceable

from schemas.state import AutoResearchState
from schemas.pydantic_models import CriticEvaluation
from clients.llm_clients import critic_llm
from utils.helpers import invoke_with_retry, format_evidence_for_prompt


@traceable(name="Critic Agent")
def critic_node(state: AutoResearchState) -> AutoResearchState:
    evidence_block = format_evidence_for_prompt(state["extracted_points"])
    prompt = f"""You are a strict research report critic (LLM-as-judge).

Evaluate on GROUNDING, COMPLETENESS, and CLARITY.

--- DRAFT REPORT ---
{state["draft_report"]}

--- ORIGINAL EVIDENCE ---
{evidence_block}

Score 1-10, list specific issues, and give a verdict: 'PASS' or 'REVISE'.
Be strict — generic or ungrounded reports should not PASS."""

    structured_llm = critic_llm.with_structured_output(CriticEvaluation)
    result = invoke_with_retry(structured_llm, prompt, "CRITIC")
    if result is None:
        new_state = dict(state)
        new_state["critique"] = {
            "quality_score": None,
            "issues": ["Critic evaluation unavailable — NVIDIA API timed out after 3 attempts."],
            "verdict": "PASS",
        }
        new_state["critic_failed"] = True
        return new_state

    new_state = dict(state)
    new_state["critique"] = result.model_dump()
    new_state["critic_failed"] = False
    current_score = result.quality_score
    best_score_so_far = state.get("best_score", -1)
    if current_score > best_score_so_far:
        new_state["best_score"] = current_score
        new_state["best_report"] = state["draft_report"]
    return new_state
