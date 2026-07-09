# agents/planner.py
# Planner Agent — breaks the topic into sub-questions.

from langsmith import traceable

from schemas.state import AutoResearchState
from schemas.pydantic_models import PlannerOutput
from clients.llm_clients import llm
from utils.helpers import invoke_with_retry


@traceable(name="Planner Agent")
def planner_node(state: AutoResearchState) -> AutoResearchState:
    topic = state["topic"]
    structured_llm = llm.with_structured_output(PlannerOutput)
    prompt = f"""You are a research planning assistant.

Given the research topic below, generate exactly 3 focused sub-questions
that together would help a researcher fully understand the topic.

Topic: {topic}

Return exactly 3 sub-questions. Be specific and research-oriented."""
    result = invoke_with_retry(structured_llm, prompt, "PLANNER")
    if result is None:
        raise RuntimeError("Planner Agent failed after 3 attempts.")
    new_state = dict(state)
    new_state["sub_questions"] = result.sub_questions
    return new_state
