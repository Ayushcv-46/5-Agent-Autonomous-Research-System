# agents/writer.py
# Writer Agent — drafts and revises the research report.

from langsmith import traceable

from schemas.state import AutoResearchState
from schemas.pydantic_models import DraftReport
from clients.llm_clients import writer_llm
from utils.helpers import invoke_with_retry, format_evidence_for_prompt, render_markdown


@traceable(name="Writer Agent")
def writer_node(state: AutoResearchState) -> AutoResearchState:
    evidence_block = format_evidence_for_prompt(state["extracted_points"])
    critique = state.get("critique")

    if critique is None:
        prompt = f"""You are a research report writer.

Using ONLY the evidence below, write a structured research report.
Every finding must be grounded in the evidence and cite its source.

CRITICAL RULES — DO NOT VIOLATE THESE:
- You MUST NOT invent citations, author names, journal names, publication
    years, statistics, percentages, or study results that do not appear in
    the evidence below.
- Every specific number, percentage, or named source in your report MUST
    trace back to something explicitly stated in the evidence block.
- If the evidence for a sub-question is thin, general, or lacks specific
    data, you MUST say so explicitly rather than filling the gap with a
    fabricated statistic or study.
- Only cite the sources actually listed in the evidence below.

--- EVIDENCE ---
{evidence_block}

Return the report as title, introduction, one finding per sub-question,
and a conclusion. If evidence is insufficient anywhere, say so plainly."""
    else:
        issues_block = "\n".join(f"- {issue}" for issue in critique["issues"])
        prompt = f"""You are revising a research report based on reviewer feedback.

--- ORIGINAL EVIDENCE ---
{evidence_block}

--- PREVIOUS DRAFT ---
{state["draft_report"]}

--- REVIEWER ISSUES TO FIX ---
{issues_block}

CRITICAL RULES — DO NOT VIOLATE THESE:
- Do NOT invent citations, statistics, or studies not present in the
    original evidence, even if the reviewer's issues ask for "more rigor".
- Only cite sources present in the original evidence block above.

Rewrite the report addressing EVERY issue above. Do not introduce new
unsupported claims. Keep what already worked; fix only what's flagged."""

    structured_llm = writer_llm.with_structured_output(DraftReport)
    result = invoke_with_retry(structured_llm, prompt, "WRITER", attempts=5)
    if result is None:
        new_state = dict(state)
        new_state["writer_failed"] = True
        new_state.pop("draft_report", None)
        return new_state

    new_state = dict(state)
    new_state["draft_report"] = render_markdown(result)
    new_state["writer_failed"] = False
    if critique is not None:
        new_state["revision_count"] = state.get("revision_count", 0) + 1
    new_state["critique"] = None
    return new_state
