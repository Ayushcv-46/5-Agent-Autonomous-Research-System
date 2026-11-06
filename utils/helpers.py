# utils/helpers.py
# Pure utility functions — no Streamlit, no LLM imports.

import time
from typing import List, Dict, Any

from schemas.pydantic_models import DraftReport, Finding


def invoke_with_retry(invoker, prompt, node_name, attempts=3, method_name="invoke"):
    method = getattr(invoker, method_name)
    for attempt in range(1, attempts + 1):
        try:
            return method(prompt)
        except Exception as e:
            print(f"[{node_name}][attempt {attempt}/{attempts}] failed: {e}")
            if attempt < attempts:
                time.sleep(2 ** attempt)
    return None

def format_evidence_for_prompt(extracted_points):
    blocks = []
    for i, point in enumerate(extracted_points, 1):
        question = point.get("question", "")
        answer = point.get("answer", "")
        sources = point.get("sources", [])
        sources_str = ", ".join(sources) if sources else "No source available"
        contexts = point.get("contexts", [])
        context_block = "\n".join(
            f"  - {context[:1600]}" for context in contexts[:3] if context
        )
        blocks.append(
            f"[{i}] Sub-Question: {question}\nOriginal Research Answer: {answer}\n"
            f"Sources: {sources_str}\nRetrieved evidence excerpts:\n{context_block or '  - No excerpts available'}"
        )
    return "\n\n".join(blocks)


def align_report_citations(report: DraftReport, extracted_points):
    """Align each finding to its planned question and allowlisted source URLs.

    Returns a normalized report and warnings about omitted or invented URLs.
    This validates citation provenance at the finding level; it does not
    claim sentence-level factual verification.
    """
    normalized_findings = []
    warnings = []
    supplied_findings = report.findings

    for index, point in enumerate(extracted_points):
        question = point.get("question", f"Research question {index + 1}")
        allowed_sources = list(dict.fromkeys(
            url for url in point.get("sources", []) if isinstance(url, str) and url
        ))

        if index < len(supplied_findings):
            finding = supplied_findings[index]
            cited_sources = list(dict.fromkeys(
                url for url in finding.sources if url in allowed_sources
            ))
            if len(cited_sources) != len(set(finding.sources)):
                warnings.append(f"Finding {index + 1}: removed citation(s) absent from its retrieved evidence.")
            if not cited_sources and allowed_sources:
                cited_sources = allowed_sources[:3]
                warnings.append(f"Finding {index + 1}: attached retrieved source(s) omitted by the Writer.")
            normalized_findings.append(Finding(
                sub_question=question,
                content=finding.content or "The Writer returned no finding for this question.",
                sources=cited_sources,
            ))
        else:
            warnings.append(f"Finding {index + 1}: missing finding was filled with an explicit evidence gap.")
            normalized_findings.append(Finding(
                sub_question=question,
                content="No finding was produced for this sub-question; treat this as an evidence gap.",
                sources=allowed_sources[:3],
            ))

    if len(supplied_findings) > len(extracted_points):
        warnings.append("Removed extra finding(s) that did not match a planned sub-question.")

    return DraftReport(
        title=report.title,
        introduction=report.introduction,
        findings=normalized_findings,
        conclusion=report.conclusion,
    ), warnings

def render_markdown(report: DraftReport) -> str:
    lines = [f"# {report.title}", "", "## Introduction", report.introduction, "", "## Findings"]
    for f in report.findings:
        lines += [f"### {f.sub_question}", f.content]
        if f.sources:
            lines.append(f"*Sources: {', '.join(f.sources)}*")
        lines.append("")
    lines += ["## Conclusion", report.conclusion]
    return "\n".join(lines)

def extract_unique_sources(extracted_points: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    unique_sources = []
    for point in extracted_points:
        for url in point.get("sources", []):
            if url and url not in seen:
                seen.add(url)
                unique_sources.append(url)
    return unique_sources

def safe_filename_from(topic: str) -> str:
    cleaned = "".join(c for c in topic if c.isalnum() or c in " _-")
    return cleaned.strip()[:50].strip() or "report"
