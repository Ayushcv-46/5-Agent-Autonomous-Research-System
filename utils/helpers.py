# utils/helpers.py
# Pure utility functions — no Streamlit, no LLM imports.

import time
from typing import List, Dict, Any

from schemas.pydantic_models import DraftReport


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
        blocks.append(
            f"[{i}] Sub-Question: {question}\nOriginal Research Answer: {answer}\nSources: {sources_str}"
        )
    return "\n\n".join(blocks)

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
