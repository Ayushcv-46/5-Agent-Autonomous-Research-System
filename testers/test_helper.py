# tests/test_helpers.py
#
# Unit tests for the pure functions in utils/helpers.py. I'm assuming the
# same 5 function names your agent's table listed (invoke_with_retry,
# format_evidence_for_prompt, render_markdown, extract_unique_sources,
# safe_filename_from) since those are what critic.py/writer.py/reader.py
# import from utils.helpers. If any name doesn't match what's actually in
# your file, just fix the import line below — the test bodies don't care.
#
# Run with:  pytest tests/test_helpers.py -v

from utils.helpers import (
    extract_unique_sources,
    safe_filename_from,
    render_markdown,
    format_evidence_for_prompt,
)
from schemas.pydantic_models import DraftReport, Finding


def test_extract_unique_sources_dedupes_across_points():
    points = [
        {"sources": ["https://a.com", "https://b.com"]},
        {"sources": ["https://b.com", "https://c.com"]},
    ]
    result = extract_unique_sources(points)
    assert result == ["https://a.com", "https://b.com", "https://c.com"]


def test_extract_unique_sources_handles_empty_and_missing():
    assert extract_unique_sources([]) == []
    assert extract_unique_sources([{"sources": []}, {}]) == []


def test_safe_filename_strips_special_characters():
    assert safe_filename_from("AI in Healthcare: 2026?!") == "AI in Healthcare 2026"


def test_safe_filename_empty_topic_has_fallback():
    assert safe_filename_from("") == "report"
    assert safe_filename_from("!!!") == "report"


def test_safe_filename_truncates_to_50_chars():
    long_topic = "a" * 100
    assert len(safe_filename_from(long_topic)) <= 50


def test_render_markdown_includes_all_sections():
    report = DraftReport(
        title="Test Report",
        introduction="Intro text.",
        findings=[
            Finding(sub_question="What is X?", content="X is Y.", sources=["https://x.com"]),
        ],
        conclusion="Conclusion text.",
    )
    md = render_markdown(report)
    assert "# Test Report" in md
    assert "## Introduction" in md
    assert "### What is X?" in md
    assert "Sources: https://x.com" in md
    assert "## Conclusion" in md


def test_format_evidence_handles_missing_sources():
    points = [{"question": "Q1", "answer": "A1", "sources": []}]
    block = format_evidence_for_prompt(points)
    assert "No source available" in block
    assert "Q1" in block and "A1" in block