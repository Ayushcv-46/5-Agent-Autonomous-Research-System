"""Run the fixed, human-referenced retrieval benchmark.

This performs real Tavily and Gemini calls for each benchmark question.
The RAGAS reference metrics use the curated expected answers, not the
system-generated answer as its own reference.
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from datasets import Dataset
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.reader import reader_node
from agents.searcher import search_node
from schemas.state import AutoResearchState


def _domain(url: str) -> str:
    hostname = (urlsplit(url).hostname or "").lower()
    return hostname[4:] if hostname.startswith("www.") else hostname


def _source_coverage(expected_urls, retrieved_urls):
    expected_domains = {_domain(url) for url in expected_urls if _domain(url)}
    retrieved_domains = {_domain(url) for url in retrieved_urls if _domain(url)}
    covered = sorted(expected_domains & retrieved_domains)
    return {
        "expected_domains": sorted(expected_domains),
        "retrieved_domains": sorted(retrieved_domains),
        "covered_domains": covered,
        "coverage": round(len(covered) / len(expected_domains), 3) if expected_domains else None,
    }


def _finite_score(value):
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _build_reader_graph():
    graph = StateGraph(AutoResearchState)
    graph.add_node("searcher", search_node)
    graph.add_node("reader", reader_node)
    graph.set_entry_point("searcher")
    graph.add_edge("searcher", "reader")
    graph.add_edge("reader", END)
    return graph.compile()


def _initial_state(question):
    return {
        "topic": question,
        "sub_questions": [question],
        "search_results": {},
        "extracted_points": [],
        "reader_failed": False,
        "writer_failed": False,
        "critic_failed": False,
        "best_report": "",
        "best_score": -1,
        "revision_count": 0,
        "critique": None,
        "run_ragas": False,
        "ragas_scores": None,
        "stage_timings": {},
        "planner_fallback_used": False,
        "citation_warnings": [],
    }


def _run_reference_metrics(rows):
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.dataset_schema import EvaluationResult
    from evaluation.ragas_eval import get_ragas_wrappers

    dataset = Dataset.from_dict({
        "user_input": [row["question"] for row in rows],
        "response": [row["answer"] for row in rows],
        "retrieved_contexts": [row["contexts"] for row in rows],
        "reference": [row["expected_answer"] for row in rows],
    })
    judge, embeddings = get_ragas_wrappers()
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]
    for metric in metrics:
        if hasattr(metric, "reproducibility"):
            metric.reproducibility = 1

    result = ragas_evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )
    if not isinstance(result, EvaluationResult):
        raise RuntimeError("RAGAS did not return completed evaluation scores.")
    return result.scores


def main():
    parser = argparse.ArgumentParser(description="Run the human-referenced AutoResearch benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N benchmark questions")
    parser.add_argument("--skip-ragas", action="store_true", help="Only report source-domain coverage")
    args = parser.parse_args()

    with (ROOT / "evaluation" / "golden_set.json").open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    if not cases:
        raise SystemExit("No benchmark questions selected.")

    app = _build_reader_graph()
    output_rows = []
    metric_rows = []
    for index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        print(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")
        result = {
            "id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_source_urls": case["source_urls"],
            "status": "failed",
        }
        try:
            state = app.invoke(_initial_state(case["question"]))
            point = (state.get("extracted_points") or [{}])[0]
            sources = point.get("sources", [])
            contexts = point.get("contexts", [])
            answer = point.get("answer", "")
            result.update({
                "status": "completed",
                "answer": answer,
                "retrieved_sources": sources,
                "source_domain_coverage": _source_coverage(case["source_urls"], sources),
                "retrieved_context_count": len(contexts),
                "reader_failed": state.get("reader_failed", False),
                "stage_timings": state.get("stage_timings", {}),
            })
            if answer and answer != "Error: could not retrieve answer." and contexts:
                metric_rows.append({
                    "id": case["id"],
                    "question": case["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "expected_answer": case["expected_answer"],
                })
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    [ERROR] {result['error']}")
        result["duration_seconds"] = round(time.perf_counter() - started, 2)
        output_rows.append(result)

    if not args.skip_ragas and metric_rows:
        try:
            scores = _run_reference_metrics(metric_rows)
            metric_by_id = {row["id"]: row for row in output_rows}
            for input_row, score_row in zip(metric_rows, scores):
                metric_by_id[input_row["id"]]["reference_metrics"] = {
                    name: _finite_score(value)
                    for name, value in score_row.items()
                }
            metric_names = scores[0].keys() if scores else []
            averages = {}
            for name in metric_names:
                values = [
                    _finite_score(row[name])
                    for row in scores
                    if row.get(name) is not None
                    and _finite_score(row[name]) is not None
                ]
                averages[name] = round(sum(values) / len(values), 3) if values else None
        except Exception as exc:
            averages = {}
            print(f"[RAGAS][ERROR] {type(exc).__name__}: {exc}")
    else:
        averages = {}

    report = {
        "benchmark": "AutoResearch human-referenced retrieval set",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_basis": "curated human-written expected answers",
        "metrics_note": (
            "Context precision/recall and answer correctness use the curated reference answers. "
            "Faithfulness and answer relevancy remain LLM-judge estimates."
        ),
        "averages": averages,
        "completed_cases": sum(row["status"] == "completed" for row in output_rows),
        "case_count": len(output_rows),
        "cases": output_rows,
    }
    results_dir = ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"golden_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nReference-based averages:")
    for name, score in averages.items():
        print(f"  {name}: {score if score is not None else 'N/A'}")
    coverage_values = [
        row["source_domain_coverage"]["coverage"]
        for row in output_rows
        if row.get("source_domain_coverage", {}).get("coverage") is not None
    ]
    coverage = sum(coverage_values) / len(coverage_values) if coverage_values else None
    print(f"Expected source-domain coverage: {coverage:.1%}" if coverage is not None else "Expected source-domain coverage: N/A")
    print(f"Detailed report: {output_path}")


if __name__ == "__main__":
    main()
