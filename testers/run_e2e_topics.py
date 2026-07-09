# tests/run_e2e_topics.py
#
# Deliverable: "Run 20 topics: technical, business, current events,
# niche. Fix every crash and bad output."
#
# NOT a pytest file — real NVIDIA + Tavily API calls, RAGAS alone is
# 1-3 min/topic. Run directly:
#
#   python tests/run_e2e_topics.py
#   python tests/run_e2e_topics.py --limit 5          # quick smoke test
#   python tests/run_e2e_topics.py --no-ragas          # skip RAGAS, faster
#
# Every topic is wrapped in its own try/except, and results are written to
# tests/e2e_report_<timestamp>.json after EVERY topic — a crash on #14
# still leaves you #1-13's results on disk.

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

# Same trick as conftest.py — make the root importable regardless of cwd,
# since this file is run directly (`python tests/run_e2e_topics.py`) and
# won't get conftest.py's sys.path injection for free.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.builder import build_graph
from utils.helpers import extract_unique_sources

DEFAULT_TOPICS = [
    # technical
    "How do vector databases handle approximate nearest neighbor search?",
    "What are the tradeoffs between REST and GraphQL APIs?",
    "How does retrieval-augmented generation reduce LLM hallucination?",
    "What is the CAP theorem and why does it matter for distributed databases?",
    "How do transformer attention mechanisms scale with sequence length?",
    # business
    "What is driving consolidation in the global semiconductor industry?",
    "How are subscription businesses measuring customer lifetime value?",
    "What factors influence startup valuation in early funding rounds?",
    "How is generative AI changing enterprise software pricing models?",
    "What are the risks of over-reliance on a single cloud vendor?",
    # current events / fast-moving (good stress test for Tavily recency)
    "What are the latest developments in India's semiconductor manufacturing push?",
    "What is the current state of AI regulation in the European Union?",
    "What progress has been made on commercial nuclear fusion in 2026?",
    "How are central banks responding to inflation trends in 2026?",
    "What is the current status of India's space program and ISRO missions?",
    # niche / edge cases
    "What is the history of the Bengaluru Metro Rail project?",
    "How does the Vedantic concept of neti neti relate to modern epistemology?",
    "What are the engineering challenges of vertical farming in urban India?",
    "asdfghjkl",  # garbage input — must not crash the pipeline
    "",  # empty input — must not crash the pipeline
]


def run_single_topic(app, topic: str, run_ragas: bool) -> dict:
    initial_state = {
        "topic": topic,
        "sub_questions": [],
        "search_results": {},
        "extracted_points": [],
        "reader_failed": False,
        "writer_failed": False,
        "critic_failed": False,
        "best_report": "",
        "best_score": -1,
        "revision_count": 0,
        "critique": None,
        "run_ragas": run_ragas,
        "ragas_scores": None,
    }

    result = {
        "topic": topic,
        "status": "unknown",
        "duration_sec": None,
        "score": None,
        "verdict": None,
        "revision_count": None,
        "reader_failed": None,
        "writer_failed": None,
        "critic_failed": None,
        "ragas_scores": None,
        "num_sources": None,
        "final_report_preview": None,
        "error": None,
    }

    start = time.time()
    try:
        running_state = dict(initial_state)
        for step in app.stream(initial_state):
            for _, update in step.items():
                running_state.update(update)

        final_report = running_state.get("final_report", "")
        critique = running_state.get("critique") or {}

        result.update({
            "status": "completed",
            "score": critique.get("quality_score"),
            "verdict": critique.get("verdict"),
            "revision_count": running_state.get("revision_count", 0),
            "reader_failed": running_state.get("reader_failed", False),
            "writer_failed": running_state.get("writer_failed", False),
            "critic_failed": running_state.get("critic_failed", False),
            "ragas_scores": running_state.get("ragas_scores"),
            "num_sources": len(extract_unique_sources(running_state.get("extracted_points", []))),
            "final_report_preview": final_report[:300],
        })
    except Exception as e:
        result["status"] = "crashed"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"    [CRASH] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        result["duration_sec"] = round(time.time() - start, 1)

    return result


def main():
    parser = argparse.ArgumentParser(description="AutoResearch end-to-end topic harness")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N topics")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS evaluation for a faster run")
    args = parser.parse_args()

    topics = DEFAULT_TOPICS[: args.limit] if args.limit else DEFAULT_TOPICS
    run_ragas = not args.no_ragas

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(os.path.dirname(__file__), f"e2e_report_{timestamp}.json")

    print(f"Running {len(topics)} topics (RAGAS {'ON' if run_ragas else 'OFF'})")
    print(f"Report will be written incrementally to: {report_path}\n")

    app = build_graph()
    results = []

    for i, topic in enumerate(topics, 1):
        label = topic if topic else "(empty string)"
        print(f"[{i}/{len(topics)}] {label[:70]}")
        result = run_single_topic(app, topic, run_ragas)
        results.append(result)

        status_str = result["status"].upper()
        extra = f"score={result['score']} verdict={result['verdict']}" if result["status"] == "completed" else result["error"]
        print(f"    -> {status_str} in {result['duration_sec']}s  {extra}")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    completed = [r for r in results if r["status"] == "completed"]
    crashed = [r for r in results if r["status"] == "crashed"]
    scored = [r for r in completed if r["score"] is not None]

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(completed)}/{len(results)} completed, {len(crashed)} crashed")
    if scored:
        avg_score = sum(r["score"] for r in scored) / len(scored)
        print(f"Average quality score: {avg_score:.1f}/10 (n={len(scored)})")
    if crashed:
        print("\nCrashed topics:")
        for r in crashed:
            print(f"  - {r['topic'][:60]!r}: {r['error']}")
    print(f"\nFull report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()