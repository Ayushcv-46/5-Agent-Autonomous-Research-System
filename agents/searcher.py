# agents/searcher.py
# Search Agent — runs independent Tavily queries with bounded concurrency.

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from langsmith import traceable

from schemas.state import AutoResearchState
from config.settings import SEARCH_CONCURRENCY
from clients.search_client import create_tavily_client


def _search_one(question: str):
    started = time.perf_counter()
    try:
        response = create_tavily_client().search(
            query=question,
            max_results=3,
            search_depth="basic",
            timeout=20,
        )
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in response.get("results", [])
        ]
        return question, results, time.perf_counter() - started, None
    except Exception as exc:
        return question, [], time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


@traceable(name="Search Agent")
def search_node(state: AutoResearchState) -> AutoResearchState:
    questions = state.get("sub_questions", [])
    results_by_question: Dict[str, List[Dict[str, str]]] = {}
    search_timings: Dict[str, float] = {}
    search_errors: Dict[str, str] = {}

    if questions:
        worker_count = max(1, min(SEARCH_CONCURRENCY, len(questions)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_search_one, question) for question in questions]
            for future in as_completed(futures):
                question, results, elapsed, error = future.result()
                results_by_question[question] = results
                search_timings[question] = round(elapsed, 3)
                if error:
                    search_errors[question] = error
                    print(f"[SEARCHER][ERROR] {question[:70]!r}: {error}")
                else:
                    print(f"[SEARCHER] {question[:70]!r}: {len(results)} results in {elapsed:.2f}s")

    # Restore planner order for readable traces and stable downstream processing.
    new_state = dict(state)
    new_state["search_results"] = {q: results_by_question.get(q, []) for q in questions}
    new_state["search_timings"] = search_timings
    new_state["search_errors"] = search_errors
    return new_state
