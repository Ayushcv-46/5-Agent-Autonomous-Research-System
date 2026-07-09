# agents/searcher.py
# Search Agent — searches the web via Tavily.

import time
from typing import Dict, List

from langsmith import traceable

from schemas.state import AutoResearchState
from clients.search_client import tavily_client


@traceable(name="Search Agent")
def search_node(state: AutoResearchState) -> AutoResearchState:
    sub_questions = state["sub_questions"]
    search_results: Dict[str, List[Dict[str, str]]] = {}
    for question in sub_questions:
        start_time = time.time()
        try:
            print(f"[SEARCHER] starting search for: {question[:80]}")
            response = tavily_client.search(query=question, max_results=3, search_depth="basic", timeout=20)
            results = [
                {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
                for item in response.get("results", [])
            ]
            search_results[question] = results
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[SEARCHER][ERROR] '{question[:50]}...' failed after {elapsed:.1f}s: {e}")
            search_results[question] = []
        time.sleep(0.5)
    new_state = dict(state)
    new_state["search_results"] = search_results
    return new_state
