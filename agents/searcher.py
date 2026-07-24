# agents/searcher.py
# Search Agent — searches the web via Tavily.

import asyncio
import time
from typing import Dict, List

from langsmith import traceable

from schemas.state import AutoResearchState
from clients.search_client import tavily_client


@traceable(name="Search Agent")
def search_node(state: AutoResearchState) -> AutoResearchState:
    sub_questions = state["sub_questions"]

    async def search_question(question: str, index: int) -> List[Dict[str, str]]:
        if index > 0:
            await asyncio.sleep(index * 0.5)
        start_time = time.time()
        try:
            print(f"[SEARCHER] starting search for: {question[:80]}")
            response = await asyncio.to_thread(
                tavily_client.search,
                query=question,
                max_results=3,
                search_depth="basic",
                timeout=20
            )
            return [
                {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")}
                for item in response.get("results", [])
            ]
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[SEARCHER][ERROR] '{question[:50]}...' failed after {elapsed:.1f}s: {e}")
            raise e

    async def run_searches() -> List[List[Dict[str, str]] | Exception]:
        tasks = [search_question(q, i) for i, q in enumerate(sub_questions)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    results = asyncio.run(run_searches())

    search_results: Dict[str, List[Dict[str, str]]] = {}
    for question, res in zip(sub_questions, results):
        if isinstance(res, Exception):
            search_results[question] = []
        else:
            search_results[question] = res

    new_state = dict(state)
    new_state["search_results"] = search_results
    return new_state
