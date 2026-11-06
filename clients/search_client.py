# clients/search_client.py
# Factory for independent Tavily clients used by concurrent search workers.

import os

from tavily import TavilyClient

from config import settings as _settings  # noqa: F401


def create_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
