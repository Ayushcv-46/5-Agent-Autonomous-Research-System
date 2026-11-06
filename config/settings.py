# config/settings.py
# Environment setup, constants, and stdout reconfiguration.

import os
import sys
from dotenv import load_dotenv

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

MAX_REVISIONS = 2
MAX_CONTEXT_CHARS = 4000  # cap per retrieved chunk fed to RAGAS
SEARCH_CONCURRENCY = int(os.getenv("SEARCH_CONCURRENCY", "3"))
FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", "6"))
FETCH_TIMEOUT_SECONDS = float(os.getenv("FETCH_TIMEOUT_SECONDS", "12"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Support the current LangSmith names and the older LANGCHAIN_* aliases.
tracing_enabled = os.getenv(
    "LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "true")
)
project_name = os.getenv(
    "LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "AutoResearch")
)
langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING"] = tracing_enabled
os.environ["LANGSMITH_PROJECT"] = project_name
LANGSMITH_TRACING_ENABLED = tracing_enabled.strip().lower() in {"1", "true", "yes", "on"}
LANGSMITH_PROJECT_NAME = project_name
LANGSMITH_CONFIGURED = LANGSMITH_TRACING_ENABLED and bool(langsmith_key)
# Older LangChain integrations still read these aliases.
os.environ["LANGCHAIN_TRACING_V2"] = tracing_enabled
os.environ["LANGCHAIN_PROJECT"] = project_name
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
