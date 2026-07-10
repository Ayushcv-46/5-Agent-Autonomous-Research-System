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

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "AutoResearch")
