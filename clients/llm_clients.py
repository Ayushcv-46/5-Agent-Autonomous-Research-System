# clients/llm_clients.py
# Gemini clients for LangChain agents and LlamaIndex retrieval.

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ensure .env is loaded before reading provider credentials.
from config import settings as _settings  # noqa: F401

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Both SDK integrations understand GOOGLE_API_KEY. Keep GEMINI_API_KEY as an
# accepted alias so existing local setups do not need to rename their secret.
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY


def _chat_model(max_tokens: int, timeout: float):
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        vertexai=False,
        temperature=1,
        max_tokens=max_tokens,
        request_timeout=timeout,
        retries=2,
    )


llm = _chat_model(max_tokens=4096, timeout=60.0)
writer_llm = _chat_model(max_tokens=12000, timeout=120.0)
critic_llm = _chat_model(max_tokens=4096, timeout=60.0)
ragas_judge_llm = _chat_model(max_tokens=16000, timeout=120.0)

# LlamaIndex uses the same Gemini key for query synthesis in the Reader.
llamaindex_llm = GoogleGenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    temperature=1,
    max_tokens=4096,
)
Settings.llm = llamaindex_llm

# ==========================================
# EMBEDDING MODEL (offline -> online fallback)
# ==========================================

try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"local_files_only": True},
    )
except Exception as exc:
    print(f"[WARNING] Local HF embedding load failed ({exc}); retrying standard load.")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
