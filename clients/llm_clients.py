# clients/llm_clients.py
# All LLM clients: ChatOpenAI (langchain), OpenAILike (llama_index),
# HuggingFace embeddings, and Settings configuration.

import os
import httpx

from langchain_openai import ChatOpenAI
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ensure config is loaded first
from config import settings as _settings  # noqa: F401

# ==========================================
# IPv4-only HTTP client
# ==========================================

ipv4_client = httpx.Client(
    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    timeout=15.0,
)

# ==========================================
# LLM CLIENTS
# ==========================================

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    openai_api_base="https://integrate.api.nvidia.com/v1",
    temperature=1, top_p=1, max_tokens=4096,
    http_client=ipv4_client, timeout=15.0, max_retries=0,
)

writer_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    openai_api_base="https://integrate.api.nvidia.com/v1",
    temperature=1, top_p=1, max_tokens=12000,
    http_client=ipv4_client, timeout=60.0, max_retries=0,
)

critic_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    openai_api_base="https://integrate.api.nvidia.com/v1",
    temperature=1, top_p=1, max_tokens=4096,
    http_client=ipv4_client, timeout=15.0, max_retries=0,
)

# ---------- RAGAS judge (long timeout!) ----------
# RAGAS fires dozens of judge calls per run; a 15s timeout would fail
# constantly. Separate client, 60s timeout, 1 retry.
ragas_judge_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=os.getenv("NVIDIA_API_KEY"),
    openai_api_base="https://integrate.api.nvidia.com/v1",
    temperature=0, top_p=1, max_tokens=8000,   # temperature=0: judges must be deterministic
    timeout=60.0, max_retries=1,
)

llamaindex_llm = OpenAILike(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("NVIDIA_API_KEY"),
    api_base="https://integrate.api.nvidia.com/v1",
    temperature=1, top_p=1,
    http_client=ipv4_client, timeout=15.0, max_retries=0,
    max_tokens=4096, context_window=128000, is_chat_model=True,
)
Settings.llm = llamaindex_llm

# ==========================================
# EMBEDDING MODEL (offline → online fallback)
# ==========================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception:
    print("[WARNING] HF embedding model not cached locally; retrying online.")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
