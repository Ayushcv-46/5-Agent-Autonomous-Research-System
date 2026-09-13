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
# HTTP client with clean timeout
# ==========================================

ipv4_client = httpx.Client(
    timeout=30.0,
)

# ==========================================
# LLM CLIENTS
# ==========================================

MODEL_NAME = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")

llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=1, top_p=1, max_tokens=4096,
    http_client=ipv4_client, timeout=60.0, max_retries=2,
)

writer_llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=1, top_p=1, max_tokens=12000,
    http_client=ipv4_client, timeout=120.0, max_retries=2,
)

critic_llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=1, top_p=1, max_tokens=4096,
    http_client=ipv4_client, timeout=60.0, max_retries=2,
)

# ---------- RAGAS judge (long timeout!) ----------
ragas_judge_llm = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0, top_p=1, max_tokens=16000,   # temperature=0: judges must be deterministic
    timeout=120.0, max_retries=2,
)

llamaindex_llm = OpenAILike(
    model=MODEL_NAME,
    api_key=API_KEY,
    api_base=BASE_URL,
    temperature=1, top_p=1,
    http_client=ipv4_client, timeout=60.0, max_retries=2,
    max_tokens=4096, context_window=128000, is_chat_model=True,
)
Settings.llm = llamaindex_llm

# ==========================================
# EMBEDDING MODEL (offline → online fallback)
# ==========================================

try:
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"local_files_only": True}
    )
except Exception as e:
    print(f"[WARNING] Local HF embedding load failed ({e}); retrying standard load.")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

