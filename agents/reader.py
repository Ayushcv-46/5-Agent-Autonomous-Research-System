# agents/reader.py
# Reader Agent — bounded async fetching, explicit chunking, hybrid RRF retrieval.

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Dict, List
from urllib.parse import urlsplit

import chromadb
import httpx
from bs4 import BeautifulSoup
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from langsmith import traceable

from clients.llm_clients import llamaindex_llm
from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FETCH_CONCURRENCY,
    FETCH_TIMEOUT_SECONDS,
    MAX_CONTEXT_CHARS,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from evaluation.ragas_eval import run_ragas_evaluation
from schemas.state import AutoResearchState
from utils.helpers import invoke_with_retry


class HybridRRF(BaseRetriever):
    """Fuse dense Chroma and BM25 rankings with reciprocal rank fusion."""

    def __init__(self, vector_retriever, lexical_retriever, top_k: int, rrf_k: int):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.lexical_retriever = lexical_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        fused_scores = defaultdict(float)
        nodes_by_id = {}
        for retriever in (self.vector_retriever, self.lexical_retriever):
            for rank, item in enumerate(retriever.retrieve(query_bundle), start=1):
                node_id = item.node.node_id
                nodes_by_id[node_id] = item.node
                fused_scores[node_id] += 1.0 / (self.rrf_k + rank)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[: self.top_k]
        return [
            NodeWithScore(node=nodes_by_id[node_id], score=fused_scores[node_id])
            for node_id in ranked_ids
        ]


async def _fetch_documents(urls, snippets):
    semaphore = asyncio.Semaphore(max(1, FETCH_CONCURRENCY))
    timeout = httpx.Timeout(FETCH_TIMEOUT_SECONDS)
    limits = httpx.Limits(
        max_connections=max(1, FETCH_CONCURRENCY),
        max_keepalive_connections=max(1, FETCH_CONCURRENCY),
    )
    headers = {"User-Agent": "AutoResearch/1.0 (research reader)"}
    fetch_timings = {}

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, follow_redirects=True, headers=headers
    ) as client:
        async def fetch_one(url):
            started = time.perf_counter()
            try:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise ValueError("unsupported URL scheme or missing host")
                async with semaphore:
                    response = await client.get(url)
                    response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                if not text:
                    raise ValueError("page contained no extractable text")
                document = Document(text=text, metadata={"url": url})
                return url, document, None, time.perf_counter() - started
            except Exception as exc:
                snippet = snippets.get(url, "").strip()
                fallback = Document(text=snippet, metadata={"url": url}) if snippet else None
                return url, fallback, f"{type(exc).__name__}: {exc}", time.perf_counter() - started

        results = await asyncio.gather(*(fetch_one(url) for url in urls))

    documents = []
    fetch_errors = {}
    for url, document, error, elapsed in results:
        fetch_timings[url] = round(elapsed, 3)
        if error:
            fetch_errors[url] = error
            if document:
                print(f"[READER][FALLBACK] used search snippet for {url}")
            else:
                print(f"[READER][FETCH_FAILED] {url}: {error}")
        if document:
            documents.append(document)
    return documents, fetch_timings, fetch_errors


@traceable(name="Reader Agent")
def reader_node(state: AutoResearchState) -> AutoResearchState:
    total_started = time.perf_counter()
    sub_questions = state.get("sub_questions", [])
    search_results = state.get("search_results", {})

    url_snippets: Dict[str, str] = {}
    all_urls = []
    seen_urls = set()
    for results_list in search_results.values():
        for item in results_list:
            url = item.get("url", "").strip()
            if not url:
                continue
            url_snippets[url] = item.get("snippet", "")
            if url not in seen_urls:
                seen_urls.add(url)
                all_urls.append(url)

    stage_timings = dict(state.get("stage_timings", {}))
    try:
        documents, fetch_timings, fetch_errors = asyncio.run(
            _fetch_documents(all_urls, url_snippets)
        )
    except RuntimeError as exc:
        documents, fetch_timings, fetch_errors = [], {}, {"fetcher": str(exc)}
    stage_timings["page_fetch_seconds"] = round(sum(fetch_timings.values()), 3)
    stage_timings["page_fetch_wall_seconds"] = round(max(fetch_timings.values(), default=0), 3)

    new_state = dict(state)
    new_state["stage_timings"] = stage_timings
    new_state["fetch_timings"] = fetch_timings
    new_state["fetch_errors"] = fetch_errors

    if not documents:
        new_state.update({
            "extracted_points": [],
            "reader_failed": True,
            "reader_error": "No pages or search snippets could be read.",
            "ragas_scores": None,
        })
        return new_state

    index_started = time.perf_counter()
    chroma_client = chromadb.EphemeralClient()
    collection_name = f"session_{uuid.uuid4().hex[:8]}"
    chroma_collection = chroma_client.create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
    stage_timings["chunk_and_index_seconds"] = round(time.perf_counter() - index_started, 3)

    vector_retriever = index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
    lexical_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=RETRIEVAL_TOP_K,
    )
    hybrid_retriever = HybridRRF(
        vector_retriever=vector_retriever,
        lexical_retriever=lexical_retriever,
        top_k=RETRIEVAL_TOP_K,
        rrf_k=RRF_K,
    )
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        llm=llamaindex_llm,
    )

    extracted_points = []
    retrieval_started = time.perf_counter()
    for question in sub_questions:
        response = invoke_with_retry(query_engine, question, "READER", method_name="query")
        if response is None:
            extracted_points.append({
                "question": question,
                "answer": "Error: could not retrieve answer.",
                "sources": [],
                "contexts": [],
            })
            continue

        sources, contexts = [], []
        for node in response.source_nodes:
            url = node.metadata.get("url", "")
            if url and url not in sources:
                sources.append(url)
            if node.text:
                contexts.append(node.text[:MAX_CONTEXT_CHARS])
        extracted_points.append({
            "question": question,
            "answer": str(response).strip(),
            "sources": sources,
            "contexts": contexts,
        })
    stage_timings["question_retrieval_seconds"] = round(time.perf_counter() - retrieval_started, 3)

    failed_count = sum(
        point["answer"] == "Error: could not retrieve answer." or not point["sources"]
        for point in extracted_points
    )
    reader_failed = bool(extracted_points) and failed_count == len(extracted_points)
    new_state["extracted_points"] = extracted_points
    new_state["reader_failed"] = reader_failed

    ragas_started = time.perf_counter()
    if state.get("run_ragas", True) and not reader_failed:
        new_state["ragas_scores"] = run_ragas_evaluation(extracted_points)
    else:
        new_state["ragas_scores"] = None
    stage_timings["ragas_seconds"] = round(time.perf_counter() - ragas_started, 3)
    stage_timings["reader_work_seconds"] = round(time.perf_counter() - total_started, 3)
    new_state["stage_timings"] = stage_timings
    return new_state
