# agents/reader.py
# Reader Agent — extracts grounded answers via LlamaIndex + ChromaDB.

import uuid
import asyncio

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from langsmith import traceable

from schemas.state import AutoResearchState
from config.settings import MAX_CONTEXT_CHARS
from utils.helpers import invoke_with_retry
from evaluation.ragas_eval import run_ragas_evaluation


@traceable(name="Reader Agent")
def reader_node(state: AutoResearchState) -> AutoResearchState:
    sub_questions = state["sub_questions"]
    search_results = state["search_results"]

    all_urls, url_set = [], set()
    url_snippets = {}
    for results_list in search_results.values():
        for item in results_list:
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            if url:
                url_snippets[url] = snippet
                if url not in url_set:
                    url_set.add(url)
                    all_urls.append(url)

    web_reader = SimpleWebPageReader()
    documents = []

    async def fetch_url(url: str):
        try:
            print(f"[READER] fetching: {url}")
            fetched_docs = await asyncio.wait_for(
                asyncio.to_thread(web_reader.load_data, urls=[url]),
                timeout=15
            )
            return url, fetched_docs or []
        except asyncio.TimeoutError:
            print(f"[READER][TIMEOUT] {url} took too long, using search snippet fallback")
        except Exception as e:
            print(f"[READER][FAILED] {url} — {e}")
        return url, []

    async def run_readers():
        tasks = [fetch_url(url) for url in all_urls]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_readers())

    for url, fetched_docs in results:
        if fetched_docs:
            documents.extend(fetched_docs)
            continue

        # Fallback to Tavily search snippet if web page fetching yielded no docs
        if url_snippets.get(url):
            print(f"[READER][FALLBACK] using search snippet for: {url}")
            documents.append(Document(text=url_snippets[url], metadata={"url": url}))

    if not documents:
        new_state = dict(state)
        new_state["extracted_points"] = []
        new_state["reader_failed"] = True
        new_state["ragas_scores"] = None
        return new_state

    chroma_client = chromadb.EphemeralClient()
    collection_name = f"session_{uuid.uuid4().hex[:8]}"
    chroma_collection = chroma_client.create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=False)
    query_engine = index.as_query_engine(similarity_top_k=3)

    extracted_points = []
    for question in sub_questions:
        response = invoke_with_retry(query_engine, question, "READER", method_name="query")
        if response is None:
            extracted_points.append({
                "question": question,
                "answer": "Error: could not retrieve answer.",
                "sources": [],
                "contexts": [],          # NEW DAY 23
            })
            continue
        answer = str(response).strip()
        sources, contexts = [], []       # NEW DAY 23: contexts list
        for node in response.source_nodes:
            url = node.metadata.get("url", "")
            if url and url not in sources:
                sources.append(url)
            # NEW DAY 23: RAGAS needs the actual chunk text, capped for
            # prompt-size safety.
            if node.text:
                contexts.append(node.text[:MAX_CONTEXT_CHARS])
        extracted_points.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "contexts": contexts,        # NEW DAY 23
        })

    failed_count = sum(p["answer"] == "Error: could not retrieve answer." for p in extracted_points)
    reader_failed = bool(extracted_points) and failed_count == len(extracted_points)

    new_state = dict(state)
    new_state["extracted_points"] = extracted_points
    new_state["reader_failed"] = reader_failed

    # NEW DAY 23: evaluate retrieval quality — optional, never fatal.
    if state.get("run_ragas", True) and not reader_failed:
        new_state["ragas_scores"] = run_ragas_evaluation(extracted_points)
    else:
        new_state["ragas_scores"] = None
    return new_state
