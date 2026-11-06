# AutoResearch

AutoResearch is a Streamlit research assistant built around five agents. It plans sub-questions, searches the web with Tavily, retrieves evidence with dense and BM25 search, drafts a cited report, and asks a critic to review it.

## What it does

1. **Planner:** creates three research questions. If planning fails, a deterministic fallback plan is used.
2. **Searcher:** searches those questions concurrently, with a configurable concurrency limit.
3. **Reader:** fetches result pages concurrently, falls back to Tavily snippets, chunks the text explicitly, and fuses Chroma vector retrieval with BM25 using Reciprocal Rank Fusion (RRF).
4. **Writer:** drafts one finding per question and cites only sources retrieved for that question.
5. **Critic:** scores grounding, completeness, and clarity and can request up to two revisions.

The app displays a local agent trace and a runtime breakdown for planner, search, page fetching, indexing, retrieval, RAGAS, writing, and critique. The Reader's RRF ranks keyword and vector results together; see the [RRF paper](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/).

## Quality evaluation

The interactive RAGAS scores are diagnostics, not ground truth. Faithfulness and answer relevancy are LLM-judge estimates. Context precision and context recall use the generated answer as a proxy reference, so the UI labels them as proxy scores.

For a separate reference-based comparison, evaluation/golden_set.json contains five fixed questions, human-written expected answers, and authoritative source URLs. Run:

~~~powershell
python evaluation/run_golden_eval.py
~~~

This runs Tavily search and Reader retrieval for each question, then evaluates the answers against the curated references with RAGAS. It also reports whether retrieved sources cover the expected source domains. This makes live API calls and can take several minutes. To check retrieval and expected-source coverage without RAGAS:

~~~powershell
python evaluation/run_golden_eval.py --limit 1 --skip-ragas
~~~

Results are written to evaluation/results/, which is ignored by Git. The five-case set is a small regression benchmark; expand it with cases from your own research domains before treating its average as a broad quality score.

## Requirements

- Python 3.13
- Google AI Studio API key for Gemini
- Tavily API key for web search
- Optional LangSmith API key for hosted tracing

requirements.in lists the pinned direct dependencies. requirements.txt is the generated lock containing pinned transitive dependencies as well. If you intentionally update dependencies, regenerate the lock with:

~~~powershell
pip install pip-tools
pip-compile --upgrade requirements.in --output-file requirements.txt
~~~

## Run locally on Windows

From the repository root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
streamlit run app.py
~~~

Set GOOGLE_API_KEY and TAVILY_API_KEY in .env. For hosted tracing, set LANGSMITH_TRACING=true, LANGSMITH_PROJECT=AutoResearch, and LANGSMITH_API_KEY. The app's sidebar links to LangSmith. The in-page Agent Trace is local to the Streamlit session.

Concurrency, chunking, and retrieval settings can be adjusted in .env:

| Variable | Default | Purpose |
|---|---:|---|
| SEARCH_CONCURRENCY | 3 | Concurrent Tavily requests |
| FETCH_CONCURRENCY | 6 | Concurrent page fetches |
| FETCH_TIMEOUT_SECONDS | 12 | Timeout for each page |
| CHUNK_SIZE | 800 | SentenceSplitter chunk size |
| CHUNK_OVERLAP | 120 | Chunk overlap |
| RETRIEVAL_TOP_K | 6 | Candidates from each retrieval method and final RRF list |
| RRF_K | 60 | RRF rank smoothing constant |

## Run with Docker

Install and start Docker Desktop, then create .env from .env.example and enter your API keys. From the repository root run:

~~~powershell
Copy-Item .env.example .env
notepad .env
docker compose up --build
~~~

Open [http://localhost:8501](http://localhost:8501). Stop the app with Ctrl+C, then remove the container with:

~~~powershell
docker compose down
~~~

The Hugging Face embedding model downloads the first time the container starts. Docker Compose keeps its cache in a named volume so later starts can reuse it. API keys are passed at runtime through .env; .env is excluded from the image build context and must never be committed.

## Project structure

~~~
app.py                         Streamlit entry point
agents/                        Planner, Searcher, Reader, Writer, Critic
clients/                       Gemini and Tavily clients
config/                        Environment settings and runtime constants
evaluation/
  golden_set.json              Human-referenced benchmark cases
  run_golden_eval.py           Reference-based benchmark runner
  ragas_eval.py                Interactive RAGAS diagnostics
graph/                         LangGraph builder, routing, terminal nodes
schemas/                       Typed state and Pydantic outputs
testers/
  test_routing.py              Fast routing tests
  test_helper.py               Fast helper tests
  run_e2e_topics.py            Live API end-to-end topic runner
ui/                            Streamlit views and components
requirements.in                Pinned direct dependencies
requirements.txt               Fully pinned dependency lock
Dockerfile
compose.yaml
~~~

## Checks

~~~powershell
python -m pytest testers/test_routing.py testers/test_helper.py -v
python testers/run_e2e_topics.py --limit 1 --no-ragas
python evaluation/run_golden_eval.py --limit 1
~~~

The first command uses no model or search API calls. The latter two use live services and may incur API usage.

## Known limitations

- The Chroma index is ephemeral for each run; report history is held in the Streamlit session.
- Live RAGAS proxy metrics are not a substitute for independently labeled references.
- The golden benchmark is intentionally small and should grow with reviewed examples from the intended research domains.
