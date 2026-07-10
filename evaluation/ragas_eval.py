# evaluation/ragas_eval.py
# RAGAS evaluation: guarded import, monkeypatch, wrappers, evaluator.

import math
from typing import List, Dict, Any, Optional

# ---------- RAGAS (guarded import) ----------
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from datasets import Dataset
    from ragas.dataset_schema import EvaluationResult
    RAGAS_AVAILABLE = True

    import ragas.dataset_schema as _ragas_ds

    def _safe_parse_run_traces(traces, run_id=None):
        if not traces:
            return []
        try:
            from ragas.callbacks import parse_run_traces as _orig
            return _orig(traces, run_id)
        except IndexError:
            return []

    _ragas_ds.parse_run_traces = _safe_parse_run_traces
except ImportError as _e:
    RAGAS_AVAILABLE = False
    print(f"[WARNING] RAGAS not available ({_e}). "
          f'Run: pip install ragas datasets langchain-huggingface "langchain-community<0.4"')

RAGAS_METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ---------- RAGAS wrappers (built once, lazily) ----------
_ragas_llm_wrapper = None
_ragas_emb_wrapper = None

def get_ragas_wrappers():
    """Build (and cache) the RAGAS LLM + embeddings adapters.
    Local BGE embeddings prevent RAGAS from defaulting to OpenAI."""
    global _ragas_llm_wrapper, _ragas_emb_wrapper
    if _ragas_llm_wrapper is None:
        from clients.llm_clients import ragas_judge_llm
        _ragas_llm_wrapper = LangchainLLMWrapper(ragas_judge_llm)
        _ragas_emb_wrapper = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        )
    return _ragas_llm_wrapper, _ragas_emb_wrapper


def run_ragas_evaluation(extracted_points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Score Reader output with RAGAS.

    Ground-truth note: AutoResearch is autonomous, so we have no human
    reference answers. We use the generated answer as a PROXY reference,
    which makes context_precision / context_recall self-consistency
    metrics. faithfulness and answer_relevancy need no ground truth and
    are the headline numbers.
    """
    if not RAGAS_AVAILABLE:
        return {"error": "ragas not installed"}

    # Only evaluate rows that actually have an answer AND contexts.
    rows = [
        p for p in extracted_points
        if p.get("contexts") and p.get("answer")
        and p["answer"] != "Error: could not retrieve answer."
    ]
    if not rows:
        return {"error": "no evaluable rows (missing contexts or answers)"}

    dataset = Dataset.from_dict({
        "user_input": [p["question"] for p in rows],
        "response": [p["answer"] for p in rows],
        "retrieved_contexts": [p["contexts"] for p in rows],
        "reference": [p["answer"] for p in rows],  # proxy reference
    })

    try:
        judge, embeddings = get_ragas_wrappers()
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            if hasattr(m, "reproducibility"):
                m.reproducibility = 1

        print(f"[RAGAS] evaluating {len(rows)} rows x 4 metrics (this takes 1-3 min)...")
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge,
            embeddings=embeddings,
            raise_exceptions=False,   # a single failed judge call must not kill the run
        )
        if not isinstance(result, EvaluationResult):
            raise RuntimeError("ragas.evaluate() returned an Executor instead of EvaluationResult")
        scores: Dict[str, Any] = {}
        for name in RAGAS_METRIC_NAMES:
            row_scores = [row.get(name) for row in result.scores]
            clean_scores = []
            for value in row_scores:
                if value is None:
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(numeric_value):
                    continue
                clean_scores.append(numeric_value)
            scores[name] = round(sum(clean_scores) / len(clean_scores), 3) if clean_scores else None
        print(f"[RAGAS] scores: {scores}")
        return scores
    except Exception as e:
        import traceback
        print("[RAGAS][ERROR] Full traceback:")
        traceback.print_exc()
        return {"error": str(e)}
