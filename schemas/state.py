# schemas/state.py
# LangGraph state TypedDict for the AutoResearch pipeline.

from typing import TypedDict, List, Dict, Any, Optional


class AutoResearchState(TypedDict, total=False):
    topic: str
    sub_questions: List[str]
    search_results: Dict[str, List[Dict[str, str]]]
    extracted_points: List[Dict[str, Any]]
    reader_failed: bool
    writer_failed: bool
    critic_failed: bool
    draft_report: str
    best_report: str
    best_score: int
    critique: Optional[Dict[str, Any]]
    revision_count: int
    final_report: str
    run_ragas: bool                       # NEW: user toggle carried in state
    ragas_scores: Optional[Dict[str, Any]]  # NEW: metric results
