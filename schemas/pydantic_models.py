# schemas/pydantic_models.py
# Pydantic structured-output models used by the agents.

from typing import List, Literal
from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    sub_questions: List[str]

class Finding(BaseModel):
    sub_question: str
    content: str
    sources: List[str] = Field(default_factory=list)

class DraftReport(BaseModel):
    title: str
    introduction: str
    findings: List[Finding]
    conclusion: str

class CriticEvaluation(BaseModel):
    quality_score: int = Field(ge=1, le=10)
    issues: List[str]
    verdict: Literal["PASS", "REVISE"]
