from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    architecture: str  # "A", "B", "C", or "D"
    timeout_s: int = 60
    session_id: str | None = None


class UsageInfo(BaseModel):
    total_tokens: int = 0
    latency_s: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    architecture: str
    hallucination_risk: bool
    usage: UsageInfo
    metadata: dict[str, Any] = {}


class ArchSummary(BaseModel):
    n: int
    accuracy_mean: float
    accuracy_ci_95: list[float] = []
    hallucination_mean: float
    hallucination_ci_95: list[float] = []
    traceability_mean: float
    traceability_ci_95: list[float] = []
    error_handling_mean: float
    error_handling_ci_95: list[float] = []
    latency_mean_s: float
    total_tokens_mean: float
    accuracy_by_level: dict[str, float] = {}


class BenchmarkResults(BaseModel):
    summary: dict[str, ArchSummary]
    csv_rows: list[dict[str, Any]]


class BenchmarkRunRequest(BaseModel):
    architectures: list[str] = ["A", "B", "C", "D"]
    level: int | None = None
    n_questions: int | None = None


class CompareRequest(BaseModel):
    question: str
    architectures: list[str] = ["A", "B", "C", "D"]


class FigureList(BaseModel):
    figures: list[str]


# ── New models ────────────────────────────────────────────────────────────────

class QuestionItem(BaseModel):
    id: str
    question: str
    level: int
    domain: str
    expected_keywords: list[str] = []


class QuestionCatalog(BaseModel):
    total: int
    questions: list[QuestionItem]


class SessionCreateResponse(BaseModel):
    session_id: str
    created_at: float


class SessionInfo(BaseModel):
    session_id: str
    turns: int
    created_at: float
    last_used: float
