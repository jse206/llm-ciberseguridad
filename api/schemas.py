from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    architecture: str  # "A", "B", "C", or "D"


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
    hallucination_mean: float
    traceability_mean: float
    error_handling_mean: float
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


class FigureList(BaseModel):
    figures: list[str]
