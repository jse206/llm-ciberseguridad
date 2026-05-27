from __future__ import annotations

import asyncio
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ArchSummary,
    BenchmarkResults,
    BenchmarkRunRequest,
    QuestionCatalog,
    QuestionItem,
)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_IMPL_DIR = _PROJECT_ROOT / "03_implementacion"
_BENCHMARK_DIR = _PROJECT_ROOT / "04_benchmark"
_RESULTS_DIR = _BENCHMARK_DIR / "resultados"

if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

_CHAINS = {
    "A": ("arquitectura_A_Text2SQL.chain", "ArchitectureAChain"),
    "B": ("arquitectura_B_API.chain", "ArchitectureBChain"),
    "C": ("arquitectura_C_GraphRAG.chain", "ArchitectureCChain"),
    "D": ("arquitectura_D_Toolformer.chain", "ArchitectureDChain"),
}


def _latest_file(pattern: str) -> Path | None:
    """Return the most recently modified file matching *pattern* in _RESULTS_DIR."""
    candidates = sorted(_RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


@router.get("/results", response_model=BenchmarkResults)
def get_results() -> BenchmarkResults:
    summary_path = _RESULTS_DIR / "results_summary.json"
    csv_path = _RESULTS_DIR / "results_metrics.csv"

    # Fall back to the latest timestamped files if the plain-named ones don't exist
    if not summary_path.exists():
        summary_path = _latest_file("results_summary_*.json")  # type: ignore[assignment]
    if not csv_path.exists():
        csv_path = _latest_file("results_metrics_*.csv")  # type: ignore[assignment]

    if not summary_path or not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No hay resultados de benchmark. Ejecuta el benchmark primero.",
        )

    summary_raw = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = {k: ArchSummary(**v) for k, v in summary_raw.items()}

    csv_rows: list[dict] = []
    if csv_path and csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

    return BenchmarkResults(summary=summary, csv_rows=csv_rows)


@router.get("/questions", response_model=QuestionCatalog)
def get_questions(
    level: int | None = None,
    domain: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> QuestionCatalog:
    questions_path = _BENCHMARK_DIR / "questions.json"
    if not questions_path.exists():
        raise HTTPException(status_code=404, detail="questions.json no encontrado.")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if level is not None:
        questions = [q for q in questions if q.get("level") == level]
    if domain:
        questions = [
            q for q in questions
            if domain.lower() in q.get("domain", "").lower()
        ]
    total = len(questions)
    page = questions[offset: offset + limit]
    return QuestionCatalog(
        total=total,
        questions=[
            QuestionItem(
                id=q["id"],
                question=q["question"],
                level=q["level"],
                domain=q.get("domain", ""),
                expected_keywords=q.get("expected_keywords", []),
            )
            for q in page
        ],
    )


@router.get("/results/detail")
def get_results_detail(
    arch: str | None = None,
    level: int | None = None,
    domain: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    csv_path = _RESULTS_DIR / "results_metrics.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No hay resultados CSV.")

    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if arch and not row.get("architecture", "").startswith(arch):
                continue
            if level is not None:
                try:
                    if int(row.get("level", 0)) != level:
                        continue
                except (ValueError, TypeError):
                    continue
            if domain and domain.lower() not in row.get("domain", "").lower():
                continue
            rows.append(row)

    total = len(rows)
    page = rows[offset: offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "rows": page}


@router.post("/run")
async def run_benchmark(request: Request, body: BenchmarkRunRequest):
    questions_path = _BENCHMARK_DIR / "questions.json"
    if not questions_path.exists():
        raise HTTPException(status_code=404, detail="questions.json no encontrado.")

    async def event_generator() -> AsyncGenerator:
        from metrics import compute_all, aggregate  # noqa: E402 — imported after sys.path setup

        questions = json.loads(questions_path.read_text(encoding="utf-8"))

        if body.level:
            questions = [q for q in questions if q["level"] == body.level]
        if body.n_questions:
            questions = questions[: body.n_questions]

        total_questions = len(questions) * len(body.architectures)
        done = 0

        all_metrics: list[dict] = []

        for arch_key in body.architectures:
            if arch_key not in _CHAINS:
                continue
            module_path, class_name = _CHAINS[arch_key]
            module = importlib.import_module(module_path)
            chain = getattr(module, class_name)()

            for question in questions:
                if await request.is_disconnected():
                    return

                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, chain.run, question["question"])
                except Exception as e:
                    result = {
                        "answer": f"Error: {e}",
                        "hallucination_risk": True,
                        "usage": {"total_tokens": 0},
                        "architecture": arch_key,
                    }

                metrics = compute_all(question, result)
                all_metrics.append(metrics)
                done += 1

                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "done": done,
                        "total": total_questions,
                        "architecture": arch_key,
                        "question_id": question["id"],
                        "accuracy": metrics["accuracy"],
                        "latency_s": metrics["latency_s"],
                    }),
                }

        # Summary final
        summary: dict[str, dict] = {}
        for arch_key in body.architectures:
            arch_metrics = [
                m for m in all_metrics
                if str(m.get("architecture", "")).startswith(arch_key)
            ]
            if arch_metrics:
                summary[arch_key] = aggregate(arch_metrics)

        yield {
            "event": "done",
            "data": json.dumps({"summary": summary}),
        }

    return EventSourceResponse(event_generator())
