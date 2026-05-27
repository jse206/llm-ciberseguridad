from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CompareRequest,
    SessionCreateResponse,
    SessionInfo,
    UsageInfo,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_IMPL_DIR = Path(__file__).parent.parent.parent / "03_implementacion"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

_CHAINS = {
    "A": ("arquitectura_A_Text2SQL.chain", "ArchitectureAChain"),
    "B": ("arquitectura_B_API.chain", "ArchitectureBChain"),
    "C": ("arquitectura_C_GraphRAG.chain", "ArchitectureCChain"),
    "D": ("arquitectura_D_Toolformer.chain", "ArchitectureDChain"),
}

_ARCH_META = {
    "A": {
        "name": "Text2SQL",
        "description": "Consulta la BD NVD/CVE local mediante SQL generado por LLM",
        "data_source": "NVD SQLite local (2018-2025)",
        "strengths": ["Sin alucinación", "Trazabilidad 100%", "Consultas de agregación"],
        "weaknesses": ["Limitado a CVEs descargados", "Mayor latencia"],
    },
    "B": {
        "name": "API Calls",
        "description": "Consulta NVD API v2.0 y AbuseIPDB en tiempo real",
        "data_source": "NVD API v2.0 + AbuseIPDB (tiempo real)",
        "strengths": ["Datos siempre actualizados", "Menor latencia"],
        "weaknesses": ["Depende de conectividad", "Alta tasa de alucinación"],
    },
    "C": {
        "name": "GraphRAG",
        "description": "Razonamiento sobre el grafo MITRE ATT&CK",
        "data_source": "MITRE ATT&CK Enterprise (NetworkX)",
        "strengths": ["Relaciones multi-hop", "Alta trazabilidad"],
        "weaknesses": ["Solo dominio ATT&CK", "No responde sobre CVEs"],
    },
    "D": {
        "name": "Toolformer",
        "description": "Agente ReAct que selecciona A, B o C según la pregunta",
        "data_source": "NVD local + NVD API + MITRE ATT&CK",
        "strengths": ["Mayor accuracy", "Sin alucinación", "Razonamiento explícito"],
        "weaknesses": ["Mayor latencia", "Coste de tokens más alto"],
    },
}

_CHAIN_INSTANCES: dict = {}

# ── Cache ─────────────────────────────────────────────────────────────────────
@dataclass
class _CacheEntry:
    data: Any
    ts: float = field(default_factory=time.time)


_response_cache: dict[str, _CacheEntry] = {}
_CACHE_TTL = 300.0


def _cache_get(key: str) -> Any | None:
    e = _response_cache.get(key)
    if e and time.time() - e.ts < _CACHE_TTL:
        return e.data
    return None


def _cache_set(key: str, data: Any) -> None:
    _response_cache[key] = _CacheEntry(data=data)
    if len(_response_cache) > 500:
        oldest = min(_response_cache, key=lambda k: _response_cache[k].ts)
        del _response_cache[oldest]


# ── Rate limiter ──────────────────────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 20  # requests per minute per IP


def _check_rate(ip: str) -> None:
    now = time.time()
    _rate_store[ip] = [t for t in _rate_store[ip] if now - t < 60]
    if len(_rate_store[ip]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: máximo {_RATE_LIMIT} peticiones/minuto por IP.",
        )
    _rate_store[ip].append(now)


# ── Sessions ──────────────────────────────────────────────────────────────────
@dataclass
class _Session:
    session_id: str
    history: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


_sessions: dict[str, _Session] = {}
_SESSION_TTL = 3600.0


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [k for k, v in _sessions.items() if now - v.last_used > _SESSION_TTL]
    for k in expired:
        del _sessions[k]


def _get_session(session_id: str) -> _Session | None:
    _cleanup_sessions()
    s = _sessions.get(session_id)
    if s:
        s.last_used = time.time()
    return s


def _get_chain(arch: str):
    if arch not in _CHAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Arquitectura '{arch}' no válida. Usa A, B, C o D.",
        )
    if arch not in _CHAIN_INSTANCES:
        module_path, class_name = _CHAINS[arch]
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        _CHAIN_INSTANCES[arch] = cls()
    return _CHAIN_INSTANCES[arch]


def _build_question_with_context(session: _Session, question: str) -> str:
    if not session.history:
        return question
    ctx_lines = []
    for turn in session.history[-3:]:  # last 3 turns for context
        ctx_lines.append(f"P: {turn['question']}\nR: {turn['answer'][:400]}")
    ctx = "\n\n".join(ctx_lines)
    return f"Contexto de la conversación anterior:\n{ctx}\n\nPregunta actual: {question}"


def _build_response(result: dict, arch_key: str) -> ChatResponse:
    usage_raw = result.get("usage", {})
    total_latency = sum(
        v for k, v in usage_raw.items()
        if k.endswith("_latency_s") and isinstance(v, (int, float))
    )
    if total_latency == 0:
        total_latency = usage_raw.get("latency_s", 0.0)
    usage = UsageInfo(
        total_tokens=usage_raw.get("total_tokens", 0),
        latency_s=round(total_latency, 3),
    )
    _STRIP = {"sql", "sql_error", "api_fallback"}
    metadata = {
        k: v for k, v in result.items()
        if k not in ("answer", "hallucination_risk", "usage", "architecture")
        and k not in _STRIP
    }
    return ChatResponse(
        answer=result.get("answer", ""),
        architecture=result.get("architecture", arch_key),
        hallucination_risk=bool(result.get("hallucination_risk", False)),
        usage=usage,
        metadata=metadata,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/session", response_model=SessionCreateResponse)
def create_session() -> SessionCreateResponse:
    _cleanup_sessions()
    sid = str(uuid.uuid4())
    _sessions[sid] = _Session(session_id=sid)
    return SessionCreateResponse(session_id=sid, created_at=_sessions[sid].created_at)


@router.get("/session/{session_id}", response_model=SessionInfo)
def get_session_info(session_id: str) -> SessionInfo:
    session = _get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o expirada.")
    return SessionInfo(
        session_id=session.session_id,
        turns=len(session.history),
        created_at=session.created_at,
        last_used=session.last_used,
    )


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id in _sessions:
        del _sessions[session_id]
    return {"deleted": session_id}


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    _check_rate(req.client.host if req.client else "unknown")

    # Session context injection
    question = request.question
    session = None
    if request.session_id:
        session = _get_session(request.session_id)
        if session:
            question = _build_question_with_context(session, request.question)

    # Cache (no session-aware requests, to avoid stale context)
    cache_key = f"{request.architecture}:{request.question}"
    if not request.session_id:
        cached = _cache_get(cache_key)
        if cached:
            return cached

    chain = _get_chain(request.architecture)
    loop = asyncio.get_event_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, chain.run, question),
            timeout=float(request.timeout_s),
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"La arquitectura {request.architecture} superó el timeout "
                f"de {request.timeout_s}s."
            ),
        )

    response = _build_response(result, request.architecture)

    if not request.session_id:
        _cache_set(cache_key, response)

    if session is not None:
        session.history.append({
            "question": request.question,
            "answer": result.get("answer", ""),
            "architecture": request.architecture,
        })

    return response


@router.get("/stream")
async def stream_chat(
    req: Request,
    question: str,
    architecture: str = "D",
    session_id: str | None = None,
    timeout_s: int = 90,
) -> EventSourceResponse:
    """SSE endpoint: streams ReAct steps for Architecture D in real-time."""
    _check_rate(req.client.host if req.client else "unknown")

    q = question
    if session_id:
        session = _get_session(session_id)
        if session:
            q = _build_question_with_context(session, question)

    chain = _get_chain(architecture)
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def on_step(step_data: dict) -> None:
        asyncio.run_coroutine_threadsafe(event_queue.put(step_data), loop)

    def run_chain() -> None:
        try:
            kwargs = {"on_step": on_step} if architecture == "D" else {}
            result = chain.run(q, **kwargs)
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": "done", "result": result}), loop
            )
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": "error", "message": str(exc)}), loop
            )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_chain)

    async def generator() -> AsyncGenerator:
        deadline = time.time() + timeout_s
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                yield {"event": "error", "data": json.dumps({"message": "Timeout"})}
                break
            try:
                event = await asyncio.wait_for(
                    event_queue.get(), timeout=min(remaining, 2.0)
                )
            except asyncio.TimeoutError:
                continue

            if event.get("type") == "done":
                result = event["result"]
                response = _build_response(result, architecture)
                if session_id:
                    s = _get_session(session_id)
                    if s:
                        s.history.append({
                            "question": question,
                            "answer": result.get("answer", ""),
                            "architecture": architecture,
                        })
                yield {"event": "done", "data": json.dumps(response.model_dump())}
                break
            elif event.get("type") == "error":
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"message": event.get("message", "Error desconocido")}
                    ),
                }
                break
            else:
                yield {"event": "step", "data": json.dumps(event)}

    return EventSourceResponse(generator())


@router.post("/compare")
async def compare(request: CompareRequest, req: Request) -> dict:
    _check_rate(req.client.host if req.client else "unknown")
    archs = [a for a in request.architectures if a in _CHAINS]
    if not archs:
        raise HTTPException(status_code=400, detail="Ninguna arquitectura válida.")

    loop = asyncio.get_event_loop()

    def run_one(arch: str) -> tuple[str, ChatResponse]:
        chain = _get_chain(arch)
        result = chain.run(request.question)
        return arch, _build_response(result, arch)

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(archs)) as executor:
        futures = {executor.submit(run_one, arch): arch for arch in archs}
        for future in concurrent.futures.as_completed(futures):
            try:
                arch, response = future.result()
                results[arch] = response.model_dump()
            except Exception as exc:
                arch = futures[future]
                results[arch] = {"error": str(exc), "architecture": arch}

    return {"question": request.question, "results": results}


@router.get("/architectures")
def list_architectures() -> dict:
    # Enrich with benchmark metrics if available
    summary: dict = {}
    try:
        summary_path = (
            _PROJECT_ROOT / "04_benchmark" / "resultados" / "results_summary.json"
        )
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        pass

    result = {}
    for key, meta in _ARCH_META.items():
        entry = dict(meta)
        if key in summary:
            entry["avg_latency_s"] = summary[key].get("latency_mean_s")
            entry["accuracy_mean"] = summary[key].get("accuracy_mean")
            entry["hallucination_mean"] = summary[key].get("hallucination_mean")
        result[key] = entry

    return {"architectures": result}


@router.get("/cache/stats")
def cache_stats() -> dict:
    now = time.time()
    active = sum(1 for e in _response_cache.values() if now - e.ts < _CACHE_TTL)
    return {
        "total_entries": len(_response_cache),
        "active_entries": active,
        "ttl_s": _CACHE_TTL,
    }


@router.delete("/cache")
def clear_cache() -> dict:
    _response_cache.clear()
    return {"cleared": True}
