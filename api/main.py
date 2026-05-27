from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Añadir 03_implementacion/ al path antes de importar los routers
_IMPL_DIR = Path(__file__).parent.parent / "03_implementacion"
if str(_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(_IMPL_DIR))

from api.routers import chat, benchmark, report  # noqa: E402

app = FastAPI(
    title="LLM Ciberseguridad — API",
    description="API para comparar arquitecturas LLM en análisis de vulnerabilidades",
    version="1.0.0",
)

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8001,http://localhost:3000,http://127.0.0.1:8001",
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(chat.router)
app.include_router(benchmark.router)
app.include_router(report.router)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Health-check state ────────────────────────────────────────────────────────
_start_time = time.time()
_health_cache: dict = {}
_health_cache_ts: float = 0.0
_HEALTH_CACHE_TTL = 30.0


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
def health() -> dict:
    global _health_cache, _health_cache_ts

    now = time.time()
    uptime_s = int(now - _start_time)

    # Return cached expensive fields if still fresh
    if now - _health_cache_ts < _HEALTH_CACHE_TTL and _health_cache:
        cached = dict(_health_cache)
        cached["uptime_s"] = uptime_s
        return cached

    # ── Expensive lookups ─────────────────────────────────────────────────────
    db_rows = 0
    graph_nodes = 0

    try:
        import config as _cfg  # type: ignore[import]
        import sqlite3

        if _cfg.NVD_DB_PATH.exists():
            conn = sqlite3.connect(str(_cfg.NVD_DB_PATH))
            row = conn.execute("SELECT COUNT(*) FROM cve").fetchone()
            conn.close()
            db_rows = row[0] if row else 0
    except Exception:
        pass

    try:
        import pickle
        import config as _cfg  # type: ignore[import]

        if _cfg.MITRE_GRAPH_PATH.exists():
            with open(_cfg.MITRE_GRAPH_PATH, "rb") as fh:
                g = pickle.load(fh)
            graph_nodes = g.number_of_nodes()
    except Exception:
        pass

    openai_key = bool(os.getenv("OPENAI_API_KEY", ""))
    nvd_key = bool(os.getenv("NVD_API_KEY", ""))

    _health_cache = {
        "status": "ok",
        "uptime_s": uptime_s,
        "db_rows": db_rows,
        "graph_nodes": graph_nodes,
        "openai_key": openai_key,
        "nvd_key": nvd_key,
    }
    _health_cache_ts = now
    return _health_cache
