"""
Extrae el esquema de la base de datos SQLite NVD/CVE y lo formatea
para incluirlo en los prompts de Text2SQL.

Referencia: Pourreza & Rafiei (2023). DIN-SQL. NeurIPS 2023.
               Dong et al. (2023). C3 Text-to-SQL. arXiv:2307.07306.

El schema linking (paso 1 de DIN-SQL) necesita conocer exactamente
las tablas, columnas, tipos y valores de ejemplo para que el LLM
pueda generar SQL preciso sin alucinar nombres de columna.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from data.nvd_db import get_conn
from config import NVD_DB_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_schema_str(db_path: str | None = None) -> str:
    """
    Devuelve el esquema completo como string para incluir en los prompts.
    Se cachea en memoria para no consultar la BD en cada llamada.
    """
    path = Path(db_path) if db_path else NVD_DB_PATH
    with get_conn(path) as conn:
        tables = _get_tables(conn)
        parts = []
        for table in tables:
            cols = _get_columns(conn, table)
            samples = _get_samples(conn, table)
            parts.append(_format_table(table, cols, samples))
    schema = "\n\n".join(parts)
    logger.debug("Schema generado (%d chars)", len(schema))
    return schema


def _get_tables(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def _get_columns(conn, table: str) -> list[dict]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [{"name": r["name"], "type": r["type"], "pk": bool(r["pk"])} for r in rows]


def _get_samples(conn, table: str, n: int = 3) -> list[dict]:
    """Devuelve n filas de ejemplo (sin columnas de texto largo)."""
    try:
        rows = conn.execute(f"SELECT * FROM {table} LIMIT {n}").fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _format_table(table: str, cols: list[dict], samples: list[dict]) -> str:
    col_defs = ", ".join(
        f"{c['name']} {c['type']}{'(PK)' if c['pk'] else ''}" for c in cols
    )
    lines = [f"TABLE {table} ({col_defs})"]
    if samples:
        lines.append("  Ejemplos de valores:")
        for row in samples:
            # Truncar valores largos para no saturar el prompt
            preview = {k: (str(v)[:60] + "…" if isinstance(v, str) and len(str(v)) > 60 else v)
                       for k, v in row.items()}
            lines.append(f"    {preview}")
    return "\n".join(lines)


def get_column_names(table: str, db_path: str | None = None) -> list[str]:
    """Utilidad para validar que las columnas generadas por el LLM existen."""
    path = Path(db_path) if db_path else NVD_DB_PATH
    with get_conn(path) as conn:
        cols = _get_columns(conn, table)
    return [c["name"] for c in cols]
