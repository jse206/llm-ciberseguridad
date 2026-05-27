"""
Pipeline principal de la Arquitectura A: LLM + Text2SQL sobre NVD/CVE.

Implementa la estrategia DIN-SQL (Pourreza & Rafiei, NeurIPS 2023)
con cuatro pasos encadenados:
  1. Schema Linking  — identifica tablas y columnas relevantes.
  2. SQL Generation  — genera la consulta SQLite.
  3. SQL Execution   — ejecuta contra la BD local y captura errores.
  4. Self-Correction — si falla, el LLM corrige la consulta (máx. 2 intentos).
  5. Synthesis       — el LLM redacta la respuesta en lenguaje natural.

Ventaja frente a Arquitectura B: consultas offline sin coste de API,
mayor velocidad y capacidad de responder preguntas de agregación complejas.
Limitación: el conocimiento está acotado a los años descargados en NVD_YEARS.

Referencia: Pourreza & Rafiei (2023). DIN-SQL. NeurIPS 2023.
               Dong et al. (2023). C3 Text-to-SQL. arXiv:2307.07306.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from shared.llm_client import chat
from data.nvd_db import get_conn
from config import NVD_DB_PATH
from arquitectura_A_Text2SQL.schema_inspector import get_schema_str
from arquitectura_A_Text2SQL.prompts import (
    SCHEMA_LINKING_SYSTEM,
    SCHEMA_LINKING_USER_TEMPLATE,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER_TEMPLATE,
    SELF_CORRECTION_SYSTEM,
    SELF_CORRECTION_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_SQL_TAG_RE = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
_MAX_CORRECTION_ATTEMPTS = 2
_MAX_RESULT_ROWS = 20      # limita filas enviadas al sintetizador
_MAX_RESULT_CHARS = 6000   # limita caracteres enviados al sintetizador

# Palabras clave que indican sentencias peligrosas aunque vayan tras un SELECT
_DANGEROUS_SQL_RE = re.compile(
    r"\b(DROP|INSERT|UPDATE|DELETE|CREATE|ALTER|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def _extract_sql(text: str) -> str | None:
    """Extrae el SQL entre etiquetas <sql>…</sql>."""
    m = _SQL_TAG_RE.search(text)
    return m.group(1).strip() if m else None


def _is_safe_sql(sql: str) -> bool:
    """Rechaza sentencias que no sean SELECT puro o que contengan DDL/DML peligroso."""
    stripped = sql.strip()
    first_token = stripped.split()[0].upper() if stripped else ""
    if first_token != "SELECT":
        return False
    return not _DANGEROUS_SQL_RE.search(stripped)


def _execute_sql(sql: str, db_path: Path = NVD_DB_PATH) -> tuple[list[dict], str | None]:
    """
    Ejecuta la consulta y devuelve (rows, error).
    rows es una lista de dicts; error es None si todo fue bien.
    """
    if not _is_safe_sql(sql):
        return [], "Solo se permiten consultas SELECT."
    try:
        with get_conn(db_path) as conn:
            cursor = conn.execute(sql)
            rows = [dict(r) for r in cursor.fetchmany(_MAX_RESULT_ROWS)]
        return rows, None
    except sqlite3.Error as e:
        return [], str(e)


def _try_api_fallback(question: str) -> tuple[list[dict], str | None]:
    """
    Fallback a NVD API cuando el SQL retorna 0 filas.
    Solo actúa si la pregunta menciona un CVE específico.
    La BD local cubre 2018-2025 desde la v2; CVEs anteriores o no descargados
    aún pueden recuperarse via API sin coste adicional de síntesis.
    """
    try:
        from arquitectura_B_API.api_client import get_cve_by_id  # noqa: PLC0415
    except ImportError:
        return [], None
    match = _CVE_RE.search(question)
    if not match:
        return [], None
    cve_id = match.group(0).upper()
    api_result = get_cve_by_id(cve_id=cve_id)
    if api_result.get("error") or not api_result.get("data"):
        return [], None
    data = api_result["data"]
    row: dict = {"fuente": f"NVD API ({cve_id})"}
    if isinstance(data, dict):
        row.update(data)
    logger.info("Fallback NVD API activado: %s (SQL devolvió 0 filas)", cve_id)
    return [row], f"NVD API — {cve_id}"


def _rows_to_str(rows: list[dict]) -> str:
    if not rows:
        return "(sin resultados)"
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(text) > _MAX_RESULT_CHARS:
        text = text[:_MAX_RESULT_CHARS] + "\n[... truncado ...]"
    return text


class ArchitectureAChain:
    """
    Arquitectura A: LLM + Text2SQL sobre la base de datos NVD/CVE local.

    Atributos del resultado (para el benchmark):
      - answer            : respuesta en lenguaje natural
      - sql               : consulta SQL finalmente ejecutada
      - sql_rows          : número de filas devueltas
      - correction_attempts: veces que el LLM autocorrigió el SQL
      - sql_error         : último error SQL si no se pudo corregir
      - hallucination_risk: True si el SQL no se pudo ejecutar correctamente
      - usage             : tokens y latencias de cada paso LLM
      - architecture      : "A_Text2SQL"
    """

    def __init__(self, db_path: Path = NVD_DB_PATH):
        self.db_path = db_path
        self._schema = get_schema_str(str(db_path))

    def run(self, question: str) -> dict:
        total_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "schema_linking_latency_s": 0,
            "sql_generation_latency_s": 0,
            "self_correction_latency_s": 0,
            "synthesis_latency_s": 0,
        }

        # ── Paso 1: Schema Linking ─────────────────────────────────────────────
        linking_text, lu = chat(
            messages=[{"role": "user", "content": SCHEMA_LINKING_USER_TEMPLATE.format(
                schema=self._schema, question=question
            )}],
            system=SCHEMA_LINKING_SYSTEM,
            temperature=0.0,
            max_tokens=512,
        )
        _add_usage(total_usage, lu)
        total_usage["schema_linking_latency_s"] = lu["latency_s"]
        linked_schema = self._parse_linking(linking_text)
        logger.info("Schema linking: %s", linked_schema)

        # ── Paso 2: SQL Generation ─────────────────────────────────────────────
        sql_text, su = chat(
            messages=[{"role": "user", "content": SQL_GENERATION_USER_TEMPLATE.format(
                schema=self._schema,
                linked_schema=json.dumps(linked_schema, ensure_ascii=False, indent=2),
                question=question,
            )}],
            system=SQL_GENERATION_SYSTEM,
            temperature=0.0,
            max_tokens=512,
        )
        _add_usage(total_usage, su)
        total_usage["sql_generation_latency_s"] = su["latency_s"]
        sql = _extract_sql(sql_text)
        logger.info("SQL generado: %s", sql)

        if not sql:
            return self._error_result(
                question, "El LLM no generó SQL válido.", total_usage
            )

        # ── Paso 3 + 4: Ejecución con Self-Correction ─────────────────────────
        correction_attempts = 0
        sql_error = None
        rows: list[dict] = []

        last_error: str | None = None
        for attempt in range(_MAX_CORRECTION_ATTEMPTS + 1):
            rows, sql_error = _execute_sql(sql, self.db_path)
            if sql_error is None:
                last_error = None
                break
            last_error = sql_error
            if attempt == _MAX_CORRECTION_ATTEMPTS:
                break

            logger.warning("SQL error (intento %d): %s", attempt + 1, sql_error)
            correction_attempts += 1

            corrected_text, cu = chat(
                messages=[{"role": "user", "content": SELF_CORRECTION_USER_TEMPLATE.format(
                    sql=sql, error=sql_error, schema=self._schema, question=question
                )}],
                system=SELF_CORRECTION_SYSTEM,
                temperature=0.0,
                max_tokens=512,
            )
            _add_usage(total_usage, cu)
            total_usage["self_correction_latency_s"] += cu["latency_s"]

            corrected = _extract_sql(corrected_text)
            if not corrected or "IMPOSIBLE_CORREGIR" in corrected:
                break
            sql = corrected
            logger.info("SQL corregido (intento %d): %s", attempt + 1, sql)
        sql_error = last_error

        # ── Fallback a NVD API si el SQL devolvió 0 filas ─────────────────────
        api_fallback_note = None
        if not rows and not sql_error:
            fallback_rows, api_fallback_note = _try_api_fallback(question)
            if fallback_rows:
                rows = fallback_rows

        # ── Paso 5: Síntesis ───────────────────────────────────────────────────
        answer, synth_u = chat(
            messages=[{"role": "user", "content": SYNTHESIZER_USER_TEMPLATE.format(
                question=question,
                sql=sql,
                results=_rows_to_str(rows),
            )}],
            system=SYNTHESIZER_SYSTEM,
            temperature=0.0,
        )
        _add_usage(total_usage, synth_u)
        total_usage["synthesis_latency_s"] = synth_u["latency_s"]

        return {
            "answer": answer,
            "sql": sql,
            "sql_rows": len(rows),
            "correction_attempts": correction_attempts,
            "sql_error": sql_error,
            "api_fallback": api_fallback_note,
            # Riesgo real solo si el SQL falló Y no hay datos de ninguna fuente
            "hallucination_risk": bool(sql_error) and len(rows) == 0,
            "usage": total_usage,
            "architecture": "A_Text2SQL",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_linking(self, text: str) -> dict:
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"tables": [], "columns": {}, "notes": "parse error"}

    def _error_result(self, question: str, reason: str, usage: dict) -> dict:
        return {
            "answer": f"No fue posible generar una consulta SQL válida: {reason}",
            "sql": None,
            "sql_rows": 0,
            "correction_attempts": 0,
            "sql_error": reason,
            "hallucination_risk": True,
            "usage": usage,
            "architecture": "A_Text2SQL",
        }


def _add_usage(total: dict, step: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[k] = total.get(k, 0) + step.get(k, 0)
