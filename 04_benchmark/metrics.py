"""
Módulo de métricas del benchmark del TFM.

Implementa las 5 métricas de evaluación definidas en el Capítulo 4.3:

  1. Accuracy          — coincidencia de keywords verificables entre respuesta y ground truth.
  2. Tasa de alucinación — proporción de respuestas marcadas con riesgo de alucinación.
  3. Trazabilidad      — proporción de respuestas que citan su fuente explícitamente.
  4. Manejo de errores — proporción de errores gestionados correctamente sin inventar datos.
  5. Latencia          — tiempo de respuesta total en segundos.

Diseño orientado al TFM:
  - Cada función devuelve un valor escalar y una justificación textual.
  - `compute_all` devuelve el dict completo listo para exportar a CSV/Excel.
  - Las funciones son deterministas (no llaman a LLMs) para reproducibilidad.

Referencia: Liang et al. (2023). HELM. Annals NYAS.
            Es et al. (2024). RAGAS. EACL 2024.
            Li et al. (2023). HaluEval. EMNLP 2023.
"""

from __future__ import annotations

import random as _random
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Semilla fija para reproducibilidad del bootstrap CI (Efron & Hastie, 2016)
_rng = _random.Random(42)

# Palabras clave que indican citación de fuente en la respuesta
_SOURCE_MARKERS = [
    "nvd", "nist", "national vulnerability", "mitre", "att&ck", "abuseipdb",
    "según", "according to", "fuente", "source", "base de datos", "database",
    "api", "grafo", "graph", "consulta sql", "sql query",
]

# Frases que indican que el sistema reconoció un error sin inventar
_ERROR_HANDLING_MARKERS = [
    "no encontr", "not found", "no dispon", "not available",
    "error", "no fue posible", "unable", "sin resultados", "no result",
    "no tengo inform", "insuficiente", "insufficient", "no puedo",
    "el sistema no", "la api", "la base de datos no",
]


def _keyword_score(keyword: str, answer_lower: str) -> float:
    """
    Score 0-1 para un keyword individual.
    Exact match → 1.0; para keywords multi-token, crédito parcial proporcional
    a la fracción de tokens encontrados. Mejora sobre la versión binaria original
    para keywords compuestos como "lateral movement" o "valid accounts".
    """
    kw_lower = keyword.lower()
    if kw_lower in answer_lower:
        return 1.0
    tokens = kw_lower.split()
    if len(tokens) <= 1:
        return 0.0
    found = sum(1 for t in tokens if t in answer_lower)
    return round(found / len(tokens), 2)


def accuracy(answer: str, expected_keywords: list[str]) -> tuple[float, str]:
    """
    Calcula la exactitud como media de los scores por keyword.

    Para keywords de una sola palabra: score binario (0 o 1).
    Para keywords multi-palabra: crédito parcial proporcional a tokens encontrados.
    Mejora respecto a la versión binaria original, que penalizaba respuestas
    correctas con sinónimos o formulaciones ligeramente distintas.

    Returns:
        (score 0.0-1.0, justificación)
    """
    if not expected_keywords:
        return 1.0, "Sin keywords de referencia"

    answer_lower = answer.lower()
    per_kw = [(kw, _keyword_score(kw, answer_lower)) for kw in expected_keywords]
    score = sum(s for _, s in per_kw) / len(per_kw)
    full_match = [kw for kw, s in per_kw if s == 1.0]
    partial_match = [f"{kw}({s:.0%})" for kw, s in per_kw if 0 < s < 1.0]
    missing = [kw for kw, s in per_kw if s == 0.0]
    justification = (
        f"Score={score:.3f} | "
        f"completas={full_match} | "
        f"parciales={partial_match} | "
        f"ausentes={missing}"
    )
    return round(score, 4), justification


def hallucination_rate(result: dict) -> tuple[float, str]:
    """
    Estima el riesgo de alucinación combinando:
      - Flag `hallucination_risk` del pipeline (señal fuerte)
      - Presencia de afirmaciones sin fuente verificable (señal débil)

    Returns:
        (0.0 = sin riesgo, 1.0 = riesgo alto, justificación)
    """
    risk_flag = result.get("hallucination_risk", False)
    api_error = bool(result.get("api_error") or result.get("sql_error"))

    if risk_flag and api_error:
        return 1.0, "Flag activo + error de fuente detectado"
    if risk_flag:
        return 0.7, "Flag de riesgo activo (sin datos de fuente verificados)"
    if api_error:
        return 0.5, "Error de API/SQL — respuesta puede basarse en conocimiento parametrizado"
    return 0.0, "Sin señales de alucinación detectadas"


def source_traceability(answer: str, result: dict) -> tuple[float, str]:
    """
    Evalúa si la respuesta cita explícitamente su fuente de información.

    Combina:
      - Presencia de markers de fuente en el texto de la respuesta
      - Presencia de `api_source` o `seed_nodes` en los metadatos del resultado

    Returns:
        (0.0-1.0, justificación)
    """
    answer_lower = answer.lower()
    markers_found = [m for m in _SOURCE_MARKERS if m in answer_lower]
    has_api_source = bool(result.get("api_source") and result["api_source"] != "N/A")
    has_seed_nodes = bool(result.get("seed_nodes"))
    has_sql = bool(result.get("sql"))

    # Puntuación: markers en texto + metadatos estructurados
    text_score = min(len(markers_found) / 2, 1.0)  # hasta 1.0 con 2+ markers
    meta_score = 1.0 if (has_api_source or has_seed_nodes or has_sql) else 0.0
    score = round((text_score + meta_score) / 2, 4)

    justification = (
        f"Markers texto: {markers_found[:3]} | "
        f"Fuente estructurada: api_source={has_api_source}, "
        f"seed_nodes={has_seed_nodes}, sql={has_sql}"
    )
    return score, justification


def error_handling(result: dict, answer: str) -> tuple[float, str]:
    """
    Evalúa si el sistema maneja correctamente los errores sin inventar datos.

    Escenarios:
      - Sin error y respuesta correcta → 1.0
      - Con error y la respuesta lo reconoce honestamente → 0.8
      - Con error y la respuesta intenta responder de todos modos → 0.3
      - Con error y la respuesta no lo menciona → 0.0

    Returns:
        (0.0-1.0, justificación)
    """
    # iterations==0 en D no es error: el LLM puede dar Final Answer sin invocar herramientas.
    # forced_final=True sí indica fallo real (límite alcanzado o formato inválido del LLM).
    has_error = bool(
        result.get("api_error") or result.get("sql_error") or
        result.get("forced_final")
    )

    if not has_error:
        return 1.0, "Sin errores detectados en el pipeline"

    answer_lower = answer.lower()
    acknowledges_error = any(m in answer_lower for m in _ERROR_HANDLING_MARKERS)

    if acknowledges_error:
        return 0.8, "Error presente pero reconocido honestamente en la respuesta"

    # Comprueba si la respuesta es anormalmente corta (posible fallo silencioso)
    if len(answer.split()) < 15:
        return 0.1, "Error presente, respuesta demasiado corta — posible fallo silencioso"

    return 0.3, "Error presente — la respuesta no lo reconoce explícitamente"


def latency(result: dict) -> tuple[float, str]:
    """
    Extrae la latencia total de la respuesta en segundos.

    Returns:
        (latencia_total_segundos, justificación con desglose)
    """
    usage = result.get("usage", {})
    arch = result.get("architecture", "")

    if arch == "A_Text2SQL":
        total = (
            usage.get("schema_linking_latency_s", 0)
            + usage.get("sql_generation_latency_s", 0)
            + usage.get("self_correction_latency_s", 0)
            + usage.get("synthesis_latency_s", 0)
        )
        detail = (
            f"linking={usage.get('schema_linking_latency_s',0):.2f}s "
            f"sql={usage.get('sql_generation_latency_s',0):.2f}s "
            f"corr={usage.get('self_correction_latency_s',0):.2f}s "
            f"synth={usage.get('synthesis_latency_s',0):.2f}s"
        )
    elif arch == "B_API":
        total = (
            usage.get("router_latency_s", 0)
            + usage.get("api_latency_s", 0)
            + usage.get("synthesis_latency_s", 0)
        )
        detail = (
            f"router={usage.get('router_latency_s',0):.2f}s "
            f"api={usage.get('api_latency_s',0):.2f}s "
            f"synth={usage.get('synthesis_latency_s',0):.2f}s"
        )
    elif arch == "C_GraphRAG":
        total = (
            usage.get("retrieval_latency_s", 0)
            + usage.get("coverage_latency_s", 0)
            + usage.get("synthesis_latency_s", 0)
        )
        detail = (
            f"retrieval={usage.get('retrieval_latency_s',0):.2f}s "
            f"coverage={usage.get('coverage_latency_s',0):.2f}s "
            f"synth={usage.get('synthesis_latency_s',0):.2f}s"
        )
    elif arch == "D_Toolformer":
        total = usage.get("total_latency_s", 0)
        detail = f"total_react={total:.2f}s iterations={result.get('iterations',0)}"
    else:
        total = usage.get("total_latency_s", usage.get("synthesis_latency_s", 0))
        detail = f"total={total:.2f}s"

    return round(total, 3), detail


def compute_all(
    question: dict,
    result: dict,
) -> dict[str, Any]:
    """
    Calcula las 5 métricas para una pregunta y devuelve un dict
    listo para guardar en CSV y generar tablas del Capítulo 5.

    Args:
        question : entrada de questions.json (con expected_keywords, level, domain…)
        result   : salida del chain.run() de cualquier arquitectura

    Returns:
        Dict con todas las métricas y metadatos.
    """
    answer = result.get("answer", "")
    expected_keywords = question.get("expected_keywords", [])

    acc_score, acc_just = accuracy(answer, expected_keywords)
    hall_score, hall_just = hallucination_rate(result)
    trace_score, trace_just = source_traceability(answer, result)
    err_score, err_just = error_handling(result, answer)
    lat_score, lat_just = latency(result)

    return {
        # Identificación
        "question_id": question["id"],
        "level": question["level"],
        "domain": question["domain"],
        "architecture": result.get("architecture", "unknown"),
        # Métricas principales
        "accuracy": acc_score,
        "hallucination_risk": hall_score,
        "traceability": trace_score,
        "error_handling": err_score,
        "latency_s": lat_score,
        # Tokens
        "total_tokens": result.get("usage", {}).get("total_tokens", 0),
        # Justificaciones (para análisis cualitativo en Cap. 5)
        "accuracy_detail": acc_just,
        "hallucination_detail": hall_just,
        "traceability_detail": trace_just,
        "error_handling_detail": err_just,
        "latency_detail": lat_just,
        # Respuesta completa
        "answer_preview": answer[:300] + ("…" if len(answer) > 300 else ""),
    }


def _bootstrap_ci(
    values: list[float],
    n_iters: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Intervalo de confianza bootstrap al (1-alpha)*100%.
    Usa muestreo con reemplazamiento sobre la muestra original.
    Implementa la metodología recomendada por Efron & Hastie (2016)
    para comparaciones de sistemas NLP (Dror et al., 2018).
    """
    if len(values) < 2:
        v = values[0] if values else 0.0
        return round(v, 4), round(v, 4)
    n = len(values)
    boot_means = sorted(
        sum(_rng.choices(values, k=n)) / n
        for _ in range(n_iters)
    )
    lo = boot_means[int(alpha / 2 * n_iters)]
    hi = boot_means[min(int((1 - alpha / 2) * n_iters), n_iters - 1)]
    return round(lo, 4), round(hi, 4)


def aggregate(results: list[dict]) -> dict[str, Any]:
    """
    Calcula métricas agregadas para una arquitectura o subconjunto.
    Incluye intervalos de confianza bootstrap al 95% para las métricas
    principales, lo que permite comparaciones estadísticamente informadas
    entre arquitecturas (Dror et al., 2018; Bouthillier et al., 2021).
    Usado por report_generator.py para las tablas comparativas.
    """
    if not results:
        return {}

    def vals(key: str) -> list[float]:
        return [r[key] for r in results if r.get(key) is not None]

    def mean(key: str) -> float:
        v = vals(key)
        return round(sum(v) / len(v), 4) if v else 0.0

    def ci(key: str) -> list[float]:
        v = vals(key)
        lo, hi = _bootstrap_ci(v) if v else (0.0, 0.0)
        return [lo, hi]

    return {
        "n": len(results),
        "accuracy_mean": mean("accuracy"),
        "accuracy_ci_95": ci("accuracy"),
        "hallucination_mean": mean("hallucination_risk"),
        "hallucination_ci_95": ci("hallucination_risk"),
        "traceability_mean": mean("traceability"),
        "traceability_ci_95": ci("traceability"),
        "error_handling_mean": mean("error_handling"),
        "error_handling_ci_95": ci("error_handling"),
        "latency_mean_s": mean("latency_s"),
        "total_tokens_mean": mean("total_tokens"),
        "accuracy_by_level": {
            lvl: mean_for_level(results, "accuracy", lvl)
            for lvl in [1, 2, 3]
        },
    }


def mean_for_level(results: list[dict], metric: str, level: int) -> float:
    subset = [r[metric] for r in results if r.get("level") == level and r.get(metric) is not None]
    return round(sum(subset) / len(subset), 4) if subset else 0.0
