"""
Pipeline principal de la Arquitectura C: LLM + GraphRAG sobre MITRE ATT&CK.

Flujo:
  Pregunta → [Entity Matching + BFS] → subgrafo relevante
           → [Serialización a texto] → contexto estructurado
           → [Synthesis LLM]         → respuesta fundamentada en el grafo

Ventaja diferencial frente a A y B:
- Razonamiento multi-salto sobre relaciones explícitas del grafo
  (p. ej. "qué grupos APT usan técnicas de la táctica Lateral Movement")
- No requiere API externa ni BD local de CVEs
- Preserva la semántica relacional del framework MITRE ATT&CK

Limitación: cobertura acotada al dominio ATT&CK; no responde sobre CVEs.

Referencia: Edge, D. et al. (2024). arXiv:2404.16130.
               Pan, S. et al. (2024). IEEE TKDE 36, 3580-3599.
"""

from __future__ import annotations

import json
import logging
import time

from shared.llm_client import chat
from arquitectura_C_GraphRAG.graph_retriever import retrieve
from arquitectura_C_GraphRAG.prompts import (
    GRAPH_QA_SYSTEM,
    GRAPH_QA_USER_TEMPLATE,
    NO_CONTEXT_SYSTEM,
    NO_CONTEXT_USER_TEMPLATE,
    COVERAGE_SYSTEM,
    COVERAGE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class ArchitectureCChain:
    """
    Arquitectura C: LLM con recuperación sobre el grafo MITRE ATT&CK.

    Atributos del resultado (para el benchmark):
      - answer          : respuesta en lenguaje natural
      - seed_nodes      : nodos semilla identificados por entity matching
      - subgraph_nodes  : total de nodos en el subgrafo recuperado
      - subgraph_edges  : total de aristas en el subgrafo recuperado
      - coverage        : dict con análisis de suficiencia del contexto
      - hallucination_risk: True si el subgrafo estaba vacío
      - usage           : tokens y latencias de cada paso LLM
      - architecture    : "C_GraphRAG"
    """

    def __init__(self, hops: int = 2):
        self.hops = hops

    def run(self, question: str) -> dict:
        total_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "retrieval_latency_s": 0,
            "coverage_latency_s": 0,
            "synthesis_latency_s": 0,
        }

        # ── Paso 1: Recuperación del subgrafo ──────────────────────────────────
        t0 = time.perf_counter()
        retrieval = retrieve(question, hops=self.hops)
        total_usage["retrieval_latency_s"] = round(time.perf_counter() - t0, 3)
        seed_nodes = retrieval["seed_nodes"]
        nodes = retrieval["nodes"]
        edges = retrieval["edges"]
        graph_context = retrieval["context"]

        # ── Paso 2: Fallback si no hay contexto ───────────────────────────────
        if not nodes:
            answer, su = chat(
                messages=[{"role": "user", "content": NO_CONTEXT_USER_TEMPLATE.format(
                    question=question
                )}],
                system=NO_CONTEXT_SYSTEM,
                temperature=0.0,
            )
            _add_usage(total_usage, su)
            total_usage["synthesis_latency_s"] = su["latency_s"]
            return {
                "answer": answer,
                "seed_nodes": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "coverage": {"sufficient": False, "missing_entities": [], "confidence": 0.0},
                "hallucination_risk": True,
                "usage": total_usage,
                "architecture": "C_GraphRAG",
            }

        # ── Paso 3: Análisis de cobertura (métrica del benchmark) ─────────────
        coverage_text, cu = chat(
            messages=[{"role": "user", "content": COVERAGE_USER_TEMPLATE.format(
                question=question,
                seed_nodes=seed_nodes,
                total_nodes=len(nodes),
                total_edges=len(edges),
            )}],
            system=COVERAGE_SYSTEM,
            temperature=0.0,
            max_tokens=256,
        )
        _add_usage(total_usage, cu)
        total_usage["coverage_latency_s"] = cu["latency_s"]
        coverage = _parse_json(coverage_text, default={
            "sufficient": bool(seed_nodes),
            "missing_entities": [],
            "confidence": 0.5,
        })

        # ── Paso 4: Síntesis ───────────────────────────────────────────────────
        # Truncar el contexto si es muy largo para evitar superar el contexto del LLM
        if len(graph_context) > 10000:
            graph_context = graph_context[:10000] + "\n[... subgrafo truncado por longitud ...]"

        answer, synth_u = chat(
            messages=[{"role": "user", "content": GRAPH_QA_USER_TEMPLATE.format(
                question=question,
                graph_context=graph_context,
            )}],
            system=GRAPH_QA_SYSTEM,
            temperature=0.0,
        )
        _add_usage(total_usage, synth_u)
        total_usage["synthesis_latency_s"] = synth_u["latency_s"]

        return {
            "answer": answer,
            "seed_nodes": seed_nodes,
            "subgraph_nodes": len(nodes),
            "subgraph_edges": len(edges),
            "coverage": coverage,
            "hallucination_risk": not coverage.get("sufficient", False),
            "usage": total_usage,
            "architecture": "C_GraphRAG",
        }


def _add_usage(total: dict, step: dict) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[k] = total.get(k, 0) + step.get(k, 0)


def _parse_json(text: str, default: dict) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return default
