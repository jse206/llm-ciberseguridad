"""
Pipeline principal de la Arquitectura D: Toolformer con ciclo ReAct.

El LLM actúa como agente que selecciona y encadena herramientas autónomamente.
Implementa el ciclo Thought → Action → Observation de ReAct (Yao et al., 2023)
con un límite de iteraciones para garantizar terminación.

Herramientas disponibles (definidas en tools.py):
  - query_nvd_local   → Arquitectura A (Text2SQL local)
  - query_nvd_api     → Arquitectura B (NVD API tiempo real)
  - query_mitre_graph → Arquitectura C (GraphRAG MITRE ATT&CK)
  - check_ip          → AbuseIPDB

Ventaja diferencial:
  - Selección autónoma de fuente según la naturaleza de la pregunta
  - Capacidad de combinar múltiples fuentes en una respuesta
  - Razonamiento explícito (Thoughts) trazable

Referencia: Schick et al. (2023). Toolformer. NeurIPS 2023.
               Yao et al. (2023). ReAct. ICLR 2023.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from shared.llm_client import chat
from arquitectura_D_Toolformer.tools import call_tool, TOOLS_BY_NAME
from arquitectura_D_Toolformer.prompts import (
    get_react_system,
    REACT_USER_TEMPLATE,
    OBSERVATION_TEMPLATE,
    FORCED_FINAL_SYSTEM,
    FORCED_FINAL_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 6
_ACTION_RE = re.compile(r"Action:\s*(\w+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL | re.IGNORECASE)


def _extract_action_input(text: str) -> dict:
    """
    Extrae el JSON de 'Action Input:' tolerando llaves anidadas.
    El regex simple falla con objetos JSON anidados porque captura
    solo hasta el primer '}'. Esta función cuenta llaves para encontrar
    el cierre correcto del objeto raíz.
    """
    marker = re.search(r"Action Input:\s*\{", text, re.IGNORECASE)
    if not marker:
        return {}
    start = marker.end() - 1  # posición del '{' de apertura
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


@dataclass
class ReActStep:
    thought: str = ""
    action: str = ""
    action_input: dict = field(default_factory=dict)
    observation: str = ""
    raw: str = ""
    latency_s: float = 0.0


class ArchitectureDChain:
    """
    Arquitectura D: Toolformer con ciclo ReAct.

    Atributos del resultado (para el benchmark):
      - answer          : respuesta en lenguaje natural (Final Answer)
      - steps           : lista de ReActStep con el trazado completo
      - tools_used      : herramientas invocadas (con repeticiones)
      - iterations      : número de ciclos Thought/Action completados
      - forced_final    : True si se forzó el cierre por límite de iteraciones
      - hallucination_risk: True si ninguna herramienta devolvió datos válidos
      - usage           : tokens y latencia total
      - architecture    : "D_Toolformer"
    """

    def __init__(self):
        self._system = get_react_system()

    def run(self, question: str, on_step=None) -> dict:
        # Sistema separado de la conversación para que chat() lo anteponga correctamente
        conversation: list[dict] = [
            {"role": "user", "content": REACT_USER_TEMPLATE.format(question=question)},
        ]

        total_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "total_latency_s": 0.0,
            "iteration_latencies_s": [],  # desglose por iteración para la API
        }
        steps: list[ReActStep] = []
        tools_used: list[str] = []
        forced_final = False
        answer = ""

        t_start = time.perf_counter()

        for iteration in range(_MAX_ITERATIONS):
            # ── Llamada al LLM (con retry via tenacity del cliente compartido) ──
            raw, step_usage = chat(
                messages=conversation,
                system=self._system,
                temperature=0.0,
                max_tokens=1024,
                stop=["Observation:"],
            )
            total_usage["prompt_tokens"] += step_usage["prompt_tokens"]
            total_usage["completion_tokens"] += step_usage["completion_tokens"]
            total_usage["total_tokens"] += step_usage["total_tokens"]
            total_usage["iteration_latencies_s"].append(step_usage["latency_s"])

            logger.debug("Iteración %d LLM output:\n%s", iteration + 1, raw)

            # ── ¿Ha llegado a Final Answer? ────────────────────────────────────
            final_match = _FINAL_RE.search(raw)
            if final_match:
                answer = final_match.group(1).strip()
                conversation.append({"role": "assistant", "content": raw})
                break

            # ── Parsear Action + Action Input ──────────────────────────────────
            step = ReActStep(raw=raw)
            action_match = _ACTION_RE.search(raw)

            if not action_match:
                logger.warning("Sin Action en iteración %d. Forzando Final Answer.", iteration + 1)
                forced_final = True
                break

            step.action = action_match.group(1).strip()
            step.action_input = _extract_action_input(raw)
            step.latency_s = step_usage["latency_s"]

            # ── Ejecutar herramienta ───────────────────────────────────────────
            if step.action not in TOOLS_BY_NAME:
                step.observation = f"Error: herramienta '{step.action}' no existe."
            else:
                tool_result = call_tool(step.action, step.action_input)
                tools_used.append(step.action)
                obs_str = json.dumps(tool_result, ensure_ascii=False, indent=2, default=str)
                if len(obs_str) > 3000:
                    obs_str = obs_str[:3000] + "\n[... truncado ...]"
                step.observation = obs_str

            steps.append(step)

            # ── Notificar al caller si hay callback ───────────────────────────
            if on_step is not None:
                try:
                    on_step({
                        "type": "step",
                        "iteration": iteration + 1,
                        "action": step.action,
                        "observation_preview": step.observation[:200],
                        "latency_s": step.latency_s,
                    })
                except Exception:
                    pass  # no dejar que el callback rompa el ciclo ReAct

            # ── Añadir al historial de conversación ───────────────────────────
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({
                "role": "user",
                "content": OBSERVATION_TEMPLATE.format(observation=step.observation),
            })

        total_usage["total_latency_s"] = round(time.perf_counter() - t_start, 3)

        # ── Añadir fuentes consultadas al final answer ────────────────────────
        if answer and tools_used:
            unique_tools = list(dict.fromkeys(tools_used))
            answer = answer + "\n\n**Fuentes consultadas:** " + ", ".join(unique_tools)

        # ── Síntesis forzada si no hubo Final Answer ───────────────────────────
        if not answer:
            forced_final = True
            obs_summary = "\n\n".join(
                f"[{s.action}]: {s.observation[:500]}" for s in steps if s.observation
            ) or "Sin observaciones disponibles."
            fallback_text, fallback_usage = chat(
                messages=[{"role": "user", "content": FORCED_FINAL_USER_TEMPLATE.format(
                    question=question, observations_summary=obs_summary,
                )}],
                system=FORCED_FINAL_SYSTEM,
                temperature=0.0,
                max_tokens=512,
            )
            answer = fallback_text
            total_usage["prompt_tokens"] += fallback_usage["prompt_tokens"]
            total_usage["completion_tokens"] += fallback_usage["completion_tokens"]
            total_usage["total_tokens"] += fallback_usage["total_tokens"]
            if tools_used:
                unique_tools = list(dict.fromkeys(tools_used))
                answer = answer + "\n\n**Fuentes consultadas:** " + ", ".join(unique_tools)

        any_valid_obs = any(
            s.observation and "error" not in s.observation.lower()
            for s in steps
        )

        return {
            "answer": answer,
            "steps": [_step_to_dict(s) for s in steps],
            "tools_used": tools_used,
            "iterations": len(steps),
            "forced_final": forced_final,
            "hallucination_risk": not any_valid_obs and not steps,
            "usage": total_usage,
            "architecture": "D_Toolformer",
        }


def _step_to_dict(step: ReActStep) -> dict:
    return {
        "thought": step.thought,
        "action": step.action,
        "action_input": step.action_input,
        "observation_preview": step.observation[:300] + "…" if len(step.observation) > 300 else step.observation,
        "latency_s": step.latency_s,
    }
