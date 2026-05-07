"""
Pipeline principal de la Arquitectura B: LLM + API Calls en tiempo real.

Flujo (dos pasos):
  Pregunta → [Router LLM] → intent + params
           → [API Call]   → datos JSON en tiempo real
           → [Synthesizer LLM] → respuesta fundamentada

Referencia: Yao et al. (2023). ReAct. ICLR 2023.
               NIST (2024). NVD API v2.0.

Uso:
    from arquitectura_B_API.chain import ArchitectureBChain
    chain = ArchitectureBChain()
    result = chain.run("¿Cuál es el CVSS score de CVE-2021-44228?")
    print(result["answer"])
"""

from __future__ import annotations

import json
import logging
from typing import Any

from shared.llm_client import chat
from arquitectura_B_API.api_client import (
    get_cve_by_id,
    search_cves,
    get_cves_by_cpe,
    check_ip,
)
from arquitectura_B_API.prompts import (
    ROUTER_SYSTEM,
    ROUTER_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM,
    SYNTHESIZER_USER_TEMPLATE,
    ERROR_SYSTEM,
    ERROR_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.65

# Mapa intent → función de API
_API_DISPATCH: dict[str, Any] = {
    "get_cve_by_id": get_cve_by_id,
    "search_cves": search_cves,
    "get_cves_by_cpe": get_cves_by_cpe,
    "check_ip": check_ip,
}


class ArchitectureBChain:
    """
    Arquitectura B: LLM con llamadas a API en tiempo real.

    Atributos de resultado (para el benchmark):
      - answer       : respuesta en lenguaje natural
      - intent       : intención detectada por el router
      - api_source   : URL de la API consultada
      - api_error    : error de API si lo hubo
      - router_conf  : confianza del router (0-1)
      - usage        : dict con tokens y latencias de ambas llamadas LLM
      - hallucination_risk: True si el router devolvió intent "unknown"
    """

    def run(self, question: str) -> dict:
        """
        Ejecuta el pipeline completo y devuelve un dict con la respuesta
        y todas las métricas necesarias para el benchmark.
        """
        total_usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "router_latency_s": 0,
            "api_latency_s": 0,
            "synthesis_latency_s": 0,
        }

        # ── Paso 1: Router ─────────────────────────────────────────────────────
        router_text, router_usage = chat(
            messages=[{"role": "user", "content": ROUTER_USER_TEMPLATE.format(question=question)}],
            system=ROUTER_SYSTEM,
            temperature=0.0,
            max_tokens=256,
        )
        total_usage["prompt_tokens"] += router_usage["prompt_tokens"]
        total_usage["completion_tokens"] += router_usage["completion_tokens"]
        total_usage["total_tokens"] += router_usage["total_tokens"]
        total_usage["router_latency_s"] = router_usage["latency_s"]

        routing = self._parse_routing(router_text)
        intent = routing.get("intent", "unknown")
        params = routing.get("params", {})
        confidence = routing.get("confidence", 0.0)

        logger.info("Router: intent=%s, confidence=%.2f, params=%s", intent, confidence, params)

        # ── Gate de confianza ──────────────────────────────────────────────────
        if confidence < _CONFIDENCE_THRESHOLD or intent == "unknown":
            return self._low_confidence_result(question, intent, confidence, total_usage)

        # ── Paso 2: Llamada a la API ───────────────────────────────────────────
        api_result = self._call_api(intent, params)
        total_usage["api_latency_s"] = api_result.get("latency_s", 0)

        if api_result.get("error"):
            # Síntesis de error: informa al usuario sin inventar datos
            answer, synth_usage = chat(
                messages=[{
                    "role": "user",
                    "content": ERROR_USER_TEMPLATE.format(
                        question=question,
                        error=api_result["error"],
                        source=api_result.get("source", "N/A"),
                    ),
                }],
                system=ERROR_SYSTEM,
                temperature=0.0,
            )
        else:
            # ── Paso 3: Síntesis de la respuesta ──────────────────────────────
            api_data_str = json.dumps(api_result.get("data"), ensure_ascii=False, indent=2)
            if len(api_data_str) > 8000:
                api_data_str = api_data_str[:8000] + "\n[... truncado por longitud ...]"

            answer, synth_usage = chat(
                messages=[{
                    "role": "user",
                    "content": SYNTHESIZER_USER_TEMPLATE.format(
                        question=question,
                        api_data=api_data_str,
                        source=api_result.get("source", "N/A"),
                    ),
                }],
                system=SYNTHESIZER_SYSTEM,
                temperature=0.0,
            )

        total_usage["prompt_tokens"] += synth_usage["prompt_tokens"]
        total_usage["completion_tokens"] += synth_usage["completion_tokens"]
        total_usage["total_tokens"] += synth_usage["total_tokens"]
        total_usage["synthesis_latency_s"] = synth_usage["latency_s"]

        api_source = api_result.get("source")
        return {
            "answer": answer,
            "intent": intent,
            "api_source": api_source,
            "api_error": api_result.get("error"),
            "router_confidence": confidence,
            "usage": total_usage,
            "hallucination_risk": bool(api_result.get("error")),
            "architecture": "B_API",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _low_confidence_result(self, question: str, intent: str, confidence: float, total_usage: dict) -> dict:
        """Devuelve un resultado honesto cuando el router no está seguro del intent."""
        answer = (
            f"No fue posible determinar con suficiente confianza qué fuente de datos consultar "
            f"para esta pregunta (confianza de routing: {confidence:.0%}). "
            "La pregunta podría requerir información no disponible en las APIs del sistema "
            "(NVD, AbuseIPDB). Reformule la pregunta especificando un CVE concreto, "
            "una dirección IP o un producto afectado."
        )
        return {
            "answer": answer,
            "intent": intent,
            "api_source": None,
            "api_error": f"Confianza insuficiente ({confidence:.2f} < {_CONFIDENCE_THRESHOLD})",
            "router_confidence": confidence,
            "usage": total_usage,
            "hallucination_risk": False,
            "architecture": "B_API",
        }

    def _parse_routing(self, raw: str) -> dict:
        """Extrae el JSON del router tolerando texto extra alrededor."""
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Router JSON parse error: %s | raw: %s", e, raw[:200])
            return {"intent": "unknown", "params": {}, "confidence": 0.0}

    def _call_api(self, intent: str, params: dict) -> dict:
        """Despacha la llamada a la función de API correspondiente."""
        fn = _API_DISPATCH.get(intent)
        if fn is None:
            return {
                "data": None,
                "source": "N/A",
                "error": f"Intent no reconocido: '{intent}'",
                "latency_s": 0,
            }
        try:
            return fn(**params)
        except TypeError as e:
            # Parámetros incorrectos generados por el router
            logger.warning("API dispatch TypeError para intent=%s: %s", intent, e)
            return {
                "data": None,
                "source": "N/A",
                "error": f"Parámetros inválidos para {intent}: {e}",
                "latency_s": 0,
            }
