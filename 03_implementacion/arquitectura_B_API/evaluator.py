"""
Interfaz de evaluación de la Arquitectura B para el benchmark.

Uso:
    python -m arquitectura_B_API.evaluator --question "¿Cuál es el CVSS de CVE-2021-44228?"
    python -m arquitectura_B_API.evaluator --demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from arquitectura_B_API.chain import ArchitectureBChain

logger = logging.getLogger(__name__)

# Preguntas de demostración que cubren los tres niveles del benchmark
_DEMO_QUESTIONS = [
    # Nivel 1 — Recuperación simple
    "¿Cuál es el score CVSS v3 de la vulnerabilidad CVE-2021-44228?",
    "¿Qué severidad tiene CVE-2023-23397?",
    # Nivel 2 — Consulta con filtros
    "Lista las 5 vulnerabilidades CRÍTICAS más recientes relacionadas con Apache",
    "¿Qué CVEs están asociados a la debilidad CWE-79?",
    # Nivel 3 — Reputación IP
    "¿Es la dirección IP 192.168.1.1 conocida por actividad maliciosa?",
]


def evaluate_single(question: str, chain: ArchitectureBChain | None = None) -> dict:
    """
    Evalúa una sola pregunta y devuelve el resultado completo.
    Formato compatible con el runner del benchmark principal.
    """
    chain = chain or ArchitectureBChain()
    result = chain.run(question)
    result["question"] = question
    return result


def print_result(result: dict) -> None:
    """Imprime el resultado formateado para inspección manual."""
    print("\n" + "─" * 60)
    print(f"❓ Pregunta: {result['question']}")
    print(f"🔍 Intent detectado: {result['intent']} (confianza: {result['router_confidence']:.0%})")
    print(f"🌐 API consultada: {result['api_source']}")
    if result.get("api_error"):
        print(f"⚠️  Error de API: {result['api_error']}")
    print(f"\n💬 Respuesta:\n{result['answer']}")
    u = result["usage"]
    print(
        f"\n📊 Métricas: tokens={u['total_tokens']} | "
        f"router={u['router_latency_s']}s | "
        f"api={u['api_latency_s']}s | "
        f"síntesis={u['synthesis_latency_s']}s"
    )
    print(f"⚠️  Riesgo alucinación: {'SÍ' if result['hallucination_risk'] else 'NO'}")
    print("─" * 60)


def run_demo() -> None:
    """Ejecuta las preguntas de demostración y muestra los resultados."""
    chain = ArchitectureBChain()
    print("\n🚀 Demo Arquitectura B — LLM + API Calls en tiempo real\n")
    for q in _DEMO_QUESTIONS:
        result = evaluate_single(q, chain)
        print_result(result)

    print("\n✅ Demo completada.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluador Arquitectura B")
    parser.add_argument("--question", "-q", type=str, help="Pregunta a evaluar")
    parser.add_argument("--demo", action="store_true", help="Ejecutar preguntas de demostración")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.question:
        result = evaluate_single(args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
    else:
        parser.print_help()
        sys.exit(1)
