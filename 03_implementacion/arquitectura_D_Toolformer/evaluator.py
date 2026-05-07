"""
Interfaz de evaluación de la Arquitectura D para el benchmark.

Uso:
    python -m arquitectura_D_Toolformer.evaluator --demo
    python -m arquitectura_D_Toolformer.evaluator -q "¿Qué técnicas usa APT29 y tienen CVEs críticos asociados?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from arquitectura_D_Toolformer.chain import ArchitectureDChain

logger = logging.getLogger(__name__)

# Preguntas de demo diseñadas para activar distintas herramientas
# y combinaciones de ellas
_DEMO_QUESTIONS = [
    # Nivel 1 — herramienta única obvia
    "¿Cuál es el CVSS score de CVE-2021-44228?",
    "¿Qué descripción tiene la técnica MITRE T1059?",
    # Nivel 2 — herramienta única con razonamiento
    "¿Cuántos CVEs con severidad CRITICAL se publicaron en 2024 relacionados con Microsoft?",
    "¿Qué grupos APT utilizan la técnica T1566 (Phishing) según MITRE ATT&CK?",
    # Nivel 3 — cruce de herramientas (multi-step)
    "¿Qué técnicas MITRE ATT&CK utiliza APT28 y cuántos CVEs críticos existen "
    "relacionados con las plataformas que atacan?",
]


def evaluate_single(question: str, chain: ArchitectureDChain | None = None) -> dict:
    chain = chain or ArchitectureDChain()
    result = chain.run(question)
    result["question"] = question
    return result


def print_result(result: dict) -> None:
    print("\n" + "─" * 60)
    print(f"❓ Pregunta: {result['question']}")
    print(f"🔁 Iteraciones ReAct: {result['iterations']}")
    print(f"🔧 Herramientas usadas: {result['tools_used']}")
    if result.get("forced_final"):
        print("⚠️  Final Answer forzada por límite de iteraciones")

    if result.get("steps"):
        print("\n📋 Trazado ReAct:")
        for i, step in enumerate(result["steps"], 1):
            print(f"  [{i}] Action: {step['action']}({json.dumps(step['action_input'], ensure_ascii=False)})")
            print(f"       Obs: {step['observation_preview']}")

    print(f"\n💬 Respuesta Final:\n{result['answer']}")
    u = result["usage"]
    print(
        f"\n📊 Métricas: tokens={u['total_tokens']} | "
        f"latencia_total={u['total_latency_s']}s"
    )
    print(f"⚠️  Riesgo alucinación: {'SÍ' if result['hallucination_risk'] else 'NO'}")
    print("─" * 60)


def run_demo() -> None:
    chain = ArchitectureDChain()
    print("\n🚀 Demo Arquitectura D — Toolformer / ReAct (selección autónoma de herramientas)\n")
    for q in _DEMO_QUESTIONS:
        result = evaluate_single(q, chain)
        print_result(result)
    print("\n✅ Demo completada.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluador Arquitectura D — Toolformer/ReAct")
    parser.add_argument("--question", "-q", type=str, help="Pregunta a evaluar")
    parser.add_argument("--demo", action="store_true", help="Ejecutar preguntas de demostración")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.question:
        result = evaluate_single(args.question)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print_result(result)
    else:
        parser.print_help()
        sys.exit(1)
