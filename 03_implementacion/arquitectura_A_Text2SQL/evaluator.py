"""
Interfaz de evaluación de la Arquitectura A para el benchmark.

Uso:
    python -m arquitectura_A_Text2SQL.evaluator --demo
    python -m arquitectura_A_Text2SQL.evaluator -q "¿Cuántos CVEs críticos hay de 2024?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from arquitectura_A_Text2SQL.chain import ArchitectureAChain

logger = logging.getLogger(__name__)

# Preguntas de demo por nivel de dificultad (benchmark Capítulo 4)
_DEMO_QUESTIONS = [
    # Nivel 1 — Recuperación simple
    "¿Cuál es el score CVSS v3 de CVE-2021-44228?",
    "¿Qué descripción tiene la vulnerabilidad CVE-2023-23397?",
    # Nivel 2 — Consulta con filtros y agregación
    "¿Cuántas vulnerabilidades críticas hay publicadas en 2024?",
    "Lista las 5 vulnerabilidades con mayor CVSS v3 score de 2023",
    "¿Cuántos CVEs están asociados a la debilidad CWE-79?",
    # Nivel 3 — Razonamiento multi-paso (join entre tablas)
    "¿Qué productos están afectados por CVE-2021-44228 y cuál es su severidad?",
]


def evaluate_single(question: str, chain: ArchitectureAChain | None = None) -> dict:
    chain = chain or ArchitectureAChain()
    result = chain.run(question)
    result["question"] = question
    return result


def print_result(result: dict) -> None:
    print("\n" + "─" * 60)
    print(f"❓ Pregunta: {result['question']}")
    print(f"🗄️  SQL generado:\n   {result.get('sql', 'N/A')}")
    print(f"📋 Filas devueltas: {result['sql_rows']}")
    if result.get("correction_attempts"):
        print(f"🔧 Autocorrecciones: {result['correction_attempts']}")
    if result.get("sql_error"):
        print(f"⚠️  Error SQL: {result['sql_error']}")
    print(f"\n💬 Respuesta:\n{result['answer']}")
    u = result["usage"]
    total_latency = (
        u["schema_linking_latency_s"]
        + u["sql_generation_latency_s"]
        + u["self_correction_latency_s"]
        + u["synthesis_latency_s"]
    )
    print(
        f"\n📊 Métricas: tokens={u['total_tokens']} | "
        f"latencia_total={total_latency:.2f}s"
    )
    print(f"⚠️  Riesgo alucinación: {'SÍ' if result['hallucination_risk'] else 'NO'}")
    print("─" * 60)


def run_demo() -> None:
    chain = ArchitectureAChain()
    print("\n🚀 Demo Arquitectura A — LLM + Text2SQL (DIN-SQL) sobre NVD/CVE\n")
    for q in _DEMO_QUESTIONS:
        result = evaluate_single(q, chain)
        print_result(result)
    print("\n✅ Demo completada.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluador Arquitectura A — Text2SQL")
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
