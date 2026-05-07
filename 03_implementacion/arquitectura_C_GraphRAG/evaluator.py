"""
Interfaz de evaluación de la Arquitectura C para el benchmark.

Uso:
    python -m arquitectura_C_GraphRAG.evaluator --demo
    python -m arquitectura_C_GraphRAG.evaluator -q "¿Qué técnicas usa el grupo APT29?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from arquitectura_C_GraphRAG.chain import ArchitectureCChain

logger = logging.getLogger(__name__)

_DEMO_QUESTIONS = [
    # Nivel 1 — Recuperación simple sobre ATT&CK
    "¿Qué descripción tiene la técnica T1059?",
    "¿En qué táctica se encuadra la técnica T1078 (Valid Accounts)?",
    # Nivel 2 — Cruce de entidades
    "¿Qué técnicas MITRE ATT&CK utiliza el grupo APT29?",
    "¿Qué mitigaciones existen para la técnica T1566 (Phishing)?",
    # Nivel 3 — Razonamiento multi-salto
    "¿Qué grupos APT utilizan técnicas de la táctica Lateral Movement "
    "y qué plataformas afectan?",
]


def evaluate_single(question: str, chain: ArchitectureCChain | None = None) -> dict:
    chain = chain or ArchitectureCChain()
    result = chain.run(question)
    result["question"] = question
    return result


def print_result(result: dict) -> None:
    print("\n" + "─" * 60)
    print(f"❓ Pregunta: {result['question']}")
    print(f"🕸  Nodos semilla: {result['seed_nodes']}")
    print(f"📊 Subgrafo: {result['subgraph_nodes']} nodos, {result['subgraph_edges']} aristas")
    cov = result.get("coverage", {})
    print(
        f"✅ Cobertura suficiente: {cov.get('sufficient')} "
        f"(confianza: {cov.get('confidence', 0):.0%})"
    )
    if cov.get("missing_entities"):
        print(f"⚠️  Entidades no encontradas: {cov['missing_entities']}")
    print(f"\n💬 Respuesta:\n{result['answer']}")
    u = result["usage"]
    total_lat = u["retrieval_latency_s"] + u["coverage_latency_s"] + u["synthesis_latency_s"]
    print(f"\n📊 Métricas: tokens={u['total_tokens']} | latencia_total={total_lat:.2f}s")
    print(f"⚠️  Riesgo alucinación: {'SÍ' if result['hallucination_risk'] else 'NO'}")
    print("─" * 60)


def run_demo() -> None:
    chain = ArchitectureCChain()
    print("\n🚀 Demo Arquitectura C — LLM + GraphRAG sobre MITRE ATT&CK\n")
    for q in _DEMO_QUESTIONS:
        result = evaluate_single(q, chain)
        print_result(result)
    print("\n✅ Demo completada.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluador Arquitectura C — GraphRAG")
    parser.add_argument("--question", "-q", type=str, help="Pregunta a evaluar")
    parser.add_argument("--demo", action="store_true", help="Ejecutar preguntas de demostración")
    parser.add_argument("--json", action="store_true", help="Salida en formato JSON")
    parser.add_argument("--hops", type=int, default=2, help="Profundidad de expansión del grafo")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.question:
        chain = ArchitectureCChain(hops=args.hops)
        result = evaluate_single(args.question, chain)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            print_result(result)
    else:
        parser.print_help()
        sys.exit(1)
