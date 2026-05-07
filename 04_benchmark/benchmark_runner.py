"""
Runner principal del benchmark del TFM.

Ejecuta las 4 arquitecturas contra las 50 preguntas de questions.json,
calcula las 5 métricas por pregunta y guarda los resultados en:
  - resultados/results_raw.json      ← respuestas completas
  - resultados/results_metrics.csv   ← métricas por pregunta y arquitectura
  - resultados/results_summary.json  ← métricas agregadas por arquitectura

Uso:
    cd 03_implementacion
    python ../04_benchmark/benchmark_runner.py              # todas las arquitecturas
    python ../04_benchmark/benchmark_runner.py --arch A    # solo Arquitectura A
    python ../04_benchmark/benchmark_runner.py --level 1   # solo preguntas nivel 1
    python ../04_benchmark/benchmark_runner.py --dry-run   # sin llamadas a LLM

El runner es robusto ante fallos: si una arquitectura falla en una pregunta,
registra el error y continúa con la siguiente. Los resultados parciales
se guardan tras cada pregunta para evitar perder trabajo ante un corte.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Añadir directorios al path para que los imports funcionen
IMPL_DIR = Path(__file__).parent.parent / "03_implementacion"
BENCHMARK_DIR_PATH = Path(__file__).parent
sys.path.insert(0, str(IMPL_DIR))
sys.path.insert(0, str(BENCHMARK_DIR_PATH))

from metrics import compute_all, aggregate  # noqa: E402

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "resultados"
QUESTIONS_PATH = BENCHMARK_DIR / "questions.json"

_ARCHITECTURES = {
    "A": "arquitectura_A_Text2SQL.chain.ArchitectureAChain",
    "B": "arquitectura_B_API.chain.ArchitectureBChain",
    "C": "arquitectura_C_GraphRAG.chain.ArchitectureCChain",
    "D": "arquitectura_D_Toolformer.chain.ArchitectureDChain",
}


def load_questions(level: int | None = None, domain: str | None = None) -> list[dict]:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if level:
        questions = [q for q in questions if q["level"] == level]
    if domain:
        questions = [q for q in questions if q["domain"] == domain]
    return questions


def _import_chain(arch_key: str):
    """Importa dinámicamente el chain de una arquitectura."""
    module_path, class_name = _ARCHITECTURES[arch_key].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_architecture(
    arch_key: str,
    questions: list[dict],
    dry_run: bool = False,
) -> list[dict]:
    """
    Ejecuta una arquitectura contra todas las preguntas.
    Devuelve lista de dicts con métricas por pregunta.
    """
    logger.info("=== Arquitectura %s ===", arch_key)
    ChainClass = _import_chain(arch_key)
    chain = ChainClass()

    metric_rows = []
    raw_results = []

    for i, question in enumerate(questions, 1):
        qid = question["id"]
        logger.info("[%d/%d] %s: %s", i, len(questions), qid, question["question"][:60])

        if dry_run:
            result = _fake_result(arch_key)
        else:
            try:
                result = chain.run(question["question"])
            except Exception as e:
                logger.error("Error en %s / %s: %s", arch_key, qid, e)
                result = _error_result(arch_key, str(e))

        metrics = compute_all(question, result)
        metric_rows.append(metrics)
        raw_results.append({"question_id": qid, "architecture": arch_key, "result": result})

        # Guardar resultado parcial tras cada pregunta
        _save_partial(arch_key, qid, result, metrics)

        logger.info(
            "  ACC=%.2f | HALL=%.2f | TRACE=%.2f | ERR=%.2f | LAT=%.1fs",
            metrics["accuracy"], metrics["hallucination_risk"],
            metrics["traceability"], metrics["error_handling"], metrics["latency_s"],
        )

    return metric_rows, raw_results


def _fake_result(arch_key: str) -> dict:
    """Resultado simulado para dry-run (sin llamadas a LLM)."""
    return {
        "answer": f"[DRY-RUN] Respuesta simulada de Arquitectura {arch_key}.",
        "hallucination_risk": False,
        "api_source": "N/A",
        "sql": None,
        "seed_nodes": [],
        "usage": {"total_tokens": 0, "latency_s": 0.01},
        "architecture": f"{arch_key}_TEST",
    }


def _error_result(arch_key: str, error_msg: str) -> dict:
    return {
        "answer": f"Error en la ejecución: {error_msg}",
        "hallucination_risk": True,
        "api_error": error_msg,
        "usage": {"total_tokens": 0},
        "architecture": arch_key,
    }


def _save_partial(arch_key: str, qid: str, result: dict, metrics: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"partial_{arch_key}_{qid}.json"
    path.write_text(
        json.dumps({"result": result, "metrics": metrics}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def save_results(
    all_metrics: list[dict],
    all_raw: list[dict],
    run_id: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON completo con respuestas
    raw_path = RESULTS_DIR / f"results_raw_{run_id}.json"
    raw_path.write_text(
        json.dumps(all_raw, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # CSV con métricas por pregunta
    import csv
    csv_path = RESULTS_DIR / f"results_metrics_{run_id}.csv"
    if all_metrics:
        fieldnames = list(all_metrics[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)

    # JSON con métricas agregadas por arquitectura
    summary: dict[str, dict] = {}
    for arch_key in _ARCHITECTURES:
        arch_metrics = [m for m in all_metrics if m["architecture"].startswith(arch_key)]
        if arch_metrics:
            summary[arch_key] = aggregate(arch_metrics)

    summary_path = RESULTS_DIR / f"results_summary_{run_id}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Copiar como "latest" para el report_generator
    for src, dst_name in [
        (raw_path, "results_raw.json"),
        (csv_path, "results_metrics.csv"),
        (summary_path, "results_summary.json"),
    ]:
        dst = RESULTS_DIR / dst_name
        dst.write_bytes(src.read_bytes())

    logger.info("Resultados guardados en %s", RESULTS_DIR)
    logger.info("  Raw:     %s", raw_path.name)
    logger.info("  Métricas: %s", csv_path.name)
    logger.info("  Resumen: %s", summary_path.name)


def print_summary_table(all_metrics: list[dict], archs: list[str]) -> None:
    print("\n" + "=" * 70)
    print("RESUMEN BENCHMARK — TFM LLM Ciberseguridad")
    print("=" * 70)
    header = f"{'Arq.':<6} {'N':>4} {'Accuracy':>10} {'Halluci.':>10} {'Traceb.':>10} {'Errors':>10} {'Latency':>10}"
    print(header)
    print("-" * 70)

    for arch_key in archs:
        arch_metrics = [m for m in all_metrics if m["architecture"].startswith(arch_key)]
        if not arch_metrics:
            continue
        agg = aggregate(arch_metrics)
        print(
            f"{arch_key:<6} {agg['n']:>4} "
            f"{agg['accuracy_mean']:>10.3f} "
            f"{agg['hallucination_mean']:>10.3f} "
            f"{agg['traceability_mean']:>10.3f} "
            f"{agg['error_handling_mean']:>10.3f} "
            f"{agg['latency_mean_s']:>9.2f}s"
        )
    print("=" * 70 + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Runner del benchmark TFM LLM Ciberseguridad")
    parser.add_argument("--arch", choices=["A", "B", "C", "D"], help="Ejecutar solo esta arquitectura")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Filtrar por nivel de dificultad")
    parser.add_argument("--domain", choices=["NVD", "MITRE", "BOTH"], help="Filtrar por dominio")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin llamadas a LLM")
    parser.add_argument("--questions", type=int, help="Limitar a N preguntas (para pruebas)")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    archs = [args.arch] if args.arch else list(_ARCHITECTURES.keys())
    questions = load_questions(level=args.level, domain=args.domain)

    if args.questions:
        questions = questions[: args.questions]

    logger.info(
        "Benchmark iniciado: %d preguntas × %d arquitecturas (run_id=%s)",
        len(questions), len(archs), run_id,
    )

    all_metrics: list[dict] = []
    all_raw: list[dict] = []

    for arch_key in archs:
        try:
            metrics, raw = run_architecture(arch_key, questions, dry_run=args.dry_run)
            all_metrics.extend(metrics)
            all_raw.extend(raw)
        except Exception as e:
            logger.error("Arquitectura %s falló: %s\n%s", arch_key, e, traceback.format_exc())

    if all_metrics:
        save_results(all_metrics, all_raw, run_id)
        print_summary_table(all_metrics, archs)
    else:
        logger.warning("No se generaron métricas.")


if __name__ == "__main__":
    main()
