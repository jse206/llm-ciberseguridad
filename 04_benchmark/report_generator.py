"""
Generador de tablas y gráficas para el Capítulo 5 del TFM.

Lee los resultados del benchmark (results_metrics.csv y results_summary.json)
y produce:
  - Tabla comparativa general (todas las arquitecturas × todas las métricas)
  - Tabla por nivel de dificultad (nivel 1/2/3)
  - Gráfica de barras agrupadas (métricas por arquitectura)
  - Radar chart (perfil de cada arquitectura)
  - Gráfica de latencia con distribución por nivel

Todos los archivos se guardan en resultados/figuras/ con nombres descriptivos
listos para incluir en la memoria del TFM.

Uso:
    cd 04_benchmark
    python report_generator.py
    python report_generator.py --csv resultados/results_metrics.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sin GUI para entorno de servidor

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

_ARCH_LABELS = {
    "A": "A — Text2SQL",
    "B": "B — API Calls",
    "C": "C — GraphRAG",
    "D": "D — Toolformer",
}
_ARCH_COLORS = {
    "A": "#2196F3",   # azul
    "B": "#4CAF50",   # verde
    "C": "#FF9800",   # naranja
    "D": "#9C27B0",   # morado
}
_METRIC_LABELS = {
    "accuracy_mean": "Exactitud (Accuracy)",
    "traceability_mean": "Trazabilidad",
    "error_handling_mean": "Manejo de errores",
    "hallucination_mean": "Riesgo alucinación",
    "latency_mean_s": "Latencia media (s)",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["arch_key"] = df["architecture"].str[0]
    return df


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _arch_key(arch_str: str) -> str:
    return arch_str[0] if arch_str else "?"


# ── Tabla 1: Comparativa general ─────────────────────────────────────────────

def table_general(summary: dict) -> pd.DataFrame:
    """Tabla principal: métricas agregadas con IC 95% por arquitectura."""
    rows = []
    for arch_key in ["A", "B", "C", "D"]:
        if arch_key not in summary:
            continue
        s = summary[arch_key]

        def _fmt_ci(mean_key: str, ci_key: str) -> str:
            mean = s.get(mean_key, 0)
            ci = s.get(ci_key, [mean, mean])
            half = (ci[1] - ci[0]) / 2
            return f"{mean:.3f} ±{half:.3f}"

        rows.append({
            "Arquitectura": _ARCH_LABELS.get(arch_key, arch_key),
            "N": s["n"],
            "Exactitud (IC95%)": _fmt_ci("accuracy_mean", "accuracy_ci_95"),
            "Alucinación (IC95%)": _fmt_ci("hallucination_mean", "hallucination_ci_95"),
            "Trazabilidad (IC95%)": _fmt_ci("traceability_mean", "traceability_ci_95"),
            "Man. Errores (IC95%)": _fmt_ci("error_handling_mean", "error_handling_ci_95"),
            "Latencia (s)": f"{s['latency_mean_s']:.2f}",
            "Tokens (media)": f"{s.get('total_tokens_mean', 0):.0f}",
        })
    df = pd.DataFrame(rows)
    return df


def table_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de exactitud desglosada por nivel de dificultad y arquitectura."""
    rows = []
    for arch_key in ["A", "B", "C", "D"]:
        arch_df = df[df["arch_key"] == arch_key]
        if arch_df.empty:
            continue
        row = {"Arquitectura": _ARCH_LABELS.get(arch_key, arch_key)}
        for level in [1, 2, 3]:
            lvl_df = arch_df[arch_df["level"] == level]
            row[f"Nivel {level} (N={len(lvl_df)})"] = (
                f"{lvl_df['accuracy'].mean():.3f}" if not lvl_df.empty else "—"
            )
        row["Global"] = f"{arch_df['accuracy'].mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ── Gráfica 1: Barras agrupadas por métrica ───────────────────────────────────

def plot_grouped_bars(summary: dict, out_dir: Path) -> Path:
    metrics = ["accuracy_mean", "traceability_mean", "error_handling_mean"]
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in summary]
    x = np.arange(len(metrics))
    width = 0.2

    ci_keys = [m.replace("_mean", "_ci_95") for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, arch_key in enumerate(arch_keys):
        values = [summary[arch_key].get(m, 0) for m in metrics]
        errors = []
        for m, ck in zip(metrics, ci_keys):
            ci = summary[arch_key].get(ck, [values[metrics.index(m)], values[metrics.index(m)]])
            errors.append((ci[1] - ci[0]) / 2)
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=_ARCH_COLORS.get(arch_key, "#999"),
            alpha=0.85,
            yerr=errors,
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in metrics], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Puntuación (0–1)", fontsize=11)
    ax.set_title(
        "Comparativa de arquitecturas LLM — Benchmark CVE/MITRE ATT&CK\n"
        "TFM: Evaluación de arquitecturas LLM en ciberseguridad",
        fontsize=12, pad=12
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "fig1_barras_metricas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 2: Radar chart ────────────────────────────────────────────────────

def plot_radar(summary: dict, out_dir: Path) -> Path:
    metrics = ["accuracy_mean", "traceability_mean", "error_handling_mean"]
    # Invertir alucinación para que "mayor = mejor" en todos los ejes
    metric_labels = ["Exactitud", "Trazabilidad", "Man. Errores"]
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in summary]

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for arch_key in arch_keys:
        values = [summary[arch_key].get(m, 0) for m in metrics]
        values += values[:1]
        ax.plot(
            angles, values, linewidth=2,
            color=_ARCH_COLORS.get(arch_key, "#999"),
            label=_ARCH_LABELS.get(arch_key, arch_key),
        )
        ax.fill(angles, values, alpha=0.08, color=_ARCH_COLORS.get(arch_key, "#999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Perfil de cada arquitectura (Radar Chart)\nTFM — LLM Ciberseguridad", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()

    out_path = out_dir / "fig2_radar_arquitecturas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 3: Latencia por nivel ─────────────────────────────────────────────

def plot_latency_by_level(df: pd.DataFrame, out_dir: Path) -> Path:
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in df["arch_key"].values]
    levels = [1, 2, 3]
    x = np.arange(len(levels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, arch_key in enumerate(arch_keys):
        means = []
        for level in levels:
            subset = df[(df["arch_key"] == arch_key) & (df["level"] == level)]
            means.append(subset["latency_s"].mean() if not subset.empty else 0)
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=_ARCH_COLORS.get(arch_key, "#999"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Nivel 1\n(Simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"], fontsize=11)
    ax.set_ylabel("Latencia media (segundos)", fontsize=11)
    ax.set_title("Latencia por nivel de dificultad y arquitectura", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "fig3_latencia_por_nivel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 4: Exactitud por nivel ────────────────────────────────────────────

def plot_accuracy_by_level(df: pd.DataFrame, out_dir: Path) -> Path:
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in df["arch_key"].values]
    levels = [1, 2, 3]
    x = np.arange(len(levels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, arch_key in enumerate(arch_keys):
        means = []
        for level in levels:
            subset = df[(df["arch_key"] == arch_key) & (df["level"] == level)]
            means.append(subset["accuracy"].mean() if not subset.empty else 0)
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=_ARCH_COLORS.get(arch_key, "#999"),
            alpha=0.85,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Nivel 1\n(Simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Exactitud media (0–1)", fontsize=11)
    ax.set_title("Exactitud (Accuracy) por nivel de dificultad y arquitectura", fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / "fig4_accuracy_por_nivel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Función principal ─────────────────────────────────────────────────────────

def generate_report(
    csv_path: Path = RESULTS_DIR / "results_metrics.csv",
    summary_path: Path = RESULTS_DIR / "results_summary.json",
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error("No se encuentra %s. Ejecuta primero benchmark_runner.py", csv_path)
        return
    if not summary_path.exists():
        logger.error("No se encuentra %s. Ejecuta primero benchmark_runner.py", summary_path)
        return

    df = load_data(csv_path)
    summary = load_summary(summary_path)

    logger.info("Generando tablas…")
    tbl_general = table_general(summary)
    tbl_level = table_by_level(df)

    # Guardar tablas como CSV para importar en Word/LaTeX
    tbl_general.to_csv(FIGURES_DIR / "tabla1_comparativa_general.csv", index=False, encoding="utf-8")
    tbl_level.to_csv(FIGURES_DIR / "tabla2_accuracy_por_nivel.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("TABLA 1 — Comparativa general de arquitecturas")
    print("=" * 70)
    print(tbl_general.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLA 2 — Exactitud por nivel de dificultad")
    print("=" * 70)
    print(tbl_level.to_string(index=False))

    logger.info("Generando gráficas…")
    plot_grouped_bars(summary, FIGURES_DIR)
    plot_radar(summary, FIGURES_DIR)
    plot_latency_by_level(df, FIGURES_DIR)
    plot_accuracy_by_level(df, FIGURES_DIR)

    print(f"\n[OK] Informe generado en: {FIGURES_DIR}")
    print("   Tablas: tabla1_comparativa_general.csv, tabla2_accuracy_por_nivel.csv")
    print("   Figuras: fig1_barras_metricas.png, fig2_radar_arquitecturas.png,")
    print("            fig3_latencia_por_nivel.png, fig4_accuracy_por_nivel.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generador de informe del benchmark TFM")
    parser.add_argument("--csv", type=Path, default=RESULTS_DIR / "results_metrics.csv")
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "results_summary.json")
    args = parser.parse_args()
    generate_report(args.csv, args.summary)
