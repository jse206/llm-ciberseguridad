"""
Generador de tablas y gráficas para el Capítulo 5.

Lee los resultados del benchmark (results_metrics.csv y results_summary.json)
y produce:
  - Tabla comparativa general (todas las arquitecturas × todas las métricas)
  - Tabla por nivel de dificultad (nivel 1/2/3)
  - Gráfica de barras agrupadas (métricas por arquitectura)
  - Radar chart (perfil de cada arquitectura)
  - Gráfica de latencia con distribución por nivel
  - Gráfica de exactitud por nivel

Todos los archivos se guardan en resultados/figuras/ con nombres descriptivos.

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

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

# ── Paleta y etiquetas ────────────────────────────────────────────────────────

_ARCH_LABELS = {
    "A": "A — Text2SQL",
    "B": "B — API Calls",
    "C": "C — GraphRAG",
    "D": "D — Toolformer",
}
_ARCH_COLORS = {
    "A": "#2563eb",
    "B": "#059669",
    "C": "#d97706",
    "D": "#7c3aed",
}
_METRIC_LABELS = {
    "accuracy_mean":       "Exactitud",
    "traceability_mean":   "Trazabilidad",
    "error_handling_mean": "Man. Errores",
    "hallucination_mean":  "Alucinación",
    "latency_mean_s":      "Latencia (s)",
}

# ── Estilo global ─────────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#cbd5e1",
        "axes.linewidth":     0.9,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.titlepad":      14,
        "axes.labelsize":     11,
        "axes.labelcolor":    "#334155",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.color":        "#64748b",
        "ytick.color":        "#64748b",
        "legend.fontsize":    10,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#e2e8f0",
        "legend.facecolor":   "white",
        "grid.color":         "#e2e8f0",
        "grid.linewidth":     0.8,
        "grid.linestyle":     "-",
        "savefig.facecolor":  "white",
        "savefig.dpi":        150,
    })


# ── Helper: barras con esquinas redondeadas ───────────────────────────────────

def _round_bars(ax: plt.Axes, rects, radius_frac: float = 0.3) -> None:
    """Sustituye los rectángulos de barras por versiones con esquinas redondeadas."""
    for rect in rects:
        rect.set_visible(False)
        x = rect.get_x()
        y = rect.get_y()
        w = rect.get_width()
        h = rect.get_height()
        if h <= 0:
            continue
        r = min(radius_frac * w, 0.38 * h)
        fancy = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=rect.get_facecolor(),
            alpha=rect.get_alpha() if rect.get_alpha() is not None else 1.0,
            linewidth=0,
            zorder=rect.get_zorder(),
        )
        ax.add_patch(fancy)


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df["arch_key"] = df["architecture"].str[0]
    return df


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _arch_key(arch_str: str) -> str:
    return arch_str[0] if arch_str else "?"


# ── Tablas ────────────────────────────────────────────────────────────────────

def table_general(summary: dict) -> pd.DataFrame:
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
            "Arquitectura":        _ARCH_LABELS.get(arch_key, arch_key),
            "N":                   s["n"],
            "Exactitud (IC95%)":   _fmt_ci("accuracy_mean",       "accuracy_ci_95"),
            "Alucinación (IC95%)": _fmt_ci("hallucination_mean",  "hallucination_ci_95"),
            "Trazabilidad (IC95%)":_fmt_ci("traceability_mean",   "traceability_ci_95"),
            "Man. Errores (IC95%)":_fmt_ci("error_handling_mean", "error_handling_ci_95"),
            "Latencia (s)":        f"{s['latency_mean_s']:.2f}",
            "Tokens (media)":      f"{s.get('total_tokens_mean', 0):.0f}",
        })
    return pd.DataFrame(rows)


def table_by_level(df: pd.DataFrame) -> pd.DataFrame:
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
    _apply_style()
    metrics   = ["accuracy_mean", "traceability_mean", "error_handling_mean"]
    ci_keys   = [m.replace("_mean", "_ci_95") for m in metrics]
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in summary]
    x = np.arange(len(metrics))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, arch_key in enumerate(arch_keys):
        values = [summary[arch_key].get(m, 0) for m in metrics]
        errors = [
            (summary[arch_key].get(ck, [v, v])[1] - summary[arch_key].get(ck, [v, v])[0]) / 2
            for v, ck in zip(values, ci_keys)
        ]
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        color = _ARCH_COLORS.get(arch_key, "#999")
        rects = ax.bar(
            x + offset, values, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=color, alpha=0.88,
            yerr=errors, capsize=4,
            error_kw={"ecolor": "#64748b", "elinewidth": 1, "capthick": 1},
        )
        _round_bars(ax, rects)
        for bar, val in zip(rects, values):
            if val > 0.04:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + max(errors) + 0.025,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, color="#334155",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Puntuación (0–1)")
    ax.set_title("Comparativa de métricas por arquitectura")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = out_dir / "fig1_barras_metricas.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 2: Radar chart ────────────────────────────────────────────────────

def plot_radar(summary: dict, out_dir: Path) -> Path:
    _apply_style()
    metrics       = ["accuracy_mean", "traceability_mean", "error_handling_mean"]
    metric_labels = ["Exactitud", "Trazabilidad", "Man. Errores"]
    arch_keys     = [k for k in ["A", "B", "C", "D"] if k in summary]

    N      = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#cbd5e1")

    for arch_key in arch_keys:
        values = [summary[arch_key].get(m, 0) for m in metrics] + [summary[arch_key].get(metrics[0], 0)]
        color  = _ARCH_COLORS.get(arch_key, "#999")
        ax.plot(angles, values, linewidth=2, color=color, label=_ARCH_LABELS.get(arch_key, arch_key))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11, color="#334155")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#94a3b8")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_title("Perfil comparativo de arquitecturas", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    fig.tight_layout()

    out_path = out_dir / "fig2_radar_arquitecturas.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 3: Latencia por nivel ─────────────────────────────────────────────

def plot_latency_by_level(df: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in df["arch_key"].values]
    levels    = [1, 2, 3]
    x         = np.arange(len(levels))
    width     = 0.19

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, arch_key in enumerate(arch_keys):
        means  = []
        for level in levels:
            subset = df[(df["arch_key"] == arch_key) & (df["level"] == level)]
            means.append(subset["latency_s"].mean() if not subset.empty else 0)
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, means, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=_ARCH_COLORS.get(arch_key, "#999"),
            alpha=0.88,
        )
        _round_bars(ax, rects)

    ax.set_xticks(x)
    ax.set_xticklabels(["Nivel 1\n(Simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"])
    ax.set_ylabel("Latencia media (segundos)")
    ax.set_title("Latencia por nivel de dificultad")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = out_dir / "fig3_latencia_por_nivel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Gráfica 4: Exactitud por nivel ────────────────────────────────────────────

def plot_accuracy_by_level(df: pd.DataFrame, out_dir: Path) -> Path:
    _apply_style()
    arch_keys = [k for k in ["A", "B", "C", "D"] if k in df["arch_key"].values]
    levels    = [1, 2, 3]
    x         = np.arange(len(levels))
    width     = 0.19

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, arch_key in enumerate(arch_keys):
        means  = []
        for level in levels:
            subset = df[(df["arch_key"] == arch_key) & (df["level"] == level)]
            means.append(subset["accuracy"].mean() if not subset.empty else 0)
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, means, width,
            label=_ARCH_LABELS.get(arch_key, arch_key),
            color=_ARCH_COLORS.get(arch_key, "#999"),
            alpha=0.88,
        )
        _round_bars(ax, rects)
        for bar, val in zip(rects, means):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.014,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#334155",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(["Nivel 1\n(Simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Exactitud media (0–1)")
    ax.set_title("Exactitud por nivel de dificultad")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = out_dir / "fig4_accuracy_por_nivel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Guardada: %s", out_path.name)
    return out_path


# ── Función principal ─────────────────────────────────────────────────────────

def generate_report(
    csv_path: Path     = RESULTS_DIR / "results_metrics.csv",
    summary_path: Path = RESULTS_DIR / "results_summary.json",
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        logger.error("No se encuentra %s", csv_path)
        return
    if not summary_path.exists():
        logger.error("No se encuentra %s", summary_path)
        return

    df      = load_data(csv_path)
    summary = load_summary(summary_path)

    tbl_general = table_general(summary)
    tbl_level   = table_by_level(df)

    tbl_general.to_csv(FIGURES_DIR / "tabla1_comparativa_general.csv", index=False, encoding="utf-8")
    tbl_level.to_csv(FIGURES_DIR / "tabla2_accuracy_por_nivel.csv",    index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("TABLA 1 — Comparativa general de arquitecturas")
    print("=" * 70)
    print(tbl_general.to_string(index=False))

    print("\n" + "=" * 70)
    print("TABLA 2 — Exactitud por nivel de dificultad")
    print("=" * 70)
    print(tbl_level.to_string(index=False))

    plot_grouped_bars(summary, FIGURES_DIR)
    plot_radar(summary, FIGURES_DIR)
    plot_latency_by_level(df, FIGURES_DIR)
    plot_accuracy_by_level(df, FIGURES_DIR)

    print(f"\n[OK] Figuras generadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generador de informe del benchmark")
    parser.add_argument("--csv",     type=Path, default=RESULTS_DIR / "results_metrics.csv")
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "results_summary.json")
    args = parser.parse_args()
    generate_report(args.csv, args.summary)
