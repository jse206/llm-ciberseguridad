"""
Genera las 4 figuras del benchmark.
Lee results_summary.json y, si existe, results_metrics.csv.

Uso:
    cd 04_benchmark
    python generate_figures_sober.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

ARCH_LABELS = {
    "A": "A — Text2SQL",
    "B": "B — API Calls",
    "C": "C — GraphRAG",
    "D": "D — Toolformer",
}
ARCH_COLORS = {
    "A": "#2563eb",
    "B": "#059669",
    "C": "#d97706",
    "D": "#7c3aed",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
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
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#e2e8f0",
        "legend.facecolor":   "white",
        "grid.color":         "#e2e8f0",
        "grid.linewidth":     0.8,
        "savefig.facecolor":  "white",
        "savefig.dpi":        200,
    })


def _round_bars(ax: plt.Axes, rects, radius_frac: float = 0.3) -> None:
    """Sustituye los rectángulos de barra por versiones con esquinas redondeadas."""
    for rect in rects:
        rect.set_visible(False)
        x = rect.get_x()
        y = rect.get_y()
        w = rect.get_width()
        h = rect.get_height()
        if h <= 0:
            continue
        r = min(radius_frac * w, 0.38 * h)
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=rect.get_facecolor(),
            alpha=rect.get_alpha() if rect.get_alpha() is not None else 1.0,
            linewidth=0,
            zorder=rect.get_zorder(),
        ))


def load_summary() -> dict:
    path = RESULTS_DIR / "results_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv() -> pd.DataFrame | None:
    path = RESULTS_DIR / "results_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8")
    df["arch_key"] = df["architecture"].str[0]
    return df


# ── Fig 1: Barras agrupadas — métricas principales ────────────────────────────

def fig1_grouped_bars(summary: dict) -> None:
    metrics       = ["accuracy_mean", "traceability_mean", "error_handling_mean", "hallucination_mean"]
    metric_labels = ["Exactitud", "Trazabilidad", "Man. errores", "Alucinación"]
    ci_keys       = [m.replace("_mean", "_ci_95") for m in metrics]
    arch_keys     = ["A", "B", "C", "D"]
    x     = np.arange(len(metrics))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, ak in enumerate(arch_keys):
        s      = summary[ak]
        values = [s.get(m, 0) for m in metrics]
        errors = [
            (s.get(ck, [v, v])[1] - s.get(ck, [v, v])[0]) / 2
            for v, ck in zip(values, ci_keys)
        ]
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, values, width,
            label=ARCH_LABELS[ak],
            color=ARCH_COLORS[ak],
            alpha=0.88,
            yerr=errors, capsize=3,
            error_kw={"ecolor": "#64748b", "elinewidth": 0.9, "capthick": 0.9},
        )
        _round_bars(ax, rects)
        for bar, val in zip(rects, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(errors) + 0.03,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#334155",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Puntuación (0–1)")
    ax.set_title("Figura 5.1. Comparativa de métricas por arquitectura (n = 100)")
    ax.legend(loc="upper right")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = FIGURES_DIR / "fig1_barras_metricas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 2: Radar chart ────────────────────────────────────────────────────────

def fig2_radar(summary: dict) -> None:
    metrics   = ["accuracy_mean", "traceability_mean", "error_handling_mean"]
    labels    = ["Exactitud", "Trazabilidad", "Man. errores"]
    arch_keys = ["A", "B", "C", "D"]
    N         = len(metrics)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#cbd5e1")

    for ak in arch_keys:
        values = [summary[ak].get(m, 0) for m in metrics] + [summary[ak].get(metrics[0], 0)]
        color  = ARCH_COLORS[ak]
        ax.plot(angles, values, linewidth=2, color=color, label=ARCH_LABELS[ak])
        ax.fill(angles, values, alpha=0.09, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, color="#334155")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#94a3b8")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_title("Figura 5.2. Perfil de rendimiento por arquitectura", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15))
    fig.tight_layout()

    out = FIGURES_DIR / "fig2_radar_arquitecturas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 3: Latencia por nivel de dificultad ───────────────────────────────────

def fig3_latency_by_level(df: pd.DataFrame | None, summary: dict) -> None:
    arch_keys    = ["A", "B", "C", "D"]
    levels       = [1, 2, 3]
    level_labels = ["Nivel 1\n(Recuperación simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"]
    x     = np.arange(len(levels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ak in enumerate(arch_keys):
        means = []
        for level in levels:
            if df is not None:
                subset = df[(df["arch_key"] == ak) & (df["level"] == level)]
                means.append(subset["latency_s"].mean() if not subset.empty else 0)
            else:
                means.append(summary[ak]["latency_mean_s"])
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, means, width,
            label=ARCH_LABELS[ak],
            color=ARCH_COLORS[ak],
            alpha=0.88,
        )
        _round_bars(ax, rects)
        for bar, val in zip(rects, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.2,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#334155",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylabel("Latencia media (segundos)")
    ax.set_title("Figura 5.3. Latencia media por nivel de dificultad y arquitectura")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = FIGURES_DIR / "fig3_latencia_por_nivel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 4: Exactitud por nivel de dificultad ─────────────────────────────────

def fig4_accuracy_by_level(summary: dict) -> None:
    arch_keys    = ["A", "B", "C", "D"]
    levels       = ["1", "2", "3"]
    level_labels = ["Nivel 1\n(Recuperación simple)", "Nivel 2\n(Cruce de fuentes)", "Nivel 3\n(Multi-paso)"]
    x     = np.arange(len(levels))
    width = 0.19

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ak in enumerate(arch_keys):
        values = [summary[ak]["accuracy_by_level"].get(lv, 0) for lv in levels]
        offset = (i - len(arch_keys) / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, values, width,
            label=ARCH_LABELS[ak],
            color=ARCH_COLORS[ak],
            alpha=0.88,
        )
        _round_bars(ax, rects)
        for bar, val in zip(rects, values):
            if val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.012,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#334155",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Exactitud media (0–1)")
    ax.set_title("Figura 5.4. Exactitud por nivel de dificultad y arquitectura")
    ax.legend()
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = FIGURES_DIR / "fig4_accuracy_por_nivel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    df      = load_csv()
    if df is not None:
        print(f"CSV cargado: {len(df)} filas")
    else:
        print("CSV no encontrado — fig3 usará medias globales")

    fig1_grouped_bars(summary)
    fig2_radar(summary)
    fig3_latency_by_level(df, summary)
    fig4_accuracy_by_level(summary)

    print(f"\nFiguras guardadas en: {FIGURES_DIR}")
