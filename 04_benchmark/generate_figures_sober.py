"""
Genera las 4 figuras del benchmark con estética de barras horizontales limpia.
Lee results_summary.json.

Uso:
    cd 04_benchmark
    python generate_figures_sober.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).parent / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

ARCH_KEYS   = ["A", "B", "C", "D"]
ARCH_LABELS = ["A — Text2SQL", "B — API Calls", "C — GraphRAG", "D — Toolformer"]
ARCH_COLORS = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
RED         = "#ef4444"


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": True,
        "axes.edgecolor":     "#cbd5e1",
        "axes.linewidth":     0.8,
        "axes.labelsize":     10,
        "axes.labelcolor":    "#64748b",
        "xtick.labelsize":    9,
        "ytick.labelsize":    10,
        "xtick.color":        "#94a3b8",
        "ytick.color":        "#334155",
        "grid.color":         "#e2e8f0",
        "grid.linewidth":     0.8,
        "savefig.facecolor":  "white",
        "savefig.dpi":        180,
    })


def _round_hbars(ax: plt.Axes, bars, radius: float = 0.12) -> None:
    """Reemplaza barras horizontales por versiones con esquinas redondeadas."""
    for bar in bars:
        bar.set_visible(False)
        x = bar.get_x()
        y = bar.get_y()
        w = bar.get_width()
        h = bar.get_height()
        if w <= 0:
            continue
        r = min(radius * h, 0.35 * w)
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=bar.get_facecolor(),
            alpha=bar.get_alpha() if bar.get_alpha() is not None else 1.0,
            linewidth=0,
            zorder=bar.get_zorder(),
        ))


def load_summary() -> dict:
    return json.loads((RESULTS_DIR / "results_summary.json").read_text(encoding="utf-8"))


# ── Fig 1: Exactitud por arquitectura ─────────────────────────────────────────

def fig1_accuracy(summary: dict) -> None:
    values = [summary[k]["accuracy_mean"] for k in ARCH_KEYS]
    pct    = [v * 100 for v in values]

    # Invert so A is at top
    labels_inv = ARCH_LABELS[::-1]
    pct_inv    = pct[::-1]
    colors_inv = ARCH_COLORS[::-1]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(labels_inv, pct_inv, color=colors_inv, height=0.52, alpha=0.9)
    _round_hbars(ax, bars)

    for bar, val in zip(bars, pct_inv):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center", ha="left", fontsize=9, color="#334155",
        )

    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", left=False, pad=6)
    fig.tight_layout(pad=1.4)

    out = FIGURES_DIR / "fig1_barras_metricas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 2: Radar — Perfil de métricas ─────────────────────────────────────────

def fig2_radar(summary: dict) -> None:
    lat_max = max(summary[k]["latency_mean_s"] for k in ARCH_KEYS)

    def get_vals(k: str) -> list[float]:
        s = summary[k]
        speed = 1 - s["latency_mean_s"] / lat_max
        return [
            s.get("accuracy_mean", 0),
            s.get("traceability_mean", 0),
            1 - s.get("hallucination_mean", 0),
            s.get("error_handling_mean", 0),
            speed,
        ]

    labels = ["Exactitud", "Trazabilidad", "Sin alucinación", "Err. handling", "Velocidad"]
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.spines["polar"].set_color("#cbd5e1")
    ax.spines["polar"].set_linewidth(0.8)

    for k, color, label in zip(ARCH_KEYS, ARCH_COLORS, ARCH_LABELS):
        vals = get_vals(k) + [get_vals(k)[0]]
        ax.plot(angles, vals, linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color="#334155")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5, color="#94a3b8")
    ax.yaxis.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.xaxis.grid(True, color="#e2e8f0", linewidth=0.7)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
        framealpha=0.95, edgecolor="#e2e8f0", fontsize=9,
    )
    fig.tight_layout(pad=1.4)

    out = FIGURES_DIR / "fig2_radar_arquitecturas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 3: Latencia media por arquitectura ────────────────────────────────────

def fig3_latency(summary: dict) -> None:
    values     = [summary[k]["latency_mean_s"] for k in ARCH_KEYS]
    labels_inv = ARCH_LABELS[::-1]
    vals_inv   = values[::-1]
    colors_inv = ARCH_COLORS[::-1]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(labels_inv, vals_inv, color=colors_inv, height=0.52, alpha=0.9)
    _round_hbars(ax, bars)

    for bar, val in zip(bars, vals_inv):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center", ha="left", fontsize=9, color="#334155",
        )

    ax.set_xlim(0, max(values) * 1.25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}s"))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", left=False, pad=6)
    fig.tight_layout(pad=1.4)

    out = FIGURES_DIR / "fig3_latencia_por_nivel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Fig 4: Tasa de alucinación por arquitectura ───────────────────────────────

def fig4_hallucination(summary: dict) -> None:
    values     = [summary[k]["hallucination_mean"] for k in ARCH_KEYS]
    pct        = [v * 100 for v in values]
    labels_inv = ARCH_LABELS[::-1]
    pct_inv    = pct[::-1]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(labels_inv, pct_inv, color=RED, height=0.52, alpha=0.88)
    _round_hbars(ax, bars)

    for bar, val in zip(bars, pct_inv):
        label_x = bar.get_width() + 1.5 if val > 0 else 1.5
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center", ha="left", fontsize=9, color="#334155",
        )

    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", left=False, pad=6)
    fig.tight_layout(pad=1.4)

    out = FIGURES_DIR / "fig4_accuracy_por_nivel.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()

    fig1_accuracy(summary)
    fig2_radar(summary)
    fig3_latency(summary)
    fig4_hallucination(summary)

    print(f"\nFiguras guardadas en: {FIGURES_DIR}")
