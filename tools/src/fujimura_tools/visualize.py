#!/usr/bin/env python3
"""visualize.py — single-run visualization for Fujimura & Hino (2019).

Reads `metrics.csv` (and, if present, `sem_fit.json`) from results/latest (or
`--results-dir`) and produces:
  (1) a time-series of silence_rate / voice_volume / climate_of_silence,
  (2) the silence-motive mix (4-motive) trajectory,
  (3) an estimated SEM path diagram (ψ → fear → acquiescent → silence, fear → voice).

Usage:
    fujimura-tools visualize
    fujimura-tools visualize --results-dir results/20260530_000000
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from socsim_tools.io import resolve_results_dir  # noqa: E402

COLOR_BG = "#FAFAF8"
COLOR_SILENCE = "#C0392B"
COLOR_VOICE = "#0F6E56"
COLOR_CLIMATE = "#534AB7"
MOTIVE_COLORS = {
    "acquiescent": "#C0392B",
    "quiescent": "#E08E0B",
    "prosocial": "#0F6E56",
    "opportunistic": "#534AB7",
}


def _pooled_mean_by_t(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average each metric across seeds at each t."""
    cols = [c for c in metrics.columns if c not in ("seed", "t")]
    return metrics.groupby("t")[cols].mean().reset_index()


def plot_timeseries(metrics: pd.DataFrame, out_path: str) -> None:
    m = _pooled_mean_by_t(metrics)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.plot(m["t"], m["silence_rate"], color=COLOR_SILENCE, lw=2.0, label="silence rate")
    ax.plot(m["t"], m["voice_volume"], color=COLOR_VOICE, lw=2.0, label="voice volume")
    ax.plot(m["t"], m["climate_of_silence"], color=COLOR_CLIMATE, lw=1.6, ls="--",
            label="climate of silence")
    # Paper anchors (design B11/B12): silence ≈ 0.38, voice ≈ 0.45.
    ax.axhline(0.38, color=COLOR_SILENCE, ls=":", lw=0.9, alpha=0.5)
    ax.axhline(0.45, color=COLOR_VOICE, ls=":", lw=0.9, alpha=0.5)
    ax.set_xlabel("time t")
    ax.set_ylabel("share")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("Silence & voice trajectory\n(dotted: paper anchors silence≈.38 / voice≈.45)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_motive_mix(metrics: pd.DataFrame, out_path: str) -> None:
    m = _pooled_mean_by_t(metrics)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    for motive, color in MOTIVE_COLORS.items():
        col = f"motive_mix_{motive}"
        if col in m.columns:
            ax.plot(m["t"], m[col], color=color, lw=2.0, label=motive)
    ax.set_xlabel("time t")
    ax.set_ylabel("within-silent motive share")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("Silence-motive mix (Knoll 4 forms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_path_diagram(sem_fit: dict | None, out_path: str) -> None:
    """Fig.1-style SEM path diagram annotated with the estimated β̃."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.axis("off")

    nodes = {
        "psafety": (0.08, 0.5, "心理的安全\nψ"),
        "fear": (0.36, 0.5, "怖れ (Quiescent)\nf"),
        "acquiescent": (0.64, 0.68, "黙従 (Acquiescent)\na"),
        "silence": (0.92, 0.68, "沈黙\nsilence"),
        "voice": (0.64, 0.22, "発言\nvoice"),
    }
    for _k, (x, y, label) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.08, y - 0.08), 0.16, 0.16,
                                   fc="#FFFFFF", ec="#333333", lw=1.4, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=3)

    def beta_for(path: str) -> float:
        if not sem_fit:
            return float("nan")
        return sem_fit.get("paths", {}).get(path, {}).get("beta", float("nan"))

    edges = [
        ("psafety", "fear", "psafety->fear", -0.54),
        ("fear", "acquiescent", "fear->acquiescent", 0.86),
        ("acquiescent", "silence", "acquiescent->silence", 0.52),
        ("fear", "voice", "fear->voice", -0.23),
    ]
    for src, dst, key, paper in edges:
        x0, y0, _ = nodes[src]
        x1, y1, _ = nodes[dst]
        est = beta_for(key)
        color = "#C0392B" if (paper > 0) else "#0F6E56"
        ax.annotate("", xy=(x1 - 0.085, y1), xytext=(x0 + 0.085, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color), zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.04
        label = f"paper {paper:+.2f}\nABM {est:+.2f}" if est == est else f"paper {paper:+.2f}"
        ax.text(mx, my, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#FAFAF8", ec=color, lw=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fujimura & Hino (2019) SEM — estimated ABM path coefficients β̃")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fujimura-tools visualize", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = str(resolve_results_dir(args.results_dir))
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    metrics = pd.read_csv(os.path.join(results_dir, "metrics.csv"))
    sem_fit = None
    sem_path = os.path.join(results_dir, "sem_fit.json")
    if os.path.exists(sem_path):
        with open(sem_path, encoding="utf-8") as f:
            sem_fit = json.load(f)

    p1 = os.path.join(output_dir, "timeseries.png")
    p2 = os.path.join(output_dir, "motive_mix.png")
    p3 = os.path.join(output_dir, "path_diagram.png")
    plot_timeseries(metrics, p1)
    plot_motive_mix(metrics, p2)
    plot_path_diagram(sem_fit, p3)
    print(f"timeseries   → {p1}")
    print(f"motive mix   → {p2}")
    print(f"path diagram → {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
