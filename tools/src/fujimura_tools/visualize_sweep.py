#!/usr/bin/env python3
"""visualize_sweep.py — sweep visualization for Fujimura & Hino (2019).

Reads `sweep_summary.csv` from results/latest (or `--results-dir`) and produces:
  (1) a forest plot of the final silence_rate / voice_volume per n_levels (the
      hierarchy-strength effect on the silence/voice levels), with paper anchors,
  (2) a heatmap of the mean climate_of_silence over the (η × network_beta) grid
      (the structural conditions for the global silence spiral).

Usage:
    fujimura-tools visualize-sweep
    fujimura-tools visualize-sweep --results-dir results/20260530_000000_sweep
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from socsim_tools.io import resolve_results_dir  # noqa: E402

COLOR_BG = "#FAFAF8"
COLOR_SILENCE = "#C0392B"
COLOR_VOICE = "#0F6E56"


def plot_level_forest(df: pd.DataFrame, out_path: str) -> None:
    g = df.groupby("n_levels").agg(
        silence_mean=("silence_rate", "mean"),
        silence_sd=("silence_rate", "std"),
        voice_mean=("voice_volume", "mean"),
        voice_sd=("voice_volume", "std"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    y = np.arange(len(g))
    ax.errorbar(g["silence_mean"], y - 0.1, xerr=g["silence_sd"].fillna(0), fmt="o",
                color=COLOR_SILENCE, capsize=3, label="silence rate")
    ax.errorbar(g["voice_mean"], y + 0.1, xerr=g["voice_sd"].fillna(0), fmt="s",
                color=COLOR_VOICE, capsize=3, label="voice volume")
    ax.axvline(0.38, color=COLOR_SILENCE, ls=":", lw=0.9, alpha=0.5)
    ax.axvline(0.45, color=COLOR_VOICE, ls=":", lw=0.9, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"L={int(v)}" for v in g["n_levels"]])
    ax.set_xlabel("share (mean ± SD across cells)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title("Hierarchy-strength effect on silence / voice\n(dotted: paper anchors .38 / .45)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_climate_heatmap(df: pd.DataFrame, out_path: str) -> None:
    pivot = df.pivot_table(index="eta", columns="network_beta",
                           values="climate_of_silence", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(COLOR_BG)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="magma",
                   vmin=0.0, vmax=max(0.5, float(np.nanmax(pivot.values))))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("network_beta (WS rewiring)")
    ax.set_ylabel("supervisor homogeneity η")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.35 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean climate of silence")
    ax.set_title("Climate of silence over (η × network_beta)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fujimura-tools visualize-sweep", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = str(resolve_results_dir(args.results_dir))
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(results_dir, "sweep_summary.csv"))
    p1 = os.path.join(output_dir, "sweep_level_forest.png")
    p2 = os.path.join(output_dir, "sweep_climate_heatmap.png")
    plot_level_forest(df, p1)
    plot_climate_heatmap(df, p2)
    print(f"level forest    → {p1}")
    print(f"climate heatmap → {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
