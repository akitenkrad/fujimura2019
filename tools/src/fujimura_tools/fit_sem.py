#!/usr/bin/env python3
"""fit_sem.py — estimate the ABM-induced SEM path coefficients β̃ (semopy).

Reads `agent_panel.csv` (long: seed, t, agent_id, psafety, fear, acquiescent,
voice, silence, motive), builds per-agent cross-sectional observations by
time-averaging each latent state and each behaviour (skipping the all-neutral
initial step t=0), then fits the paper's structural model

    fear        ~ psafety               (paper β = −.54)
    acquiescent ~ fear                  (paper β = +.86)
    voice       ~ fear                  (paper β = −.23)
    silence     ~ acquiescent           (paper β = +.52)

with semopy, reporting standardized β̃, their 95% CI (Wald), the SEM fit
indices (CFI / GFI / RMSEA / χ²), and the silence ⊥ voice correlation
(paper H5 r = .02; target |r| < .10). Falls back to per-path OLS (statsmodels-
free, numpy) when semopy is unavailable or fails on the data.

Writes `sem_fit.json` to the results directory.

Usage:
    fujimura-tools fit-sem --results-dir results/latest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

from socsim_tools.io import resolve_results_dir

# Paper anchors (design §5): path → (value, expected sign).
PAPER_PATHS = {
    "psafety->fear": -0.54,
    "fear->acquiescent": 0.86,
    "fear->voice": -0.23,
    "acquiescent->silence": 0.52,
}

# Mapping of our path keys to (outcome, predictor) in the agent-level frame.
PATH_VARS = {
    "psafety->fear": ("fear", "psafety"),
    "fear->acquiescent": ("acquiescent", "fear"),
    "fear->voice": ("voice", "fear"),
    "acquiescent->silence": ("silence", "acquiescent"),
}

SEM_SPEC = (
    "fear ~ psafety\n"
    "acquiescent ~ fear\n"
    "voice ~ fear\n"
    "silence ~ acquiescent\n"
)


def build_agent_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long panel to one row per (seed, agent_id): time-averaged
    latent states + behaviour frequencies (excluding the t=0 all-neutral row)."""
    df = panel[panel["t"] > 0].copy()
    grouped = (
        df.groupby(["seed", "agent_id"])
        .agg(
            psafety=("psafety", "mean"),
            fear=("fear", "mean"),
            acquiescent=("acquiescent", "mean"),
            voice=("voice", "mean"),
            silence=("silence", "mean"),
        )
        .reset_index()
    )
    return grouped


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        sd = out[c].std(ddof=0)
        if sd > 0:
            out[c] = (out[c] - out[c].mean()) / sd
        else:
            out[c] = 0.0
    return out


def ols_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Simple OLS slope of standardized y on standardized x; returns (beta, se)."""
    n = len(x)
    if n < 3 or x.std() == 0:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return float("nan"), float("nan")
    beta = float(((x - xm) * (y - ym)).sum() / sxx)
    resid = y - (ym + beta * (x - xm))
    dof = n - 2
    sigma2 = float((resid**2).sum() / dof) if dof > 0 else float("nan")
    se = math.sqrt(sigma2 / sxx) if sigma2 == sigma2 and sxx > 0 else float("nan")
    return beta, se


def fit_ols_paths(agent_std: pd.DataFrame) -> dict[str, dict]:
    """Per-path standardized OLS (always available)."""
    out: dict[str, dict] = {}
    for key, (outcome, predictor) in PATH_VARS.items():
        y = agent_std[outcome].to_numpy()
        x = agent_std[predictor].to_numpy()
        beta, se = ols_beta(y, x)
        ci = (beta - 1.96 * se, beta + 1.96 * se) if se == se else (float("nan"), float("nan"))
        out[key] = {"beta": beta, "se": se, "ci_low": ci[0], "ci_high": ci[1]}
    return out


def fit_semopy(agent_std: pd.DataFrame) -> tuple[dict[str, dict], dict[str, float]]:
    """semopy SEM fit. Returns (path estimates, fit indices). Raises on failure."""
    from semopy import Model, calc_stats

    m = Model(SEM_SPEC)
    m.fit(agent_std)
    inspect = m.inspect(std_est=True)
    # inspect columns: lval, op, rval, Estimate, Est. Std, Std. Err, z-value, p-value
    paths: dict[str, dict] = {}
    for key, (outcome, predictor) in PATH_VARS.items():
        row = inspect[
            (inspect["lval"] == outcome)
            & (inspect["op"] == "~")
            & (inspect["rval"] == predictor)
        ]
        if row.empty:
            paths[key] = {"beta": float("nan"), "se": float("nan"),
                          "ci_low": float("nan"), "ci_high": float("nan")}
            continue
        r = row.iloc[0]
        beta = float(r.get("Est. Std", r.get("Estimate", float("nan"))))
        se_raw = r.get("Std. Err", float("nan"))
        try:
            se = float(se_raw)
        except (TypeError, ValueError):
            se = float("nan")
        ci = (beta - 1.96 * se, beta + 1.96 * se) if se == se else (float("nan"), float("nan"))
        paths[key] = {"beta": beta, "se": se, "ci_low": ci[0], "ci_high": ci[1]}

    stats = calc_stats(m)
    if isinstance(stats, pd.DataFrame) and not stats.empty:
        srow = stats.iloc[0].to_dict()
    elif isinstance(stats, dict):
        srow = stats
    else:
        srow = {}

    def _get(*keys: str) -> float:
        for k in keys:
            for kk in (k, k.lower(), k.upper()):
                if kk in srow:
                    try:
                        return float(srow[kk])
                    except (TypeError, ValueError):
                        pass
        return float("nan")

    fit = {
        "chi2": _get("chi2"),
        "dof": _get("DoF"),
        "p": _get("chi2 p-value"),
        "cfi": _get("CFI"),
        "gfi": _get("GFI"),
        "rmsea": _get("RMSEA"),
    }
    return paths, fit


def corr_silence_voice(agent: pd.DataFrame) -> float:
    s = agent["silence"].to_numpy()
    v = agent["voice"].to_numpy()
    if s.std() == 0 or v.std() == 0 or len(s) < 2:
        return 0.0
    return float(np.corrcoef(s, v)[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fujimura-tools fit-sem", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = str(resolve_results_dir(args.results_dir))
    panel_path = os.path.join(results_dir, "agent_panel.csv")
    if not os.path.exists(panel_path):
        print(f"error: agent_panel.csv not found in {results_dir}", file=sys.stderr)
        return 1

    panel = pd.read_csv(panel_path)
    agent = build_agent_frame(panel)
    agent_std = standardize(agent, ["psafety", "fear", "acquiescent", "voice", "silence"])

    method = "semopy"
    fit: dict[str, float] = {}
    try:
        paths, fit = fit_semopy(agent_std)
        # If semopy returned all-NaN betas, fall back.
        if all(math.isnan(p["beta"]) for p in paths.values()):
            raise RuntimeError("semopy returned no usable path estimates")
    except Exception as exc:  # noqa: BLE001
        print(f"warning: semopy fit unavailable/failed ({exc}); using per-path OLS", file=sys.stderr)
        method = "ols"
        paths = fit_ols_paths(agent_std)
        fit = {"chi2": float("nan"), "dof": float("nan"), "p": float("nan"),
               "cfi": float("nan"), "gfi": float("nan"), "rmsea": float("nan")}

    csv_corr = corr_silence_voice(agent)

    # Anchor reconciliation: sign match + |Δ| vs paper.
    anchors = []
    for key, paper_val in PAPER_PATHS.items():
        est = paths[key]["beta"]
        sign_match = (est == est) and (math.copysign(1, est) == math.copysign(1, paper_val))
        anchors.append({
            "path": key,
            "paper_beta": paper_val,
            "abm_beta": est,
            "ci_low": paths[key]["ci_low"],
            "ci_high": paths[key]["ci_high"],
            "sign_match": bool(sign_match),
            "abs_diff": abs(est - paper_val) if est == est else float("nan"),
        })

    n_sign = sum(1 for a in anchors if a["sign_match"])
    result = {
        "method": method,
        "n_agents": int(len(agent)),
        "n_seeds": int(agent["seed"].nunique()),
        "paths": paths,
        "fit_indices": fit,
        "corr_silence_voice": csv_corr,
        "corr_h5_target": "|r| < 0.10",
        "corr_h5_pass": abs(csv_corr) < 0.10,
        "anchors": anchors,
        "sign_match_count": n_sign,
        "sign_match_total": len(anchors),
    }

    out_path = os.path.join(results_dir, "sem_fit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Pretty report.
    print("=" * 66)
    print(f"ABM-induced SEM fit  (method={method}, N={len(agent)} agents, {result['n_seeds']} seeds)")
    print("=" * 66)
    print(f"{'path':<22}{'paper β':>10}{'ABM β̃':>12}{'sign':>7}")
    print("-" * 66)
    for a in anchors:
        sign = "OK" if a["sign_match"] else "MISS"
        print(f"{a['path']:<22}{a['paper_beta']:>10.2f}{a['abm_beta']:>12.3f}{sign:>7}")
    print("-" * 66)
    print(f"sign-match: {n_sign}/{len(anchors)} paths")
    print(f"corr(silence, voice) = {csv_corr:+.3f}  (H5 target |r|<.10 → "
          f"{'PASS' if result['corr_h5_pass'] else 'FAIL'})")
    fi = fit
    print(f"fit: CFI={fi['cfi']:.3f}  GFI={fi['gfi']:.3f}  RMSEA={fi['rmsea']:.3f}  "
          f"χ²={fi['chi2']:.2f} (df={fi['dof']})")
    print("=" * 66)
    print(f"[fit-sem] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
