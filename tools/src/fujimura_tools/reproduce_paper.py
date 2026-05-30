#!/usr/bin/env python3
"""reproduce_paper.py — Fig.1-style SEM path diagram + B1--B5 anchor reconciliation.

Ensures `sem_fit.json` exists (runs `fit_sem` on `agent_panel.csv` if not), then:
  (1) renders the Fig.1-style SEM path diagram annotated with the estimated β̃,
  (2) prints + writes a B1--B5 reconciliation table comparing the ABM-induced
      paths and corr(silence, voice) against the paper anchors (design §5).

Usage:
    fujimura-tools reproduce
    fujimura-tools reproduce --results-dir results/latest
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from socsim_tools.io import resolve_results_dir

from fujimura_tools import fit_sem, visualize

# Design §5 anchors B1--B5.
B_TABLE = [
    ("B1", "psafety->fear", -0.54, "符号一致 + 95%CI 重なり"),
    ("B2", "fear->acquiescent", 0.86, "符号一致 + 95%CI 重なり"),
    ("B3", "fear->voice", -0.23, "符号一致 + 95%CI 重なり"),
    ("B4", "acquiescent->silence", 0.52, "符号一致 + 95%CI 重なり"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fujimura-tools reproduce", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = str(resolve_results_dir(args.results_dir))
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    sem_path = os.path.join(results_dir, "sem_fit.json")
    if not os.path.exists(sem_path):
        print("sem_fit.json not found — running fit-sem first ...")
        rc = fit_sem.main(["--results-dir", results_dir])
        if rc != 0:
            return rc
    with open(sem_path, encoding="utf-8") as f:
        sem = json.load(f)

    # Fig.1-style path diagram.
    diagram = os.path.join(output_dir, "paper_fig1_path_diagram.png")
    visualize.plot_path_diagram(sem, diagram)

    # B1--B5 reconciliation.
    paths = sem.get("paths", {})
    print("=" * 74)
    print("Fujimura & Hino (2019) — 再現照合 (B1--B5)")
    print("=" * 74)
    print(f"{'#':<4}{'path':<22}{'paper β':>10}{'ABM β̃':>12}{'sign':>7}{'CI overlap?':>12}")
    print("-" * 74)
    report = {"anchors": [], "corr_silence_voice": sem.get("corr_silence_voice")}
    for tag, key, paper_val, _crit in B_TABLE:
        est = paths.get(key, {}).get("beta", float("nan"))
        ci_low = paths.get(key, {}).get("ci_low", float("nan"))
        ci_high = paths.get(key, {}).get("ci_high", float("nan"))
        sign_ok = (est == est) and ((est >= 0) == (paper_val >= 0))
        ci_overlap = (ci_low == ci_low) and (ci_low <= paper_val <= ci_high)
        print(f"{tag:<4}{key:<22}{paper_val:>10.2f}{est:>12.3f}"
              f"{'OK' if sign_ok else 'MISS':>7}{'yes' if ci_overlap else 'no':>12}")
        report["anchors"].append({
            "tag": tag, "path": key, "paper_beta": paper_val, "abm_beta": est,
            "sign_match": bool(sign_ok), "ci_overlap": bool(ci_overlap),
        })
    csv_corr = sem.get("corr_silence_voice", float("nan"))
    csv_pass = abs(csv_corr) < 0.10 if csv_corr == csv_corr else False
    print("-" * 74)
    print(f"B5  corr(silence,voice) = {csv_corr:+.3f}  (paper r=.02, target |r|<.10 → "
          f"{'PASS' if csv_pass else 'FAIL'})")
    fi = sem.get("fit_indices", {})
    print(f"    fit: CFI={fi.get('cfi', float('nan')):.3f}  GFI={fi.get('gfi', float('nan')):.3f}  "
          f"RMSEA={fi.get('rmsea', float('nan')):.3f}")
    n_sign = sum(1 for a in report["anchors"] if a["sign_match"])
    print("=" * 74)
    print(f"sign-match: {n_sign}/{len(report['anchors'])} paths | method: {sem.get('method')}")

    report["b5_pass"] = bool(csv_pass)
    report["sign_match_count"] = n_sign
    report["fit_indices"] = fi
    out_json = os.path.join(output_dir, "reproduction_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"path diagram → {diagram}")
    print(f"report       → {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
