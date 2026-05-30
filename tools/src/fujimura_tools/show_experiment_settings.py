"""fujimura-tools show-experiment-settings — print a results directory's settings.

Reads `config.json` (run / cultural-compare) or `sweep_config.json` (sweep) plus,
if present, `llm_meta.json`, and renders the run parameters. `results/latest` is
resolved automatically.

Usage:
    fujimura-tools show-experiment-settings
    fujimura-tools show-experiment-settings --results-dir results/20260530_000000
    fujimura-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import resolve_results_dir

FIELD_LABELS = {
    "decision_mode": "決定モード       ",
    "locale": "ロケール         ",
    "hierarchy_strength": "階層強度 L       ",
    "n_teams": "チーム数         ",
    "team_size": "チームサイズ     ",
    "n_employees": "従業員数 N       ",
    "eta": "上司均質性 η     ",
    "network_k": "平均次数 k       ",
    "network_beta": "再配線率 β       ",
    "prompt_variant": "プロンプト変種   ",
    "psafety_mean": "心理的安全平均   ",
    "fear_mean": "怖れ平均         ",
    "acquiescent_mean": "黙従平均         ",
    "ivt_mean": "IVT 強度平均     ",
    "p_retaliate": "報復確率         ",
    "t_max": "最大 tick T      ",
    "runs": "試行数 runs      ",
    "seed": "シード           ",
}


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json または sweep_config.json"
    )


def _load_llm_meta(results_dir: Path) -> dict | None:
    p = results_dir / "llm_meta.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def render_run_config(cfg: dict, source: Path) -> str:
    lines = ["=" * 70, "実行設定 (run)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    for key, label in FIELD_LABELS.items():
        if key in cfg:
            lines.append(f"{label}: {cfg[key]}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    def join(key: str) -> str:
        return ", ".join(str(v) for v in cfg.get(key, []))

    lines = ["=" * 70, "実行設定 (sweep)", "=" * 70, f"設定ファイル: {source}", "-" * 70]
    lines.append(f"決定モード       : {cfg.get('decision_mode', '-')}")
    lines.append(f"ロケール         : {cfg.get('locale', '-')}")
    lines.append(f"階層強度 L       : {join('n_levels_values')}")
    lines.append(f"上司均質性 η     : {join('eta_values')}")
    lines.append(f"再配線率 β       : {join('network_beta_values')}")
    lines.append(f"チーム数         : {cfg.get('n_teams', '-')}")
    lines.append(f"チームサイズ     : {cfg.get('team_size', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"最大 tick T      : {cfg.get('t_max', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_meta(meta: dict) -> str:
    lines = ["-" * 70, "LLM メタ情報", "-" * 70]
    for k in ("decision_mode", "model", "endpoint", "temperature", "seed",
              "calls", "cache_hits", "cache_hit_rate", "parse_failures", "parse_fail_rate"):
        if k in meta:
            lines.append(f"  {k:<16}: {meta[k]}")
    lines.append("-" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fujimura-tools show-experiment-settings",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    meta = _load_llm_meta(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm_meta": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_llm_meta(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
