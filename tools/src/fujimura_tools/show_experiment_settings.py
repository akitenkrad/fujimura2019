"""fujimura-tools show-experiment-settings — print a run directory's settings.

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．どのサブコマンドの run かは `run.json` の
`subcommand` が答える．旧 `llm_meta.json` の中身は run スコープの指標
(`llm_calls` / `llm_cache_hits` / …) と `run.json` の `llm` ブロックに移ったので，
そちらから拾う．legacy な flat `config.json` / `sweep_config.json` / `llm_meta.json`
も従来どおり読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment fujimura-silence --latest --subcommand run
    runvault path --experiment fujimura-silence --latest --subcommand sweep

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

from runvault.read import config_parameters, load_run_meta, run_scope_metrics

from fujimura_tools import runs

FIELD_LABELS = {
    "decision_mode": "決定モード       ",
    "locale": "ロケール         ",
    "hierarchy_strength": "階層強度 L       ",
    "n_teams": "チーム数         ",
    "team_size": "チームサイズ     ",
    "n_employees": "従業員数 N       ",
    "n_levels": "階層数 n_levels  ",
    "eta": "上司均質性 η     ",
    "network_k": "平均次数 k       ",
    "network_beta": "再配線率 β       ",
    "prompt_variant": "プロンプト変種   ",
    "p_retaliate": "報復確率         ",
    "t_max": "最大 tick T      ",
    "runs": "試行数 runs      ",
    "seed": "シード           ",
    "base_seed": "シード基点       ",
}

#: `init` (初期分布) のうち表示する平均．SD は既定値から動かないので省く．
INIT_LABELS = {
    "psafety_mean": "心理的安全平均   ",
    "fear_mean": "怖れ平均         ",
    "acquiescent_mean": "黙従平均         ",
    "ivt_mean": "IVT 強度平均     ",
}


def _flatten(cfg: dict) -> dict:
    """入れ子の `init` を旧 flat 形式の名前に開く．

    移行後の `parameters` は `init` / `beta` を入れ子で持つ (条件をすべて
    `config_hash` に載せるため)．表示だけは旧来の平らな並びを保つ．
    """
    flat = dict(cfg)
    init = cfg.get("init")
    if isinstance(init, dict):
        flat.update({k: v for k, v in init.items() if k in INIT_LABELS})
    return flat


def render_config(cfg: dict, run_dir: Path, subcommand: str) -> str:
    flat = _flatten(cfg)
    lines = ["=" * 70, f"実行設定 ({subcommand})", "=" * 70, f"run: {run_dir}", "-" * 70]

    def join(key: str) -> str:
        return ", ".join(str(v) for v in cfg.get(key, []))

    # sweep 親と cultural-compare 親はグリッドそのものを持つので候補列を並べる．
    for key, label in (
        ("n_levels_values", "階層強度 L       "),
        ("eta_values", "上司均質性 η     "),
        ("network_beta_values", "再配線率 β       "),
        ("locales", "ロケール         "),
    ):
        if key in cfg:
            lines.append(f"{label}: {join(key)}")

    for key, label in {**FIELD_LABELS, **INIT_LABELS}.items():
        if key in flat and f"{key}_values" not in cfg:
            lines.append(f"{label}: {flat[key]}")
    lines.append("=" * 70)
    return "\n".join(lines)


def llm_summary(run_dir: Path) -> dict | None:
    """LLM の同定情報と呼び出しの内訳．

    移行後は `run.json` の `llm` ブロック (provider / model / temperature) と run
    スコープの指標に分かれている．rule モードの run はどちらも持たないので `None`．
    legacy な `llm_meta.json` があればそれをそのまま返す．
    """
    legacy = run_dir / "llm_meta.json"
    if legacy.exists():
        with legacy.open(encoding="utf-8") as f:
            return json.load(f)

    meta = load_run_meta(run_dir, required=False)
    if meta is None:
        return None
    block = meta.get("llm")
    scoped = {k: v for k, v in run_scope_metrics(run_dir).items() if k.startswith("llm_")}
    if not block and not scoped:
        return None
    out: dict = {}
    if block:
        out.update(
            {
                "provider": block.get("provider"),
                "model": block.get("model_snapshot"),
                "temperature": block.get("temperature"),
            }
        )
    out["seed"] = (meta.get("rng") or {}).get("master_seed")
    out.update(scoped)
    return out


def render_llm(meta: dict) -> str:
    lines = ["-" * 70, "LLM メタ情報", "-" * 70]
    for k, v in meta.items():
        lines.append(f"  {k:<20}: {v}")
    lines.append("-" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fujimura-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None)
    parser.add_argument(
        "--subcommand",
        default="run",
        help="--results-dir 省略時に runvault へ聞くサブコマンド (run / sweep / cultural-compare)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    run_dir = runs.resolve_run_dir(args.results_dir, subcommand=args.subcommand)
    if not run_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {run_dir}", file=sys.stderr)
        return 1

    meta = load_run_meta(run_dir, required=False)
    subcommand = str(meta["subcommand"]) if meta else "legacy"

    cfg = config_parameters(run_dir, required=False)
    if cfg is None:
        # legacy の sweep は sweep_config.json に条件を持つ．
        legacy_sweep = run_dir / "sweep_config.json"
        if not legacy_sweep.exists():
            print(f"エラー: 設定が見つかりません: {run_dir}", file=sys.stderr)
            return 1
        with legacy_sweep.open(encoding="utf-8") as f:
            cfg = json.load(f)
        subcommand = "sweep"

    llm = llm_summary(run_dir)

    if args.json:
        payload = {
            "run_dir": str(run_dir),
            "subcommand": subcommand,
            "parameters": cfg,
            "llm": llm,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_config(cfg, run_dir, subcommand))
        if llm is not None:
            print(render_llm(llm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
