"""runs.py — どの run を読むか，そして反復をどう束ねるか．

Rust 側は `run` / `cultural-compare` 1 回を «親 run 1 本 + 反復ごとの子 run» として
記録する．旧 `metrics.csv` / `agent_panel.csv` が持っていた «同一条件の反復を 1 枚に
プールした表» は，runvault では子 run に分かれている — `metrics.csv` の主キーは
(`run_uid`, `step`, `step_unit`, `scope`, `name`) なので，seed の違う行が同じ `t` を
名乗ることはできないからである．

このモジュールはその表を読み側で組み直す．旧形式と同じ列 (`seed`, `t`, …) を返すので，
`visualize` / `fit-sem` の計算そのものは移行前後で 1 セルも変わらない．

run ディレクトリの解決は `runvault path` に任せる — `results/` を走査して新しそうな
ディレクトリを当てにいかない．run_slug には条件と環境のハッシュが入るので，こちらで
名前を組み立てることはできないし，できたとしてもすべきでない．
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from runvault.read import (
    artifacts_dir,
    load_run_meta,
    run_subcommand,
    sweep_children,
)
from runvault.read import runvault_path as _runvault_path

EXPERIMENT = "fujimura-silence"

#: 反復を子 run として持つ («親» の) サブコマンド．
PARENT_SUBCOMMANDS = ("run", "cultural-compare", "sweep")


def resolve_run_dir(
    results_dir: str | os.PathLike | None,
    *,
    subcommand: str = "run",
    results_root: str = "results",
) -> Path:
    """読む run ディレクトリを決める．

    `results_dir` が与えられていればそれを使う (legacy な `results/<timestamp>/` も
    そのまま渡せる)．与えられていなければ `runvault path --latest` に聞く．
    """
    if results_dir is not None:
        path = Path(results_dir)
        # legacy の `results/latest` シンボリックリンクは実体へ解決する．
        if path.is_symlink():
            return Path(os.path.realpath(path))
        return path
    return Path(_runvault_path(EXPERIMENT, results_root, subcommand=subcommand))


def is_runvault_run(run_dir: str | os.PathLike) -> bool:
    """runvault の run ディレクトリか (`run.json` があるか)．"""
    return load_run_meta(run_dir, required=False) is not None


def replicate_dirs(run_dir: str | os.PathLike) -> list[Path]:
    """反復 1 本ずつの run ディレクトリ．

    親 run を渡すとその子を，単独の run を渡すとそれ自身 1 本を返す．legacy な
    ディレクトリもそれ自身 1 本になる (1 枚の CSV に反復が入っているため)．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return [run_dir]
    if run_subcommand(run_dir) in PARENT_SUBCOMMANDS:
        children = [Path(c) for c in sweep_children(run_dir)]
        if not children:
            raise SystemExit(
                f"エラー: 親 run に子がありません: {run_dir}\n"
                "  子は lineage.parent_run_uid で親を指します．親と子が同じ"
                " results root にあるか確認してください．"
            )
        return children
    return [run_dir]


def master_seed(run_dir: str | os.PathLike) -> int:
    """その run を駆動したシード．旧 `metrics.csv` の `seed` 列にあたる．"""
    meta = load_run_meta(run_dir)
    assert meta is not None  # required=True raises rather than returning None
    seed = (meta.get("rng") or {}).get("master_seed")
    if seed is None:
        raise SystemExit(f"エラー: master_seed を持たない run です: {run_dir}")
    return int(seed)


def _metrics_wide(metrics_path: Path) -> pd.DataFrame:
    """long 形式の `metrics.csv` を «1 ステップ 1 行» に戻す．

    `runvault.read.metrics_wide` と同じ形を返すが，読み込みに
    `float_precision="round_trip"` を渡す点だけが違う．pandas の既定の C パーサは
    f64 を 1 ULP 落とすことがあり (例: `0.26666666666666666` → `0.2666666666666666`)，
    移行前後の値の照合が «実際には一致しているのに一致しない» と出る．記録された
    数をそのまま読むために，ここでは自前で読む．
    """
    df = pd.read_csv(metrics_path, float_precision="round_trip")
    stepped = df[df["step"].notna()]
    return (
        stepped.pivot_table(index="step", columns="name", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .astype({"step": int})
        .sort_values("step")
        .reset_index(drop=True)
    )


def pooled_metrics(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復をプールしたステップ指標．旧 `metrics.csv` と同じ列を返す．

    子 run の long 形式 `metrics.csv` を wide に戻し，`seed` 列を `run.json` の
    `master_seed` から補って縦に積む．legacy な wide `metrics.csv` はそのまま返す．
    """
    run_dir = Path(run_dir)
    if not is_runvault_run(run_dir):
        return pd.read_csv(run_dir / "metrics.csv", float_precision="round_trip")

    frames = []
    for child in replicate_dirs(run_dir):
        wide = _metrics_wide(child / "metrics.csv").rename(columns={"step": "t"})
        wide.insert(0, "seed", master_seed(child))
        frames.append(wide)
    return pd.concat(frames, ignore_index=True)


def pooled_panel(run_dir: str | os.PathLike) -> pd.DataFrame:
    """反復をプールしたエージェントパネル．旧 `agent_panel.csv` と同じ列を返す．

    パネルは表のまま子 run の `artifacts/` にあるので，読んで縦に積むだけである
    (`seed` 列はもともとファイルの中にある)．
    """
    run_dir = Path(run_dir)
    frames = []
    for child in replicate_dirs(run_dir):
        path = Path(artifacts_dir(child)) / "agent_panel.csv"
        if not path.exists():
            raise SystemExit(f"エラー: agent_panel.csv が見つかりません: {path}")
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    return pd.concat(frames, ignore_index=True)


def analysis_output_dir(run_dir: str | os.PathLike, override: str | None) -> Path:
    """図や `sem_fit.json` の置き場．

    `manifest.csv` は `finish()` が確定させるので，run が終わったあとに作ったものを
    `artifacts/` に置くとハッシュを持たない (＝記録の一部でない) ファイルが混ざる．
    runvault の run には `runvault.read.figures_dir` が示す run ディレクトリの外を使う．
    """
    if override is not None:
        out = Path(override)
    elif is_runvault_run(run_dir):
        from runvault.read import figures_dir

        out = Path(figures_dir(run_dir))
    else:
        out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
