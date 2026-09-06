# CLI リファレンス

[English](cli.md) | [日本語]

`fujimura` バイナリは 4 サブコマンドを持つ．決定モードは排他切替：`--decision-mode rule`（ロジスティック ablation，LLM 不要）または `--decision-mode llm`（`socsim-llm`，Ollama 第一 → OpenAI フォールバック）．

## `run`

単一設定を `--runs` 回，派生シードで繰り返す．条件と反復リストを宣言する親 run 1 本と，シードごとの `run-replicate` 子 run を記録する．子はそれぞれ自分の tick 単位 `metrics.csv` と `artifacts/agent_panel.csv` を持つ．

| フラグ | 既定 | 意味 |
|------|------|------|
| `--decision-mode` | `rule` | `rule` \| `llm` |
| `--locale` | `ja-JP` | `ja-JP`（L=4, ῑ=.55）\| `en-US`（L=3, ῑ=.40） |
| `--n-teams` | 5 | チーム数 |
| `--team-size` | 80 | チームあたり従業員数 |
| `--n-levels` | ロケール既定 | 階層強度 L（ロケール既定を上書き） |
| `--eta` | 0.7 | 上司シグナル均質性 η |
| `--network-beta` | 0.05 | Watts–Strogatz 再配線率 β |
| `--network-k` | 6 | Watts–Strogatz 平均次数 k |
| `--ivt-strength-mean` | ロケール既定 | IVT 強度初期平均 |
| `--psafety-mean` | 0.613 | 心理的安全初期平均 |
| `--prompt-variant` | `A` | プロンプト wording A / B / C |
| `--p-retaliate` | 0.05 | エージェント・tick 単位の報復確率 |
| `--shock-t` | なし | 任意の σ ショック tick |
| `--t-max` | 12 | 最大 tick |
| `--runs` | 1 | 独立シード数（プール） |
| `--seed` | 2019 | root シード |
| `--llm-temperature` | 0.0 | LLM 温度 |
| `--llm-seed` | 0 | LLM シードオフセット |
| `--llm-model` | `llama3.1` | モデルヒント（参考） |
| `--cache-path` | `.llm_cache/cache.json` | プロンプト→応答キャッシュ（LLM モード） |
| `--output-dir` | `results` | 出力基点 |

## `sweep`

`--n-levels-values × (η レンジ) × --network-beta-values × seeds` の直積．グリッドを宣言する親 run 1 本と，セルごとの `sweep-point` 子 run を記録する．セル内の `--runs` 本の試行は子の `events.jsonl` の `terminal` 行（旧 `sweep_summary.csv` の 1 行にあたる），子の `metrics.csv` は試行をまたいだ平均を持つ．sweep が見るのは各試行の最終ステップだけで，試行は自分の時系列を持たない — `run` では反復を run にし，ここでは試行をイベントにする理由がこれである．

主フラグ：`--n-levels-values 2,3,4,5`，`--eta-min/--eta-max/--eta-step`，`--network-beta-values 0.05,0.10,0.20`，`--runs`，`--t-max`，`--seed`．

## `cultural-compare`

JP と EN ロケールを（それぞれのロケール既定で）並走させる．パス係数の文化 ablation 比較用．親 run 1 本と，(ロケール, シード) ごとの `cultural-replicate` 子 run を記録する．ロケールは子の `parameters` に入るので，JP と EN の行はシードではなく条件で見分けられる．

主フラグ：`--decision-mode`，`--n-teams`，`--team-size`，`--eta`，`--t-max`，`--runs`，`--seed`，`--cache-path`．

## `reproduce`

4 パス β̃ を推定し §5 アンカーと照合する Python `fit-sem` / `reproduce` ツールのエントリを表示する．
