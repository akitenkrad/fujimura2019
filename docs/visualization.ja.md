# 可視化

[English](visualization.md) | [日本語]

`--results-dir` を省くと，各ツールが必要とするサブコマンドの最新の run を runvault に聞く（`runvault path --experiment fujimura-silence --latest --subcommand run` / `--subcommand sweep`）．`--results-dir` には runvault の親 run・単独の run・runvault 以前の `results/<timestamp>/` のいずれも渡せる．

図と `sem_fit.json` は run ディレクトリの外（`results/fujimura-silence/figures/<run_slug>/`）に書く．`manifest.csv` は run の完了時に確定するので，あとから描いた図は記録の一部にできない．`--output-dir` で上書きできる．

## `fit-sem`

run の反復（子 run）をまたいでプールしたエージェントパネルから ABM 由来 SEM を [semopy](https://semopy.com/) で推定する：エージェントごとに 1 行の横断観測（t=0 を除く潜在状態時間平均 + 行動頻度）を構成し，

```
fear ~ psafety ; acquiescent ~ fear ; voice ~ fear ; silence ~ acquiescent
```

をフィットして `sem_fit.json` に標準化 β̃・Wald 95%CI・適合度（CFI / GFI / RMSEA / χ²）・`corr_silence_voice`・原著アンカーとの符号一致数を書く．semopy 不在時は per-path OLS にフォールバック．

## `visualize`

反復の tick 単位指標（あれば `sem_fit.json`）を読んで出力：

- `timeseries.png` — silence_rate / voice_volume / climate_of_silence の t 推移．原著アンカー（沈黙≈.38，発言≈.45）を点線で重ねる．
- `motive_mix.png` — 4 動機の within-silent 構成比推移．
- `path_diagram.png` — 推定 β̃ を原著値と並記した SEM パス図．

## `visualize-sweep`

sweep の子 run の `terminal` イベントを «試行 1 本 = 1 行» の表に戻して出力：

- `sweep_level_forest.png` — 階層 L ごとの最終 silence_rate / voice_volume（平均 ± SD），原著アンカー付き．
- `sweep_climate_heatmap.png` — （η × network_beta）格子上の平均 climate_of_silence — 大域沈黙螺旋の構造条件．

## `reproduce`

必要なら `fit-sem` を実行し，`paper_fig1_path_diagram.png`（図 1 相当）と `reproduction_report.json` を書き，B1–B5 照合表（符号一致 + CI 重なり + H5 沈黙 ⊥ 発言チェック）を表示する．

## `show-experiment-settings`

run の実験条件（`config.json` の `parameters`）と，LLM run なら `llm` ブロックと `llm_*` の run スコープ指標を表示する．`--results-dir` 省略時にどの run を解決するかは `--subcommand` で選ぶ．`--json` で機械可読出力．
