//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `cultural-compare` のどれでも
//! 同一なので，ここ 1 箇所で組み立てる．ステップごとの指標，run 全体を 1 つの値で
//! 表す指標，スイープ 1 セルぶんの集約もここに集める．
//!
//! # 反復 (replicate) の形
//!
//! 旧 `metrics.csv` の第 1 列は `t` ではなく `seed` で，1 ファイルに同一条件の
//! 反復が複数本入っていた．runvault の `metrics.csv` は
//! (`run_uid`, `step`, `step_unit`, `scope`, `name`) が主キーなので，この形は
//! そのままでは載らない — 同じ `t` の行が seed の数だけ同じ主キーを名乗る．
//!
//! そこで **反復 1 本を 1 つの子 run にした**．`run` / `cultural-compare` は
//! «条件 1 つ + その反復» なので，親が [`RunOptions::sweep_parent`] を名乗って
//! グリッド (というより反復リスト) を宣言し，子がそれぞれ自分の `master_seed` と
//! `replicate_index` を持つ．同一条件の反復は `config_hash` が一致するので，
//! «どれとどれが同じ条件の繰り返しか» は機械が答えられる．
//!
//! 決め手は «反復が時系列を持つかどうか» である．`run` / `cultural-compare` は
//! 全ステップを観測して残すので，反復ごとに `metrics.csv` が要る (時系列の置き場は
//! そこしかない)．`sweep` は各試行の最終ステップしか見ないので，試行は時系列を
//! 持たず，`events.jsonl` の `terminal` 行 1 本で言い尽くせる — こちらは
//! hegselmann2002 と同じく «セル 1 つ = 子 run 1 本，試行 = terminal 行» にした．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::config::Config;
use crate::simulation::{MetricsRow, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "fujimura-silence";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "fujimura2019";
/// 分野．
///
/// LLM で駆動できるモデルだが `llm-safety` ではない — 測っているのは
/// モデルの安全性ではなく，組織網の上の沈黙と発言だからである．LLM 側の同一性は
/// `llm` ブロック ([`llm_block`]) が持つ．`simulation` を名乗ると `master_seed` が
/// 必須になるが，実際に潜在状態・網・スケジューラを乱数で引くので正しい．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは離散 tick なので語彙の `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．集団指標はどれも従業員全体の集約なので `run`．
const SCOPE: &str = "run";

/// 試行イベントの種別．コア語彙に無いので `x.<repo_id>.<name>` を使う…
/// のではなく，試行の終端は生存時間解析と同じ形をしているのでコア語彙の
/// `terminal` をそのまま使う．
const TERMINAL: &str = "terminal";

// ---------------------------------------------------------------------------
// 論文メタデータ
// ---------------------------------------------------------------------------

/// この再現実験が対象としている論文．
///
/// 『組織学会大会論文集』8(1) には DOI が無い (設計書の書誌表も「記載なし」) ので，
/// 同定は vault の `paper-id` で行う．
///
/// target は 2 つ持つ．原著の主張は «心理的安全 → 怖れ → 黙従 → 沈黙 / 怖れ →
/// 発言» という媒介経路 (B1–B4) と，«沈黙と発言は別個の行動次元» (B5) の 2 本で，
/// 後者だけが Rust 側で直接測れる量 (`corr_silence_voice`) に対応する．
pub fn replication() -> Replication {
    Work::paper_id("P00001824")
        .title("組織における沈黙と発言の規定要因 ―心理的安全と沈黙動機の影響過程―")
        .year(2019)
        .source_version("published")
        .target(Target::claim(
            "psafety-fear-acquiescence-silence-path",
            "Psychological safety lowers fear, fear raises acquiescence and suppresses voice, and acquiescence drives silence",
        ))
        .target(Target::claim(
            "silence-voice-independence",
            "Silence and voice are separate behavioural dimensions rather than opposites",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/fujimura2019/設計書.md")
}

/// 原著が報告した値のうち，この実装が run 1 本の中で直接測るもの．
///
/// 書けるのは `corr_silence_voice` の 1 つだけである．原著 §5 の
/// 「沈黙経験と発言経験は無相関 ($r=.02$, n.s.)」がそれで，本文にその数字がある．
///
/// 4 つのパス係数 $\tilde\beta$ (B1–B4) はここには書かない — Rust 側は係数を
/// 推定せず，`agent_panel.csv` から Python (semopy) が推定するので，run スコープに
/// 対応する観測値が無い．B11/B12 の «平均沈黙率 ≈ .38 / 平均発言率 ≈ .45» も
/// 書かない．原著が報告しているのは 4 件法の M = 2.15 / 2.36 であって，
/// .38 / .45 はこの再現実装が $[0,1]$ へ線形変換した値である．変換値を
/// 「論文が報告した値」の欄に置くと，原著が印字した数と自前の換算が後から
/// 見分けられなくなる (±0.05 の帯に至ってはこちらが決めたものなので当然除く)．
pub fn log_paper_reference(run: &mut Run) {
    run.log_reference("corr_silence_voice", 0.02)
        .target("silence-voice-independence")
        .source("藤村・日野 (2019) §5: 沈黙経験と発言経験の相関 r=.02 (n.s.)")
        .send()
        .expect("原著の報告値の記録に失敗");
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが
/// 答えたか» を決める (`mock://…` はオフラインの scripted クライアント，
/// それ以外はホスト名で Ollama / OpenAI を分ける)．
///
/// rule モードではこれを呼ばない．LLM を 1 回も叩かない run に `llm` ブロックを
/// 付けると，存在しないモデルを名乗ることになる．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // プロンプトはエージェントごとに組み立てられ，固定の system prompt を
        // 持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// 実験条件 (parameters)
// ---------------------------------------------------------------------------

/// 条件そのもの．シードも反復数も含まない．
///
/// 旧 `config.json` に対して `beta` (SEM パスを補正する係数群) と `init` 全体を
/// 加えてある．どちらも結果を決める量なので，落とすと «同じ条件» を名乗る 2 本の
/// run が実は違う係数で回っていた，ということが起こりうる．`config_hash` は
/// 条件の同一性を判定するためのものなので，条件はすべて入れる．
#[derive(Serialize)]
pub struct ConditionParameters {
    pub decision_mode: &'static str,
    pub locale: &'static str,
    pub hierarchy_strength: u8,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_employees: usize,
    pub n_levels: u8,
    pub network_k: usize,
    pub network_beta: f64,
    pub eta: f64,
    pub prompt_variant: &'static str,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub init: crate::config::InitDist,
    pub beta: crate::config::BetaGroup,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    /// プロンプト → 応答キャッシュの置き場．条件ではなく置き場なので
    /// `hash_exclude` で `config_hash` から外す ([`HASH_EXCLUDE`])．
    pub llm_cache_path: Option<String>,
}

/// `config_hash` から外すポインタ．
///
/// キャッシュのパスは «どこに置いたか» であって条件ではない．同じ条件の run を
/// 別のキャッシュファイルで回しても同じ条件である．
pub const HASH_EXCLUDE: [&str; 1] = ["/llm_cache_path"];

impl ConditionParameters {
    /// [`Config`] から条件だけを取り出す．
    pub fn from_config(cfg: &Config) -> Self {
        ConditionParameters {
            decision_mode: cfg.decision_mode.label(),
            locale: cfg.locale.label(),
            hierarchy_strength: cfg.locale.default_hierarchy_strength(),
            n_teams: cfg.n_teams,
            team_size: cfg.team_size,
            n_employees: cfg.n_employees(),
            n_levels: cfg.n_levels,
            network_k: cfg.network_k,
            network_beta: cfg.network_beta,
            eta: cfg.eta,
            prompt_variant: cfg.prompt_variant.label(),
            p_retaliate: cfg.p_retaliate,
            shock_t: cfg.shock_t,
            shock_magnitude: cfg.shock_magnitude,
            t_max: cfg.t_max,
            init: cfg.init,
            beta: cfg.beta,
            llm_temperature: cfg.llm.temperature,
            llm_seed: cfg.llm.seed,
            llm_cache_path: cfg.llm.cache_path.clone(),
        }
    }
}

/// 反復 1 本 (子 run) の実験条件．
///
/// `seed` はこの反復が実際に使ったシードで，`master_seed` と同じ値である．
/// `seed_pointers` で seed として宣言するので `config_hash` からは外れ，
/// 同一条件の反復どうしは `config_hash` が一致する．
#[derive(Serialize)]
pub struct ReplicateParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub seed: u64,
}

/// 「1 条件 + その反復群」の実験条件．
///
/// `run` の親 run と，`sweep` のセル 1 つぶんの子 run が同じ形をしている．
/// どちらも «この条件を `runs` 本回す» という宣言である．`base_seed` は
/// 反復ごとのシードを派生させる元で，それ自体でシミュレーションを回す値ではない．
#[derive(Serialize)]
pub struct ReplicateGroupParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub runs: usize,
    pub base_seed: u64,
}

/// `run` 親 / `sweep` セル子の seed ポインタ．
pub const GROUP_SEED_POINTERS: [&str; 1] = ["/base_seed"];
/// 反復 (子 run) の seed ポインタ．
pub const REPLICATE_SEED_POINTERS: [&str; 1] = ["/seed"];

// ---------------------------------------------------------------------------
// ステップごとの指標
// ---------------------------------------------------------------------------

/// 反復 1 本ぶんの記録．
///
/// ステップごとの 7 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `corr_silence_voice` / `final_round` を書く．旧 `metrics.csv` の
/// `seed` 列は run の同一性そのもの (`master_seed`) になったので指標にはしない．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_rows {
        log_step(run, m);
    }
    run.log_metrics(
        SCOPE,
        &[
            // 時刻ごとの値ではなく，エージェントごとの時間平均どうしの Pearson r
            // (原著 H5)．ステップ指標と同じ名前にすると step の有無だけが両者の
            // 違いになるが，こちらにステップごとの相関は無いので衝突しない．
            ("corr_silence_voice", result.corr_silence_voice),
            ("final_round", result.final_round as f64),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// [`MetricsRow`] の数値フィールドを 1 ステップぶんまとめて書く．
///
/// `motive_mix_*` の 4 本は «沈黙動機というカテゴリに番号を振ったもの» ではなく，
/// 各動機が沈黙者に占める割合という «ステップごとの数» が 4 つあるだけである
/// (沈黙が 1 人でもいれば和は 1，いなければ全部 0)．
fn log_step(run: &mut Run, m: &MetricsRow) {
    run.log_metrics_at(
        m.t,
        T_UNIT,
        SCOPE,
        &[
            ("silence_rate", m.silence_rate),
            ("voice_volume", m.voice_volume),
            ("climate_of_silence", m.climate_of_silence),
            ("motive_mix_acquiescent", m.motive_mix_acquiescent),
            ("motive_mix_quiescent", m.motive_mix_quiescent),
            ("motive_mix_prosocial", m.motive_mix_prosocial),
            ("motive_mix_opportunistic", m.motive_mix_opportunistic),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// LLM 呼び出しの内訳を run スコープの指標として書く．
///
/// rule モードでは呼ばない．0 回という数を書くこと自体は嘘ではないが，
/// LLM を配線していない run に LLM の指標が並ぶと，`llm` ブロックの有無と
/// 食い違って見える．
///
/// `tokens_in` / `tokens_out` / `cost_usd` は書かない — socsim-llm の
/// `MetadataCollector` はトークン数も費用も持たないので，予約名に入れる値が無い．
pub fn log_llm_usage(run: &mut Run, result: &SimulationResult) {
    let (fails, total) = result.parse_fail;
    let parse_fail_rate = if total > 0 {
        fails as f64 / total as f64
    } else {
        0.0
    };
    run.log_metrics(
        SCOPE,
        &[
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
            ("llm_parse_failures", fails as f64),
            ("llm_parse_fail_rate", parse_fail_rate),
        ],
    )
    .expect("LLM 呼び出しの内訳の記録に失敗");
}

// ---------------------------------------------------------------------------
// sweep の試行 (events.jsonl)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く観測行．
///
/// 予約キーだけを持つ．数はここには書かない — 試行の最終値は下の
/// [`TrialTerminal`] が正本なので，同じ数を 2 箇所に置かない．この行が持つのは
/// 「その試行をいつ見たか」という時間軸だけである．
///
/// `terminal` 行だけでも生存時間解析は組めるが (`schema/v1/event.json` の注記)，
/// `runvault verify --deep` は terminal の `unit_id` が observation にも現れる
/// ことを要求するので，観測した時刻を明示的に残す．
#[derive(Serialize)]
struct TrialObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// `events.jsonl` に書く試行の終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは自由欄で，旧 `sweep_summary.csv` の 1 行がこの 1 行に対応する．
///
/// シードの欄を `seed` としているのに対し，セル子 run の parameters 側は
/// `base_seed` という別の名前にしてある．`sweep_events_table` は parameters の
/// 列を event の同名列に上書きするので，同じ名前にすると試行ごとのシードが
/// base seed で黙って潰れる．
#[derive(Serialize)]
struct TrialTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    seed: u64,
    silence_rate: f64,
    voice_volume: f64,
    climate_of_silence: f64,
    corr_silence_voice: f64,
    motive_mix_acquiescent: f64,
    motive_mix_quiescent: f64,
}

/// 1 試行の最終値．セル 1 つぶんの集約の材料でもある．
pub struct TrialOutcome {
    pub final_round: u64,
    pub silence_rate: f64,
    pub voice_volume: f64,
    pub climate_of_silence: f64,
    pub corr_silence_voice: f64,
    pub motive_mix_acquiescent: f64,
    pub motive_mix_quiescent: f64,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終ステップから取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let last = result
            .metrics_rows
            .last()
            .expect("metrics_rows は t=0 を含む");
        TrialOutcome {
            final_round: result.final_round,
            silence_rate: last.silence_rate,
            voice_volume: last.voice_volume,
            climate_of_silence: last.climate_of_silence,
            corr_silence_voice: result.corr_silence_voice,
            motive_mix_acquiescent: last.motive_mix_acquiescent,
            motive_mix_quiescent: last.motive_mix_quiescent,
        }
    }
}

/// 試行 1 本を `terminal` イベントとして書く．
///
/// モデルは収束判定を持たず必ず `t_max` まで回るので，`outcome` は常に
/// `horizon`，`censored` は常に真である (打ち切りの行は `t == budget` で
/// なければならず，`final_round == t_max` なのでこれを満たす)．
pub fn log_trial(run: &mut Run, index: usize, seed: u64, t_max: u64, outcome: &TrialOutcome) {
    let unit_id = format!("trial-{index}");
    // sweep が見るのは各試行の最終ステップだけなので，観測時刻もそこ 1 点．
    run.log_event(
        "observation",
        &TrialObservation {
            unit_id: &unit_id,
            t: outcome.final_round,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の observation の記録に失敗: {e}"));

    run.log_event(
        TERMINAL,
        &TrialTerminal {
            unit_id: &unit_id,
            t: outcome.final_round,
            t_unit: T_UNIT,
            outcome: "horizon",
            censored: true,
            budget: t_max,
            seed,
            silence_rate: outcome.silence_rate,
            voice_volume: outcome.voice_volume,
            climate_of_silence: outcome.climate_of_silence,
            corr_silence_voice: outcome.corr_silence_voice,
            motive_mix_acquiescent: outcome.motive_mix_acquiescent,
            motive_mix_quiescent: outcome.motive_mix_quiescent,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

/// 1 セル (1 つの `(n_levels, η, network_beta)`) を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行
/// ごとの `silence_rate` を指標にすると (`run_uid`, `step`, `scope`, `name`) が
/// 重複する．散らばりが要る図は `events.jsonl` から組み直す．
pub fn log_cell_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("mean_silence_rate", mean(&|t| t.silence_rate)),
            ("mean_voice_volume", mean(&|t| t.voice_volume)),
            ("mean_climate_of_silence", mean(&|t| t.climate_of_silence)),
            ("mean_corr_silence_voice", mean(&|t| t.corr_silence_voice)),
        ],
    )
    .expect("セル集約の記録に失敗");
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 反復 1 本のシードを base seed から決定的に派生させる (`run`)．
pub fn replicate_seed(base: u64, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[index as u64])
}

/// 反復 1 本のシードを base seed から決定的に派生させる (`cultural-compare`)．
///
/// ロケールは階層強度 `L` で区別する — JP は `L=4`，EN は `L=3` なので，
/// 同じ `index` でもロケールが違えば別のシードになる．
pub fn cultural_seed(base: u64, hierarchy_strength: u8, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[hierarchy_strength as u64, index as u64])
}

/// 試行 1 本のシードを base seed とセル座標から決定的に派生させる (`sweep`)．
pub fn trial_seed(base: u64, n_levels: u8, eta: f64, network_beta: f64, index: usize) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            n_levels as u64,
            (eta * 1000.0) as u64,
            (network_beta * 1000.0) as u64,
            index as u64,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(replicate_seed(2019, 3), replicate_seed(2019, 3));
        assert_eq!(cultural_seed(2019, 4, 3), cultural_seed(2019, 4, 3));
        assert_eq!(
            trial_seed(2019, 3, 0.5, 0.05, 1),
            trial_seed(2019, 3, 0.5, 0.05, 1)
        );
    }

    #[test]
    fn each_coordinate_changes_the_seed() {
        let base = trial_seed(2019, 3, 0.5, 0.05, 0);
        assert_ne!(
            base,
            trial_seed(2020, 3, 0.5, 0.05, 0),
            "base が効いていない"
        );
        assert_ne!(
            base,
            trial_seed(2019, 4, 0.5, 0.05, 0),
            "n_levels が効いていない"
        );
        assert_ne!(
            base,
            trial_seed(2019, 3, 0.6, 0.05, 0),
            "eta が効いていない"
        );
        assert_ne!(
            base,
            trial_seed(2019, 3, 0.5, 0.10, 0),
            "beta が効いていない"
        );
        assert_ne!(
            base,
            trial_seed(2019, 3, 0.5, 0.05, 1),
            "index が効いていない"
        );
    }

    #[test]
    fn locales_get_distinct_seeds() {
        // JP (L=4) と EN (L=3) は同じ index でも別のシードになる．
        assert_ne!(cultural_seed(2019, 4, 0), cultural_seed(2019, 3, 0));
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは
    /// 過去の run と結果を比較できなくなっている．Cargo.lock が socsim の commit を
    /// 固定しているので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(replicate_seed(2019, 0), 13_938_057_335_077_554_196);
        assert_eq!(replicate_seed(2019, 1), 13_938_058_434_589_182_407);
        assert_eq!(cultural_seed(2019, 4, 0), 73_382_701_531_917_656);
        assert_eq!(cultural_seed(2019, 3, 0), 80_078_727_346_398_071);
        assert_eq!(
            trial_seed(2019, 3, 0.5, 0.05, 0),
            15_632_339_161_558_697_201
        );
    }
}
