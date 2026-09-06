//! Fujimura & Hino (2019) — Silence and voice in the organization CLI.
//!
//! `run`              : single configuration; `--decision-mode {rule|llm}` exclusive switch.
//! `sweep`            : Cartesian product over `n_levels × η × network_beta × seeds`.
//! `cultural-compare` : runs the JP and EN locales side by side (rule or LLM).
//! `reproduce`        : pointer to the Python `fit-sem` / `reproduce` tooling.
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．
//!
//! `run` と `cultural-compare` は «条件 + その反復» なので，親 run が反復リストを
//! 宣言し，反復 1 本ずつが子 run になる (`fujimura_silence::record` の冒頭を参照)．
//! `sweep` はセル 1 つが子 run で，そのセルの試行は `events.jsonl` の `terminal`
//! 行になる．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions, Stage};
use serde::Serialize;

use fujimura_silence::config::{
    parse_decision_mode, parse_prompt_variant, Config, InitDist, LlmSettings,
};
use fujimura_silence::llm::{build_live_client, SilenceClient};
use fujimura_silence::record::{
    self, ConditionParameters, ReplicateGroupParameters, ReplicateParameters, DOMAIN, EXPERIMENT,
    GROUP_SEED_POINTERS, HASH_EXCLUDE, REPLICATE_SEED_POINTERS, REPO_ID,
};
use fujimura_silence::simulation::{run_with_client_observed, save_agent_panel, SimulationResult};
use fujimura_silence::world::{parse_locale, Locale};

use socsim_llm::LlmClient;

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "fujimura",
    about = "Fujimura & Hino (2019) — Silence and voice in the organization (rule vs LLM)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration (rule or LLM decision mode).
    Run(RunArgs),
    /// Sweep n_levels × η × network_beta across seeds; one child run per cell.
    Sweep(SweepArgs),
    /// Run the JP and EN locales side by side (cultural-comparison ablation).
    CulturalCompare(CulturalCompareArgs),
    /// Pointer to the Python `fit-sem` / `reproduce` tooling.
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Decision mechanism (rule = logistic ablation; llm = socsim-llm).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Cultural locale (ja-JP / en-US).
    #[arg(long, default_value = "ja-JP")]
    locale: String,
    /// Number of teams.
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    /// Employees per team.
    #[arg(long, default_value_t = 80)]
    team_size: usize,
    /// Number of hierarchical levels (overrides the locale default when set).
    #[arg(long)]
    n_levels: Option<u8>,
    /// Supervisor-signal homogeneity η.
    #[arg(long, default_value_t = 0.7)]
    eta: f64,
    /// Watts–Strogatz rewiring β.
    #[arg(long, default_value_t = 0.05)]
    network_beta: f64,
    /// Watts–Strogatz mean degree k.
    #[arg(long, default_value_t = 6)]
    network_k: usize,
    /// Initial IVT-strength mean (overrides the locale default when set).
    #[arg(long)]
    ivt_strength_mean: Option<f64>,
    /// Initial psychological-safety mean.
    #[arg(long, default_value_t = 0.613)]
    psafety_mean: f64,
    /// Prompt-wording variant (A / B / C).
    #[arg(long, default_value = "A")]
    prompt_variant: String,
    /// Per-agent per-step retaliation probability.
    #[arg(long, default_value_t = 0.05)]
    p_retaliate: f64,
    /// Optional exogenous σ-shock time step.
    #[arg(long)]
    shock_t: Option<u64>,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 12)]
    t_max: u64,
    /// Number of independent runs (different seeds; one child run each).
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Random seed (governs the socsim core layer).
    #[arg(long, default_value_t = 2019)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,
    /// LLM generation seed offset.
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// LLM model hint (advisory; provider env vars take precedence).
    #[arg(long, default_value = "llama3.1")]
    llm_model: String,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// Output base directory (runvault results root).
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Decision mechanism (rule / llm).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Locale (ja-JP / en-US).
    #[arg(long, default_value = "ja-JP")]
    locale: String,
    /// n_levels sweep values (comma-separated).
    #[arg(long, default_value = "2,3,4,5")]
    n_levels_values: String,
    /// η minimum.
    #[arg(long, default_value_t = 0.3)]
    eta_min: f64,
    /// η maximum.
    #[arg(long, default_value_t = 0.9)]
    eta_max: f64,
    /// η step.
    #[arg(long, default_value_t = 0.1)]
    eta_step: f64,
    /// network_beta sweep values (comma-separated).
    #[arg(long, default_value = "0.05,0.10,0.20")]
    network_beta_values: String,
    /// Teams.
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    /// Team size.
    #[arg(long, default_value_t = 40)]
    team_size: usize,
    /// Runs (seeds) per cell.
    #[arg(long, default_value_t = 20)]
    runs: usize,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 12)]
    t_max: u64,
    /// Base seed.
    #[arg(long, default_value_t = 2019)]
    seed: u64,
    /// Output base directory (runvault results root).
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct CulturalCompareArgs {
    /// Decision mechanism (rule / llm).
    #[arg(long, default_value = "rule")]
    decision_mode: String,
    /// Teams.
    #[arg(long, default_value_t = 5)]
    n_teams: usize,
    /// Team size.
    #[arg(long, default_value_t = 40)]
    team_size: usize,
    /// Supervisor-signal homogeneity η.
    #[arg(long, default_value_t = 0.7)]
    eta: f64,
    /// Maximum simulation step.
    #[arg(long, default_value_t = 12)]
    t_max: u64,
    /// Runs (seeds) per locale.
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Base seed.
    #[arg(long, default_value_t = 2019)]
    seed: u64,
    /// Prompt → response cache path (LLM mode only).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// Output base directory (runvault results root).
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Output base directory (runvault results root).
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split([',', ' '])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect()
}

fn parse_u8_list(s: &str) -> Vec<u8> {
    s.split([',', ' '])
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.trim().parse::<u8>().ok())
        .collect()
}

fn cfg_from_run_args(args: &RunArgs) -> Config {
    let locale = parse_locale(&args.locale).unwrap_or_else(|e| panic!("{e}"));
    let ivt_mean = args
        .ivt_strength_mean
        .unwrap_or_else(|| locale.default_ivt_mean());
    Config {
        n_teams: args.n_teams,
        team_size: args.team_size,
        n_levels: args
            .n_levels
            .unwrap_or_else(|| locale.default_hierarchy_strength()),
        network_k: args.network_k,
        network_beta: args.network_beta,
        locale,
        eta: args.eta,
        decision_mode: parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}")),
        beta: Default::default(),
        init: InitDist {
            psafety_mean: args.psafety_mean,
            ivt_mean,
            ..InitDist::default()
        },
        prompt_variant: parse_prompt_variant(&args.prompt_variant)
            .unwrap_or_else(|e| panic!("{e}")),
        p_retaliate: args.p_retaliate,
        shock_t: args.shock_t,
        shock_magnitude: 0.3,
        t_max: args.t_max,
        runs: args.runs,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        // 出力先は runvault が決めるので Config は持たない．
        output_dir: String::new(),
    }
}

fn print_run_line(idx: usize, n: usize, seed: u64, r: &SimulationResult) {
    let last = r.metrics_rows.last();
    println!(
        "[{}/{}] seed={} silence={:.3} voice={:.3} C={:.3} corr(s,v)={:+.3}",
        idx,
        n,
        seed,
        last.map(|m| m.silence_rate).unwrap_or(0.0),
        last.map(|m| m.voice_volume).unwrap_or(0.0),
        last.map(|m| m.climate_of_silence).unwrap_or(0.0),
        r.corr_silence_voice,
    );
}

/// LLM モードなら本番クライアントを組み立てる．rule モードでは `None`．
///
/// `Run::start` より前に組み立てるのは，`llm` ブロックに書く model / endpoint を
/// クライアント自身から採るためである (名前を推測で書かない)．
fn build_client(cfg: &Config) -> Option<SilenceClient> {
    if !cfg.decision_mode.is_llm() {
        return None;
    }
    Some(build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM client build failed: {e}")))
}

/// LLM キャッシュの置き場を用意する (LLM モードのみ)．
fn ensure_cache_dir(cfg: &Config, cache_path: &str) {
    if cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }
}

/// 反復 1 本を子 run として回し，記録する．
///
/// `run` と `cultural-compare` で共通．子の subcommand 名だけが違う — `run` の子は
/// 「単一ロケールの 1 本」，`cultural-compare` の子は「JP と EN を並べたうちの
/// 1 本」で，同じ名前にすると `runvault path --subcommand` がどちらを返すか
/// 分からなくなる．
///
/// 進捗の 1 単位は 1 ステップ．費用がそこにあるからで，1 ステップは全従業員に
/// ついて決定を出し，LLM モードではその 1 つ 1 つがモデル呼び出しになる．反復を
/// 単位にすると，ライブの 1 本は 0/1 と出したきり終わりまで黙る．`stage` は
/// 呼び出し側が開ける — コマンド全体で 1 つにすることで，反復をまたいでも割合が
/// 途中で 100% に戻らない．
#[allow(clippy::too_many_arguments)]
fn run_replicate(
    subcommand: &str,
    results_root: &str,
    cfg: &Config,
    seed: u64,
    replicate_index: usize,
    lineage: &Lineage,
    stage: &mut Stage,
) -> SimulationResult {
    let client = build_client(cfg);
    let llm = client
        .as_ref()
        .map(|c| record::llm_block(c.inner().model(), c.inner().endpoint(), cfg.llm.temperature));

    let parameters = ReplicateParameters {
        condition: ConditionParameters::from_config(cfg),
        seed,
    };

    let mut options = RunOptions::new(EXPERIMENT, subcommand)
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(results_root)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .hash_exclude(HASH_EXCLUDE)
        .seed_pointers(REPLICATE_SEED_POINTERS)
        .master_seed(seed)
        .replicate_index(replicate_index as u64)
        .lineage(lineage.clone())
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }

    let mut child = Run::start(options).expect("runvault: 子 run の開始に失敗");

    let result = run_with_client_observed(cfg, client, |_| stage.tick())
        .unwrap_or_else(|e| panic!("replicate run failed: {e}"));

    record::log_simulation(&mut child, &result);
    if cfg.decision_mode.is_llm() {
        record::log_llm_usage(&mut child, &result);
    }
    record::log_paper_reference(&mut child);
    save_agent_panel(
        &result.panel_rows,
        &child.dir().join("artifacts").to_string_lossy(),
    );
    child.finish().expect("runvault: 子 run の完了に失敗");

    result
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let base_cfg = cfg_from_run_args(&args);
    ensure_cache_dir(&base_cfg, &args.cache_path);

    let runs = base_cfg.runs.max(1);

    // 親 run: 条件と反復リストを宣言するだけで，シミュレーションは回さない．
    // 反復ごとの派生シードで駆動されるので単一の master_seed は名乗らない
    // (base seed は /base_seed と seed_pointers 経由で execution_hash に残る)．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&ReplicateGroupParameters {
                condition: ConditionParameters::from_config(&base_cfg),
                runs,
                base_seed: base_cfg.seed,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== Fujimura & Hino (2019) — Silence and voice ===");
    println!(
        "decision-mode: {} | locale: {} (L={}) | teams: {}×{} (={}) | η={:.2} | WS k={} β={:.2}",
        base_cfg.decision_mode.label(),
        base_cfg.locale.label(),
        base_cfg.locale.default_hierarchy_strength(),
        base_cfg.n_teams,
        base_cfg.team_size,
        base_cfg.n_employees(),
        base_cfg.eta,
        base_cfg.network_k,
        base_cfg.network_beta,
    );
    println!(
        "t_max={} runs={} seed={} | output: {}",
        base_cfg.t_max,
        runs,
        base_cfg.seed,
        parent.dir().display()
    );
    println!("----------------------------------------------------------------------");

    let mut last: Option<SimulationResult> = None;
    // 親 run に stage を 1 つ．反復はすべて同じ条件・同じ t_max なので重みでは
    // なく数える．子 run ごとに開け直すと小さな 100% が並ぶだけになる．
    let mut stage = parent.stage("steps", runs * base_cfg.t_max as usize);
    for run_idx in 0..runs {
        let seed = record::replicate_seed(base_cfg.seed, run_idx);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run_replicate(
            "run-replicate",
            &args.output_dir,
            &cfg,
            seed,
            run_idx,
            &lineage,
            &mut stage,
        );
        print_run_line(run_idx + 1, runs, seed, &result);
        last = Some(result);
    }
    // manifest.csv は finish() で封をされる．その後に 1 行足せば，manifest が
    // 食い違うダイジェストを持つことになる．
    stage.close();

    let dir = parent.finish().expect("runvault: 親 run の完了に失敗");

    println!("----------------------------------------------------------------------");
    if let Some(result) = &last {
        println!(
            "LLM calls (最後の反復): {} | cache-hit: {} ({:.1}%) | model: {}",
            result.metadata.total(),
            result.metadata.cache_hits(),
            result.metadata.cache_hit_rate() * 100.0,
            result.llm_model,
        );
    }
    println!("親 run   → {}", dir.display());
    println!("反復 {runs} 本 → 子 run (subcommand=run-replicate)．metrics.csv がステップごとの時系列，artifacts/agent_panel.csv がパネル．");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(Serialize)]
struct SweepParameters {
    decision_mode: &'static str,
    locale: &'static str,
    n_levels_values: Vec<u8>,
    eta_values: Vec<f64>,
    network_beta_values: Vec<f64>,
    n_teams: usize,
    team_size: usize,
    runs: usize,
    t_max: u64,
    base_seed: u64,
}

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let locale = parse_locale(&args.locale).unwrap_or_else(|e| panic!("{e}"));

    let n_levels_vals = parse_u8_list(&args.n_levels_values);
    let mut eta_vals: Vec<f64> = Vec::new();
    let mut e = args.eta_min;
    while e <= args.eta_max + 1e-9 {
        eta_vals.push((e * 1000.0).round() / 1000.0);
        e += args.eta_step;
    }
    let beta_vals = parse_f64_list(&args.network_beta_values);

    let n_cells = n_levels_vals.len() * eta_vals.len() * beta_vals.len();
    let n_total = n_cells * args.runs;

    // 親 run: グリッド定義そのものを parameters に持つ．個別セルの指標は書かない．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&SweepParameters {
                decision_mode: decision_mode.label(),
                locale: locale.label(),
                n_levels_values: n_levels_vals.clone(),
                eta_values: eta_vals.clone(),
                network_beta_values: beta_vals.clone(),
                n_teams: args.n_teams,
                team_size: args.team_size,
                runs: args.runs,
                t_max: args.t_max,
                base_seed: args.seed,
            })
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== fujimura-sweep ===");
    println!(
        "decision_mode: {} | locale: {} | n_levels={:?} η={:?} network_beta={:?} | runs/cell={} | total {} runs",
        decision_mode.label(),
        locale.label(),
        n_levels_vals,
        eta_vals,
        beta_vals,
        args.runs,
        n_total,
    );
    println!("base seed: {}", args.seed);
    println!("output: {}", parent.dir().display());
    println!("------------------------------------------------------------");

    let mut idx = 0usize;
    // グリッド全体で stage を 1 つ．掃引しているのは階層数・η・β で，どれも
    // 仕事の量を変えない (従業員数も t_max も固定) ので，重みではなく数える．
    let mut stage = parent.stage("steps", n_total * args.t_max as usize);

    for &nl in &n_levels_vals {
        for &eta in &eta_vals {
            for &nb in &beta_vals {
                let cell_cfg = Config {
                    n_teams: args.n_teams,
                    team_size: args.team_size,
                    n_levels: nl,
                    network_beta: nb,
                    locale,
                    eta,
                    decision_mode,
                    init: InitDist {
                        ivt_mean: locale.default_ivt_mean(),
                        ..InitDist::default()
                    },
                    t_max: args.t_max,
                    runs: args.runs,
                    seed: args.seed,
                    llm: LlmSettings {
                        cache_path: Some(".llm_cache/cache.json".to_string()),
                        ..LlmSettings::default()
                    },
                    ..Config::default()
                };

                // 子は «そのセルの試行群» そのもの．base seed とセル座標から
                // すべての試行シードが決まるので master_seed は base seed であり，
                // 同一セルの繰り返しは無いので replicate_index は 0．
                let mut child = Run::start(
                    RunOptions::new(EXPERIMENT, "sweep-point")
                        .repo_id(REPO_ID)
                        .domain(DOMAIN)
                        .results_root(&args.output_dir)
                        .parameters(&ReplicateGroupParameters {
                            condition: ConditionParameters::from_config(&cell_cfg),
                            runs: args.runs,
                            base_seed: args.seed,
                        })
                        .expect("runvault: 子 run の parameters の組み立てに失敗")
                        .hash_exclude(HASH_EXCLUDE)
                        .seed_pointers(GROUP_SEED_POINTERS)
                        .master_seed(args.seed)
                        .replicate_index(0)
                        .lineage(lineage.clone())
                        .replication(record::replication()),
                )
                .expect("runvault: sweep 子 run の開始に失敗");

                let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
                for run_idx in 0..args.runs {
                    idx += 1;
                    let seed = record::trial_seed(args.seed, nl, eta, nb, run_idx);
                    let cfg = Config {
                        seed,
                        runs: 1,
                        ..cell_cfg.clone()
                    };
                    let client = build_client(&cfg);
                    let result = run_with_client_observed(&cfg, client, |_| stage.tick())
                        .unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                    let outcome = record::TrialOutcome::from_result(&result);
                    record::log_trial(&mut child, run_idx, seed, args.t_max, &outcome);
                    if idx.is_multiple_of(20) || idx == n_total {
                        println!(
                            "[{}/{}] L={} η={:.2} β={:.2} run={} silence={:.3}",
                            idx, n_total, nl, eta, nb, run_idx, outcome.silence_rate
                        );
                    }
                    trials.push(outcome);
                }
                record::log_cell_summary(&mut child, &trials);
                child.finish().expect("runvault: sweep 子 run の完了に失敗");
            }
        }
    }

    stage.close();

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("親 run     → {}", dir.display());
    println!("セル {n_cells} 個 → 子 run (subcommand=sweep-point)．試行 1 本が events.jsonl の terminal 行 1 本．");
}

// --------------------------------------------------------------------------- //
// cultural-compare
// --------------------------------------------------------------------------- //

/// cultural-compare 親 run の実験条件．
///
/// 2 つのロケールを並べる指示そのものなので `ConditionParameters` は持たない —
/// JP と EN は別々の条件で，どちらか一方を親の条件として名乗ることはできない．
#[derive(Serialize)]
struct CulturalParameters {
    decision_mode: &'static str,
    locales: [&'static str; 2],
    n_teams: usize,
    team_size: usize,
    eta: f64,
    t_max: u64,
    runs: usize,
    base_seed: u64,
}

fn cmd_cultural_compare(args: CulturalCompareArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let runs = args.runs.max(1);

    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "cultural-compare")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&CulturalParameters {
                decision_mode: decision_mode.label(),
                locales: [Locale::JaJp.label(), Locale::EnUs.label()],
                n_teams: args.n_teams,
                team_size: args.team_size,
                eta: args.eta,
                t_max: args.t_max,
                runs,
                base_seed: args.seed,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(GROUP_SEED_POINTERS)
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: 親 run の開始に失敗");

    let lineage = Lineage {
        sweep_id: parent.sweep_id().map(str::to_string),
        parent_run_uid: Some(parent.run_uid().to_string()),
        ..Default::default()
    };

    println!("=== fujimura cultural-compare (JP vs EN) ===");
    println!(
        "decision_mode: {} | runs/locale: {} | output: {}",
        decision_mode.label(),
        runs,
        parent.dir().display()
    );
    println!("------------------------------------------------------------");

    // 2 ロケール分で stage を 1 つ．ロケールが変えるのは階層の深さと IVT の
    // 初期平均だけで，1 ステップの仕事の量は同じなので重みではなく数える．
    let mut stage = parent.stage("steps", 2 * runs * args.t_max as usize);

    for locale in [Locale::JaJp, Locale::EnUs] {
        println!("-- locale: {} --", locale.label());
        for run_idx in 0..runs {
            let seed =
                record::cultural_seed(args.seed, locale.default_hierarchy_strength(), run_idx);
            let cfg = Config {
                n_teams: args.n_teams,
                team_size: args.team_size,
                n_levels: locale.default_hierarchy_strength(),
                locale,
                eta: args.eta,
                decision_mode,
                init: InitDist {
                    ivt_mean: locale.default_ivt_mean(),
                    ..InitDist::default()
                },
                t_max: args.t_max,
                runs: 1,
                seed,
                llm: LlmSettings {
                    cache_path: Some(args.cache_path.clone()),
                    ..LlmSettings::default()
                },
                ..Config::default()
            };
            ensure_cache_dir(&cfg, &args.cache_path);
            // ロケールは子 run の parameters に入る．旧 `agent_panel.csv` は JP と
            // EN を 1 ファイルに混ぜたうえロケール列を持たず，どの行がどちらの
            // ものかシードからしか辿れなかった．
            let result = run_replicate(
                "cultural-replicate",
                &args.output_dir,
                &cfg,
                seed,
                run_idx,
                &lineage,
                &mut stage,
            );
            print_run_line(run_idx + 1, runs, seed, &result);
        }
    }
    stage.close();

    let dir = parent.finish().expect("runvault: 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("cultural-compare done. 親 run → {}", dir.display());
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn cmd_reproduce(_args: ReproduceArgs) {
    println!("The SEM β̃ estimation + Fig.1 path-diagram reproduction lives in the Python tooling:");
    println!();
    println!("  uv run fujimura-tools fit-sem");
    println!("  uv run fujimura-tools reproduce");
    println!();
    println!("They pool the replicates' agent_panel.csv, fit the ABM-induced SEM (semopy),");
    println!("estimate the 4 path coefficients (ψ→fear, fear→acquiescent, fear→voice,");
    println!("acquiescent→silence) + CFI/GFI/RMSEA, and reconcile them against the §5 paper");
    println!("anchors (B1–B5).");
    println!();
    println!("Run a simulation first:");
    println!(
        "  cargo run --release -- run --decision-mode rule --locale ja-JP --runs 10 --seed 2019"
    );
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::CulturalCompare(args) => cmd_cultural_compare(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
