//! Fujimura & Hino (2019) — Silence and voice in the organization CLI.
//!
//! `run`              : single configuration; `--decision-mode {rule|llm}` exclusive switch.
//! `sweep`            : Cartesian product over `n_levels × η × network_beta × seeds`.
//! `cultural-compare` : runs the JP and EN locales side by side (rule or LLM).
//! `reproduce`        : pointer to the Python `fit-sem` / `reproduce` tooling.

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use fujimura_silence::config::{
    parse_decision_mode, parse_prompt_variant, Config, InitDist, LlmSettings,
};
use fujimura_silence::simulation::{
    ensure_output_dir, llm_meta_json, run, save_agent_panel, save_metrics, write_json_file,
    SimulationResult,
};
use fujimura_silence::world::{parse_locale, Locale};

use socsim_core::derive_seed;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

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
    /// Sweep n_levels × η × network_beta across seeds; aggregate to `sweep_summary.csv`.
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
    /// Number of independent runs (different seeds; outputs are pooled).
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
    /// Output base directory.
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
    /// Output base directory.
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
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Output base directory.
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
        output_dir: args.output_dir.clone(),
    }
}

fn print_run_line(idx: usize, n: usize, r: &SimulationResult) {
    let last = r.metrics_rows.last();
    println!(
        "[{}/{}] seed={} silence={:.3} voice={:.3} C={:.3} corr(s,v)={:+.3}",
        idx,
        n,
        r.seed,
        last.map(|m| m.silence_rate).unwrap_or(0.0),
        last.map(|m| m.voice_volume).unwrap_or(0.0),
        last.map(|m| m.climate_of_silence).unwrap_or(0.0),
        r.corr_silence_voice,
    );
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let ts = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, ts);
    ensure_output_dir(&output_dir);

    let mut base_cfg = cfg_from_run_args(&args);
    base_cfg.output_dir = output_dir.clone();
    if base_cfg.decision_mode.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

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
        "t_max={} runs={} seed={} | output: {output_dir}",
        base_cfg.t_max, base_cfg.runs, base_cfg.seed
    );
    println!("----------------------------------------------------------------------");

    write_json(
        &base_cfg.to_run_config_json(),
        format!("{output_dir}/config.json"),
    )
    .expect("failed to write config.json");

    let runs = base_cfg.runs.max(1);
    let mut all_metrics = Vec::new();
    let mut all_panel = Vec::new();
    let mut last_result: Option<SimulationResult> = None;
    for run_idx in 0..runs {
        let seed = derive_seed(base_cfg.seed, &[run_idx as u64]);
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
        print_run_line(run_idx + 1, runs, &result);
        all_metrics.extend(result.metrics_rows.clone());
        all_panel.extend(result.panel_rows.clone());
        last_result = Some(result);
    }

    let result = last_result.expect("at least one run");
    save_metrics(&all_metrics, &output_dir);
    save_agent_panel(&all_panel, &output_dir);
    let meta = llm_meta_json(&base_cfg, &result);
    write_json_file(&meta, &format!("{output_dir}/llm_meta.json"));

    let _ = refresh_latest_symlink(&args.output_dir, &ts);

    println!("----------------------------------------------------------------------");
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("agent_panel → {output_dir}/agent_panel.csv");
    println!("metrics     → {output_dir}/metrics.csv");
    println!("llm_meta    → {output_dir}/llm_meta.json");
    println!("config      → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    decision_mode: String,
    locale: String,
    n_levels: u8,
    eta: f64,
    network_beta: f64,
    run: usize,
    seed: u64,
    final_round: u64,
    silence_rate: f64,
    voice_volume: f64,
    climate_of_silence: f64,
    corr_silence_voice: f64,
    motive_mix_acquiescent: f64,
    motive_mix_quiescent: f64,
}

fn cmd_sweep(args: SweepArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let locale = parse_locale(&args.locale).unwrap_or_else(|e| panic!("{e}"));
    let ts = timestamp();
    let dir_name = format!("{ts}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");

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
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    let config_json = serde_json::json!({
        "command": "sweep",
        "decision_mode": decision_mode.label(),
        "locale": locale.label(),
        "n_levels_values": n_levels_vals,
        "eta_values": eta_vals,
        "network_beta_values": beta_vals,
        "n_teams": args.n_teams,
        "team_size": args.team_size,
        "runs": args.runs,
        "t_max": args.t_max,
        "seed": args.seed,
    });
    write_json(&config_json, format!("{sweep_dir}/sweep_config.json"))
        .expect("failed to write sweep_config.json");

    let mut rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut idx = 0usize;
    for &nl in &n_levels_vals {
        for &eta in &eta_vals {
            for &nb in &beta_vals {
                for run_idx in 0..args.runs {
                    idx += 1;
                    let seed = derive_seed(
                        args.seed,
                        &[
                            nl as u64,
                            (eta * 1000.0) as u64,
                            (nb * 1000.0) as u64,
                            run_idx as u64,
                        ],
                    );
                    let cfg = Config {
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
                        runs: 1,
                        seed,
                        llm: LlmSettings {
                            cache_path: Some(".llm_cache/cache.json".to_string()),
                            ..LlmSettings::default()
                        },
                        ..Config::default()
                    };
                    let result = run(&cfg).unwrap_or_else(|e| panic!("sweep run failed: {e}"));
                    let last = result
                        .metrics_rows
                        .last()
                        .expect("metrics must not be empty");
                    rows.push(SweepRow {
                        decision_mode: decision_mode.label().to_string(),
                        locale: locale.label().to_string(),
                        n_levels: nl,
                        eta,
                        network_beta: nb,
                        run: run_idx,
                        seed,
                        final_round: result.final_round,
                        silence_rate: last.silence_rate,
                        voice_volume: last.voice_volume,
                        climate_of_silence: last.climate_of_silence,
                        corr_silence_voice: result.corr_silence_voice,
                        motive_mix_acquiescent: last.motive_mix_acquiescent,
                        motive_mix_quiescent: last.motive_mix_quiescent,
                    });
                    if idx.is_multiple_of(20) || idx == n_total {
                        println!(
                            "[{}/{}] L={} η={:.2} β={:.2} run={} silence={:.3}",
                            idx, n_total, nl, eta, nb, run_idx, last.silence_rate
                        );
                    }
                }
            }
        }
    }

    write_csv(&rows, format!("{sweep_dir}/sweep_summary.csv"))
        .expect("failed to write sweep_summary.csv");
    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
}

// --------------------------------------------------------------------------- //
// cultural-compare
// --------------------------------------------------------------------------- //

fn cmd_cultural_compare(args: CulturalCompareArgs) {
    let decision_mode = parse_decision_mode(&args.decision_mode).unwrap_or_else(|e| panic!("{e}"));
    let ts = timestamp();
    let output_dir = format!("{}/{}_cultural", args.output_dir, ts);
    ensure_output_dir(&output_dir);
    if decision_mode.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    println!("=== fujimura cultural-compare (JP vs EN) ===");
    println!(
        "decision_mode: {} | runs/locale: {} | output: {output_dir}",
        decision_mode.label(),
        args.runs
    );
    println!("------------------------------------------------------------");

    let mut all_panel = Vec::new();
    let mut all_metrics = Vec::new();
    let mut last_result: Option<SimulationResult> = None;
    for locale in [Locale::JaJp, Locale::EnUs] {
        println!("-- locale: {} --", locale.label());
        for run_idx in 0..args.runs.max(1) {
            let seed = derive_seed(
                args.seed,
                &[locale.default_hierarchy_strength() as u64, run_idx as u64],
            );
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
            let result = run(&cfg).unwrap_or_else(|e| panic!("cultural-compare run failed: {e}"));
            print_run_line(run_idx + 1, args.runs.max(1), &result);
            // Tag panel rows by encoding locale into the seed column is not ideal;
            // instead write per-locale subdirs are overkill — pool with locale in metrics.
            all_panel.extend(result.panel_rows.clone());
            all_metrics.extend(result.metrics_rows.clone());
            last_result = Some(result);
        }
    }

    save_agent_panel(&all_panel, &output_dir);
    save_metrics(&all_metrics, &output_dir);
    let config_json = serde_json::json!({
        "command": "cultural-compare",
        "decision_mode": decision_mode.label(),
        "locales": ["ja-JP", "en-US"],
        "n_teams": args.n_teams,
        "team_size": args.team_size,
        "eta": args.eta,
        "runs": args.runs,
        "t_max": args.t_max,
        "seed": args.seed,
    });
    write_json(&config_json, format!("{output_dir}/config.json"))
        .expect("failed to write config.json");
    if let Some(r) = &last_result {
        let cfg = Config {
            decision_mode,
            ..Config::default()
        };
        write_json_file(
            &llm_meta_json(&cfg, r),
            &format!("{output_dir}/llm_meta.json"),
        );
    }
    let _ = refresh_latest_symlink(&args.output_dir, &format!("{ts}_cultural"));
    println!("------------------------------------------------------------");
    println!("cultural-compare done. agent_panel → {output_dir}/agent_panel.csv");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

fn cmd_reproduce(_args: ReproduceArgs) {
    println!("The SEM β̃ estimation + Fig.1 path-diagram reproduction lives in the Python tooling:");
    println!();
    println!("  uv run fujimura-tools fit-sem    --results-dir results/latest");
    println!("  uv run fujimura-tools reproduce  --results-dir results/latest");
    println!();
    println!("They fit the ABM-induced SEM to agent_panel.csv (semopy), estimate the 4 path");
    println!("coefficients (ψ→fear, fear→acquiescent, fear→voice, acquiescent→silence) + CFI/GFI/");
    println!("RMSEA, and reconcile them against the §5 paper anchors (B1–B5).");
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
