//! Offline (no live LLM) smoke: a scripted mock drives the LLM pipeline
//! end-to-end and records the same run shape as the production `run` (no live
//! LLM dependency — the sandbox cannot reach localhost:11434).
//!
//! 反復は 1 本なので親 run は作らない．`run` の子 (`run-replicate`) とは
//! subcommand 名で区別する — 中身は «scripted client で駆動した 1 本» であって，
//! 本番の LLM 経路とは別物である．
//!
//! Usage:
//!     cargo run --release --example mock_smoke -- results

use fujimura_silence::config::{Config, DecisionMode, InitDist, LlmSettings};
use fujimura_silence::llm::wrap_client;
use fujimura_silence::record::{
    self, ConditionParameters, ReplicateParameters, DOMAIN, EXPERIMENT, HASH_EXCLUDE,
    REPLICATE_SEED_POINTERS, REPO_ID,
};
use fujimura_silence::simulation::{run_with_client, save_agent_panel};
use fujimura_silence::world::Locale;

use runvault::{Run, RunOptions};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

fn main() {
    let results_root = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results".to_string());

    let cfg = Config {
        n_teams: 4,
        team_size: 20,
        n_levels: 4,
        locale: Locale::JaJp,
        decision_mode: DecisionMode::Llm,
        t_max: 12,
        runs: 1,
        seed: 2019,
        init: InitDist::default(),
        llm: LlmSettings::default(), // cache_path = None → in-memory; no save()
        output_dir: String::new(),
        ..Config::default()
    };

    // Scripted backend: psychologically-safe-looking prompts → VOICE; fearful
    // ones → SILENCE with a motive read off the prompt's fear level.
    let backend = ScriptedClient::new("mock-fujimura", |prompt: &str| {
        let fearful = prompt.contains("怖れ動機の現在水準は 0.6")
            || prompt.contains("怖れ動機の現在水準は 0.7")
            || prompt.contains("怖れ動機の現在水準は 0.8");
        if fearful {
            r#"{"decision":"SILENCE","motive":"quiescent","rationale":"報復が怖い"}"#.to_string()
        } else if prompt.contains("黙る傾向") {
            r#"{"decision":"SILENCE","motive":"acquiescent","rationale":"言っても無駄"}"#
                .to_string()
        } else {
            r#"{"decision":"VOICE","motive":null,"rationale":"改善したい"}"#.to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&results_root)
            .parameters(&ReplicateParameters {
                condition: ConditionParameters::from_config(&cfg),
                seed: cfg.seed,
            })
            .expect("runvault: parameters の組み立てに失敗")
            .hash_exclude(HASH_EXCLUDE)
            .seed_pointers(REPLICATE_SEED_POINTERS)
            .master_seed(cfg.seed)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, Some(client)).expect("mock run failed");

    record::log_simulation(&mut rv, &result);
    record::log_llm_usage(&mut rv, &result);
    record::log_paper_reference(&mut rv);
    save_agent_panel(
        &result.panel_rows,
        &rv.dir().join("artifacts").to_string_lossy(),
    );

    let last = result.metrics_rows.last().unwrap().clone();
    let calls = result.metadata.total();
    let hit_rate = result.metadata.cache_hit_rate();
    let corr = result.corr_silence_voice;

    let dir = rv.finish().expect("runvault: run の完了に失敗");

    println!("mock smoke wrote: {}", dir.display());
    println!("LLM calls: {} (cache-hit {:.1}%)", calls, hit_rate * 100.0);
    println!(
        "final silence={:.3} voice={:.3} corr(s,v)={:+.3}",
        last.silence_rate, last.voice_volume, corr
    );
}
