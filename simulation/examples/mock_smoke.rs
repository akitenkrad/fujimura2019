//! Offline (no live LLM) smoke: a scripted mock drives the LLM pipeline
//! end-to-end and writes the same outputs as the production `run` (no live LLM
//! dependency — the sandbox cannot reach localhost:11434).
//!
//! Usage:
//!     cargo run --release --example mock_smoke -- results

use fujimura_silence::config::{Config, DecisionMode, InitDist, LlmSettings};
use fujimura_silence::llm::wrap_client;
use fujimura_silence::simulation::{
    ensure_output_dir, llm_meta_json, run_with_client, save_agent_panel, save_metrics,
    write_json_file,
};
use fujimura_silence::world::Locale;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use socsim_results::{timestamp, write_json};

fn main() {
    let base = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "results".to_string());
    let ts = timestamp();
    let output_dir = format!("{}/{}_mock", base, ts);
    ensure_output_dir(&output_dir);

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
        output_dir: output_dir.clone(),
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

    let result = run_with_client(&cfg, Some(client)).expect("mock run failed");
    save_metrics(&result.metrics_rows, &output_dir);
    save_agent_panel(&result.panel_rows, &output_dir);
    write_json(
        &cfg.to_run_config_json(),
        format!("{}/config.json", output_dir),
    )
    .expect("failed to write config.json");
    write_json_file(
        &llm_meta_json(&cfg, &result),
        &format!("{}/llm_meta.json", output_dir),
    );

    println!("mock smoke wrote: {output_dir}");
    println!(
        "LLM calls: {} (cache-hit {:.1}%)",
        result.metadata.total(),
        result.metadata.cache_hit_rate() * 100.0
    );
    let last = result.metrics_rows.last().unwrap();
    println!(
        "final silence={:.3} voice={:.3} corr(s,v)={:+.3}",
        last.silence_rate, last.voice_volume, result.corr_silence_voice
    );
}
