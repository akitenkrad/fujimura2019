//! Integration tests for the Fujimura & Hino (2019) silence-and-voice simulation.
//!
//! **No live LLM required.** Rule mode needs no LLM; the LLM path is driven by
//! `socsim_llm::mock::ScriptedClient`. Tests cover rule-mode end-to-end
//! bit-determinism, the LLM path via a scripted client, init-distribution
//! ranges, and the silence ⊥ voice independence (panel flags can co-vary).

use fujimura_silence::config::{Config, DecisionMode, InitDist, LlmSettings};
use fujimura_silence::llm::wrap_client;
use fujimura_silence::simulation::{run_with_client, SimulationResult};
use fujimura_silence::world::Locale;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn small_rule_cfg() -> Config {
    Config {
        n_teams: 3,
        team_size: 12,
        n_levels: 4,
        network_k: 4,
        network_beta: 0.05,
        locale: Locale::JaJp,
        eta: 0.7,
        decision_mode: DecisionMode::Rule,
        t_max: 8,
        runs: 1,
        seed: 2019,
        init: InitDist::default(),
        llm: LlmSettings::default(),
        output_dir: "results".to_string(),
        ..Config::default()
    }
}

fn small_llm_cfg() -> Config {
    Config {
        decision_mode: DecisionMode::Llm,
        ..small_rule_cfg()
    }
}

/// Scripted client cycling VOICE / SILENCE(quiescent) / SILENCE(acquiescent).
fn scripted_client() -> fujimura_silence::llm::SilenceClient {
    let backend = ScriptedClient::new("mock-fujimura", |prompt: &str| {
        let h = prompt.len() % 3;
        match h {
            0 => r#"{"decision":"VOICE","motive":null,"rationale":"speak"}"#.to_string(),
            1 => r#"{"decision":"SILENCE","motive":"quiescent","rationale":"怖い"}"#.to_string(),
            _ => r#"{"decision":"SILENCE","motive":"acquiescent","rationale":"無駄"}"#.to_string(),
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

// --------------------------------------------------------------------------- //
// Rule mode
// --------------------------------------------------------------------------- //

#[test]
fn rule_mode_smoke_run() {
    let r: SimulationResult = run_with_client(&small_rule_cfg(), None).unwrap();
    assert!(!r.metrics_rows.is_empty(), "must produce per-step metrics");
    assert_eq!(r.metadata.total(), 0, "rule mode makes 0 LLM calls");
    for row in &r.metrics_rows {
        assert!((0.0..=1.0).contains(&row.silence_rate));
        assert!((0.0..=1.0).contains(&row.voice_volume));
        let s = row.motive_mix_acquiescent
            + row.motive_mix_quiescent
            + row.motive_mix_prosocial
            + row.motive_mix_opportunistic;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
    }
}

#[test]
fn rule_mode_is_bit_deterministic() {
    let a = run_with_client(&small_rule_cfg(), None).unwrap();
    let b = run_with_client(&small_rule_cfg(), None).unwrap();
    assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
    for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
        assert_eq!(ra.t, rb.t);
        assert!((ra.silence_rate - rb.silence_rate).abs() < 1e-15);
        assert!((ra.voice_volume - rb.voice_volume).abs() < 1e-15);
        assert!((ra.climate_of_silence - rb.climate_of_silence).abs() < 1e-15);
        assert!((ra.motive_mix_acquiescent - rb.motive_mix_acquiescent).abs() < 1e-15);
    }
    assert_eq!(a.panel_rows.len(), b.panel_rows.len());
    for (pa, pb) in a.panel_rows.iter().zip(b.panel_rows.iter()) {
        assert_eq!(pa.agent_id, pb.agent_id);
        assert_eq!(pa.voice, pb.voice);
        assert_eq!(pa.silence, pb.silence);
        assert!((pa.fear - pb.fear).abs() < 1e-15);
    }
}

#[test]
fn init_distributions_in_range_and_levels_valid() {
    use fujimura_silence::simulation::init_world;
    let cfg = small_rule_cfg();
    let w = init_world(&cfg, cfg.seed);
    for e in w.employees.values() {
        for v in [
            e.psych_safety,
            e.fear,
            e.acquiescent,
            e.ivt_strength,
            e.voice_threshold,
        ] {
            assert!((0.0..=1.0).contains(&v));
        }
        assert!(e.level >= 1 && e.level <= w.hierarchy_strength);
    }
}

#[test]
fn voice_and_silence_are_independent_flags() {
    // An agent may be voice in some steps and silence in others — they are not
    // the mechanical complement of each other (paper H5).
    let r = run_with_client(&small_rule_cfg(), None).unwrap();
    // voice and silence are two independent behavioural channels: an employee
    // may voice on one concern and withhold on another in the same step, so
    // both flags can be 1 together (this is exactly what keeps them weakly
    // correlated — paper H5). The panel records them as separate columns.
    let has_voice = r.panel_rows.iter().any(|row| row.voice == 1);
    let has_silence = r.panel_rows.iter().any(|row| row.silence == 1);
    assert!(has_voice && has_silence, "both channels must be exercised");
}

// --------------------------------------------------------------------------- //
// LLM mode (mock; no live LLM)
// --------------------------------------------------------------------------- //

#[test]
fn llm_mode_smoke_with_scripted_client() {
    let cfg = small_llm_cfg();
    let client = scripted_client();
    let r = run_with_client(&cfg, Some(client)).unwrap();
    assert!(!r.metrics_rows.is_empty());
    assert!(r.metadata.total() > 0, "LLM mode must call the LLM");
    // The scripted client always returns valid JSON → 0 parse failures.
    assert_eq!(
        r.parse_fail.0, 0,
        "no parse failures expected from scripted client"
    );
    for row in &r.metrics_rows {
        let s = row.motive_mix_acquiescent
            + row.motive_mix_quiescent
            + row.motive_mix_prosocial
            + row.motive_mix_opportunistic;
        assert!(s.abs() < 1e-9 || (s - 1.0).abs() < 1e-9);
    }
}

#[test]
fn en_locale_runs() {
    let cfg = Config {
        locale: Locale::EnUs,
        n_levels: 3,
        ..small_rule_cfg()
    };
    let r = run_with_client(&cfg, None).unwrap();
    assert!(!r.metrics_rows.is_empty());
}
