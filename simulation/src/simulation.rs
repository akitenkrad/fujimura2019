//! Initialisation + run driver (`SimulationBuilder` wiring).
//!
//! Two-layer determinism:
//! - **lower (deterministic socsim core)** — `derive_seed(root, &[RNG_WORLD_INIT])`
//!   seeds the IID latent-state sampling, `&[RNG_NETWORK]` the Watts–Strogatz
//!   network, `&[RNG_ENGINE]` the engine (scheduler / Bernoulli draws).
//!   Bit-reproducible in rule mode.
//! - **upper (non-deterministic LLM)** — confined to `voice_decision` via
//!   `socsim-llm`'s cached Ollama → OpenAI client. `temperature=0` +
//!   `(agent_id, t)`-derived seed + prompt→response cache pseudo-determinise it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::rc::Rc;

use csv::Writer;
use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_net::SocialNetwork;

use crate::config::{Config, DecisionMode};
use crate::llm::{build_live_client, SilenceClient};
use crate::mechanisms::{
    AcquiescentUpdate, ClimateSilence, FearAppraisal, IssueSalience, PsafetyUpdate,
    RetaliationEvent, SharedClient, SharedMetadata, SharedParseFail, SilenceSpiral,
    VoiceDecisionLlm, VoiceDecisionRule,
};
use crate::metrics::{step_metrics, StepMetrics};
use crate::world::{Employee, SilenceWorld, Team};

/// RNG stream label: world init (latent-state sampling).
pub const RNG_WORLD_INIT: u64 = 0;
/// RNG stream label: socsim engine (scheduler / Bernoulli draws).
pub const RNG_ENGINE: u64 = 1;
/// RNG stream label: Watts–Strogatz network generation.
pub const RNG_NETWORK: u64 = 2;
/// RNG stream label: LLM `(agent_id, t)` seed.
pub const RNG_LLM_ROOT: u64 = 3;

// --------------------------------------------------------------------------- //
// Result containers
// --------------------------------------------------------------------------- //

/// One long-format `agent_panel.csv` row.
#[derive(Debug, Clone, Serialize)]
pub struct AgentPanelRow {
    pub seed: u64,
    pub t: u64,
    pub agent_id: u64,
    pub psafety: f64,
    pub fear: f64,
    pub acquiescent: f64,
    pub voice: u8,
    pub silence: u8,
    pub motive: String,
}

/// One `metrics.csv` row.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsRow {
    pub seed: u64,
    pub t: u64,
    pub silence_rate: f64,
    pub voice_volume: f64,
    pub climate_of_silence: f64,
    pub motive_mix_acquiescent: f64,
    pub motive_mix_quiescent: f64,
    pub motive_mix_prosocial: f64,
    pub motive_mix_opportunistic: f64,
}

/// Result of a single run.
pub struct SimulationResult {
    pub final_round: u64,
    pub seed: u64,
    pub metrics_rows: Vec<MetricsRow>,
    pub panel_rows: Vec<AgentPanelRow>,
    pub corr_silence_voice: f64,
    pub metadata: MetadataCollector,
    pub parse_fail: (usize, usize),
    pub llm_model: String,
    pub llm_endpoint: String,
}

// --------------------------------------------------------------------------- //
// World initialisation
// --------------------------------------------------------------------------- //

/// Sample a clipped-Normal value in [0,1].
fn sample_clipped_normal(rng: &mut SimRng, mean: f64, sd: f64) -> f64 {
    // Box–Muller using two uniforms from the deterministic stream.
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (mean + sd * z).clamp(0.0, 1.0)
}

/// Initialise a [`SilenceWorld`] with per-employee latent states drawn IID from
/// the design §4.3.1 distributions, plus a Watts–Strogatz network.
pub fn init_world(cfg: &Config, root: u64) -> SilenceWorld {
    let n = cfg.n_employees();
    let l = cfg.locale.default_hierarchy_strength();
    let mut rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));

    let mut employees: BTreeMap<AgentId, Employee> = BTreeMap::new();
    for i in 0..n {
        let team = i / cfg.team_size.max(1);
        // Level 1..=L, weighted toward the bottom (pyramidal).
        let level = (1 + (i % l.max(1) as usize)) as u8;
        let tenure: u32 = rng.gen_range(1..240);
        let mut e = Employee::neutral(team, level, tenure);
        e.psych_safety =
            sample_clipped_normal(&mut rng, cfg.init.psafety_mean, cfg.init.psafety_sd);
        e.fear = sample_clipped_normal(&mut rng, cfg.init.fear_mean, cfg.init.fear_sd);
        e.acquiescent =
            sample_clipped_normal(&mut rng, cfg.init.acquiescent_mean, cfg.init.acquiescent_sd);
        e.ivt_strength = sample_clipped_normal(&mut rng, cfg.init.ivt_mean, cfg.init.ivt_sd);
        e.voice_threshold = sample_clipped_normal(&mut rng, cfg.init.theta_mean, cfg.init.theta_sd);
        employees.insert(AgentId(i as u64), e);
    }

    // Teams: supervisor openness homogenised by η around a shared mean signal.
    let shared_signal: f64 = rng.gen_range(-0.3..0.5);
    let mut teams = Vec::with_capacity(cfg.n_teams);
    for _ in 0..cfg.n_teams {
        let idiosyncratic: f64 = rng.gen_range(-0.6..0.6);
        let u = cfg.eta * shared_signal + (1.0 - cfg.eta) * idiosyncratic;
        teams.push(Team {
            supervisor_openness: u.clamp(-1.0, 1.0),
        });
    }

    // Network (separate RNG stream).
    let mut net_rng = SimRng::from_seed(derive_seed(root, &[RNG_NETWORK]));
    let ids: Vec<AgentId> = (0..n).map(|i| AgentId(i as u64)).collect();
    let network =
        SocialNetwork::watts_strogatz(&ids, cfg.network_k.max(2), cfg.network_beta, &mut net_rng);

    SilenceWorld {
        clock: SimClock::new(cfg.t_max),
        employees,
        teams,
        network,
        issue_salience: 0.5,
        climate_of_silence: 0.0,
        locale: cfg.locale,
        hierarchy_strength: l,
        eta: cfg.eta,
        retaliation_this_step: Vec::new(),
    }
}

// --------------------------------------------------------------------------- //
// Run driver
// --------------------------------------------------------------------------- //

/// Build the production LLM client (LLM mode) or run rule mode directly.
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    if cfg.decision_mode.is_llm() {
        let client =
            build_live_client(&cfg.llm).map_err(|e| format!("LLM client build failed: {e}"))?;
        run_with_client(cfg, Some(client))
    } else {
        run_with_client(cfg, None)
    }
}

/// Run with an optional pre-built [`SilenceClient`] — production via
/// [`build_live_client`], tests via [`crate::llm::wrap_client`] over a mock.
pub fn run_with_client(
    cfg: &Config,
    client: Option<SilenceClient>,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed;
    let world = init_world(cfg, root);

    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_parse_fail: SharedParseFail = Rc::new(RefCell::new((0, 0)));
    let (llm_model, llm_endpoint, shared_client): (String, String, Option<SharedClient>) =
        match client {
            Some(c) => {
                let model = c.inner().model().to_string();
                let endpoint = c.inner().endpoint().to_string();
                (model, endpoint, Some(Rc::new(RefCell::new(c))))
            }
            None => ("none".to_string(), "none".to_string(), None),
        };

    let mut builder = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]));

    // Environment
    builder = builder.add_mechanism(Box::new(IssueSalience::new(
        cfg.shock_t,
        cfg.shock_magnitude,
    )));
    builder = builder.add_mechanism(Box::new(RetaliationEvent::new(cfg.p_retaliate)));

    // Decision
    builder = builder.add_mechanism(Box::new(FearAppraisal::new(cfg.beta)));
    match (cfg.decision_mode, &shared_client) {
        (DecisionMode::Rule, _) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionRule::new(cfg.beta)));
        }
        (DecisionMode::Llm, Some(sc)) => {
            builder = builder.add_mechanism(Box::new(VoiceDecisionLlm::new(
                Rc::clone(sc),
                Rc::clone(&shared_meta),
                Rc::clone(&shared_parse_fail),
                cfg.llm.clone(),
                cfg.prompt_variant,
                derive_seed(root, &[RNG_LLM_ROOT]),
            )));
        }
        (DecisionMode::Llm, None) => {
            return Err("LLM decision mode selected but no client supplied".to_string());
        }
    }

    // Interaction
    builder = builder.add_mechanism(Box::new(AcquiescentUpdate::new(cfg.beta)));
    builder = builder.add_mechanism(Box::new(SilenceSpiral::new(cfg.beta)));

    // PostStep
    builder = builder.add_mechanism(Box::new(PsafetyUpdate::new()));
    builder = builder.add_mechanism(Box::new(ClimateSilence));

    let mut sim = builder.build();

    let mut metrics_rows: Vec<MetricsRow> = Vec::new();
    let mut panel_rows: Vec<AgentPanelRow> = Vec::new();

    // Record initial state (t=0).
    record_step(sim.world(), 0, root, &mut metrics_rows, &mut panel_rows);

    let mut final_round = 0u64;
    sim.run_observed(|report| {
        let t = report.t;
        record_step(report.world, t, root, &mut metrics_rows, &mut panel_rows);
        final_round = t;
    })
    .map_err(|e| format!("simulation run failed: {e}"))?;

    // Persist LLM cache (file-backed only; load-on-open / save-on-demand).
    if let Some(sc) = &shared_client {
        if cfg.llm.cache_path.is_some() {
            sc.borrow()
                .cache()
                .save()
                .map_err(|e| format!("cache save failed: {e}"))?;
        }
    }

    // corr(silence_i, voice_i) over per-agent time-averaged flags.
    let corr_silence_voice = compute_corr_silence_voice(&panel_rows);

    let metadata = shared_meta.borrow().clone();
    let parse_fail = *shared_parse_fail.borrow();
    Ok(SimulationResult {
        final_round,
        seed: root,
        metrics_rows,
        panel_rows,
        corr_silence_voice,
        metadata,
        parse_fail,
        llm_model,
        llm_endpoint,
    })
}

fn record_step(
    world: &SilenceWorld,
    t: u64,
    seed: u64,
    metrics_rows: &mut Vec<MetricsRow>,
    panel_rows: &mut Vec<AgentPanelRow>,
) {
    let m: StepMetrics = step_metrics(world);
    metrics_rows.push(MetricsRow {
        seed,
        t,
        silence_rate: m.silence_rate,
        voice_volume: m.voice_volume,
        climate_of_silence: m.climate_of_silence,
        motive_mix_acquiescent: m.motive_mix[0],
        motive_mix_quiescent: m.motive_mix[1],
        motive_mix_prosocial: m.motive_mix[2],
        motive_mix_opportunistic: m.motive_mix[3],
    });
    for (&id, emp) in &world.employees {
        panel_rows.push(AgentPanelRow {
            seed,
            t,
            agent_id: id.0,
            psafety: emp.psych_safety,
            fear: emp.fear,
            acquiescent: emp.acquiescent,
            voice: u8::from(emp.voiced),
            silence: u8::from(emp.silenced),
            motive: emp
                .silence_motive
                .map(|mt| mt.label().to_string())
                .unwrap_or_else(|| "-".to_string()),
        });
    }
}

/// Pearson r between per-agent time-averaged silence and voice flags (H5).
fn compute_corr_silence_voice(rows: &[AgentPanelRow]) -> f64 {
    use std::collections::BTreeMap as Map;
    let mut sil: Map<u64, (f64, u64)> = Map::new();
    let mut voi: Map<u64, (f64, u64)> = Map::new();
    for r in rows {
        if r.t == 0 {
            continue; // skip the all-neutral initial step
        }
        let s = sil.entry(r.agent_id).or_insert((0.0, 0));
        s.0 += r.silence as f64;
        s.1 += 1;
        let v = voi.entry(r.agent_id).or_insert((0.0, 0));
        v.0 += r.voice as f64;
        v.1 += 1;
    }
    let mut sv: Vec<f64> = Vec::new();
    let mut vv: Vec<f64> = Vec::new();
    for (id, (ssum, scnt)) in &sil {
        if let Some((vsum, vcnt)) = voi.get(id) {
            if *scnt > 0 && *vcnt > 0 {
                sv.push(ssum / *scnt as f64);
                vv.push(vsum / *vcnt as f64);
            }
        }
    }
    crate::metrics::pearson(&sv, &vv)
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

/// Create the output directory.
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("failed to create output directory");
}

/// Write `agent_panel.csv` (long format).
pub fn save_agent_panel(rows: &[AgentPanelRow], output_dir: &str) {
    let path = format!("{output_dir}/agent_panel.csv");
    let file = File::create(&path).expect("failed to create agent_panel.csv");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for r in rows {
        wtr.serialize(r).expect("failed to write agent_panel row");
    }
    wtr.flush().expect("failed to flush agent_panel.csv");
}

/// Write `metrics.csv`.
pub fn save_metrics(rows: &[MetricsRow], output_dir: &str) {
    let path = format!("{output_dir}/metrics.csv");
    let file = File::create(&path).expect("failed to create metrics.csv");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for r in rows {
        wtr.serialize(r).expect("failed to write metrics row");
    }
    wtr.flush().expect("failed to flush metrics.csv");
}

/// Build the `llm_meta.json` value from a result's [`MetadataCollector`].
pub fn llm_meta_json(cfg: &Config, result: &SimulationResult) -> serde_json::Value {
    let (fails, total) = result.parse_fail;
    let parse_fail_rate = if total > 0 {
        fails as f64 / total as f64
    } else {
        0.0
    };
    serde_json::json!({
        "decision_mode": cfg.decision_mode.label(),
        "model": result.llm_model,
        "endpoint": result.llm_endpoint,
        "temperature": cfg.llm.temperature,
        "seed": result.seed,
        "calls": result.metadata.total(),
        "cache_hits": result.metadata.cache_hits(),
        "cache_hit_rate": result.metadata.cache_hit_rate(),
        "parse_failures": fails,
        "parse_fail_rate": parse_fail_rate,
        "determinism_note": "LLM output is outside socsim bit-reproducibility; the prompt->response \
                             cache (temperature=0, (agent_id, t)-derived seed) is the reproducibility \
                             mechanism. The deterministic socsim core (init, network, scheduling, the \
                             7 non-LLM mechanisms) is bit-reproducible given the seed; rule mode makes \
                             zero LLM calls."
    })
}

/// Write an arbitrary JSON value to a file.
pub fn write_json_file(value: &serde_json::Value, path: &str) {
    let file = File::create(path).expect("failed to create JSON file");
    let mut w = BufWriter::new(file);
    let s = serde_json::to_string_pretty(value).expect("failed to serialise JSON");
    w.write_all(s.as_bytes()).expect("failed to write JSON");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Locale;

    fn small_cfg() -> Config {
        Config {
            n_teams: 3,
            team_size: 10,
            n_levels: 4,
            network_k: 4,
            network_beta: 0.05,
            locale: Locale::JaJp,
            t_max: 6,
            runs: 1,
            seed: 2019,
            ..Config::default()
        }
    }

    #[test]
    fn rule_run_is_deterministic() {
        let a = run_with_client(&small_cfg(), None).unwrap();
        let b = run_with_client(&small_cfg(), None).unwrap();
        assert_eq!(a.metrics_rows.len(), b.metrics_rows.len());
        for (ra, rb) in a.metrics_rows.iter().zip(b.metrics_rows.iter()) {
            assert!((ra.silence_rate - rb.silence_rate).abs() < 1e-15);
            assert!((ra.voice_volume - rb.voice_volume).abs() < 1e-15);
        }
        assert_eq!(a.metadata.total(), 0, "rule mode makes 0 LLM calls");
    }

    #[test]
    fn init_distributions_in_range() {
        let w = init_world(&small_cfg(), 2019);
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
    fn panel_has_initial_and_per_step_rows() {
        let cfg = small_cfg();
        let r = run_with_client(&cfg, None).unwrap();
        let n = cfg.n_employees() as u64;
        // (t_max + 1) snapshots × n agents.
        assert_eq!(r.panel_rows.len() as u64, (cfg.t_max + 1) * n);
    }
}
