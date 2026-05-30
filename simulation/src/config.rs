//! Simulation configuration for Fujimura & Hino (2019).
//!
//! Holds every knob surfaced by the `run` / `sweep` / `cultural-compare` /
//! `reproduce` CLI: organisation shape (`n_teams`, `team_size`, `n_levels`),
//! the Watts–Strogatz network, locale + cultural structural defaults, the
//! sign-constrained `β` group calibrating the SEM paths (ψ→fear −, fear→voice −,
//! fear→acquiescent +, acquiescent→silence +), retaliation / salience, and the
//! LLM settings used when `decision_mode == Llm`.

use serde::Serialize;

pub use socsim_llm::LlmSettings;

use crate::world::Locale;

// --------------------------------------------------------------------------- //
// DecisionMode — rule-based ablation vs LLM-driven decision
// --------------------------------------------------------------------------- //

/// Decision-mechanism selector. The driver wires **exactly one** of
/// `voice_decision_rule` (`Rule`) and `voice_decision` (`Llm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// `voice_decision_rule` — logistic ablation (LLM-free, bit-deterministic).
    Rule,
    /// `voice_decision` — LLM-driven (Japanese-localised prompt).
    Llm,
}

impl DecisionMode {
    pub fn label(&self) -> &'static str {
        match self {
            DecisionMode::Rule => "rule",
            DecisionMode::Llm => "llm",
        }
    }

    pub fn is_llm(&self) -> bool {
        matches!(self, DecisionMode::Llm)
    }
}

/// Parse a [`DecisionMode`] from a CLI string.
pub fn parse_decision_mode(s: &str) -> Result<DecisionMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rule" | "rules" | "logistic" => Ok(DecisionMode::Rule),
        "llm" | "ollama" | "openai" => Ok(DecisionMode::Llm),
        _ => Err(format!("invalid decision-mode: \"{s}\" (rule / llm)")),
    }
}

// --------------------------------------------------------------------------- //
// PromptVariant — prompt-wording sensitivity (A / B / C)
// --------------------------------------------------------------------------- //

/// Prompt-wording variant for the prompt-sensitivity ablation (design §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptVariant {
    A,
    B,
    C,
}

impl PromptVariant {
    pub fn label(&self) -> &'static str {
        match self {
            PromptVariant::A => "A",
            PromptVariant::B => "B",
            PromptVariant::C => "C",
        }
    }
}

/// Parse a [`PromptVariant`] from a CLI string.
pub fn parse_prompt_variant(s: &str) -> Result<PromptVariant, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "A" => Ok(PromptVariant::A),
        "B" => Ok(PromptVariant::B),
        "C" => Ok(PromptVariant::C),
        _ => Err(format!("invalid prompt-variant: \"{s}\" (A / B / C)")),
    }
}

// --------------------------------------------------------------------------- //
// InitDist — initial latent-state distributions (design §4.3.1 table)
// --------------------------------------------------------------------------- //

/// Mean/SD of the IID-sampled initial latent states (clipped to [0,1]).
///
/// Defaults track the paper's Table 1 M/SD (rescaled from the 7-point Likert to
/// `[0,1]`): ψ N(.613,.139), fear N(.450,.214), acquiescent N(.457,.197),
/// θ N(.40,.15). `ivt_mean` is locale-dependent (set from the [`Locale`]).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct InitDist {
    pub psafety_mean: f64,
    pub psafety_sd: f64,
    pub fear_mean: f64,
    pub fear_sd: f64,
    pub acquiescent_mean: f64,
    pub acquiescent_sd: f64,
    pub ivt_mean: f64,
    pub ivt_sd: f64,
    pub theta_mean: f64,
    pub theta_sd: f64,
}

impl Default for InitDist {
    fn default() -> Self {
        InitDist {
            psafety_mean: 0.613,
            psafety_sd: 0.139,
            fear_mean: 0.450,
            fear_sd: 0.214,
            acquiescent_mean: 0.457,
            acquiescent_sd: 0.197,
            ivt_mean: 0.55,
            ivt_sd: 0.15,
            theta_mean: 0.40,
            theta_sd: 0.15,
        }
    }
}

// --------------------------------------------------------------------------- //
// BetaGroup — sign-constrained coefficients calibrating the 4 SEM paths
// --------------------------------------------------------------------------- //

/// Sign-constrained coefficients used by the deterministic mechanisms (and the
/// `voice_decision_rule` logit). Calibrated so the agent-level cross-section
/// regression `β̃` reproduces the paper's path signs/magnitudes:
/// ψ→fear `−.54`, fear→acquiescent `+.86`, fear→voice `−.23`, acquiescent→silence `+.52`.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BetaGroup {
    // ── fear_appraisal: ψ → fear (paper β = −.54) ───────────────────────────
    /// Fear-update learning rate `α_f`.
    pub alpha_f: f64,
    /// `(1-ψ)` loading on the fear target (drives the negative ψ→fear path).
    pub beta_psafety_fear: f64,
    /// `(1-u)` supervisor-signal loading on the fear target.
    pub beta_supervisor_fear: f64,
    /// Retaliation loading on the fear target.
    pub beta_retaliation_fear: f64,

    // ── acquiescent_update: fear → acquiescent (paper β = +.86) ─────────────
    /// Acquiescent-update learning rate `α_a`.
    pub alpha_a: f64,
    /// Strong fear→acquiescent coupling `γ_fa` (saturating; paper's .86).
    pub gamma_fa: f64,

    // ── silence_spiral: acquiescent → silence (paper β = +.52) ──────────────
    /// Acquiescent loading on the silence logit.
    pub beta_acq_silence: f64,
    /// Neighbour-silence (climate) loading on the silence logit.
    pub beta_climate_silence: f64,
    /// Voice-threshold protection against the cascade.
    pub beta_theta_silence: f64,

    // ── voice_decision_rule logit (fear → voice, paper β = −.23) ────────────
    pub voice_intercept: f64,
    /// `+ψ` loading for VOICE.
    pub beta_psafety_voice: f64,
    /// `+u` supervisor-openness loading for VOICE.
    pub beta_supervisor_voice: f64,
    /// `−fear` loading for VOICE (the fear→voice negative path).
    pub beta_fear_voice: f64,
    /// `−ι` IVT-strength loading for VOICE.
    pub beta_ivt_voice: f64,
    /// `−ρ` neighbour-silence loading for VOICE.
    pub beta_rho_voice: f64,
}

impl Default for BetaGroup {
    fn default() -> Self {
        BetaGroup {
            // fear_appraisal — ψ dominates so ψ→fear is strongly negative.
            alpha_f: 0.45,
            beta_psafety_fear: 0.95,
            beta_supervisor_fear: 0.20,
            beta_retaliation_fear: 0.30,
            // acquiescent_update — strong fear coupling for the .86 path.
            alpha_a: 0.55,
            gamma_fa: 3.4,
            // silence_spiral — acquiescent dominates so acquiescent→silence > 0.
            // Lower magnitude + low intercept leaves a large NEUTRAL buffer so
            // that silence and voice stay (near) uncorrelated across agents (H5).
            beta_acq_silence: 3.0,
            beta_climate_silence: 0.5,
            beta_theta_silence: 1.0,
            // voice_decision_rule — proactive traits (ψ, supervisor openness)
            // drive voice; fear suppresses it (the fear→voice negative path) but
            // not so strongly that voice becomes the mechanical complement of
            // silence. The independent trait basis keeps corr(silence,voice)≈0.
            voice_intercept: -0.35,
            beta_psafety_voice: 0.8,
            beta_supervisor_voice: 1.8,
            beta_fear_voice: 0.8,
            beta_ivt_voice: 0.3,
            beta_rho_voice: 0.3,
        }
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single run.
#[derive(Debug, Clone)]
pub struct Config {
    // ── organisation shape ─────────────────────────────────────────────────
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,

    // ── network ────────────────────────────────────────────────────────────
    /// Watts–Strogatz `k` (mean degree).
    pub network_k: usize,
    /// Watts–Strogatz rewiring `β`.
    pub network_beta: f64,

    // ── locale / cultural structure ────────────────────────────────────────
    pub locale: Locale,
    /// Supervisor-signal homogeneity `η ∈ [0,1]`.
    pub eta: f64,

    // ── decision mode + calibration ────────────────────────────────────────
    pub decision_mode: DecisionMode,
    pub beta: BetaGroup,
    pub init: InitDist,
    pub prompt_variant: PromptVariant,

    // ── exogenous drivers ──────────────────────────────────────────────────
    /// Per-agent per-step retaliation probability.
    pub p_retaliate: f64,
    /// Optional exogenous σ shock time step.
    pub shock_t: Option<u64>,
    /// σ shock magnitude.
    pub shock_magnitude: f64,

    // ── horizon / repeats ──────────────────────────────────────────────────
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,

    // ── LLM settings (used iff `decision_mode == Llm`) ─────────────────────
    pub llm: LlmSettings,

    // ── output ─────────────────────────────────────────────────────────────
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_teams: 5,
            team_size: 80,
            n_levels: 4,
            network_k: 6,
            network_beta: 0.05,
            locale: Locale::JaJp,
            eta: 0.7,
            decision_mode: DecisionMode::Rule,
            beta: BetaGroup::default(),
            init: InitDist::default(),
            prompt_variant: PromptVariant::A,
            p_retaliate: 0.05,
            shock_t: None,
            shock_magnitude: 0.3,
            t_max: 12,
            runs: 1,
            seed: 2019,
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.n_teams.saturating_mul(self.team_size)
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_levels: u8,
    pub n_employees: usize,
    pub network_k: usize,
    pub network_beta: f64,
    pub locale: String,
    pub hierarchy_strength: u8,
    pub eta: f64,
    pub decision_mode: DecisionMode,
    pub prompt_variant: String,
    pub p_retaliate: f64,
    pub shock_t: Option<u64>,
    pub shock_magnitude: f64,
    pub t_max: u64,
    pub runs: usize,
    pub seed: u64,
    pub psafety_mean: f64,
    pub fear_mean: f64,
    pub acquiescent_mean: f64,
    pub ivt_mean: f64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub llm_cache_path: Option<String>,
    pub output_dir: String,
}

impl Config {
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_teams: self.n_teams,
            team_size: self.team_size,
            n_levels: self.n_levels,
            n_employees: self.n_employees(),
            network_k: self.network_k,
            network_beta: self.network_beta,
            locale: self.locale.label().to_string(),
            hierarchy_strength: self.locale.default_hierarchy_strength(),
            eta: self.eta,
            decision_mode: self.decision_mode,
            prompt_variant: self.prompt_variant.label().to_string(),
            p_retaliate: self.p_retaliate,
            shock_t: self.shock_t,
            shock_magnitude: self.shock_magnitude,
            t_max: self.t_max,
            runs: self.runs,
            seed: self.seed,
            psafety_mean: self.init.psafety_mean,
            fear_mean: self.init.fear_mean,
            acquiescent_mean: self.init.acquiescent_mean,
            ivt_mean: self.init.ivt_mean,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            llm_cache_path: self.llm.cache_path.clone(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_decision_mode_variants() {
        assert_eq!(parse_decision_mode("rule").unwrap(), DecisionMode::Rule);
        assert_eq!(parse_decision_mode("LLM").unwrap(), DecisionMode::Llm);
        assert!(parse_decision_mode("bogus").is_err());
    }

    #[test]
    fn parse_prompt_variant_variants() {
        assert_eq!(parse_prompt_variant("a").unwrap(), PromptVariant::A);
        assert_eq!(parse_prompt_variant("C").unwrap(), PromptVariant::C);
        assert!(parse_prompt_variant("Z").is_err());
    }

    #[test]
    fn default_config_is_jp_baseline() {
        let c = Config::default();
        assert_eq!(c.locale, Locale::JaJp);
        assert_eq!(c.n_employees(), 400);
    }
}
