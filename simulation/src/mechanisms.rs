//! 8 mechanisms across the socsim 6-phase loop.
//!
//! | # | Mechanism            | Phase        | Role (paper SEM path) |
//! |---|----------------------|--------------|-----------------------|
//! | 1 | `IssueSalience`      | Environment  | Update σ(t); optional shock at `shock_t` |
//! | 2 | `RetaliationEvent`   | Environment  | Probabilistic retaliation → `retaliation_this_step` |
//! | 3 | `FearAppraisal`      | Decision     | ψ → fear (paper β = −.54) |
//! | 4 | `VoiceDecisionRule` / `VoiceDecisionLlm` | Decision | **★ exclusive**: fear → voice (β = −.23) + 4-motive |
//! | 5 | `AcquiescentUpdate`  | Interaction  | fear → acquiescent (paper β = +.86, saturating) |
//! | 6 | `SilenceSpiral`      | Interaction  | acquiescent → silence (paper β = +.52, neighbour ρ_i) |
//! | 7 | `PsafetyUpdate`      | PostStep     | voice/no-retaliation slowly raises ψ |
//! | 8 | `ClimateSilence`     | PostStep     | aggregate C(t) for next-step Decision |
//!
//! The LLM call is confined to `voice_decision`; everything else is
//! deterministic. Decision mechanisms snapshot all employees at step start and
//! apply the new expressions/motives from the snapshot (synchronous update).

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;
use socsim_core::{
    derive_seed, AgentId, Mechanism, Phase, Result, SocsimError, StepContext, WorldState,
};
use socsim_llm::MetadataCollector;

use crate::config::{BetaGroup, LlmSettings, PromptVariant};
use crate::llm::{llm_config, SilenceClient};
use crate::prompts::{build_voice_prompt, parse_voice_decision};
use crate::world::{Expression, Motive, SilenceWorld};

/// Shared LLM client between driver + mechanism (`Rc<RefCell>` pattern).
pub type SharedClient = Rc<RefCell<SilenceClient>>;
/// Shared metadata collector for cache-hit rate / call count.
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

/// Shared parse-failure counter (LLM mode), surfaced into `llm_meta.json`.
pub type SharedParseFail = Rc<RefCell<(usize, usize)>>; // (failures, total)

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn clip01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

// --------------------------------------------------------------------------- //
// 1. IssueSalience (Environment)
// --------------------------------------------------------------------------- //

/// Mean-reverting `σ(t)` with an optional exogenous shock at `shock_t`.
pub struct IssueSalience {
    decay: f64,
    target: f64,
    shock_t: Option<u64>,
    shock_magnitude: f64,
}

impl IssueSalience {
    pub fn new(shock_t: Option<u64>, shock_magnitude: f64) -> Self {
        IssueSalience {
            decay: 0.10,
            target: 0.5,
            shock_t,
            shock_magnitude,
        }
    }
}

impl Mechanism<SilenceWorld> for IssueSalience {
    fn name(&self) -> &str {
        "issue_salience"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let sigma = ctx.world.issue_salience;
        let mut new_sigma = sigma + self.decay * (self.target - sigma);
        if let Some(t_shock) = self.shock_t {
            if ctx.clock.t() == t_shock {
                new_sigma = (new_sigma + self.shock_magnitude).clamp(0.0, 1.0);
            }
        }
        ctx.world.issue_salience = new_sigma.clamp(0.0, 1.0);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 2. RetaliationEvent (Environment)
// --------------------------------------------------------------------------- //

/// With probability `p_retaliate` per agent, mark them as retaliated this step.
pub struct RetaliationEvent {
    p_retaliate: f64,
}

impl RetaliationEvent {
    pub fn new(p_retaliate: f64) -> Self {
        RetaliationEvent {
            p_retaliate: p_retaliate.clamp(0.0, 1.0),
        }
    }
}

impl Mechanism<SilenceWorld> for RetaliationEvent {
    fn name(&self) -> &str {
        "retaliation_event"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.retaliation_this_step.clear();
        if self.p_retaliate <= 0.0 {
            return Ok(());
        }
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            if ctx.rng.gen::<f64>() < self.p_retaliate {
                ctx.world.retaliation_this_step.push(id);
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 3. FearAppraisal (Decision) — ψ → fear (paper β = −.54)
// --------------------------------------------------------------------------- //

/// `f_i ← clip(f_i + α_f (g_f − f_i))` with
/// `g_f = β_ψ⁻(1-ψ) + β_u⁻(1-u⁺) + β_r·retaliated`. Higher ψ lowers the fear
/// target, producing the negative ψ→fear path.
pub struct FearAppraisal {
    beta: BetaGroup,
}

impl FearAppraisal {
    pub fn new(beta: BetaGroup) -> Self {
        FearAppraisal { beta }
    }
}

impl Mechanism<SilenceWorld> for FearAppraisal {
    fn name(&self) -> &str {
        "fear_appraisal"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let team_idx = ctx.world.employees[&id].team;
            let u = ctx.world.teams[team_idx]
                .supervisor_openness
                .clamp(-1.0, 1.0);
            let u_pos = u.max(0.0);
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let r = if retaliated.contains(&id) { 1.0 } else { 0.0 };
            let g_f = self.beta.beta_psafety_fear * (1.0 - emp.psych_safety)
                + self.beta.beta_supervisor_fear * (1.0 - u_pos)
                + self.beta.beta_retaliation_fear * r;
            let g_f = clip01(g_f);
            emp.fear = clip01(emp.fear + self.beta.alpha_f * (g_f - emp.fear));
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4a. VoiceDecisionRule (Decision) — logistic ablation (fear → voice, β = −.23)
// --------------------------------------------------------------------------- //

/// `P(VOICE) = σ(β0 + β_ψ ψ + β_u u − β_f f − β_ι ι − β_ρ ρ)`. On SILENCE, the
/// motive is chosen by a sign-constrained softmax over the 4 motives.
pub struct VoiceDecisionRule {
    beta: BetaGroup,
}

impl VoiceDecisionRule {
    pub fn new(beta: BetaGroup) -> Self {
        VoiceDecisionRule { beta }
    }
}

/// Snapshot of the features `voice_decision_rule` reads for one agent.
struct RuleFeatures {
    psafety: f64,
    fear: f64,
    acquiescent: f64,
    ivt: f64,
    rho: f64,
    u: f64,
}

fn motive_softmax(x: &RuleFeatures, u_motive: f64) -> Motive {
    // Sign-constrained logits over (AS, QS, PS, OS).
    // AS (黙従): driven by acquiescent + ρ − ψ.
    // QS (怖れ): driven by fear − ψ.
    // PS (配慮): driven by supervisor openness (protect others) − small.
    // OS (自己都合): driven by ι (self-censorship habit) − ψ.
    let logits = [
        2.0 * x.acquiescent + 1.0 * x.rho - 1.0 * x.psafety,
        2.5 * x.fear - 1.0 * x.psafety,
        0.6 * x.u.max(0.0) + 0.3,
        1.2 * x.ivt - 0.8 * x.psafety,
    ];
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    let probs: Vec<f64> = exps.iter().map(|e| e / s).collect();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u_motive < acc {
            return Motive::ALL[i];
        }
    }
    Motive::Acquiescent
}

impl Mechanism<SilenceWorld> for VoiceDecisionRule {
    fn name(&self) -> &str {
        "voice_decision_rule"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        // Synchronous: snapshot features first.
        let mut snapshot: Vec<(AgentId, RuleFeatures)> = Vec::with_capacity(ids.len());
        for &id in &ids {
            let emp = &ctx.world.employees[&id];
            let team = &ctx.world.teams[emp.team];
            snapshot.push((
                id,
                RuleFeatures {
                    psafety: emp.psych_safety,
                    fear: emp.fear,
                    acquiescent: emp.acquiescent,
                    ivt: emp.ivt_strength,
                    rho: ctx.world.neighbour_silence_ratio(id),
                    u: team.supervisor_openness,
                },
            ));
        }

        let mut updates: Vec<(AgentId, bool)> = Vec::with_capacity(snapshot.len());
        for (id, x) in snapshot {
            let voice_logit = self.beta.voice_intercept
                + self.beta.beta_psafety_voice * x.psafety
                + self.beta.beta_supervisor_voice * x.u.max(0.0)
                - self.beta.beta_fear_voice * x.fear
                - self.beta.beta_ivt_voice * x.ivt
                - self.beta.beta_rho_voice * x.rho;
            let p_voice = sigmoid(voice_logit);
            let u_voice: f64 = ctx.rng.gen();
            updates.push((id, u_voice < p_voice));
        }
        // voice_decision only sets the independent `voiced` flag; the silence
        // dimension is resolved later by `silence_spiral` (paper H5: silence ⊥
        // voice). Reset per-step flags here so the step starts clean.
        for (id, voiced) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.voiced = voiced;
            emp.silenced = false;
            emp.silence_motive = None;
            emp.expression = if voiced {
                Expression::Voice
            } else {
                Expression::Neutral
            };
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 4b. VoiceDecisionLlm (Decision) — LLM-driven (Japanese-localised)
// --------------------------------------------------------------------------- //

/// LLM-driven voice decision. Per-agent prompt with `temperature=0` + a
/// `(agent_id, t)`-derived seed + the prompt→response cache pseudo-determinise
/// generation. Parse failures fall back to `Neutral + None` and are counted.
pub struct VoiceDecisionLlm {
    client: SharedClient,
    metadata: SharedMetadata,
    parse_fail: SharedParseFail,
    settings: LlmSettings,
    variant: PromptVariant,
    llm_seed_root: u64,
}

impl VoiceDecisionLlm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        parse_fail: SharedParseFail,
        settings: LlmSettings,
        variant: PromptVariant,
        llm_seed_root: u64,
    ) -> Self {
        VoiceDecisionLlm {
            client,
            metadata,
            parse_fail,
            settings,
            variant,
            llm_seed_root,
        }
    }
}

impl Mechanism<SilenceWorld> for VoiceDecisionLlm {
    fn name(&self) -> &str {
        "voice_decision"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        let t = ctx.clock.t();
        // Snapshot prompts before mutating (synchronous update).
        let mut prompts: Vec<(AgentId, String, u64)> = Vec::with_capacity(ids.len());
        for id in ids {
            let prompt = build_voice_prompt(ctx.world, id, self.variant);
            let llm_seed = derive_seed(self.llm_seed_root, &[3, id.0, t]);
            prompts.push((id, prompt, llm_seed));
        }

        let mut updates: Vec<(AgentId, Expression, Option<Motive>)> =
            Vec::with_capacity(prompts.len());
        for (id, prompt, llm_seed) in prompts {
            let mut cfg = llm_config(&self.settings);
            cfg.seed = llm_seed;
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client.complete(&prompt, &cfg).map_err(|e| {
                    SocsimError::Mechanism(format!("voice_decision LLM call failed: {e}"))
                })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let verdict = parse_voice_decision(&text);
            {
                let mut pf = self.parse_fail.borrow_mut();
                pf.1 += 1;
                if verdict.parse_failed {
                    pf.0 += 1;
                }
            }
            updates.push((id, verdict.expression, verdict.motive));
        }
        for (id, expr, m) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.expression = expr;
            emp.silence_motive = m;
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 5. AcquiescentUpdate (Interaction) — fear → acquiescent (paper β = +.86)
// --------------------------------------------------------------------------- //

/// `a_i ← clip((1-α_a) a_i + α_a σ_sat(γ_fa f_i + ι_i − c))`. Strong γ_fa
/// reproduces the paper's exceptionally strong fear→acquiescent path.
pub struct AcquiescentUpdate {
    beta: BetaGroup,
}

impl AcquiescentUpdate {
    pub fn new(beta: BetaGroup) -> Self {
        AcquiescentUpdate { beta }
    }
}

impl Mechanism<SilenceWorld> for AcquiescentUpdate {
    fn name(&self) -> &str {
        "acquiescent_update"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            // Centre the saturating function so a∈[0,1] is well spread.
            let target = sigmoid(self.beta.gamma_fa * (emp.fear - 0.5) + emp.ivt_strength - 0.5);
            emp.acquiescent =
                clip01((1.0 - self.beta.alpha_a) * emp.acquiescent + self.beta.alpha_a * target);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 6. SilenceSpiral (Interaction) — acquiescent → silence (paper β = +.52)
// --------------------------------------------------------------------------- //

/// Resolves the final expression for **non-voicers** only: an agent who did
/// *not* VOICE this step becomes SILENCE with probability driven by its
/// acquiescent motive (the `acquiescent → silence` path) plus a mild,
/// equilibrium-centred neighbour-silence climate term; otherwise it stays
/// NEUTRAL. A high neighbour-voice ratio above θ_i protects against silence.
///
/// Voicers are **never** flipped, which keeps voice an independent behavioural
/// dimension (paper H5: silence ⊥ voice, `r ≈ .02`). The climate term is
/// centred on `CLIMATE_ANCHOR` so it cannot drive a runaway all-silence
/// cascade — it only modulates around the equilibrium silence level.
pub struct SilenceSpiral {
    beta: BetaGroup,
}

/// Equilibrium neighbour-silence level the climate term is centred on.
const CLIMATE_ANCHOR: f64 = 0.4;

impl SilenceSpiral {
    pub fn new(beta: BetaGroup) -> Self {
        SilenceSpiral { beta }
    }
}

impl Mechanism<SilenceWorld> for SilenceSpiral {
    fn name(&self) -> &str {
        "silence_spiral"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        // Snapshot ρ and ρ^V (synchronous).
        let mut snap: Vec<(AgentId, f64, f64)> = Vec::with_capacity(ids.len());
        for &id in &ids {
            snap.push((
                id,
                ctx.world.neighbour_silence_ratio(id),
                ctx.world.neighbour_voice_ratio(id),
            ));
        }
        let mut updates: Vec<(AgentId, bool, Option<Motive>)> = Vec::with_capacity(snap.len());
        for (id, rho, rho_v) in snap {
            let emp = &ctx.world.employees[&id];
            let protect = if rho_v > emp.voice_threshold {
                1.0
            } else {
                0.0
            };
            let logit = self.beta.beta_acq_silence * (emp.acquiescent - 0.52)
                + self.beta.beta_climate_silence * (rho - CLIMATE_ANCHOR)
                - self.beta.beta_theta_silence * protect;
            let p_silence = sigmoid(logit);
            let u: f64 = ctx.rng.gen();
            if u < p_silence {
                let x = RuleFeatures {
                    psafety: emp.psych_safety,
                    fear: emp.fear,
                    acquiescent: emp.acquiescent,
                    ivt: emp.ivt_strength,
                    rho,
                    u: 0.0,
                };
                let u_m: f64 = ctx.rng.gen();
                updates.push((id, true, Some(motive_softmax(&x, u_m))));
            } else {
                updates.push((id, false, None));
            }
        }
        // The silence dimension is independent of `voiced`: an agent may voice
        // on one concern and stay silent on another. `expression` summarises the
        // step (Voice if voiced, else Silence if silenced, else Neutral).
        for (id, silenced, motive) in updates {
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            emp.silenced = silenced;
            emp.silence_motive = motive;
            emp.expression = if emp.voiced {
                Expression::Voice
            } else if silenced {
                Expression::Silence
            } else {
                Expression::Neutral
            };
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 7. PsafetyUpdate (PostStep)
// --------------------------------------------------------------------------- //

/// `ψ_i ← clip(ψ_i + η_ψ(ψ* − ψ_i))` where the target ψ* rises when the agent
/// voiced without retaliation and falls when retaliated.
pub struct PsafetyUpdate {
    eta_psi: f64,
}

impl PsafetyUpdate {
    pub fn new() -> Self {
        PsafetyUpdate { eta_psi: 0.10 }
    }
}

impl Default for PsafetyUpdate {
    fn default() -> Self {
        Self::new()
    }
}

impl Mechanism<SilenceWorld> for PsafetyUpdate {
    fn name(&self) -> &str {
        "psafety_update"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        let retaliated: std::collections::HashSet<AgentId> =
            ctx.world.retaliation_this_step.iter().copied().collect();
        let ids: Vec<AgentId> = ctx.world.agent_ids();
        for id in ids {
            let was_retaliated = retaliated.contains(&id);
            let emp = ctx.world.employees.get_mut(&id).expect("agent missing");
            let voiced = emp.voiced;
            let target = if was_retaliated {
                0.2
            } else if voiced {
                0.85
            } else {
                emp.psych_safety
            };
            emp.psych_safety =
                clip01(emp.psych_safety + self.eta_psi * (target - emp.psych_safety));
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// 8. ClimateSilence (PostStep)
// --------------------------------------------------------------------------- //

/// Aggregate the whole-organisation climate of silence `C(t)` so the next step's
/// Decision/Interaction phases observe it.
pub struct ClimateSilence;

impl Mechanism<SilenceWorld> for ClimateSilence {
    fn name(&self) -> &str {
        "climate_silence"
    }
    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }
    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, SilenceWorld>) -> Result<()> {
        ctx.world.climate_of_silence = crate::metrics::climate_of_silence(ctx.world);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn motive_softmax_high_fear_picks_quiescent_region() {
        // With high fear and low everything else, deterministic-ish at u=0 should
        // land on AS or QS (the fear/acquiescent-driven rows).
        let x = RuleFeatures {
            psafety: 0.1,
            fear: 0.95,
            acquiescent: 0.9,
            ivt: 0.1,
            rho: 0.8,
            u: -0.2,
        };
        let m = motive_softmax(&x, 0.5);
        assert!(matches!(m, Motive::Acquiescent | Motive::Quiescent));
    }
}
