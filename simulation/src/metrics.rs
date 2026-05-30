//! Aggregate metrics for the Fujimura & Hino (2019) silence-and-voice model.
//!
//! - **silence_rate** — fraction with `expression == Silence`.
//! - **voice_volume** — fraction with `expression == Voice`.
//! - **climate_of_silence** — `C(t) = (1/N) Σ 1[Silence ∧ acquiescent > 0.5]`
//!   (resigned silence under disagreement; Morrison & Milliken 2000 proxy).
//! - **motive_mix_{acquiescent,quiescent,prosocial,opportunistic}** — within-
//!   silent motive shares (Σ = 1 when any silence exists; 0 otherwise).
//! - **corr_silence_voice** — Pearson r between per-agent silence/voice flags
//!   (paper H5: r = .02, target |r| < .10).
//!
//! The 4 SEM path coefficients `β̃` are estimated Python-side from
//! `agent_panel.csv` (semopy), not here.

use crate::world::SilenceWorld;

/// Fraction of employees in `Silence`.
pub fn silence_rate(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let s = world.employees.values().filter(|e| e.silenced).count();
    s as f64 / n as f64
}

/// Fraction of employees in `Voice`.
pub fn voice_volume(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let v = world.employees.values().filter(|e| e.voiced).count();
    v as f64 / n as f64
}

/// `C(t) = (1/N) Σ 1[Silence ∧ acquiescent > 0.5]` — resigned silence proxy.
pub fn climate_of_silence(world: &SilenceWorld) -> f64 {
    let n = world.n_employees();
    if n == 0 {
        return 0.0;
    }
    let c = world
        .employees
        .values()
        .filter(|e| e.silenced && e.acquiescent > 0.5)
        .count();
    c as f64 / n as f64
}

/// 4-vector `(acquiescent, quiescent, prosocial, opportunistic)` of within-
/// silent motive shares (Σ = 1 if any silence exists, else the zero vector).
pub fn motive_mix(world: &SilenceWorld) -> [f64; 4] {
    let mut counts = [0u64; 4];
    let mut total = 0u64;
    for e in world.employees.values() {
        if e.silenced {
            if let Some(m) = e.silence_motive {
                counts[m.index()] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return [0.0; 4];
    }
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = counts[i] as f64 / total as f64;
    }
    out
}

/// Pearson correlation. Returns 0 on degenerate inputs.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        0.0
    } else {
        sxy / denom
    }
}

/// Per-step metric snapshot from the current world.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub silence_rate: f64,
    pub voice_volume: f64,
    pub climate_of_silence: f64,
    pub motive_mix: [f64; 4],
}

pub fn step_metrics(world: &SilenceWorld) -> StepMetrics {
    StepMetrics {
        silence_rate: silence_rate(world),
        voice_volume: voice_volume(world),
        climate_of_silence: climate_of_silence(world),
        motive_mix: motive_mix(world),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Employee, Expression, Locale, Motive, SilenceWorld, Team};
    use socsim_core::{AgentId, SimClock, SimRng};
    use socsim_net::SocialNetwork;
    use std::collections::BTreeMap;

    fn mini(exprs: &[Expression], motives: &[Option<Motive>]) -> SilenceWorld {
        let mut rng = SimRng::from_seed(0);
        let ids: Vec<AgentId> = (0..exprs.len()).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.5, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            let mut e = Employee::neutral(0, 1, 0);
            e.expression = exprs[i];
            e.voiced = exprs[i] == Expression::Voice;
            e.silenced = exprs[i] == Expression::Silence;
            e.silence_motive = motives[i];
            e.acquiescent = 0.8;
            emps.insert(id, e);
        }
        SilenceWorld {
            clock: SimClock::new(1),
            employees: emps,
            teams: vec![Team::default()],
            network: net,
            issue_salience: 0.5,
            climate_of_silence: 0.0,
            locale: Locale::JaJp,
            hierarchy_strength: 4,
            eta: 0.7,
            retaliation_this_step: Vec::new(),
        }
    }

    #[test]
    fn motive_mix_sums_to_one_when_silence() {
        let w = mini(
            &[Expression::Silence, Expression::Silence, Expression::Voice],
            &[Some(Motive::Acquiescent), Some(Motive::Quiescent), None],
        );
        let mix = motive_mix(&w);
        assert!((mix.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rates_sum_le_one() {
        let w = mini(
            &[Expression::Silence, Expression::Voice, Expression::Neutral],
            &[Some(Motive::Acquiescent), None, None],
        );
        assert!((silence_rate(&w) - 1.0 / 3.0).abs() < 1e-12);
        assert!((voice_volume(&w) - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
    }
}
