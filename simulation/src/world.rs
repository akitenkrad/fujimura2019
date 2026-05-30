//! World state for Fujimura & Hino (2019) silence-and-voice SEM model.
//!
//! Implements socsim's [`WorldState`] over employees living on a
//! [`SocialNetwork`] (Watts–Strogatz small-world organisational network). Each
//! employee carries the latent psychological states of the paper's structural
//! equation model — psychological safety `ψ`, fear/Quiescent motive `f`,
//! acquiescent motive `a`, implicit-voice-theory strength `ι`, voice threshold
//! `θ` — plus a 4-way silence motive (`Acquiescent / Quiescent / Prosocial /
//! Opportunistic`, `None` when the agent voices).
//!
//! ## Terminology (critical — design-doc warning box)
//!
//! Following the paper *body* and Fig. 1 (the primary source), **not** the
//! English abstract:
//! - `Quiescent` = 怖れ (fear-based withholding)
//! - `Acquiescent` = 黙従 (resigned "nothing will change" withholding)

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

// --------------------------------------------------------------------------- //
// Locale
// --------------------------------------------------------------------------- //

/// Cultural locale governing prompt language + structural defaults
/// (hierarchy strength, IVT strength, supervisor-signal homogeneity).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Locale {
    /// Japanese workplace (high hierarchy `L = 4`, strong IVT, closed peers).
    JaJp,
    /// English-speaking workplace (flatter `L = 3`, open-door vocabulary).
    EnUs,
}

impl Locale {
    /// Stable label used in CSV / JSON / directory names.
    pub fn label(&self) -> &'static str {
        match self {
            Locale::JaJp => "ja-JP",
            Locale::EnUs => "en-US",
        }
    }

    /// Default hierarchy strength `L` for this locale (design §4.3.1).
    pub fn default_hierarchy_strength(&self) -> u8 {
        match self {
            Locale::JaJp => 4,
            Locale::EnUs => 3,
        }
    }

    /// Default initial IVT-strength mean `ῑ` for this locale (design §4.3.1).
    pub fn default_ivt_mean(&self) -> f64 {
        match self {
            Locale::JaJp => 0.55,
            Locale::EnUs => 0.40,
        }
    }
}

/// Parse a [`Locale`] from a CLI string.
pub fn parse_locale(s: &str) -> Result<Locale, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ja-jp" | "ja" | "jp" | "japan" | "japanese" => Ok(Locale::JaJp),
        "en-us" | "en" | "us" | "english" => Ok(Locale::EnUs),
        _ => Err(format!("invalid locale: \"{s}\" (ja-JP / en-US)")),
    }
}

// --------------------------------------------------------------------------- //
// Motive / Expression
// --------------------------------------------------------------------------- //

/// 4-form silence motive (Knoll & van Dick 2013; same 4 motives the paper uses).
///
/// - `Acquiescent` (黙従) — resigned "nothing will change" withholding.
/// - `Quiescent` (怖れ) — fear-based self-protective withholding.
/// - `Prosocial` (配慮) — protective, other-oriented silence.
/// - `Opportunistic` (自己都合) — self-interested strategic withholding.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Motive {
    Acquiescent,
    Quiescent,
    Prosocial,
    Opportunistic,
}

impl Motive {
    /// Stable lowercase label (CSV / JSON friendly).
    pub fn label(&self) -> &'static str {
        match self {
            Motive::Acquiescent => "acquiescent",
            Motive::Quiescent => "quiescent",
            Motive::Prosocial => "prosocial",
            Motive::Opportunistic => "opportunistic",
        }
    }

    /// All 4 motives in canonical order.
    pub const ALL: [Motive; 4] = [
        Motive::Acquiescent,
        Motive::Quiescent,
        Motive::Prosocial,
        Motive::Opportunistic,
    ];

    /// Index this motive into 0..4 in canonical order.
    pub fn index(&self) -> usize {
        match self {
            Motive::Acquiescent => 0,
            Motive::Quiescent => 1,
            Motive::Prosocial => 2,
            Motive::Opportunistic => 3,
        }
    }
}

/// Public expression at step `t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expression {
    Voice,
    Silence,
    Neutral,
}

impl Expression {
    pub fn label(&self) -> &'static str {
        match self {
            Expression::Voice => "voice",
            Expression::Silence => "silence",
            Expression::Neutral => "neutral",
        }
    }
}

// --------------------------------------------------------------------------- //
// Employee / Team
// --------------------------------------------------------------------------- //

/// Per-employee latent state (the paper's SEM observed/latent variables).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Employee {
    /// Hierarchical level `ℓ_i ∈ {1..L}`.
    pub level: u8,
    /// Team membership index `k(i)`.
    pub team: usize,
    /// Tenure in months `τ_i`.
    pub tenure: u32,
    /// Psychological safety `ψ_i ∈ [0,1]` (Edmondson 1999 / Baer & Frese 2003).
    pub psych_safety: f64,
    /// Fear / Quiescent motive `f_i ∈ [0,1]` (Kish-Gephart 2009).
    pub fear: f64,
    /// Acquiescent motive `a_i ∈ [0,1]`.
    pub acquiescent: f64,
    /// Implicit-voice-theory strength `ι_i ∈ [0,1]` (Detert 2011).
    pub ivt_strength: f64,
    /// Current public expression `b̂_i` (primary behaviour this step).
    pub expression: Expression,
    /// Silence motive when silent this step; `None` otherwise.
    pub silence_motive: Option<Motive>,
    /// VOICE threshold `θ_i ∈ [0,1]` (Kuran 1995).
    pub voice_threshold: f64,
    /// Whether the agent voiced this step (independent behavioural dimension).
    pub voiced: bool,
    /// Whether the agent stayed silent on a concern this step (independent of
    /// `voiced`: an employee can voice some issues and withhold others — the
    /// paper measures voice and silence as two separate scales, `r ≈ .02`).
    pub silenced: bool,
}

impl Employee {
    /// Neutral employee with mid-range latent state (overwritten at world init).
    pub fn neutral(team: usize, level: u8, tenure: u32) -> Self {
        Employee {
            level,
            team,
            tenure,
            psych_safety: 0.613,
            fear: 0.450,
            acquiescent: 0.457,
            ivt_strength: 0.55,
            expression: Expression::Neutral,
            silence_motive: None,
            voice_threshold: 0.40,
            voiced: false,
            silenced: false,
        }
    }
}

/// Per-team state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Team {
    /// Supervisor openness `u_k ∈ [-1, 1]` (mean `η`-homogenised at init).
    pub supervisor_openness: f64,
}

impl Default for Team {
    fn default() -> Self {
        Team {
            supervisor_openness: 0.0,
        }
    }
}

// --------------------------------------------------------------------------- //
// SilenceWorld
// --------------------------------------------------------------------------- //

/// World state for the Fujimura & Hino (2019) silence-and-voice model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SilenceWorld {
    pub clock: SimClock,
    /// Employees keyed by sorted [`AgentId`] (sorted order = determinism).
    pub employees: BTreeMap<AgentId, Employee>,
    pub teams: Vec<Team>,
    /// Inter-employee social network (Watts–Strogatz).
    pub network: SocialNetwork,
    /// Issue salience `σ(t) ∈ [0,1]`.
    pub issue_salience: f64,
    /// Whole-organisation climate of silence `C(t)` (emergent aggregate).
    pub climate_of_silence: f64,
    /// Cultural locale (prompt language + structural defaults).
    pub locale: Locale,
    /// Hierarchy strength `L` (4 = JP, 3 = EN).
    pub hierarchy_strength: u8,
    /// Supervisor-signal homogeneity `η ∈ [0,1]`.
    pub eta: f64,
    /// Agents touched by retaliation in the current step (transient until PostStep).
    pub retaliation_this_step: Vec<AgentId>,
}

impl SilenceWorld {
    /// Total number of employees.
    pub fn n_employees(&self) -> usize {
        self.employees.len()
    }

    /// Perceived neighbour-*silence* ratio `ρ_i` for `id` over its network
    /// neighbours. Isolated nodes return 0.
    pub fn neighbour_silence_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut silent = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.silenced {
                    silent += 1;
                }
            }
        }
        silent as f64 / neighbours.len() as f64
    }

    /// Perceived neighbour-*voice* ratio `ρ^V_i` for `id`. Isolated nodes return 0.
    pub fn neighbour_voice_ratio(&self, id: AgentId) -> f64 {
        let neighbours = self.network.neighbors(id);
        if neighbours.is_empty() {
            return 0.0;
        }
        let mut voice = 0usize;
        for nb in &neighbours {
            if let Some(e) = self.employees.get(nb) {
                if e.voiced {
                    voice += 1;
                }
            }
        }
        voice as f64 / neighbours.len() as f64
    }
}

impl WorldState for SilenceWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        self.employees.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_core::SimRng;

    #[test]
    fn motive_index_round_trips() {
        for m in Motive::ALL {
            assert_eq!(Motive::ALL[m.index()], m);
        }
    }

    #[test]
    fn locale_defaults() {
        assert_eq!(Locale::JaJp.default_hierarchy_strength(), 4);
        assert_eq!(Locale::EnUs.default_hierarchy_strength(), 3);
        assert!((Locale::JaJp.default_ivt_mean() - 0.55).abs() < 1e-12);
        assert!((Locale::EnUs.default_ivt_mean() - 0.40).abs() < 1e-12);
    }

    #[test]
    fn parse_locale_variants() {
        assert_eq!(parse_locale("ja-JP").unwrap(), Locale::JaJp);
        assert_eq!(parse_locale("EN").unwrap(), Locale::EnUs);
        assert!(parse_locale("xx").is_err());
    }

    #[test]
    fn neighbour_ratios_isolated_is_zero() {
        let mut rng = SimRng::from_seed(7);
        let ids: Vec<AgentId> = (0..4).map(|i| AgentId(i as u64)).collect();
        let net = SocialNetwork::erdos_renyi(&ids, 0.0, &mut rng);
        let mut emps: BTreeMap<AgentId, Employee> = BTreeMap::new();
        for &id in &ids {
            emps.insert(id, Employee::neutral(0, 1, 0));
        }
        let world = SilenceWorld {
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
        };
        assert_eq!(world.neighbour_silence_ratio(AgentId(0)), 0.0);
        assert_eq!(world.neighbour_voice_ratio(AgentId(0)), 0.0);
    }
}
