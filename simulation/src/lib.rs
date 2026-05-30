//! Fujimura & Hino (2019) — Silence and voice in the organization.
//!
//! A socsim-based LLM-agent ABM that reproduces the paper's structural-equation
//! model (psychological safety → fear/Quiescent → acquiescence/黙従 → silence,
//! plus fear → suppressed voice) on a Watts–Strogatz organisational network.
//!
//! Two **mutually exclusive** decision modes are wired by `config.decision_mode`:
//!
//! - `--decision-mode rule` — `VoiceDecisionRule`: a sign-constrained logistic
//!   ablation over the latent states (LLM-free, bit-deterministic).
//! - `--decision-mode llm` — `VoiceDecisionLlm`: a Japanese-localised
//!   (or English-contrast) prompt is sent to an LLM (Ollama-first, OpenAI
//!   fallback) via `socsim-llm`'s shared harness, returning a JSON
//!   `{decision, motive, rationale}`.
//!
//! Seven further deterministic mechanisms run unconditionally each step across
//! socsim's 6-phase loop: `IssueSalience`, `RetaliationEvent`, `FearAppraisal`,
//! `AcquiescentUpdate`, `SilenceSpiral`, `PsafetyUpdate`, `ClimateSilence`.
//!
//! The 4 SEM path coefficients `β̃` are estimated Python-side (`semopy`) from
//! the long-format `agent_panel.csv`.

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;
