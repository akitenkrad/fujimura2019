//! LLM harness — thin re-exports of the shared `socsim-llm` harness.
//!
//! Every LLM-driven replication shares the same boilerplate (`LlmSettings`,
//! `LiveClient = CachingClient<Box<dyn LlmClient>>`, `wrap_client`, `llm_config`,
//! `build_live_client_from_settings`). It lives in `socsim-llm::harness`; this
//! module re-exports the bits under the repo-local `crate::llm::*` ergonomics
//! (mirroring the knoll2013 / detert2011 / noelleneumann1974 convention).
//!
//! Provider order at runtime is **Ollama first → OpenAI fallback**
//! (`OLLAMA_HOST` / `OLLAMA_MODEL` / `OPENAI_API_KEY` / `OPENAI_MODEL`).
//! `temperature=0` + a per-`(agent_id, t)` seed (set in the mechanism) + the
//! prompt→response cache pseudo-determinise generation, keeping the
//! deterministic socsim core reproducible.

pub use socsim_llm::build_live_client_from_settings as build_live_client;
pub use socsim_llm::{llm_config, wrap_client, LiveClient as SilenceClient, LlmSettings};
