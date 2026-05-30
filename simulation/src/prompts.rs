//! LLM prompt construction (Japanese-localised + English contrast) and response
//! parsing for the `voice_decision` mechanism.
//!
//! The LLM is asked, given an employee's local organisational context, to
//! return a JSON decision of the form
//!
//! ```json
//! {"decision": "VOICE" | "SILENCE",
//!  "motive": "acquiescent" | "quiescent" | "prosocial" | "opportunistic" | null,
//!  "rationale": "短い理由"}
//! ```
//!
//! On parse failure we fall back to `Neutral + None` and flag `parse_failed`
//! (recorded as the parse-failure rate in `llm_meta.json`).

use serde::Deserialize;
use serde_json::Value;
use socsim_core::AgentId;

use crate::config::PromptVariant;
use crate::world::{Expression, Locale, Motive, SilenceWorld};

// --------------------------------------------------------------------------- //
// Prompt construction
// --------------------------------------------------------------------------- //

/// Local context snapshot fed into a prompt (taken before mutation).
struct PromptContext {
    level_name: String,
    tenure: u32,
    team_size: usize,
    peer_silence_pct: f64,
    psych_safety: f64,
    fear: f64,
    salience: f64,
    retaliation_recent: bool,
    supervisor_open: bool,
}

fn gather_context(world: &SilenceWorld, id: AgentId) -> PromptContext {
    let emp = &world.employees[&id];
    let team = &world.teams[emp.team];
    let rho = world.neighbour_silence_ratio(id);
    let team_size = world
        .employees
        .values()
        .filter(|e| e.team == emp.team)
        .count();
    let retaliation_recent = world.retaliation_this_step.contains(&id)
        || world
            .network
            .neighbors(id)
            .iter()
            .any(|nb| world.retaliation_this_step.contains(nb));
    let level_name = level_name_ja(emp.level, world.hierarchy_strength);
    PromptContext {
        level_name,
        tenure: emp.tenure,
        team_size,
        peer_silence_pct: rho * 100.0,
        psych_safety: emp.psych_safety,
        fear: emp.fear,
        salience: world.issue_salience,
        retaliation_recent,
        supervisor_open: team.supervisor_openness > 0.0,
    }
}

fn level_name_ja(level: u8, l: u8) -> String {
    // level 1 = 一般社員 ... level L = 部長相当．
    match (level, l) {
        (1, _) => "一般社員".to_string(),
        (2, _) => "主任".to_string(),
        (3, _) => "課長".to_string(),
        (lv, ll) if lv >= ll => "部長".to_string(),
        _ => "中堅社員".to_string(),
    }
}

fn level_name_en(level: u8, l: u8) -> String {
    match (level, l) {
        (1, _) => "staff member".to_string(),
        (2, _) => "team lead".to_string(),
        (lv, ll) if lv >= ll => "department head".to_string(),
        _ => "manager".to_string(),
    }
}

/// Build the voice-decision prompt for `id` in the world's locale.
pub fn build_voice_prompt(world: &SilenceWorld, id: AgentId, variant: PromptVariant) -> String {
    match world.locale {
        Locale::JaJp => build_voice_prompt_ja(world, id, variant),
        Locale::EnUs => build_voice_prompt_en(world, id, variant),
    }
}

/// Japanese workplace prompt (design §4.3.3). High hierarchy, closed peers.
pub fn build_voice_prompt_ja(world: &SilenceWorld, id: AgentId, variant: PromptVariant) -> String {
    let c = gather_context(world, id);
    let supervisor_relation = if c.supervisor_open {
        "比較的相談しやすい"
    } else {
        "気軽に意見を言いにくい"
    };
    let retaliation_recent = if c.retaliation_recent {
        "ありました"
    } else {
        "ありませんでした"
    };
    let issue_description = match variant {
        PromptVariant::A => "業務プロセスの非効率",
        PromptVariant::B => "コンプライアンス上の懸念",
        PromptVariant::C => "顧客に不利益を与えかねない慣行",
    };
    format!(
        "あなたは日本の企業に勤める{level_name}（在職 {tenure} ヶ月）の正社員です．\n\
         直属上司との関係は{supervisor_relation}状況で，所属チーム（{team_size}名）のうち\n\
         近頃 {peer_silence_pct:.0}% の同僚は懸念があっても黙る傾向にあります．\n\
         あなた自身の心理的安全の感覚は {psych_safety:.2}/1.00，\n\
         怖れ動機の現在水準は {fear:.2}/1.00 です．\n\n\
         組織で{issue_description}（深刻度 {salience:.2}/1.00）の問題に気づきました．\n\
         直近のチームで報復イベントが{retaliation_recent}．\n\n\
         発言すべきか沈黙すべきか，そしてその理由を以下の 4 類型から選んでください．\n\
         - acquiescent（黙従：「言っても何も変わらない」）\n\
         - quiescent（怖れ：「報復・評価低下が怖い」）\n\
         - prosocial（配慮：「上司・組織・同僚を守りたい」）\n\
         - opportunistic（自己都合：「自分の負担を増やしたくない」）\n\n\
         JSON で {{\"decision\": \"VOICE\"|\"SILENCE\", \"motive\": <類型>, \"rationale\": <40字以内>}}\n\
         の形式で 1 行で答えてください．VOICE のときは motive を null にしてください．",
        level_name = c.level_name,
        tenure = c.tenure,
        supervisor_relation = supervisor_relation,
        team_size = c.team_size,
        peer_silence_pct = c.peer_silence_pct,
        psych_safety = c.psych_safety,
        fear = c.fear,
        issue_description = issue_description,
        salience = c.salience,
        retaliation_recent = retaliation_recent,
    )
}

/// English-contrast prompt (flatter hierarchy, open-door vocabulary). Design §4.3.3.
pub fn build_voice_prompt_en(world: &SilenceWorld, id: AgentId, variant: PromptVariant) -> String {
    let emp = &world.employees[&id];
    let c = gather_context(world, id);
    let level_name = level_name_en(emp.level, world.hierarchy_strength);
    let supervisor_relation = if c.supervisor_open {
        "an approachable, open-door manager"
    } else {
        "a manager who is hard to approach"
    };
    let retaliation_recent = if c.retaliation_recent {
        "there was"
    } else {
        "there was no"
    };
    let issue_description = match variant {
        PromptVariant::A => "an inefficiency in a work process",
        PromptVariant::B => "a compliance concern",
        PromptVariant::C => "a practice that could harm customers",
    };
    format!(
        "You are a {level_name} (tenure {tenure} months) at a company. Your relationship \
         with your direct manager is {supervisor_relation}. In your team meeting of \
         {team_size} people, about {peer_silence_pct:.0}% of colleagues tend to stay \
         quiet about concerns lately.\n\
         Your sense of psychological safety is {psych_safety:.2}/1.00, and your current \
         fear level is {fear:.2}/1.00.\n\n\
         You have noticed {issue_description} (severity {salience:.2}/1.00). Recently \
         {retaliation_recent} a retaliation event in your team.\n\n\
         Decide whether to SPEAK UP (voice) or REMAIN SILENT, and pick a reason from:\n\
         - acquiescent (resigned: \"nothing will change\")\n\
         - quiescent (fear: \"afraid of retaliation or a bad evaluation\")\n\
         - prosocial (protective: \"want to protect my manager, org, or peers\")\n\
         - opportunistic (self-interest: \"do not want extra work for myself\")\n\n\
         Reply with a SINGLE-LINE JSON: {{\"decision\": \"VOICE\"|\"SILENCE\", \"motive\": <type>, \
         \"rationale\": <short>}}. If decision = VOICE, motive must be null.",
        level_name = level_name,
        tenure = c.tenure,
        supervisor_relation = supervisor_relation,
        team_size = c.team_size,
        peer_silence_pct = c.peer_silence_pct,
        psych_safety = c.psych_safety,
        fear = c.fear,
        issue_description = issue_description,
        salience = c.salience,
        retaliation_recent = retaliation_recent,
    )
}

// --------------------------------------------------------------------------- //
// Response parsing
// --------------------------------------------------------------------------- //

/// Parsed voice-decision verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceDecisionVerdict {
    pub expression: Expression,
    pub motive: Option<Motive>,
    pub rationale: String,
    /// True when parsing failed and we fell back to `Neutral + None`.
    pub parse_failed: bool,
}

#[derive(Deserialize)]
struct RawDecision {
    decision: Option<String>,
    motive: Option<String>,
    rationale: Option<String>,
}

/// Parse an LLM response into a verdict. Lenient: extracts the first balanced
/// `{...}` object, accepts mixed-case labels, and on any failure falls back to
/// `Neutral + None` (`parse_failed = true`).
pub fn parse_voice_decision(text: &str) -> VoiceDecisionVerdict {
    let fallback = VoiceDecisionVerdict {
        expression: Expression::Neutral,
        motive: None,
        rationale: String::new(),
        parse_failed: true,
    };

    let json_str = match extract_json_object(text) {
        Some(s) => s,
        None => return fallback,
    };

    if let Ok(raw) = serde_json::from_str::<RawDecision>(&json_str) {
        return finalise_verdict(raw);
    }
    if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
        let raw = RawDecision {
            decision: val
                .get("decision")
                .and_then(|v| v.as_str().map(str::to_string)),
            motive: val.get("motive").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_str().map(str::to_string)
                }
            }),
            rationale: val
                .get("rationale")
                .and_then(|v| v.as_str().map(str::to_string)),
        };
        return finalise_verdict(raw);
    }
    fallback
}

fn finalise_verdict(raw: RawDecision) -> VoiceDecisionVerdict {
    let decision = raw
        .decision
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let expression = match decision.as_str() {
        "voice" | "speak" | "speak_up" => Expression::Voice,
        "silence" | "silent" | "withhold" => Expression::Silence,
        _ => {
            return VoiceDecisionVerdict {
                expression: Expression::Neutral,
                motive: None,
                rationale: raw.rationale.unwrap_or_default(),
                parse_failed: true,
            };
        }
    };

    let motive = if expression == Expression::Voice {
        None
    } else {
        match raw
            .motive
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("acquiescent") | Some("黙従") => Some(Motive::Acquiescent),
            Some("quiescent") | Some("怖れ") => Some(Motive::Quiescent),
            Some("prosocial") | Some("配慮") => Some(Motive::Prosocial),
            Some("opportunistic") | Some("自己都合") => Some(Motive::Opportunistic),
            // Silence without a recognised motive → parse_failed, default None.
            _ => {
                return VoiceDecisionVerdict {
                    expression: Expression::Silence,
                    motive: None,
                    rationale: raw.rationale.unwrap_or_default(),
                    parse_failed: true,
                };
            }
        }
    };

    VoiceDecisionVerdict {
        expression,
        motive,
        rationale: raw.rationale.unwrap_or_default(),
        parse_failed: false,
    }
}

/// Extract the first balanced `{...}` substring from `text`.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_voice() {
        let v = parse_voice_decision(r#"{"decision":"VOICE","motive":null,"rationale":"ok"}"#);
        assert_eq!(v.expression, Expression::Voice);
        assert_eq!(v.motive, None);
        assert!(!v.parse_failed);
    }

    #[test]
    fn parses_silence_with_motive() {
        let v = parse_voice_decision(
            r#"{"decision":"SILENCE","motive":"acquiescent","rationale":"何も変わらない"}"#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Acquiescent));
        assert!(!v.parse_failed);
    }

    #[test]
    fn parses_japanese_motive_label() {
        let v = parse_voice_decision(r#"{"decision":"SILENCE","motive":"怖れ","rationale":"x"}"#);
        assert_eq!(v.motive, Some(Motive::Quiescent));
    }

    #[test]
    fn tolerates_surrounding_text() {
        let v = parse_voice_decision(
            r#"はい: {"decision":"silence","motive":"quiescent","rationale":"fear"} 以上"#,
        );
        assert_eq!(v.expression, Expression::Silence);
        assert_eq!(v.motive, Some(Motive::Quiescent));
    }

    #[test]
    fn no_json_falls_back_to_neutral() {
        let v = parse_voice_decision("no json here");
        assert!(v.parse_failed);
        assert_eq!(v.expression, Expression::Neutral);
    }
}
