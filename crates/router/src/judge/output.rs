// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict, bounded validation of version-1 pairwise judge output.

use std::collections::{BTreeSet, VecDeque};
use std::fmt;

use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value as Json};
use unicode_normalization::UnicodeNormalization;

use super::contains_sensitive_free_text;
use crate::config::{JUDGE_MAX_RATIONALE_BYTES, JUDGE_OUTPUT_SCHEMA_V1};
use crate::fingerprint::sha256_hex;
use crate::projection::is_sensitive_projection_key;

const JUDGE_OUTPUT_OVERHEAD_BYTES: usize = 8 * 1024;
const MAX_VALIDATION_ISSUES: usize = 16;
const MAX_VALIDATION_ISSUE_BYTES: usize = 256;
const REQUIRED_FIELDS: [&str; 5] = [
    "hard_failures",
    "judge_confidence",
    "rationale",
    "response_equivalence",
    "trajectory_equivalence",
];
const CREDENTIAL_ISSUE: &str = "output.credential_bearing";
const TRUNCATED_ISSUES: &str = "issues.truncated";
const MALFORMED_KEY_CONTEXT_BYTES: usize = 256;
const MALFORMED_KEY_PROBE_BYTES: usize = 256;

/// Published version-1 hard-failure categories in canonical enum order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeHardFailureV1 {
    /// The candidate violated a declared tool contract.
    ToolContract,
    /// The candidate violated the response schema.
    ResponseSchema,
    /// The candidate produced unsafe output.
    Safety,
    /// The candidate response could not be interpreted.
    MalformedCandidate,
}

impl JudgeHardFailureV1 {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "tool_contract" => Some(Self::ToolContract),
            "response_schema" => Some(Self::ResponseSchema),
            "safety" => Some(Self::Safety),
            "malformed_candidate" => Some(Self::MalformedCandidate),
            _ => None,
        }
    }
}

/// A fully validated version-1 judge result.
///
/// This type intentionally does not implement `Deserialize`; callers must use
/// [`validate_pairwise_judge_output`] so numeric, text, and secret bounds cannot
/// be bypassed.
#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct PairwiseJudgeResultV1 {
    /// Semantic equivalence of the final response.
    response_equivalence: f64,
    /// Equivalence of the observed future-local trajectory.
    trajectory_equivalence: f64,
    /// Judge confidence in the comparison.
    judge_confidence: f64,
    /// Unique failures in published enum order.
    hard_failures: Vec<JudgeHardFailureV1>,
    /// Nonempty, NFC, control-free, credential-free rationale.
    rationale: String,
}

impl PairwiseJudgeResultV1 {
    /// Return the validated response-equivalence score.
    pub(crate) const fn response_equivalence(&self) -> f64 {
        self.response_equivalence
    }

    /// Return the validated trajectory-equivalence score.
    pub(crate) const fn trajectory_equivalence(&self) -> f64 {
        self.trajectory_equivalence
    }

    /// Return the validated judge-confidence score.
    pub(crate) const fn judge_confidence(&self) -> f64 {
        self.judge_confidence
    }

    /// Return unique hard failures in published enum order.
    pub(crate) fn hard_failures(&self) -> &[JudgeHardFailureV1] {
        &self.hard_failures
    }

    /// Return the validated rationale.
    pub(crate) fn rationale(&self) -> &str {
        &self.rationale
    }
}

/// One stable, bounded judge-output validation issue.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub(crate) struct JudgeValidationIssueV1(String);

impl JudgeValidationIssueV1 {
    fn new(value: impl Into<String>) -> Self {
        let value = value.into();
        if value.len() <= MAX_VALIDATION_ISSUE_BYTES {
            Self(value)
        } else {
            Self(format!("issue.sha256.{}", sha256_hex(value.as_bytes())))
        }
    }

    /// Return the stable issue text.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for JudgeValidationIssueV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if value.len() > MAX_VALIDATION_ISSUE_BYTES {
            return Err(serde::de::Error::custom(
                "judge validation issue exceeds its byte bound",
            ));
        }
        Ok(Self(value))
    }
}

/// Why invalid output could not be retained verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InvalidJudgeOutputMarkerV1 {
    /// The original output exceeded the checked raw-output bound.
    Oversized,
    /// The original or decoded output contained credential-shaped material.
    CredentialBearing,
}

/// Secret-safe representation of one invalid assistant output.
///
/// `Debug` is deliberately omitted because `Retained` contains exact
/// assistant-controlled text.
#[derive(Serialize)]
#[serde(tag = "retention", rename_all = "snake_case")]
pub(crate) enum SafeInvalidJudgeOutputV1 {
    /// Exact bounded output that passed the free-text credential scan.
    Retained {
        /// Exact assistant output.
        output: String,
        /// SHA-256 of the exact UTF-8 bytes.
        sha256: String,
        /// Original UTF-8 byte count.
        byte_count: u64,
    },
    /// Hash-only evidence for unsafe or oversized output.
    Redacted {
        /// Stable structural reason for redaction.
        marker: InvalidJudgeOutputMarkerV1,
        /// SHA-256 of the exact UTF-8 bytes.
        sha256: String,
        /// Original UTF-8 byte count.
        byte_count: u64,
    },
}

impl SafeInvalidJudgeOutputV1 {
    /// Return exact invalid output only when retention was safe.
    pub(crate) fn retained_output(&self) -> Option<&str> {
        match self {
            Self::Retained { output, .. } => Some(output),
            Self::Redacted { .. } => None,
        }
    }

    /// Return the stable redaction marker, if any.
    pub(crate) fn marker(&self) -> Option<InvalidJudgeOutputMarkerV1> {
        match self {
            Self::Retained { .. } => None,
            Self::Redacted { marker, .. } => Some(*marker),
        }
    }

    /// Return the SHA-256 of the original output bytes.
    pub(crate) fn sha256(&self) -> &str {
        match self {
            Self::Retained { sha256, .. } | Self::Redacted { sha256, .. } => sha256,
        }
    }

    /// Return the original UTF-8 byte count.
    pub(crate) fn byte_count(&self) -> u64 {
        match self {
            Self::Retained { byte_count, .. } | Self::Redacted { byte_count, .. } => *byte_count,
        }
    }
}

/// Validated evidence retained for one invalid judge output.
///
/// `Debug` is deliberately omitted because the nested safe representation may
/// contain exact assistant-controlled text.
#[derive(Serialize)]
pub(crate) struct InvalidJudgeOutputV1 {
    output: SafeInvalidJudgeOutputV1,
    issues: Vec<JudgeValidationIssueV1>,
}

impl InvalidJudgeOutputV1 {
    /// Return the safe invalid-output representation.
    pub(crate) fn output(&self) -> &SafeInvalidJudgeOutputV1 {
        &self.output
    }

    /// Return the sorted, bounded validation issues.
    pub(crate) fn issues(&self) -> &[JudgeValidationIssueV1] {
        &self.issues
    }
}

/// Stable output-validation failures that are operational, not judge results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeOutputOperationalFailureV1 {
    /// The configured rationale bound was outside the Task 2 contract.
    InvalidRationaleBound,
    /// The raw output size could not be represented durably.
    OutputSizeUnrepresentable,
}

/// Result of validating one readable assistant output.
///
/// `Debug` is deliberately omitted because `Invalid` may retain exact output.
#[derive(Serialize)]
#[serde(tag = "outcome", content = "value", rename_all = "snake_case")]
pub(crate) enum JudgeAttemptOutcomeV1 {
    /// A strict version-1 judge result.
    Valid(PairwiseJudgeResultV1),
    /// Readable but invalid judge output with bounded evidence.
    Invalid(InvalidJudgeOutputV1),
    /// Local configuration prevented safe validation.
    Operational(JudgeOutputOperationalFailureV1),
}

/// Ordinal of one judge invocation in the initial-plus-repair sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeAttemptOrdinalV1 {
    /// Initial judge invocation, ordinal zero.
    Initial,
    /// Sole permitted repair invocation, ordinal one.
    Repair,
}

impl JudgeAttemptOrdinalV1 {
    /// Return the durable numeric ordinal.
    pub(crate) const fn as_u8(self) -> u8 {
        match self {
            Self::Initial => 0,
            Self::Repair => 1,
        }
    }
}

/// One non-cloneable judge attempt and its validated outcome.
///
/// `Debug` is deliberately omitted because the outcome may retain exact output.
#[derive(Serialize)]
pub(crate) struct JudgeAttemptV1 {
    ordinal: JudgeAttemptOrdinalV1,
    outcome: JudgeAttemptOutcomeV1,
}

impl JudgeAttemptV1 {
    /// Wrap an initial judge outcome.
    pub(crate) fn initial(outcome: JudgeAttemptOutcomeV1) -> Self {
        Self {
            ordinal: JudgeAttemptOrdinalV1::Initial,
            outcome,
        }
    }

    /// Wrap the sole repair judge outcome.
    pub(crate) fn repair(outcome: JudgeAttemptOutcomeV1) -> Self {
        Self {
            ordinal: JudgeAttemptOrdinalV1::Repair,
            outcome,
        }
    }

    /// Return this attempt's ordinal.
    pub(crate) const fn ordinal(&self) -> JudgeAttemptOrdinalV1 {
        self.ordinal
    }

    /// Return this attempt's validated outcome.
    pub(crate) const fn outcome(&self) -> &JudgeAttemptOutcomeV1 {
        &self.outcome
    }
}

/// Progress after resolving the initial attempt and any supplied repair.
///
/// `Debug` is deliberately omitted because a completed valid state contains a
/// judge rationale.
#[derive(Serialize)]
#[serde(tag = "progress", rename_all = "snake_case")]
pub(crate) enum JudgeAttemptProgressV1 {
    /// The initial output was invalid and may receive its sole repair.
    RepairRequired {
        /// Owned initial invalid attempt consumed by the repair transition.
        initial_attempt: JudgeAttemptV1,
    },
    /// No further judge invocation is permitted.
    Final {
        /// Typed terminal state.
        final_state: JudgeFinalStateV1,
    },
}

impl JudgeAttemptProgressV1 {
    /// Return whether the initial invalid output requires one repair.
    pub(crate) const fn is_repair_required(&self) -> bool {
        matches!(self, Self::RepairRequired { .. })
    }

    /// Return the terminal state, if resolution is complete.
    pub(crate) const fn final_state(&self) -> Option<&JudgeFinalStateV1> {
        match self {
            Self::RepairRequired { .. } => None,
            Self::Final { final_state } => Some(final_state),
        }
    }

    /// Consume repair-required progress into evidence and a single-use token.
    pub(crate) fn into_repair(
        self,
    ) -> Result<(JudgeRepairEvidenceV1, JudgeRepairPendingV1), JudgeAttemptResolutionErrorV1> {
        let Self::RepairRequired { initial_attempt } = self else {
            return Err(JudgeAttemptResolutionErrorV1::AlreadyFinal);
        };
        let evidence = build_judge_repair_evidence(initial_attempt)
            .map_err(JudgeAttemptResolutionErrorV1::RepairEvidenceRejected)?;
        Ok((
            evidence,
            JudgeRepairPendingV1 {
                state: "repair_pending",
            },
        ))
    }
}

/// Terminal outcome after the initial attempt and optional sole repair.
///
/// Invalid and operational variants contain no result or quality label.
/// `Debug` is deliberately omitted because valid variants contain a rationale.
#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(crate) enum JudgeFinalStateV1 {
    /// The initial judge output validated successfully.
    InitialValid {
        /// Strict validated result.
        result: PairwiseJudgeResultV1,
    },
    /// The repair output validated successfully.
    RepairedValid {
        /// Strict validated result.
        result: PairwiseJudgeResultV1,
    },
    /// Both readable outputs were invalid.
    JudgeOutputInvalid,
    /// The initial attempt ended operationally and received no repair.
    InitialOperational {
        /// Stable operational class.
        failure: JudgeOutputOperationalFailureV1,
    },
    /// The repair attempt ended operationally.
    RepairOperational {
        /// Stable operational class.
        failure: JudgeOutputOperationalFailureV1,
    },
}

impl JudgeFinalStateV1 {
    /// Return a validated result only for successful initial or repair output.
    pub(crate) const fn result(&self) -> Option<&PairwiseJudgeResultV1> {
        match self {
            Self::InitialValid { result } | Self::RepairedValid { result } => Some(result),
            Self::JudgeOutputInvalid
            | Self::InitialOperational { .. }
            | Self::RepairOperational { .. } => None,
        }
    }

    /// Return the stable operational failure for operational terminal states.
    pub(crate) const fn operational_failure(&self) -> Option<JudgeOutputOperationalFailureV1> {
        match self {
            Self::InitialOperational { failure } | Self::RepairOperational { failure } => {
                Some(*failure)
            }
            Self::InitialValid { .. } | Self::RepairedValid { .. } | Self::JudgeOutputInvalid => {
                None
            }
        }
    }
}

/// Stable reason an attempt sequence cannot be resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeAttemptResolutionErrorV1 {
    /// The first supplied attempt was not ordinal zero.
    ExpectedInitialAttempt,
    /// The second supplied attempt was not ordinal one.
    ExpectedRepairAttempt,
    /// A completed sequence cannot start a repair.
    AlreadyFinal,
    /// The repair evidence invariant failed.
    RepairEvidenceRejected(JudgeRepairRejectionV1),
}

/// Single-use proof that an initial invalid output authorized one repair.
///
/// The private field and lack of `Clone` prevent constructing or reusing this
/// token outside the repair-required transition.
#[derive(Serialize)]
pub(crate) struct JudgeRepairPendingV1 {
    state: &'static str,
}

impl JudgeRepairPendingV1 {
    /// Consume the sole repair attempt into a terminal state.
    pub(crate) fn resolve(
        self,
        repair: JudgeAttemptV1,
    ) -> Result<JudgeAttemptProgressV1, JudgeAttemptResolutionErrorV1> {
        if repair.ordinal != JudgeAttemptOrdinalV1::Repair {
            return Err(JudgeAttemptResolutionErrorV1::ExpectedRepairAttempt);
        }
        let final_state = match repair.outcome {
            JudgeAttemptOutcomeV1::Valid(result) => JudgeFinalStateV1::RepairedValid { result },
            JudgeAttemptOutcomeV1::Invalid(_) => JudgeFinalStateV1::JudgeOutputInvalid,
            JudgeAttemptOutcomeV1::Operational(failure) => {
                JudgeFinalStateV1::RepairOperational { failure }
            }
        };
        Ok(JudgeAttemptProgressV1::Final { final_state })
    }
}

/// Consume one initial attempt into repair-required progress or a terminal state.
pub(crate) fn resolve_judge_attempt_progress(
    initial: JudgeAttemptV1,
) -> Result<JudgeAttemptProgressV1, JudgeAttemptResolutionErrorV1> {
    if initial.ordinal != JudgeAttemptOrdinalV1::Initial {
        return Err(JudgeAttemptResolutionErrorV1::ExpectedInitialAttempt);
    }
    match initial.outcome {
        JudgeAttemptOutcomeV1::Valid(result) => Ok(JudgeAttemptProgressV1::Final {
            final_state: JudgeFinalStateV1::InitialValid { result },
        }),
        JudgeAttemptOutcomeV1::Operational(failure) => Ok(JudgeAttemptProgressV1::Final {
            final_state: JudgeFinalStateV1::InitialOperational { failure },
        }),
        JudgeAttemptOutcomeV1::Invalid(invalid) => Ok(JudgeAttemptProgressV1::RepairRequired {
            initial_attempt: JudgeAttemptV1 {
                ordinal: JudgeAttemptOrdinalV1::Initial,
                outcome: JudgeAttemptOutcomeV1::Invalid(invalid),
            },
        }),
    }
}

/// Stable reason an attempt cannot produce repair evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeRepairRejectionV1 {
    /// A valid output must proceed to evaluation, not repair.
    ValidOutput,
    /// An operational outcome must not trigger a paid repair call.
    OperationalFailure,
    /// Ordinal one cannot recursively construct another repair.
    RepairAttempt,
    /// The compiled Task 2 output schema was not valid UTF-8.
    InvalidOutputSchema,
}

/// Complete safe evidence used to build the sole repair request.
///
/// `Debug` is deliberately omitted because bounded invalid output may be
/// retained exactly.
#[derive(Serialize)]
pub(crate) struct JudgeRepairEvidenceV1 {
    output_schema: &'static str,
    invalid_output: SafeInvalidJudgeOutputV1,
    validation_issues: Vec<JudgeValidationIssueV1>,
}

impl JudgeRepairEvidenceV1 {
    /// Return the exact immutable Task 2 output-schema bytes.
    pub(crate) fn output_schema(&self) -> &str {
        self.output_schema
    }

    /// Return the secret-safe invalid-output representation.
    pub(crate) fn invalid_output(&self) -> &SafeInvalidJudgeOutputV1 {
        &self.invalid_output
    }

    /// Return the stable sorted validation issues.
    pub(crate) fn validation_issues(&self) -> &[JudgeValidationIssueV1] {
        &self.validation_issues
    }
}

/// Consume an initial invalid attempt to construct its sole repair evidence.
pub(crate) fn build_judge_repair_evidence(
    attempt: JudgeAttemptV1,
) -> Result<JudgeRepairEvidenceV1, JudgeRepairRejectionV1> {
    if attempt.ordinal == JudgeAttemptOrdinalV1::Repair {
        return Err(JudgeRepairRejectionV1::RepairAttempt);
    }
    match attempt.outcome {
        JudgeAttemptOutcomeV1::Invalid(invalid) => {
            let output_schema = std::str::from_utf8(JUDGE_OUTPUT_SCHEMA_V1)
                .map_err(|_| JudgeRepairRejectionV1::InvalidOutputSchema)?;
            Ok(JudgeRepairEvidenceV1 {
                output_schema,
                invalid_output: invalid.output,
                validation_issues: invalid.issues,
            })
        }
        JudgeAttemptOutcomeV1::Valid(_) => Err(JudgeRepairRejectionV1::ValidOutput),
        JudgeAttemptOutcomeV1::Operational(_) => Err(JudgeRepairRejectionV1::OperationalFailure),
    }
}

struct TopLevelFieldVisitor;

impl<'de> Visitor<'de> for TopLevelFieldVisitor {
    type Value = BTreeSet<String>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a top-level JSON object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut duplicates = BTreeSet::new();
        while let Some(field) = map.next_key::<String>()? {
            if !seen.insert(field.clone()) {
                duplicates.insert(field.clone());
            }
            map.next_value::<IgnoredAny>()?;
        }
        Ok(duplicates)
    }
}

fn duplicate_top_level_fields(raw: &str) -> BTreeSet<String> {
    if raw.trim_start().as_bytes().first() != Some(&b'{') {
        return BTreeSet::new();
    }
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    let Ok(duplicates) = deserializer.deserialize_map(TopLevelFieldVisitor) else {
        return BTreeSet::new();
    };
    if deserializer.end().is_err() {
        return BTreeSet::new();
    }
    duplicates
}

fn scan_malformed_output(text: &str) -> bool {
    if scan_sensitive_text_once(text) {
        return true;
    }
    let normalized = tolerant_unescape_normal_form(text);
    normalized != text && scan_sensitive_text_once(&normalized)
}

fn scan_sensitive_text_once(text: &str) -> bool {
    if contains_sensitive_free_text(text) {
        return true;
    }
    let bytes = text.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        match bytes[cursor] {
            quote @ (b'"' | b'\'') => {
                if let Some(fragment) = decode_tolerant_quoted_key_probe(text, cursor, quote)
                    && is_sensitive_projection_key(fragment.decoded.trim())
                    && malformed_quoted_key_context(text, fragment.end)
                {
                    return true;
                }
                cursor += 1;
            }
            byte if is_bare_key_start(byte) => {
                let start = cursor;
                cursor += 1;
                while cursor < bytes.len() && is_bare_key_continue(bytes[cursor]) {
                    cursor += 1;
                }
                if text[cursor..]
                    .trim_start_matches(char::is_whitespace)
                    .starts_with(':')
                    && is_sensitive_projection_key(&text[start..cursor])
                {
                    return true;
                }
            }
            byte if byte.is_ascii() => cursor += 1,
            _ => {
                let Some(character) = text[cursor..].chars().next() else {
                    break;
                };
                cursor += character.len_utf8();
            }
        }
    }
    false
}

fn malformed_quoted_key_context(text: &str, start: usize) -> bool {
    let suffix = &text[start..];
    let bytes = suffix.as_bytes();
    let limit = suffix.len().min(MALFORMED_KEY_CONTEXT_BYTES);
    let mut cursor = 0;
    let mut allows_adjacent_junk = true;
    while cursor < limit {
        match bytes[cursor] {
            b':' => return true,
            byte if byte.is_ascii_whitespace() => {
                cursor += 1;
                allows_adjacent_junk = false;
            }
            b'/' if bytes.get(cursor + 1) == Some(&b'*') => {
                cursor += 2;
                let mut terminated = false;
                while cursor < limit {
                    if bytes[cursor] == b'*' && bytes.get(cursor + 1) == Some(&b'/') {
                        cursor += 2;
                        terminated = true;
                        break;
                    }
                    cursor += 1;
                }
                if !terminated {
                    return suffix.len() > limit;
                }
                allows_adjacent_junk = false;
            }
            b'/' if bytes.get(cursor + 1) == Some(&b'/') => {
                cursor += 2;
                while cursor < limit && !matches!(bytes[cursor], b'\n' | b'\r') {
                    cursor += 1;
                }
                allows_adjacent_junk = false;
            }
            b'\\' => {
                let Some(escaped_start) = cursor.checked_add(1) else {
                    return true;
                };
                let Some(&escaped) = bytes.get(escaped_start) else {
                    return true;
                };
                if escaped == b'u' {
                    if decode_hex_quad(bytes, escaped_start.saturating_add(1)).is_none() {
                        return true;
                    }
                    cursor = escaped_start.saturating_add(5);
                } else if escaped.is_ascii() {
                    cursor = escaped_start.saturating_add(1);
                } else {
                    let Some(character) = suffix[escaped_start..].chars().next() else {
                        return true;
                    };
                    cursor = escaped_start.saturating_add(character.len_utf8());
                }
                allows_adjacent_junk = false;
            }
            b',' | b'{' | b'}' | b'[' | b']' | b'"' | b'\'' => {
                return false;
            }
            byte if byte.is_ascii() && allows_adjacent_junk => {
                cursor += 1;
                while cursor < limit && is_bare_key_continue(bytes[cursor]) {
                    cursor += 1;
                }
                allows_adjacent_junk = false;
            }
            byte if byte.is_ascii() => return false,
            _ => {
                let Some(character) = suffix[cursor..].chars().next() else {
                    return true;
                };
                if character.is_whitespace() || allows_adjacent_junk {
                    cursor += character.len_utf8();
                    allows_adjacent_junk = false;
                } else {
                    return false;
                }
            }
        }
    }
    suffix.len() > limit
}

struct TolerantKeyProbe {
    decoded: String,
    end: usize,
}

fn decode_tolerant_quoted_key_probe(
    text: &str,
    start: usize,
    quote: u8,
) -> Option<TolerantKeyProbe> {
    let bytes = text.as_bytes();
    let mut decoded = String::new();
    let mut cursor = start.saturating_add(1);
    let limit = cursor
        .saturating_add(MALFORMED_KEY_PROBE_BYTES)
        .min(bytes.len());
    while cursor < limit {
        match bytes[cursor] {
            byte if byte == quote => {
                return Some(TolerantKeyProbe {
                    decoded,
                    end: cursor + 1,
                });
            }
            b'\\' => {
                cursor = decode_tolerant_escape(text, cursor, &mut decoded).0;
                if cursor > limit {
                    return None;
                }
            }
            byte if byte.is_ascii() => {
                decoded.push(char::from(byte));
                cursor += 1;
            }
            _ => {
                let Some(character) = text[cursor..].chars().next() else {
                    break;
                };
                decoded.push(character);
                cursor += character.len_utf8();
            }
        }
    }
    None
}

fn tolerant_unescape_normal_form(text: &str) -> String {
    let mut pending = VecDeque::from(text.as_bytes().to_vec());
    let mut decoded = Vec::with_capacity(text.len());
    while let Some(byte) = pending.pop_front() {
        if byte != b'\\' {
            decoded.push(byte);
            continue;
        }

        let Some(escaped) = pending.pop_front() else {
            decoded.push(b'\\');
            break;
        };
        match escaped {
            b'\\' => pending.push_front(b'\\'),
            b'"' => decoded.push(b'"'),
            b'\'' => decoded.push(b'\''),
            b'/' => decoded.push(b'/'),
            b'b' => decoded.push(0x08),
            b'f' => decoded.push(0x0c),
            b'n' => decoded.push(b'\n'),
            b'r' => decoded.push(b'\r'),
            b't' => decoded.push(b'\t'),
            b'u' => {
                if let Some(character) = pop_unicode_escape(&mut pending) {
                    push_normalized_character(character, &mut pending, &mut decoded);
                } else {
                    decoded.push(b'u');
                }
            }
            byte => decoded.push(byte),
        }
    }
    String::from_utf8(decoded).unwrap_or_else(|_| text.to_string())
}

fn pop_unicode_escape(pending: &mut VecDeque<u8>) -> Option<char> {
    let first = peek_hex_quad(pending, 0)?;
    for _ in 0..4 {
        pending.pop_front();
    }
    let scalar = if (0xd800..=0xdbff).contains(&first) {
        if pending.front() != Some(&b'\\') || pending.get(1) != Some(&b'u') {
            return Some('\u{fffd}');
        }
        let second = peek_hex_quad(pending, 2)?;
        if !(0xdc00..=0xdfff).contains(&second) {
            return Some('\u{fffd}');
        }
        for _ in 0..6 {
            pending.pop_front();
        }
        let high = u32::from(first) - 0xd800;
        let low = u32::from(second) - 0xdc00;
        0x1_0000 + (high << 10) + low
    } else {
        u32::from(first)
    };
    Some(char::from_u32(scalar).unwrap_or('\u{fffd}'))
}

fn peek_hex_quad(pending: &VecDeque<u8>, offset: usize) -> Option<u16> {
    (0..4).try_fold(0u16, |value, index| {
        let digit = pending
            .get(offset.checked_add(index)?)?
            .to_ascii_lowercase();
        let digit = match digit {
            b'0'..=b'9' => u16::from(digit - b'0'),
            b'a'..=b'f' => u16::from(digit - b'a' + 10),
            _ => return None,
        };
        value.checked_mul(16)?.checked_add(digit)
    })
}

fn push_normalized_character(character: char, pending: &mut VecDeque<u8>, decoded: &mut Vec<u8>) {
    if character == '\\' {
        pending.push_front(b'\\');
        return;
    }
    let mut encoded = [0; 4];
    decoded.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
}

fn decode_tolerant_escape(text: &str, slash: usize, output: &mut String) -> (usize, bool) {
    let bytes = text.as_bytes();
    let Some(&escaped) = bytes.get(slash.saturating_add(1)) else {
        output.push('\\');
        return (bytes.len(), false);
    };
    let after_escape = slash.saturating_add(2);
    match escaped {
        b'"' => output.push('"'),
        b'\'' => output.push('\''),
        b'\\' => output.push('\\'),
        b'/' => output.push('/'),
        b'b' => output.push('\u{0008}'),
        b'f' => output.push('\u{000c}'),
        b'n' => output.push('\n'),
        b'r' => output.push('\r'),
        b't' => output.push('\t'),
        b'u' => {
            if let Some((character, end)) = decode_unicode_escape(bytes, slash) {
                output.push(character);
                return (end, true);
            }
            output.push('u');
        }
        byte if byte.is_ascii() => output.push(char::from(byte)),
        _ => {
            let Some(character_start) = slash.checked_add(1) else {
                return (bytes.len(), true);
            };
            let Some(character) = text[character_start..].chars().next() else {
                return (bytes.len(), true);
            };
            output.push(character);
            let end = character_start
                .checked_add(character.len_utf8())
                .unwrap_or(bytes.len());
            return (end, true);
        }
    }
    (after_escape, true)
}

fn decode_unicode_escape(bytes: &[u8], slash: usize) -> Option<(char, usize)> {
    let digits = slash.checked_add(2)?;
    let first = decode_hex_quad(bytes, digits)?;
    let first_end = digits.checked_add(4)?;
    let scalar = if (0xd800..=0xdbff).contains(&first) {
        let marker_end = first_end.checked_add(1)?;
        if bytes.get(first_end) != Some(&b'\\') || bytes.get(marker_end) != Some(&b'u') {
            return Some(('\u{fffd}', first_end));
        }
        let second_digits = first_end.checked_add(2)?;
        let second = decode_hex_quad(bytes, second_digits)?;
        if !(0xdc00..=0xdfff).contains(&second) {
            return Some(('\u{fffd}', first_end));
        }
        let high = u32::from(first) - 0xd800;
        let low = u32::from(second) - 0xdc00;
        let scalar = 0x1_0000 + (high << 10) + low;
        let second_end = second_digits.checked_add(4)?;
        return char::from_u32(scalar).map(|character| (character, second_end));
    } else {
        u32::from(first)
    };
    Some((char::from_u32(scalar).unwrap_or('\u{fffd}'), first_end))
}

fn decode_hex_quad(bytes: &[u8], start: usize) -> Option<u16> {
    let end = start.checked_add(4)?;
    bytes.get(start..end)?.iter().try_fold(0u16, |value, byte| {
        let digit = byte.to_ascii_lowercase();
        let digit = match digit {
            b'0'..=b'9' => u16::from(digit - b'0'),
            b'a'..=b'f' => u16::from(digit - b'a' + 10),
            _ => return None,
        };
        value.checked_mul(16)?.checked_add(digit)
    })
}

fn is_bare_key_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$')
}

fn is_bare_key_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'$' | b'.')
}

/// Strictly validate one readable assistant output against the Task 2 contract.
pub(crate) fn validate_pairwise_judge_output(
    raw: &str,
    max_rationale_bytes: usize,
) -> JudgeAttemptOutcomeV1 {
    let Some(raw_output_bound) = checked_raw_output_bound(max_rationale_bytes) else {
        return JudgeAttemptOutcomeV1::Operational(
            JudgeOutputOperationalFailureV1::InvalidRationaleBound,
        );
    };
    if raw.len() > raw_output_bound {
        return invalid_outcome(
            raw,
            InvalidRetention::Oversized,
            BTreeSet::from([JudgeValidationIssueV1::new("output.too_large")]),
        );
    }

    let raw_scan = scan_malformed_output(raw);
    let duplicate_fields = duplicate_top_level_fields(raw);
    let value = match serde_json::from_str::<Json>(raw) {
        Ok(value) => value,
        Err(_) => {
            let mut issues = BTreeSet::from([JudgeValidationIssueV1::new("output.invalid_json")]);
            if raw_scan {
                issues.insert(JudgeValidationIssueV1::new(CREDENTIAL_ISSUE));
            }
            return invalid_outcome(
                raw,
                if raw_scan {
                    InvalidRetention::CredentialBearing
                } else {
                    InvalidRetention::Retained
                },
                issues,
            );
        }
    };

    let credential_bearing = raw_scan || json_contains_sensitive_text(&value);
    let Some(object) = value.as_object() else {
        let mut issues =
            BTreeSet::from([JudgeValidationIssueV1::new("output.type.object_required")]);
        if credential_bearing {
            issues.insert(JudgeValidationIssueV1::new(CREDENTIAL_ISSUE));
        }
        return invalid_outcome(raw, retention_for_credential(credential_bearing), issues);
    };

    validate_object(
        raw,
        object,
        max_rationale_bytes,
        credential_bearing,
        &duplicate_fields,
    )
}

fn checked_raw_output_bound(max_rationale_bytes: usize) -> Option<usize> {
    if max_rationale_bytes == 0 || max_rationale_bytes > JUDGE_MAX_RATIONALE_BYTES {
        return None;
    }
    max_rationale_bytes.checked_add(JUDGE_OUTPUT_OVERHEAD_BYTES)
}

fn validate_object(
    raw: &str,
    object: &Map<String, Json>,
    max_rationale_bytes: usize,
    credential_bearing: bool,
    duplicate_fields: &BTreeSet<String>,
) -> JudgeAttemptOutcomeV1 {
    let mut issues = BTreeSet::new();
    for field in duplicate_fields {
        let issue = if REQUIRED_FIELDS.contains(&field.as_str()) {
            format!("field.{field}.duplicate")
        } else {
            format!(
                "field.unknown_duplicate.sha256.{}",
                sha256_hex(field.as_bytes())
            )
        };
        issues.insert(JudgeValidationIssueV1::new(issue));
    }
    for field in REQUIRED_FIELDS {
        if !object.contains_key(field) {
            issues.insert(JudgeValidationIssueV1::new(format!(
                "field.{field}.missing"
            )));
        }
    }
    let mut unknown_fields = object
        .keys()
        .filter(|field| !REQUIRED_FIELDS.contains(&field.as_str()))
        .collect::<Vec<_>>();
    unknown_fields.sort_unstable();
    for field in unknown_fields {
        issues.insert(JudgeValidationIssueV1::new(format!(
            "field.unknown.sha256.{}",
            sha256_hex(field.as_bytes())
        )));
    }

    let response_equivalence = validate_score(object, "response_equivalence", &mut issues);
    let trajectory_equivalence = validate_score(object, "trajectory_equivalence", &mut issues);
    let judge_confidence = validate_score(object, "judge_confidence", &mut issues);
    let hard_failures = validate_hard_failures(object, &mut issues);
    let rationale = validate_rationale(object, max_rationale_bytes, &mut issues);
    if credential_bearing {
        issues.insert(JudgeValidationIssueV1::new(CREDENTIAL_ISSUE));
    }

    if issues.is_empty()
        && let (
            Some(response_equivalence),
            Some(trajectory_equivalence),
            Some(judge_confidence),
            Some(hard_failures),
            Some(rationale),
        ) = (
            response_equivalence,
            trajectory_equivalence,
            judge_confidence,
            hard_failures,
            rationale,
        )
    {
        return JudgeAttemptOutcomeV1::Valid(PairwiseJudgeResultV1 {
            response_equivalence,
            trajectory_equivalence,
            judge_confidence,
            hard_failures,
            rationale,
        });
    }

    invalid_outcome(raw, retention_for_credential(credential_bearing), issues)
}

fn validate_score(
    object: &Map<String, Json>,
    field: &str,
    issues: &mut BTreeSet<JudgeValidationIssueV1>,
) -> Option<f64> {
    let value = object.get(field)?;
    let Json::Number(number) = value else {
        issues.insert(JudgeValidationIssueV1::new(format!(
            "field.{field}.type.number_required"
        )));
        return None;
    };
    let Some(value) = number.as_f64().filter(|value| value.is_finite()) else {
        issues.insert(JudgeValidationIssueV1::new(format!(
            "field.{field}.number.finite_required"
        )));
        return None;
    };
    if !(0.0..=1.0).contains(&value) {
        issues.insert(JudgeValidationIssueV1::new(format!(
            "field.{field}.number.range_0_1_required"
        )));
        return None;
    }
    Some(value)
}

fn validate_hard_failures(
    object: &Map<String, Json>,
    issues: &mut BTreeSet<JudgeValidationIssueV1>,
) -> Option<Vec<JudgeHardFailureV1>> {
    let value = object.get("hard_failures")?;
    let Json::Array(values) = value else {
        issues.insert(JudgeValidationIssueV1::new(
            "field.hard_failures.type.array_required",
        ));
        return None;
    };
    if values.len() > 4 {
        issues.insert(JudgeValidationIssueV1::new(
            "field.hard_failures.items.too_many",
        ));
    }
    let mut failures = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let Json::String(value) = value else {
            issues.insert(JudgeValidationIssueV1::new(format!(
                "field.hard_failures.item.{index:05}.type.string_required"
            )));
            continue;
        };
        let Some(failure) = JudgeHardFailureV1::parse(value) else {
            issues.insert(JudgeValidationIssueV1::new(format!(
                "field.hard_failures.item.{index:05}.value.unknown"
            )));
            continue;
        };
        if !failures.insert(failure) {
            issues.insert(JudgeValidationIssueV1::new(format!(
                "field.hard_failures.item.{index:05}.value.duplicate"
            )));
        }
    }
    Some(failures.into_iter().collect())
}

fn validate_rationale(
    object: &Map<String, Json>,
    max_rationale_bytes: usize,
    issues: &mut BTreeSet<JudgeValidationIssueV1>,
) -> Option<String> {
    let value = object.get("rationale")?;
    let Json::String(value) = value else {
        issues.insert(JudgeValidationIssueV1::new(
            "field.rationale.type.string_required",
        ));
        return None;
    };
    if value.is_empty() {
        issues.insert(JudgeValidationIssueV1::new(
            "field.rationale.text.nonempty_required",
        ));
    }
    if value.len() > max_rationale_bytes {
        issues.insert(JudgeValidationIssueV1::new(
            "field.rationale.text.too_large",
        ));
    }
    if value.nfc().ne(value.chars()) {
        issues.insert(JudgeValidationIssueV1::new(
            "field.rationale.text.nfc_required",
        ));
    }
    if value.chars().any(char::is_control) {
        issues.insert(JudgeValidationIssueV1::new(
            "field.rationale.text.control_free_required",
        ));
    }
    Some(value.clone())
}

fn json_contains_sensitive_text(value: &Json) -> bool {
    match value {
        Json::String(value) => scan_malformed_output(value),
        Json::Array(values) => values.iter().any(json_contains_sensitive_text),
        Json::Object(values) => values.iter().any(|(key, value)| {
            is_sensitive_projection_key(key)
                || contains_sensitive_free_text(key)
                || scan_malformed_output(key)
                || json_contains_sensitive_text(value)
        }),
        Json::Null | Json::Bool(_) | Json::Number(_) => false,
    }
}

enum InvalidRetention {
    Retained,
    Oversized,
    CredentialBearing,
}

fn retention_for_credential(credential_bearing: bool) -> InvalidRetention {
    if credential_bearing {
        InvalidRetention::CredentialBearing
    } else {
        InvalidRetention::Retained
    }
}

fn invalid_outcome(
    raw: &str,
    retention: InvalidRetention,
    issues: BTreeSet<JudgeValidationIssueV1>,
) -> JudgeAttemptOutcomeV1 {
    let sha256 = sha256_hex(raw.as_bytes());
    let Ok(byte_count) = u64::try_from(raw.len()) else {
        return JudgeAttemptOutcomeV1::Operational(
            JudgeOutputOperationalFailureV1::OutputSizeUnrepresentable,
        );
    };
    let output = match retention {
        InvalidRetention::Retained => SafeInvalidJudgeOutputV1::Retained {
            output: raw.to_string(),
            sha256,
            byte_count,
        },
        InvalidRetention::Oversized => SafeInvalidJudgeOutputV1::Redacted {
            marker: InvalidJudgeOutputMarkerV1::Oversized,
            sha256,
            byte_count,
        },
        InvalidRetention::CredentialBearing => SafeInvalidJudgeOutputV1::Redacted {
            marker: InvalidJudgeOutputMarkerV1::CredentialBearing,
            sha256,
            byte_count,
        },
    };
    JudgeAttemptOutcomeV1::Invalid(InvalidJudgeOutputV1 {
        output,
        issues: bounded_issues(issues),
    })
}

fn bounded_issues(issues: BTreeSet<JudgeValidationIssueV1>) -> Vec<JudgeValidationIssueV1> {
    if issues.len() <= MAX_VALIDATION_ISSUES {
        return issues.into_iter().collect();
    }

    let credential_issue = issues
        .iter()
        .find(|issue| issue.as_str() == CREDENTIAL_ISSUE)
        .cloned();
    let mut retained = issues
        .into_iter()
        .take(MAX_VALIDATION_ISSUES - 1)
        .collect::<BTreeSet<_>>();
    if let Some(credential_issue) = credential_issue
        && !retained.contains(&credential_issue)
    {
        retained.pop_last();
        retained.insert(credential_issue);
    }
    retained.insert(JudgeValidationIssueV1::new(TRUNCATED_ISSUES));
    retained.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value as Json, json};

    use super::{
        CREDENTIAL_ISSUE, InvalidJudgeOutputMarkerV1, JUDGE_OUTPUT_OVERHEAD_BYTES,
        JudgeAttemptOrdinalV1, JudgeAttemptOutcomeV1, JudgeAttemptResolutionErrorV1,
        JudgeAttemptV1, JudgeFinalStateV1, JudgeHardFailureV1, JudgeOutputOperationalFailureV1,
        JudgeRepairRejectionV1, MAX_VALIDATION_ISSUE_BYTES, MAX_VALIDATION_ISSUES,
        PairwiseJudgeResultV1, TRUNCATED_ISSUES, build_judge_repair_evidence,
        resolve_judge_attempt_progress, validate_pairwise_judge_output,
    };
    use crate::config::{JUDGE_MAX_RATIONALE_BYTES, JUDGE_OUTPUT_SCHEMA_V1};
    use crate::fingerprint::sha256_hex;

    const RATIONALE_LIMIT: usize = 4_096;

    fn output_with_rationale(rationale: &str) -> String {
        json!({
            "response_equivalence": 0.9,
            "trajectory_equivalence": 0.8,
            "judge_confidence": 0.95,
            "hard_failures": [],
            "rationale": rationale,
        })
        .to_string()
    }

    fn valid(outcome: &JudgeAttemptOutcomeV1) -> &PairwiseJudgeResultV1 {
        match outcome {
            JudgeAttemptOutcomeV1::Valid(result) => result,
            JudgeAttemptOutcomeV1::Invalid(_) | JudgeAttemptOutcomeV1::Operational(_) => {
                panic!("expected valid output")
            }
        }
    }

    fn invalid(outcome: &JudgeAttemptOutcomeV1) -> &super::InvalidJudgeOutputV1 {
        match outcome {
            JudgeAttemptOutcomeV1::Invalid(invalid) => invalid,
            JudgeAttemptOutcomeV1::Valid(_) | JudgeAttemptOutcomeV1::Operational(_) => {
                panic!("expected invalid output")
            }
        }
    }

    fn issue_texts(outcome: &JudgeAttemptOutcomeV1) -> Vec<&str> {
        invalid(outcome)
            .issues()
            .iter()
            .map(super::JudgeValidationIssueV1::as_str)
            .collect()
    }

    #[test]
    fn valid_output_accepts_score_boundaries_and_canonicalizes_hard_failures() {
        let raw = json!({
            "response_equivalence": 0.0,
            "trajectory_equivalence": 1.0,
            "judge_confidence": 1.0,
            "hard_failures": [
                "malformed_candidate",
                "safety",
                "response_schema",
                "tool_contract"
            ],
            "rationale": "All declared checks completed."
        })
        .to_string();
        let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
        let result = valid(&outcome);
        assert_eq!(result.response_equivalence(), 0.0);
        assert_eq!(result.trajectory_equivalence(), 1.0);
        assert_eq!(result.judge_confidence(), 1.0);
        assert_eq!(
            result.hard_failures(),
            [
                JudgeHardFailureV1::ToolContract,
                JudgeHardFailureV1::ResponseSchema,
                JudgeHardFailureV1::Safety,
                JudgeHardFailureV1::MalformedCandidate,
            ]
        );
        assert_eq!(result.rationale(), "All declared checks completed.");
    }

    #[test]
    fn task2_schema_and_manual_validator_contract_stay_in_lockstep() {
        let schema: Json = serde_json::from_slice(JUDGE_OUTPUT_SCHEMA_V1).unwrap();
        let properties = schema["properties"].as_object().unwrap();
        assert_eq!(
            properties.keys().map(String::as_str).collect::<Vec<_>>(),
            super::REQUIRED_FIELDS
        );
        assert_eq!(
            schema["required"]
                .as_array()
                .unwrap()
                .iter()
                .map(Json::as_str)
                .collect::<Option<Vec<_>>>()
                .unwrap(),
            [
                "response_equivalence",
                "trajectory_equivalence",
                "judge_confidence",
                "hard_failures",
                "rationale",
            ]
        );
        assert_eq!(schema["additionalProperties"], false);
        for field in [
            "response_equivalence",
            "trajectory_equivalence",
            "judge_confidence",
        ] {
            assert_eq!(properties[field]["minimum"], 0);
            assert_eq!(properties[field]["maximum"], 1);
        }

        let hard_failure_items = &properties["hard_failures"]["items"];
        assert_eq!(properties["hard_failures"]["maxItems"], 4);
        assert!(
            properties["hard_failures"].get("uniqueItems").is_none(),
            "the provider schema must stay inside the OpenAI strict-schema subset"
        );
        assert_eq!(
            hard_failure_items["enum"],
            serde_json::to_value([
                JudgeHardFailureV1::ToolContract,
                JudgeHardFailureV1::ResponseSchema,
                JudgeHardFailureV1::Safety,
                JudgeHardFailureV1::MalformedCandidate,
            ])
            .unwrap()
        );
        assert_eq!(properties["rationale"]["minLength"], 1);
        assert_eq!(
            properties["rationale"]["maxLength"],
            JUDGE_MAX_RATIONALE_BYTES
        );
    }

    #[test]
    fn unknown_missing_and_type_issues_are_stable_and_sorted() {
        let first =
            r#"{"z_unknown":1,"response_equivalence":"bad","hard_failures":{},"a_unknown":2}"#;
        let second =
            r#"{"a_unknown":2,"hard_failures":{},"response_equivalence":"bad","z_unknown":1}"#;
        let first = validate_pairwise_judge_output(first, RATIONALE_LIMIT);
        let second = validate_pairwise_judge_output(second, RATIONALE_LIMIT);
        let first_issues = issue_texts(&first);
        let second_issues = issue_texts(&second);
        assert_eq!(first_issues, second_issues);
        assert!(first_issues.windows(2).all(|pair| pair[0] < pair[1]));
        for expected in [
            "field.judge_confidence.missing",
            "field.rationale.missing",
            "field.response_equivalence.type.number_required",
            "field.trajectory_equivalence.missing",
            "field.hard_failures.type.array_required",
        ] {
            assert!(first_issues.contains(&expected), "missing {expected}");
        }
        assert_eq!(
            first_issues
                .iter()
                .filter(|issue| issue.starts_with("field.unknown.sha256."))
                .count(),
            2
        );
    }

    #[test]
    fn every_duplicate_known_top_level_field_is_invalid() {
        let fields = [
            ("response_equivalence", "0.9"),
            ("trajectory_equivalence", "0.8"),
            ("judge_confidence", "0.95"),
            ("hard_failures", "[]"),
            ("rationale", r#""safe""#),
        ];
        for (duplicate, _) in fields {
            let mut entries = Vec::new();
            for (field, value) in fields {
                entries.push(format!(r#""{field}":{value}"#));
                if field == duplicate {
                    entries.push(format!(r#""{field}":{value}"#));
                }
            }
            let raw = format!("{{{}}}", entries.join(","));
            let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
            let invalid = invalid(&outcome);
            assert_eq!(invalid.output().retained_output(), Some(raw.as_str()));
            let expected = format!("field.{duplicate}.duplicate");
            assert!(
                issue_texts(&outcome).contains(&expected.as_str()),
                "missing duplicate issue for {duplicate}"
            );
        }
    }

    #[test]
    fn malformed_nonfinite_typed_and_out_of_range_scores_are_invalid() {
        let fixtures = [
            r#"{"response_equivalence":NaN,"trajectory_equivalence":0.5,"judge_confidence":0.5,"hard_failures":[],"rationale":"safe"}"#,
            r#"{"response_equivalence":1e400,"trajectory_equivalence":0.5,"judge_confidence":0.5,"hard_failures":[],"rationale":"safe"}"#,
            r#"{"response_equivalence":"0.5","trajectory_equivalence":0.5,"judge_confidence":0.5,"hard_failures":[],"rationale":"safe"}"#,
            r#"{"response_equivalence":-0.01,"trajectory_equivalence":0.5,"judge_confidence":0.5,"hard_failures":[],"rationale":"safe"}"#,
            r#"{"response_equivalence":1.01,"trajectory_equivalence":0.5,"judge_confidence":0.5,"hard_failures":[],"rationale":"safe"}"#,
        ];
        for raw in fixtures {
            assert!(matches!(
                validate_pairwise_judge_output(raw, RATIONALE_LIMIT),
                JudgeAttemptOutcomeV1::Invalid(_)
            ));
        }
    }

    #[test]
    fn hard_failures_reject_unknown_duplicate_nonstring_and_too_many_values() {
        for hard_failures in [
            json!(["safety", "safety"]),
            json!(["not_published"]),
            json!([7]),
            json!([
                "tool_contract",
                "response_schema",
                "safety",
                "malformed_candidate",
                "safety"
            ]),
        ] {
            let raw = json!({
                "response_equivalence": 0.5,
                "trajectory_equivalence": 0.5,
                "judge_confidence": 0.5,
                "hard_failures": hard_failures,
                "rationale": "safe"
            })
            .to_string();
            assert!(matches!(
                validate_pairwise_judge_output(&raw, RATIONALE_LIMIT),
                JudgeAttemptOutcomeV1::Invalid(_)
            ));
        }
    }

    #[test]
    fn rationale_requires_nonempty_bounded_nfc_control_free_utf8() {
        let fixtures = [
            (output_with_rationale(""), 8usize),
            (output_with_rationale("abcde"), 4usize),
            (output_with_rationale("e\u{301}"), 8usize),
            (output_with_rationale("line\nbreak"), 32usize),
        ];
        for (raw, limit) in fixtures {
            assert!(matches!(
                validate_pairwise_judge_output(&raw, limit),
                JudgeAttemptOutcomeV1::Invalid(_)
            ));
        }

        let exact_utf8 = output_with_rationale("éé");
        assert!(matches!(
            validate_pairwise_judge_output(&exact_utf8, 4),
            JudgeAttemptOutcomeV1::Valid(_)
        ));
        assert!(matches!(
            validate_pairwise_judge_output(&output_with_rationale("   "), 3),
            JudgeAttemptOutcomeV1::Valid(_)
        ));
    }

    #[test]
    fn credential_bearing_output_is_hash_only_while_near_misses_remain_valid() {
        let credentials = [
            "prefix sk-12345678901234567890 suffix",
            "JWT eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123.",
            "key AKIA1234567890ABCDEF",
            "-----BEGIN PRIVATE KEY----- material -----END PRIVATE KEY-----",
            "Bearer abcDEF0123456789xyz-_",
            "Bearer : abcdefghijklmnopqrstuv",
            "token=not-safe",
            "api_key = not-safe",
            r#"{"api_key":"opaque"}"#,
        ];
        for credential in credentials {
            let raw = output_with_rationale(credential);
            let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
            let invalid = invalid(&outcome);
            assert_eq!(
                invalid.output().marker(),
                Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
            );
            assert_eq!(invalid.output().retained_output(), None);
            assert_eq!(invalid.output().sha256(), sha256_hex(raw.as_bytes()));
            assert_eq!(
                invalid.output().byte_count(),
                u64::try_from(raw.len()).unwrap()
            );
            assert!(issue_texts(&outcome).contains(&CREDENTIAL_ISSUE));
            assert!(!serde_json::to_string(invalid).unwrap().contains(credential));
        }

        for near_miss in [
            "a basic explanation",
            "the bearer of responsibility",
            "sketch a token budget",
            "use the authorization policy",
            "a secretariat meeting",
            "AKIA is only a four-letter example",
        ] {
            assert!(matches!(
                validate_pairwise_judge_output(&output_with_rationale(near_miss), RATIONALE_LIMIT),
                JudgeAttemptOutcomeV1::Valid(_)
            ));
        }

        let escaped = r#"{"response_equivalence":0.9,"trajectory_equivalence":0.8,"judge_confidence":0.95,"hard_failures":[],"rationale":"Bearer\u0020abcDEF0123456789xyz-_"}"#;
        let outcome = validate_pairwise_judge_output(escaped, RATIONALE_LIMIT);
        assert_eq!(
            invalid(&outcome).output().marker(),
            Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
        );

        let sensitive_key = r#"{"api_key":"not-safe"}"#;
        let outcome = validate_pairwise_judge_output(sensitive_key, RATIONALE_LIMIT);
        assert_eq!(
            invalid(&outcome).output().marker(),
            Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
        );
        assert!(
            !serde_json::to_string(invalid(&outcome))
                .unwrap()
                .contains("not-safe")
        );
    }

    #[test]
    fn malformed_json_fragments_redact_escaped_credentials_and_sensitive_keys() {
        for raw in [
            r#"{"api_key":"opaque",}"#,
            r#"{"\u0061pi_key":"opaque",}"#,
            r#"{"note":"\u0073k-12345678901234567890",}"#,
            r#"{"note":"Bearer\u0020abcDEF0123456789xyz-_",}"#,
            r#"{"note":"\u0073k-12345678901234567890"#,
            r#"{"note":"\u0073k-12345678901234567890\q"}"#,
            r#"{api_key:\"opaque\",}"#,
            r#"{'api_key':'opaque'}"#,
            r#""{\"api_key\":\"opaque\"}""#,
            r#"{\"api_key\"/*x*/:\"opaque\"}"#,
            r#"{\"api_key\"\q:\"opaque\"}"#,
        ] {
            let outcome = validate_pairwise_judge_output(raw, RATIONALE_LIMIT);
            let invalid = invalid(&outcome);
            assert_eq!(
                invalid.output().marker(),
                Some(InvalidJudgeOutputMarkerV1::CredentialBearing),
                "failed to redact {raw}"
            );
            assert_eq!(invalid.output().retained_output(), None);
            assert_eq!(invalid.output().sha256(), sha256_hex(raw.as_bytes()));
            assert!(issue_texts(&outcome).contains(&CREDENTIAL_ISSUE));
            let serialized = serde_json::to_string(invalid).unwrap();
            assert!(!serialized.contains("opaque"));
            assert!(!serialized.contains("12345678901234567890"));
            assert!(!serialized.contains("abcDEF0123456789xyz-_"));
        }

        for safe in [
            r#"{"note":"safe",}"#,
            r#"{"note":"sketch a safe explanation"#,
            r#"{"note":"opaque\q"}"#,
            r#"{note:\"safe\",}"#,
            r#"{'note':'safe'}"#,
            r#""{\"note\":\"safe\"}""#,
            r#"{\"note\"/*x*/:\"safe\"}"#,
            r#"{\"note\"\q:\"safe\"}"#,
            r#"{\"api_key\"/* contains: a colon */ \"safe\"}"#,
            r#"{\"api_key\",\"note\":\"safe\"}"#,
            r#"\"api_key\" is a field name: \"safe\""#,
        ] {
            let outcome = validate_pairwise_judge_output(safe, RATIONALE_LIMIT);
            assert_eq!(
                invalid(&outcome).output().retained_output(),
                Some(safe),
                "unexpected redaction for {safe}"
            );
        }
    }

    #[test]
    fn valid_credential_free_backslash_rationales_do_not_fail_closed() {
        let escaped_backslashes = r"\u005c".repeat(32);
        let raw = format!(
            r#"{{"response_equivalence":0.9,"trajectory_equivalence":0.8,"judge_confidence":0.95,"hard_failures":[],"rationale":"{escaped_backslashes}"}}"#
        );
        let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
        assert_eq!(valid(&outcome).rationale(), "\\".repeat(32));

        let literal_escapes = output_with_rationale(&escaped_backslashes);
        let outcome = validate_pairwise_judge_output(&literal_escapes, RATIONALE_LIMIT);
        assert_eq!(valid(&outcome).rationale(), escaped_backslashes);

        let safe_layers = format!("{}ordinary", r"\u005c".repeat(40));
        let safe_output = output_with_rationale(&safe_layers);
        let outcome = validate_pairwise_judge_output(&safe_output, RATIONALE_LIMIT);
        assert_eq!(valid(&outcome).rationale(), safe_layers);
    }

    #[test]
    fn normal_form_scan_finds_credentials_beyond_the_old_depth_limit() {
        let encoded_credential = format!("{}u0061pi_key:opaque", r"\u005c".repeat(40));
        let raw = output_with_rationale(&encoded_credential);
        let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
        let invalid = invalid(&outcome);
        assert_eq!(
            invalid.output().marker(),
            Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
        );
        assert_eq!(invalid.output().retained_output(), None);
        assert!(issue_texts(&outcome).contains(&CREDENTIAL_ISSUE));
        assert!(!serde_json::to_string(invalid).unwrap().contains("opaque"));
    }

    #[test]
    fn malformed_key_context_handles_unicode_whitespace_on_char_boundaries() {
        for whitespace in ['\u{00a0}', '\u{2003}'] {
            let raw = format!(r#"{{\"api_key\"{whitespace}:\"opaque\"}}"#);
            let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
            assert_eq!(
                invalid(&outcome).output().marker(),
                Some(InvalidJudgeOutputMarkerV1::CredentialBearing),
                "failed to redact key separated by {whitespace:?}"
            );

            for safe in [
                format!(r#"{{\"note\"{whitespace}:\"safe\"}}"#),
                format!(r#"\"api_key\"{whitespace}is a field name: \"safe\""#),
            ] {
                let outcome = validate_pairwise_judge_output(&safe, RATIONALE_LIMIT);
                assert_eq!(
                    invalid(&outcome).output().retained_output(),
                    Some(safe.as_str()),
                    "unexpected Unicode-whitespace redaction for {safe}"
                );
            }
        }
    }

    #[test]
    fn malformed_key_context_never_splits_escaped_unicode_scalars() {
        let boundary_padding = " ".repeat(super::MALFORMED_KEY_CONTEXT_BYTES - 1);
        let fixtures = [
            r#"{"api_key"\é:"opaque"}"#.to_string(),
            r#"{"api_key"\"#.to_string(),
            r#"{"api_key"\u00"#.to_string(),
            format!(r#"{{"api_key"{boundary_padding}\é:"opaque"}}"#),
            format!(r#"{{"api_key"{boundary_padding}\"#),
        ];

        for raw in fixtures {
            let outcome =
                std::panic::catch_unwind(|| validate_pairwise_judge_output(&raw, RATIONALE_LIMIT))
                    .unwrap_or_else(|_| panic!("key-context scan panicked for {raw:?}"));
            let invalid = invalid(&outcome);
            assert_eq!(
                invalid.output().marker(),
                Some(InvalidJudgeOutputMarkerV1::CredentialBearing),
                "failed closed incorrectly for {raw:?}"
            );
            assert_eq!(invalid.output().retained_output(), None);
            assert!(issue_texts(&outcome).contains(&CREDENTIAL_ISSUE));
        }
    }

    #[test]
    fn bounded_nonsecret_invalid_output_is_retained_exactly() {
        let raw = "not valid json, but safe to retain";
        let outcome = validate_pairwise_judge_output(raw, RATIONALE_LIMIT);
        let invalid = invalid(&outcome);
        assert_eq!(invalid.output().retained_output(), Some(raw));
        assert_eq!(invalid.output().marker(), None);
        assert_eq!(invalid.output().sha256(), sha256_hex(raw.as_bytes()));
        assert_eq!(
            invalid.output().byte_count(),
            u64::try_from(raw.len()).unwrap()
        );
        assert_eq!(issue_texts(&outcome), ["output.invalid_json"]);
    }

    #[test]
    fn raw_output_cap_is_exact_checked_and_hash_only_when_exceeded() {
        let max_rationale_bytes = 1;
        let exact = "x".repeat(max_rationale_bytes + JUDGE_OUTPUT_OVERHEAD_BYTES);
        let exact_outcome = validate_pairwise_judge_output(&exact, max_rationale_bytes);
        assert_eq!(
            invalid(&exact_outcome).output().retained_output(),
            Some(exact.as_str())
        );

        let oversized = format!("{exact}x");
        let oversized_outcome = validate_pairwise_judge_output(&oversized, max_rationale_bytes);
        let oversized_invalid = invalid(&oversized_outcome);
        assert_eq!(
            oversized_invalid.output().marker(),
            Some(InvalidJudgeOutputMarkerV1::Oversized)
        );
        assert_eq!(oversized_invalid.output().retained_output(), None);
        assert_eq!(
            oversized_invalid.output().byte_count(),
            u64::try_from(oversized.len()).unwrap()
        );
        assert_eq!(
            oversized_invalid.output().sha256(),
            sha256_hex(oversized.as_bytes())
        );

        for invalid_bound in [0, usize::MAX] {
            assert!(matches!(
                validate_pairwise_judge_output("{}", invalid_bound),
                JudgeAttemptOutcomeV1::Operational(
                    JudgeOutputOperationalFailureV1::InvalidRationaleBound
                )
            ));
        }
    }

    #[test]
    fn validation_issues_are_count_and_byte_bounded_with_long_unknown_keys() {
        let mut object = Map::from_iter([
            ("response_equivalence".to_string(), json!(0.9)),
            ("trajectory_equivalence".to_string(), json!(0.8)),
            ("judge_confidence".to_string(), json!(0.95)),
            ("hard_failures".to_string(), json!([])),
            ("rationale".to_string(), json!("safe")),
        ]);
        for index in 0..32 {
            object.insert(format!("unknown_{index:02}"), Json::Null);
        }
        object.insert("x".repeat(1_024), Json::Null);
        let raw = Json::Object(object).to_string();
        let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
        let issues = invalid(&outcome).issues();
        assert_eq!(issues.len(), MAX_VALIDATION_ISSUES);
        assert!(
            issues
                .iter()
                .all(|issue| issue.as_str().len() <= MAX_VALIDATION_ISSUE_BYTES)
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.as_str() == TRUNCATED_ISSUES)
        );
        assert!(issues.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn credential_issue_survives_issue_truncation_without_retaining_raw_output() {
        let mut object = Map::from_iter([
            ("response_equivalence".to_string(), json!(0.9)),
            ("trajectory_equivalence".to_string(), json!(0.8)),
            ("judge_confidence".to_string(), json!(0.95)),
            ("hard_failures".to_string(), json!([])),
            (
                "rationale".to_string(),
                json!("Bearer abcDEF0123456789xyz-_"),
            ),
        ]);
        for index in 0..32 {
            object.insert(format!("unknown_{index:02}"), Json::Null);
        }
        let raw = Json::Object(object).to_string();
        let outcome = validate_pairwise_judge_output(&raw, RATIONALE_LIMIT);
        let invalid = invalid(&outcome);
        assert_eq!(invalid.issues().len(), MAX_VALIDATION_ISSUES);
        assert!(issue_texts(&outcome).contains(&CREDENTIAL_ISSUE));
        assert_eq!(
            invalid.output().marker(),
            Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
        );
    }

    #[test]
    fn attempt_resolution_covers_every_valid_and_terminal_transition() {
        let initial_valid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            &output_with_rationale("initial valid"),
            RATIONALE_LIMIT,
        ));
        let progress = resolve_judge_attempt_progress(initial_valid).unwrap();
        let final_state = progress.final_state().unwrap();
        assert!(matches!(
            final_state,
            JudgeFinalStateV1::InitialValid { .. }
        ));
        assert_eq!(final_state.result().unwrap().rationale(), "initial valid");
        assert!(final_state.operational_failure().is_none());
        assert!(
            serde_json::to_string(&progress)
                .unwrap()
                .contains("initial_valid")
        );

        let initial_invalid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            "invalid initial output",
            RATIONALE_LIMIT,
        ));
        let progress = resolve_judge_attempt_progress(initial_invalid).unwrap();
        assert!(progress.is_repair_required());
        assert!(progress.final_state().is_none());
        assert!(
            serde_json::to_string(&progress)
                .unwrap()
                .contains("repair_required")
        );
        let (repair_evidence, repair_pending) = progress.into_repair().unwrap();
        assert_eq!(
            repair_evidence.invalid_output().retained_output(),
            Some("invalid initial output")
        );
        assert!(
            serde_json::to_string(&repair_pending)
                .unwrap()
                .contains("repair_pending")
        );

        let repair_valid = JudgeAttemptV1::repair(validate_pairwise_judge_output(
            &output_with_rationale("repair valid"),
            RATIONALE_LIMIT,
        ));
        let progress = repair_pending.resolve(repair_valid).unwrap();
        let final_state = progress.final_state().unwrap();
        assert!(matches!(
            final_state,
            JudgeFinalStateV1::RepairedValid { .. }
        ));
        assert_eq!(final_state.result().unwrap().rationale(), "repair valid");
        assert!(
            serde_json::to_string(&progress)
                .unwrap()
                .contains("repaired_valid")
        );

        let repair_invalid = JudgeAttemptV1::repair(validate_pairwise_judge_output(
            "invalid repair output",
            RATIONALE_LIMIT,
        ));
        let initial_invalid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            "invalid initial output",
            RATIONALE_LIMIT,
        ));
        let (_, repair_pending) = resolve_judge_attempt_progress(initial_invalid)
            .unwrap()
            .into_repair()
            .unwrap();
        let judge_output_invalid = repair_pending.resolve(repair_invalid).unwrap();
        assert!(matches!(
            judge_output_invalid.final_state(),
            Some(JudgeFinalStateV1::JudgeOutputInvalid)
        ));

        let initial_operational = JudgeAttemptV1::initial(validate_pairwise_judge_output("{}", 0));
        let initial_operational = resolve_judge_attempt_progress(initial_operational).unwrap();
        assert!(matches!(
            initial_operational.final_state(),
            Some(JudgeFinalStateV1::InitialOperational { .. })
        ));

        let repair_operational = JudgeAttemptV1::repair(validate_pairwise_judge_output("{}", 0));
        let initial_invalid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            "invalid initial output",
            RATIONALE_LIMIT,
        ));
        let (_, repair_pending) = resolve_judge_attempt_progress(initial_invalid)
            .unwrap()
            .into_repair()
            .unwrap();
        let repair_operational = repair_pending.resolve(repair_operational).unwrap();
        assert!(matches!(
            repair_operational.final_state(),
            Some(JudgeFinalStateV1::RepairOperational { .. })
        ));

        for terminal in [
            judge_output_invalid,
            initial_operational,
            repair_operational,
        ] {
            let final_state = terminal.final_state().unwrap();
            assert!(final_state.result().is_none());
            let serialized = serde_json::to_string(&terminal).unwrap();
            assert!(!serialized.contains("result"));
            assert!(!serialized.contains("label"));
        }
    }

    #[test]
    fn attempt_resolution_rejects_every_illegal_transition() {
        let wrong_initial = JudgeAttemptV1::repair(validate_pairwise_judge_output(
            &output_with_rationale("safe"),
            RATIONALE_LIMIT,
        ));
        assert!(matches!(
            resolve_judge_attempt_progress(wrong_initial),
            Err(JudgeAttemptResolutionErrorV1::ExpectedInitialAttempt)
        ));

        let initial_valid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            &output_with_rationale("safe"),
            RATIONALE_LIMIT,
        ));
        let final_progress = resolve_judge_attempt_progress(initial_valid).unwrap();
        assert!(matches!(
            final_progress.into_repair(),
            Err(JudgeAttemptResolutionErrorV1::AlreadyFinal)
        ));

        let initial_invalid = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            "invalid initial output",
            RATIONALE_LIMIT,
        ));
        let (_, repair_pending) = resolve_judge_attempt_progress(initial_invalid)
            .unwrap()
            .into_repair()
            .unwrap();
        let wrong_repair = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            "invalid repair output",
            RATIONALE_LIMIT,
        ));
        assert!(matches!(
            repair_pending.resolve(wrong_repair),
            Err(JudgeAttemptResolutionErrorV1::ExpectedRepairAttempt)
        ));
    }

    #[test]
    fn repair_consumes_only_an_initial_invalid_attempt_and_carries_exact_schema() {
        let raw = "bounded invalid output";
        let initial = JudgeAttemptV1::initial(validate_pairwise_judge_output(raw, RATIONALE_LIMIT));
        assert_eq!(initial.ordinal(), JudgeAttemptOrdinalV1::Initial);
        assert_eq!(initial.ordinal().as_u8(), 0);
        assert!(matches!(
            initial.outcome(),
            JudgeAttemptOutcomeV1::Invalid(_)
        ));
        let repair = build_judge_repair_evidence(initial).unwrap();
        assert_eq!(repair.output_schema().as_bytes(), JUDGE_OUTPUT_SCHEMA_V1);
        let serialized = serde_json::to_value(&repair).unwrap();
        assert_eq!(
            serialized["output_schema"].as_str().unwrap().as_bytes(),
            JUDGE_OUTPUT_SCHEMA_V1
        );
        assert_eq!(repair.invalid_output().retained_output(), Some(raw));
        assert_eq!(repair.validation_issues().len(), 1);

        let credential = "Bearer abcDEF0123456789xyz-_";
        let credential_raw = output_with_rationale(credential);
        let credential_repair = build_judge_repair_evidence(JudgeAttemptV1::initial(
            validate_pairwise_judge_output(&credential_raw, RATIONALE_LIMIT),
        ))
        .unwrap();
        assert_eq!(
            credential_repair.invalid_output().marker(),
            Some(InvalidJudgeOutputMarkerV1::CredentialBearing)
        );
        assert_eq!(credential_repair.invalid_output().retained_output(), None);
        assert_eq!(
            credential_repair.invalid_output().byte_count(),
            u64::try_from(credential_raw.len()).unwrap()
        );
        assert!(
            !serde_json::to_string(&credential_repair)
                .unwrap()
                .contains(credential)
        );

        let oversized_raw = "x".repeat(1 + JUDGE_OUTPUT_OVERHEAD_BYTES + 1);
        let oversized_repair = build_judge_repair_evidence(JudgeAttemptV1::initial(
            validate_pairwise_judge_output(&oversized_raw, 1),
        ))
        .unwrap();
        assert_eq!(
            oversized_repair.invalid_output().marker(),
            Some(InvalidJudgeOutputMarkerV1::Oversized)
        );
        assert_eq!(oversized_repair.invalid_output().retained_output(), None);
        assert_eq!(
            oversized_repair.invalid_output().byte_count(),
            u64::try_from(oversized_raw.len()).unwrap()
        );
        assert!(
            !serde_json::to_string(&oversized_repair)
                .unwrap()
                .contains(&oversized_raw)
        );

        let valid_attempt = JudgeAttemptV1::initial(validate_pairwise_judge_output(
            &output_with_rationale("safe"),
            RATIONALE_LIMIT,
        ));
        assert!(matches!(
            build_judge_repair_evidence(valid_attempt),
            Err(JudgeRepairRejectionV1::ValidOutput)
        ));

        let operational = JudgeAttemptV1::initial(validate_pairwise_judge_output("{}", 0));
        assert!(matches!(
            build_judge_repair_evidence(operational),
            Err(JudgeRepairRejectionV1::OperationalFailure)
        ));

        let repair_attempt = JudgeAttemptV1::repair(validate_pairwise_judge_output(
            "invalid repair output",
            RATIONALE_LIMIT,
        ));
        assert_eq!(repair_attempt.ordinal().as_u8(), 1);
        assert!(matches!(
            build_judge_repair_evidence(repair_attempt),
            Err(JudgeRepairRejectionV1::RepairAttempt)
        ));
    }
}
