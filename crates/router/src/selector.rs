// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Restricted selector compilation, matching, and exact overlap checks.

use std::collections::{BTreeMap, BTreeSet};

use nemo_relay::plugin::{ConfigDiagnostic, ConfigPolicy};
use nemo_relay_types::Json;
use nemo_relay_types::api::llm::LlmExecutionContextSnapshot;
use nemo_relay_types::api::scope::ScopeType;
use unicode_normalization::UnicodeNormalization;

use crate::canonical_json::canonical_json;
use crate::config::{
    PATH_PATTERN_MAX_BYTES, PoolSelectorConfig, validate_selector_identity, validate_unknown_fields,
};
use crate::diagnostics::{DUPLICATE_ID, INVALID_RANGE, INVALID_REFERENCE, error};

/// One parsed scope-type path pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopePathPattern {
    #[cfg(test)]
    raw: String,
    prefix: Vec<PathSegment>,
    suffix_wildcard: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PathSegment {
    Literal(String),
    One,
}

impl ScopePathPattern {
    /// Parses a restricted scope-type path pattern.
    ///
    /// A pattern contains slash-separated stable scope-type wire names, `*`
    /// for one segment, and an optional terminal `**` for zero or more
    /// segments.
    pub fn parse(value: &str) -> Result<Self, String> {
        if value.is_empty() {
            return Err("scope path pattern must not be empty".to_string());
        }
        if value.len() > PATH_PATTERN_MAX_BYTES {
            return Err(format!(
                "scope path pattern exceeds the {PATH_PATTERN_MAX_BYTES}-byte limit"
            ));
        }
        if value.chars().any(char::is_control) {
            return Err("scope path pattern must not contain control characters".to_string());
        }
        if !value.nfc().eq(value.chars()) {
            return Err("scope path pattern must use NFC Unicode normalization".to_string());
        }
        if value.starts_with('/') || value.ends_with('/') || value.contains("//") {
            return Err(
                "scope path pattern must not have leading, trailing, or empty segments".to_string(),
            );
        }

        let raw_segments = value.split('/').collect::<Vec<_>>();
        let mut prefix = Vec::with_capacity(raw_segments.len());
        let mut suffix_wildcard = false;
        for (index, segment) in raw_segments.iter().enumerate() {
            match *segment {
                "*" => prefix.push(PathSegment::One),
                "**" if index + 1 == raw_segments.len() => suffix_wildcard = true,
                "**" => return Err("'**' is allowed only as the terminal segment".to_string()),
                literal if literal.contains('*') => {
                    return Err("wildcards must occupy an entire path segment".to_string());
                }
                literal if !is_scope_type_wire_name(literal) => {
                    return Err(format!("'{literal}' is not a stable ScopeType wire name"));
                }
                literal => prefix.push(PathSegment::Literal(literal.to_string())),
            }
        }

        Ok(Self {
            #[cfg(test)]
            raw: value.to_string(),
            prefix,
            suffix_wildcard,
        })
    }

    /// Returns the original validated pattern text.
    #[cfg(test)]
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Returns whether this pattern matches an owner-to-parent scope-type path.
    pub fn matches(&self, path: &[ScopeType]) -> bool {
        if path.len() < self.prefix.len()
            || (!self.suffix_wildcard && path.len() != self.prefix.len())
        {
            return false;
        }
        self.prefix
            .iter()
            .zip(path)
            .all(|(expected, actual)| match expected {
                PathSegment::One => true,
                PathSegment::Literal(expected) => expected == actual.as_str(),
            })
    }

    /// Returns whether this pattern and another pattern share any path witness.
    #[cfg(test)]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.overlap_witness_length(other)
            .is_some_and(|witness_length| {
                (0..witness_length).all(|index| {
                    segments_compatible(self.prefix.get(index), other.prefix.get(index))
                })
            })
    }

    fn overlaps_with_owner(&self, other: &Self, owner: ScopeType) -> bool {
        self.overlap_witness_length(other)
            .is_some_and(|witness_length| {
                let witness_length = witness_length.max(1);
                (0..witness_length).all(|index| {
                    let left = self.prefix.get(index);
                    let right = other.prefix.get(index);
                    segments_compatible(left, right)
                        && (index != 0
                            || segment_accepts(left, owner) && segment_accepts(right, owner))
                })
            })
    }

    fn overlap_witness_length(&self, other: &Self) -> Option<usize> {
        let witness_length = match (self.suffix_wildcard, other.suffix_wildcard) {
            (false, false) if self.prefix.len() == other.prefix.len() => self.prefix.len(),
            (false, false) => return None,
            (false, true) if self.prefix.len() >= other.prefix.len() => self.prefix.len(),
            (true, false) if other.prefix.len() >= self.prefix.len() => other.prefix.len(),
            (false, true) | (true, false) => return None,
            (true, true) => self.prefix.len().max(other.prefix.len()),
        };
        Some(witness_length)
    }
}

fn segments_compatible(left: Option<&PathSegment>, right: Option<&PathSegment>) -> bool {
    match (left, right) {
        (Some(PathSegment::Literal(left)), Some(PathSegment::Literal(right))) => left == right,
        _ => true,
    }
}

fn segment_accepts(segment: Option<&PathSegment>, scope_type: ScopeType) -> bool {
    match segment {
        Some(PathSegment::Literal(literal)) => literal == scope_type.as_str(),
        Some(PathSegment::One) | None => true,
    }
}

/// Compiled selector predicates for one pool.
#[derive(Debug, Clone)]
pub struct CompiledSelector {
    tenant_ids: Option<BTreeSet<String>>,
    agent_ids: Option<BTreeSet<String>>,
    owner_scope_types: Option<BTreeSet<String>>,
    metadata_equals: BTreeMap<String, CanonicalScalar>,
    scope_path_patterns: Option<Vec<ScopePathPattern>>,
}

impl CompiledSelector {
    /// Compiles a selector that has passed configuration validation.
    pub fn compile(config: &PoolSelectorConfig) -> Result<Self, String> {
        let metadata_equals = config
            .metadata_equals
            .iter()
            .map(|(key, value)| {
                CanonicalScalar::from_json(value)
                    .map(|value| (key.clone(), value))
                    .ok_or_else(|| format!("metadata selector '{key}' is not scalar"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let scope_path_patterns = config
            .scope_path_patterns
            .as_ref()
            .map(|patterns| {
                patterns
                    .iter()
                    .map(|pattern| ScopePathPattern::parse(pattern))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        Ok(Self {
            tenant_ids: config
                .tenant_ids
                .as_ref()
                .map(|values| values.iter().cloned().collect()),
            agent_ids: config
                .agent_ids
                .as_ref()
                .map(|values| values.iter().cloned().collect()),
            owner_scope_types: config.owner_scope_types.as_ref().map(|values| {
                values
                    .iter()
                    .map(|value| value.as_str().to_string())
                    .collect()
            }),
            metadata_equals,
            scope_path_patterns,
        })
    }

    /// Returns whether all predicates match one frozen V2 execution context.
    pub fn matches(&self, context: &LlmExecutionContextSnapshot) -> bool {
        if !optional_identity_matches(&self.tenant_ids, context.tenant_id.as_deref())
            || !optional_identity_matches(&self.agent_ids, context.agent_id.as_deref())
        {
            return false;
        }

        if let Some(owner_scope_types) = &self.owner_scope_types {
            let Some(owner) = context.trajectory_owner_path.first() else {
                return false;
            };
            if !owner_scope_types.contains(owner.scope_type.as_str()) {
                return false;
            }
        }

        if !self.metadata_equals.iter().all(|(key, expected)| {
            context
                .sanitized_metadata
                .get(key)
                .and_then(CanonicalScalar::from_json)
                .is_some_and(|actual| actual == *expected)
        }) {
            return false;
        }

        if let Some(patterns) = &self.scope_path_patterns {
            let path = context
                .trajectory_owner_path
                .iter()
                .map(|snapshot| snapshot.scope_type)
                .collect::<Vec<_>>();
            if !patterns.iter().any(|pattern| pattern.matches(&path)) {
                return false;
            }
        }

        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CanonicalScalar {
    Null,
    Bool(bool),
    Number(String),
    String(String),
}

impl CanonicalScalar {
    fn from_json(value: &Json) -> Option<Self> {
        match value {
            Json::Null => Some(Self::Null),
            Json::Bool(value) => Some(Self::Bool(*value)),
            Json::Number(_) => canonical_json(value).ok().map(Self::Number),
            Json::String(value) => Some(Self::String(value.clone())),
            Json::Array(_) | Json::Object(_) => None,
        }
    }
}

pub(crate) fn validate_selector(
    selector: &PoolSelectorConfig,
    pool_prefix: &str,
    policy: &ConfigPolicy,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.selector");
    validate_unknown_fields(diagnostics, policy, &selector.unknown_fields, &prefix);
    validate_identity_list(
        selector.tenant_ids.as_ref(),
        &format!("{prefix}.tenant_ids"),
        diagnostics,
    );
    validate_identity_list(
        selector.agent_ids.as_ref(),
        &format!("{prefix}.agent_ids"),
        diagnostics,
    );

    if let Some(scope_types) = &selector.owner_scope_types {
        if scope_types.is_empty() {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{prefix}.owner_scope_types")),
                "present owner_scope_types must not be empty",
            ));
        }
        let mut seen = BTreeMap::new();
        for (index, scope_type) in scope_types.iter().enumerate() {
            if let Some(previous) = seen.insert(scope_type.as_str(), index) {
                diagnostics.push(error(
                    DUPLICATE_ID,
                    Some(format!("{prefix}.owner_scope_types[{index}]")),
                    format!(
                        "owner scope type '{}' duplicates owner_scope_types[{previous}]",
                        scope_type.as_str()
                    ),
                ));
            }
        }
    }

    for (key, value) in &selector.metadata_equals {
        validate_selector_identity(key, &format!("{prefix}.metadata_equals.{key}"), diagnostics);
        if !matches!(
            value,
            Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_)
        ) {
            diagnostics.push(error(
                INVALID_REFERENCE,
                Some(format!("{prefix}.metadata_equals.{key}")),
                "metadata selector values must be scalar JSON",
            ));
        }
    }

    if let Some(patterns) = &selector.scope_path_patterns {
        if patterns.is_empty() {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{prefix}.scope_path_patterns")),
                "present scope_path_patterns must not be empty",
            ));
        }
        let mut seen = BTreeMap::new();
        for (index, pattern) in patterns.iter().enumerate() {
            if let Some(previous) = seen.insert(pattern.as_str(), index) {
                diagnostics.push(error(
                    DUPLICATE_ID,
                    Some(format!("{prefix}.scope_path_patterns[{index}]")),
                    format!("path pattern '{pattern}' duplicates scope_path_patterns[{previous}]"),
                ));
            }
            if let Err(message) = ScopePathPattern::parse(pattern) {
                diagnostics.push(error(
                    if message.contains("byte limit") {
                        INVALID_RANGE
                    } else {
                        INVALID_REFERENCE
                    },
                    Some(format!("{prefix}.scope_path_patterns[{index}]")),
                    message,
                ));
            }
        }
    }
}

/// Returns whether two selector domains share at least one frozen-context witness.
#[cfg(test)]
pub fn selector_domains_overlap(left: &PoolSelectorConfig, right: &PoolSelectorConfig) -> bool {
    selectors_overlap(left, right)
}

pub(crate) fn selectors_overlap(left: &PoolSelectorConfig, right: &PoolSelectorConfig) -> bool {
    optional_sets_overlap(left.tenant_ids.as_ref(), right.tenant_ids.as_ref())
        && optional_sets_overlap(left.agent_ids.as_ref(), right.agent_ids.as_ref())
        && owner_path_domains_overlap(
            left.owner_scope_types.as_ref(),
            right.owner_scope_types.as_ref(),
            left.scope_path_patterns.as_ref(),
            right.scope_path_patterns.as_ref(),
        )
        && metadata_domains_overlap(&left.metadata_equals, &right.metadata_equals)
}

fn validate_identity_list(
    values: Option<&Vec<String>>,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let Some(values) = values else {
        return;
    };
    if values.is_empty() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!(
                "present {} must not be empty",
                field.rsplit('.').next().unwrap_or(field)
            ),
        ));
    }
    let mut seen = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        validate_selector_identity(value, &format!("{field}[{index}]"), diagnostics);
        if let Some(previous) = seen.insert(value.as_str(), index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(format!("{field}[{index}]")),
                format!("identity '{value}' duplicates {field}[{previous}]"),
            ));
        }
    }
}

fn optional_identity_matches(values: &Option<BTreeSet<String>>, actual: Option<&str>) -> bool {
    match values {
        None => true,
        Some(values) => actual.is_some_and(|actual| values.contains(actual)),
    }
}

fn optional_sets_overlap(left: Option<&Vec<String>>, right: Option<&Vec<String>>) -> bool {
    match (left, right) {
        (None, _) | (_, None) => true,
        (Some(left), Some(right)) => left.iter().any(|value| right.contains(value)),
    }
}

fn owner_path_domains_overlap(
    left: Option<&Vec<ScopeType>>,
    right: Option<&Vec<ScopeType>>,
    left_paths: Option<&Vec<String>>,
    right_paths: Option<&Vec<String>>,
) -> bool {
    all_scope_types().into_iter().any(|owner| {
        optional_scope_types_accepts(left, owner)
            && optional_scope_types_accepts(right, owner)
            && path_domains_overlap_for_owner(left_paths, right_paths, owner)
    })
}

fn metadata_domains_overlap(left: &BTreeMap<String, Json>, right: &BTreeMap<String, Json>) -> bool {
    left.iter().all(|(key, left_value)| {
        right.get(key).is_none_or(|right_value| {
            CanonicalScalar::from_json(left_value) == CanonicalScalar::from_json(right_value)
        })
    })
}

fn optional_scope_types_accepts(values: Option<&Vec<ScopeType>>, owner: ScopeType) -> bool {
    values.is_none_or(|values| values.contains(&owner))
}

fn path_domains_overlap_for_owner(
    left: Option<&Vec<String>>,
    right: Option<&Vec<String>>,
    owner: ScopeType,
) -> bool {
    let wildcard = ["**".to_string()];
    let left = left.map(Vec::as_slice).unwrap_or(&wildcard);
    let right = right.map(Vec::as_slice).unwrap_or(&wildcard);
    let left = left
        .iter()
        .map(|pattern| ScopePathPattern::parse(pattern))
        .collect::<Result<Vec<_>, _>>();
    let right = right
        .iter()
        .map(|pattern| ScopePathPattern::parse(pattern))
        .collect::<Result<Vec<_>, _>>();
    match (left, right) {
        (Ok(left), Ok(right)) => left.iter().any(|left| {
            right
                .iter()
                .any(|right| left.overlaps_with_owner(right, owner))
        }),
        _ => false,
    }
}

fn is_scope_type_wire_name(value: &str) -> bool {
    all_scope_types()
        .iter()
        .any(|scope_type| scope_type.as_str() == value)
}

fn all_scope_types() -> [ScopeType; 11] {
    [
        ScopeType::Agent,
        ScopeType::Function,
        ScopeType::Tool,
        ScopeType::Llm,
        ScopeType::Retriever,
        ScopeType::Embedder,
        ScopeType::Reranker,
        ScopeType::Guardrail,
        ScopeType::Evaluator,
        ScopeType::Custom,
        ScopeType::Unknown,
    ]
}

#[cfg(test)]
mod tests {
    use super::ScopePathPattern;

    #[test]
    fn restricted_pattern_overlap_is_exact_for_suffix_wildcards() {
        let cases = [
            ("agent/*", "agent/tool", true),
            ("agent/**", "agent", true),
            ("agent/**", "agent/tool/llm", true),
            ("agent/tool", "agent/llm", false),
            ("agent/*/llm", "agent/**", true),
            ("agent/*/llm", "agent/tool", false),
        ];
        for (left, right, expected) in cases {
            assert_eq!(
                ScopePathPattern::parse(left)
                    .unwrap()
                    .overlaps(&ScopePathPattern::parse(right).unwrap()),
                expected,
                "{left} versus {right}"
            );
        }
    }
}
