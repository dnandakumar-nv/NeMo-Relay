// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic pool matching against frozen V2 call context.

use std::collections::BTreeSet;

use nemo_relay::plugin::{ConfigDiagnostic, DiagnosticLevel};
use nemo_relay_types::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot,
};

use crate::config::RouterConfig;
use crate::selector::CompiledSelector;

/// Stable health reason recorded for an unexpected runtime double match.
pub const AMBIGUOUS_POOL_REASON: &str = "router.runtime.ambiguous_pool";

/// Result of matching one call against a validated pool set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolMatchOutcome<'a> {
    /// Stable ID of the sole matching pool, or `None` for a bypass/no-match.
    pub matched_pool_id: Option<&'a str>,
    /// Stable non-secret health reason when an invariant failed at runtime.
    pub health_reason: Option<&'static str>,
}

/// Immutable deterministic matcher compiled from a valid Router configuration.
#[derive(Debug, Clone)]
pub struct PoolMatcher {
    pools: Vec<CompiledPool>,
}

#[derive(Debug, Clone)]
struct CompiledPool {
    id: String,
    api_family: LlmApiFamily,
    anchor_models: BTreeSet<String>,
    selector: CompiledSelector,
}

impl PoolMatcher {
    /// Compiles a matcher after running complete typed configuration validation.
    ///
    /// Error diagnostics are returned without retaining any partial matcher.
    pub fn compile(config: &RouterConfig) -> Result<Self, Vec<ConfigDiagnostic>> {
        let diagnostics = config.validate();
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
        {
            return Err(diagnostics);
        }

        let pools = config
            .pools
            .iter()
            .map(|pool| {
                CompiledSelector::compile(&pool.selector).map(|selector| CompiledPool {
                    id: pool.id.clone(),
                    api_family: pool.api_family,
                    anchor_models: pool.anchor_models.iter().cloned().collect(),
                    selector,
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("validated selectors must compile");
        Ok(Self { pools })
    }

    /// Matches one anchor model and frozen execution context.
    ///
    /// Non-Primary, streaming, and stateful calls bypass before selector work.
    /// An unexpected double match returns no pool and the stable ambiguity
    /// reason for bounded runtime health accounting.
    pub fn match_pool<'a>(
        &'a self,
        anchor_model: &str,
        context: &LlmExecutionContextSnapshot,
    ) -> PoolMatchOutcome<'a> {
        if context.call_role != LlmCallRole::Primary
            || context
                .attributes
                .intersects(LlmAttributes::STREAMING | LlmAttributes::STATEFUL)
        {
            return no_match();
        }

        let mut matches = self.pools.iter().filter(|pool| {
            pool.api_family == context.api_family
                && pool.anchor_models.contains(anchor_model)
                && pool.selector.matches(context)
        });
        let first = matches.next();
        match (first, matches.next()) {
            (Some(pool), None) => PoolMatchOutcome {
                matched_pool_id: Some(&pool.id),
                health_reason: None,
            },
            (Some(_), Some(_)) => PoolMatchOutcome {
                matched_pool_id: None,
                health_reason: Some(AMBIGUOUS_POOL_REASON),
            },
            (None, _) => no_match(),
        }
    }
}

fn no_match() -> PoolMatchOutcome<'static> {
    PoolMatchOutcome {
        matched_pool_id: None,
        health_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use nemo_relay_types::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot,
        LlmTrajectoryScopeSnapshot,
    };
    use nemo_relay_types::api::scope::ScopeType;
    use uuid::Uuid;

    use super::{AMBIGUOUS_POOL_REASON, CompiledPool, PoolMatcher};
    use crate::config::PoolSelectorConfig;
    use crate::selector::CompiledSelector;

    #[test]
    fn unexpected_runtime_double_match_is_a_health_no_match() {
        let selector = CompiledSelector::compile(&PoolSelectorConfig::default()).unwrap();
        let pool = |id: &str| CompiledPool {
            id: id.to_string(),
            api_family: LlmApiFamily::OpenAIResponses,
            anchor_models: BTreeSet::from(["anchor".to_string()]),
            selector: selector.clone(),
        };
        let matcher = PoolMatcher {
            pools: vec![pool("a"), pool("b")],
        };
        let context = LlmExecutionContextSnapshot {
            call_uuid: Uuid::new_v4(),
            root_uuid: Uuid::new_v4(),
            parent_uuid: Uuid::new_v4(),
            trajectory_owner_uuid: Uuid::new_v4(),
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: Uuid::new_v4(),
                name: "must-not-be-read".to_string(),
                scope_type: ScopeType::Agent,
            }],
            api_family: LlmApiFamily::OpenAIResponses,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: None,
            agent_id: None,
            sanitized_metadata: BTreeMap::new(),
        };
        let outcome = matcher.match_pool("anchor", &context);
        assert_eq!(outcome.matched_pool_id, None);
        assert_eq!(outcome.health_reason, Some(AMBIGUOUS_POOL_REASON));
    }
}
