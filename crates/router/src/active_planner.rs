// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fresh Active query gates, hysteresis, first-winner selection, and audit finalization.

#![allow(dead_code)] // Task 11 wires the completed planner into foreground execution.

use std::collections::{BTreeMap, BTreeSet};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use uuid::{Uuid, Variant};

use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::decision_audit::{
    ActiveDecisionParentBindingV2, DecisionAuditV1, DecisionFinalReasonV1,
};
use crate::ledger::repository::active_learning::{
    ActiveAuthorizationObserveAck, ActiveAuthorizationSnapshot, ActiveNeighborhoodKey,
    ActiveNeighborhoodMutationAck, ActiveNeighborhoodObserveAck, ActiveNeighborhoodPromotion,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::live_embedding::LiveEmbeddingResult;
use crate::recommendation::{
    EvaluatedRecommendationQueryPlanV1, PreparedRecommendationV1,
    RecommendationCandidateGateResolverV1, RecommendationCandidateGateV1,
    RecommendationCandidateSearchIdentityV1, RecommendationErrorV1,
    run_recommendation_query_plan_with_resolver_until,
};
use crate::sqlite_vector_store::SqliteVecStore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivePlannerErrorV2 {
    InvalidIdentity,
    Recommendation,
    AuthorityUnavailable,
    NoWinner,
    NeighborhoodChanged,
    Audit,
}

impl From<RecommendationErrorV1> for ActivePlannerErrorV2 {
    fn from(_: RecommendationErrorV1) -> Self {
        Self::Recommendation
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveCandidateExperimentAuthorityV2 {
    pub(crate) candidate_id: String,
    pub(crate) active_experiment_id: Uuid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveGateBasisV2 {
    Promotion,
    Retention,
    Restoration,
    Blocked,
}

impl ActiveGateBasisV2 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Promotion => "promotion",
            Self::Retention => "retention",
            Self::Restoration => "restoration",
            Self::Blocked => "blocked",
        }
    }
}

#[derive(Debug, Clone)]
struct ActiveResolvedCandidateGateV2 {
    identity: RecommendationCandidateSearchIdentityV1,
    experiment_id: Uuid,
    authorization: Option<ActiveAuthorizationSnapshot>,
    key: Option<ActiveNeighborhoodKey>,
    neighborhood_state_event_id: Option<Uuid>,
    basis: ActiveGateBasisV2,
    threshold_bits: u64,
    externally_authorized: bool,
}

pub(crate) struct ActiveGateResolverV2 {
    writer: LedgerWriterClient,
    experiments: BTreeMap<String, Uuid>,
    config_generation_id: String,
    learning_generation_id: Uuid,
    cohort_generation_id: Uuid,
    canonical_query_hash: String,
    promotion_lower_bound: f64,
    retention_lower_bound: f64,
    observed_at_unix_ms: i64,
    deadline: Instant,
    resolved: BTreeMap<String, ActiveResolvedCandidateGateV2>,
}

impl ActiveGateResolverV2 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        writer: LedgerWriterClient,
        experiments: Vec<ActiveCandidateExperimentAuthorityV2>,
        config_generation_id: String,
        learning_generation_id: Uuid,
        cohort_generation_id: Uuid,
        canonical_query_hash: String,
        promotion_lower_bound: f64,
        retention_lower_bound: f64,
        observed_at_unix_ms: i64,
        deadline: Instant,
    ) -> Result<Self, ActivePlannerErrorV2> {
        let mut experiment_map = BTreeMap::new();
        let mut experiment_ids = BTreeSet::new();
        for experiment in experiments {
            if experiment.candidate_id.is_empty()
                || experiment.candidate_id.len() > 128
                || !is_uuid_v7(experiment.active_experiment_id)
                || experiment_map
                    .insert(experiment.candidate_id, experiment.active_experiment_id)
                    .is_some()
                || !experiment_ids.insert(experiment.active_experiment_id)
            {
                return Err(ActivePlannerErrorV2::InvalidIdentity);
            }
        }
        if experiment_map.is_empty()
            || !is_hash(&config_generation_id)
            || !is_uuid_v7(learning_generation_id)
            || !is_uuid_v7(cohort_generation_id)
            || !is_hash(&canonical_query_hash)
            || !promotion_lower_bound.is_finite()
            || !(0.0..=1.0).contains(&promotion_lower_bound)
            || !retention_lower_bound.is_finite()
            || !(0.0..promotion_lower_bound).contains(&retention_lower_bound)
            || observed_at_unix_ms < 0
            || Instant::now() >= deadline
        {
            return Err(ActivePlannerErrorV2::InvalidIdentity);
        }
        Ok(Self {
            writer,
            experiments: experiment_map,
            config_generation_id,
            learning_generation_id,
            cohort_generation_id,
            canonical_query_hash,
            promotion_lower_bound,
            retention_lower_bound,
            observed_at_unix_ms,
            deadline,
            resolved: BTreeMap::new(),
        })
    }

    async fn resolve(
        &mut self,
        identity: RecommendationCandidateSearchIdentityV1,
    ) -> Result<RecommendationCandidateGateV1, RecommendationErrorV1> {
        let experiment_id = self
            .experiments
            .get(&identity.candidate_id)
            .copied()
            .ok_or(RecommendationErrorV1::InvalidIdentity)?;
        if self.resolved.contains_key(&identity.candidate_id) {
            return Err(RecommendationErrorV1::InvalidIdentity);
        }
        let authorization = self
            .writer
            .observe_active_authorization_until(
                experiment_id,
                self.observed_at_unix_ms,
                self.deadline,
            )
            .await
            .map_err(|_| RecommendationErrorV1::RuntimeFailure)?;
        let authorization = match authorization {
            ActiveAuthorizationObserveAck::Current(snapshot) => Some(snapshot),
            ActiveAuthorizationObserveAck::Unavailable
            | ActiveAuthorizationObserveAck::TransactionNotStarted => None,
        };
        let key = authorization.map(|authorization| ActiveNeighborhoodKey {
            active_experiment_id: experiment_id,
            canonical_query_hash: self.canonical_query_hash.clone(),
            sorted_neighbor_hash: identity.sorted_neighbor_hash.clone(),
            config_generation_id: self.config_generation_id.clone(),
            learning_generation_id: self.learning_generation_id,
            cohort_generation_id: self.cohort_generation_id,
            active_authorization_state_event_id: authorization.authorization_state_event_id,
        });
        let neighborhood = match key.as_ref() {
            Some(key) => self
                .writer
                .observe_active_neighborhood_until(
                    key.clone(),
                    self.observed_at_unix_ms,
                    self.deadline,
                )
                .await
                .map_err(|_| RecommendationErrorV1::RuntimeFailure)?,
            None => ActiveNeighborhoodObserveAck::Unavailable,
        };
        let (basis, threshold, externally_authorized, neighborhood_state_event_id) =
            match neighborhood {
                ActiveNeighborhoodObserveAck::Current(snapshot) if snapshot.authorizing => (
                    ActiveGateBasisV2::Retention,
                    self.retention_lower_bound,
                    true,
                    Some(snapshot.active_neighborhood_state_event_id),
                ),
                ActiveNeighborhoodObserveAck::Current(snapshot) if snapshot.restorable => (
                    ActiveGateBasisV2::Restoration,
                    self.promotion_lower_bound,
                    true,
                    None,
                ),
                ActiveNeighborhoodObserveAck::Absent {
                    promotion_authorized: true,
                    ..
                } => (
                    ActiveGateBasisV2::Promotion,
                    self.promotion_lower_bound,
                    true,
                    None,
                ),
                ActiveNeighborhoodObserveAck::Current(_)
                | ActiveNeighborhoodObserveAck::Absent { .. }
                | ActiveNeighborhoodObserveAck::Unavailable
                | ActiveNeighborhoodObserveAck::TransactionNotStarted => (
                    ActiveGateBasisV2::Blocked,
                    self.promotion_lower_bound,
                    false,
                    None,
                ),
            };
        let gate = RecommendationCandidateGateV1 {
            candidate_id: identity.candidate_id.clone(),
            lower_bound_threshold: threshold,
            externally_authorized,
        };
        self.resolved.insert(
            identity.candidate_id.clone(),
            ActiveResolvedCandidateGateV2 {
                identity,
                experiment_id,
                authorization,
                key,
                neighborhood_state_event_id,
                basis,
                threshold_bits: threshold.to_bits(),
                externally_authorized,
            },
        );
        Ok(gate)
    }
}

impl RecommendationCandidateGateResolverV1 for ActiveGateResolverV2 {
    fn resolve_gate<'a>(
        &'a mut self,
        identity: RecommendationCandidateSearchIdentityV1,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<RecommendationCandidateGateV1, RecommendationErrorV1>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(self.resolve(identity))
    }
}

pub(crate) struct ActiveEvaluatedQueryV2 {
    plan: EvaluatedRecommendationQueryPlanV1,
    resolver: ActiveGateResolverV2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveWinnerAuthorityV2 {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) active_authorization_state_event_id: Uuid,
    pub(crate) active_neighborhood_state_event_id: Uuid,
    pub(crate) active_outcome_look_id: Option<Uuid>,
    pub(crate) open_tranche_ordinal: u64,
    pub(crate) noninferiority_lower_bits: Option<u64>,
    pub(crate) noninferiority_upper_bits: Option<u64>,
    pub(crate) neighborhood_key: ActiveNeighborhoodKey,
}

pub(crate) struct ActiveAuditedQueryV2 {
    pub(crate) audit: Arc<DecisionAuditV1>,
    pub(crate) winner: ActiveWinnerAuthorityV2,
    pub(crate) fresh_gate_audit_json: String,
    pub(crate) fresh_gate_audit_hash: String,
    pub(crate) anchor_shadow_lower_bound_bits: Option<u64>,
}

pub(crate) async fn evaluate_active_query_until(
    store: &SqliteVecStore,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
    mut resolver: ActiveGateResolverV2,
) -> Result<ActiveEvaluatedQueryV2, ActivePlannerErrorV2> {
    let plan = run_recommendation_query_plan_with_resolver_until(
        store,
        prepared,
        embedding,
        &mut resolver,
    )
    .await?;
    Ok(ActiveEvaluatedQueryV2 { plan, resolver })
}

impl ActiveEvaluatedQueryV2 {
    pub(crate) fn winner_candidate_id(&self) -> Option<&str> {
        self.plan.winner_candidate_id.as_deref()
    }

    pub(crate) async fn authorize_winner_neighborhood_until(
        &mut self,
    ) -> Result<ActiveWinnerAuthorityV2, ActivePlannerErrorV2> {
        let winner_id = self
            .plan
            .winner_candidate_id
            .as_deref()
            .ok_or(ActivePlannerErrorV2::NoWinner)?;
        let resolved = self
            .resolver
            .resolved
            .get_mut(winner_id)
            .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
        let authorization = resolved
            .authorization
            .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
        if !resolved.externally_authorized {
            return Err(ActivePlannerErrorV2::AuthorityUnavailable);
        }
        let neighborhood_state_event_id = match resolved.neighborhood_state_event_id {
            Some(event_id) => event_id,
            None if matches!(
                resolved.basis,
                ActiveGateBasisV2::Promotion | ActiveGateBasisV2::Restoration
            ) =>
            {
                let key = resolved
                    .key
                    .clone()
                    .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
                let acknowledgement = self
                    .resolver
                    .writer
                    .promote_active_neighborhood_until(
                        ActiveNeighborhoodPromotion {
                            active_neighborhood_state_event_id: Uuid::now_v7(),
                            key,
                            stable_reason: "fresh_active_gate_v1".to_string(),
                        },
                        self.resolver.deadline,
                    )
                    .await
                    .map_err(|_| ActivePlannerErrorV2::AuthorityUnavailable)?;
                let snapshot = match acknowledgement {
                    ActiveNeighborhoodMutationAck::Applied(snapshot)
                    | ActiveNeighborhoodMutationAck::AlreadyApplied(snapshot)
                    | ActiveNeighborhoodMutationAck::AlreadyCurrent(snapshot)
                        if snapshot.authorizing =>
                    {
                        snapshot
                    }
                    _ => return Err(ActivePlannerErrorV2::NeighborhoodChanged),
                };
                resolved.neighborhood_state_event_id =
                    Some(snapshot.active_neighborhood_state_event_id);
                snapshot.active_neighborhood_state_event_id
            }
            None => return Err(ActivePlannerErrorV2::AuthorityUnavailable),
        };
        Ok(ActiveWinnerAuthorityV2 {
            active_experiment_id: resolved.experiment_id,
            active_authorization_state_event_id: authorization.authorization_state_event_id,
            active_neighborhood_state_event_id: neighborhood_state_event_id,
            active_outcome_look_id: authorization.active_outcome_look_id,
            open_tranche_ordinal: authorization
                .open_tranche_ordinal
                .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?,
            noninferiority_lower_bits: authorization.noninferiority_lower_bits,
            noninferiority_upper_bits: authorization.noninferiority_upper_bits,
            neighborhood_key: resolved
                .key
                .clone()
                .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?,
        })
    }

    pub(crate) fn into_audited_query(
        self,
        binding: ActiveDecisionParentBindingV2,
        final_reason: DecisionFinalReasonV1,
    ) -> Result<ActiveAuditedQueryV2, ActivePlannerErrorV2> {
        let winner_id = self
            .plan
            .winner_candidate_id
            .clone()
            .ok_or(ActivePlannerErrorV2::NoWinner)?;
        let resolved = self
            .resolver
            .resolved
            .get(&winner_id)
            .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
        let authorization = resolved
            .authorization
            .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
        let neighborhood_state_event_id = resolved
            .neighborhood_state_event_id
            .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
        if binding.active_experiment_id != Some(resolved.experiment_id)
            || binding.active_authorization_state_event_id
                != Some(authorization.authorization_state_event_id)
        {
            return Err(ActivePlannerErrorV2::InvalidIdentity);
        }
        let audit = self
            .plan
            .into_active_audit(binding, final_reason)
            .map_err(|_| ActivePlannerErrorV2::Audit)?;
        let (fresh_gate_audit_json, fresh_gate_audit_hash) =
            build_fresh_gate_audit(&audit, &self.resolver.resolved, &winner_id)?;
        let anchor_shadow_lower_bound_bits = audit
            .summaries
            .iter()
            .find(|summary| summary.candidate_id == winner_id)
            .and_then(|summary| summary.lower_bound)
            .map(|value| value.bits());
        Ok(ActiveAuditedQueryV2 {
            audit: Arc::new(audit),
            winner: ActiveWinnerAuthorityV2 {
                active_experiment_id: resolved.experiment_id,
                active_authorization_state_event_id: authorization.authorization_state_event_id,
                active_neighborhood_state_event_id: neighborhood_state_event_id,
                active_outcome_look_id: authorization.active_outcome_look_id,
                open_tranche_ordinal: authorization
                    .open_tranche_ordinal
                    .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?,
                noninferiority_lower_bits: authorization.noninferiority_lower_bits,
                noninferiority_upper_bits: authorization.noninferiority_upper_bits,
                neighborhood_key: resolved
                    .key
                    .clone()
                    .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?,
            },
            fresh_gate_audit_json,
            fresh_gate_audit_hash,
            anchor_shadow_lower_bound_bits,
        })
    }
}

fn build_fresh_gate_audit(
    audit: &DecisionAuditV1,
    resolved: &BTreeMap<String, ActiveResolvedCandidateGateV2>,
    winner_id: &str,
) -> Result<(String, String), ActivePlannerErrorV2> {
    let winner = resolved
        .get(winner_id)
        .ok_or(ActivePlannerErrorV2::AuthorityUnavailable)?;
    let candidates = audit
        .summaries
        .iter()
        .map(|summary| {
            let gate = resolved.get(&summary.candidate_id);
            json!({
                "candidate_id": summary.candidate_id,
                "cost_rank": summary.cost_rank,
                "terminal_reason": summary.terminal_reason.as_str(),
                "threshold_bits": format!("{:016x}", summary.promotion_lower_bound.bits()),
                "lower_bound_bits": summary.lower_bound.map(|value| format!("{:016x}", value.bits())),
                "partition_gate": summary.partition_gate_passed,
                "points_gate": summary.points_gate_passed,
                "roots_gate": summary.roots_gate_passed,
                "coverage_gate": summary.coverage_gate_passed,
                "weight_gate": summary.weight_gate_passed,
                "effective_samples_gate": summary.effective_samples_gate_passed,
                "beta_quantile_gate": summary.beta_quantile_gate_passed,
                "lower_bound_gate": summary.lower_bound_gate_passed,
                "sorted_neighbor_hash": gate.map(|gate| gate.identity.sorted_neighbor_hash.clone()),
                "active_experiment_id": gate.map(|gate| gate.experiment_id),
                "active_authorization_state_event_id": gate.and_then(|gate| gate.authorization.map(|value| value.authorization_state_event_id)),
                "authorization_state": gate.and_then(|gate| gate.authorization.map(|value| value.state.as_str())),
                "basis": gate.map(|gate| gate.basis.as_str()),
                "externally_authorized": gate.is_some_and(|gate| gate.externally_authorized),
            })
        })
        .collect::<Vec<_>>();
    let document = json!({
        "schema": "nemo.relay.router.active-fresh-gates@1",
        "decision_id": audit.parent.decision_id,
        "as_of_unix_ms": audit.parent.as_of_unix_ms.to_string(),
        "winner": {
            "candidate_id": winner_id,
            "active_experiment_id": winner.experiment_id,
            "active_authorization_state_event_id": winner.authorization.map(|value| value.authorization_state_event_id),
            "active_neighborhood_state_event_id": winner.neighborhood_state_event_id,
            "sorted_neighbor_hash": winner.identity.sorted_neighbor_hash,
            "basis": winner.basis.as_str(),
            "threshold_bits": format!("{:016x}", winner.threshold_bits),
        },
        "candidates": candidates,
    });
    let canonical = canonical_json(&document).map_err(|_| ActivePlannerErrorV2::Audit)?;
    let hash = canonical_sha256(&document).map_err(|_| ActivePlannerErrorV2::Audit)?;
    Ok((canonical, hash))
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
