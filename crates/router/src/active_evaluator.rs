// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded two-phase Active outcome-look evaluation outside SQLite transactions.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::time::{MissedTickBehavior, interval_at};
use uuid::Uuid;

use crate::active_math::{
    ActiveLookEvaluationV1, ActiveMathErrorV1, evaluate_active_look_with_control_v1,
};
use crate::background::BackgroundCancellation;
use crate::canonical_json::canonical_sha256;
use crate::ledger::repository::active_learning::{
    ACTIVE_LOOK_LEASE_MAX_MILLIS, ActiveFrozenLookMember, ActiveLookClaimAck,
    ActiveLookClaimRequest, ActiveLookCommit, ActiveLookCommitAck, ActiveLookFailure,
    ActiveLookFailureAck, ActiveLookFailureKind, ActiveLookLeaseAck, ActiveLookLeaseRenewal,
    ActiveLookWorkCandidate,
};
use crate::ledger::writer::LedgerWriterClient;

const WRITER_TIMEOUT: Duration = Duration::from_secs(5);
const LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveEvaluatorOutcome {
    Applied,
    AlreadyApplied,
    Deferred,
    Stale,
    Cancelled,
    PermanentFailure,
}

pub(crate) async fn execute_active_look(
    writer: &LedgerWriterClient,
    work: ActiveLookWorkCandidate,
    cancellation: BackgroundCancellation,
) -> ActiveEvaluatorOutcome {
    if cancellation.is_cancelled() {
        return ActiveEvaluatorOutcome::Cancelled;
    }
    let lease_token_hash = match canonical_sha256(&serde_json::json!({
        "shape": "active_look_worker_lease_v1",
        "nonce": Uuid::now_v7(),
    })) {
        Ok(value) => value,
        Err(_) => return ActiveEvaluatorOutcome::PermanentFailure,
    };
    let claim = match writer
        .claim_active_look_until(
            ActiveLookClaimRequest {
                active_look_claim_id: Uuid::now_v7(),
                claim_state_event_id: Uuid::now_v7(),
                evaluating_authorization_state_event_id: Uuid::now_v7(),
                skipped_failure_id: Uuid::now_v7(),
                active_experiment_id: work.active_experiment_id,
                lease_token_hash,
                lease_duration_millis: ACTIVE_LOOK_LEASE_MAX_MILLIS,
            },
            writer_deadline(),
        )
        .await
    {
        Ok(
            ActiveLookClaimAck::Claimed(claim)
            | ActiveLookClaimAck::AlreadyOwned(claim)
            | ActiveLookClaimAck::Reclaimed(claim),
        ) => claim,
        Ok(ActiveLookClaimAck::Skipped(_)) => return ActiveEvaluatorOutcome::Applied,
        Ok(ActiveLookClaimAck::Busy { .. } | ActiveLookClaimAck::NoBoundary) => {
            return ActiveEvaluatorOutcome::Deferred;
        }
        Ok(ActiveLookClaimAck::AuthorityChanged) => return ActiveEvaluatorOutcome::Stale,
        Ok(ActiveLookClaimAck::TransactionNotStarted) => {
            return ActiveEvaluatorOutcome::Cancelled;
        }
        Err(_) => return ActiveEvaluatorOutcome::Deferred,
    };
    let members = claim
        .members
        .iter()
        .copied()
        .map(ActiveFrozenLookMember::math_member)
        .collect::<Vec<_>>();
    let continue_evaluation = Arc::new(AtomicBool::new(true));
    let blocking_control = continue_evaluation.clone();
    let policy = claim.policy;
    let as_of_unix_ms = claim.as_of_unix_ms;
    let mut evaluation = tokio::task::spawn_blocking(move || {
        evaluate_active_look_with_control_v1(policy, as_of_unix_ms, &members, |_| {
            blocking_control.load(Ordering::Acquire)
        })
    });
    let mut renewal = interval_at(
        tokio::time::Instant::now() + LEASE_RENEW_INTERVAL,
        LEASE_RENEW_INTERVAL,
    );
    renewal.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let evaluated = loop {
        tokio::select! {
            biased;
            () = cancellation.cancelled() => {
                continue_evaluation.store(false, Ordering::Release);
                let _ = evaluation.await;
                return ActiveEvaluatorOutcome::Cancelled;
            }
            result = &mut evaluation => {
                break match result {
                    Ok(result) => result,
                    Err(_) => return ActiveEvaluatorOutcome::PermanentFailure,
                };
            }
            _ = renewal.tick() => {
                let acknowledgement = writer
                    .renew_active_look_lease_until(
                        ActiveLookLeaseRenewal {
                            active_look_claim_id: claim.active_look_claim_id,
                            claim_state_event_id: Uuid::now_v7(),
                            lease_token_hash: claim.lease_token_hash.clone(),
                            lease_duration_millis: ACTIVE_LOOK_LEASE_MAX_MILLIS,
                        },
                        writer_deadline(),
                    )
                    .await;
                if !matches!(
                    acknowledgement,
                    Ok(ActiveLookLeaseAck::Renewed { .. }
                        | ActiveLookLeaseAck::AlreadyRenewed { .. })
                ) {
                    continue_evaluation.store(false, Ordering::Release);
                    let _ = evaluation.await;
                    return match acknowledgement {
                        Ok(ActiveLookLeaseAck::LeaseLost | ActiveLookLeaseAck::AuthorityChanged) => {
                            ActiveEvaluatorOutcome::Stale
                        }
                        Ok(ActiveLookLeaseAck::TransactionNotStarted) => {
                            ActiveEvaluatorOutcome::Cancelled
                        }
                        Err(_) => ActiveEvaluatorOutcome::Deferred,
                        Ok(_) => ActiveEvaluatorOutcome::PermanentFailure,
                    };
                }
            }
        }
    };
    match evaluated {
        Ok(ActiveLookEvaluationV1::Completed(audit)) => {
            match writer
                .commit_active_look_until(
                    ActiveLookCommit {
                        active_look_claim_id: claim.active_look_claim_id,
                        lease_token_hash: claim.lease_token_hash,
                        active_outcome_look_id: Uuid::now_v7(),
                        claim_terminal_state_event_id: Uuid::now_v7(),
                        authorization_state_event_id: Uuid::now_v7(),
                        audit,
                    },
                    writer_deadline(),
                )
                .await
            {
                Ok(ActiveLookCommitAck::Applied(_)) => ActiveEvaluatorOutcome::Applied,
                Ok(ActiveLookCommitAck::AlreadyApplied(_)) => {
                    ActiveEvaluatorOutcome::AlreadyApplied
                }
                Ok(ActiveLookCommitAck::LeaseLost | ActiveLookCommitAck::AuthorityChanged) => {
                    ActiveEvaluatorOutcome::Stale
                }
                Ok(ActiveLookCommitAck::TransactionNotStarted) => ActiveEvaluatorOutcome::Cancelled,
                Ok(ActiveLookCommitAck::Conflict) => ActiveEvaluatorOutcome::PermanentFailure,
                Err(_) => ActiveEvaluatorOutcome::Deferred,
            }
        }
        Ok(ActiveLookEvaluationV1::Skipped(_)) => {
            persist_evaluation_failure(
                writer,
                &claim,
                ActiveLookFailureKind::IntegrityFailure,
                "active_math.raw_minimum_mismatch",
            )
            .await
        }
        Err(ActiveMathErrorV1::Canceled) => ActiveEvaluatorOutcome::Cancelled,
        Err(error) => {
            let kind = match error {
                ActiveMathErrorV1::InvalidInput
                | ActiveMathErrorV1::InvalidOrder
                | ActiveMathErrorV1::InvalidShape
                | ActiveMathErrorV1::FutureSkew => ActiveLookFailureKind::IntegrityFailure,
                _ => ActiveLookFailureKind::NumericFailure,
            };
            persist_evaluation_failure(writer, &claim, kind, active_math_failure_reason(error))
                .await
        }
    }
}

async fn persist_evaluation_failure(
    writer: &LedgerWriterClient,
    claim: &crate::ledger::repository::active_learning::ActiveLookClaimReceipt,
    failure_kind: ActiveLookFailureKind,
    stable_reason: &str,
) -> ActiveEvaluatorOutcome {
    match writer
        .fail_active_look_until(
            ActiveLookFailure {
                active_look_failure_id: Uuid::now_v7(),
                active_look_claim_id: claim.active_look_claim_id,
                claim_terminal_state_event_id: Uuid::now_v7(),
                authorization_state_event_id: Uuid::now_v7(),
                lease_token_hash: claim.lease_token_hash.clone(),
                failure_kind,
                stable_reason: stable_reason.to_string(),
            },
            writer_deadline(),
        )
        .await
    {
        Ok(ActiveLookFailureAck::Applied) => ActiveEvaluatorOutcome::Applied,
        Ok(ActiveLookFailureAck::AlreadyApplied) => ActiveEvaluatorOutcome::AlreadyApplied,
        Ok(ActiveLookFailureAck::LeaseLost | ActiveLookFailureAck::AuthorityChanged) => {
            ActiveEvaluatorOutcome::Stale
        }
        Ok(ActiveLookFailureAck::TransactionNotStarted) => ActiveEvaluatorOutcome::Cancelled,
        Ok(ActiveLookFailureAck::Conflict) => ActiveEvaluatorOutcome::PermanentFailure,
        Err(_) => ActiveEvaluatorOutcome::Deferred,
    }
}

const fn active_math_failure_reason(error: ActiveMathErrorV1) -> &'static str {
    match error {
        ActiveMathErrorV1::InvalidInput => "active_math.invalid_input",
        ActiveMathErrorV1::InvalidOrder => "active_math.invalid_order",
        ActiveMathErrorV1::FutureSkew => "active_math.future_skew",
        ActiveMathErrorV1::Nonfinite => "active_math.nonfinite",
        ActiveMathErrorV1::NumericFailure => "active_math.numeric_failure",
        ActiveMathErrorV1::InvalidShape => "active_math.invalid_shape",
        ActiveMathErrorV1::CdfFailure => "active_math.cdf_failure",
        ActiveMathErrorV1::SymmetryDisagreement => "active_math.symmetry_disagreement",
        ActiveMathErrorV1::NonmonotoneCdf => "active_math.nonmonotone_cdf",
        ActiveMathErrorV1::QuantileUnresolved => "active_math.quantile_unresolved",
        ActiveMathErrorV1::CdfBudgetExceeded => "active_math.cdf_budget_exceeded",
        ActiveMathErrorV1::DisjointOrientations => "active_math.disjoint_orientations",
        ActiveMathErrorV1::PosteriorResolutionExceeded => {
            "active_math.posterior_resolution_exceeded"
        }
        ActiveMathErrorV1::Canceled => "active_math.canceled",
    }
}

fn writer_deadline() -> Instant {
    Instant::now()
        .checked_add(WRITER_TIMEOUT)
        .unwrap_or_else(Instant::now)
}

#[cfg(test)]
mod tests {
    use rusqlite::Connection;

    use super::*;
    use crate::ledger::repository::active::ActiveSignalDisposition;
    use crate::ledger::repository::active_learning::{
        ActiveAuthorizationObserveAck, ActiveAuthorizationState, select_active_look_work,
    };
    use crate::ledger::writer::LedgerWriterOwner;

    #[tokio::test]
    async fn bounded_worker_discovers_evaluates_renews_and_commits() {
        let mut fixture =
            crate::ledger::repository::active::tests::fixture_with_max_canary_roots(128);
        crate::ledger::repository::active_learning::tests::fill_drained_boundary_with_labels(
            &mut fixture,
            32,
            32,
            ActiveSignalDisposition::Success,
            ActiveSignalDisposition::Failure,
        );
        let project_uuid = fixture.activated.identity.project_uuid;
        let config_generation_id = fixture.activated.identity.config_generation_id.clone();
        let active_experiment_id = fixture.admission.active_experiment_id;
        let database_path = fixture.config.database_path.clone();
        let discovery_connection = Connection::open(&database_path).unwrap();
        let selected = select_active_look_work(
            &discovery_connection,
            project_uuid,
            &config_generation_id,
            1,
        )
        .unwrap();
        assert_eq!(
            selected,
            vec![ActiveLookWorkCandidate {
                active_experiment_id,
            }]
        );
        drop(discovery_connection);
        let repository = fixture.activated.repository;
        let (mut owner, writer) = LedgerWriterOwner::start(repository, 16).unwrap();
        assert_eq!(
            execute_active_look(
                &writer,
                ActiveLookWorkCandidate {
                    active_experiment_id,
                },
                BackgroundCancellation::new(),
            )
            .await,
            ActiveEvaluatorOutcome::Applied
        );
        let authorization = writer
            .observe_active_authorization_until(
                active_experiment_id,
                chrono::Utc::now().timestamp_millis(),
                Instant::now() + Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert!(matches!(
            authorization,
            ActiveAuthorizationObserveAck::Current(snapshot)
                if snapshot.state == ActiveAuthorizationState::Passed && snapshot.authorizing
        ));
        owner
            .drain_until(Instant::now() + Duration::from_secs(5))
            .await
            .unwrap();
        let connection = Connection::open(database_path).unwrap();
        let look_count: i64 = connection
            .query_row(
                "SELECT count(*) FROM active_outcome_looks
                 WHERE active_experiment_id = ?1",
                [active_experiment_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(look_count, 1);
        let renewal_count: i64 = connection
            .query_row(
                "SELECT count(*) FROM active_look_claim_state_events AS state
                 JOIN active_look_claims AS claim
                   ON claim.active_look_claim_id = state.active_look_claim_id
                 WHERE claim.active_experiment_id = ?1
                   AND state.state = 'renewed'",
                [active_experiment_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(renewal_count >= 1);
        let remaining =
            select_active_look_work(&connection, project_uuid, &config_generation_id, 1).unwrap();
        assert!(remaining.is_empty());
    }
}
