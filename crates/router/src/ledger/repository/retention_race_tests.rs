// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-component races between retention, search snapshots, and decision writes.

use std::sync::{Arc, Barrier};
use std::thread;

use rusqlite::{Connection, params};
use uuid::Uuid;

use super::decision::DecisionAuditAck;
use super::decision::tests::{existing_neighbor_audit, insert_test_decision_graphs};
use super::materialization::{VerifiedVectorLinkSourceLoad, load_verified_vector_link_source};
use super::process::{HeartbeatAck, HeartbeatRenewal};
use super::retention::{RetentionAck, RetentionRequest};
use super::vector_search::{search_projected_neighbors, search_projected_neighbors_in_transaction};
use super::{LedgerRepository, TransactionStartGuard, ready_evaluated_runtime_fixture};
use crate::config::LearningConfig;
use crate::vector::VectorRecordId;

const RETENTION_DAY_MILLIS: i64 = 86_400_000;
const DEPENDENT_DECISION_COUNT: usize = 1_001;

struct PauseAfterBegin {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl TransactionStartGuard for PauseAfterBegin {
    fn permits_transaction(&self) -> bool {
        self.entered.wait();
        self.release.wait();
        true
    }
}

fn complete_learning(config: &mut crate::config::RouterConfig) {
    let embedder = config.pools[0].learning.as_ref().unwrap().embedder.clone();
    config.pools[0].learning = Some(LearningConfig {
        version: 1,
        embedder,
        top_k: Some(1),
        radius: Some(1.0),
        min_points: Some(1),
        min_independent_roots: Some(1),
        min_effective_samples: Some(1.0),
        min_coverage: Some(0.0),
        time_decay_half_life_seconds: Some(3_600.0),
        prior_success: Some(1.0),
        prior_failure: Some(1.0),
        familywise_credible_level: Some(0.95),
        promotion_lower_bound: Some(0.0),
        retention_lower_bound: None,
        holdout_probability: None,
        active_canary_fraction: None,
    });
}

fn scalar(connection: &Connection, sql: &str) -> i64 {
    connection.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[test]
fn retention_linearizes_marker_before_queued_write_and_fresh_search() {
    let (
        _temporary,
        activation_config,
        mut activated,
        vector_space_id,
        root,
        evidence_vector_link_id,
        _evaluation_id,
    ) = ready_evaluated_runtime_fixture();
    let mut audit_config = activation_config.clone();
    complete_learning(&mut audit_config);

    let source = match load_verified_vector_link_source(
        &activated.repository.connection,
        &vector_space_id,
        VectorRecordId::new(evidence_vector_link_id).unwrap(),
    )
    .unwrap()
    {
        VerifiedVectorLinkSourceLoad::Verified(source) => source,
        _ => panic!("ready evaluated source did not verify"),
    };
    let partition_id = source.partition_id;
    let query = source
        .vector_record
        .as_ref()
        .expect("ready source must retain a vector")
        .vector()
        .vector()
        .clone();

    let audits = (0..DEPENDENT_DECISION_COUNT)
        .map(|index| {
            existing_neighbor_audit(
                &activated,
                &audit_config,
                Uuid::now_v7(),
                evidence_vector_link_id,
                100 + i64::try_from(index).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    insert_test_decision_graphs(&mut activated.repository.connection, &audits);
    let queued_audit = existing_neighbor_audit(
        &activated,
        &audit_config,
        Uuid::now_v7(),
        evidence_vector_link_id,
        2_000,
    );
    drop(audits);

    let mut decision_writer = LedgerRepository::activate_at(&activation_config, 0)
        .unwrap()
        .repository;
    activated.repository.retention_days = 1;
    let retention_at = RETENTION_DAY_MILLIS + 10_000;
    assert!(matches!(
        activated
            .repository
            .renew_heartbeat(HeartbeatRenewal::new(retention_at - 1).unwrap())
            .unwrap(),
        HeartbeatAck::Applied { .. }
    ));
    activated
        .repository
        .connection
        .execute_batch(
            "CREATE TRIGGER fail_decision_retention_receipt
             BEFORE INSERT ON decision_retention_receipts
             BEGIN
                SELECT RAISE(ABORT, 'injected decision retention receipt failure');
             END;",
        )
        .unwrap();
    let failed_request =
        RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), retention_at).unwrap();
    assert!(activated.repository.run_retention(&failed_request).is_err());
    assert_eq!(
        scalar(
            &activated.repository.connection,
            "SELECT count(*) FROM decision_retiring_anchors"
        ),
        0
    );
    assert_eq!(
        scalar(
            &activated.repository.connection,
            "SELECT count(*) FROM decisions"
        ),
        i64::try_from(DEPENDENT_DECISION_COUNT).unwrap()
    );
    assert_eq!(
        scalar(
            &activated.repository.connection,
            "SELECT count(*) FROM decision_retention_receipts"
        ),
        0
    );
    assert_eq!(
        scalar(
            &activated.repository.connection,
            "SELECT count(*) FROM vector_source_change_events"
        ),
        1
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                params![evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    activated
        .repository
        .connection
        .execute_batch("DROP TRIGGER fail_decision_retention_receipt")
        .unwrap();
    let request = RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), retention_at + 1).unwrap();

    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let retention_entered = entered.clone();
    let retention_release = release.clone();
    let mut retention_repository = activated.repository;
    let retention = thread::spawn(move || {
        let result = retention_repository.run_retention_with_start_check(&request, || {
            Some(PauseAfterBegin {
                entered: retention_entered,
                release: retention_release,
            })
        });
        (retention_repository, result)
    });
    entered.wait();

    let mut reader = Connection::open(&activation_config.database_path).unwrap();
    reader.execute_batch("PRAGMA query_only = ON").unwrap();
    let snapshot = reader.transaction().unwrap();
    assert_eq!(
        search_projected_neighbors_in_transaction(
            &snapshot,
            &vector_space_id,
            partition_id,
            &query,
            1,
        )
        .unwrap()
        .len(),
        1
    );

    let decision_attempting = Arc::new(Barrier::new(2));
    let writer_attempting = decision_attempting.clone();
    let writer_release = Arc::new(Barrier::new(2));
    let decision_release = writer_release.clone();
    let max_evidence_records = activation_config.max_evidence_records;
    let decision = thread::spawn(move || {
        let result = decision_writer.record_decision_audit_with_start_check::<()>(
            &queued_audit,
            max_evidence_records,
            Uuid::now_v7(),
            || {
                writer_attempting.wait();
                decision_release.wait();
                Some(())
            },
        );
        (decision_writer, result)
    });
    decision_attempting.wait();
    release.wait();

    let (mut retention_repository, applied) = retention.join().unwrap();
    let RetentionAck::Applied {
        summary,
        observation,
    } = applied.unwrap()
    else {
        panic!("first retention request did not apply");
    };
    assert_eq!(summary.retention_batch_id, request.retention_batch_id);
    assert!(observation.more_cleanup);
    writer_release.wait();
    let (_decision_writer, decision_result) = decision.join().unwrap();
    assert_eq!(decision_result.unwrap(), DecisionAuditAck::SourceRetiring);

    assert_eq!(
        search_projected_neighbors_in_transaction(
            &snapshot,
            &vector_space_id,
            partition_id,
            &query,
            1,
        )
        .unwrap()
        .len(),
        1
    );
    snapshot.commit().unwrap();
    assert!(
        search_projected_neighbors(&reader, &vector_space_id, partition_id, &query, 1)
            .unwrap()
            .is_empty()
    );

    assert_eq!(
        scalar(
            &retention_repository.connection,
            "SELECT count(*) FROM decision_retiring_anchors"
        ),
        1
    );
    assert_eq!(
        scalar(
            &retention_repository.connection,
            "SELECT count(*) FROM decisions"
        ),
        1
    );
    assert_eq!(
        retention_repository
            .connection
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                params![evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    let receipt = retention_repository
        .connection
        .query_row(
            "SELECT source_forced, new_marker_count, deleted_decision_count,
                    blocked_anchor_count, deleted_anchor_count, more_cleanup,
                    canonical_payload_hash
             FROM decision_retention_receipts WHERE retention_batch_id = ?1",
            [request.retention_batch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, String>(6)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(receipt.0, 1);
    assert_eq!(receipt.1, 1);
    assert_eq!(receipt.2, 1_000);
    assert_eq!(receipt.3, 1);
    assert_eq!(receipt.4, 0);
    assert_eq!(receipt.5, 1);

    let replay = retention_repository.run_retention(&request).unwrap();
    assert!(matches!(
        replay,
        RetentionAck::AlreadyApplied {
            summary: replayed,
            observation: replayed_observation,
        } if replayed == summary && replayed_observation == observation
    ));
    assert_eq!(
        scalar(
            &retention_repository.connection,
            "SELECT count(*) FROM decision_retention_receipts"
        ),
        1
    );
    assert_eq!(
        retention_repository
            .connection
            .query_row(
                "SELECT canonical_payload_hash FROM decision_retention_receipts
                 WHERE retention_batch_id = ?1",
                [request.retention_batch_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        receipt.6
    );
}
