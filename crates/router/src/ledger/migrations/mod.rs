// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered, embedded Router ledger migration manifest.

/// SQLite application identifier reserved by the Router ledger schema.
pub(crate) const LEDGER_APPLICATION_ID: i32 = 1_313_690_194;

/// One immutable source-controlled SQLite migration.
#[derive(Clone, Copy)]
pub(crate) struct Migration {
    /// Ordered positive schema version.
    pub(crate) version: i64,
    /// Stable source-controlled migration name.
    pub(crate) name: &'static str,
    /// Canonical lowercase SHA-256 of the exact embedded SQL bytes.
    pub(crate) checksum_sha256: &'static str,
    /// Exact SQL applied inside the migration transaction.
    pub(crate) sql: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/router_migration_checksums.rs"));

/// Highest schema version understood by this Router build.
pub(crate) const CURRENT_SCHEMA_VERSION: i64 = MIGRATIONS[MIGRATIONS.len() - 1].version;

#[cfg(test)]
mod tests {
    use rusqlite::types::Value;
    use rusqlite::{Connection, TransactionBehavior, params};

    use super::{CURRENT_SCHEMA_VERSION, LEDGER_APPLICATION_ID, MIGRATIONS};

    const PROJECT_UUID: &str = "018f0000-0000-7000-8000-000000000001";
    const PROCESS_UUID: &str = "018f0000-0000-7000-8000-000000000002";
    const LEARNING_UUID: &str = "018f0000-0000-7000-8000-000000000003";
    const ANCHOR_UUID: &str = "018f0000-0000-7000-8000-000000000004";
    const BATCH_UUID: &str = "018f0000-0000-7000-8000-000000000005";
    const SHADOW_UUID: &str = "018f0000-0000-7000-8000-000000000006";
    const OPERATION_UUID: &str = "018f0000-0000-7000-8000-000000000007";
    const CONFIG_HASH: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const POLICY_HASH: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DEPENDENCY_KEY_ID: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PAYLOAD_HASH: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn manifest_is_ordered_and_embeds_the_generated_checksum() {
        let expected = [
            (
                1,
                "0001_initial",
                "40c2c629836e26984d257a0caf7efc3336d9b262a7310bf168870f96439b08df",
            ),
            (
                2,
                "0002_task10_recovery_leases",
                "bb5724a82f2e98c05bd8bcbfe9ac05ca738b41220547690165b88be11f7cdf63",
            ),
            (
                3,
                "0003_task11_retention",
                "bc6cbd6819b7de2ad0dd92edb98ada737176cd03e74d91ae4ac9a4da1ab9c1c4",
            ),
            (
                4,
                "0004_spec06_embedding_vectors",
                "cf13a252e041261351651350e5b1faffd7fecbdd5460bcf05c09e8d5b2fcaa23",
            ),
            (
                5,
                "0005_spec07_recommend_decisions",
                "5205c2d2248aac5c3e94e23ff344f06fa56fbe1adedb7524e365215be23f840c",
            ),
            (
                6,
                "0006_spec08_active_canary",
                "7171249250a3f275f3e652c70ce3ad8857df46afb0a4e1c67c62a961fbf8652c",
            ),
            (
                7,
                "0007_spec10_operator_operations",
                "93056b96a66e812fed842ef3975f1e2ad993c0a8df9103d612ef7dfd504bbd16",
            ),
        ];
        assert_eq!(MIGRATIONS.len(), expected.len());
        for (migration, (version, name, checksum)) in MIGRATIONS.iter().zip(expected) {
            assert_eq!(migration.version, version);
            assert_eq!(migration.name, name);
            assert_eq!(migration.checksum_sha256, checksum);
            assert!(migration.sql.ends_with('\n'));
        }
        assert_eq!(CURRENT_SCHEMA_VERSION, 7);
        assert_eq!(MIGRATIONS.last().unwrap().version, CURRENT_SCHEMA_VERSION);
        assert!(
            MIGRATIONS[0]
                .sql
                .contains(&format!("PRAGMA application_id = {LEDGER_APPLICATION_ID};"))
        );
    }

    #[test]
    fn second_migration_preserves_dependency_sequences_and_indexes() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        connection.execute_batch(MIGRATIONS[0].sql).unwrap();
        seed_dependency_operation(&connection);
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    event_seq, dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, cooloff_until_unix_ms,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (4, ?1, ?2, ?3, ?4, 'admitted', 2, 99, 10, ?5)",
                params![
                    "018f0000-0000-7000-8000-000000000008",
                    DEPENDENCY_KEY_ID,
                    OPERATION_UUID,
                    ANCHOR_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    event_seq, dependency_state_event_id, dependency_key_id,
                    state, consecutive_failures, cooloff_until_unix_ms,
                    failure_class, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (9, ?1, ?2, 'failure', 2, 99,
                           'router.provider.timeout', 11, ?3)",
                params![
                    "018f0000-0000-7000-8000-000000000009",
                    DEPENDENCY_KEY_ID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();

        connection.execute_batch(MIGRATIONS[1].sql).unwrap();

        let rows = connection
            .prepare("SELECT event_seq, state FROM dependency_state_events ORDER BY event_seq")
            .unwrap()
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            rows,
            vec![(4, "admitted".to_string()), (9, "failure".to_string())]
        );
        let admitted_hash = connection
            .query_row(
                "SELECT canonical_payload_hash FROM dependency_state_events
                 WHERE event_seq = 4",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        assert_eq!(admitted_hash, PAYLOAD_HASH);

        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, cooloff_until_unix_ms,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, 'orphaned_in_flight', 2, 99, 12, ?5)",
                params![
                    "018f0000-0000-7000-8000-000000000010",
                    DEPENDENCY_KEY_ID,
                    OPERATION_UUID,
                    ANCHOR_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT max(event_seq) FROM dependency_state_events",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            10
        );
        assert!(
            connection
                .execute(
                    "INSERT INTO dependency_state_events (
                        dependency_state_event_id, dependency_key_id,
                        dependency_operation_id, anchor_id, state,
                        consecutive_failures, created_at_unix_ms,
                        canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, 'success', 0, 13, ?5)",
                    params![
                        "018f0000-0000-7000-8000-000000000011",
                        DEPENDENCY_KEY_ID,
                        OPERATION_UUID,
                        ANCHOR_UUID,
                        PAYLOAD_HASH,
                    ],
                )
                .is_err()
        );

        let indexes = connection
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'index' AND tbl_name = 'dependency_state_events'
                   AND name NOT LIKE 'sqlite_autoindex%'
                 ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            indexes,
            vec![
                "idx_dependency_state_latest",
                "uq_dependency_operation_claim_state",
                "uq_dependency_operation_terminal_state",
            ]
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn second_migration_adds_lease_snapshots_and_orphan_terminal_shape() {
        let connection = Connection::open_in_memory().unwrap();
        connection.execute_batch(MIGRATIONS[0].sql).unwrap();
        connection.execute_batch(MIGRATIONS[1].sql).unwrap();

        let columns = connection
            .prepare(
                "SELECT name, type, \"notnull\", coalesce(dflt_value, '')
                 FROM pragma_table_info('embedding_job_state_events')
                 WHERE name IN (
                    'lease_token', 'lease_expires_at_unix_ms',
                    'attempt_count', 'next_eligible_at_unix_ms'
                 ) ORDER BY cid",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            columns,
            vec![
                (
                    "lease_token".to_string(),
                    "TEXT".to_string(),
                    0,
                    String::new()
                ),
                (
                    "lease_expires_at_unix_ms".to_string(),
                    "INTEGER".to_string(),
                    0,
                    String::new(),
                ),
                (
                    "attempt_count".to_string(),
                    "INTEGER".to_string(),
                    1,
                    "0".to_string()
                ),
                (
                    "next_eligible_at_unix_ms".to_string(),
                    "INTEGER".to_string(),
                    1,
                    "0".to_string(),
                ),
            ]
        );

        connection
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, cooloff_until_unix_ms,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, 'orphaned_in_flight', 3, 99, 20, ?5)",
                params![
                    "018f0000-0000-7000-8000-000000000020",
                    DEPENDENCY_KEY_ID,
                    "018f0000-0000-7000-8000-000000000021",
                    "018f0000-0000-7000-8000-000000000022",
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        assert!(
            connection
                .execute(
                    "INSERT INTO dependency_state_events (
                        dependency_state_event_id, dependency_key_id,
                        dependency_operation_id, anchor_id, state,
                        consecutive_failures, cooloff_until_unix_ms,
                        failure_class, created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, 'orphaned_in_flight', 1, 99,
                               'router.provider.timeout', 21, ?5)",
                    params![
                        "018f0000-0000-7000-8000-000000000023",
                        DEPENDENCY_KEY_ID,
                        "018f0000-0000-7000-8000-000000000024",
                        "018f0000-0000-7000-8000-000000000025",
                        PAYLOAD_HASH,
                    ],
                )
                .is_err()
        );
    }

    #[test]
    fn third_migration_preserves_v2_data_sequences_indexes_and_foreign_keys() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        connection.execute_batch(MIGRATIONS[0].sql).unwrap();
        connection.execute_batch(MIGRATIONS[1].sql).unwrap();
        seed_dependency_operation(&connection);

        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    event_seq, dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, cooloff_until_unix_ms,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (4, ?1, ?2, ?3, ?4, 'admitted', 2, 99, 10, ?5)",
                params![
                    "018f0000-0000-7000-8000-000000000008",
                    DEPENDENCY_KEY_ID,
                    OPERATION_UUID,
                    ANCHOR_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    event_seq, dependency_state_event_id, dependency_key_id,
                    state, consecutive_failures, cooloff_until_unix_ms,
                    failure_class, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (9, ?1, ?2, 'failure', 2, 99,
                           'router.provider.timeout', 11, ?3)",
                params![
                    "018f0000-0000-7000-8000-000000000009",
                    DEPENDENCY_KEY_ID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    event_seq, dependency_state_event_id, dependency_key_id,
                    state, consecutive_failures, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (20, ?1, ?2, 'success', 0, 12, ?3)",
                params![
                    "018f0000-0000-7000-8000-000000000020",
                    DEPENDENCY_KEY_ID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "DELETE FROM dependency_state_events WHERE event_seq = 20",
                [],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO retention_batches (
                    retention_batch_id, project_uuid, process_instance_id,
                    age_expired, count_excess, selected_count,
                    selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                    selection_hash, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 1, 0, 0, 10, 20, ?4, 21, ?4)",
                params![
                    "018f0000-0000-7000-8000-000000000021",
                    PROJECT_UUID,
                    PROCESS_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();

        let dependency_rows_before = dependency_state_rows(&connection);
        let retention_before = retention_batch_shape(&connection);
        let dependency_foreign_keys_before = foreign_keys(&connection, "dependency_state_events");
        let retention_foreign_keys_before = foreign_keys(&connection, "retention_batches");
        assert_eq!(sequence_value(&connection, "dependency_state_events"), 20);

        connection.execute_batch(MIGRATIONS[2].sql).unwrap();

        assert_eq!(dependency_state_rows(&connection), dependency_rows_before);
        assert_eq!(retention_batch_shape(&connection), retention_before);
        assert_eq!(
            connection
                .query_row(
                    "SELECT summary_shape_version FROM retention_batches",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            2
        );
        assert_eq!(
            foreign_keys(&connection, "dependency_state_events"),
            dependency_foreign_keys_before
        );
        assert_eq!(
            foreign_keys(&connection, "retention_batches"),
            retention_foreign_keys_before
        );
        assert_eq!(sequence_value(&connection, "dependency_state_events"), 20);
        assert_eq!(
            explicit_indexes(&connection, "dependency_state_events"),
            vec![
                "idx_dependency_state_anchor",
                "idx_dependency_state_latest",
                "uq_dependency_operation_claim_state",
                "uq_dependency_operation_terminal_state",
            ]
        );
        assert_eq!(
            index_columns(&connection, "idx_dependency_operation_anchor"),
            vec!["anchor_id"]
        );
        assert_eq!(
            index_columns(&connection, "idx_dependency_state_anchor"),
            vec!["anchor_id"]
        );
        assert_eq!(
            index_columns(&connection, "idx_health_event_anchor"),
            vec!["anchor_id"]
        );
        assert_eq!(
            index_columns(&connection, "idx_evidence_vector_link_anchor"),
            vec!["anchor_id"]
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);

        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    dependency_state_event_id, dependency_key_id, state,
                    consecutive_failures, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, 'orphaned_in_flight', 0, 22, ?3)",
                params![
                    "018f0000-0000-7000-8000-000000000022",
                    DEPENDENCY_KEY_ID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        assert_eq!(sequence_value(&connection, "dependency_state_events"), 21);
    }

    #[test]
    fn third_migration_enforces_dependency_and_retention_shapes() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_dependency_operation(&connection);

        let insert_unlinked = |event_id: &str,
                               state: &str,
                               consecutive_failures: i64,
                               cooloff_until_unix_ms: Option<i64>,
                               failure_class: Option<&str>| {
            connection.execute(
                "INSERT INTO dependency_state_events (
                        dependency_state_event_id, dependency_key_id, state,
                        consecutive_failures, cooloff_until_unix_ms,
                        failure_class, created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 30, ?7)",
                params![
                    event_id,
                    DEPENDENCY_KEY_ID,
                    state,
                    consecutive_failures,
                    cooloff_until_unix_ms,
                    failure_class,
                    PAYLOAD_HASH,
                ],
            )
        };
        insert_unlinked(
            "018f0000-0000-7000-8000-000000000030",
            "success",
            0,
            None,
            None,
        )
        .unwrap();
        insert_unlinked(
            "018f0000-0000-7000-8000-000000000031",
            "failure",
            1,
            Some(40),
            Some("router.provider.timeout"),
        )
        .unwrap();
        insert_unlinked(
            "018f0000-0000-7000-8000-000000000032",
            "skipped_cooloff",
            1,
            Some(40),
            None,
        )
        .unwrap();
        insert_unlinked(
            "018f0000-0000-7000-8000-000000000033",
            "orphaned_in_flight",
            0,
            None,
            None,
        )
        .unwrap();
        assert!(
            insert_unlinked(
                "018f0000-0000-7000-8000-000000000034",
                "admitted",
                0,
                None,
                None,
            )
            .is_err()
        );
        connection
            .execute(
                "INSERT INTO dependency_state_events (
                    dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, 'admitted', 0, 31, ?5)",
                params![
                    "018f0000-0000-7000-8000-000000000035",
                    DEPENDENCY_KEY_ID,
                    OPERATION_UUID,
                    ANCHOR_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();

        let insert_retention = |batch_id: &str,
                                age_expired: i64,
                                count_excess: i64,
                                selected_count: i64,
                                lower_bound: Option<i64>,
                                upper_bound: Option<i64>| {
            connection.execute(
                "INSERT INTO retention_batches (
                        retention_batch_id, project_uuid, process_instance_id,
                        conflict_health_event_id,
                        age_expired, count_excess, selected_count,
                        selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                        selection_hash, created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 50, ?10)",
                params![
                    batch_id,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    "018f0000-0000-7000-8000-000000000049",
                    age_expired,
                    count_excess,
                    selected_count,
                    lower_bound,
                    upper_bound,
                    PAYLOAD_HASH,
                ],
            )
        };
        insert_retention("018f0000-0000-7000-8000-000000000040", 0, 0, 0, None, None).unwrap();
        insert_retention(
            "018f0000-0000-7000-8000-000000000041",
            1,
            0,
            2,
            Some(10),
            Some(20),
        )
        .unwrap();
        assert!(
            insert_retention("018f0000-0000-7000-8000-000000000042", 1, 0, 0, None, None,).is_err()
        );
        assert!(
            insert_retention(
                "018f0000-0000-7000-8000-000000000043",
                0,
                0,
                0,
                Some(10),
                Some(20),
            )
            .is_err()
        );
        assert!(
            insert_retention(
                "018f0000-0000-7000-8000-000000000044",
                0,
                0,
                2,
                Some(10),
                Some(20),
            )
            .is_err()
        );
        assert!(
            insert_retention(
                "018f0000-0000-7000-8000-000000000045",
                0,
                1,
                2,
                None,
                Some(20),
            )
            .is_err()
        );
        assert!(
            insert_retention(
                "018f0000-0000-7000-8000-000000000046",
                0,
                1,
                2,
                Some(20),
                Some(10),
            )
            .is_err()
        );
        let embedding_lookup_plan = connection
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT 1 FROM evidence_vector_link_state_events
                 WHERE embedding_id = ?1",
            )
            .unwrap()
            .query_map(params!["018f0000-0000-7000-8000-000000000047"], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(
            embedding_lookup_plan
                .iter()
                .any(|detail| { detail.contains("idx_evidence_vector_link_state_embedding") })
        );
        let evidence_link_delete_plan = connection
            .prepare(
                "EXPLAIN QUERY PLAN
                 DELETE FROM evidence_vector_links
                 WHERE evidence_vector_link_id = ?1",
            )
            .unwrap()
            .query_map(params!["018f0000-0000-7000-8000-000000000048"], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(
            evidence_link_delete_plan
                .iter()
                .any(|detail| { detail.contains("idx_decision_neighbors_evidence_vector_link") })
        );
        let sample_batch_delete_plan = connection
            .prepare(
                "EXPLAIN QUERY PLAN
                 DELETE FROM sample_batches
                 WHERE sample_batch_id = ?1",
            )
            .unwrap()
            .query_map(params!["018f0000-0000-7000-8000-000000000050"], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(
            sample_batch_delete_plan
                .iter()
                .any(|detail| detail.contains("idx_shadow_attempt_sample_batch"))
        );
        assert!(
            !sample_batch_delete_plan
                .iter()
                .any(|detail| detail.contains("SCAN shadow_attempts"))
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn fourth_migration_replaces_placeholders_and_preserves_sequences() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..3] {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch(
                "DELETE FROM sqlite_sequence
                 WHERE name IN (
                    'embedding_job_state_events', 'evidence_vector_link_state_events'
                 );
                 INSERT INTO sqlite_sequence (name, seq)
                 VALUES
                    ('embedding_job_state_events', 41),
                    ('evidence_vector_link_state_events', 73);",
            )
            .unwrap();

        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        transaction.execute_batch(MIGRATIONS[3].sql).unwrap();
        transaction.commit().unwrap();

        assert_eq!(
            sequence_value(&connection, "embedding_job_state_events"),
            41
        );
        assert_eq!(
            sequence_value(&connection, "evidence_vector_link_state_events"),
            73
        );
        let expected_tables = [
            "canonical_routing_queries",
            "embedder_profiles",
            "embedding_job_state_events",
            "embedding_jobs",
            "embeddings",
            "evidence_vector_link_state_events",
            "evidence_vector_links",
            "pool_vector_space_mappings",
            "routing_partitions",
            "vector_index_manifest",
            "vector_index_rebuild_leases",
            "vector_materialization_job_state_events",
            "vector_materialization_jobs",
            "vector_source_change_events",
            "vector_space_source_sequences",
            "vector_spaces",
            "vectorization_outcomes",
        ];
        let actual_tables = connection
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name IN (
                    'canonical_routing_queries',
                    'embedder_profiles',
                    'embedding_job_state_events',
                    'embedding_jobs',
                    'embeddings',
                    'evidence_vector_link_state_events',
                    'evidence_vector_links',
                    'pool_vector_space_mappings',
                    'routing_partitions',
                    'vector_index_manifest',
                    'vector_index_rebuild_leases',
                    'vector_materialization_job_state_events',
                    'vector_materialization_jobs',
                    'vector_source_change_events',
                    'vector_space_source_sequences',
                    'vector_spaces',
                    'vectorization_outcomes'
                 ) ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(actual_tables, expected_tables);
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name LIKE 'spec06_vector_%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            explicit_indexes(&connection, "embedding_jobs"),
            vec![
                "idx_embedding_jobs_claim",
                "idx_embedding_jobs_failure_propagation",
                "idx_embedding_jobs_owner",
            ]
        );
        assert_eq!(
            explicit_indexes(&connection, "vector_materialization_jobs"),
            vec![
                "idx_vector_materialization_jobs_claim",
                "idx_vector_materialization_jobs_embedding_job",
                "idx_vector_materialization_jobs_owner",
            ]
        );
        assert_eq!(
            index_columns(&connection, "idx_vector_materialization_jobs_embedding_job"),
            vec!["embedding_job_id", "evidence_vector_link_id"]
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT sql FROM sqlite_schema
                     WHERE type = 'index'
                       AND name = 'idx_vector_materialization_jobs_embedding_job'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            concat!(
                "CREATE INDEX idx_vector_materialization_jobs_embedding_job\n",
                "    ON vector_materialization_jobs(embedding_job_id, evidence_vector_link_id)\n",
                "    WHERE embedding_job_id IS NOT NULL"
            )
        );
        let failure_propagation_plan = connection
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT evidence_vector_link_id
                 FROM vector_materialization_jobs
                 WHERE embedding_job_id = ?1
                   AND evidence_vector_link_id > ?2
                 ORDER BY evidence_vector_link_id
                 LIMIT 256",
            )
            .unwrap()
            .query_map(params!["a".repeat(64), PROCESS_UUID], |row| {
                row.get::<_, String>(3)
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(failure_propagation_plan.iter().any(|detail| {
            detail.contains("USING COVERING INDEX idx_vector_materialization_jobs_embedding_job")
        }));
        assert!(
            !failure_propagation_plan
                .iter()
                .any(|detail| detail.contains("SCAN vector_materialization_jobs")
                    || detail.contains("USE TEMP B-TREE"))
        );
        assert_eq!(
            explicit_indexes(&connection, "vector_index_manifest"),
            vec![
                "idx_vector_index_manifest_cleanup",
                "uq_vector_index_manifest_building",
                "uq_vector_index_manifest_current",
            ]
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn fourth_migration_rejects_each_nonempty_placeholder_atomically() {
        let cases = [
            (
                "vector_spaces",
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    '018f0000-0000-7000-8000-000000000001', 0,
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
                 )",
            ),
            (
                "embedding_jobs",
                "INSERT INTO embedding_jobs (
                    embedding_job_id, vector_space_id, canonical_query_hash, content_hash,
                    next_eligible_at_unix_ms, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                    'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd',
                    0, 0, 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
                 )",
            ),
            (
                "embedding_job_state_events",
                "INSERT INTO embedding_job_state_events (
                    embedding_job_state_event_id, embedding_job_id, state,
                    attempt_generation, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000001',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'pending', 0, 0,
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
                 )",
            ),
            (
                "embeddings",
                "INSERT INTO embeddings (
                    embedding_id, vector_space_id, canonical_query_hash, content_hash,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000001',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                    0, 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
                 )",
            ),
            (
                "evidence_vector_links",
                "INSERT INTO evidence_vector_links (
                    evidence_vector_link_id, shadow_attempt_id, anchor_id, root_uuid,
                    learning_generation_id, vector_space_id, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000001',
                    '018f0000-0000-7000-8000-000000000002',
                    '018f0000-0000-7000-8000-000000000003',
                    '018f0000-0000-7000-8000-000000000004',
                    '018f0000-0000-7000-8000-000000000005',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    0, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
                 )",
            ),
            (
                "evidence_vector_link_state_events",
                "INSERT INTO evidence_vector_link_state_events (
                    evidence_vector_link_state_event_id, evidence_vector_link_id,
                    state, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000001',
                    '018f0000-0000-7000-8000-000000000002', 'pending', 0,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 )",
            ),
        ];

        for (table, seed) in cases {
            let mut connection = Connection::open_in_memory().unwrap();
            for migration in &MIGRATIONS[..3] {
                connection.execute_batch(migration.sql).unwrap();
            }
            connection
                .execute_batch("PRAGMA foreign_keys = OFF;")
                .unwrap();
            connection.execute_batch(seed).unwrap();
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Exclusive)
                .unwrap();
            assert!(
                transaction.execute_batch(MIGRATIONS[3].sql).is_err(),
                "{table}"
            );
            drop(transaction);

            assert_eq!(
                connection
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                1,
                "{table}"
            );
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_schema
                         WHERE name LIKE 'spec06_vector_%'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0,
                "{table}"
            );
        }
    }

    #[test]
    fn fourth_migration_rolls_back_a_late_schema_failure() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..3] {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch(
                "CREATE TABLE vector_index_rebuild_leases (
                    injected_marker INTEGER NOT NULL
                 ) STRICT;",
            )
            .unwrap();
        let legacy_vector_spaces_sql = connection
            .query_row(
                "SELECT sql FROM sqlite_schema
                 WHERE type = 'table' AND name = 'vector_spaces'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();

        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        assert!(transaction.execute_batch(MIGRATIONS[3].sql).is_err());
        drop(transaction);

        assert_eq!(
            connection
                .query_row(
                    "SELECT sql FROM sqlite_schema
                     WHERE type = 'table' AND name = 'vector_spaces'",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            legacy_vector_spaces_sql
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name IN ('embedder_profiles', 'vector_index_manifest')",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('vector_index_rebuild_leases')
                     WHERE name = 'injected_marker'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name LIKE 'spec06_vector_%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn fourth_migration_manifest_shape_and_root_name_are_strict() {
        let connection = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();
        let space = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let objects_hash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let payload_hash = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        let insert = |root: &str, objects: &str| {
            connection.execute(
                "INSERT INTO vector_index_manifest (
                    vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256,
                    base_source_seq, applied_source_seq, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, 1, 'building', ?2, 2, ?3, ?4, 0, 0, 0, ?5)",
                params![space, root, objects, objects_hash, payload_hash],
            )
        };
        assert!(
            insert(
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1",
                "[0,0,0,0,0]",
            )
            .is_ok()
        );
        connection
            .execute("DELETE FROM vector_index_manifest", [])
            .unwrap();
        assert!(
            insert(
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g2",
                "[0,0,0,0,0]",
            )
            .is_err()
        );
        assert!(
            insert(
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1",
                "[0,0,0,0]",
            )
            .is_err()
        );

        let root = "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1";
        let insert_state = |state: &str,
                            activated_at: Option<i64>,
                            retired_at: Option<i64>,
                            source_count: Option<i64>,
                            source_fingerprint: Option<&str>| {
            connection.execute(
                "INSERT INTO vector_index_manifest (
                    vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256,
                    base_source_seq, applied_source_seq, source_record_count,
                    source_fingerprint_sha256, created_at_unix_ms,
                    activated_at_unix_ms, retired_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, 1, ?2, ?3, 2, '[0,0,0,0,0]', ?4, 0, 0, ?5, ?6, 0, ?7, ?8, ?9
                 )",
                params![
                    space,
                    state,
                    root,
                    objects_hash,
                    source_count,
                    source_fingerprint,
                    activated_at,
                    retired_at,
                    payload_hash,
                ],
            )
        };
        assert!(insert_state("active", Some(1), None, None, None).is_err());
        assert!(insert_state("active", Some(1), None, Some(0), Some(objects_hash)).is_ok());
        connection
            .execute("DELETE FROM vector_index_manifest", [])
            .unwrap();
        assert!(insert_state("retired", Some(1), Some(2), None, None).is_err());
        assert!(insert_state("retired", Some(1), Some(2), Some(0), Some(objects_hash),).is_ok());
    }

    #[test]
    fn fourth_migration_preserves_shadow_terminal_and_nullable_label_semantics() {
        let connection = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();
        let insert =
            |terminal_class: &str, evaluation_id: Option<&str>, quality_label: Option<&str>| {
                connection.execute(
                    "INSERT INTO evidence_vector_links (
                    evidence_vector_link_id, vectorization_outcome_id,
                    shadow_attempt_id, anchor_id, root_uuid,
                    learning_generation_id, vector_space_id, partition_id,
                    canonical_query_hash, terminal_class, evaluation_id,
                    quality_label, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000001', ?1,
                    '018f0000-0000-7000-8000-000000000002',
                    '018f0000-0000-7000-8000-000000000003',
                    '018f0000-0000-7000-8000-000000000004',
                    '018f0000-0000-7000-8000-000000000005', ?2, 1, ?3,
                    ?4, ?5, ?6, 0, ?7
                 )",
                    params![
                        "d".repeat(64),
                        "a".repeat(64),
                        "b".repeat(64),
                        terminal_class,
                        evaluation_id,
                        quality_label,
                        "c".repeat(64),
                    ],
                )
            };

        let evaluation_id = "018f0000-0000-7000-8000-000000000006";
        assert!(insert("completed", Some(evaluation_id), None).is_ok());
        connection
            .execute("DELETE FROM evidence_vector_links", [])
            .unwrap();
        assert!(insert("completed", Some(evaluation_id), Some("pass")).is_ok());
        connection
            .execute("DELETE FROM evidence_vector_links", [])
            .unwrap();
        assert!(insert("operational_failure", None, None).is_ok());
        connection
            .execute("DELETE FROM evidence_vector_links", [])
            .unwrap();
        assert!(insert("binary", Some(evaluation_id), Some("pass")).is_err());
        assert!(insert("completed", None, None).is_err());
        assert!(insert("operational_failure", Some(evaluation_id), None).is_err());
        assert!(insert("operational_failure", None, Some("fail")).is_err());
    }

    #[test]
    fn fifth_migration_preserves_a_populated_v4_nondecision_graph() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..4] {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_dependency_operation(&connection);
        let dependency_rows_before = dependency_state_rows(&connection);

        connection.execute_batch(MIGRATIONS[4].sql).unwrap();

        assert_eq!(dependency_state_rows(&connection), dependency_rows_before);
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM anchors", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM shadow_attempts", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('decisions')
                     WHERE name = 'canonical_query_hash'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn fifth_migration_rejects_each_nonempty_reserved_table_atomically() {
        let cases = [
            (
                "decisions",
                "INSERT INTO decisions (
                    decision_id, project_uuid, process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, pool_id,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000021',
                    '018f0000-0000-7000-8000-000000000022',
                    '018f0000-0000-7000-8000-000000000023',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    '018f0000-0000-7000-8000-000000000024', 'pool', 0,
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
                 )",
            ),
            (
                "decision_candidate_summaries",
                "INSERT INTO decision_candidate_summaries (
                    decision_id, candidate_id, rank_ordinal, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000021', 'candidate', 0,
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
                 )",
            ),
            (
                "decision_neighbors",
                "INSERT INTO decision_neighbors (
                    decision_id, neighbor_ordinal, evidence_vector_link_id,
                    canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000021', 0,
                    '018f0000-0000-7000-8000-000000000025',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
                 )",
            ),
            (
                "outcomes",
                "INSERT INTO outcomes (
                    outcome_id, decision_id, project_uuid,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000026',
                    '018f0000-0000-7000-8000-000000000021',
                    '018f0000-0000-7000-8000-000000000022', 0,
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc'
                 )",
            ),
        ];

        for (table, seed) in cases {
            let mut connection = Connection::open_in_memory().unwrap();
            for migration in &MIGRATIONS[..4] {
                connection.execute_batch(migration.sql).unwrap();
            }
            connection
                .execute_batch("PRAGMA foreign_keys = OFF;")
                .unwrap();
            connection.execute_batch(seed).unwrap();
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Exclusive)
                .unwrap();
            assert!(
                transaction.execute_batch(MIGRATIONS[4].sql).is_err(),
                "{table}"
            );
            drop(transaction);

            assert_eq!(
                connection
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                1,
                "{table}"
            );
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_schema
                         WHERE name LIKE 'spec07_decision_%'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0,
                "{table}"
            );
        }
    }

    #[test]
    fn fifth_migration_rolls_back_after_a_late_schema_collision() {
        let mut connection = Connection::open_in_memory().unwrap();
        for migration in &MIGRATIONS[..4] {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch(
                "CREATE INDEX uq_pool_vector_space_mapping_exact
                     ON anchors(opened_at_unix_ms);",
            )
            .unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        assert!(transaction.execute_batch(MIGRATIONS[4].sql).is_err());
        drop(transaction);

        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('decisions')
                     WHERE name = 'canonical_query_hash'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE type = 'index' AND name = 'uq_pool_vector_space_mapping_exact'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn fifth_migration_requires_query_identity_and_restores_exact_foreign_keys() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..5] {
            connection.execute_batch(migration.sql).unwrap();
        }

        let query_column = connection
            .query_row(
                "SELECT type, \"notnull\" FROM pragma_table_info('decisions')
                 WHERE name = 'canonical_query_hash'",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap();
        assert_eq!(query_column, ("TEXT".to_string(), 1));
        assert!(
            foreign_keys(&connection, "decisions")
                .iter()
                .any(|(_, _, table, from, to, _)| {
                    table == "canonical_routing_queries"
                        && from == "canonical_query_hash"
                        && to == "canonical_query_hash"
                })
        );
        assert!(foreign_keys(&connection, "decision_neighbors").iter().any(
            |(_, _, table, from, to, on_delete)| {
                table == "evidence_vector_links"
                    && from == "evidence_vector_link_id"
                    && to == "evidence_vector_link_id"
                    && on_delete == "NO ACTION"
            }
        ));
        assert!(foreign_keys(&connection, "outcomes").iter().any(
            |(_, _, table, from, to, on_delete)| {
                table == "decisions"
                    && from == "decision_id"
                    && to == "decision_id"
                    && on_delete == "CASCADE"
            }
        ));
        assert!(
            foreign_keys(&connection, "decision_retiring_anchors")
                .iter()
                .any(|(_, _, table, from, to, on_delete)| {
                    table == "anchors"
                        && from == "anchor_id"
                        && to == "anchor_id"
                        && on_delete == "CASCADE"
                })
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn fifth_migration_enforces_whole_audit_and_retention_bounds() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..5] {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_spec07_decision_graph(&connection);

        assert!(
            connection
                .execute(
                    "UPDATE decision_candidate_summaries SET coverage_bits = NULL",
                    [],
                )
                .is_err()
        );
        assert!(
            connection
                .execute("UPDATE decisions SET neighbor_count = 4096", [])
                .is_err()
        );

        assert!(
            connection
                .execute(
                    "DELETE FROM evidence_vector_links
                     WHERE evidence_vector_link_id = ?1",
                    ["018f0000-0000-7000-8000-000000000030"],
                )
                .is_err()
        );
        connection
            .execute(
                "DELETE FROM decisions
                 WHERE decision_id = '018f0000-0000-7000-8000-000000000031'",
                [],
            )
            .unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM decision_neighbors", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
        connection
            .execute(
                "DELETE FROM evidence_vector_links
                 WHERE evidence_vector_link_id = ?1",
                ["018f0000-0000-7000-8000-000000000030"],
            )
            .unwrap();

        assert!(
            connection
                .execute(
                    "UPDATE decision_retention_receipts
                     SET deleted_child_count = 4160",
                    [],
                )
                .is_err()
        );
        assert!(
            connection
                .execute(
                    "UPDATE decision_retention_receipts
                     SET verified_aggregate_bytes = 33554433",
                    [],
                )
                .is_err()
        );
        connection
            .execute(
                "INSERT INTO retention_batches (
                    retention_batch_id, project_uuid, process_instance_id,
                    conflict_health_event_id, summary_shape_version,
                    age_expired, count_excess, selected_count,
                    selection_hash, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000035', ?1, ?2,
                    '018f0000-0000-7000-8000-000000000036', 3,
                    0, 0, 0, ?3, 2, ?3
                 )",
                params![PROJECT_UUID, PROCESS_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decision_retention_receipts (
                    retention_batch_id, project_uuid, process_instance_id,
                    decision_age_expired, decision_count_excess, source_forced,
                    new_marker_count, new_marker_hash,
                    deleted_decision_count, deleted_summary_count,
                    deleted_neighbor_count, deleted_child_count,
                    verified_aggregate_bytes, decision_selection_hash,
                    blocked_anchor_count, blocked_anchor_hash,
                    deleted_anchor_count, deleted_anchor_hash, more_cleanup,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000035', ?1, ?2,
                    0, 0, 0, 0, ?3, 0, 0, 0, 0, 0, ?3,
                    0, ?3, 0, ?3, 0, 2, ?3
                 )",
                params![PROJECT_UUID, PROCESS_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn sixth_migration_applies_fresh_complete_active_schema() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }

        let required_tables = [
            "outcome_policy_versions",
            "config_generation_state_events",
            "process_writer_capabilities",
            "controls",
            "control_mutation_receipts",
            "control_history_checkpoints",
            "active_experiments",
            "active_experiment_state_events",
            "active_root_windows",
            "active_root_window_state_events",
            "active_root_signals",
            "active_root_decision_links",
            "active_assignments",
            "active_dispatches",
            "active_dispatch_terminal_events",
            "outcomes",
            "active_experiment_tranches",
            "active_tranche_state_events",
            "active_look_claims",
            "active_look_claim_state_events",
            "active_look_failures",
            "active_look_members",
            "active_outcome_looks",
            "active_outcome_look_audits",
            "active_authorization_state_events",
            "active_neighborhood_state_events",
            "active_decision_facts",
            "active_retirement_markers",
            "active_retirement_receipts",
            "active_retirement_checkpoints",
        ];
        for table in required_tables {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_schema
                         WHERE type = 'table' AND name = ?1",
                        [table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                1,
                "{table}"
            );
        }

        assert_eq!(
            connection
                .query_row(
                    "SELECT \"notnull\" FROM pragma_table_info('active_outcome_look_audits')
                     WHERE name = 'max_monotonic_repair_bits'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert!(foreign_keys(&connection, "outcomes").iter().any(
            |(_, _, table, from, to, on_delete)| {
                table == "decisions"
                    && from == "representative_decision_id"
                    && to == "decision_id"
                    && on_delete == "NO ACTION"
            }
        ));
        assert!(
            !foreign_keys(&connection, "outcomes").iter().any(
                |(_, _, table, _, _, on_delete)| table == "decisions" && on_delete == "CASCADE"
            )
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name LIKE 'spec08_v5_%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert!(MIGRATIONS[5].sql.contains("length(CAST(actor AS BLOB))"));
        assert!(MIGRATIONS[5].sql.contains("writer_protocol = 'active-v6'"));
        assert!(
            MIGRATIONS[5]
                .sql
                .contains("max_monotonic_repair <= 0.00000000001")
        );
        assert_eq!(
            connection
                .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn sixth_migration_preserves_populated_v5_decision_graph_exactly() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..5] {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_spec07_decision_graph(&connection);

        let decisions_before = table_rows(&connection, "decisions");
        let summaries_before = table_rows(&connection, "decision_candidate_summaries");
        let neighbors_before = table_rows(&connection, "decision_neighbors");
        let decision_indexes_before = explicit_indexes(&connection, "decisions");
        let summary_indexes_before = explicit_indexes(&connection, "decision_candidate_summaries");
        let neighbor_indexes_before = explicit_indexes(&connection, "decision_neighbors");
        let decision_foreign_keys_before = foreign_keys(&connection, "decisions");
        let summary_foreign_keys_before = foreign_keys(&connection, "decision_candidate_summaries");
        let neighbor_foreign_keys_before = foreign_keys(&connection, "decision_neighbors");
        let sequences_before = sequence_rows(&connection);

        connection.execute_batch(MIGRATIONS[5].sql).unwrap();

        assert_eq!(table_rows(&connection, "decisions"), decisions_before);
        assert_eq!(
            table_rows(&connection, "decision_candidate_summaries"),
            summaries_before
        );
        assert_eq!(
            table_rows(&connection, "decision_neighbors"),
            neighbors_before
        );
        assert_eq!(
            explicit_indexes(&connection, "decisions"),
            decision_indexes_before
        );
        assert_eq!(
            explicit_indexes(&connection, "decision_candidate_summaries"),
            summary_indexes_before
        );
        assert_eq!(
            explicit_indexes(&connection, "decision_neighbors"),
            neighbor_indexes_before
        );
        assert_eq!(
            foreign_keys(&connection, "decisions"),
            decision_foreign_keys_before
        );
        assert_eq!(
            foreign_keys(&connection, "decision_candidate_summaries"),
            summary_foreign_keys_before
        );
        assert_eq!(
            foreign_keys(&connection, "decision_neighbors"),
            neighbor_foreign_keys_before
        );
        assert_eq!(sequence_rows(&connection), sequences_before);
        assert_eq!(
            connection
                .query_row(
                    "SELECT decision_shape_version, algorithm_version, mode,
                            canonical_payload_hash
                     FROM decisions",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, String>(3)?,
                        ))
                    },
                )
                .unwrap(),
            (1, 1, "recommend".to_string(), PAYLOAD_HASH.to_string())
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn sixth_migration_rejects_reserved_rows_and_authorization_sequence() {
        let cases = [
            (
                "controls",
                "INSERT INTO controls (
                    control_id, control_generation, project_uuid,
                    process_instance_id, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061', 1,
                    '018f0000-0000-7000-8000-000000000062',
                    '018f0000-0000-7000-8000-000000000063', 1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 )",
                1,
            ),
            (
                "outcomes",
                "INSERT INTO outcomes (
                    outcome_id, decision_id, project_uuid,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061',
                    '018f0000-0000-7000-8000-000000000062',
                    '018f0000-0000-7000-8000-000000000063', 1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 )",
                1,
            ),
            (
                "active_experiments",
                "INSERT INTO active_experiments (
                    active_experiment_id, project_uuid, pool_id, candidate_id,
                    partition_hash, learning_generation_id, config_generation_id,
                    cohort_generation_id, outcome_policy_hash,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061',
                    '018f0000-0000-7000-8000-000000000062', 'pool', 'candidate',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                    '018f0000-0000-7000-8000-000000000063',
                    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    '018f0000-0000-7000-8000-000000000064',
                    'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc', 1,
                    'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
                 )",
                1,
            ),
            (
                "active_outcome_looks",
                "INSERT INTO active_outcome_looks (
                    active_outcome_look_id, active_experiment_id, look_ordinal,
                    boundary_root_count, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061',
                    '018f0000-0000-7000-8000-000000000062', 1, 1, 1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 )",
                1,
            ),
            (
                "active_authorization_state_events",
                "INSERT INTO active_authorization_state_events (
                    active_authorization_state_event_id, active_experiment_id,
                    state, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061',
                    '018f0000-0000-7000-8000-000000000062', 'collecting', 1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 )",
                1,
            ),
            (
                "active_authorization_state_events sequence",
                "INSERT INTO active_authorization_state_events (
                    active_authorization_state_event_id, active_experiment_id,
                    state, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000061',
                    '018f0000-0000-7000-8000-000000000062', 'collecting', 1,
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                 );
                 DELETE FROM active_authorization_state_events;",
                0,
            ),
        ];

        for (case, seed, expected_rows) in cases {
            let mut connection = Connection::open_in_memory().unwrap();
            for migration in &MIGRATIONS[..5] {
                connection.execute_batch(migration.sql).unwrap();
            }
            connection
                .execute_batch("PRAGMA foreign_keys = OFF;")
                .unwrap();
            connection.execute_batch(seed).unwrap();

            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Exclusive)
                .unwrap();
            assert!(
                transaction.execute_batch(MIGRATIONS[5].sql).is_err(),
                "{case}"
            );
            drop(transaction);

            let table = case.split(' ').next().unwrap();
            assert_eq!(
                connection
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap(),
                expected_rows,
                "{case}"
            );
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_schema WHERE name LIKE 'spec08_v5_%'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0,
                "{case}"
            );
        }
    }

    #[test]
    fn sixth_migration_rolls_back_after_late_schema_collision() {
        let mut connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..5] {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_spec07_decision_graph(&connection);
        let decisions_before = table_rows(&connection, "decisions");
        let summaries_before = table_rows(&connection, "decision_candidate_summaries");
        let neighbors_before = table_rows(&connection, "decision_neighbors");
        let sequences_before = sequence_rows(&connection);
        connection
            .execute_batch(
                "CREATE TABLE active_retirement_checkpoints (
                    injected_marker INTEGER NOT NULL
                 ) STRICT;",
            )
            .unwrap();

        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        assert!(transaction.execute_batch(MIGRATIONS[5].sql).is_err());
        drop(transaction);

        assert_eq!(table_rows(&connection, "decisions"), decisions_before);
        assert_eq!(
            table_rows(&connection, "decision_candidate_summaries"),
            summaries_before
        );
        assert_eq!(
            table_rows(&connection, "decision_neighbors"),
            neighbors_before
        );
        assert_eq!(sequence_rows(&connection), sequences_before);
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('outcomes')
                     WHERE name = 'decision_id'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name LIKE 'spec08_v5_%'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn seventh_migration_adds_strict_operator_receipt_graph() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }

        for table in [
            "operator_mutation_receipts",
            "operator_mutation_generation_edges",
            "operator_history_entries",
        ] {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sqlite_schema
                         WHERE type = 'table' AND name = ?1 AND sql LIKE '%STRICT%'",
                        [table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                1,
                "{table}"
            );
        }
        assert_eq!(
            explicit_indexes(&connection, "operator_mutation_receipts"),
            vec![
                "idx_operator_mutation_receipts_created".to_string(),
                "idx_operator_mutation_receipts_kind_result".to_string(),
            ]
        );
        assert_eq!(
            explicit_indexes(&connection, "operator_mutation_generation_edges"),
            vec![
                "idx_operator_mutation_edges_cohort".to_string(),
                "idx_operator_mutation_edges_learning".to_string(),
            ]
        );
        assert_eq!(
            explicit_indexes(&connection, "operator_history_entries"),
            vec![
                "idx_operator_history_entries_created".to_string(),
                "idx_operator_history_entries_kind".to_string(),
            ]
        );
        assert!(
            foreign_keys(&connection, "operator_mutation_generation_edges")
                .iter()
                .any(|(_, _, table, from, to, on_delete)| {
                    table == "learning_generations"
                        && from == "learning_generation_id"
                        && to == "learning_generation_id"
                        && on_delete == "NO ACTION"
                })
        );
        assert!(
            foreign_keys(&connection, "operator_mutation_generation_edges")
                .iter()
                .any(|(_, _, table, from, to, on_delete)| {
                    table == "cohort_generations"
                        && from == "cohort_generation_id"
                        && to == "cohort_generation_id"
                        && on_delete == "NO ACTION"
                })
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn seventh_migration_preserves_populated_v6_authority_and_sequences() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        for migration in &MIGRATIONS[..6] {
            connection.execute_batch(migration.sql).unwrap();
        }
        seed_v6_generation_authority(&connection);

        let tables = [
            "project_metadata",
            "config_generations",
            "process_instances",
            "learning_generations",
            "learning_generation_state_events",
            "cohort_generations",
            "cohort_generation_state_events",
            "config_generation_state_events",
            "process_writer_capabilities",
            "controls",
        ];
        let rows_before = tables
            .iter()
            .map(|table| ((*table).to_string(), table_rows(&connection, table)))
            .collect::<Vec<_>>();
        let sequences_before = sequence_rows(&connection);

        connection.execute_batch(MIGRATIONS[6].sql).unwrap();

        for (table, rows) in rows_before {
            assert_eq!(table_rows(&connection, &table), rows, "{table}");
        }
        let sequences_after = sequence_rows(&connection);
        for sequence in sequences_before {
            assert!(sequences_after.contains(&sequence), "{sequence:?}");
        }
        assert_eq!(sequence_value(&connection, "operator_history_entries"), 1);
        assert_eq!(
            connection
                .query_row(
                    "SELECT entry_kind, audit_id, control_mutation_id,
                            operator_mutation_id, created_at_unix_ms
                     FROM operator_history_entries",
                    [],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, i64>(4)?,
                        ))
                    },
                )
                .unwrap(),
            (
                "control".to_string(),
                "018f0000-0000-7000-8000-000000000045".to_string(),
                Some("018f0000-0000-7000-8000-000000000045".to_string()),
                None,
                1,
            )
        );
        assert_eq!(foreign_key_violation_count(&connection), 0);
    }

    #[test]
    fn seventh_migration_rejects_reserved_object_injection_without_changes() {
        let mut connection = Connection::open_in_memory().unwrap();
        for migration in &MIGRATIONS[..6] {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch(
                "CREATE TABLE operator_mutation_receipts (
                    injected_marker INTEGER NOT NULL
                 ) STRICT;",
            )
            .unwrap();
        let sequences_before = sequence_rows(&connection);

        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        assert!(transaction.execute_batch(MIGRATIONS[6].sql).is_err());
        drop(transaction);

        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('operator_mutation_receipts')
                     WHERE name = 'injected_marker'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name = 'operator_mutation_generation_edges'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(sequence_rows(&connection), sequences_before);
    }

    #[test]
    fn seventh_migration_rolls_back_after_late_index_collision() {
        let mut connection = Connection::open_in_memory().unwrap();
        for migration in &MIGRATIONS[..6] {
            connection.execute_batch(migration.sql).unwrap();
        }
        connection
            .execute_batch(
                "CREATE INDEX idx_operator_history_entries_kind
                     ON project_metadata(project_uuid);",
            )
            .unwrap();
        let sequences_before = sequence_rows(&connection);

        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        assert!(transaction.execute_batch(MIGRATIONS[6].sql).is_err());
        drop(transaction);

        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE name IN (
                        'operator_mutation_receipts',
                        'operator_mutation_generation_edges',
                        'idx_operator_mutation_receipts_created',
                        'idx_operator_mutation_receipts_kind_result',
                        'idx_operator_mutation_edges_learning',
                        'idx_operator_mutation_edges_cohort',
                        'operator_history_entries',
                        'idx_operator_history_entries_created'
                     )",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM sqlite_schema
                     WHERE type = 'index' AND name = 'idx_operator_history_entries_kind'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(sequence_rows(&connection), sequences_before);
    }

    fn seed_v6_generation_authority(connection: &Connection) {
        let cohort_id = "018f0000-0000-7000-8000-000000000041";
        connection
            .execute(
                "INSERT INTO project_metadata (
                    singleton_key, project_uuid, project_id, created_at_unix_ms,
                    application_version, canonical_payload_hash
                 ) VALUES (1, ?1, 'migration-test', 0, 'test', ?2)",
                params![PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO config_generations (
                    config_generation_id, project_uuid, canonical_config_json,
                    canonical_payload_hash, created_at_unix_ms, application_version
                 ) VALUES (?1, ?2, '{}', ?1, 0, 'test')",
                params![CONFIG_HASH, PROJECT_UUID],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO process_instances (
                    process_instance_id, project_uuid, config_generation_id,
                    application_version, sqlite_version, started_at_unix_ms,
                    heartbeat_expires_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'test', 'test', 0, 30000, ?4)",
                params![PROCESS_UUID, PROJECT_UUID, CONFIG_HASH, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO learning_generations (
                    learning_generation_id, project_uuid, pool_id, actor, reason,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'pool', 'test', 'test', 0, ?3)",
                params![LEARNING_UUID, PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO learning_generation_state_events (
                    learning_state_event_id, project_uuid, pool_id,
                    learning_generation_id, state, actor, reason,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000042', ?1, 'pool', ?2,
                    'current', 'test', 'test', 0, ?3
                 )",
                params![PROJECT_UUID, LEARNING_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO cohort_generations (
                    cohort_generation_id, project_uuid, cohort_salt,
                    assignment_algorithm, salt_fingerprint_sha256,
                    actor, reason, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, zeroblob(32), 'hmac-sha256-v1', ?3,
                           'test', 'test', 0, ?3)",
                params![cohort_id, PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO cohort_generation_state_events (
                    cohort_state_event_id, project_uuid, cohort_generation_id,
                    state, actor, reason, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000043', ?1, ?2,
                    'current', 'test', 'test', 0, ?3
                 )",
                params![PROJECT_UUID, cohort_id, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO config_generation_state_events (
                    config_generation_state_event_id, config_epoch, project_uuid,
                    config_generation_id, predecessor_event_hash,
                    process_instance_id, created_at_unix_ms, event_hash,
                    canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000044', 1, ?1, ?2, NULL,
                    ?3, 0, ?4, ?4
                 )",
                params![PROJECT_UUID, CONFIG_HASH, PROCESS_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO process_writer_capabilities (
                    process_instance_id, project_uuid, writer_protocol,
                    schema_version, verified_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'active-v6', 6, 0, ?3)",
                params![PROCESS_UUID, PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO controls (
                    control_id, control_generation, originating_history_ordinal,
                    project_uuid, scope_kind, pool_id, force_anchor, paused,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
                 ) VALUES (
                    '00000000-0000-0000-0000-000000000000', 0, 0, ?1,
                    'all', NULL, 0, 0, 'test', 'test', ?2, 0, ?3, ?3
                 )",
                params![PROJECT_UUID, PROCESS_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO control_mutation_receipts (
                    mutation_id, canonical_payload_hash, history_ordinal,
                    predecessor_chain_hash, chain_tip_hash, project_uuid,
                    result, result_control_generation, control_id, control_record_hash,
                    scope_kind, pool_id, operation_kind, requested_value,
                    expected_control_generation, actor, reason,
                    process_instance_id, created_at_unix_ms
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000045', ?1, 1, ?1, ?2, ?3,
                    'conflict', 0, NULL, NULL, 'all', NULL, 'set_paused', 1, 1,
                    'test', 'test', ?4, 1
                 )",
                params![PAYLOAD_HASH, CONFIG_HASH, PROJECT_UUID, PROCESS_UUID],
            )
            .unwrap();
    }

    #[derive(Debug, PartialEq, Eq)]
    struct DependencyStateRow {
        event_seq: i64,
        event_id: String,
        dependency_key_id: String,
        dependency_operation_id: Option<String>,
        anchor_id: Option<String>,
        state: String,
        consecutive_failures: i64,
        cooloff_until_unix_ms: Option<i64>,
        failure_class: Option<String>,
        created_at_unix_ms: i64,
        canonical_payload_hash: String,
    }

    fn dependency_state_rows(connection: &Connection) -> Vec<DependencyStateRow> {
        connection
            .prepare(
                "SELECT event_seq, dependency_state_event_id, dependency_key_id,
                        dependency_operation_id, anchor_id, state,
                        consecutive_failures, cooloff_until_unix_ms,
                        failure_class, created_at_unix_ms, canonical_payload_hash
                 FROM dependency_state_events ORDER BY event_seq",
            )
            .unwrap()
            .query_map([], |row| {
                Ok(DependencyStateRow {
                    event_seq: row.get(0)?,
                    event_id: row.get(1)?,
                    dependency_key_id: row.get(2)?,
                    dependency_operation_id: row.get(3)?,
                    anchor_id: row.get(4)?,
                    state: row.get(5)?,
                    consecutive_failures: row.get(6)?,
                    cooloff_until_unix_ms: row.get(7)?,
                    failure_class: row.get(8)?,
                    created_at_unix_ms: row.get(9)?,
                    canonical_payload_hash: row.get(10)?,
                })
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    #[derive(Debug, PartialEq, Eq)]
    struct RetentionBatchRow {
        retention_batch_id: String,
        project_uuid: String,
        process_instance_id: String,
        age_expired: i64,
        count_excess: i64,
        selected_count: i64,
        selection_lower_bound_unix_ms: Option<i64>,
        selection_upper_bound_unix_ms: Option<i64>,
        selection_hash: String,
        created_at_unix_ms: i64,
        canonical_payload_hash: String,
    }

    fn retention_batch_shape(connection: &Connection) -> RetentionBatchRow {
        connection
            .query_row(
                "SELECT retention_batch_id, project_uuid, process_instance_id,
                        age_expired, count_excess, selected_count,
                        selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                        selection_hash, created_at_unix_ms, canonical_payload_hash
                 FROM retention_batches",
                [],
                |row| {
                    Ok(RetentionBatchRow {
                        retention_batch_id: row.get(0)?,
                        project_uuid: row.get(1)?,
                        process_instance_id: row.get(2)?,
                        age_expired: row.get(3)?,
                        count_excess: row.get(4)?,
                        selected_count: row.get(5)?,
                        selection_lower_bound_unix_ms: row.get(6)?,
                        selection_upper_bound_unix_ms: row.get(7)?,
                        selection_hash: row.get(8)?,
                        created_at_unix_ms: row.get(9)?,
                        canonical_payload_hash: row.get(10)?,
                    })
                },
            )
            .unwrap()
    }

    fn sequence_value(connection: &Connection, table: &str) -> i64 {
        connection
            .query_row(
                "SELECT seq FROM sqlite_sequence WHERE name = ?1",
                [table],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn sequence_rows(connection: &Connection) -> Vec<(String, i64)> {
        connection
            .prepare("SELECT name, seq FROM sqlite_sequence ORDER BY name")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn table_rows(connection: &Connection, table: &str) -> Vec<Vec<Value>> {
        assert!(
            table
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        );
        let mut statement = connection
            .prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))
            .unwrap();
        let column_count = statement.column_count();
        statement
            .query_map([], |row| {
                (0..column_count)
                    .map(|index| row.get(index))
                    .collect::<rusqlite::Result<Vec<Value>>>()
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn explicit_indexes(connection: &Connection, table: &str) -> Vec<String> {
        connection
            .prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'index' AND tbl_name = ?1
                   AND name NOT LIKE 'sqlite_autoindex%'
                 ORDER BY name",
            )
            .unwrap()
            .query_map([table], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn index_columns(connection: &Connection, index: &str) -> Vec<String> {
        connection
            .prepare("SELECT name FROM pragma_index_info(?1) ORDER BY seqno")
            .unwrap()
            .query_map([index], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn foreign_keys(
        connection: &Connection,
        table: &str,
    ) -> Vec<(i64, i64, String, String, String, String)> {
        connection
            .prepare(
                "SELECT id, seq, \"table\", \"from\", \"to\", on_delete
                 FROM pragma_foreign_key_list(?1) ORDER BY id, seq",
            )
            .unwrap()
            .query_map([table], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn seed_spec07_decision_graph(connection: &Connection) {
        const PROFILE_HASH: &str =
            "1111111111111111111111111111111111111111111111111111111111111111";
        const SPACE_HASH: &str = "2222222222222222222222222222222222222222222222222222222222222222";
        const CANONICALIZER_HASH: &str =
            "3333333333333333333333333333333333333333333333333333333333333333";
        const QUERY_HASH: &str = "4444444444444444444444444444444444444444444444444444444444444444";
        const PARTITION_HASH: &str =
            "5555555555555555555555555555555555555555555555555555555555555555";
        const VECTORIZATION_OUTCOME_ID: &str =
            "6666666666666666666666666666666666666666666666666666666666666666";
        const DECISION_ID: &str = "018f0000-0000-7000-8000-000000000031";
        const RETENTION_BATCH_ID: &str = "018f0000-0000-7000-8000-000000000032";

        seed_dependency_operation(connection);
        connection
            .execute(
                "INSERT INTO embedder_profiles (
                    embedder_profile_version_id, profile_id, protocol, endpoint_url,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    timeout_ms, max_in_flight, batch_size, egress_class,
                    canonical_profile_json, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, 'profile', 'openai-embeddings-v1', 'https://example.invalid/v1',
                    ?2, 'embedder', 'revision', 2, 1000, 1, 1, 'remote_https',
                    '{}', 1, ?1
                 )",
                params![PROFILE_HASH, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    metric, normalization, canonical_space_json,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, ?4, '{}', ?5, 'embedder', 'revision', 2,
                    'cosine', 'l2_f32_v1', '{}', 1, ?1
                 )",
                params![
                    SPACE_HASH,
                    PROJECT_UUID,
                    PROFILE_HASH,
                    CANONICALIZER_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO pool_vector_space_mappings (
                    project_uuid, config_generation_id, pool_id, policy_version_id,
                    profile_id, embedder_profile_version_id, canonicalizer_version_id,
                    vector_space_id, canonical_mapping_json,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'pool', ?3, 'profile', ?4, ?5, ?6, '{}', 1, ?7)",
                params![
                    PROJECT_UUID,
                    CONFIG_HASH,
                    POLICY_HASH,
                    PROFILE_HASH,
                    CANONICALIZER_HASH,
                    SPACE_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO routing_partitions (
                    partition_hash, canonical_partition_json, project_uuid, pool_id,
                    tenant_policy_hash, agent_policy_hash, policy_version_id,
                    learning_generation_id, api_family, transport_identity,
                    anchor_model, anchor_revision, candidate_id, candidate_model,
                    candidate_model_revision, decoding_fingerprint, evaluator_version,
                    vector_space_id, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, '{}', ?2, 'pool', ?3, ?3, ?4, ?5,
                    'openai_chat_completions', 'transport', 'anchor', 'revision',
                    'candidate', 'model', 'revision', ?3, ?3, ?6, 1, ?1
                 )",
                params![
                    PARTITION_HASH,
                    PROJECT_UUID,
                    PAYLOAD_HASH,
                    POLICY_HASH,
                    LEARNING_UUID,
                    SPACE_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, '{}', 2, 1, ?1)",
                [QUERY_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vectorization_outcomes (
                    vectorization_outcome_id, shadow_attempt_id, anchor_id,
                    learning_generation_id, vector_space_id, canonical_query_hash,
                    outcome, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'canonicalized', 1, ?7)",
                params![
                    VECTORIZATION_OUTCOME_ID,
                    SHADOW_UUID,
                    ANCHOR_UUID,
                    LEARNING_UUID,
                    SPACE_HASH,
                    QUERY_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO evidence_vector_links (
                    evidence_vector_link_id, vectorization_outcome_id,
                    shadow_attempt_id, anchor_id, root_uuid, learning_generation_id,
                    vector_space_id, partition_id, canonical_query_hash,
                    terminal_class, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    '018f0000-0000-7000-8000-000000000030', ?1, ?2, ?3,
                    '018f0000-0000-7000-8000-000000000012', ?4, ?5,
                    (SELECT partition_id FROM routing_partitions WHERE partition_hash = ?6),
                    ?7, 'operational_failure', 1, ?8
                 )",
                params![
                    VECTORIZATION_OUTCOME_ID,
                    SHADOW_UUID,
                    ANCHOR_UUID,
                    LEARNING_UUID,
                    SPACE_HASH,
                    PARTITION_HASH,
                    QUERY_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decisions (
                    decision_id, decision_shape_version, algorithm_version,
                    project_uuid, process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, pool_id,
                    primary_call_uuid, mode, canonical_query_hash,
                    partition_base_json, partition_base_hash, vector_space_id,
                    candidate_set_hash, candidate_count, recommended_model,
                    recommended_model_revision, served_model, served_model_revision,
                    as_of_unix_ms, decision_latency_ms, final_reason,
                    summary_count, summary_aggregate_hash, neighbor_count,
                    neighbor_aggregate_hash, aggregate_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, 1, 1, ?2, ?3, ?4, ?5, ?6, 'pool',
                    '018f0000-0000-7000-8000-000000000034', 'recommend', ?7,
                    '{}', ?8, ?9, ?10, 1, 'anchor', 'revision', 'anchor', 'revision',
                    1, 1, 'no_candidate_passed', 1, ?11, 1, ?11, 1024, 1, ?11
                 )",
                params![
                    DECISION_ID,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    CONFIG_HASH,
                    POLICY_HASH,
                    LEARNING_UUID,
                    QUERY_HASH,
                    PAYLOAD_HASH,
                    SPACE_HASH,
                    PAYLOAD_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decision_candidate_summaries (
                    decision_id, candidate_id, rank_ordinal, candidate_model,
                    candidate_model_revision, cost_rank, learning_generation_id,
                    vector_space_id, partition_hash, partition_id, decoding_fingerprint,
                    top_k, radius, radius_bits, min_points, min_independent_roots,
                    min_effective_samples, min_effective_samples_bits,
                    min_coverage, min_coverage_bits,
                    time_decay_half_life_seconds, time_decay_half_life_seconds_bits,
                    prior_success, prior_success_bits, prior_failure, prior_failure_bits,
                    familywise_credible_level, familywise_credible_level_bits,
                    candidate_alpha, candidate_alpha_bits,
                    promotion_lower_bound, promotion_lower_bound_bits,
                    returned_neighbor_count, within_radius_count, labeled_point_count,
                    attempted_root_count, labeled_root_count, selected_root_count,
                    coverage, coverage_bits, partition_gate_passed, points_gate_passed,
                    terminal_reason, neighbor_count, neighbor_aggregate_hash,
                    canonical_payload_hash
                 ) VALUES (
                    ?1, 'candidate', 0, 'model', 'revision', 0, ?2, ?3, ?4,
                    (SELECT partition_id FROM routing_partitions WHERE partition_hash = ?4),
                    ?5, 1, 1.0, 0, 1, 1, 1.0, 0, 0.5, 0,
                    60.0, 0, 1.0, 0, 1.0, 0, 0.95, 0, 0.05, 0, 0.8, 0,
                    1, 1, 0, 1, 0, 0, 0.0, 0, 1, 0, 'sparse_points', 1, ?6, ?6
                 )",
                params![
                    DECISION_ID,
                    LEARNING_UUID,
                    SPACE_HASH,
                    PARTITION_HASH,
                    PAYLOAD_HASH,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decision_neighbors (
                    decision_id, neighbor_ordinal, candidate_id,
                    candidate_neighbor_ordinal, evidence_vector_link_id,
                    shadow_attempt_id, anchor_id, learning_generation_id,
                    distance, distance_f32_bits, selected_for_root,
                    root_group_ordinal, exclusion_reason, canonical_payload_hash
                 ) VALUES (
                    ?1, 0, 'candidate', 0,
                    '018f0000-0000-7000-8000-000000000030', ?2, ?3, ?4,
                    0.5, 1056964608, 0, 0, 'ineligible_quality', ?5
                 )",
                params![
                    DECISION_ID,
                    SHADOW_UUID,
                    ANCHOR_UUID,
                    LEARNING_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO retention_batches (
                    retention_batch_id, project_uuid, process_instance_id,
                    conflict_health_event_id, summary_shape_version,
                    age_expired, count_excess, selected_count,
                    selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                    selection_hash, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, '018f0000-0000-7000-8000-000000000033',
                    3, 1, 0, 1, 1, 1, ?4, 1, ?4
                 )",
                params![RETENTION_BATCH_ID, PROJECT_UUID, PROCESS_UUID, PAYLOAD_HASH,],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decision_retiring_anchors (
                    anchor_id, project_uuid, first_retention_batch_id,
                    age_expired, count_excess, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 1, 0, 1, ?4)",
                params![ANCHOR_UUID, PROJECT_UUID, RETENTION_BATCH_ID, PAYLOAD_HASH,],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO decision_retention_receipts (
                    retention_batch_id, project_uuid, process_instance_id,
                    decision_age_expired, decision_count_excess, source_forced,
                    new_marker_count, new_marker_hash,
                    deleted_decision_count, deleted_summary_count,
                    deleted_neighbor_count, deleted_child_count,
                    verified_aggregate_bytes, decision_selection_hash,
                    blocked_anchor_count, blocked_anchor_hash,
                    deleted_anchor_count, deleted_anchor_hash, more_cleanup,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, 0, 0, 1, 1, ?4, 0, 0, 0, 0, 0, ?4,
                    1, ?4, 0, ?4, 1, 1, ?4
                 )",
                params![RETENTION_BATCH_ID, PROJECT_UUID, PROCESS_UUID, PAYLOAD_HASH,],
            )
            .unwrap();
    }

    fn seed_dependency_operation(connection: &Connection) {
        connection
            .execute(
                "INSERT INTO project_metadata (
                    singleton_key, project_uuid, project_id, created_at_unix_ms,
                    application_version, canonical_payload_hash
                 ) VALUES (1, ?1, 'migration-test', 0, 'test', ?2)",
                params![PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO config_generations (
                    config_generation_id, project_uuid, canonical_config_json,
                    canonical_payload_hash, created_at_unix_ms, application_version
                 ) VALUES (?1, ?2, '{}', ?1, 0, 'test')",
                params![CONFIG_HASH, PROJECT_UUID],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO policy_versions (
                    policy_version_id, project_uuid, pool_id,
                    canonical_policy_json, canonical_payload_hash,
                    created_at_unix_ms
                 ) VALUES (?1, ?2, 'pool', '{}', ?1, 0)",
                params![POLICY_HASH, PROJECT_UUID],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO process_instances (
                    process_instance_id, project_uuid, config_generation_id,
                    application_version, sqlite_version, started_at_unix_ms,
                    heartbeat_expires_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'test', 'test', 0, 30, ?4)",
                params![PROCESS_UUID, PROJECT_UUID, CONFIG_HASH, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO learning_generations (
                    learning_generation_id, project_uuid, pool_id, actor,
                    reason, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'pool', 'test', 'test', 0, ?3)",
                params![LEARNING_UUID, PROJECT_UUID, PAYLOAD_HASH],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO anchors (
                    anchor_id, project_uuid, process_instance_id,
                    config_generation_id, policy_version_id,
                    learning_generation_id, pool_id, anchor_call_uuid,
                    root_uuid, owner_uuid, owner_path_json, api_family,
                    transport_identity, anchor_model, anchor_model_revision,
                    replay_capability_fingerprint, decoding_fingerprint,
                    request_projection_json, routing_context_projection_json,
                    candidate_facts_json, requested_progress, opened_at_unix_ms,
                    deadline_at_unix_ms, non_resumable, pending_hash,
                    canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, 'pool',
                    '018f0000-0000-7000-8000-000000000011',
                    '018f0000-0000-7000-8000-000000000012',
                    '018f0000-0000-7000-8000-000000000013',
                    '[]', 'openai_chat_completions', 'transport',
                    'anchor', 'revision', ?7, ?7, '{}', '{}', '[]',
                    1, 0, 1, 1, ?7, ?7
                 )",
                params![
                    ANCHOR_UUID,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    CONFIG_HASH,
                    POLICY_HASH,
                    LEARNING_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO sample_batches (
                    sample_batch_id, anchor_id, project_uuid,
                    process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, pool_id,
                    reserved_candidate_count, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pool', 1, 1, ?8)",
                params![
                    BATCH_UUID,
                    ANCHOR_UUID,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    CONFIG_HASH,
                    POLICY_HASH,
                    LEARNING_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO shadow_attempts (
                    shadow_attempt_id, sample_batch_id, anchor_id, project_uuid,
                    process_instance_id, config_generation_id, policy_version_id,
                    learning_generation_id, pool_id, candidate_id,
                    candidate_model, candidate_model_revision, cost_rank,
                    api_family, transport_identity, anchor_model,
                    anchor_model_revision, decoding_fingerprint,
                    evaluator_version, tenant_policy_hash, agent_policy_hash,
                    eligible, request_projection_json, partition_inputs_json,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pool', 'candidate',
                    'model', 'revision', 0, 'openai_chat_completions',
                    'transport', 'anchor', 'revision', ?9, ?9, ?9, ?9,
                    1, '{}', '{}', 1, ?9
                 )",
                params![
                    SHADOW_UUID,
                    BATCH_UUID,
                    ANCHOR_UUID,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    CONFIG_HASH,
                    POLICY_HASH,
                    LEARNING_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_keys (
                    dependency_key_id, project_uuid, key_kind,
                    canonical_identity_json, canonical_payload_hash
                 ) VALUES (?1, ?2, 'candidate', '{}', ?1)",
                params![DEPENDENCY_KEY_ID, PROJECT_UUID],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO dependency_operations (
                    dependency_operation_id, dependency_key_id, project_uuid,
                    process_instance_id, anchor_id, shadow_attempt_id,
                    base_cooloff_seconds, max_cooloff_seconds,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, 2, 1, ?7)",
                params![
                    OPERATION_UUID,
                    DEPENDENCY_KEY_ID,
                    PROJECT_UUID,
                    PROCESS_UUID,
                    ANCHOR_UUID,
                    SHADOW_UUID,
                    PAYLOAD_HASH,
                ],
            )
            .unwrap();
    }

    fn foreign_key_violation_count(connection: &Connection) -> i64 {
        connection
            .prepare("PRAGMA foreign_key_check")
            .unwrap()
            .query_map([], |_| Ok(()))
            .unwrap()
            .count()
            .try_into()
            .unwrap()
    }
}
