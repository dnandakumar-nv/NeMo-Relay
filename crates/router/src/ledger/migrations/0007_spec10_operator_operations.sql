-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE operator_mutation_receipts (
    operator_ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
    mutation_id TEXT NOT NULL UNIQUE CHECK (length(mutation_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    mutation_kind TEXT NOT NULL CHECK (
        mutation_kind IN ('reset_pool', 'reset_all', 'rotate_cohort')
    ),
    scope_pool_id TEXT CHECK (
        scope_pool_id IS NULL OR
        length(CAST(scope_pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    canonical_request_hash TEXT NOT NULL CHECK (
        length(canonical_request_hash) = 64
        AND canonical_request_hash NOT GLOB '*[^0-9a-f]*'
    ),
    result TEXT NOT NULL CHECK (result IN ('applied', 'conflict')),
    confirm_project_id TEXT NOT NULL CHECK (
        length(CAST(confirm_project_id AS BLOB)) BETWEEN 1 AND 128
    ),
    expected_generation_count INTEGER NOT NULL CHECK (
        expected_generation_count BETWEEN 1 AND 9223372036854775807
    ),
    prior_generation_count INTEGER NOT NULL CHECK (
        prior_generation_count = expected_generation_count
    ),
    resulting_generation_count INTEGER NOT NULL CHECK (
        resulting_generation_count = prior_generation_count
    ),
    superseded_generation_count INTEGER NOT NULL CHECK (
        superseded_generation_count >= 0
    ),
    actor TEXT NOT NULL CHECK (
        length(CAST(actor AS BLOB)) BETWEEN 1 AND 128 AND trim(actor) <> ''
    ),
    reason TEXT NOT NULL CHECK (
        length(CAST(reason AS BLOB)) BETWEEN 1 AND 512 AND trim(reason) <> ''
    ),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    record_hash TEXT NOT NULL UNIQUE CHECK (
        length(record_hash) = 64 AND record_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = record_hash
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    CHECK (
        (mutation_kind = 'reset_pool' AND scope_pool_id IS NOT NULL)
        OR (mutation_kind IN ('reset_all', 'rotate_cohort') AND scope_pool_id IS NULL)
    ),
    CHECK (
        (result = 'applied'
            AND superseded_generation_count = prior_generation_count)
        OR (result = 'conflict' AND superseded_generation_count = 0)
    ),
    UNIQUE (project_uuid, mutation_id),
    UNIQUE (project_uuid, operator_ordinal),
    UNIQUE (project_uuid, record_hash)
) STRICT;

CREATE INDEX idx_operator_mutation_receipts_created
    ON operator_mutation_receipts(
        project_uuid, created_at_unix_ms DESC, operator_ordinal DESC, mutation_id
    );

CREATE INDEX idx_operator_mutation_receipts_kind_result
    ON operator_mutation_receipts(
        project_uuid, mutation_kind, result, operator_ordinal DESC
    );

CREATE TABLE operator_mutation_generation_edges (
    mutation_id TEXT NOT NULL,
    project_uuid TEXT NOT NULL,
    edge_role TEXT NOT NULL CHECK (
        edge_role IN ('expected', 'prior', 'resulting', 'superseded')
    ),
    edge_ordinal INTEGER NOT NULL CHECK (edge_ordinal >= 0),
    scope_key TEXT NOT NULL CHECK (
        length(CAST(scope_key AS BLOB)) BETWEEN 1 AND 128
    ),
    generation_kind TEXT NOT NULL CHECK (
        generation_kind IN ('learning', 'cohort')
    ),
    pool_id TEXT CHECK (
        pool_id IS NULL OR length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    learning_generation_id TEXT CHECK (
        learning_generation_id IS NULL OR length(learning_generation_id) = 36
    ),
    cohort_generation_id TEXT CHECK (
        cohort_generation_id IS NULL OR length(cohort_generation_id) = 36
    ),
    FOREIGN KEY (project_uuid, mutation_id)
        REFERENCES operator_mutation_receipts(project_uuid, mutation_id),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(
            project_uuid, pool_id, learning_generation_id
        ),
    FOREIGN KEY (project_uuid, cohort_generation_id)
        REFERENCES cohort_generations(project_uuid, cohort_generation_id),
    CHECK (
        (generation_kind = 'learning'
            AND pool_id IS NOT NULL
            AND scope_key = pool_id
            AND learning_generation_id IS NOT NULL
            AND cohort_generation_id IS NULL)
        OR (generation_kind = 'cohort'
            AND pool_id IS NULL
            AND scope_key = 'cohort'
            AND learning_generation_id IS NULL
            AND cohort_generation_id IS NOT NULL)
    ),
    PRIMARY KEY (mutation_id, edge_role, edge_ordinal),
    UNIQUE (mutation_id, edge_role, scope_key)
) STRICT;

CREATE INDEX idx_operator_mutation_edges_learning
    ON operator_mutation_generation_edges(
        project_uuid, pool_id, learning_generation_id, mutation_id, edge_role
    ) WHERE learning_generation_id IS NOT NULL;

CREATE INDEX idx_operator_mutation_edges_cohort
    ON operator_mutation_generation_edges(
        project_uuid, cohort_generation_id, mutation_id, edge_role
    ) WHERE cohort_generation_id IS NOT NULL;

CREATE TABLE operator_history_entries (
    history_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    audit_id TEXT NOT NULL CHECK (length(audit_id) = 36),
    entry_kind TEXT NOT NULL CHECK (entry_kind IN ('control', 'operator')),
    control_mutation_id TEXT,
    operator_mutation_id TEXT,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    FOREIGN KEY (control_mutation_id)
        REFERENCES control_mutation_receipts(mutation_id) ON DELETE CASCADE,
    FOREIGN KEY (project_uuid, operator_mutation_id)
        REFERENCES operator_mutation_receipts(project_uuid, mutation_id)
        ON DELETE CASCADE,
    CHECK (
        (entry_kind = 'control'
            AND control_mutation_id = audit_id
            AND operator_mutation_id IS NULL)
        OR (entry_kind = 'operator'
            AND control_mutation_id IS NULL
            AND operator_mutation_id = audit_id)
    ),
    UNIQUE (project_uuid, audit_id),
    UNIQUE (project_uuid, control_mutation_id),
    UNIQUE (project_uuid, operator_mutation_id)
) STRICT;

CREATE INDEX idx_operator_history_entries_created
    ON operator_history_entries(
        project_uuid, created_at_unix_ms DESC, audit_id DESC, history_sequence
    );

CREATE INDEX idx_operator_history_entries_kind
    ON operator_history_entries(project_uuid, entry_kind, history_sequence DESC);

INSERT INTO operator_history_entries (
    project_uuid, audit_id, entry_kind, control_mutation_id,
    operator_mutation_id, created_at_unix_ms
)
SELECT
    project_uuid, mutation_id, 'control', mutation_id, NULL, created_at_unix_ms
FROM control_mutation_receipts
ORDER BY created_at_unix_ms, mutation_id;
