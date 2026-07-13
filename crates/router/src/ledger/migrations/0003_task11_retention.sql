-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

ALTER TABLE dependency_state_events RENAME TO dependency_state_events_v2;

CREATE TABLE dependency_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    dependency_state_event_id TEXT NOT NULL UNIQUE CHECK (length(dependency_state_event_id) = 36),
    dependency_key_id TEXT NOT NULL REFERENCES dependency_keys(dependency_key_id),
    dependency_operation_id TEXT,
    anchor_id TEXT REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (
        state IN ('admitted', 'skipped_cooloff', 'success', 'failure', 'orphaned_in_flight')
    ),
    consecutive_failures INTEGER NOT NULL CHECK (consecutive_failures >= 0),
    cooloff_until_unix_ms INTEGER CHECK (
        cooloff_until_unix_ms IS NULL OR cooloff_until_unix_ms >= 0
    ),
    failure_class TEXT CHECK (
        failure_class IS NULL OR (
            length(failure_class) BETWEEN 1 AND 128
            AND failure_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (dependency_operation_id, dependency_key_id, anchor_id)
        REFERENCES dependency_operations(
            dependency_operation_id, dependency_key_id, anchor_id
        ) ON DELETE CASCADE,
    CHECK (
        (dependency_operation_id IS NULL AND anchor_id IS NULL
            AND state IN (
                'skipped_cooloff', 'success', 'failure', 'orphaned_in_flight'
            ))
        OR (dependency_operation_id IS NOT NULL AND anchor_id IS NOT NULL)
    ),
    CHECK (
        (state = 'success' AND consecutive_failures = 0
            AND cooloff_until_unix_ms IS NULL AND failure_class IS NULL)
        OR (state = 'failure' AND consecutive_failures > 0
            AND cooloff_until_unix_ms IS NOT NULL AND failure_class IS NOT NULL)
        OR (state = 'skipped_cooloff' AND consecutive_failures > 0
            AND cooloff_until_unix_ms IS NOT NULL AND failure_class IS NULL)
        OR (state IN ('admitted', 'orphaned_in_flight') AND failure_class IS NULL AND (
            (consecutive_failures = 0 AND cooloff_until_unix_ms IS NULL)
            OR (consecutive_failures > 0 AND cooloff_until_unix_ms IS NOT NULL)
        ))
    ),
    UNIQUE (dependency_operation_id, state)
) STRICT;

INSERT INTO dependency_state_events (
    event_seq, dependency_state_event_id, dependency_key_id,
    dependency_operation_id, anchor_id, state, consecutive_failures,
    cooloff_until_unix_ms, failure_class, created_at_unix_ms,
    canonical_payload_hash
)
SELECT
    event_seq, dependency_state_event_id, dependency_key_id,
    dependency_operation_id, anchor_id, state, consecutive_failures,
    cooloff_until_unix_ms, failure_class, created_at_unix_ms,
    canonical_payload_hash
FROM dependency_state_events_v2
ORDER BY event_seq;

INSERT INTO sqlite_sequence (name, seq)
SELECT 'dependency_state_events', seq
FROM sqlite_sequence
WHERE name = 'dependency_state_events_v2'
  AND NOT EXISTS (
      SELECT 1 FROM sqlite_sequence WHERE name = 'dependency_state_events'
  );

UPDATE sqlite_sequence
SET seq = max(
    seq,
    coalesce(
        (SELECT seq FROM sqlite_sequence WHERE name = 'dependency_state_events_v2'),
        seq
    )
)
WHERE name = 'dependency_state_events';

DROP TABLE dependency_state_events_v2;

CREATE UNIQUE INDEX uq_dependency_operation_terminal_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL AND state <> 'admitted';

CREATE UNIQUE INDEX uq_dependency_operation_claim_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL
      AND state IN ('admitted', 'skipped_cooloff');

CREATE INDEX idx_dependency_state_latest
    ON dependency_state_events(dependency_key_id, event_seq DESC);

ALTER TABLE retention_batches RENAME TO retention_batches_v2;

CREATE TABLE retention_batches (
    retention_batch_id TEXT PRIMARY KEY CHECK (length(retention_batch_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT NOT NULL,
    conflict_health_event_id TEXT CHECK (
        conflict_health_event_id IS NULL OR length(conflict_health_event_id) = 36
    ),
    summary_shape_version INTEGER NOT NULL DEFAULT 3 CHECK (summary_shape_version IN (2, 3)),
    age_expired INTEGER NOT NULL CHECK (age_expired IN (0, 1)),
    count_excess INTEGER NOT NULL CHECK (count_excess IN (0, 1)),
    selected_count INTEGER NOT NULL CHECK (selected_count BETWEEN 0 AND 1000),
    selection_lower_bound_unix_ms INTEGER CHECK (
        selection_lower_bound_unix_ms IS NULL OR selection_lower_bound_unix_ms >= 0
    ),
    selection_upper_bound_unix_ms INTEGER CHECK (
        selection_upper_bound_unix_ms IS NULL OR selection_upper_bound_unix_ms >= 0
    ),
    selection_hash TEXT NOT NULL CHECK (
        length(selection_hash) = 64 AND selection_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    CHECK (
        (summary_shape_version = 2
            AND conflict_health_event_id IS NULL
            AND (age_expired = 1 OR count_excess = 1)
            AND (
                selection_lower_bound_unix_ms IS NULL
                OR selection_upper_bound_unix_ms IS NULL
                OR selection_lower_bound_unix_ms <= selection_upper_bound_unix_ms
            ))
        OR (summary_shape_version = 3 AND (
            conflict_health_event_id IS NOT NULL
            AND conflict_health_event_id <> retention_batch_id
            AND ((selected_count = 0
                AND age_expired = 0
                AND count_excess = 0
                AND selection_lower_bound_unix_ms IS NULL
                AND selection_upper_bound_unix_ms IS NULL)
            OR (selected_count > 0
                AND (age_expired = 1 OR count_excess = 1)
                AND selection_lower_bound_unix_ms IS NOT NULL
                AND selection_upper_bound_unix_ms IS NOT NULL
                AND selection_lower_bound_unix_ms <= selection_upper_bound_unix_ms))
        ))
    )
) STRICT;

INSERT INTO retention_batches (
    retention_batch_id, project_uuid, process_instance_id, conflict_health_event_id,
    summary_shape_version, age_expired, count_excess, selected_count,
    selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
    selection_hash, created_at_unix_ms, canonical_payload_hash
)
SELECT
    retention_batch_id, project_uuid, process_instance_id, NULL,
    2, age_expired, count_excess, selected_count,
    selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
    selection_hash, created_at_unix_ms, canonical_payload_hash
FROM retention_batches_v2;

DROP TABLE retention_batches_v2;

CREATE INDEX idx_dependency_operation_anchor
    ON dependency_operations(anchor_id);
CREATE INDEX idx_dependency_state_anchor
    ON dependency_state_events(anchor_id)
    WHERE anchor_id IS NOT NULL;
CREATE INDEX idx_health_event_anchor
    ON health_events(anchor_id)
    WHERE anchor_id IS NOT NULL;
CREATE INDEX idx_evidence_vector_link_anchor
    ON evidence_vector_links(anchor_id);
CREATE INDEX idx_evidence_vector_link_state_embedding
    ON evidence_vector_link_state_events(embedding_id)
    WHERE embedding_id IS NOT NULL;
CREATE INDEX idx_decision_neighbors_evidence_vector_link
    ON decision_neighbors(evidence_vector_link_id);
CREATE INDEX idx_shadow_attempt_sample_batch
    ON shadow_attempts(sample_batch_id);
