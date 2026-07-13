-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

ALTER TABLE embedding_job_state_events
    ADD COLUMN lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36);
ALTER TABLE embedding_job_state_events
    ADD COLUMN lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    );
ALTER TABLE embedding_job_state_events
    ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0);
ALTER TABLE embedding_job_state_events
    ADD COLUMN next_eligible_at_unix_ms INTEGER NOT NULL DEFAULT 0 CHECK (
        next_eligible_at_unix_ms >= 0
    );

ALTER TABLE dependency_state_events RENAME TO dependency_state_events_v1;

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
            AND state IN ('success', 'failure'))
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
FROM dependency_state_events_v1
ORDER BY event_seq;

DROP TABLE dependency_state_events_v1;

CREATE UNIQUE INDEX uq_dependency_operation_terminal_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL AND state <> 'admitted';

CREATE UNIQUE INDEX uq_dependency_operation_claim_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL
      AND state IN ('admitted', 'skipped_cooloff');

CREATE INDEX idx_dependency_state_latest
    ON dependency_state_events(dependency_key_id, event_seq DESC);
