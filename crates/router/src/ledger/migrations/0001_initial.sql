-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

PRAGMA application_id = 1313690194;

CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL UNIQUE CHECK (name <> ''),
    checksum_sha256 TEXT NOT NULL CHECK (
        length(checksum_sha256) = 64
        AND checksum_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    applied_at_unix_ms INTEGER NOT NULL CHECK (applied_at_unix_ms >= 0),
    application_version TEXT NOT NULL CHECK (application_version <> ''),
    sqlite_version TEXT NOT NULL CHECK (sqlite_version <> '')
) STRICT;

CREATE TABLE project_metadata (
    singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
    project_uuid TEXT NOT NULL UNIQUE CHECK (length(project_uuid) = 36),
    project_id TEXT NOT NULL UNIQUE CHECK (length(project_id) BETWEEN 1 AND 128),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    application_version TEXT NOT NULL CHECK (application_version <> ''),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;

CREATE TABLE config_generations (
    config_generation_id TEXT PRIMARY KEY CHECK (
        length(config_generation_id) = 64
        AND config_generation_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    canonical_config_json TEXT NOT NULL CHECK (json_valid(canonical_config_json)),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = config_generation_id
        AND length(canonical_payload_hash) = 64
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    application_version TEXT NOT NULL CHECK (application_version <> ''),
    UNIQUE (project_uuid, config_generation_id)
) STRICT;

CREATE TABLE policy_versions (
    policy_version_id TEXT NOT NULL CHECK (
        length(policy_version_id) = 64
        AND policy_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    canonical_policy_json TEXT NOT NULL CHECK (json_valid(canonical_policy_json)),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = policy_version_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    PRIMARY KEY (project_uuid, pool_id, policy_version_id)
) STRICT;

CREATE TABLE process_instances (
    process_instance_id TEXT PRIMARY KEY CHECK (length(process_instance_id) = 36),
    project_uuid TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    application_version TEXT NOT NULL CHECK (application_version <> ''),
    sqlite_version TEXT NOT NULL CHECK (sqlite_version <> ''),
    started_at_unix_ms INTEGER NOT NULL CHECK (started_at_unix_ms >= 0),
    heartbeat_expires_at_unix_ms INTEGER NOT NULL CHECK (
        heartbeat_expires_at_unix_ms >= started_at_unix_ms
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    UNIQUE (project_uuid, process_instance_id),
    UNIQUE (project_uuid, process_instance_id, config_generation_id)
) STRICT;

CREATE TABLE process_instance_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    process_state_event_id TEXT NOT NULL UNIQUE CHECK (length(process_state_event_id) = 36),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (state IN ('started', 'stopped', 'reconciled')),
    subject_process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state = 'reconciled' AND subject_process_instance_id <> process_instance_id)
        OR (state IN ('started', 'stopped') AND subject_process_instance_id = process_instance_id)
    ),
    UNIQUE (subject_process_instance_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_process_terminal_state
    ON process_instance_state_events(subject_process_instance_id)
    WHERE state IN ('stopped', 'reconciled');
CREATE INDEX idx_process_state_latest
    ON process_instance_state_events(subject_process_instance_id, event_seq DESC);
CREATE INDEX idx_process_heartbeat
    ON process_instances(heartbeat_expires_at_unix_ms, process_instance_id);

CREATE TABLE cohort_generations (
    cohort_generation_id TEXT PRIMARY KEY CHECK (length(cohort_generation_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    cohort_salt BLOB NOT NULL CHECK (length(cohort_salt) = 32),
    assignment_algorithm TEXT NOT NULL CHECK (assignment_algorithm = 'hmac-sha256-v1'),
    salt_fingerprint_sha256 TEXT NOT NULL CHECK (
        length(salt_fingerprint_sha256) = 64
        AND salt_fingerprint_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    actor TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 128 AND trim(actor) <> ''),
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 512 AND trim(reason) <> ''),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (project_uuid, cohort_generation_id),
    UNIQUE (project_uuid, salt_fingerprint_sha256)
) STRICT;

CREATE TABLE cohort_generation_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    cohort_state_event_id TEXT NOT NULL UNIQUE CHECK (length(cohort_state_event_id) = 36),
    project_uuid TEXT NOT NULL,
    cohort_generation_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state = 'current'),
    actor TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 128 AND trim(actor) <> ''),
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 512 AND trim(reason) <> ''),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, cohort_generation_id)
        REFERENCES cohort_generations(project_uuid, cohort_generation_id),
    UNIQUE (project_uuid, cohort_generation_id)
) STRICT;

CREATE INDEX idx_cohort_state_latest
    ON cohort_generation_state_events(project_uuid, event_seq DESC);

CREATE TABLE learning_generations (
    learning_generation_id TEXT PRIMARY KEY CHECK (length(learning_generation_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    actor TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 128 AND trim(actor) <> ''),
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 512 AND trim(reason) <> ''),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (project_uuid, pool_id, learning_generation_id)
) STRICT;

CREATE TABLE learning_generation_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    learning_state_event_id TEXT NOT NULL UNIQUE CHECK (length(learning_state_event_id) = 36),
    project_uuid TEXT NOT NULL,
    pool_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state = 'current'),
    actor TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 128 AND trim(actor) <> ''),
    reason TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 512 AND trim(reason) <> ''),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    UNIQUE (project_uuid, pool_id, learning_generation_id)
) STRICT;

CREATE INDEX idx_learning_state_latest
    ON learning_generation_state_events(project_uuid, pool_id, event_seq DESC);

CREATE TABLE anchors (
    anchor_id TEXT PRIMARY KEY CHECK (length(anchor_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    anchor_call_uuid TEXT NOT NULL CHECK (length(anchor_call_uuid) = 36),
    root_uuid TEXT NOT NULL CHECK (length(root_uuid) = 36),
    owner_uuid TEXT NOT NULL CHECK (length(owner_uuid) = 36),
    owner_path_json TEXT NOT NULL CHECK (json_valid(owner_path_json)),
    api_family TEXT NOT NULL CHECK (
        api_family IN ('openai_chat_completions', 'openai_responses', 'anthropic_messages')
    ),
    transport_identity TEXT NOT NULL CHECK (length(transport_identity) BETWEEN 1 AND 256),
    anchor_model TEXT NOT NULL CHECK (length(anchor_model) BETWEEN 1 AND 512),
    anchor_model_revision TEXT NOT NULL CHECK (length(anchor_model_revision) BETWEEN 1 AND 128),
    replay_capability_fingerprint TEXT NOT NULL CHECK (
        length(replay_capability_fingerprint) = 64
        AND replay_capability_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    decoding_fingerprint TEXT NOT NULL CHECK (
        length(decoding_fingerprint) = 64
        AND decoding_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    request_projection_json TEXT NOT NULL CHECK (json_valid(request_projection_json)),
    routing_context_projection_json TEXT NOT NULL CHECK (json_valid(routing_context_projection_json)),
    candidate_facts_json TEXT NOT NULL CHECK (json_valid(candidate_facts_json)),
    requested_progress INTEGER NOT NULL CHECK (requested_progress > 0),
    opened_at_unix_ms INTEGER NOT NULL CHECK (opened_at_unix_ms >= 0),
    deadline_at_unix_ms INTEGER NOT NULL CHECK (deadline_at_unix_ms >= opened_at_unix_ms),
    non_resumable INTEGER NOT NULL CHECK (non_resumable IN (0, 1)),
    pending_hash TEXT NOT NULL CHECK (
        length(pending_hash) = 64
        AND pending_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = pending_hash),
    FOREIGN KEY (project_uuid, process_instance_id, config_generation_id)
        REFERENCES process_instances(project_uuid, process_instance_id, config_generation_id),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    UNIQUE (config_generation_id, learning_generation_id, pool_id, anchor_call_uuid),
    UNIQUE (anchor_id, root_uuid),
    UNIQUE (
        anchor_id, project_uuid, process_instance_id, config_generation_id,
        pool_id, policy_version_id, learning_generation_id
    )
) STRICT;

CREATE TABLE anchor_results (
    anchor_id TEXT PRIMARY KEY REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    normalized_response_json TEXT NOT NULL CHECK (json_valid(normalized_response_json)),
    semantic_response_fingerprint TEXT NOT NULL CHECK (
        length(semantic_response_fingerprint) = 64
        AND semantic_response_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;

CREATE TABLE anchor_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    anchor_state_event_id TEXT NOT NULL UNIQUE CHECK (length(anchor_state_event_id) = 36),
    anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    dead_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN ('pending', 'closed', 'rejected', 'not_scheduled_queue_full', 'orphaned_non_resumable')
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state = 'orphaned_non_resumable' AND dead_process_instance_id IS NOT NULL)
        OR (state <> 'orphaned_non_resumable' AND dead_process_instance_id IS NULL)
    ),
    UNIQUE (anchor_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_anchor_terminal_state
    ON anchor_state_events(anchor_id)
    WHERE state <> 'pending';
CREATE INDEX idx_anchor_state_latest
    ON anchor_state_events(anchor_id, event_seq DESC);

CREATE TABLE anchor_windows (
    anchor_id TEXT PRIMARY KEY REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    requested_progress INTEGER NOT NULL CHECK (requested_progress > 0),
    observed_progress INTEGER NOT NULL CHECK (observed_progress >= 0),
    terminal_kind TEXT NOT NULL CHECK (terminal_kind IN ('closed', 'rejected')),
    trigger TEXT CHECK (
        trigger IS NULL OR trigger IN (
            'progress_reached', 'owner_terminated', 'deadline_elapsed', 'shutdown'
        )
    ),
    rejection_reason TEXT CHECK (
        rejection_reason IS NULL OR rejection_reason IN (
            'event_loss', 'overflow', 'contradictory_ownership',
            'canceled_before_anchor_end', 'rejected_delivery_barrier'
        )
    ),
    is_partial INTEGER NOT NULL CHECK (is_partial IN (0, 1)),
    promotion_eligible INTEGER NOT NULL CHECK (promotion_eligible IN (0, 1)),
    closed_at_unix_ms INTEGER NOT NULL CHECK (closed_at_unix_ms >= 0),
    diagnostics_json TEXT NOT NULL CHECK (json_valid(diagnostics_json)),
    terminal_hash TEXT NOT NULL CHECK (
        length(terminal_hash) = 64
        AND terminal_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (terminal_kind = 'closed' AND trigger IS NOT NULL AND rejection_reason IS NULL)
        OR (terminal_kind = 'rejected' AND trigger IS NULL AND rejection_reason IS NOT NULL)
    ),
    CHECK (is_partial = 0 OR promotion_eligible = 0)
) STRICT;

CREATE TABLE trajectory_events (
    anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    ingest_seq INTEGER NOT NULL CHECK (ingest_seq >= 0),
    event_uuid TEXT NOT NULL CHECK (length(event_uuid) = 36),
    parent_uuid TEXT CHECK (parent_uuid IS NULL OR length(parent_uuid) = 36),
    kind TEXT NOT NULL CHECK (kind IN ('scope', 'mark')),
    phase TEXT,
    category TEXT,
    call_role TEXT CHECK (call_role IS NULL OR call_role IN ('primary', 'shadow', 'judge')),
    name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 512),
    event_time_unix_ms INTEGER NOT NULL CHECK (event_time_unix_ms >= 0),
    schema_id TEXT NOT NULL CHECK (length(schema_id) BETWEEN 1 AND 256),
    sanitized_payload_json TEXT NOT NULL CHECK (json_valid(sanitized_payload_json)),
    canonical_size_bytes INTEGER NOT NULL CHECK (canonical_size_bytes >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (anchor_id, ingest_seq)
) STRICT;

CREATE TABLE sample_batches (
    sample_batch_id TEXT PRIMARY KEY CHECK (length(sample_batch_id) = 36),
    anchor_id TEXT NOT NULL UNIQUE REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    reserved_candidate_count INTEGER NOT NULL CHECK (reserved_candidate_count > 0),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (
        anchor_id, project_uuid, process_instance_id, config_generation_id,
        pool_id, policy_version_id, learning_generation_id
    ) REFERENCES anchors(
        anchor_id, project_uuid, process_instance_id, config_generation_id,
        pool_id, policy_version_id, learning_generation_id
    ),
    UNIQUE (
        sample_batch_id, anchor_id, project_uuid, process_instance_id,
        config_generation_id, pool_id, policy_version_id, learning_generation_id
    )
) STRICT;

CREATE TABLE sample_batch_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_batch_state_event_id TEXT NOT NULL UNIQUE CHECK (length(sample_batch_state_event_id) = 36),
    sample_batch_id TEXT NOT NULL REFERENCES sample_batches(sample_batch_id) ON DELETE CASCADE,
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    dead_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN ('open', 'closed', 'orphaned_before_schedule', 'orphaned_in_flight', 'canceled_shutdown')
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state IN ('orphaned_before_schedule', 'orphaned_in_flight') AND dead_process_instance_id IS NOT NULL)
        OR (state NOT IN ('orphaned_before_schedule', 'orphaned_in_flight') AND dead_process_instance_id IS NULL)
    ),
    UNIQUE (sample_batch_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_sample_batch_terminal_state
    ON sample_batch_state_events(sample_batch_id)
    WHERE state <> 'open';
CREATE INDEX idx_sample_batch_state_latest
    ON sample_batch_state_events(sample_batch_id, event_seq DESC);

CREATE TABLE shadow_attempts (
    shadow_attempt_id TEXT PRIMARY KEY CHECK (length(shadow_attempt_id) = 36),
    sample_batch_id TEXT NOT NULL REFERENCES sample_batches(sample_batch_id) ON DELETE CASCADE,
    anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
    candidate_model TEXT NOT NULL CHECK (length(candidate_model) BETWEEN 1 AND 512),
    candidate_model_revision TEXT NOT NULL CHECK (length(candidate_model_revision) BETWEEN 1 AND 128),
    cost_rank INTEGER NOT NULL CHECK (cost_rank >= 0),
    api_family TEXT NOT NULL CHECK (
        api_family IN ('openai_chat_completions', 'openai_responses', 'anthropic_messages')
    ),
    transport_identity TEXT NOT NULL CHECK (length(transport_identity) BETWEEN 1 AND 256),
    anchor_model TEXT NOT NULL CHECK (length(anchor_model) BETWEEN 1 AND 512),
    anchor_model_revision TEXT NOT NULL CHECK (length(anchor_model_revision) BETWEEN 1 AND 128),
    decoding_fingerprint TEXT NOT NULL CHECK (
        length(decoding_fingerprint) = 64
        AND decoding_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    evaluator_version TEXT NOT NULL CHECK (
        length(evaluator_version) = 64
        AND evaluator_version NOT GLOB '*[^0-9a-f]*'
    ),
    tenant_policy_hash TEXT NOT NULL CHECK (
        length(tenant_policy_hash) = 64
        AND tenant_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    agent_policy_hash TEXT NOT NULL CHECK (
        length(agent_policy_hash) = 64
        AND agent_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    request_projection_json TEXT NOT NULL CHECK (json_valid(request_projection_json)),
    partition_inputs_json TEXT NOT NULL CHECK (json_valid(partition_inputs_json)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (
        sample_batch_id, anchor_id, project_uuid, process_instance_id,
        config_generation_id, pool_id, policy_version_id, learning_generation_id
    ) REFERENCES sample_batches(
        sample_batch_id, anchor_id, project_uuid, process_instance_id,
        config_generation_id, pool_id, policy_version_id, learning_generation_id
    ),
    UNIQUE (anchor_id, candidate_id, candidate_model_revision),
    UNIQUE (shadow_attempt_id, anchor_id),
    UNIQUE (shadow_attempt_id, anchor_id, learning_generation_id),
    UNIQUE (shadow_attempt_id, evaluator_version),
    UNIQUE (shadow_attempt_id, learning_generation_id, evaluator_version),
    UNIQUE (shadow_attempt_id, process_instance_id, learning_generation_id, evaluator_version),
    UNIQUE (shadow_attempt_id, anchor_id, process_instance_id),
    UNIQUE (shadow_attempt_id, anchor_id, project_uuid, process_instance_id),
    UNIQUE (
        shadow_attempt_id, anchor_id, process_instance_id,
        learning_generation_id, evaluator_version
    )
) STRICT;

CREATE TABLE shadow_attempt_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    shadow_attempt_state_event_id TEXT NOT NULL UNIQUE CHECK (length(shadow_attempt_state_event_id) = 36),
    shadow_attempt_id TEXT NOT NULL REFERENCES shadow_attempts(shadow_attempt_id) ON DELETE CASCADE,
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    dead_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'reserved', 'started', 'completed', 'deterministic_failure', 'operational_failure',
            'skipped_cooloff', 'orphaned_before_schedule', 'orphaned_in_flight', 'canceled_shutdown'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state IN ('orphaned_before_schedule', 'orphaned_in_flight') AND dead_process_instance_id IS NOT NULL)
        OR (state NOT IN ('orphaned_before_schedule', 'orphaned_in_flight') AND dead_process_instance_id IS NULL)
    ),
    UNIQUE (shadow_attempt_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_shadow_attempt_terminal_state
    ON shadow_attempt_state_events(shadow_attempt_id)
    WHERE state NOT IN ('reserved', 'started');
CREATE INDEX idx_shadow_attempt_state_latest
    ON shadow_attempt_state_events(shadow_attempt_id, event_seq DESC);
CREATE INDEX idx_shadow_attempt_process
    ON shadow_attempts(process_instance_id, shadow_attempt_id);

CREATE TABLE evaluations (
    evaluation_id TEXT PRIMARY KEY CHECK (length(evaluation_id) = 36),
    shadow_attempt_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL CHECK (
        length(evaluator_version) = 64
        AND evaluator_version NOT GLOB '*[^0-9a-f]*'
    ),
    source TEXT NOT NULL CHECK (source IN ('deterministic_validator', 'judge')),
    judge_model TEXT CHECK (judge_model IS NULL OR length(judge_model) BETWEEN 1 AND 512),
    judge_model_revision TEXT CHECK (
        judge_model_revision IS NULL OR length(judge_model_revision) BETWEEN 1 AND 128
    ),
    prompt_version TEXT CHECK (prompt_version IS NULL OR length(prompt_version) BETWEEN 1 AND 128),
    prompt_sha256 TEXT CHECK (
        prompt_sha256 IS NULL OR (
            length(prompt_sha256) = 64 AND prompt_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    rubric_version TEXT CHECK (rubric_version IS NULL OR length(rubric_version) BETWEEN 1 AND 128),
    rubric_sha256 TEXT CHECK (
        rubric_sha256 IS NULL OR (
            length(rubric_sha256) = 64 AND rubric_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    output_schema_version INTEGER CHECK (output_schema_version IS NULL OR output_schema_version = 1),
    output_schema_sha256 TEXT CHECK (
        output_schema_sha256 IS NULL OR (
            length(output_schema_sha256) = 64
            AND output_schema_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    response_equivalence REAL CHECK (
        response_equivalence IS NULL OR response_equivalence BETWEEN 0.0 AND 1.0
    ),
    response_equivalence_bits INTEGER,
    trajectory_equivalence REAL CHECK (
        trajectory_equivalence IS NULL OR trajectory_equivalence BETWEEN 0.0 AND 1.0
    ),
    trajectory_equivalence_bits INTEGER,
    judge_confidence REAL CHECK (judge_confidence IS NULL OR judge_confidence BETWEEN 0.0 AND 1.0),
    judge_confidence_bits INTEGER,
    response_weight REAL CHECK (response_weight IS NULL OR response_weight BETWEEN 0.0 AND 1.0),
    response_weight_bits INTEGER,
    trajectory_weight REAL CHECK (
        trajectory_weight IS NULL OR trajectory_weight BETWEEN 0.0 AND 1.0
    ),
    trajectory_weight_bits INTEGER,
    aggregate_score REAL CHECK (
        aggregate_score IS NULL OR aggregate_score BETWEEN 0.0 AND 1.000000001
    ),
    aggregate_score_bits INTEGER,
    label TEXT NOT NULL CHECK (label IN ('pass', 'fail', 'ambiguous')),
    binary_label TEXT CHECK (binary_label IS NULL OR binary_label IN ('pass', 'fail')),
    rationale TEXT CHECK (
        rationale IS NULL OR length(CAST(rationale AS BLOB)) BETWEEN 1 AND 16384
    ),
    is_partial INTEGER NOT NULL CHECK (is_partial IN (0, 1)),
    promotion_eligible INTEGER NOT NULL CHECK (promotion_eligible IN (0, 1)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (shadow_attempt_id, evaluator_version)
        REFERENCES shadow_attempts(shadow_attempt_id, evaluator_version) ON DELETE CASCADE,
    CHECK ((response_equivalence IS NULL) = (response_equivalence_bits IS NULL)),
    CHECK ((trajectory_equivalence IS NULL) = (trajectory_equivalence_bits IS NULL)),
    CHECK ((judge_confidence IS NULL) = (judge_confidence_bits IS NULL)),
    CHECK ((response_weight IS NULL) = (response_weight_bits IS NULL)),
    CHECK ((trajectory_weight IS NULL) = (trajectory_weight_bits IS NULL)),
    CHECK ((aggregate_score IS NULL) = (aggregate_score_bits IS NULL)),
    CHECK (
        (label = 'pass' AND binary_label = 'pass')
        OR (label = 'fail' AND binary_label = 'fail')
        OR (label = 'ambiguous' AND binary_label IS NULL)
    ),
    CHECK (
        promotion_eligible = CASE
            WHEN is_partial = 0 AND binary_label IS NOT NULL THEN 1 ELSE 0
        END
    ),
    CHECK (
        (
            source = 'deterministic_validator'
            AND judge_model IS NULL AND judge_model_revision IS NULL
            AND prompt_version IS NULL AND prompt_sha256 IS NULL
            AND rubric_version IS NULL AND rubric_sha256 IS NULL
            AND output_schema_version IS NULL AND output_schema_sha256 IS NULL
            AND response_equivalence IS NULL AND trajectory_equivalence IS NULL
            AND judge_confidence IS NULL AND response_weight IS NULL
            AND trajectory_weight IS NULL AND aggregate_score IS NULL
            AND rationale IS NULL AND label = 'fail' AND binary_label = 'fail'
        )
        OR
        (
            source = 'judge'
            AND judge_model IS NOT NULL AND judge_model_revision IS NOT NULL
            AND prompt_version IS NOT NULL AND prompt_sha256 IS NOT NULL
            AND rubric_version IS NOT NULL AND rubric_sha256 IS NOT NULL
            AND output_schema_version = 1 AND output_schema_sha256 IS NOT NULL
            AND response_equivalence IS NOT NULL AND trajectory_equivalence IS NOT NULL
            AND judge_confidence IS NOT NULL AND response_weight IS NOT NULL
            AND trajectory_weight IS NOT NULL AND aggregate_score IS NOT NULL
            AND rationale IS NOT NULL
        )
    ),
    CHECK (
        source <> 'judge'
        OR abs(response_weight + trajectory_weight - 1.0) <= 0.000000001
    ),
    UNIQUE (shadow_attempt_id, evaluator_version),
    UNIQUE (evaluation_id, shadow_attempt_id)
) STRICT;

CREATE TABLE shadow_results (
    shadow_result_id TEXT PRIMARY KEY CHECK (length(shadow_result_id) = 36),
    shadow_attempt_id TEXT NOT NULL UNIQUE REFERENCES shadow_attempts(shadow_attempt_id) ON DELETE CASCADE,
    terminal_class TEXT NOT NULL CHECK (
        terminal_class IN (
            'completed', 'deterministic_failure', 'operational_failure', 'skipped_cooloff',
            'canceled_shutdown', 'orphaned_before_schedule', 'orphaned_in_flight'
        )
    ),
    normalized_response_json TEXT CHECK (
        normalized_response_json IS NULL OR json_valid(normalized_response_json)
    ),
    response_fingerprint TEXT CHECK (
        response_fingerprint IS NULL OR (
            length(response_fingerprint) = 64
            AND response_fingerprint NOT GLOB '*[^0-9a-f]*'
        )
    ),
    deterministic_hard_failure TEXT CHECK (
        deterministic_hard_failure IS NULL OR deterministic_hard_failure IN (
            'tool_contract', 'response_schema', 'malformed_candidate'
        )
    ),
    operational_failure_class TEXT CHECK (
        operational_failure_class IS NULL OR (
            length(operational_failure_class) BETWEEN 1 AND 128
            AND operational_failure_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    latency_ms INTEGER CHECK (latency_ms IS NULL OR latency_ms >= 0),
    usage_json TEXT CHECK (usage_json IS NULL OR json_valid(usage_json)),
    evaluation_id TEXT,
    canonicalizable INTEGER NOT NULL CHECK (canonicalizable IN (0, 1)),
    query_inputs_json TEXT CHECK (query_inputs_json IS NULL OR json_valid(query_inputs_json)),
    partition_inputs_json TEXT NOT NULL CHECK (json_valid(partition_inputs_json)),
    vector_source_hash TEXT CHECK (
        vector_source_hash IS NULL OR (
            length(vector_source_hash) = 64
            AND vector_source_hash NOT GLOB '*[^0-9a-f]*'
        )
    ),
    noncanonicalizable_reason TEXT CHECK (
        noncanonicalizable_reason IS NULL OR (
            length(noncanonicalizable_reason) BETWEEN 1 AND 128
            AND noncanonicalizable_reason NOT GLOB '*[^a-z0-9_]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (terminal_class = 'deterministic_failure' AND deterministic_hard_failure IS NOT NULL)
        OR (terminal_class <> 'deterministic_failure' AND deterministic_hard_failure IS NULL)
    ),
    CHECK (
        (terminal_class = 'operational_failure' AND operational_failure_class IS NOT NULL)
        OR (terminal_class <> 'operational_failure' AND operational_failure_class IS NULL)
    ),
    CHECK (terminal_class <> 'completed' OR normalized_response_json IS NOT NULL),
    CHECK (
        (terminal_class IN ('completed', 'deterministic_failure') AND evaluation_id IS NOT NULL)
        OR
        (terminal_class NOT IN ('completed', 'deterministic_failure') AND evaluation_id IS NULL)
    ),
    CHECK (
        (canonicalizable = 1 AND query_inputs_json IS NOT NULL
            AND vector_source_hash IS NOT NULL AND noncanonicalizable_reason IS NULL)
        OR (canonicalizable = 0 AND query_inputs_json IS NULL
            AND vector_source_hash IS NULL AND noncanonicalizable_reason IS NOT NULL)
    ),
    FOREIGN KEY (evaluation_id, shadow_attempt_id)
        REFERENCES evaluations(evaluation_id, shadow_attempt_id)
        DEFERRABLE INITIALLY DEFERRED
) STRICT;

CREATE TABLE judge_attempts (
    judge_attempt_id TEXT PRIMARY KEY CHECK (length(judge_attempt_id) = 36),
    shadow_attempt_id TEXT NOT NULL,
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    learning_generation_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL CHECK (
        length(evaluator_version) = 64
        AND evaluator_version NOT GLOB '*[^0-9a-f]*'
    ),
    judge_input_sha256 TEXT NOT NULL CHECK (
        length(judge_input_sha256) = 64
        AND judge_input_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    candidate_response_json TEXT NOT NULL CHECK (json_valid(candidate_response_json)),
    candidate_response_fingerprint TEXT NOT NULL CHECK (
        length(candidate_response_fingerprint) = 64
        AND candidate_response_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    judge_model TEXT NOT NULL CHECK (length(judge_model) BETWEEN 1 AND 512),
    judge_model_revision TEXT NOT NULL CHECK (length(judge_model_revision) BETWEEN 1 AND 128),
    prompt_version TEXT NOT NULL CHECK (prompt_version <> ''),
    prompt_sha256 TEXT NOT NULL CHECK (
        length(prompt_sha256) = 64 AND prompt_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    rubric_version TEXT NOT NULL CHECK (rubric_version <> ''),
    rubric_sha256 TEXT NOT NULL CHECK (
        length(rubric_sha256) = 64 AND rubric_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    output_schema_version INTEGER NOT NULL CHECK (output_schema_version = 1),
    output_schema_sha256 TEXT NOT NULL CHECK (
        length(output_schema_sha256) = 64
        AND output_schema_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    attempt_ordinal INTEGER NOT NULL CHECK (attempt_ordinal IN (0, 1)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (
        shadow_attempt_id, process_instance_id, learning_generation_id, evaluator_version
    )
        REFERENCES shadow_attempts(
            shadow_attempt_id, process_instance_id, learning_generation_id, evaluator_version
        ) ON DELETE CASCADE,
    UNIQUE (shadow_attempt_id, evaluator_version, attempt_ordinal)
) STRICT;

CREATE TABLE judge_attempt_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    judge_attempt_state_event_id TEXT NOT NULL UNIQUE CHECK (length(judge_attempt_state_event_id) = 36),
    judge_attempt_id TEXT NOT NULL REFERENCES judge_attempts(judge_attempt_id) ON DELETE CASCADE,
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    dead_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'started', 'valid', 'invalid', 'transport_failure',
            'orphaned_in_flight', 'canceled_shutdown'
        )
    ),
    parse_result TEXT CHECK (
        parse_result IS NULL OR parse_result IN ('valid', 'invalid', 'operational')
    ),
    safe_output_json TEXT CHECK (safe_output_json IS NULL OR json_valid(safe_output_json)),
    raw_output_sha256 TEXT CHECK (
        raw_output_sha256 IS NULL OR (
            length(raw_output_sha256) = 64
            AND raw_output_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    raw_output_bytes INTEGER CHECK (raw_output_bytes IS NULL OR raw_output_bytes >= 0),
    stable_error_class TEXT CHECK (
        stable_error_class IS NULL OR (
            length(stable_error_class) BETWEEN 1 AND 128
            AND stable_error_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state = 'orphaned_in_flight' AND dead_process_instance_id IS NOT NULL)
        OR (state <> 'orphaned_in_flight' AND dead_process_instance_id IS NULL)
    ),
    CHECK ((raw_output_sha256 IS NULL) = (raw_output_bytes IS NULL)),
    CHECK (
        (state = 'valid' AND parse_result = 'valid' AND safe_output_json IS NOT NULL
            AND raw_output_sha256 IS NOT NULL AND raw_output_bytes IS NOT NULL
            AND stable_error_class IS NULL)
        OR (state = 'invalid' AND parse_result = 'invalid' AND safe_output_json IS NOT NULL
            AND raw_output_sha256 IS NOT NULL AND raw_output_bytes IS NOT NULL
            AND stable_error_class IS NULL)
        OR (state = 'transport_failure' AND parse_result = 'operational'
            AND safe_output_json IS NULL AND raw_output_sha256 IS NULL
            AND raw_output_bytes IS NULL AND stable_error_class IS NOT NULL)
        OR (state IN ('started', 'orphaned_in_flight', 'canceled_shutdown')
            AND parse_result IS NULL AND safe_output_json IS NULL
            AND raw_output_sha256 IS NULL AND raw_output_bytes IS NULL
            AND stable_error_class IS NULL)
    ),
    UNIQUE (judge_attempt_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_judge_attempt_terminal_state
    ON judge_attempt_state_events(judge_attempt_id)
    WHERE state <> 'started';
CREATE INDEX idx_judge_attempt_state_latest
    ON judge_attempt_state_events(judge_attempt_id, event_seq DESC);

CREATE TABLE evaluation_hard_failures (
    evaluation_id TEXT NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
    hard_failure TEXT NOT NULL CHECK (
        hard_failure IN ('tool_contract', 'response_schema', 'safety', 'malformed_candidate')
    ),
    canonical_ordinal INTEGER NOT NULL CHECK (canonical_ordinal BETWEEN 0 AND 3),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (hard_failure = 'tool_contract' AND canonical_ordinal = 0)
        OR (hard_failure = 'response_schema' AND canonical_ordinal = 1)
        OR (hard_failure = 'safety' AND canonical_ordinal = 2)
        OR (hard_failure = 'malformed_candidate' AND canonical_ordinal = 3)
    ),
    PRIMARY KEY (evaluation_id, hard_failure),
    UNIQUE (evaluation_id, canonical_ordinal)
) STRICT;

CREATE TABLE vector_spaces (
    vector_space_id TEXT PRIMARY KEY CHECK (
        length(vector_space_id) = 64
        AND vector_space_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;

CREATE TABLE embedding_jobs (
    embedding_job_id TEXT PRIMARY KEY CHECK (
        length(embedding_job_id) = 64
        AND embedding_job_id NOT GLOB '*[^0-9a-f]*'
    ),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_query_hash TEXT NOT NULL CHECK (
        length(canonical_query_hash) = 64
        AND canonical_query_hash NOT GLOB '*[^0-9a-f]*'
    ),
    content_hash TEXT NOT NULL CHECK (
        length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    lease_owner_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    ),
    attempt_generation INTEGER NOT NULL DEFAULT 0 CHECK (attempt_generation >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    next_eligible_at_unix_ms INTEGER NOT NULL CHECK (next_eligible_at_unix_ms >= 0),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (lease_owner_process_instance_id IS NULL AND lease_token IS NULL AND lease_expires_at_unix_ms IS NULL)
        OR
        (lease_owner_process_instance_id IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at_unix_ms IS NOT NULL)
    ),
    UNIQUE (vector_space_id, canonical_query_hash)
) STRICT;

CREATE INDEX idx_embedding_jobs_claim
    ON embedding_jobs(next_eligible_at_unix_ms, lease_expires_at_unix_ms, embedding_job_id);
CREATE INDEX idx_embedding_jobs_owner
    ON embedding_jobs(lease_owner_process_instance_id, lease_token);

CREATE TABLE embedding_job_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_job_state_event_id TEXT NOT NULL UNIQUE CHECK (length(embedding_job_state_event_id) = 36),
    embedding_job_id TEXT NOT NULL REFERENCES embedding_jobs(embedding_job_id) ON DELETE CASCADE,
    process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        length(state) BETWEEN 1 AND 64 AND state NOT GLOB '*[^a-z0-9_]*'
    ),
    attempt_generation INTEGER NOT NULL CHECK (attempt_generation >= 0),
    stable_error_class TEXT CHECK (
        stable_error_class IS NULL OR (
            length(stable_error_class) BETWEEN 1 AND 128
            AND stable_error_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (embedding_job_id, attempt_generation, state)
) STRICT;

CREATE INDEX idx_embedding_job_state_latest
    ON embedding_job_state_events(embedding_job_id, event_seq DESC);

CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY CHECK (length(embedding_id) = 36),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_query_hash TEXT NOT NULL CHECK (
        length(canonical_query_hash) = 64
        AND canonical_query_hash NOT GLOB '*[^0-9a-f]*'
    ),
    content_hash TEXT NOT NULL CHECK (
        length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (vector_space_id, canonical_query_hash)
) STRICT;

CREATE TABLE evidence_vector_links (
    evidence_vector_link_id TEXT PRIMARY KEY CHECK (length(evidence_vector_link_id) = 36),
    shadow_attempt_id TEXT NOT NULL,
    anchor_id TEXT NOT NULL,
    root_uuid TEXT NOT NULL CHECK (length(root_uuid) = 36),
    learning_generation_id TEXT NOT NULL,
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    evaluation_id TEXT,
    quality_label TEXT CHECK (quality_label IS NULL OR quality_label IN ('pass', 'fail')),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (shadow_attempt_id, anchor_id, learning_generation_id)
        REFERENCES shadow_attempts(shadow_attempt_id, anchor_id, learning_generation_id)
        ON DELETE CASCADE,
    FOREIGN KEY (anchor_id, root_uuid)
        REFERENCES anchors(anchor_id, root_uuid) ON DELETE CASCADE,
    FOREIGN KEY (evaluation_id, shadow_attempt_id)
        REFERENCES evaluations(evaluation_id, shadow_attempt_id),
    CHECK (quality_label IS NULL OR evaluation_id IS NOT NULL),
    UNIQUE (shadow_attempt_id, vector_space_id)
) STRICT;

CREATE TABLE evidence_vector_link_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_vector_link_state_event_id TEXT NOT NULL UNIQUE CHECK (length(evidence_vector_link_state_event_id) = 36),
    evidence_vector_link_id TEXT NOT NULL REFERENCES evidence_vector_links(evidence_vector_link_id) ON DELETE CASCADE,
    embedding_id TEXT REFERENCES embeddings(embedding_id),
    state TEXT NOT NULL CHECK (
        length(state) BETWEEN 1 AND 64 AND state NOT GLOB '*[^a-z0-9_]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK ((state = 'ready' AND embedding_id IS NOT NULL) OR state <> 'ready')
) STRICT;

CREATE INDEX idx_evidence_vector_link_state_latest
    ON evidence_vector_link_state_events(evidence_vector_link_id, event_seq DESC);

CREATE TABLE active_experiments (
    active_experiment_id TEXT PRIMARY KEY CHECK (length(active_experiment_id) = 36),
    project_uuid TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
    partition_hash TEXT NOT NULL CHECK (
        length(partition_hash) = 64 AND partition_hash NOT GLOB '*[^0-9a-f]*'
    ),
    learning_generation_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT NOT NULL,
    outcome_policy_hash TEXT NOT NULL CHECK (
        length(outcome_policy_hash) = 64
        AND outcome_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, cohort_generation_id)
        REFERENCES cohort_generations(project_uuid, cohort_generation_id),
    UNIQUE (
        pool_id, candidate_id, partition_hash, learning_generation_id,
        config_generation_id, cohort_generation_id, outcome_policy_hash
    ),
    UNIQUE (
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    )
) STRICT;

CREATE TABLE active_outcome_looks (
    active_outcome_look_id TEXT PRIMARY KEY CHECK (length(active_outcome_look_id) = 36),
    active_experiment_id TEXT NOT NULL REFERENCES active_experiments(active_experiment_id),
    look_ordinal INTEGER NOT NULL CHECK (look_ordinal >= 0),
    boundary_root_count INTEGER NOT NULL CHECK (boundary_root_count > 0),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (active_experiment_id, look_ordinal, boundary_root_count),
    UNIQUE (active_outcome_look_id, active_experiment_id)
) STRICT;

CREATE TABLE active_authorization_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_authorization_state_event_id TEXT NOT NULL UNIQUE CHECK (length(active_authorization_state_event_id) = 36),
    active_experiment_id TEXT NOT NULL REFERENCES active_experiments(active_experiment_id),
    active_outcome_look_id TEXT,
    state TEXT NOT NULL CHECK (state IN ('collecting', 'passed', 'rollback', 'exhausted')),
    valid_until_unix_ms INTEGER CHECK (
        valid_until_unix_ms IS NULL OR valid_until_unix_ms >= 0
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_outcome_look_id, active_experiment_id)
        REFERENCES active_outcome_looks(active_outcome_look_id, active_experiment_id),
    CHECK (
        (state = 'passed' AND active_outcome_look_id IS NOT NULL AND valid_until_unix_ms IS NOT NULL)
        OR (state <> 'passed' AND valid_until_unix_ms IS NULL)
    ),
    CHECK (
        state = 'collecting'
        OR active_outcome_look_id IS NOT NULL
    ),
    UNIQUE (active_authorization_state_event_id, active_experiment_id)
) STRICT;

CREATE INDEX idx_active_authorization_state_latest
    ON active_authorization_state_events(active_experiment_id, event_seq DESC);

CREATE TABLE decisions (
    decision_id TEXT PRIMARY KEY CHECK (length(decision_id) = 36),
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT,
    active_experiment_id TEXT,
    active_authorization_state_event_id TEXT,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    candidate_id TEXT CHECK (candidate_id IS NULL OR length(candidate_id) BETWEEN 1 AND 128),
    root_key TEXT CHECK (
        root_key IS NULL OR (
            length(root_key) = 64 AND root_key NOT GLOB '*[^0-9a-f]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id, config_generation_id)
        REFERENCES process_instances(project_uuid, process_instance_id, config_generation_id),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    FOREIGN KEY (project_uuid, cohort_generation_id)
        REFERENCES cohort_generations(project_uuid, cohort_generation_id),
    FOREIGN KEY (
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    ) REFERENCES active_experiments(
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    ),
    FOREIGN KEY (active_authorization_state_event_id, active_experiment_id)
        REFERENCES active_authorization_state_events(
            active_authorization_state_event_id, active_experiment_id
        ),
    CHECK (
        (active_experiment_id IS NULL AND active_authorization_state_event_id IS NULL)
        OR (active_experiment_id IS NOT NULL AND active_authorization_state_event_id IS NOT NULL
            AND cohort_generation_id IS NOT NULL AND candidate_id IS NOT NULL)
    ),
    UNIQUE (decision_id, project_uuid)
) STRICT;

CREATE TABLE decision_candidate_summaries (
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
    rank_ordinal INTEGER NOT NULL CHECK (rank_ordinal >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (decision_id, candidate_id),
    UNIQUE (decision_id, rank_ordinal)
) STRICT;

CREATE TABLE decision_neighbors (
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
    neighbor_ordinal INTEGER NOT NULL CHECK (neighbor_ordinal >= 0),
    evidence_vector_link_id TEXT NOT NULL REFERENCES evidence_vector_links(evidence_vector_link_id) ON DELETE CASCADE,
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (decision_id, neighbor_ordinal),
    UNIQUE (decision_id, evidence_vector_link_id)
) STRICT;

CREATE TABLE outcomes (
    outcome_id TEXT PRIMARY KEY CHECK (length(outcome_id) = 36),
    decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (decision_id, project_uuid)
        REFERENCES decisions(decision_id, project_uuid) ON DELETE CASCADE
) STRICT;

CREATE TABLE controls (
    control_id TEXT PRIMARY KEY CHECK (length(control_id) = 36),
    control_generation INTEGER NOT NULL UNIQUE CHECK (control_generation >= 0),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id)
) STRICT;

CREATE TABLE dependency_keys (
    dependency_key_id TEXT PRIMARY KEY CHECK (
        length(dependency_key_id) = 64
        AND dependency_key_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    key_kind TEXT NOT NULL CHECK (key_kind IN ('candidate', 'judge')),
    canonical_identity_json TEXT NOT NULL CHECK (json_valid(canonical_identity_json)),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = dependency_key_id),
    UNIQUE (dependency_key_id, project_uuid)
) STRICT;

CREATE TABLE dependency_operations (
    dependency_operation_id TEXT PRIMARY KEY CHECK (length(dependency_operation_id) = 36),
    dependency_key_id TEXT NOT NULL,
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    shadow_attempt_id TEXT NOT NULL,
    base_cooloff_seconds INTEGER NOT NULL CHECK (base_cooloff_seconds > 0),
    max_cooloff_seconds INTEGER NOT NULL CHECK (
        max_cooloff_seconds >= base_cooloff_seconds
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (dependency_key_id, project_uuid)
        REFERENCES dependency_keys(dependency_key_id, project_uuid),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    FOREIGN KEY (shadow_attempt_id, anchor_id, project_uuid, process_instance_id)
        REFERENCES shadow_attempts(
            shadow_attempt_id, anchor_id, project_uuid, process_instance_id
        ) ON DELETE CASCADE,
    UNIQUE (dependency_operation_id, dependency_key_id),
    UNIQUE (dependency_operation_id, dependency_key_id, anchor_id)
) STRICT;

CREATE TABLE dependency_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    dependency_state_event_id TEXT NOT NULL UNIQUE CHECK (length(dependency_state_event_id) = 36),
    dependency_key_id TEXT NOT NULL REFERENCES dependency_keys(dependency_key_id),
    dependency_operation_id TEXT,
    anchor_id TEXT REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('admitted', 'skipped_cooloff', 'success', 'failure')),
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
        OR (state = 'admitted' AND failure_class IS NULL AND (
            (consecutive_failures = 0 AND cooloff_until_unix_ms IS NULL)
            OR (consecutive_failures > 0 AND cooloff_until_unix_ms IS NOT NULL)
        ))
    ),
    UNIQUE (dependency_operation_id, state)
) STRICT;

CREATE UNIQUE INDEX uq_dependency_operation_terminal_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL AND state <> 'admitted';

CREATE UNIQUE INDEX uq_dependency_operation_claim_state
    ON dependency_state_events(dependency_operation_id)
    WHERE dependency_operation_id IS NOT NULL
      AND state IN ('admitted', 'skipped_cooloff');

CREATE INDEX idx_dependency_state_latest
    ON dependency_state_events(dependency_key_id, event_seq DESC);

CREATE TABLE health_events (
    health_event_id TEXT PRIMARY KEY CHECK (length(health_event_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT,
    requested_anchor_id TEXT CHECK (
        requested_anchor_id IS NULL OR length(requested_anchor_id) = 36
    ),
    requested_dependency_key_id TEXT CHECK (
        requested_dependency_key_id IS NULL OR (
            length(requested_dependency_key_id) = 64
            AND requested_dependency_key_id NOT GLOB '*[^0-9a-f]*'
        )
    ),
    anchor_id TEXT REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    dependency_key_id TEXT REFERENCES dependency_keys(dependency_key_id),
    stable_class TEXT NOT NULL CHECK (
        length(stable_class) BETWEEN 1 AND 128
        AND stable_class NOT GLOB '*[^a-z0-9_.]*'
    ),
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'degraded')),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id)
) STRICT;

CREATE TABLE retention_batches (
    retention_batch_id TEXT PRIMARY KEY CHECK (length(retention_batch_id) = 36),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT NOT NULL,
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
    CHECK (age_expired = 1 OR count_excess = 1),
    CHECK (
        selection_lower_bound_unix_ms IS NULL
        OR selection_upper_bound_unix_ms IS NULL
        OR selection_lower_bound_unix_ms <= selection_upper_bound_unix_ms
    )
) STRICT;

CREATE INDEX idx_anchor_retention
    ON anchor_windows(closed_at_unix_ms, anchor_id);
CREATE INDEX idx_anchor_process
    ON anchors(process_instance_id, anchor_id);
CREATE INDEX idx_sample_batch_process
    ON sample_batches(process_instance_id, sample_batch_id);
CREATE INDEX idx_judge_attempt_process
    ON judge_attempts(process_instance_id, judge_attempt_id);
CREATE INDEX idx_dependency_operation_process
    ON dependency_operations(process_instance_id, dependency_operation_id);
CREATE INDEX idx_shadow_results_partition
    ON shadow_attempts(
        learning_generation_id, api_family, transport_identity,
        anchor_model, candidate_id, candidate_model_revision, evaluator_version
    );
