-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE spec06_vector_placeholder_guard (
    singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
    placeholder_row_count INTEGER NOT NULL CHECK (placeholder_row_count = 0)
) STRICT;

INSERT INTO spec06_vector_placeholder_guard (singleton_key, placeholder_row_count)
SELECT
    1,
    (SELECT count(*) FROM vector_spaces)
    + (SELECT count(*) FROM embedding_jobs)
    + (SELECT count(*) FROM embedding_job_state_events)
    + (SELECT count(*) FROM embeddings)
    + (SELECT count(*) FROM evidence_vector_links)
    + (SELECT count(*) FROM evidence_vector_link_state_events);

DROP TABLE spec06_vector_placeholder_guard;

CREATE TABLE spec06_vector_sequence_snapshot (
    table_name TEXT PRIMARY KEY CHECK (
        table_name IN ('embedding_job_state_events', 'evidence_vector_link_state_events')
    ),
    sequence_value INTEGER NOT NULL CHECK (sequence_value >= 0)
) STRICT;

INSERT INTO spec06_vector_sequence_snapshot (table_name, sequence_value)
SELECT name, seq
FROM sqlite_sequence
WHERE name IN ('embedding_job_state_events', 'evidence_vector_link_state_events');

DROP TABLE embedding_job_state_events;
DROP TABLE evidence_vector_link_state_events;
DROP TABLE evidence_vector_links;
DROP TABLE embeddings;
DROP TABLE embedding_jobs;
DROP TABLE vector_spaces;

CREATE TABLE embedder_profiles (
    embedder_profile_version_id TEXT PRIMARY KEY CHECK (
        length(embedder_profile_version_id) = 64
        AND embedder_profile_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    profile_id TEXT NOT NULL CHECK (length(profile_id) BETWEEN 1 AND 128),
    protocol TEXT NOT NULL CHECK (protocol = 'openai-embeddings-v1'),
    endpoint_url TEXT NOT NULL CHECK (length(endpoint_url) BETWEEN 1 AND 2048),
    endpoint_identity_sha256 TEXT NOT NULL CHECK (
        length(endpoint_identity_sha256) = 64
        AND endpoint_identity_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    model TEXT NOT NULL CHECK (length(model) BETWEEN 1 AND 512),
    provider_revision TEXT NOT NULL CHECK (length(provider_revision) BETWEEN 1 AND 128),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    credential_env_name_sha256 TEXT CHECK (
        credential_env_name_sha256 IS NULL OR (
            length(credential_env_name_sha256) = 64
            AND credential_env_name_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    timeout_ms INTEGER NOT NULL CHECK (timeout_ms BETWEEN 1 AND 55000),
    max_in_flight INTEGER NOT NULL CHECK (max_in_flight BETWEEN 1 AND 256),
    batch_size INTEGER NOT NULL CHECK (batch_size BETWEEN 1 AND 128),
    egress_class TEXT NOT NULL CHECK (
        egress_class IN ('loopback_http', 'loopback_https', 'remote_https')
    ),
    canonical_profile_json TEXT NOT NULL CHECK (json_valid(canonical_profile_json)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = embedder_profile_version_id
    ),
    UNIQUE (profile_id, embedder_profile_version_id)
) STRICT;

CREATE TABLE vector_spaces (
    vector_space_id TEXT PRIMARY KEY CHECK (
        length(vector_space_id) = 64
        AND vector_space_id NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    embedder_profile_version_id TEXT NOT NULL
        REFERENCES embedder_profiles(embedder_profile_version_id),
    canonicalizer_version_id TEXT NOT NULL CHECK (
        length(canonicalizer_version_id) = 64
        AND canonicalizer_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    canonicalizer_identity_json TEXT NOT NULL CHECK (
        json_valid(canonicalizer_identity_json)
    ),
    endpoint_identity_sha256 TEXT NOT NULL CHECK (
        length(endpoint_identity_sha256) = 64
        AND endpoint_identity_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    model TEXT NOT NULL CHECK (length(model) BETWEEN 1 AND 512),
    provider_revision TEXT NOT NULL CHECK (length(provider_revision) BETWEEN 1 AND 128),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    metric TEXT NOT NULL CHECK (metric = 'cosine'),
    normalization TEXT NOT NULL CHECK (normalization = 'l2_f32_v1'),
    canonical_space_json TEXT NOT NULL CHECK (json_valid(canonical_space_json)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = vector_space_id),
    UNIQUE (vector_space_id, dimensions),
    UNIQUE (vector_space_id, embedder_profile_version_id, canonicalizer_version_id)
) STRICT;

CREATE TABLE pool_vector_space_mappings (
    project_uuid TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    policy_version_id TEXT NOT NULL CHECK (
        length(policy_version_id) = 64
        AND policy_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    profile_id TEXT NOT NULL CHECK (length(profile_id) BETWEEN 1 AND 128),
    embedder_profile_version_id TEXT NOT NULL
        REFERENCES embedder_profiles(embedder_profile_version_id),
    canonicalizer_version_id TEXT NOT NULL CHECK (
        length(canonicalizer_version_id) = 64
        AND canonicalizer_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_mapping_json TEXT NOT NULL CHECK (json_valid(canonical_mapping_json)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (project_uuid, config_generation_id, pool_id, policy_version_id),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    FOREIGN KEY (profile_id, embedder_profile_version_id)
        REFERENCES embedder_profiles(profile_id, embedder_profile_version_id),
    FOREIGN KEY (
        vector_space_id, embedder_profile_version_id, canonicalizer_version_id
    ) REFERENCES vector_spaces(
        vector_space_id, embedder_profile_version_id, canonicalizer_version_id
    )
) STRICT;

CREATE INDEX idx_pool_vector_space_mapping_space
    ON pool_vector_space_mappings(vector_space_id, project_uuid, pool_id);

CREATE TABLE routing_partitions (
    partition_id INTEGER PRIMARY KEY AUTOINCREMENT CHECK (partition_id > 0),
    partition_hash TEXT NOT NULL UNIQUE CHECK (
        length(partition_hash) = 64 AND partition_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_partition_json TEXT NOT NULL CHECK (json_valid(canonical_partition_json)),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    tenant_policy_hash TEXT NOT NULL CHECK (
        length(tenant_policy_hash) = 64
        AND tenant_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    agent_policy_hash TEXT NOT NULL CHECK (
        length(agent_policy_hash) = 64
        AND agent_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    policy_version_id TEXT NOT NULL CHECK (
        length(policy_version_id) = 64
        AND policy_version_id NOT GLOB '*[^0-9a-f]*'
    ),
    learning_generation_id TEXT NOT NULL CHECK (length(learning_generation_id) = 36),
    api_family TEXT NOT NULL CHECK (
        api_family IN (
            'openai_chat_completions', 'openai_responses', 'anthropic_messages'
        )
    ),
    transport_identity TEXT NOT NULL CHECK (length(transport_identity) BETWEEN 1 AND 256),
    anchor_model TEXT NOT NULL CHECK (length(anchor_model) BETWEEN 1 AND 512),
    anchor_revision TEXT NOT NULL CHECK (length(anchor_revision) BETWEEN 1 AND 128),
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
    candidate_model TEXT NOT NULL CHECK (length(candidate_model) BETWEEN 1 AND 512),
    candidate_model_revision TEXT NOT NULL CHECK (
        length(candidate_model_revision) BETWEEN 1 AND 128
    ),
    decoding_fingerprint TEXT NOT NULL CHECK (
        length(decoding_fingerprint) = 64
        AND decoding_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    evaluator_version TEXT NOT NULL CHECK (
        length(evaluator_version) = 64
        AND evaluator_version NOT GLOB '*[^0-9a-f]*'
    ),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = partition_hash),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    UNIQUE (partition_id, vector_space_id)
) STRICT;

CREATE INDEX idx_routing_partitions_space
    ON routing_partitions(vector_space_id, partition_id);

CREATE TABLE canonical_routing_queries (
    canonical_query_hash TEXT PRIMARY KEY CHECK (
        length(canonical_query_hash) = 64
        AND canonical_query_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_query_json TEXT NOT NULL CHECK (json_valid(canonical_query_json)),
    canonical_size_bytes INTEGER NOT NULL CHECK (
        canonical_size_bytes > 0
        AND canonical_size_bytes <= 33554432
        AND canonical_size_bytes = length(CAST(canonical_query_json AS BLOB))
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (canonical_payload_hash = canonical_query_hash)
) STRICT;

CREATE TABLE vectorization_outcomes (
    vectorization_outcome_id TEXT PRIMARY KEY CHECK (
        length(vectorization_outcome_id) = 64
        AND vectorization_outcome_id NOT GLOB '*[^0-9a-f]*'
    ),
    shadow_attempt_id TEXT NOT NULL,
    anchor_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_query_hash TEXT REFERENCES canonical_routing_queries(canonical_query_hash),
    outcome TEXT NOT NULL CHECK (
        outcome IN ('canonicalized', 'noncanonicalizable', 'source_expired')
    ),
    stable_reason TEXT CHECK (
        stable_reason IS NULL OR (
            length(stable_reason) BETWEEN 1 AND 128
            AND stable_reason NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (shadow_attempt_id, anchor_id, learning_generation_id)
        REFERENCES shadow_attempts(shadow_attempt_id, anchor_id, learning_generation_id)
        ON DELETE CASCADE,
    CHECK (
        (outcome = 'canonicalized' AND canonical_query_hash IS NOT NULL AND stable_reason IS NULL)
        OR (outcome IN ('noncanonicalizable', 'source_expired')
            AND canonical_query_hash IS NULL AND stable_reason IS NOT NULL)
    ),
    UNIQUE (shadow_attempt_id, vector_space_id),
    UNIQUE (vectorization_outcome_id, shadow_attempt_id, vector_space_id, canonical_query_hash)
) STRICT;

CREATE INDEX idx_vectorization_outcomes_backfill
    ON vectorization_outcomes(vector_space_id, outcome, shadow_attempt_id);

CREATE TABLE embedding_jobs (
    embedding_job_id TEXT PRIMARY KEY CHECK (
        length(embedding_job_id) = 64
        AND embedding_job_id NOT GLOB '*[^0-9a-f]*'
    ),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_query_hash TEXT NOT NULL
        REFERENCES canonical_routing_queries(canonical_query_hash),
    content_hash TEXT NOT NULL CHECK (
        length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    lease_owner_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    ),
    attempt_generation INTEGER NOT NULL DEFAULT 0 CHECK (attempt_generation >= 0),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 5),
    next_eligible_at_unix_ms INTEGER NOT NULL CHECK (next_eligible_at_unix_ms >= 0),
    terminal_error_class TEXT CHECK (
        terminal_error_class IS NULL OR (
            length(terminal_error_class) BETWEEN 1 AND 128
            AND terminal_error_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    failure_propagation_cursor TEXT CHECK (
        failure_propagation_cursor IS NULL OR length(failure_propagation_cursor) = 36
    ),
    failure_propagation_complete INTEGER NOT NULL DEFAULT 0 CHECK (
        failure_propagation_complete IN (0, 1)
    ),
    reset_actor TEXT CHECK (
        reset_actor IS NULL OR (length(reset_actor) BETWEEN 1 AND 128 AND trim(reset_actor) <> '')
    ),
    reset_reason TEXT CHECK (
        reset_reason IS NULL OR (
            length(reset_reason) BETWEEN 1 AND 512 AND trim(reset_reason) <> ''
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (lease_owner_process_instance_id IS NULL
            AND lease_token IS NULL AND lease_expires_at_unix_ms IS NULL)
        OR (lease_owner_process_instance_id IS NOT NULL
            AND lease_token IS NOT NULL AND lease_expires_at_unix_ms IS NOT NULL)
    ),
    CHECK (
        (terminal_error_class IS NULL
            AND failure_propagation_cursor IS NULL AND failure_propagation_complete = 0)
        OR terminal_error_class IS NOT NULL
    ),
    CHECK ((reset_actor IS NULL) = (reset_reason IS NULL)),
    UNIQUE (vector_space_id, canonical_query_hash),
    UNIQUE (embedding_job_id, vector_space_id, canonical_query_hash),
    UNIQUE (embedding_job_id, vector_space_id, canonical_query_hash, content_hash)
) STRICT;

CREATE INDEX idx_embedding_jobs_claim
    ON embedding_jobs(next_eligible_at_unix_ms, lease_expires_at_unix_ms, embedding_job_id);
CREATE INDEX idx_embedding_jobs_owner
    ON embedding_jobs(lease_owner_process_instance_id, lease_token);
CREATE INDEX idx_embedding_jobs_failure_propagation
    ON embedding_jobs(failure_propagation_complete, embedding_job_id)
    WHERE terminal_error_class IS NOT NULL;

CREATE TABLE embedding_job_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_job_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(embedding_job_state_event_id) = 36
    ),
    embedding_job_id TEXT NOT NULL
        REFERENCES embedding_jobs(embedding_job_id) ON DELETE CASCADE,
    process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'pending', 'claimed', 'released', 'retry_scheduled',
            'completed', 'terminal_failure', 'quarantined',
            'reset', 'orphaned_in_flight'
        )
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
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 5),
    next_eligible_at_unix_ms INTEGER NOT NULL DEFAULT 0 CHECK (
        next_eligible_at_unix_ms >= 0
    ),
    reset_actor TEXT CHECK (
        reset_actor IS NULL OR (length(reset_actor) BETWEEN 1 AND 128 AND trim(reset_actor) <> '')
    ),
    reset_reason TEXT CHECK (
        reset_reason IS NULL OR (
            length(reset_reason) BETWEEN 1 AND 512 AND trim(reset_reason) <> ''
        )
    ),
    CHECK ((lease_token IS NULL) = (lease_expires_at_unix_ms IS NULL)),
    CHECK (
        (state IN ('retry_scheduled', 'terminal_failure', 'quarantined')
            AND stable_error_class IS NOT NULL)
        OR (state NOT IN ('retry_scheduled', 'terminal_failure', 'quarantined')
            AND stable_error_class IS NULL)
    ),
    CHECK (
        (state = 'reset' AND reset_actor IS NOT NULL AND reset_reason IS NOT NULL)
        OR (state <> 'reset' AND reset_actor IS NULL AND reset_reason IS NULL)
    ),
    UNIQUE (embedding_job_id, attempt_generation, state)
) STRICT;

CREATE INDEX idx_embedding_job_state_latest
    ON embedding_job_state_events(embedding_job_id, event_seq DESC);

CREATE TABLE embeddings (
    embedding_id TEXT PRIMARY KEY CHECK (length(embedding_id) = 36),
    vector_space_id TEXT NOT NULL,
    canonical_query_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL CHECK (
        length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'
    ),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    vector_blob BLOB NOT NULL CHECK (length(vector_blob) = dimensions * 4),
    vector_checksum TEXT NOT NULL CHECK (
        length(vector_checksum) = 64 AND vector_checksum NOT GLOB '*[^0-9a-f]*'
    ),
    source TEXT NOT NULL CHECK (source IN ('provider', 'cache')),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (vector_space_id, dimensions)
        REFERENCES vector_spaces(vector_space_id, dimensions),
    FOREIGN KEY (canonical_query_hash)
        REFERENCES canonical_routing_queries(canonical_query_hash),
    UNIQUE (vector_space_id, canonical_query_hash),
    UNIQUE (embedding_id, vector_space_id, canonical_query_hash)
) STRICT;

CREATE TABLE evidence_vector_links (
    evidence_vector_link_id TEXT PRIMARY KEY CHECK (length(evidence_vector_link_id) = 36),
    vectorization_outcome_id TEXT NOT NULL,
    shadow_attempt_id TEXT NOT NULL,
    anchor_id TEXT NOT NULL,
    root_uuid TEXT NOT NULL CHECK (length(root_uuid) = 36),
    learning_generation_id TEXT NOT NULL,
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    partition_id INTEGER NOT NULL,
    canonical_query_hash TEXT NOT NULL REFERENCES canonical_routing_queries(canonical_query_hash),
    terminal_class TEXT NOT NULL CHECK (
        terminal_class IN (
            'completed', 'deterministic_failure', 'operational_failure', 'skipped_cooloff',
            'canceled_shutdown', 'orphaned_before_schedule', 'orphaned_in_flight'
        )
    ),
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
    FOREIGN KEY (partition_id, vector_space_id)
        REFERENCES routing_partitions(partition_id, vector_space_id),
    FOREIGN KEY (
        vectorization_outcome_id, shadow_attempt_id, vector_space_id, canonical_query_hash
    ) REFERENCES vectorization_outcomes(
        vectorization_outcome_id, shadow_attempt_id, vector_space_id, canonical_query_hash
    ) ON DELETE CASCADE,
    CHECK (
        (terminal_class IN ('completed', 'deterministic_failure')
            AND evaluation_id IS NOT NULL)
        OR (terminal_class NOT IN ('completed', 'deterministic_failure')
            AND evaluation_id IS NULL)
    ),
    CHECK (quality_label IS NULL OR evaluation_id IS NOT NULL),
    UNIQUE (shadow_attempt_id, vector_space_id),
    UNIQUE (evidence_vector_link_id, vector_space_id, canonical_query_hash)
) STRICT;

CREATE INDEX idx_evidence_vector_link_anchor
    ON evidence_vector_links(anchor_id);
CREATE INDEX idx_evidence_vector_links_partition
    ON evidence_vector_links(partition_id, evidence_vector_link_id);

CREATE TABLE evidence_vector_link_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_vector_link_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(evidence_vector_link_state_event_id) = 36
    ),
    evidence_vector_link_id TEXT NOT NULL
        REFERENCES evidence_vector_links(evidence_vector_link_id) ON DELETE CASCADE,
    embedding_id TEXT REFERENCES embeddings(embedding_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'pending_embedding', 'pending_index', 'ready',
            'failed_embedding', 'failed_index', 'canceled_retention'
        )
    ),
    attempt_generation INTEGER NOT NULL DEFAULT 0 CHECK (
        attempt_generation BETWEEN 0 AND 64
    ),
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
        (state = 'pending_embedding' AND embedding_id IS NULL AND stable_error_class IS NULL)
        OR (state IN ('pending_index', 'ready')
            AND embedding_id IS NOT NULL AND stable_error_class IS NULL)
        OR (state = 'failed_embedding'
            AND embedding_id IS NULL AND stable_error_class IS NOT NULL)
        OR (state = 'failed_index'
            AND embedding_id IS NOT NULL
            AND attempt_generation = 64
            AND stable_error_class = 'router.vector.materialization_attempt_limit')
        OR (state = 'canceled_retention' AND stable_error_class IS NULL)
    ),
    UNIQUE (evidence_vector_link_id, attempt_generation, state)
) STRICT;

CREATE INDEX idx_evidence_vector_link_state_latest
    ON evidence_vector_link_state_events(evidence_vector_link_id, event_seq DESC);
CREATE INDEX idx_evidence_vector_link_state_embedding
    ON evidence_vector_link_state_events(embedding_id)
    WHERE embedding_id IS NOT NULL;

CREATE TABLE vector_materialization_jobs (
    vector_materialization_job_id TEXT PRIMARY KEY CHECK (
        length(vector_materialization_job_id) = 64
        AND vector_materialization_job_id NOT GLOB '*[^0-9a-f]*'
    ),
    evidence_vector_link_id TEXT NOT NULL UNIQUE
        REFERENCES evidence_vector_links(evidence_vector_link_id) ON DELETE CASCADE,
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    canonical_query_hash TEXT NOT NULL REFERENCES canonical_routing_queries(canonical_query_hash),
    embedding_job_id TEXT,
    embedding_id TEXT,
    lease_owner_process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    ),
    attempt_generation INTEGER NOT NULL DEFAULT 0 CHECK (
        attempt_generation BETWEEN 0 AND 64
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 64),
    next_eligible_at_unix_ms INTEGER NOT NULL CHECK (next_eligible_at_unix_ms >= 0),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (lease_owner_process_instance_id IS NULL
            AND lease_token IS NULL AND lease_expires_at_unix_ms IS NULL)
        OR (lease_owner_process_instance_id IS NOT NULL
            AND lease_token IS NOT NULL AND lease_expires_at_unix_ms IS NOT NULL)
    ),
    CHECK (embedding_job_id IS NOT NULL OR embedding_id IS NOT NULL),
    FOREIGN KEY (evidence_vector_link_id, vector_space_id, canonical_query_hash)
        REFERENCES evidence_vector_links(
            evidence_vector_link_id, vector_space_id, canonical_query_hash
        ) ON DELETE CASCADE,
    FOREIGN KEY (embedding_job_id, vector_space_id, canonical_query_hash)
        REFERENCES embedding_jobs(
            embedding_job_id, vector_space_id, canonical_query_hash
        ),
    FOREIGN KEY (embedding_id, vector_space_id, canonical_query_hash)
        REFERENCES embeddings(embedding_id, vector_space_id, canonical_query_hash)
) STRICT;

CREATE INDEX idx_vector_materialization_jobs_claim
    ON vector_materialization_jobs(
        next_eligible_at_unix_ms, lease_expires_at_unix_ms, vector_materialization_job_id
    );
CREATE INDEX idx_vector_materialization_jobs_owner
    ON vector_materialization_jobs(lease_owner_process_instance_id, lease_token);
CREATE INDEX idx_vector_materialization_jobs_embedding_job
    ON vector_materialization_jobs(embedding_job_id, evidence_vector_link_id)
    WHERE embedding_job_id IS NOT NULL;

CREATE TABLE vector_materialization_job_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    vector_materialization_job_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(vector_materialization_job_state_event_id) = 36
    ),
    vector_materialization_job_id TEXT NOT NULL
        REFERENCES vector_materialization_jobs(vector_materialization_job_id)
        ON DELETE CASCADE,
    process_instance_id TEXT REFERENCES process_instances(process_instance_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'pending_embedding', 'claimed', 'released', 'retry_scheduled',
            'pending_index', 'ready', 'failed_embedding', 'failed_index',
            'canceled_retention', 'orphaned_in_flight'
        )
    ),
    attempt_generation INTEGER NOT NULL CHECK (attempt_generation BETWEEN 0 AND 64),
    stable_error_class TEXT CHECK (
        stable_error_class IS NULL OR (
            length(stable_error_class) BETWEEN 1 AND 128
            AND stable_error_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    lease_token TEXT CHECK (lease_token IS NULL OR length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER CHECK (
        lease_expires_at_unix_ms IS NULL OR lease_expires_at_unix_ms >= 0
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 64),
    next_eligible_at_unix_ms INTEGER NOT NULL DEFAULT 0 CHECK (
        next_eligible_at_unix_ms >= 0
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK ((lease_token IS NULL) = (lease_expires_at_unix_ms IS NULL)),
    CHECK (
        (state IN ('retry_scheduled', 'failed_embedding', 'failed_index')
            AND stable_error_class IS NOT NULL)
        OR (state NOT IN ('retry_scheduled', 'failed_embedding', 'failed_index')
            AND stable_error_class IS NULL)
    ),
    CHECK (
        state <> 'failed_index'
        OR (
            attempt_generation = 64
            AND attempt_count = 64
            AND stable_error_class = 'router.vector.materialization_attempt_limit'
        )
    ),
    UNIQUE (vector_materialization_job_id, attempt_generation, state)
) STRICT;

CREATE INDEX idx_vector_materialization_job_state_latest
    ON vector_materialization_job_state_events(
        vector_materialization_job_id, event_seq DESC
    );

CREATE TABLE vector_space_source_sequences (
    vector_space_id TEXT PRIMARY KEY REFERENCES vector_spaces(vector_space_id),
    source_seq INTEGER NOT NULL CHECK (source_seq >= 0),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    )
) STRICT;

CREATE TABLE vector_source_change_events (
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    source_seq INTEGER NOT NULL CHECK (source_seq > 0),
    operation TEXT NOT NULL CHECK (operation IN ('insert', 'delete')),
    record_id TEXT NOT NULL CHECK (length(record_id) = 36),
    partition_id INTEGER NOT NULL,
    vector_checksum TEXT NOT NULL CHECK (
        length(vector_checksum) = 64 AND vector_checksum NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (vector_space_id, source_seq),
    FOREIGN KEY (partition_id, vector_space_id)
        REFERENCES routing_partitions(partition_id, vector_space_id)
) STRICT;

CREATE INDEX idx_vector_source_change_record
    ON vector_source_change_events(vector_space_id, record_id, source_seq);

CREATE TABLE vector_index_manifest (
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    generation INTEGER NOT NULL CHECK (generation > 0),
    state TEXT NOT NULL CHECK (
        state IN ('building', 'active', 'unavailable', 'corrupt', 'retired', 'dropped')
    ),
    root_table_name TEXT NOT NULL UNIQUE CHECK (
        length(root_table_name) BETWEEN 78 AND 96
        AND root_table_name NOT GLOB '*[^a-z0-9_]*'
        AND root_table_name = 'router_vec_' || vector_space_id || '_g' || generation
    ),
    dimensions INTEGER NOT NULL CHECK (dimensions BETWEEN 1 AND 8192),
    expected_schema_objects_json TEXT NOT NULL CHECK (
        json_valid(expected_schema_objects_json)
        AND json_type(expected_schema_objects_json) = 'array'
        AND json_array_length(expected_schema_objects_json) = 5
    ),
    expected_schema_objects_sha256 TEXT NOT NULL CHECK (
        length(expected_schema_objects_sha256) = 64
        AND expected_schema_objects_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    base_source_seq INTEGER NOT NULL CHECK (base_source_seq >= 0),
    applied_source_seq INTEGER NOT NULL CHECK (
        applied_source_seq >= base_source_seq
    ),
    build_cursor_record_id TEXT CHECK (
        build_cursor_record_id IS NULL OR length(build_cursor_record_id) = 36
    ),
    source_record_count INTEGER CHECK (
        source_record_count IS NULL OR source_record_count >= 0
    ),
    source_fingerprint_sha256 TEXT CHECK (
        source_fingerprint_sha256 IS NULL OR (
            length(source_fingerprint_sha256) = 64
            AND source_fingerprint_sha256 NOT GLOB '*[^0-9a-f]*'
        )
    ),
    stable_error_class TEXT CHECK (
        stable_error_class IS NULL OR (
            length(stable_error_class) BETWEEN 1 AND 128
            AND stable_error_class NOT GLOB '*[^a-z0-9_.]*'
        )
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    activated_at_unix_ms INTEGER CHECK (
        activated_at_unix_ms IS NULL OR activated_at_unix_ms >= created_at_unix_ms
    ),
    retired_at_unix_ms INTEGER CHECK (
        retired_at_unix_ms IS NULL OR retired_at_unix_ms >= created_at_unix_ms
    ),
    dropped_at_unix_ms INTEGER CHECK (
        dropped_at_unix_ms IS NULL OR dropped_at_unix_ms >= created_at_unix_ms
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (vector_space_id, generation),
    FOREIGN KEY (vector_space_id, dimensions)
        REFERENCES vector_spaces(vector_space_id, dimensions),
    CHECK ((source_record_count IS NULL) = (source_fingerprint_sha256 IS NULL)),
    CHECK (
        (state = 'building'
            AND activated_at_unix_ms IS NULL
            AND retired_at_unix_ms IS NULL AND dropped_at_unix_ms IS NULL
            AND stable_error_class IS NULL)
        OR (state = 'active'
            AND activated_at_unix_ms IS NOT NULL
            AND retired_at_unix_ms IS NULL AND dropped_at_unix_ms IS NULL
            AND source_record_count IS NOT NULL AND stable_error_class IS NULL)
        OR (state IN ('unavailable', 'corrupt')
            AND activated_at_unix_ms IS NOT NULL
            AND retired_at_unix_ms IS NULL AND dropped_at_unix_ms IS NULL
            AND source_record_count IS NOT NULL AND stable_error_class IS NOT NULL)
        OR (state = 'retired'
            AND activated_at_unix_ms IS NOT NULL
            AND retired_at_unix_ms IS NOT NULL AND dropped_at_unix_ms IS NULL
            AND source_record_count IS NOT NULL)
        OR (state = 'dropped'
            AND activated_at_unix_ms IS NOT NULL
            AND retired_at_unix_ms IS NOT NULL AND dropped_at_unix_ms IS NOT NULL
            AND source_record_count IS NOT NULL
            AND dropped_at_unix_ms >= retired_at_unix_ms)
    )
) STRICT;

CREATE UNIQUE INDEX uq_vector_index_manifest_current
    ON vector_index_manifest(vector_space_id)
    WHERE state IN ('active', 'unavailable', 'corrupt');
CREATE UNIQUE INDEX uq_vector_index_manifest_building
    ON vector_index_manifest(vector_space_id)
    WHERE state = 'building';
CREATE INDEX idx_vector_index_manifest_cleanup
    ON vector_index_manifest(state, vector_space_id, generation);

CREATE TABLE vector_index_rebuild_leases (
    vector_space_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL CHECK (generation > 0),
    lease_generation INTEGER NOT NULL CHECK (lease_generation > 0),
    lease_owner_process_instance_id TEXT NOT NULL
        REFERENCES process_instances(process_instance_id),
    lease_token TEXT NOT NULL UNIQUE CHECK (length(lease_token) = 36),
    lease_expires_at_unix_ms INTEGER NOT NULL CHECK (lease_expires_at_unix_ms >= 0),
    base_source_seq INTEGER NOT NULL CHECK (base_source_seq >= 0),
    applied_source_seq INTEGER NOT NULL CHECK (applied_source_seq >= base_source_seq),
    build_cursor_record_id TEXT CHECK (
        build_cursor_record_id IS NULL OR length(build_cursor_record_id) = 36
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms >= created_at_unix_ms),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (vector_space_id, generation)
        REFERENCES vector_index_manifest(vector_space_id, generation)
        ON DELETE CASCADE
) STRICT;

CREATE INDEX idx_vector_index_rebuild_lease_expiry
    ON vector_index_rebuild_leases(lease_expires_at_unix_ms, vector_space_id);
CREATE INDEX idx_vector_index_rebuild_lease_owner
    ON vector_index_rebuild_leases(lease_owner_process_instance_id, lease_token);

INSERT INTO sqlite_sequence (name, seq)
SELECT snapshot.table_name, snapshot.sequence_value
FROM spec06_vector_sequence_snapshot AS snapshot
WHERE NOT EXISTS (
    SELECT 1 FROM sqlite_sequence WHERE name = snapshot.table_name
);

UPDATE sqlite_sequence
SET seq = max(
    seq,
    coalesce(
        (
            SELECT snapshot.sequence_value
            FROM spec06_vector_sequence_snapshot AS snapshot
            WHERE snapshot.table_name = sqlite_sequence.name
        ),
        seq
    )
)
WHERE name IN ('embedding_job_state_events', 'evidence_vector_link_state_events');

DROP TABLE spec06_vector_sequence_snapshot;
