-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- Pre-v6 binaries were forbidden from writing these reserved tables. Prove that
-- invariant before replacing any placeholder or rebuilding the populated
-- Recommend decision graph.
CREATE TABLE spec08_placeholder_guard (
    singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
    control_row_count INTEGER NOT NULL CHECK (control_row_count = 0),
    outcome_row_count INTEGER NOT NULL CHECK (outcome_row_count = 0),
    experiment_row_count INTEGER NOT NULL CHECK (experiment_row_count = 0),
    look_row_count INTEGER NOT NULL CHECK (look_row_count = 0),
    authorization_row_count INTEGER NOT NULL CHECK (authorization_row_count = 0),
    authorization_sequence INTEGER NOT NULL CHECK (authorization_sequence = 0)
) STRICT;

INSERT INTO spec08_placeholder_guard (
    singleton_key, control_row_count, outcome_row_count,
    experiment_row_count, look_row_count, authorization_row_count,
    authorization_sequence
)
SELECT
    1,
    (SELECT count(*) FROM controls),
    (SELECT count(*) FROM outcomes),
    (SELECT count(*) FROM active_experiments),
    (SELECT count(*) FROM active_outcome_looks),
    (SELECT count(*) FROM active_authorization_state_events),
    coalesce((
        SELECT seq FROM sqlite_sequence
        WHERE name = 'active_authorization_state_events'
    ), 0);

DROP TABLE spec08_placeholder_guard;

DROP TABLE outcomes;
DROP TABLE controls;
DROP TABLE active_authorization_state_events;
DROP TABLE active_outcome_looks;
DROP TABLE active_experiments;

ALTER TABLE decision_neighbors
    RENAME TO spec08_v5_decision_neighbors;
ALTER TABLE decision_candidate_summaries
    RENAME TO spec08_v5_decision_candidate_summaries;
ALTER TABLE decisions
    RENAME TO spec08_v5_decisions;

CREATE TABLE outcome_policy_versions (
    outcome_policy_hash TEXT PRIMARY KEY CHECK (
        length(outcome_policy_hash) = 64
        AND outcome_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (
        length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    canonical_policy_json TEXT NOT NULL CHECK (
        json_valid(canonical_policy_json)
        AND length(CAST(canonical_policy_json AS BLOB)) BETWEEN 2 AND 1048576
    ),
    matcher_algorithm_id TEXT NOT NULL CHECK (
        length(CAST(matcher_algorithm_id AS BLOB)) BETWEEN 1 AND 128
    ),
    reducer_algorithm_id TEXT NOT NULL CHECK (
        length(CAST(reducer_algorithm_id AS BLOB)) BETWEEN 1 AND 128
    ),
    decay_algorithm_id TEXT NOT NULL CHECK (
        length(CAST(decay_algorithm_id AS BLOB)) BETWEEN 1 AND 128
    ),
    active_math_algorithm_id_sha256 TEXT NOT NULL CHECK (
        length(active_math_algorithm_id_sha256) = 64
        AND active_math_algorithm_id_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = outcome_policy_hash
    ),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    UNIQUE (project_uuid, pool_id, outcome_policy_hash)
) STRICT;

CREATE INDEX idx_outcome_policy_versions_generation
    ON outcome_policy_versions(project_uuid, config_generation_id, pool_id);

CREATE TABLE config_generation_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    config_generation_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(config_generation_state_event_id) = 36
    ),
    config_epoch INTEGER NOT NULL UNIQUE CHECK (
        config_epoch BETWEEN 1 AND 9223372036854775807
    ),
    project_uuid TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    predecessor_event_hash TEXT CHECK (
        predecessor_event_hash IS NULL OR (
            length(predecessor_event_hash) = 64
            AND predecessor_event_hash NOT GLOB '*[^0-9a-f]*'
        )
    ),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    event_hash TEXT NOT NULL UNIQUE CHECK (
        length(event_hash) = 64 AND event_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = event_hash
    ),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    FOREIGN KEY (predecessor_event_hash)
        REFERENCES config_generation_state_events(event_hash),
    CHECK (
        (config_epoch = 1 AND predecessor_event_hash IS NULL)
        OR (config_epoch > 1 AND predecessor_event_hash IS NOT NULL)
    ),
    UNIQUE (project_uuid, config_generation_id),
    UNIQUE (project_uuid, config_epoch),
    UNIQUE (project_uuid, event_hash)
) STRICT;

CREATE INDEX idx_config_generation_state_latest
    ON config_generation_state_events(project_uuid, config_epoch DESC);

CREATE TABLE process_writer_capabilities (
    process_instance_id TEXT NOT NULL,
    project_uuid TEXT NOT NULL,
    writer_protocol TEXT NOT NULL CHECK (writer_protocol = 'active-v6'),
    schema_version INTEGER NOT NULL CHECK (schema_version = 6),
    verified_at_unix_ms INTEGER NOT NULL CHECK (verified_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    PRIMARY KEY (process_instance_id, writer_protocol),
    UNIQUE (project_uuid, process_instance_id, writer_protocol)
) STRICT;

CREATE INDEX idx_process_writer_capabilities_protocol
    ON process_writer_capabilities(project_uuid, writer_protocol, process_instance_id);

CREATE TABLE controls (
    control_id TEXT PRIMARY KEY CHECK (length(control_id) = 36),
    control_generation INTEGER NOT NULL UNIQUE CHECK (control_generation >= 0),
    originating_history_ordinal INTEGER NOT NULL UNIQUE CHECK (
        originating_history_ordinal >= 0
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('all', 'pool')),
    pool_id TEXT CHECK (
        pool_id IS NULL OR length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    force_anchor INTEGER NOT NULL CHECK (force_anchor IN (0, 1)),
    paused INTEGER NOT NULL CHECK (paused IN (0, 1)),
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
        (scope_kind = 'all' AND pool_id IS NULL)
        OR (scope_kind = 'pool' AND pool_id IS NOT NULL)
    ),
    CHECK (
        (control_generation = 0
            AND originating_history_ordinal = 0
            AND control_id = '00000000-0000-0000-0000-000000000000'
            AND scope_kind = 'all')
        OR (control_generation > 0
            AND originating_history_ordinal > 0
            AND control_id <> '00000000-0000-0000-0000-000000000000')
    ),
    UNIQUE (project_uuid, control_generation)
) STRICT;

CREATE INDEX idx_controls_scope_latest
    ON controls(project_uuid, scope_kind, pool_id, control_generation DESC);

CREATE TABLE control_mutation_receipts (
    mutation_id TEXT PRIMARY KEY CHECK (length(mutation_id) = 36),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    history_ordinal INTEGER NOT NULL UNIQUE CHECK (history_ordinal > 0),
    predecessor_chain_hash TEXT NOT NULL CHECK (
        length(predecessor_chain_hash) = 64
        AND predecessor_chain_hash NOT GLOB '*[^0-9a-f]*'
    ),
    chain_tip_hash TEXT NOT NULL UNIQUE CHECK (
        length(chain_tip_hash) = 64 AND chain_tip_hash NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    result TEXT NOT NULL CHECK (result IN ('applied', 'no_op', 'conflict')),
    result_control_generation INTEGER NOT NULL CHECK (result_control_generation >= 0),
    control_id TEXT,
    control_record_hash TEXT,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('all', 'pool')),
    pool_id TEXT CHECK (
        pool_id IS NULL OR length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    operation_kind TEXT NOT NULL CHECK (
        operation_kind IN ('set_force_anchor', 'set_paused')
    ),
    requested_value INTEGER NOT NULL CHECK (requested_value IN (0, 1)),
    expected_control_generation INTEGER NOT NULL CHECK (
        expected_control_generation >= 0
    ),
    actor TEXT NOT NULL CHECK (
        length(CAST(actor AS BLOB)) BETWEEN 1 AND 128 AND trim(actor) <> ''
    ),
    reason TEXT NOT NULL CHECK (
        length(CAST(reason AS BLOB)) BETWEEN 1 AND 512 AND trim(reason) <> ''
    ),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    CHECK (
        (scope_kind = 'all' AND pool_id IS NULL)
        OR (scope_kind = 'pool' AND pool_id IS NOT NULL)
    ),
    CHECK (
        (result = 'applied'
            AND control_id = mutation_id
            AND control_record_hash IS NOT NULL)
        OR (result IN ('no_op', 'conflict')
            AND control_id IS NULL
            AND control_record_hash IS NULL)
    ),
    CHECK (
        control_record_hash IS NULL OR (
            length(control_record_hash) = 64
            AND control_record_hash NOT GLOB '*[^0-9a-f]*'
        )
    ),
    UNIQUE (project_uuid, history_ordinal),
    UNIQUE (project_uuid, chain_tip_hash)
) STRICT;

CREATE INDEX idx_control_mutation_receipts_created
    ON control_mutation_receipts(project_uuid, created_at_unix_ms, history_ordinal);

CREATE TABLE control_history_checkpoints (
    checkpoint_hash TEXT PRIMARY KEY CHECK (
        length(checkpoint_hash) = 64
        AND checkpoint_hash NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    level INTEGER NOT NULL CHECK (level BETWEEN 0 AND 2147483647),
    first_history_ordinal INTEGER NOT NULL CHECK (first_history_ordinal > 0),
    last_history_ordinal INTEGER NOT NULL CHECK (
        last_history_ordinal >= first_history_ordinal
    ),
    first_predecessor_hash TEXT NOT NULL CHECK (
        length(first_predecessor_hash) = 64
        AND first_predecessor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    covered_chain_tip_hash TEXT NOT NULL CHECK (
        length(covered_chain_tip_hash) = 64
        AND covered_chain_tip_hash NOT GLOB '*[^0-9a-f]*'
    ),
    receipt_count INTEGER NOT NULL CHECK (receipt_count > 0),
    applied_control_count INTEGER NOT NULL CHECK (applied_control_count >= 0),
    range_hash TEXT NOT NULL CHECK (
        length(range_hash) = 64 AND range_hash NOT GLOB '*[^0-9a-f]*'
    ),
    ending_states_json TEXT NOT NULL CHECK (
        json_valid(ending_states_json)
        AND length(CAST(ending_states_json AS BLOB)) BETWEEN 2 AND 33554432
    ),
    ending_states_hash TEXT NOT NULL CHECK (
        length(ending_states_hash) = 64
        AND ending_states_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = checkpoint_hash
    ),
    UNIQUE (project_uuid, first_history_ordinal, last_history_ordinal, level),
    UNIQUE (project_uuid, covered_chain_tip_hash)
) STRICT;

CREATE INDEX idx_control_history_checkpoints_order
    ON control_history_checkpoints(
        project_uuid, first_history_ordinal, last_history_ordinal, level, checkpoint_hash
    );

CREATE TABLE active_experiments (
    active_experiment_id TEXT PRIMARY KEY CHECK (length(active_experiment_id) = 36),
    experiment_shape_version INTEGER NOT NULL CHECK (experiment_shape_version = 1),
    project_uuid TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (
        length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    candidate_id TEXT NOT NULL CHECK (
        length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    partition_hash TEXT NOT NULL CHECK (
        length(partition_hash) = 64 AND partition_hash NOT GLOB '*[^0-9a-f]*'
    ),
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    outcome_policy_hash TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT NOT NULL,
    vector_space_id TEXT NOT NULL,
    control_generation INTEGER NOT NULL CHECK (control_generation >= 0),
    assignment_algorithm_id TEXT NOT NULL CHECK (
        assignment_algorithm_id = 'cohort_assignment_v1'
    ),
    active_math_algorithm_id_sha256 TEXT NOT NULL CHECK (
        length(active_math_algorithm_id_sha256) = 64
        AND active_math_algorithm_id_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    outcome_evaluation_batch_size INTEGER NOT NULL CHECK (
        outcome_evaluation_batch_size BETWEEN 32 AND 65536
    ),
    max_canary_roots INTEGER NOT NULL CHECK (
        max_canary_roots BETWEEN outcome_evaluation_batch_size AND 1048576
    ),
    max_looks INTEGER NOT NULL CHECK (max_looks BETWEEN 1 AND 256),
    min_treatment_roots INTEGER NOT NULL CHECK (
        min_treatment_roots BETWEEN 32 AND 1048576
    ),
    min_control_roots INTEGER NOT NULL CHECK (
        min_control_roots BETWEEN 32 AND 1048576
    ),
    min_treatment_effective_weight REAL NOT NULL CHECK (
        min_treatment_effective_weight > 0.0
    ),
    min_treatment_effective_weight_bits INTEGER NOT NULL,
    min_control_effective_weight REAL NOT NULL CHECK (
        min_control_effective_weight > 0.0
    ),
    min_control_effective_weight_bits INTEGER NOT NULL,
    noninferiority_margin REAL NOT NULL CHECK (
        noninferiority_margin BETWEEN 0.0 AND 0.25
    ),
    noninferiority_margin_bits INTEGER NOT NULL,
    noninferiority_probability REAL NOT NULL CHECK (
        noninferiority_probability >= 0.99 AND noninferiority_probability < 1.0
    ),
    noninferiority_probability_bits INTEGER NOT NULL,
    rollback_probability REAL NOT NULL CHECK (
        rollback_probability >= 0.95 AND rollback_probability < 1.0
    ),
    rollback_probability_bits INTEGER NOT NULL,
    promotion_lower_bound REAL NOT NULL CHECK (
        promotion_lower_bound BETWEEN 0.0 AND 1.0
    ),
    promotion_lower_bound_bits INTEGER NOT NULL,
    retention_lower_bound REAL NOT NULL CHECK (
        retention_lower_bound >= 0.0 AND retention_lower_bound < promotion_lower_bound
    ),
    retention_lower_bound_bits INTEGER NOT NULL,
    holdout_probability REAL NOT NULL CHECK (
        holdout_probability > 0.0 AND holdout_probability <= 0.25
    ),
    holdout_probability_bits INTEGER NOT NULL,
    active_canary_fraction REAL NOT NULL CHECK (
        active_canary_fraction > 0.0 AND active_canary_fraction < 1.0
    ),
    active_canary_fraction_bits INTEGER NOT NULL,
    actual_outcome_half_life_seconds INTEGER NOT NULL CHECK (
        actual_outcome_half_life_seconds BETWEEN 1 AND 31536000
    ),
    anchor_shadow_half_life_seconds INTEGER NOT NULL CHECK (
        anchor_shadow_half_life_seconds BETWEEN 1 AND 31536000
    ),
    authorization_ttl_seconds INTEGER NOT NULL CHECK (
        authorization_ttl_seconds BETWEEN 1 AND 31536000
    ),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, config_generation_id)
        REFERENCES config_generations(project_uuid, config_generation_id),
    FOREIGN KEY (project_uuid, pool_id, policy_version_id)
        REFERENCES policy_versions(project_uuid, pool_id, policy_version_id),
    FOREIGN KEY (project_uuid, pool_id, outcome_policy_hash)
        REFERENCES outcome_policy_versions(project_uuid, pool_id, outcome_policy_hash),
    FOREIGN KEY (project_uuid, pool_id, learning_generation_id)
        REFERENCES learning_generations(project_uuid, pool_id, learning_generation_id),
    FOREIGN KEY (project_uuid, cohort_generation_id)
        REFERENCES cohort_generations(project_uuid, cohort_generation_id),
    FOREIGN KEY (
        project_uuid, config_generation_id, pool_id, policy_version_id, vector_space_id
    ) REFERENCES pool_vector_space_mappings(
        project_uuid, config_generation_id, pool_id, policy_version_id, vector_space_id
    ),
    FOREIGN KEY (project_uuid, process_instance_id, config_generation_id)
        REFERENCES process_instances(project_uuid, process_instance_id, config_generation_id),
    CHECK (holdout_probability + active_canary_fraction < 1.0),
    CHECK (min_treatment_roots <= max_canary_roots),
    CHECK (min_control_roots <= max_canary_roots),
    CHECK (max_canary_roots <= max_looks * outcome_evaluation_batch_size),
    CHECK (
        max_canary_roots > (max_looks - 1) * outcome_evaluation_batch_size
    ),
    UNIQUE (
        pool_id, candidate_id, partition_hash, learning_generation_id,
        config_generation_id, cohort_generation_id, outcome_policy_hash
    ),
    UNIQUE (
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    )
) STRICT;

CREATE INDEX idx_active_experiments_generation
    ON active_experiments(
        project_uuid, config_generation_id, pool_id, learning_generation_id,
        cohort_generation_id, active_experiment_id
    );

CREATE TABLE active_experiment_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_experiment_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_experiment_state_event_id) = 36
    ),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    state TEXT NOT NULL CHECK (
        state IN ('collecting', 'cap_draining', 'terminal')
    ),
    terminal_reason TEXT CHECK (
        terminal_reason IS NULL OR terminal_reason IN (
            'rollback', 'invalidated', 'exhausted', 'closed_passed', 'superseded'
        )
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (
        (state = 'terminal' AND terminal_reason IS NOT NULL)
        OR (state <> 'terminal' AND terminal_reason IS NULL)
    ),
    UNIQUE (active_experiment_id, active_experiment_state_event_id)
) STRICT;

CREATE INDEX idx_active_experiment_state_latest
    ON active_experiment_state_events(active_experiment_id, event_seq DESC);
CREATE UNIQUE INDEX uq_active_experiment_terminal_state
    ON active_experiment_state_events(active_experiment_id)
    WHERE state = 'terminal';

CREATE TABLE active_experiment_tranches (
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    tranche_ordinal INTEGER NOT NULL CHECK (tranche_ordinal >= 1),
    nonholdout_limit INTEGER NOT NULL CHECK (nonholdout_limit >= 1),
    total_limit INTEGER NOT NULL CHECK (
        total_limit >= nonholdout_limit AND total_limit <= 2 * nonholdout_limit
    ),
    opened_by_process_instance_id TEXT NOT NULL
        REFERENCES process_instances(process_instance_id),
    opened_at_unix_ms INTEGER NOT NULL CHECK (opened_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    PRIMARY KEY (active_experiment_id, tranche_ordinal),
    UNIQUE (active_experiment_id, canonical_payload_hash)
) STRICT;

CREATE TABLE active_tranche_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_tranche_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_tranche_state_event_id) = 36
    ),
    active_experiment_id TEXT NOT NULL,
    tranche_ordinal INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('open', 'closed', 'drained', 'evaluating', 'complete')
    ),
    total_assignment_count INTEGER NOT NULL CHECK (total_assignment_count >= 0),
    nonholdout_assignment_count INTEGER NOT NULL CHECK (
        nonholdout_assignment_count BETWEEN 0 AND total_assignment_count
    ),
    unresolved_assignment_count INTEGER NOT NULL CHECK (
        unresolved_assignment_count BETWEEN 0 AND total_assignment_count
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_experiment_id, tranche_ordinal)
        REFERENCES active_experiment_tranches(active_experiment_id, tranche_ordinal),
    CHECK (state NOT IN ('drained', 'evaluating', 'complete') OR unresolved_assignment_count = 0),
    UNIQUE (active_experiment_id, tranche_ordinal, active_tranche_state_event_id)
) STRICT;

CREATE INDEX idx_active_tranche_state_latest
    ON active_tranche_state_events(
        active_experiment_id, tranche_ordinal, event_seq DESC
    );
CREATE UNIQUE INDEX uq_active_tranche_open_state
    ON active_tranche_state_events(active_experiment_id, tranche_ordinal)
    WHERE state = 'open';
CREATE UNIQUE INDEX uq_active_tranche_complete_state
    ON active_tranche_state_events(active_experiment_id, tranche_ordinal)
    WHERE state = 'complete';

CREATE TABLE active_look_claims (
    active_look_claim_id TEXT PRIMARY KEY CHECK (length(active_look_claim_id) = 36),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    boundary_tranche_ordinal INTEGER NOT NULL CHECK (boundary_tranche_ordinal >= 1),
    boundary_nonholdout_count INTEGER NOT NULL CHECK (boundary_nonholdout_count >= 1),
    expected_look_ordinal INTEGER NOT NULL CHECK (expected_look_ordinal >= 1),
    expected_prior_authorization_state_event_id TEXT NOT NULL,
    input_aggregate_hash TEXT NOT NULL CHECK (
        length(input_aggregate_hash) = 64
        AND input_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    as_of_unix_ms INTEGER NOT NULL CHECK (as_of_unix_ms >= 0),
    treatment_denominator INTEGER NOT NULL CHECK (treatment_denominator >= 0),
    control_denominator INTEGER NOT NULL CHECK (control_denominator >= 0),
    owner_process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_experiment_id, boundary_tranche_ordinal)
        REFERENCES active_experiment_tranches(active_experiment_id, tranche_ordinal),
    FOREIGN KEY (
        expected_prior_authorization_state_event_id, active_experiment_id
    ) REFERENCES active_authorization_state_events(
        active_authorization_state_event_id, active_experiment_id
    ),
    UNIQUE (active_look_claim_id, active_experiment_id)
) STRICT;

CREATE INDEX idx_active_look_claims_boundary
    ON active_look_claims(
        active_experiment_id, boundary_tranche_ordinal,
        boundary_nonholdout_count, created_at_unix_ms
    );

CREATE TABLE active_look_claim_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_look_claim_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_look_claim_state_event_id) = 36
    ),
    active_look_claim_id TEXT NOT NULL,
    active_experiment_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (
        state IN ('claimed', 'renewed', 'committed', 'failed', 'released', 'reclaimed', 'cancelled')
    ),
    lease_token_hash TEXT CHECK (
        lease_token_hash IS NULL OR (
            length(lease_token_hash) = 64
            AND lease_token_hash NOT GLOB '*[^0-9a-f]*'
        )
    ),
    lease_expires_at_unix_ms INTEGER CHECK (lease_expires_at_unix_ms >= 0),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_look_claim_id, active_experiment_id)
        REFERENCES active_look_claims(active_look_claim_id, active_experiment_id),
    CHECK (
        (state IN ('claimed', 'renewed', 'reclaimed')
            AND lease_token_hash IS NOT NULL
            AND lease_expires_at_unix_ms IS NOT NULL
            AND lease_expires_at_unix_ms - created_at_unix_ms BETWEEN 1 AND 30000)
        OR (state IN ('committed', 'failed', 'released', 'cancelled')
            AND lease_token_hash IS NULL
            AND lease_expires_at_unix_ms IS NULL)
    )
) STRICT;

CREATE INDEX idx_active_look_claim_state_latest
    ON active_look_claim_state_events(active_look_claim_id, event_seq DESC);
CREATE UNIQUE INDEX uq_active_look_claim_terminal_state
    ON active_look_claim_state_events(active_look_claim_id)
    WHERE state IN ('committed', 'failed', 'released', 'cancelled');

CREATE TABLE active_outcome_looks (
    active_outcome_look_id TEXT PRIMARY KEY CHECK (length(active_outcome_look_id) = 36),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    active_look_claim_id TEXT NOT NULL UNIQUE,
    look_ordinal INTEGER NOT NULL CHECK (look_ordinal >= 1),
    boundary_tranche_ordinal INTEGER NOT NULL CHECK (boundary_tranche_ordinal >= 1),
    boundary_nonholdout_count INTEGER NOT NULL CHECK (boundary_nonholdout_count >= 1),
    as_of_unix_ms INTEGER NOT NULL CHECK (as_of_unix_ms >= 0),
    treatment_denominator INTEGER NOT NULL CHECK (treatment_denominator > 0),
    treatment_labeled INTEGER NOT NULL CHECK (
        treatment_labeled BETWEEN 0 AND treatment_denominator
    ),
    control_denominator INTEGER NOT NULL CHECK (control_denominator > 0),
    control_labeled INTEGER NOT NULL CHECK (
        control_labeled BETWEEN 0 AND control_denominator
    ),
    input_aggregate_hash TEXT NOT NULL CHECK (
        length(input_aggregate_hash) = 64
        AND input_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    result_state TEXT NOT NULL CHECK (
        result_state IN ('collecting', 'passed', 'rollback')
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_look_claim_id, active_experiment_id)
        REFERENCES active_look_claims(active_look_claim_id, active_experiment_id),
    FOREIGN KEY (active_experiment_id, boundary_tranche_ordinal)
        REFERENCES active_experiment_tranches(active_experiment_id, tranche_ordinal),
    UNIQUE (active_experiment_id, look_ordinal),
    UNIQUE (
        active_experiment_id, boundary_tranche_ordinal, boundary_nonholdout_count
    ),
    UNIQUE (active_outcome_look_id, active_experiment_id)
) STRICT;

CREATE INDEX idx_active_outcome_looks_boundary
    ON active_outcome_looks(
        active_experiment_id, boundary_tranche_ordinal, boundary_nonholdout_count
    );

CREATE TABLE active_outcome_look_audits (
    active_outcome_look_id TEXT PRIMARY KEY
        REFERENCES active_outcome_looks(active_outcome_look_id),
    audit_shape_version INTEGER NOT NULL CHECK (audit_shape_version = 1),
    active_math_build_id TEXT NOT NULL CHECK (
        length(active_math_build_id) = 64
        AND active_math_build_id NOT GLOB '*[^0-9a-f]*'
    ),
    active_math_algorithm_id_sha256 TEXT NOT NULL CHECK (
        length(active_math_algorithm_id_sha256) = 64
        AND active_math_algorithm_id_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    input_identity_sha256 TEXT NOT NULL CHECK (
        length(input_identity_sha256) = 64
        AND input_identity_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    actual_outcome_half_life_seconds INTEGER NOT NULL CHECK (
        actual_outcome_half_life_seconds BETWEEN 1 AND 31536000
    ),
    min_treatment_roots INTEGER NOT NULL CHECK (min_treatment_roots >= 32),
    min_control_roots INTEGER NOT NULL CHECK (min_control_roots >= 32),
    min_treatment_effective_weight REAL NOT NULL CHECK (min_treatment_effective_weight > 0.0),
    min_treatment_effective_weight_bits INTEGER NOT NULL,
    min_control_effective_weight REAL NOT NULL CHECK (min_control_effective_weight > 0.0),
    min_control_effective_weight_bits INTEGER NOT NULL,
    noninferiority_margin REAL NOT NULL CHECK (noninferiority_margin BETWEEN 0.0 AND 0.25),
    noninferiority_margin_bits INTEGER NOT NULL,
    noninferiority_probability REAL NOT NULL CHECK (
        noninferiority_probability >= 0.99 AND noninferiority_probability < 1.0
    ),
    noninferiority_probability_bits INTEGER NOT NULL,
    per_look_noninferiority_threshold REAL NOT NULL CHECK (
        per_look_noninferiority_threshold > 0.0
        AND per_look_noninferiority_threshold < 1.0
    ),
    per_look_noninferiority_threshold_bits INTEGER NOT NULL,
    rollback_probability REAL NOT NULL CHECK (
        rollback_probability >= 0.95 AND rollback_probability < 1.0
    ),
    rollback_probability_bits INTEGER NOT NULL,
    max_looks INTEGER NOT NULL CHECK (max_looks BETWEEN 1 AND 256),
    treatment_denominator INTEGER NOT NULL CHECK (treatment_denominator > 0),
    treatment_labeled INTEGER NOT NULL CHECK (treatment_labeled BETWEEN 0 AND treatment_denominator),
    treatment_successes INTEGER NOT NULL CHECK (treatment_successes >= 0),
    treatment_failures INTEGER NOT NULL CHECK (treatment_failures >= 0),
    treatment_success_weight REAL NOT NULL CHECK (treatment_success_weight >= 0.0),
    treatment_success_weight_bits INTEGER NOT NULL,
    treatment_failure_weight REAL NOT NULL CHECK (treatment_failure_weight >= 0.0),
    treatment_failure_weight_bits INTEGER NOT NULL,
    treatment_effective_weight REAL NOT NULL CHECK (treatment_effective_weight >= 0.0),
    treatment_effective_weight_bits INTEGER NOT NULL,
    treatment_beta_alpha REAL NOT NULL CHECK (treatment_beta_alpha > 0.0),
    treatment_beta_alpha_bits INTEGER NOT NULL,
    treatment_beta_beta REAL NOT NULL CHECK (treatment_beta_beta > 0.0),
    treatment_beta_beta_bits INTEGER NOT NULL,
    treatment_label_rate REAL NOT NULL CHECK (treatment_label_rate BETWEEN 0.0 AND 1.0),
    treatment_label_rate_bits INTEGER NOT NULL,
    control_denominator INTEGER NOT NULL CHECK (control_denominator > 0),
    control_labeled INTEGER NOT NULL CHECK (control_labeled BETWEEN 0 AND control_denominator),
    control_successes INTEGER NOT NULL CHECK (control_successes >= 0),
    control_failures INTEGER NOT NULL CHECK (control_failures >= 0),
    control_success_weight REAL NOT NULL CHECK (control_success_weight >= 0.0),
    control_success_weight_bits INTEGER NOT NULL,
    control_failure_weight REAL NOT NULL CHECK (control_failure_weight >= 0.0),
    control_failure_weight_bits INTEGER NOT NULL,
    control_effective_weight REAL NOT NULL CHECK (control_effective_weight >= 0.0),
    control_effective_weight_bits INTEGER NOT NULL,
    control_beta_alpha REAL NOT NULL CHECK (control_beta_alpha > 0.0),
    control_beta_alpha_bits INTEGER NOT NULL,
    control_beta_beta REAL NOT NULL CHECK (control_beta_beta > 0.0),
    control_beta_beta_bits INTEGER NOT NULL,
    control_label_rate REAL NOT NULL CHECK (control_label_rate BETWEEN 0.0 AND 1.0),
    control_label_rate_bits INTEGER NOT NULL,
    noninferiority_g_lower REAL NOT NULL CHECK (noninferiority_g_lower BETWEEN 0.0 AND 1.0),
    noninferiority_g_lower_bits INTEGER NOT NULL,
    noninferiority_g_upper REAL NOT NULL CHECK (noninferiority_g_upper BETWEEN 0.0 AND 1.0),
    noninferiority_g_upper_bits INTEGER NOT NULL,
    noninferiority_h_lower REAL NOT NULL CHECK (noninferiority_h_lower BETWEEN 0.0 AND 1.0),
    noninferiority_h_lower_bits INTEGER NOT NULL,
    noninferiority_h_upper REAL NOT NULL CHECK (noninferiority_h_upper BETWEEN 0.0 AND 1.0),
    noninferiority_h_upper_bits INTEGER NOT NULL,
    noninferiority_lower REAL NOT NULL CHECK (noninferiority_lower BETWEEN 0.0 AND 1.0),
    noninferiority_lower_bits INTEGER NOT NULL,
    noninferiority_upper REAL NOT NULL CHECK (noninferiority_upper BETWEEN 0.0 AND 1.0),
    noninferiority_upper_bits INTEGER NOT NULL,
    rollback_lower REAL NOT NULL CHECK (rollback_lower BETWEEN 0.0 AND 1.0),
    rollback_lower_bits INTEGER NOT NULL,
    treatment_raw_roots_gate INTEGER NOT NULL CHECK (treatment_raw_roots_gate IN (0, 1)),
    control_raw_roots_gate INTEGER NOT NULL CHECK (control_raw_roots_gate IN (0, 1)),
    treatment_effective_weight_gate INTEGER NOT NULL CHECK (treatment_effective_weight_gate IN (0, 1)),
    control_effective_weight_gate INTEGER NOT NULL CHECK (control_effective_weight_gate IN (0, 1)),
    treatment_attribution_gate INTEGER NOT NULL CHECK (treatment_attribution_gate IN (0, 1)),
    control_attribution_gate INTEGER NOT NULL CHECK (control_attribution_gate IN (0, 1)),
    differential_attribution_gate INTEGER NOT NULL CHECK (differential_attribution_gate IN (0, 1)),
    noninferiority_gate INTEGER NOT NULL CHECK (noninferiority_gate IN (0, 1)),
    rollback_gate INTEGER NOT NULL CHECK (rollback_gate IN (0, 1)),
    result_state TEXT NOT NULL CHECK (result_state IN ('collecting', 'passed', 'rollback')),
    raw_beta_cdf_calls INTEGER NOT NULL CHECK (raw_beta_cdf_calls BETWEEN 0 AND 2200000),
    operational_cdf_queries INTEGER NOT NULL CHECK (operational_cdf_queries BETWEEN 0 AND 1100000),
    max_symmetry_disagreement REAL NOT NULL CHECK (max_symmetry_disagreement >= 0.0),
    max_symmetry_disagreement_bits INTEGER NOT NULL,
    max_monotonic_repair REAL NOT NULL CHECK (
        max_monotonic_repair >= 0.0 AND max_monotonic_repair <= 0.00000000001
    ),
    max_monotonic_repair_bits INTEGER NOT NULL,
    max_quantile_width REAL NOT NULL CHECK (
        max_quantile_width >= 0.0 AND max_quantile_width <= 0.000244144625
    ),
    max_quantile_width_bits INTEGER NOT NULL,
    cdf_transcript_sha256 TEXT NOT NULL CHECK (
        length(cdf_transcript_sha256) = 64
        AND cdf_transcript_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_audit_json TEXT NOT NULL CHECK (
        json_valid(canonical_audit_json)
        AND length(CAST(canonical_audit_json AS BLOB)) BETWEEN 2 AND 1048576
    ),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (treatment_successes + treatment_failures = treatment_labeled),
    CHECK (control_successes + control_failures = control_labeled),
    CHECK (noninferiority_g_lower <= noninferiority_g_upper),
    CHECK (noninferiority_h_lower <= noninferiority_h_upper),
    CHECK (noninferiority_lower <= noninferiority_upper)
) STRICT;

CREATE TABLE active_look_failures (
    active_look_failure_id TEXT PRIMARY KEY CHECK (length(active_look_failure_id) = 36),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    active_look_claim_id TEXT,
    boundary_tranche_ordinal INTEGER NOT NULL CHECK (boundary_tranche_ordinal >= 1),
    boundary_nonholdout_count INTEGER NOT NULL CHECK (boundary_nonholdout_count >= 1),
    failure_kind TEXT NOT NULL CHECK (
        failure_kind IN ('skipped', 'numeric_failure', 'integrity_failure', 'claim_lost')
    ),
    stable_reason TEXT NOT NULL CHECK (
        length(CAST(stable_reason AS BLOB)) BETWEEN 1 AND 128
    ),
    input_aggregate_hash TEXT NOT NULL CHECK (
        length(input_aggregate_hash) = 64
        AND input_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_look_claim_id, active_experiment_id)
        REFERENCES active_look_claims(active_look_claim_id, active_experiment_id),
    FOREIGN KEY (active_experiment_id, boundary_tranche_ordinal)
        REFERENCES active_experiment_tranches(active_experiment_id, tranche_ordinal),
    CHECK (
        (failure_kind = 'skipped' AND active_look_claim_id IS NULL)
        OR (failure_kind <> 'skipped' AND active_look_claim_id IS NOT NULL)
    ),
    UNIQUE (
        active_experiment_id, boundary_tranche_ordinal,
        boundary_nonholdout_count, input_aggregate_hash, failure_kind
    )
) STRICT;

CREATE INDEX idx_active_look_failures_boundary
    ON active_look_failures(
        active_experiment_id, boundary_tranche_ordinal, boundary_nonholdout_count
    );

CREATE TABLE active_authorization_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_authorization_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_authorization_state_event_id) = 36
    ),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    predecessor_authorization_state_event_id TEXT,
    active_outcome_look_id TEXT,
    active_look_claim_id TEXT,
    state TEXT NOT NULL CHECK (
        state IN (
            'collecting', 'evaluating', 'passed', 'rollback', 'expired',
            'invalidated', 'superseded_draining', 'superseded',
            'closed_passed', 'exhausted'
        )
    ),
    valid_until_unix_ms INTEGER CHECK (
        valid_until_unix_ms IS NULL OR valid_until_unix_ms >= 0
    ),
    control_generation INTEGER NOT NULL CHECK (control_generation >= 0),
    transition_identity_hash TEXT NOT NULL CHECK (
        length(transition_identity_hash) = 64
        AND transition_identity_hash NOT GLOB '*[^0-9a-f]*'
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (
        predecessor_authorization_state_event_id, active_experiment_id
    ) REFERENCES active_authorization_state_events(
        active_authorization_state_event_id, active_experiment_id
    ),
    FOREIGN KEY (active_outcome_look_id, active_experiment_id)
        REFERENCES active_outcome_looks(active_outcome_look_id, active_experiment_id),
    FOREIGN KEY (active_look_claim_id, active_experiment_id)
        REFERENCES active_look_claims(active_look_claim_id, active_experiment_id),
    CHECK (
        (state = 'passed'
            AND active_outcome_look_id IS NOT NULL
            AND active_look_claim_id IS NULL
            AND valid_until_unix_ms IS NOT NULL
            AND valid_until_unix_ms > created_at_unix_ms)
        OR (state <> 'passed' AND valid_until_unix_ms IS NULL)
    ),
    CHECK (
        (state IN ('rollback') AND active_outcome_look_id IS NOT NULL)
        OR state <> 'rollback'
    ),
    CHECK (
        (state = 'evaluating'
            AND active_look_claim_id IS NOT NULL
            AND active_outcome_look_id IS NULL)
        OR (state <> 'evaluating' AND active_look_claim_id IS NULL)
    ),
    CHECK (
        state IN ('collecting', 'passed', 'rollback', 'evaluating')
        OR active_outcome_look_id IS NULL
    ),
    UNIQUE (active_authorization_state_event_id, active_experiment_id),
    UNIQUE (active_experiment_id, transition_identity_hash)
) STRICT;

CREATE INDEX idx_active_authorization_state_latest
    ON active_authorization_state_events(active_experiment_id, event_seq DESC);
CREATE UNIQUE INDEX uq_active_authorization_look_state
    ON active_authorization_state_events(active_outcome_look_id)
    WHERE active_outcome_look_id IS NOT NULL;
CREATE UNIQUE INDEX uq_active_authorization_claim_state
    ON active_authorization_state_events(active_look_claim_id)
    WHERE active_look_claim_id IS NOT NULL;

-- Keep every v5 column in place. Active shape 2 uses the same complete query
-- audit and stores active-only facts one-to-one below.
CREATE TABLE decisions (
    decision_id TEXT PRIMARY KEY CHECK (length(decision_id) = 36),
    decision_shape_version INTEGER NOT NULL CHECK (decision_shape_version IN (1, 2)),
    algorithm_version INTEGER NOT NULL CHECK (algorithm_version IN (1, 2)),
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT,
    active_experiment_id TEXT,
    active_authorization_state_event_id TEXT,
    pool_id TEXT NOT NULL CHECK (
        length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    candidate_id TEXT CHECK (
        candidate_id IS NULL
        OR length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    root_key TEXT CHECK (
        root_key IS NULL OR (
            length(root_key) = 64 AND root_key NOT GLOB '*[^0-9a-f]*'
        )
    ),
    primary_call_uuid TEXT NOT NULL CHECK (length(primary_call_uuid) = 36),
    mode TEXT NOT NULL CHECK (mode IN ('recommend', 'active')),
    canonical_query_hash TEXT NOT NULL
        REFERENCES canonical_routing_queries(canonical_query_hash),
    partition_base_json TEXT NOT NULL CHECK (json_valid(partition_base_json)),
    partition_base_hash TEXT NOT NULL CHECK (
        length(partition_base_hash) = 64
        AND partition_base_hash NOT GLOB '*[^0-9a-f]*'
    ),
    vector_space_id TEXT NOT NULL REFERENCES vector_spaces(vector_space_id),
    candidate_set_hash TEXT NOT NULL CHECK (
        length(candidate_set_hash) = 64
        AND candidate_set_hash NOT GLOB '*[^0-9a-f]*'
    ),
    candidate_count INTEGER NOT NULL CHECK (candidate_count BETWEEN 1 AND 64),
    recommended_model TEXT NOT NULL CHECK (
        length(CAST(recommended_model AS BLOB)) BETWEEN 1 AND 512
    ),
    recommended_model_revision TEXT NOT NULL CHECK (
        length(CAST(recommended_model_revision AS BLOB)) BETWEEN 1 AND 128
    ),
    served_model TEXT NOT NULL CHECK (
        length(CAST(served_model AS BLOB)) BETWEEN 1 AND 512
    ),
    served_model_revision TEXT NOT NULL CHECK (
        length(CAST(served_model_revision AS BLOB)) BETWEEN 1 AND 128
    ),
    as_of_unix_ms INTEGER NOT NULL CHECK (as_of_unix_ms >= 0),
    decision_latency_ms INTEGER NOT NULL CHECK (decision_latency_ms >= 0),
    final_reason TEXT NOT NULL CHECK (
        final_reason IN (
            'embedding_unavailable', 'vector_unhealthy', 'version_mismatch',
            'no_partition', 'sparse_points', 'insufficient_roots', 'low_coverage',
            'insufficient_effective_samples', 'lower_bound_below_threshold',
            'invalid_evidence_time', 'numeric_error', 'no_candidate_passed',
            'recommend_observe_only', 'active_candidate', 'active_anchor_control',
            'active_anchor_holdout', 'active_force_anchor', 'active_paused',
            'active_ineligible', 'active_exhausted', 'active_storage_fallback',
            'active_authorization_stale', 'active_cap_reached'
        )
    ),
    summary_count INTEGER NOT NULL CHECK (summary_count BETWEEN 1 AND 64),
    summary_aggregate_hash TEXT NOT NULL CHECK (
        length(summary_aggregate_hash) = 64
        AND summary_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    neighbor_count INTEGER NOT NULL CHECK (neighbor_count BETWEEN 0 AND 4095),
    neighbor_aggregate_hash TEXT NOT NULL CHECK (
        length(neighbor_aggregate_hash) = 64
        AND neighbor_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    aggregate_size_bytes INTEGER NOT NULL CHECK (
        aggregate_size_bytes BETWEEN 1 AND 33554432
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
    FOREIGN KEY (
        project_uuid, config_generation_id, pool_id, policy_version_id, vector_space_id
    ) REFERENCES pool_vector_space_mappings(
        project_uuid, config_generation_id, pool_id, policy_version_id, vector_space_id
    ),
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
    CHECK (summary_count = candidate_count),
    CHECK (
        (decision_shape_version = 1 AND algorithm_version = 1 AND mode = 'recommend')
        OR (decision_shape_version = 2 AND algorithm_version = 2 AND mode = 'active')
    ),
    CHECK (
        decision_shape_version = 2 OR (
            (candidate_id IS NULL
                AND final_reason <> 'recommend_observe_only'
                AND recommended_model = served_model
                AND recommended_model_revision = served_model_revision)
            OR (candidate_id IS NOT NULL AND final_reason = 'recommend_observe_only')
        )
    ),
    CHECK (
        decision_shape_version = 1 OR final_reason IN (
            'active_candidate', 'active_anchor_control', 'active_anchor_holdout',
            'active_force_anchor', 'active_paused', 'active_ineligible',
            'active_exhausted', 'active_storage_fallback',
            'active_authorization_stale', 'active_cap_reached'
        )
    ),
    CHECK (
        (active_experiment_id IS NULL AND active_authorization_state_event_id IS NULL)
        OR (active_experiment_id IS NOT NULL AND active_authorization_state_event_id IS NOT NULL
            AND cohort_generation_id IS NOT NULL AND candidate_id IS NOT NULL)
    ),
    CHECK (
        decision_shape_version = 2
        OR (active_experiment_id IS NULL AND active_authorization_state_event_id IS NULL)
    ),
    UNIQUE (decision_id, project_uuid),
    UNIQUE (decision_id, vector_space_id),
    UNIQUE (decision_id, decision_shape_version),
    UNIQUE (project_uuid, primary_call_uuid)
) STRICT;

CREATE TABLE decision_candidate_summaries (
    decision_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL CHECK (
        length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    rank_ordinal INTEGER NOT NULL CHECK (rank_ordinal BETWEEN 0 AND 63),
    candidate_model TEXT NOT NULL CHECK (
        length(CAST(candidate_model AS BLOB)) BETWEEN 1 AND 512
    ),
    candidate_model_revision TEXT NOT NULL CHECK (
        length(CAST(candidate_model_revision AS BLOB)) BETWEEN 1 AND 128
    ),
    cost_rank INTEGER NOT NULL CHECK (cost_rank >= 0),
    learning_generation_id TEXT NOT NULL CHECK (length(learning_generation_id) = 36),
    vector_space_id TEXT NOT NULL,
    partition_hash TEXT NOT NULL CHECK (
        length(partition_hash) = 64 AND partition_hash NOT GLOB '*[^0-9a-f]*'
    ),
    partition_id INTEGER,
    decoding_fingerprint TEXT NOT NULL CHECK (
        length(decoding_fingerprint) = 64
        AND decoding_fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    top_k INTEGER NOT NULL CHECK (top_k BETWEEN 1 AND 4095),
    radius REAL NOT NULL CHECK (radius > 0.0 AND radius <= 2.0),
    radius_bits INTEGER NOT NULL,
    min_points INTEGER NOT NULL CHECK (min_points >= 1 AND min_points <= top_k),
    min_independent_roots INTEGER NOT NULL CHECK (
        min_independent_roots >= 1 AND min_independent_roots <= top_k
    ),
    min_effective_samples REAL NOT NULL CHECK (
        min_effective_samples > 0.0 AND min_effective_samples <= CAST(top_k AS REAL)
    ),
    min_effective_samples_bits INTEGER NOT NULL,
    min_coverage REAL NOT NULL CHECK (min_coverage BETWEEN 0.0 AND 1.0),
    min_coverage_bits INTEGER NOT NULL,
    time_decay_half_life_seconds REAL NOT NULL CHECK (
        time_decay_half_life_seconds > 0.0
    ),
    time_decay_half_life_seconds_bits INTEGER NOT NULL,
    prior_success REAL NOT NULL CHECK (prior_success > 0.0),
    prior_success_bits INTEGER NOT NULL,
    prior_failure REAL NOT NULL CHECK (prior_failure > 0.0),
    prior_failure_bits INTEGER NOT NULL,
    familywise_credible_level REAL NOT NULL CHECK (
        familywise_credible_level > 0.5 AND familywise_credible_level < 1.0
    ),
    familywise_credible_level_bits INTEGER NOT NULL,
    candidate_alpha REAL NOT NULL CHECK (candidate_alpha > 0.0 AND candidate_alpha < 0.5),
    candidate_alpha_bits INTEGER NOT NULL,
    promotion_lower_bound REAL NOT NULL CHECK (promotion_lower_bound BETWEEN 0.0 AND 1.0),
    promotion_lower_bound_bits INTEGER NOT NULL,
    returned_neighbor_count INTEGER NOT NULL CHECK (
        returned_neighbor_count >= 0 AND returned_neighbor_count <= top_k
    ),
    within_radius_count INTEGER NOT NULL CHECK (
        within_radius_count >= 0 AND within_radius_count <= returned_neighbor_count
    ),
    labeled_point_count INTEGER NOT NULL CHECK (
        labeled_point_count >= 0 AND labeled_point_count <= within_radius_count
    ),
    attempted_root_count INTEGER NOT NULL CHECK (
        attempted_root_count >= 0 AND attempted_root_count <= within_radius_count
    ),
    labeled_root_count INTEGER NOT NULL CHECK (
        labeled_root_count >= 0
        AND labeled_root_count <= labeled_point_count
        AND labeled_root_count <= attempted_root_count
    ),
    selected_root_count INTEGER NOT NULL CHECK (
        selected_root_count >= 0 AND selected_root_count <= labeled_root_count
    ),
    coverage REAL CHECK (coverage IS NULL OR coverage BETWEEN 0.0 AND 1.0),
    coverage_bits INTEGER,
    sum_weight REAL CHECK (sum_weight IS NULL OR sum_weight >= 0.0),
    sum_weight_bits INTEGER,
    sum_weighted_label REAL CHECK (
        sum_weighted_label IS NULL OR sum_weighted_label >= 0.0
    ),
    sum_weighted_label_bits INTEGER,
    sum_squared_weight REAL CHECK (
        sum_squared_weight IS NULL OR sum_squared_weight >= 0.0
    ),
    sum_squared_weight_bits INTEGER,
    p_hat REAL CHECK (p_hat IS NULL OR p_hat BETWEEN 0.0 AND 1.0),
    p_hat_bits INTEGER,
    effective_sample_size REAL CHECK (
        effective_sample_size IS NULL OR effective_sample_size >= 0.0
    ),
    effective_sample_size_bits INTEGER,
    beta_alpha REAL CHECK (beta_alpha IS NULL OR beta_alpha > 0.0),
    beta_alpha_bits INTEGER,
    beta_beta REAL CHECK (beta_beta IS NULL OR beta_beta > 0.0),
    beta_beta_bits INTEGER,
    lower_bound REAL CHECK (lower_bound IS NULL OR lower_bound BETWEEN 0.0 AND 1.0),
    lower_bound_bits INTEGER,
    partition_gate_passed INTEGER CHECK (
        partition_gate_passed IS NULL OR partition_gate_passed IN (0, 1)
    ),
    points_gate_passed INTEGER CHECK (
        points_gate_passed IS NULL OR points_gate_passed IN (0, 1)
    ),
    roots_gate_passed INTEGER CHECK (
        roots_gate_passed IS NULL OR roots_gate_passed IN (0, 1)
    ),
    coverage_gate_passed INTEGER CHECK (
        coverage_gate_passed IS NULL OR coverage_gate_passed IN (0, 1)
    ),
    weight_gate_passed INTEGER CHECK (
        weight_gate_passed IS NULL OR weight_gate_passed IN (0, 1)
    ),
    effective_samples_gate_passed INTEGER CHECK (
        effective_samples_gate_passed IS NULL OR effective_samples_gate_passed IN (0, 1)
    ),
    beta_quantile_gate_passed INTEGER CHECK (
        beta_quantile_gate_passed IS NULL OR beta_quantile_gate_passed IN (0, 1)
    ),
    lower_bound_gate_passed INTEGER CHECK (
        lower_bound_gate_passed IS NULL OR lower_bound_gate_passed IN (0, 1)
    ),
    terminal_reason TEXT NOT NULL CHECK (
        terminal_reason IN (
            'no_partition', 'sparse_points', 'insufficient_roots', 'low_coverage',
            'insufficient_effective_samples', 'lower_bound_below_threshold',
            'invalid_evidence_time', 'numeric_error', 'passed',
            'not_evaluated_after_winner', 'not_evaluated_after_fallback'
        )
    ),
    neighbor_count INTEGER NOT NULL CHECK (neighbor_count BETWEEN 0 AND top_k),
    neighbor_aggregate_hash TEXT NOT NULL CHECK (
        length(neighbor_aggregate_hash) = 64
        AND neighbor_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (decision_id, vector_space_id)
        REFERENCES decisions(decision_id, vector_space_id) ON DELETE CASCADE,
    FOREIGN KEY (partition_id, vector_space_id)
        REFERENCES routing_partitions(partition_id, vector_space_id),
    CHECK (returned_neighbor_count = neighbor_count),
    CHECK ((coverage IS NULL) = (coverage_bits IS NULL)),
    CHECK ((sum_weight IS NULL) = (sum_weight_bits IS NULL)),
    CHECK ((sum_weighted_label IS NULL) = (sum_weighted_label_bits IS NULL)),
    CHECK ((sum_squared_weight IS NULL) = (sum_squared_weight_bits IS NULL)),
    CHECK ((p_hat IS NULL) = (p_hat_bits IS NULL)),
    CHECK ((effective_sample_size IS NULL) = (effective_sample_size_bits IS NULL)),
    CHECK ((beta_alpha IS NULL) = (beta_alpha_bits IS NULL)),
    CHECK ((beta_beta IS NULL) = (beta_beta_bits IS NULL)),
    CHECK ((lower_bound IS NULL) = (lower_bound_bits IS NULL)),
    CHECK (
        (terminal_reason = 'no_partition'
            AND partition_id IS NULL AND partition_gate_passed = 0
            AND returned_neighbor_count = 0)
        OR (terminal_reason IN (
                'not_evaluated_after_winner', 'not_evaluated_after_fallback'
            )
            AND partition_id IS NULL AND partition_gate_passed IS NULL
            AND returned_neighbor_count = 0)
        OR (terminal_reason NOT IN (
                'no_partition', 'not_evaluated_after_winner',
                'not_evaluated_after_fallback'
            )
            AND partition_id IS NOT NULL AND partition_gate_passed = 1)
    ),
    PRIMARY KEY (decision_id, candidate_id),
    UNIQUE (decision_id, rank_ordinal)
) STRICT;

CREATE TABLE decision_neighbors (
    decision_id TEXT NOT NULL,
    neighbor_ordinal INTEGER NOT NULL CHECK (neighbor_ordinal BETWEEN 0 AND 4094),
    candidate_id TEXT NOT NULL CHECK (
        length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    candidate_neighbor_ordinal INTEGER NOT NULL CHECK (
        candidate_neighbor_ordinal BETWEEN 0 AND 4094
    ),
    evidence_vector_link_id TEXT NOT NULL
        REFERENCES evidence_vector_links(evidence_vector_link_id),
    shadow_attempt_id TEXT NOT NULL CHECK (length(shadow_attempt_id) = 36),
    anchor_id TEXT NOT NULL CHECK (length(anchor_id) = 36),
    evaluation_id TEXT CHECK (evaluation_id IS NULL OR length(evaluation_id) = 36),
    learning_generation_id TEXT NOT NULL CHECK (length(learning_generation_id) = 36),
    distance REAL NOT NULL CHECK (distance >= 0.0),
    distance_f32_bits INTEGER NOT NULL CHECK (
        distance_f32_bits BETWEEN 0 AND 4294967295
    ),
    age_seconds REAL CHECK (age_seconds IS NULL OR age_seconds >= 0.0),
    age_seconds_bits INTEGER,
    similarity_weight REAL CHECK (
        similarity_weight IS NULL OR similarity_weight BETWEEN 0.0 AND 1.0
    ),
    similarity_weight_bits INTEGER,
    time_weight REAL CHECK (time_weight IS NULL OR time_weight BETWEEN 0.0 AND 1.0),
    time_weight_bits INTEGER,
    final_weight REAL CHECK (final_weight IS NULL OR final_weight BETWEEN 0.0 AND 1.0),
    final_weight_bits INTEGER,
    binary_label TEXT CHECK (binary_label IS NULL OR binary_label IN ('pass', 'fail')),
    selected_for_root INTEGER NOT NULL CHECK (selected_for_root IN (0, 1)),
    root_group_ordinal INTEGER NOT NULL CHECK (root_group_ordinal BETWEEN 0 AND 4094),
    exclusion_reason TEXT NOT NULL CHECK (
        exclusion_reason IN (
            'outside_radius', 'ineligible_quality', 'duplicate_root', 'included'
        )
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (decision_id, candidate_id)
        REFERENCES decision_candidate_summaries(decision_id, candidate_id)
        ON DELETE CASCADE,
    FOREIGN KEY (shadow_attempt_id, anchor_id, learning_generation_id)
        REFERENCES shadow_attempts(shadow_attempt_id, anchor_id, learning_generation_id),
    FOREIGN KEY (evaluation_id, shadow_attempt_id)
        REFERENCES evaluations(evaluation_id, shadow_attempt_id),
    CHECK ((age_seconds IS NULL) = (age_seconds_bits IS NULL)),
    CHECK ((similarity_weight IS NULL) = (similarity_weight_bits IS NULL)),
    CHECK ((time_weight IS NULL) = (time_weight_bits IS NULL)),
    CHECK ((final_weight IS NULL) = (final_weight_bits IS NULL)),
    CHECK (binary_label IS NULL OR evaluation_id IS NOT NULL),
    CHECK (
        (exclusion_reason = 'included' AND selected_for_root = 1)
        OR (exclusion_reason <> 'included' AND selected_for_root = 0)
    ),
    PRIMARY KEY (decision_id, neighbor_ordinal),
    UNIQUE (decision_id, candidate_id, candidate_neighbor_ordinal),
    UNIQUE (decision_id, evidence_vector_link_id)
) STRICT;

INSERT INTO decisions SELECT * FROM spec08_v5_decisions;
INSERT INTO decision_candidate_summaries
    SELECT * FROM spec08_v5_decision_candidate_summaries;
INSERT INTO decision_neighbors SELECT * FROM spec08_v5_decision_neighbors;

DROP TABLE spec08_v5_decision_neighbors;
DROP TABLE spec08_v5_decision_candidate_summaries;
DROP TABLE spec08_v5_decisions;

CREATE INDEX idx_decisions_project_created
    ON decisions(project_uuid, created_at_unix_ms, decision_id);
CREATE INDEX idx_decisions_canonical_query
    ON decisions(canonical_query_hash, decision_id);
CREATE INDEX idx_decisions_vector_space
    ON decisions(vector_space_id, decision_id);
CREATE INDEX idx_decisions_policy_learning
    ON decisions(
        project_uuid, pool_id, policy_version_id, learning_generation_id, decision_id
    );
CREATE INDEX idx_decisions_mapping
    ON decisions(
        project_uuid, config_generation_id, pool_id,
        policy_version_id, vector_space_id, decision_id
    );
CREATE INDEX idx_decision_summaries_partition
    ON decision_candidate_summaries(partition_id, decision_id)
    WHERE partition_id IS NOT NULL;
CREATE INDEX idx_decision_neighbors_evidence_vector_link
    ON decision_neighbors(evidence_vector_link_id);
CREATE INDEX idx_decision_neighbors_anchor
    ON decision_neighbors(anchor_id, decision_id);
CREATE INDEX idx_decision_neighbors_shadow_attempt
    ON decision_neighbors(shadow_attempt_id, decision_id);
CREATE INDEX idx_decision_neighbors_evaluation
    ON decision_neighbors(evaluation_id, decision_id)
    WHERE evaluation_id IS NOT NULL;

CREATE TABLE active_root_windows (
    active_root_window_id TEXT PRIMARY KEY CHECK (length(active_root_window_id) = 36),
    active_experiment_id TEXT NOT NULL,
    project_uuid TEXT NOT NULL,
    pool_id TEXT NOT NULL CHECK (
        length(CAST(pool_id AS BLOB)) BETWEEN 1 AND 128
    ),
    candidate_id TEXT NOT NULL CHECK (
        length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    root_key TEXT NOT NULL CHECK (
        length(root_key) = 64 AND root_key NOT GLOB '*[^0-9a-f]*'
    ),
    owner_relation_hash TEXT NOT NULL CHECK (
        length(owner_relation_hash) = 64
        AND owner_relation_hash NOT GLOB '*[^0-9a-f]*'
    ),
    config_generation_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT NOT NULL,
    outcome_policy_hash TEXT NOT NULL,
    opened_after_ingest_seq INTEGER NOT NULL CHECK (opened_after_ingest_seq >= 0),
    attribution_deadline_unix_ms INTEGER NOT NULL CHECK (
        attribution_deadline_unix_ms >= 0
    ),
    process_instance_id TEXT NOT NULL,
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    ) REFERENCES active_experiments(
        active_experiment_id, project_uuid, pool_id, candidate_id,
        learning_generation_id, config_generation_id, cohort_generation_id
    ),
    FOREIGN KEY (project_uuid, process_instance_id, config_generation_id)
        REFERENCES process_instances(project_uuid, process_instance_id, config_generation_id),
    CHECK (attribution_deadline_unix_ms >= created_at_unix_ms),
    UNIQUE (active_root_window_id, active_experiment_id),
    UNIQUE (project_uuid, cohort_generation_id, root_key)
) STRICT;

CREATE INDEX idx_active_root_windows_deadline
    ON active_root_windows(
        project_uuid, attribution_deadline_unix_ms, active_root_window_id
    );
CREATE INDEX idx_active_root_windows_process
    ON active_root_windows(process_instance_id, active_root_window_id);

CREATE TABLE active_root_window_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_root_window_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_root_window_state_event_id) = 36
    ),
    active_root_window_id TEXT NOT NULL
        REFERENCES active_root_windows(active_root_window_id),
    state TEXT NOT NULL CHECK (
        state IN (
            'open', 'completed', 'unattributed', 'orphaned',
            'shutdown_orphaned', 'ambiguous_exposure'
        )
    ),
    label_complete INTEGER NOT NULL CHECK (label_complete IN (0, 1)),
    signal_count INTEGER NOT NULL CHECK (signal_count BETWEEN 0 AND 64),
    signal_size_bytes INTEGER NOT NULL CHECK (
        signal_size_bytes BETWEEN 0 AND 65536
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (state = 'completed' OR label_complete = 0),
    UNIQUE (active_root_window_id, active_root_window_state_event_id)
) STRICT;

CREATE INDEX idx_active_root_window_state_latest
    ON active_root_window_state_events(active_root_window_id, event_seq DESC);
CREATE UNIQUE INDEX uq_active_root_window_open_state
    ON active_root_window_state_events(active_root_window_id)
    WHERE state = 'open';
CREATE UNIQUE INDEX uq_active_root_window_terminal_state
    ON active_root_window_state_events(active_root_window_id)
    WHERE state <> 'open';

CREATE TABLE active_root_signals (
    active_root_signal_id TEXT PRIMARY KEY CHECK (length(active_root_signal_id) = 36),
    active_root_window_id TEXT NOT NULL
        REFERENCES active_root_windows(active_root_window_id),
    signal_ordinal INTEGER NOT NULL CHECK (signal_ordinal BETWEEN 0 AND 63),
    signal_identity_hash TEXT NOT NULL CHECK (
        length(signal_identity_hash) = 64
        AND signal_identity_hash NOT GLOB '*[^0-9a-f]*'
    ),
    event_kind TEXT NOT NULL CHECK (
        length(CAST(event_kind AS BLOB)) BETWEEN 1 AND 64
    ),
    scope_phase TEXT NOT NULL CHECK (
        length(CAST(scope_phase AS BLOB)) BETWEEN 1 AND 64
    ),
    category TEXT NOT NULL CHECK (
        length(CAST(category AS BLOB)) BETWEEN 1 AND 256
    ),
    name TEXT NOT NULL CHECK (
        length(CAST(name AS BLOB)) BETWEEN 1 AND 256
    ),
    disposition TEXT NOT NULL CHECK (
        disposition IN ('success', 'failure', 'ignored')
    ),
    observed_at_unix_ms INTEGER NOT NULL CHECK (observed_at_unix_ms >= 0),
    canonical_signal_json TEXT NOT NULL CHECK (
        json_valid(canonical_signal_json)
        AND length(CAST(canonical_signal_json AS BLOB)) BETWEEN 2 AND 65536
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (active_root_window_id, signal_ordinal),
    UNIQUE (active_root_window_id, signal_identity_hash)
) STRICT;

CREATE INDEX idx_active_root_signals_window
    ON active_root_signals(active_root_window_id, signal_ordinal);

CREATE TABLE active_root_decision_links (
    active_root_window_id TEXT NOT NULL,
    decision_id TEXT NOT NULL,
    owner_relation_hash TEXT NOT NULL CHECK (
        length(owner_relation_hash) = 64
        AND owner_relation_hash NOT GLOB '*[^0-9a-f]*'
    ),
    linked_at_unix_ms INTEGER NOT NULL CHECK (linked_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_root_window_id)
        REFERENCES active_root_windows(active_root_window_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id),
    PRIMARY KEY (active_root_window_id, decision_id),
    UNIQUE (decision_id)
) STRICT;

CREATE TABLE active_assignments (
    active_assignment_id TEXT PRIMARY KEY CHECK (length(active_assignment_id) = 36),
    active_experiment_id TEXT NOT NULL,
    active_root_window_id TEXT NOT NULL UNIQUE,
    decision_id TEXT NOT NULL UNIQUE,
    tranche_ordinal INTEGER NOT NULL CHECK (tranche_ordinal >= 1),
    total_ordinal INTEGER NOT NULL CHECK (total_ordinal >= 1),
    cap_ordinal INTEGER CHECK (cap_ordinal IS NULL OR cap_ordinal >= 1),
    tranche_total_ordinal INTEGER NOT NULL CHECK (tranche_total_ordinal >= 1),
    tranche_nonholdout_ordinal INTEGER CHECK (
        tranche_nonholdout_ordinal IS NULL OR tranche_nonholdout_ordinal >= 1
    ),
    arm TEXT NOT NULL CHECK (
        arm IN ('candidate_treatment', 'anchor_control', 'anchor_holdout', 'non_learning')
    ),
    cohort_threshold_numerator BLOB NOT NULL CHECK (
        length(cohort_threshold_numerator) = 8
    ),
    selection_threshold_numerator BLOB NOT NULL CHECK (
        length(selection_threshold_numerator) = 8
    ),
    configured_holdout_probability REAL NOT NULL CHECK (
        configured_holdout_probability BETWEEN 0.0 AND 1.0
    ),
    configured_holdout_probability_bits INTEGER NOT NULL,
    configured_canary_probability REAL NOT NULL CHECK (
        configured_canary_probability BETWEEN 0.0 AND 1.0
    ),
    configured_canary_probability_bits INTEGER NOT NULL,
    effective_arm_probability REAL NOT NULL CHECK (
        effective_arm_probability > 0.0 AND effective_arm_probability <= 1.0
    ),
    effective_arm_probability_bits INTEGER NOT NULL,
    conditional_selection_probability REAL NOT NULL CHECK (
        conditional_selection_probability > 0.0
        AND conditional_selection_probability <= 1.0
    ),
    conditional_selection_probability_bits INTEGER NOT NULL,
    propensity REAL NOT NULL CHECK (propensity > 0.0 AND propensity <= 1.0),
    propensity_bits INTEGER NOT NULL,
    control_generation INTEGER NOT NULL CHECK (control_generation >= 0),
    admission_unix_ms INTEGER NOT NULL CHECK (admission_unix_ms >= 0),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_experiment_id, tranche_ordinal)
        REFERENCES active_experiment_tranches(active_experiment_id, tranche_ordinal),
    FOREIGN KEY (active_root_window_id, active_experiment_id)
        REFERENCES active_root_windows(active_root_window_id, active_experiment_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id),
    CHECK (
        (arm IN ('candidate_treatment', 'anchor_control')
            AND cap_ordinal IS NOT NULL
            AND tranche_nonholdout_ordinal IS NOT NULL)
        OR (arm IN ('anchor_holdout', 'non_learning')
            AND cap_ordinal IS NULL
            AND tranche_nonholdout_ordinal IS NULL)
    ),
    UNIQUE (active_experiment_id, total_ordinal),
    UNIQUE (active_experiment_id, tranche_ordinal, tranche_total_ordinal),
    UNIQUE (active_assignment_id, active_experiment_id),
    UNIQUE (active_assignment_id, active_root_window_id)
) STRICT;

CREATE UNIQUE INDEX uq_active_assignments_cap_ordinal
    ON active_assignments(active_experiment_id, cap_ordinal)
    WHERE cap_ordinal IS NOT NULL;
CREATE UNIQUE INDEX uq_active_assignments_tranche_nonholdout
    ON active_assignments(
        active_experiment_id, tranche_ordinal, tranche_nonholdout_ordinal
    )
    WHERE tranche_nonholdout_ordinal IS NOT NULL;
CREATE INDEX idx_active_assignments_tranche
    ON active_assignments(
        active_experiment_id, tranche_ordinal, tranche_total_ordinal
    );

CREATE TABLE active_dispatches (
    active_dispatch_id TEXT PRIMARY KEY CHECK (length(active_dispatch_id) = 36),
    active_experiment_id TEXT NOT NULL,
    active_assignment_id TEXT NOT NULL UNIQUE,
    decision_id TEXT NOT NULL UNIQUE,
    candidate_id TEXT NOT NULL CHECK (
        length(CAST(candidate_id AS BLOB)) BETWEEN 1 AND 128
    ),
    request_identity_hash TEXT NOT NULL CHECK (
        length(request_identity_hash) = 64
        AND request_identity_hash NOT GLOB '*[^0-9a-f]*'
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    admitted_at_unix_ms INTEGER NOT NULL CHECK (admitted_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_assignment_id, active_experiment_id)
        REFERENCES active_assignments(active_assignment_id, active_experiment_id),
    FOREIGN KEY (decision_id) REFERENCES decisions(decision_id),
    UNIQUE (active_dispatch_id, active_assignment_id),
    UNIQUE (active_dispatch_id, active_experiment_id)
) STRICT;

CREATE INDEX idx_active_dispatches_process
    ON active_dispatches(process_instance_id, admitted_at_unix_ms, active_dispatch_id);

CREATE TABLE active_dispatch_terminal_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_dispatch_terminal_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_dispatch_terminal_event_id) = 36
    ),
    active_dispatch_id TEXT NOT NULL UNIQUE,
    active_assignment_id TEXT NOT NULL,
    terminal_state TEXT NOT NULL CHECK (
        terminal_state IN (
            'completed', 'provider_error', 'cancelled_before_handoff',
            'cancelled_after_handoff', 'panicked_after_handoff',
            'aborted_before_handoff', 'unknown_after_crash'
        )
    ),
    representative_status TEXT CHECK (
        representative_status IS NULL OR representative_status IN ('completed', 'error')
    ),
    stable_error_class TEXT CHECK (
        stable_error_class IS NULL OR (
            length(CAST(stable_error_class AS BLOB)) BETWEEN 1 AND 128
            AND stable_error_class NOT GLOB '*[^a-z0-9._-]*'
        )
    ),
    provider_receipt_hash TEXT CHECK (
        provider_receipt_hash IS NULL OR (
            length(provider_receipt_hash) = 64
            AND provider_receipt_hash NOT GLOB '*[^0-9a-f]*'
        )
    ),
    handed_off_at_unix_ms INTEGER CHECK (handed_off_at_unix_ms >= 0),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_dispatch_id, active_assignment_id)
        REFERENCES active_dispatches(active_dispatch_id, active_assignment_id),
    CHECK (
        (terminal_state IN ('completed', 'provider_error')
            AND handed_off_at_unix_ms IS NOT NULL)
        OR terminal_state NOT IN ('completed', 'provider_error')
    ),
    CHECK (
        terminal_state NOT IN ('cancelled_after_handoff', 'panicked_after_handoff')
        OR handed_off_at_unix_ms IS NOT NULL
    ),
    CHECK (
        terminal_state NOT IN ('cancelled_before_handoff', 'aborted_before_handoff')
        OR handed_off_at_unix_ms IS NULL
    ),
    CHECK (
        (terminal_state = 'completed' AND representative_status = 'completed')
        OR (terminal_state = 'provider_error' AND representative_status = 'error')
        OR (terminal_state NOT IN ('completed', 'provider_error')
            AND representative_status IS NULL)
    ),
    CHECK (
        terminal_state <> 'provider_error' OR stable_error_class IS NOT NULL
    )
) STRICT;

CREATE INDEX idx_active_dispatch_terminal_assignment
    ON active_dispatch_terminal_events(active_assignment_id, event_seq);

CREATE TABLE outcomes (
    outcome_id TEXT PRIMARY KEY CHECK (length(outcome_id) = 36),
    active_experiment_id TEXT NOT NULL,
    active_root_window_id TEXT NOT NULL,
    active_assignment_id TEXT NOT NULL UNIQUE,
    representative_decision_id TEXT,
    root_key TEXT NOT NULL CHECK (
        length(root_key) = 64 AND root_key NOT GLOB '*[^0-9a-f]*'
    ),
    outcome_policy_hash TEXT NOT NULL CHECK (
        length(outcome_policy_hash) = 64
        AND outcome_policy_hash NOT GLOB '*[^0-9a-f]*'
    ),
    source_signal_aggregate_hash TEXT NOT NULL CHECK (
        length(source_signal_aggregate_hash) = 64
        AND source_signal_aggregate_hash NOT GLOB '*[^0-9a-f]*'
    ),
    arm TEXT NOT NULL CHECK (
        arm IN ('candidate_treatment', 'anchor_control', 'anchor_holdout', 'non_learning')
    ),
    total_ordinal INTEGER NOT NULL CHECK (total_ordinal >= 1),
    cap_ordinal INTEGER CHECK (cap_ordinal IS NULL OR cap_ordinal >= 1),
    label TEXT CHECK (label IS NULL OR label IN ('success', 'failure')),
    attribution_status TEXT NOT NULL CHECK (
        attribution_status IN (
            'eligible_treatment', 'eligible_control', 'monitoring_only',
            'unattributed', 'orphaned', 'ambiguous_exposure'
        )
    ),
    interval_start_unix_ms INTEGER NOT NULL CHECK (interval_start_unix_ms >= 0),
    interval_end_unix_ms INTEGER NOT NULL CHECK (
        interval_end_unix_ms >= interval_start_unix_ms
    ),
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0.0),
    latency_ms_bits INTEGER NOT NULL,
    stable_terminal_error_class TEXT CHECK (
        stable_terminal_error_class IS NULL OR (
            length(CAST(stable_terminal_error_class AS BLOB)) BETWEEN 1 AND 128
            AND stable_terminal_error_class NOT GLOB '*[^a-z0-9._-]*'
        )
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_assignment_id, active_experiment_id)
        REFERENCES active_assignments(active_assignment_id, active_experiment_id),
    FOREIGN KEY (active_assignment_id, active_root_window_id)
        REFERENCES active_assignments(active_assignment_id, active_root_window_id),
    FOREIGN KEY (representative_decision_id)
        REFERENCES decisions(decision_id),
    CHECK (
        (attribution_status IN ('eligible_treatment', 'eligible_control', 'monitoring_only')
            AND label IS NOT NULL)
        OR (attribution_status IN ('unattributed', 'orphaned', 'ambiguous_exposure')
            AND label IS NULL)
    ),
    CHECK (
        (arm IN ('candidate_treatment', 'anchor_control') AND cap_ordinal IS NOT NULL)
        OR (arm IN ('anchor_holdout', 'non_learning') AND cap_ordinal IS NULL)
    ),
    UNIQUE (outcome_id, active_assignment_id),
    UNIQUE (outcome_id, active_experiment_id),
    UNIQUE (active_experiment_id, total_ordinal)
) STRICT;

CREATE INDEX idx_outcomes_experiment_cap
    ON outcomes(active_experiment_id, cap_ordinal, outcome_id);
CREATE INDEX idx_outcomes_root_window
    ON outcomes(active_root_window_id, outcome_id);
CREATE INDEX idx_outcomes_representative_decision
    ON outcomes(representative_decision_id)
    WHERE representative_decision_id IS NOT NULL;

CREATE TABLE active_look_members (
    active_outcome_look_id TEXT NOT NULL,
    delta_member_ordinal INTEGER NOT NULL CHECK (delta_member_ordinal >= 1),
    active_experiment_id TEXT NOT NULL,
    active_assignment_id TEXT NOT NULL,
    outcome_id TEXT,
    cap_ordinal INTEGER NOT NULL CHECK (cap_ordinal >= 1),
    arm TEXT NOT NULL CHECK (arm IN ('candidate_treatment', 'anchor_control')),
    label TEXT CHECK (label IS NULL OR label IN ('success', 'failure')),
    inclusion_status TEXT NOT NULL CHECK (
        inclusion_status IN (
            'eligible', 'unattributed', 'orphaned', 'ambiguous_exposure'
        )
    ),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_outcome_look_id, active_experiment_id)
        REFERENCES active_outcome_looks(active_outcome_look_id, active_experiment_id),
    FOREIGN KEY (active_assignment_id, active_experiment_id)
        REFERENCES active_assignments(active_assignment_id, active_experiment_id),
    FOREIGN KEY (outcome_id, active_assignment_id)
        REFERENCES outcomes(outcome_id, active_assignment_id),
    CHECK (
        (inclusion_status = 'eligible' AND outcome_id IS NOT NULL AND label IS NOT NULL)
        OR (inclusion_status <> 'eligible' AND label IS NULL)
    ),
    PRIMARY KEY (active_outcome_look_id, delta_member_ordinal),
    UNIQUE (active_outcome_look_id, active_assignment_id),
    UNIQUE (active_outcome_look_id, cap_ordinal)
) STRICT;

CREATE INDEX idx_active_look_members_assignment
    ON active_look_members(active_assignment_id, active_outcome_look_id);

CREATE TABLE active_neighborhood_state_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_neighborhood_state_event_id TEXT NOT NULL UNIQUE CHECK (
        length(active_neighborhood_state_event_id) = 36
    ),
    active_experiment_id TEXT NOT NULL
        REFERENCES active_experiments(active_experiment_id),
    neighborhood_identity_hash TEXT NOT NULL CHECK (
        length(neighborhood_identity_hash) = 64
        AND neighborhood_identity_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_query_hash TEXT NOT NULL
        REFERENCES canonical_routing_queries(canonical_query_hash),
    sorted_neighbor_hash TEXT NOT NULL CHECK (
        length(sorted_neighbor_hash) = 64
        AND sorted_neighbor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    config_generation_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT NOT NULL,
    active_authorization_state_event_id TEXT NOT NULL,
    predecessor_neighborhood_state_event_id TEXT,
    state TEXT NOT NULL CHECK (
        state IN ('promoted', 'invalidated', 'cooloff', 'restored')
    ),
    cooloff_until_unix_ms INTEGER CHECK (
        cooloff_until_unix_ms IS NULL OR cooloff_until_unix_ms >= 0
    ),
    cause_active_dispatch_id TEXT,
    cause_outcome_id TEXT,
    stable_reason TEXT NOT NULL CHECK (
        length(CAST(stable_reason AS BLOB)) BETWEEN 1 AND 128
    ),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (active_authorization_state_event_id, active_experiment_id)
        REFERENCES active_authorization_state_events(
            active_authorization_state_event_id, active_experiment_id
        ),
    FOREIGN KEY (
        predecessor_neighborhood_state_event_id, active_experiment_id
    ) REFERENCES active_neighborhood_state_events(
        active_neighborhood_state_event_id, active_experiment_id
    ),
    FOREIGN KEY (cause_active_dispatch_id, active_experiment_id)
        REFERENCES active_dispatches(active_dispatch_id, active_experiment_id),
    FOREIGN KEY (cause_outcome_id, active_experiment_id)
        REFERENCES outcomes(outcome_id, active_experiment_id),
    CHECK (
        (state = 'cooloff'
            AND cooloff_until_unix_ms IS NOT NULL
            AND cooloff_until_unix_ms > created_at_unix_ms)
        OR (state <> 'cooloff' AND cooloff_until_unix_ms IS NULL)
    ),
    CHECK (
        state IN ('invalidated', 'cooloff')
        OR (cause_active_dispatch_id IS NULL AND cause_outcome_id IS NULL)
    ),
    UNIQUE (active_neighborhood_state_event_id, active_experiment_id),
    UNIQUE (
        active_experiment_id, neighborhood_identity_hash,
        active_neighborhood_state_event_id
    )
) STRICT;

CREATE INDEX idx_active_neighborhood_state_latest
    ON active_neighborhood_state_events(
        active_experiment_id, neighborhood_identity_hash, event_seq DESC
    );
CREATE INDEX idx_active_neighborhood_query
    ON active_neighborhood_state_events(
        canonical_query_hash, sorted_neighbor_hash, event_seq DESC
    );

CREATE TABLE active_decision_facts (
    decision_id TEXT PRIMARY KEY,
    decision_shape_version INTEGER NOT NULL CHECK (decision_shape_version = 2),
    active_experiment_id TEXT,
    active_authorization_state_event_id TEXT,
    active_neighborhood_state_event_id TEXT,
    active_root_window_id TEXT,
    active_assignment_id TEXT,
    active_dispatch_id TEXT,
    active_outcome_look_id TEXT,
    planned_route TEXT NOT NULL CHECK (
        planned_route IN (
            'candidate', 'anchor_control', 'anchor_holdout',
            'anchor_forced', 'anchor_paused', 'anchor_fallback'
        )
    ),
    assignment_arm TEXT NOT NULL CHECK (
        assignment_arm IN (
            'candidate_treatment', 'anchor_control', 'anchor_holdout', 'non_learning'
        )
    ),
    control_generation INTEGER NOT NULL CHECK (control_generation >= 0),
    promotion_lower_bound REAL NOT NULL CHECK (
        promotion_lower_bound BETWEEN 0.0 AND 1.0
    ),
    promotion_lower_bound_bits INTEGER NOT NULL,
    retention_lower_bound REAL NOT NULL CHECK (
        retention_lower_bound >= 0.0 AND retention_lower_bound <= promotion_lower_bound
    ),
    retention_lower_bound_bits INTEGER NOT NULL,
    configured_holdout_probability REAL NOT NULL CHECK (
        configured_holdout_probability BETWEEN 0.0 AND 1.0
    ),
    configured_holdout_probability_bits INTEGER NOT NULL,
    configured_canary_probability REAL NOT NULL CHECK (
        configured_canary_probability BETWEEN 0.0 AND 1.0
    ),
    configured_canary_probability_bits INTEGER NOT NULL,
    effective_arm_probability REAL NOT NULL CHECK (
        effective_arm_probability > 0.0 AND effective_arm_probability <= 1.0
    ),
    effective_arm_probability_bits INTEGER NOT NULL,
    conditional_selection_probability REAL NOT NULL CHECK (
        conditional_selection_probability > 0.0
        AND conditional_selection_probability <= 1.0
    ),
    conditional_selection_probability_bits INTEGER NOT NULL,
    propensity REAL NOT NULL CHECK (propensity > 0.0 AND propensity <= 1.0),
    propensity_bits INTEGER NOT NULL,
    actual_outcome_noninferiority_lower REAL CHECK (
        actual_outcome_noninferiority_lower IS NULL
        OR actual_outcome_noninferiority_lower BETWEEN 0.0 AND 1.0
    ),
    actual_outcome_noninferiority_lower_bits INTEGER,
    actual_outcome_noninferiority_upper REAL CHECK (
        actual_outcome_noninferiority_upper IS NULL
        OR actual_outcome_noninferiority_upper BETWEEN 0.0 AND 1.0
    ),
    actual_outcome_noninferiority_upper_bits INTEGER,
    anchor_shadow_lower_bound REAL CHECK (
        anchor_shadow_lower_bound IS NULL OR anchor_shadow_lower_bound BETWEEN 0.0 AND 1.0
    ),
    anchor_shadow_lower_bound_bits INTEGER,
    fallback_reason TEXT CHECK (
        fallback_reason IS NULL OR length(CAST(fallback_reason AS BLOB)) BETWEEN 1 AND 128
    ),
    fresh_gate_audit_json TEXT NOT NULL CHECK (
        json_valid(fresh_gate_audit_json)
        AND length(CAST(fresh_gate_audit_json AS BLOB)) BETWEEN 2 AND 1048576
    ),
    fresh_gate_audit_hash TEXT NOT NULL CHECK (
        length(fresh_gate_audit_hash) = 64
        AND fresh_gate_audit_hash NOT GLOB '*[^0-9a-f]*'
    ),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (decision_id, decision_shape_version)
        REFERENCES decisions(decision_id, decision_shape_version),
    FOREIGN KEY (active_authorization_state_event_id, active_experiment_id)
        REFERENCES active_authorization_state_events(
            active_authorization_state_event_id, active_experiment_id
        ),
    FOREIGN KEY (active_neighborhood_state_event_id, active_experiment_id)
        REFERENCES active_neighborhood_state_events(
            active_neighborhood_state_event_id, active_experiment_id
        ),
    FOREIGN KEY (active_root_window_id, active_experiment_id)
        REFERENCES active_root_windows(active_root_window_id, active_experiment_id),
    FOREIGN KEY (active_assignment_id, active_experiment_id)
        REFERENCES active_assignments(active_assignment_id, active_experiment_id),
    FOREIGN KEY (active_dispatch_id, active_experiment_id)
        REFERENCES active_dispatches(active_dispatch_id, active_experiment_id),
    FOREIGN KEY (active_outcome_look_id, active_experiment_id)
        REFERENCES active_outcome_looks(active_outcome_look_id, active_experiment_id),
    CHECK (
        (actual_outcome_noninferiority_lower IS NULL)
            = (actual_outcome_noninferiority_lower_bits IS NULL)
    ),
    CHECK (
        (actual_outcome_noninferiority_upper IS NULL)
            = (actual_outcome_noninferiority_upper_bits IS NULL)
    ),
    CHECK (
        (anchor_shadow_lower_bound IS NULL) = (anchor_shadow_lower_bound_bits IS NULL)
    ),
    CHECK (
        actual_outcome_noninferiority_lower IS NULL
        OR actual_outcome_noninferiority_upper IS NULL
        OR actual_outcome_noninferiority_lower <= actual_outcome_noninferiority_upper
    ),
    CHECK (
        (planned_route = 'candidate'
            AND assignment_arm = 'candidate_treatment'
            AND active_experiment_id IS NOT NULL
            AND active_authorization_state_event_id IS NOT NULL
            AND active_neighborhood_state_event_id IS NOT NULL
            AND active_root_window_id IS NOT NULL
            AND active_assignment_id IS NOT NULL
            AND active_dispatch_id IS NOT NULL)
        OR (planned_route <> 'candidate' AND active_dispatch_id IS NULL)
    ),
    CHECK (
        (active_experiment_id IS NULL
            AND active_authorization_state_event_id IS NULL
            AND active_neighborhood_state_event_id IS NULL
            AND active_root_window_id IS NULL
            AND active_assignment_id IS NULL
            AND active_outcome_look_id IS NULL)
        OR active_experiment_id IS NOT NULL
    )
) STRICT;

CREATE INDEX idx_active_decision_facts_experiment
    ON active_decision_facts(active_experiment_id, decision_id)
    WHERE active_experiment_id IS NOT NULL;
CREATE INDEX idx_active_decision_facts_assignment
    ON active_decision_facts(active_assignment_id)
    WHERE active_assignment_id IS NOT NULL;

CREATE TABLE active_retirement_markers (
    active_retirement_marker_id TEXT PRIMARY KEY CHECK (
        length(active_retirement_marker_id) = 36
    ),
    active_experiment_id TEXT NOT NULL UNIQUE
        REFERENCES active_experiments(active_experiment_id) ON DELETE CASCADE,
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    first_retention_batch_id TEXT NOT NULL
        REFERENCES retention_batches(retention_batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    age_expired INTEGER NOT NULL CHECK (age_expired IN (0, 1)),
    count_pressure INTEGER NOT NULL CHECK (count_pressure IN (0, 1)),
    marked_by_process_instance_id TEXT NOT NULL
        REFERENCES process_instances(process_instance_id),
    marked_at_unix_ms INTEGER NOT NULL CHECK (marked_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL UNIQUE CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (age_expired = 1 OR count_pressure = 1),
    UNIQUE (active_retirement_marker_id, active_experiment_id)
) STRICT;

CREATE INDEX idx_active_retirement_markers_batch
    ON active_retirement_markers(first_retention_batch_id, active_experiment_id);

CREATE TABLE active_retirement_receipts (
    receipt_ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
    active_retirement_receipt_id TEXT NOT NULL UNIQUE CHECK (
        length(active_retirement_receipt_id) = 36
    ),
    active_retirement_marker_id TEXT NOT NULL CHECK (
        length(active_retirement_marker_id) = 36
    ),
    active_experiment_id TEXT NOT NULL CHECK (length(active_experiment_id) = 36),
    retention_batch_id TEXT NOT NULL
        REFERENCES retention_batches(retention_batch_id),
    predecessor_chain_hash TEXT NOT NULL CHECK (
        length(predecessor_chain_hash) = 64
        AND predecessor_chain_hash NOT GLOB '*[^0-9a-f]*'
    ),
    chain_tip_hash TEXT NOT NULL UNIQUE CHECK (
        length(chain_tip_hash) = 64 AND chain_tip_hash NOT GLOB '*[^0-9a-f]*'
    ),
    cursor_json TEXT NOT NULL CHECK (
        json_valid(cursor_json)
        AND length(CAST(cursor_json AS BLOB)) BETWEEN 2 AND 65536
    ),
    cursor_hash TEXT NOT NULL CHECK (
        length(cursor_hash) = 64 AND cursor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    parent_rows_deleted INTEGER NOT NULL CHECK (
        parent_rows_deleted BETWEEN 0 AND 1000
    ),
    child_rows_deleted INTEGER NOT NULL CHECK (
        child_rows_deleted BETWEEN 0 AND 4095
    ),
    bytes_deleted INTEGER NOT NULL CHECK (bytes_deleted BETWEEN 0 AND 33554432),
    completed INTEGER NOT NULL CHECK (completed IN (0, 1)),
    process_instance_id TEXT NOT NULL REFERENCES process_instances(process_instance_id),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = chain_tip_hash
    ),
    UNIQUE (active_experiment_id, receipt_ordinal),
    UNIQUE (active_retirement_marker_id, receipt_ordinal)
) STRICT;

CREATE INDEX idx_active_retirement_receipts_experiment
    ON active_retirement_receipts(
        active_experiment_id, receipt_ordinal, active_retirement_receipt_id
    );
CREATE INDEX idx_active_retirement_receipts_completed
    ON active_retirement_receipts(completed, receipt_ordinal);

CREATE TABLE active_retirement_checkpoints (
    checkpoint_hash TEXT PRIMARY KEY CHECK (
        length(checkpoint_hash) = 64
        AND checkpoint_hash NOT GLOB '*[^0-9a-f]*'
    ),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    level INTEGER NOT NULL CHECK (level BETWEEN 0 AND 2147483647),
    first_receipt_ordinal INTEGER NOT NULL CHECK (first_receipt_ordinal >= 1),
    last_receipt_ordinal INTEGER NOT NULL CHECK (
        last_receipt_ordinal >= first_receipt_ordinal
    ),
    first_predecessor_hash TEXT NOT NULL CHECK (
        length(first_predecessor_hash) = 64
        AND first_predecessor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    covered_chain_tip_hash TEXT NOT NULL CHECK (
        length(covered_chain_tip_hash) = 64
        AND covered_chain_tip_hash NOT GLOB '*[^0-9a-f]*'
    ),
    receipt_count INTEGER NOT NULL CHECK (receipt_count > 0),
    parent_rows_deleted INTEGER NOT NULL CHECK (parent_rows_deleted >= 0),
    child_rows_deleted INTEGER NOT NULL CHECK (child_rows_deleted >= 0),
    bytes_deleted INTEGER NOT NULL CHECK (bytes_deleted >= 0),
    range_hash TEXT NOT NULL CHECK (
        length(range_hash) = 64 AND range_hash NOT GLOB '*[^0-9a-f]*'
    ),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        canonical_payload_hash = checkpoint_hash
    ),
    UNIQUE (project_uuid, first_receipt_ordinal, last_receipt_ordinal, level),
    UNIQUE (project_uuid, covered_chain_tip_hash)
) STRICT;

CREATE INDEX idx_active_retirement_checkpoints_order
    ON active_retirement_checkpoints(
        project_uuid, first_receipt_ordinal, last_receipt_ordinal,
        level, checkpoint_hash
    );
