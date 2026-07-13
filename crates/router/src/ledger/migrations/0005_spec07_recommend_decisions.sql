-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE spec07_decision_placeholder_guard (
    singleton_key INTEGER PRIMARY KEY CHECK (singleton_key = 1),
    decision_row_count INTEGER NOT NULL CHECK (decision_row_count = 0),
    summary_row_count INTEGER NOT NULL CHECK (summary_row_count = 0),
    neighbor_row_count INTEGER NOT NULL CHECK (neighbor_row_count = 0),
    outcome_row_count INTEGER NOT NULL CHECK (outcome_row_count = 0)
) STRICT;

INSERT INTO spec07_decision_placeholder_guard (
    singleton_key, decision_row_count, summary_row_count,
    neighbor_row_count, outcome_row_count
)
SELECT
    1,
    (SELECT count(*) FROM decisions),
    (SELECT count(*) FROM decision_candidate_summaries),
    (SELECT count(*) FROM decision_neighbors),
    (SELECT count(*) FROM outcomes);

DROP TABLE spec07_decision_placeholder_guard;

DROP TABLE outcomes;
DROP TABLE decision_neighbors;
DROP TABLE decision_candidate_summaries;
DROP TABLE decisions;

CREATE UNIQUE INDEX uq_pool_vector_space_mapping_exact
    ON pool_vector_space_mappings(
        project_uuid, config_generation_id, pool_id, policy_version_id, vector_space_id
    );

CREATE TABLE decisions (
    decision_id TEXT PRIMARY KEY CHECK (length(decision_id) = 36),
    decision_shape_version INTEGER NOT NULL CHECK (decision_shape_version = 1),
    algorithm_version INTEGER NOT NULL CHECK (algorithm_version = 1),
    project_uuid TEXT NOT NULL,
    process_instance_id TEXT NOT NULL,
    config_generation_id TEXT NOT NULL,
    policy_version_id TEXT NOT NULL,
    learning_generation_id TEXT NOT NULL,
    cohort_generation_id TEXT,
    active_experiment_id TEXT,
    active_authorization_state_event_id TEXT,
    pool_id TEXT NOT NULL CHECK (length(pool_id) BETWEEN 1 AND 128),
    candidate_id TEXT CHECK (
        candidate_id IS NULL OR length(candidate_id) BETWEEN 1 AND 128
    ),
    root_key TEXT CHECK (
        root_key IS NULL OR (
            length(root_key) = 64 AND root_key NOT GLOB '*[^0-9a-f]*'
        )
    ),
    primary_call_uuid TEXT NOT NULL CHECK (length(primary_call_uuid) = 36),
    mode TEXT NOT NULL CHECK (mode = 'recommend'),
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
    recommended_model TEXT NOT NULL CHECK (length(recommended_model) BETWEEN 1 AND 512),
    recommended_model_revision TEXT NOT NULL CHECK (
        length(recommended_model_revision) BETWEEN 1 AND 128
    ),
    served_model TEXT NOT NULL CHECK (length(served_model) BETWEEN 1 AND 512),
    served_model_revision TEXT NOT NULL CHECK (
        length(served_model_revision) BETWEEN 1 AND 128
    ),
    as_of_unix_ms INTEGER NOT NULL CHECK (as_of_unix_ms >= 0),
    decision_latency_ms INTEGER NOT NULL CHECK (decision_latency_ms >= 0),
    final_reason TEXT NOT NULL CHECK (
        final_reason IN (
            'embedding_unavailable', 'vector_unhealthy', 'version_mismatch',
            'no_partition', 'sparse_points', 'insufficient_roots', 'low_coverage',
            'insufficient_effective_samples', 'lower_bound_below_threshold',
            'invalid_evidence_time', 'numeric_error', 'no_candidate_passed',
            'recommend_observe_only'
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
        (candidate_id IS NULL
            AND final_reason <> 'recommend_observe_only'
            AND recommended_model = served_model
            AND recommended_model_revision = served_model_revision)
        OR (candidate_id IS NOT NULL AND final_reason = 'recommend_observe_only')
    ),
    CHECK (
        (active_experiment_id IS NULL AND active_authorization_state_event_id IS NULL)
        OR (active_experiment_id IS NOT NULL AND active_authorization_state_event_id IS NOT NULL
            AND cohort_generation_id IS NOT NULL AND candidate_id IS NOT NULL)
    ),
    UNIQUE (decision_id, project_uuid),
    UNIQUE (decision_id, vector_space_id),
    UNIQUE (project_uuid, primary_call_uuid)
) STRICT;

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

CREATE TABLE decision_candidate_summaries (
    decision_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
    rank_ordinal INTEGER NOT NULL CHECK (rank_ordinal BETWEEN 0 AND 63),
    candidate_model TEXT NOT NULL CHECK (length(candidate_model) BETWEEN 1 AND 512),
    candidate_model_revision TEXT NOT NULL CHECK (
        length(candidate_model_revision) BETWEEN 1 AND 128
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

CREATE INDEX idx_decision_summaries_partition
    ON decision_candidate_summaries(partition_id, decision_id)
    WHERE partition_id IS NOT NULL;

CREATE TABLE decision_neighbors (
    decision_id TEXT NOT NULL,
    neighbor_ordinal INTEGER NOT NULL CHECK (neighbor_ordinal BETWEEN 0 AND 4094),
    candidate_id TEXT NOT NULL CHECK (length(candidate_id) BETWEEN 1 AND 128),
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

CREATE INDEX idx_decision_neighbors_evidence_vector_link
    ON decision_neighbors(evidence_vector_link_id);
CREATE INDEX idx_decision_neighbors_anchor
    ON decision_neighbors(anchor_id, decision_id);
CREATE INDEX idx_decision_neighbors_shadow_attempt
    ON decision_neighbors(shadow_attempt_id, decision_id);
CREATE INDEX idx_decision_neighbors_evaluation
    ON decision_neighbors(evaluation_id, decision_id)
    WHERE evaluation_id IS NOT NULL;

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

CREATE INDEX idx_outcomes_decision ON outcomes(decision_id);

CREATE TABLE decision_retiring_anchors (
    anchor_id TEXT PRIMARY KEY
        REFERENCES anchors(anchor_id) ON DELETE CASCADE,
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    first_retention_batch_id TEXT NOT NULL
        REFERENCES retention_batches(retention_batch_id)
        DEFERRABLE INITIALLY DEFERRED,
    age_expired INTEGER NOT NULL CHECK (age_expired IN (0, 1)),
    count_excess INTEGER NOT NULL CHECK (count_excess IN (0, 1)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK (age_expired = 1 OR count_excess = 1)
) STRICT;

CREATE INDEX idx_decision_retiring_anchors_project
    ON decision_retiring_anchors(project_uuid, created_at_unix_ms, anchor_id);

CREATE TABLE decision_retention_receipts (
    retention_batch_id TEXT PRIMARY KEY
        REFERENCES retention_batches(retention_batch_id),
    project_uuid TEXT NOT NULL REFERENCES project_metadata(project_uuid),
    process_instance_id TEXT NOT NULL,
    decision_age_expired INTEGER NOT NULL CHECK (decision_age_expired IN (0, 1)),
    decision_count_excess INTEGER NOT NULL CHECK (decision_count_excess IN (0, 1)),
    source_forced INTEGER NOT NULL CHECK (source_forced IN (0, 1)),
    new_marker_count INTEGER NOT NULL CHECK (new_marker_count BETWEEN 0 AND 1000),
    new_marker_hash TEXT NOT NULL CHECK (
        length(new_marker_hash) = 64 AND new_marker_hash NOT GLOB '*[^0-9a-f]*'
    ),
    deleted_decision_count INTEGER NOT NULL CHECK (
        deleted_decision_count BETWEEN 0 AND 1000
    ),
    deleted_summary_count INTEGER NOT NULL CHECK (
        deleted_summary_count BETWEEN 0 AND 4159
    ),
    deleted_neighbor_count INTEGER NOT NULL CHECK (
        deleted_neighbor_count BETWEEN 0 AND 4159
    ),
    deleted_child_count INTEGER NOT NULL CHECK (
        deleted_child_count BETWEEN 0 AND 4159
    ),
    verified_aggregate_bytes INTEGER NOT NULL CHECK (
        verified_aggregate_bytes BETWEEN 0 AND 33554432
    ),
    selection_lower_bound_unix_ms INTEGER CHECK (
        selection_lower_bound_unix_ms IS NULL OR selection_lower_bound_unix_ms >= 0
    ),
    selection_upper_bound_unix_ms INTEGER CHECK (
        selection_upper_bound_unix_ms IS NULL OR selection_upper_bound_unix_ms >= 0
    ),
    decision_selection_hash TEXT NOT NULL CHECK (
        length(decision_selection_hash) = 64
        AND decision_selection_hash NOT GLOB '*[^0-9a-f]*'
    ),
    blocked_anchor_count INTEGER NOT NULL CHECK (blocked_anchor_count BETWEEN 0 AND 1000),
    blocked_anchor_hash TEXT NOT NULL CHECK (
        length(blocked_anchor_hash) = 64
        AND blocked_anchor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    deleted_anchor_count INTEGER NOT NULL CHECK (deleted_anchor_count BETWEEN 0 AND 1000),
    deleted_anchor_hash TEXT NOT NULL CHECK (
        length(deleted_anchor_hash) = 64
        AND deleted_anchor_hash NOT GLOB '*[^0-9a-f]*'
    ),
    more_cleanup INTEGER NOT NULL CHECK (more_cleanup IN (0, 1)),
    created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms >= 0),
    canonical_payload_hash TEXT NOT NULL CHECK (
        length(canonical_payload_hash) = 64
        AND canonical_payload_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY (project_uuid, process_instance_id)
        REFERENCES process_instances(project_uuid, process_instance_id),
    CHECK (deleted_child_count = deleted_summary_count + deleted_neighbor_count),
    CHECK (
        (deleted_decision_count = 0
            AND deleted_child_count = 0
            AND selection_lower_bound_unix_ms IS NULL
            AND selection_upper_bound_unix_ms IS NULL
            AND verified_aggregate_bytes = 0)
        OR (deleted_decision_count > 0
            AND deleted_summary_count >= deleted_decision_count
            AND selection_lower_bound_unix_ms IS NOT NULL
            AND selection_upper_bound_unix_ms IS NOT NULL
            AND selection_lower_bound_unix_ms <= selection_upper_bound_unix_ms
            AND verified_aggregate_bytes > 0)
    )
) STRICT;
