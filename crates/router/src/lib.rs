// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Source-only Shadow, Recommend, and bounded Active routing for NeMo Relay.
//!
//! [`RouterMode::Shadow`] collects counterfactual evidence while preserving the
//! anchor response. [`RouterMode::Recommend`] records a bounded observational
//! decision and still preserves the anchor. [`RouterMode::Active`] can serve one
//! durably authorized, randomized model-only treatment while control and
//! holdout roots preserve the anchor. Every managed continuation is invoked
//! exactly once; an Active provider error is never retried against the anchor.
//!
//! Custom Rust hosts configure this crate independently from `nemo-relay` and
//! must call [`register_router_component`] before plugin validation or
//! initialization. The Core runtime does not depend on Router. Bundled CLI,
//! Python, Node.js, and C FFI hosts register Router automatically; Go reaches
//! that registration through the bundled generic FFI plugin path. Generic FFI
//! and Go calls cannot originate the V2 replay context required for eligible
//! routing.

#[cfg(test)]
pub(crate) static TEST_GLOBAL_CONTEXT_MUTEX: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

mod active_evaluator;
#[allow(dead_code)] // Task 9 wires the bounded active look engine into durable claims.
mod active_math;
#[cfg(test)]
mod active_math_simulation_tests;
#[cfg(test)]
mod active_math_tests;
mod active_planner;
mod active_runtime;
#[cfg(test)]
mod active_runtime_integration_tests;
mod adapter;
mod background;
mod background_driver;
#[allow(dead_code)] // The Task 10 supervisor driver wires these bounded operations next.
mod background_jobs;
#[cfg(test)]
mod background_jobs_tests;
#[allow(dead_code)] // Spec 07 recommendation consumes the exact live set identity.
mod candidate_set;
mod canonical_json;
#[allow(dead_code)] // Task 7 consumes the pure canonical query artifact.
mod canonical_query;
mod confidence;
pub mod config;
pub mod control;
mod coordinator;
#[allow(dead_code)] // Task 7 builds and Task 6 storage persists validated decision graphs.
mod decision_audit;
pub mod diagnostics;
mod eligibility;
#[allow(dead_code)] // Task 10 wires the bounded provider into supervised workers.
mod embedder;
#[cfg(test)]
mod embedder_tests;
mod embedding_identity;
mod evaluator;
mod fingerprint;
mod health;
mod host;
pub mod inspection;
mod judge;
mod ledger;
#[allow(dead_code)] // Spec 07 consumes the crate-private live embedding service.
mod live_embedding;
#[cfg(test)]
mod live_embedding_tests;
mod matcher;
#[allow(dead_code)] // Tasks 8 and 11 wire the pure classifier into persistence and runtime.
mod outcome;
mod plugin_component;
mod preflight;
pub mod projection;
mod provider_admission;
mod recommendation;
mod recommendation_delivery;
mod recommendation_runtime;
#[cfg(test)]
mod recommendation_runtime_integration_tests;
mod response_validator;
#[allow(dead_code)] // Task 7 consumes the pure strict-partition artifact.
mod routing_partition;
mod runtime;
mod sampling;
mod scheduler;
mod scheduler_admission;
mod selector;
#[cfg(test)]
mod selector_integration_tests;
mod sink;
mod sqlite_sink;
#[cfg(test)]
mod sqlite_vec_capability_tests;
mod sqlite_vec_extension;
mod sqlite_vec_schema;
#[allow(dead_code)] // Tasks 6-8 consume the internal async production backend.
mod sqlite_vector_store;
mod trajectory;
#[allow(dead_code)] // Tasks 6 and 7 consume the checked vector representation.
mod vector;
#[allow(dead_code)] // Task 6 implements the production backend for this contract.
mod vector_store;

pub use config::{
    CANDIDATE_ID_MAX_BYTES, CandidateCapabilities, CandidateConfig, CanonicalizerConfig,
    ConcurrencyConfig, EMBEDDER_BATCH_SIZE_MAX, EMBEDDER_MAX_IN_FLIGHT, EMBEDDER_REQUEST_BYTES_MAX,
    EMBEDDER_RESPONSE_BYTES_MAX, EMBEDDER_TIMEOUT_MS_MAX, EMBEDDING_DIMENSIONS_MAX,
    EVIDENCE_RECORDS_MAX, EmbedderConfig, ID_MAX_BYTES, JUDGE_MAX_RATIONALE_BYTES,
    JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_PROMPT_TEMPLATE_SHA256_V1, JUDGE_PROMPT_VERSION_V1,
    JUDGE_RUBRIC_TEMPLATE_SHA256_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig, LearningConfig,
    LookaheadConfig, MODEL_ID_MAX_BYTES, OUTCOME_DURATION_SECONDS_MAX,
    OUTCOME_EVALUATION_BATCH_SIZE_MAX, OUTCOME_LABEL_MAX_BYTES, OUTCOME_MATCHER_TEXT_MAX_BYTES,
    OUTCOME_MATCHERS_MAX, OUTCOME_MAX_CANARY_ROOTS_MAX, OUTCOME_MAX_LOOKS,
    OUTCOME_METADATA_EQUALS_MAX, OUTCOME_POLICY_IDENTITY_MAX_BYTES, OutcomeConfig,
    OutcomeDisposition, OutcomeMatcher, OutcomeMatcherEventKind, OutcomeTerminalStatus,
    PATH_PATTERN_MAX_BYTES, PROJECT_ID_MAX_BYTES, PoolConfig, PoolSelectorConfig,
    REVISION_MAX_BYTES, RouterConfig, RouterMode, SCHEDULER_MAX_SLOTS, SELECTOR_IDENTITY_MAX_BYTES,
};
pub use control::{
    CONTROL_ACTOR_MAX_BYTES, CONTROL_REASON_MAX_BYTES, CONTROL_TRANSACTION_START_TIMEOUT_MS_MAX,
    ControlMutation, ControlMutationOptions, ControlMutationReceipt, ControlMutationResult,
    ControlOperation, ControlRecord, ControlScope, RouterControlError, RouterControlService,
    RouterControlSnapshot, RouterControlState, RouterPoolControlSnapshot, router_control_service,
};
pub use diagnostics::{RouterConfigValidation, validate_router_config};
pub use host::{
    ROUTER_LEDGER_SCHEMA_VERSION, RouterDatabaseSchemaReport, RouterDatabaseSchemaState,
    RouterNativeVectorCapability, RouterNativeVectorCapabilityError, inspect_router_database,
    probe_native_vector_capability,
};
pub use plugin_component::{
    ComponentSpec, ROUTER_PLUGIN_KIND, deregister_router_component, register_router_component,
};
pub use routing_partition::RoutingPartitionV1;
