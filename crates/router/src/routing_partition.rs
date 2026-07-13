// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure strict-partition identity construction for canonical vector search.

use nemo_relay_types::api::llm::LlmApiFamily;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use uuid::{Uuid, Variant};

use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{CANDIDATE_ID_MAX_BYTES, MODEL_ID_MAX_BYTES, REVISION_MAX_BYTES};
use crate::ledger::repository::shadow::{ReservedShadowAttempt, SampleBatchReservation};
use crate::preflight::contains_sensitive_control_material;
use crate::projection::{
    REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, projection_semantic_fingerprint,
};

const TRANSPORT_IDENTITY_MAX_BYTES: usize = 256;

/// Exact version-1 partition applied before any vector-distance search.
///
/// Field declaration order follows the contract. RFC 8785 canonical JSON sorts
/// object keys lexicographically when producing the authoritative bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingPartitionV1 {
    /// Versioned tenant-policy partition hash.
    pub tenant_policy_hash: String,
    /// Versioned agent-policy partition hash.
    pub agent_policy_hash: String,
    /// Exact pool policy version.
    pub policy_version_id: String,
    /// Current per-pool learning generation.
    pub learning_generation_id: Uuid,
    /// Authoritative LLM API family.
    pub api_family: LlmApiFamily,
    /// Stable replay transport identity.
    pub transport_identity: String,
    /// Anchor model identifier.
    pub anchor_model: String,
    /// Pinned anchor model revision.
    pub anchor_revision: String,
    /// Candidate identifier.
    pub candidate_id: String,
    /// Candidate model identifier.
    pub candidate_model: String,
    /// Pinned candidate model revision.
    pub candidate_model_revision: String,
    /// Canonical decoding-policy fingerprint.
    pub decoding_fingerprint: String,
    /// Exact evaluator policy version.
    pub evaluator_version: String,
    /// Exact embedding vector-space identity.
    pub vector_space_id: String,
}

/// Candidate-independent strict facts shared by one live decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RoutingPartitionBaseV1 {
    pub(crate) tenant_policy_hash: String,
    pub(crate) agent_policy_hash: String,
    pub(crate) policy_version_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) api_family: LlmApiFamily,
    pub(crate) transport_identity: String,
    pub(crate) anchor_model: String,
    pub(crate) anchor_revision: String,
    pub(crate) evaluator_version: String,
    pub(crate) vector_space_id: String,
}

/// Validated live or frozen inputs for one exact candidate partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingPartitionInputV1 {
    pub(crate) base: RoutingPartitionBaseV1,
    pub(crate) candidate_id: String,
    pub(crate) candidate_model: String,
    pub(crate) candidate_model_revision: String,
    pub(crate) decoding_fingerprint: String,
}

/// Canonical document and digest used for repository allocation and collision checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingPartitionArtifactV1 {
    pub(crate) partition: RoutingPartitionV1,
    pub(crate) canonical_json: String,
    pub(crate) partition_hash: String,
}

/// Canonical common-field document and digest stored with a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingPartitionBaseArtifactV1 {
    pub(crate) base: RoutingPartitionBaseV1,
    pub(crate) canonical_json: String,
    pub(crate) partition_base_hash: String,
}

/// Result of comparing a proposed partition with a stored hash and full document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingPartitionComparison {
    /// Hash and full canonical JSON both match.
    Exact,
    /// The hash differs, so the documents belong to different partitions.
    Different,
    /// The hash matches but the full canonical JSON differs.
    HashCollision,
}

/// Stable construction failures that never include configuration values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoutingPartitionError {
    AttemptNotReserved,
    IneligibleAttempt,
    InvalidHash(&'static str),
    InvalidText(&'static str),
    InvalidUuid(&'static str),
    InconsistentFrozenFact(&'static str),
    Canonicalization,
}

impl RoutingPartitionArtifactV1 {
    /// Compare a stored identity using both its digest and authoritative document.
    pub(crate) fn compare(
        &self,
        stored_partition_hash: &str,
        stored_canonical_json: &str,
    ) -> RoutingPartitionComparison {
        if stored_partition_hash != self.partition_hash {
            RoutingPartitionComparison::Different
        } else if stored_canonical_json == self.canonical_json {
            RoutingPartitionComparison::Exact
        } else {
            RoutingPartitionComparison::HashCollision
        }
    }
}

/// Build one strict partition from already-frozen durable facts and an active space.
pub(crate) fn build_routing_partition_v1(
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    active_vector_space_id: &str,
) -> Result<RoutingPartitionArtifactV1, RoutingPartitionError> {
    if reservation
        .attempts
        .iter()
        .filter(|reserved| *reserved == attempt)
        .count()
        != 1
    {
        return Err(RoutingPartitionError::AttemptNotReserved);
    }
    if !attempt.eligible {
        return Err(RoutingPartitionError::IneligibleAttempt);
    }

    let input = RoutingPartitionInputV1 {
        base: RoutingPartitionBaseV1 {
            tenant_policy_hash: attempt.tenant_policy_hash.clone(),
            agent_policy_hash: attempt.agent_policy_hash.clone(),
            policy_version_id: reservation.policy_version_id.clone(),
            learning_generation_id: reservation.learning_generation_id,
            api_family: attempt.api_family,
            transport_identity: attempt.transport_identity.clone(),
            anchor_model: attempt.anchor_model.clone(),
            anchor_revision: attempt.anchor_model_revision.clone(),
            evaluator_version: attempt.evaluator_version.clone(),
            vector_space_id: active_vector_space_id.to_string(),
        },
        candidate_id: attempt.candidate_id.clone(),
        candidate_model: attempt.candidate_model.clone(),
        candidate_model_revision: attempt.candidate_model_revision.clone(),
        decoding_fingerprint: attempt.decoding_fingerprint.clone(),
    };
    let artifact = build_routing_partition_from_input_v1(&input)?;
    validate_frozen_request(attempt)?;
    Ok(artifact)
}

/// Build the candidate-independent partition base for one live decision.
pub(crate) fn build_routing_partition_base_v1(
    base: &RoutingPartitionBaseV1,
) -> Result<RoutingPartitionBaseArtifactV1, RoutingPartitionError> {
    validate_partition_base(base)?;
    let value = serde_json::to_value(base).map_err(|_| RoutingPartitionError::Canonicalization)?;
    let canonical_json =
        canonical_json(&value).map_err(|_| RoutingPartitionError::Canonicalization)?;
    let partition_base_hash =
        canonical_sha256(&value).map_err(|_| RoutingPartitionError::Canonicalization)?;
    Ok(RoutingPartitionBaseArtifactV1 {
        base: base.clone(),
        canonical_json,
        partition_base_hash,
    })
}

/// Build one exact partition from validated live or durable facts.
pub(crate) fn build_routing_partition_from_input_v1(
    input: &RoutingPartitionInputV1,
) -> Result<RoutingPartitionArtifactV1, RoutingPartitionError> {
    validate_partition_base(&input.base)?;
    validate_hash(&input.decoding_fingerprint, "decoding_fingerprint")?;
    validate_stable_id(&input.candidate_id, CANDIDATE_ID_MAX_BYTES, "candidate_id")?;
    validate_provider_identifier(&input.candidate_model, "candidate_model")?;
    validate_revision(&input.candidate_model_revision, "candidate_model_revision")?;

    let partition = RoutingPartitionV1 {
        tenant_policy_hash: input.base.tenant_policy_hash.clone(),
        agent_policy_hash: input.base.agent_policy_hash.clone(),
        policy_version_id: input.base.policy_version_id.clone(),
        learning_generation_id: input.base.learning_generation_id,
        api_family: input.base.api_family,
        transport_identity: input.base.transport_identity.clone(),
        anchor_model: input.base.anchor_model.clone(),
        anchor_revision: input.base.anchor_revision.clone(),
        candidate_id: input.candidate_id.clone(),
        candidate_model: input.candidate_model.clone(),
        candidate_model_revision: input.candidate_model_revision.clone(),
        decoding_fingerprint: input.decoding_fingerprint.clone(),
        evaluator_version: input.base.evaluator_version.clone(),
        vector_space_id: input.base.vector_space_id.clone(),
    };
    let value =
        serde_json::to_value(&partition).map_err(|_| RoutingPartitionError::Canonicalization)?;
    let canonical_json =
        canonical_json(&value).map_err(|_| RoutingPartitionError::Canonicalization)?;
    let partition_hash =
        canonical_sha256(&value).map_err(|_| RoutingPartitionError::Canonicalization)?;
    Ok(RoutingPartitionArtifactV1 {
        partition,
        canonical_json,
        partition_hash,
    })
}

/// Revalidate a deserialized public partition and rebuild its canonical identity.
pub(crate) fn artifact_from_routing_partition_v1(
    partition: &RoutingPartitionV1,
) -> Result<RoutingPartitionArtifactV1, RoutingPartitionError> {
    let artifact = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
        base: RoutingPartitionBaseV1 {
            tenant_policy_hash: partition.tenant_policy_hash.clone(),
            agent_policy_hash: partition.agent_policy_hash.clone(),
            policy_version_id: partition.policy_version_id.clone(),
            learning_generation_id: partition.learning_generation_id,
            api_family: partition.api_family,
            transport_identity: partition.transport_identity.clone(),
            anchor_model: partition.anchor_model.clone(),
            anchor_revision: partition.anchor_revision.clone(),
            evaluator_version: partition.evaluator_version.clone(),
            vector_space_id: partition.vector_space_id.clone(),
        },
        candidate_id: partition.candidate_id.clone(),
        candidate_model: partition.candidate_model.clone(),
        candidate_model_revision: partition.candidate_model_revision.clone(),
        decoding_fingerprint: partition.decoding_fingerprint.clone(),
    })?;
    if artifact.partition != *partition {
        return Err(RoutingPartitionError::InconsistentFrozenFact(
            "public_partition",
        ));
    }
    Ok(artifact)
}

fn validate_partition_base(base: &RoutingPartitionBaseV1) -> Result<(), RoutingPartitionError> {
    validate_hash(&base.tenant_policy_hash, "tenant_policy_hash")?;
    validate_hash(&base.agent_policy_hash, "agent_policy_hash")?;
    validate_hash(&base.policy_version_id, "policy_version_id")?;
    validate_hash(&base.evaluator_version, "evaluator_version")?;
    validate_hash(&base.vector_space_id, "vector_space_id")?;
    validate_uuid_v7(base.learning_generation_id, "learning_generation_id")?;
    validate_provider_identifier(&base.anchor_model, "anchor_model")?;
    validate_revision(&base.anchor_revision, "anchor_revision")?;
    validate_transport_identity(&base.transport_identity)
}

fn validate_frozen_request(attempt: &ReservedShadowAttempt) -> Result<(), RoutingPartitionError> {
    let projection = &attempt.request_projection;
    if projection.schema != REQUEST_PROJECTION_SCHEMA_V1 {
        return Err(RoutingPartitionError::InconsistentFrozenFact(
            "request_projection_schema",
        ));
    }
    if projection.sanitizer_version != ROUTER_SANITIZER_VERSION {
        return Err(RoutingPartitionError::InconsistentFrozenFact(
            "request_projection_sanitizer",
        ));
    }
    if projection.family != attempt.api_family {
        return Err(RoutingPartitionError::InconsistentFrozenFact("api_family"));
    }
    if projection.normalized_request.model.as_deref() != Some(attempt.candidate_model.as_str()) {
        return Err(RoutingPartitionError::InconsistentFrozenFact(
            "candidate_model",
        ));
    }

    let expected = projection_semantic_fingerprint(projection)
        .map_err(|_| RoutingPartitionError::InconsistentFrozenFact("request_projection_bounds"))?;
    if projection.semantic_request_fingerprint != expected {
        return Err(RoutingPartitionError::InconsistentFrozenFact(
            "semantic_request_fingerprint",
        ));
    }
    Ok(())
}

fn validate_hash(value: &str, field: &'static str) -> Result<(), RoutingPartitionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RoutingPartitionError::InvalidHash(field));
    }
    Ok(())
}

fn validate_uuid_v7(value: Uuid, field: &'static str) -> Result<(), RoutingPartitionError> {
    if value.get_version_num() != 7 || value.get_variant() != Variant::RFC4122 {
        return Err(RoutingPartitionError::InvalidUuid(field));
    }
    Ok(())
}

fn validate_normalized_text(
    value: &str,
    max_bytes: usize,
    field: &'static str,
) -> Result<(), RoutingPartitionError> {
    if value.trim().is_empty()
        || value.len() > max_bytes
        || value.chars().any(char::is_control)
        || !value.nfc().eq(value.chars())
    {
        return Err(RoutingPartitionError::InvalidText(field));
    }
    Ok(())
}

fn validate_stable_id(
    value: &str,
    max_bytes: usize,
    field: &'static str,
) -> Result<(), RoutingPartitionError> {
    validate_normalized_text(value, max_bytes, field)?;
    let mut characters = value.chars();
    if !characters.next().is_some_and(char::is_alphanumeric)
        || !characters.all(|character| {
            character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        })
    {
        return Err(RoutingPartitionError::InvalidText(field));
    }
    Ok(())
}

fn validate_provider_identifier(
    value: &str,
    field: &'static str,
) -> Result<(), RoutingPartitionError> {
    validate_normalized_text(value, MODEL_ID_MAX_BYTES, field)?;
    let mut characters = value.chars();
    if !characters.next().is_some_and(char::is_alphanumeric)
        || !characters.all(|character| {
            character.is_alphanumeric() || matches!(character, '/' | ':' | '.' | '_' | '-')
        })
    {
        return Err(RoutingPartitionError::InvalidText(field));
    }
    Ok(())
}

fn validate_revision(value: &str, field: &'static str) -> Result<(), RoutingPartitionError> {
    validate_stable_id(value, REVISION_MAX_BYTES, field)?;
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "latest" | "current" | "unversioned" | "*"
    ) {
        return Err(RoutingPartitionError::InvalidText(field));
    }
    Ok(())
}

fn validate_transport_identity(value: &str) -> Result<(), RoutingPartitionError> {
    if value.is_empty()
        || value.len() > TRANSPORT_IDENTITY_MAX_BYTES
        || value.bytes().any(|byte| !byte.is_ascii_graphic())
        || contains_sensitive_control_material(value)
    {
        return Err(RoutingPartitionError::InvalidText("transport_identity"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::fingerprint::fingerprint_json;
    use crate::projection::{RouterRequestProjectionV1, SanitizedAnnotatedLlmRequest};

    fn hash(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn uuid(value: &str) -> Uuid {
        Uuid::parse_str(value).unwrap()
    }

    fn request_projection(family: LlmApiFamily, model: &str) -> RouterRequestProjectionV1 {
        let mut projection = RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some(model.to_string()),
                params: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                truncation: None,
                reasoning: None,
                service_tier: None,
                parallel_tool_calls: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
            },
            ordered_instructions: Vec::new(),
            response_format: None,
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            sanitizer_version: ROUTER_SANITIZER_VERSION,
            semantic_request_fingerprint: String::new(),
        };
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = fingerprint_json(&value).unwrap();
        projection
    }

    fn reservation() -> SampleBatchReservation {
        let attempt = ReservedShadowAttempt::new(
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a1"),
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a2"),
            "candidate-a",
            "candidate-model",
            "candidate-r1",
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-v1",
            "anchor-model",
            "anchor-r1",
            hash('d'),
            hash('e'),
            hash('a'),
            hash('b'),
            true,
            request_projection(LlmApiFamily::OpenAIChatCompletions, "candidate-model"),
            1,
        )
        .unwrap();
        SampleBatchReservation::new(
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a3"),
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a4"),
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a5"),
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a6"),
            hash('f'),
            hash('c'),
            uuid("01890f31-a2b3-7c4d-8e5f-0123456789a7"),
            "pool-a",
            vec![attempt],
            1,
        )
        .unwrap()
    }

    fn build(
        reservation: &SampleBatchReservation,
        vector_space_id: &str,
    ) -> RoutingPartitionArtifactV1 {
        build_routing_partition_v1(reservation, &reservation.attempts[0], vector_space_id).unwrap()
    }

    fn refresh_projection(attempt: &mut ReservedShadowAttempt) {
        attempt.request_projection =
            request_projection(attempt.api_family, &attempt.candidate_model);
    }

    #[test]
    fn partition_canonical_json_and_hash_match_golden() {
        let reservation = reservation();
        let artifact = build(&reservation, &hash('6'));
        let expected = concat!(
            "{\"agent_policy_hash\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",",
            "\"anchor_model\":\"anchor-model\",\"anchor_revision\":\"anchor-r1\",",
            "\"api_family\":\"openai_chat_completions\",\"candidate_id\":\"candidate-a\",",
            "\"candidate_model\":\"candidate-model\",\"candidate_model_revision\":\"candidate-r1\",",
            "\"decoding_fingerprint\":\"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\",",
            "\"evaluator_version\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",",
            "\"learning_generation_id\":\"01890f31-a2b3-7c4d-8e5f-0123456789a7\",",
            "\"policy_version_id\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",",
            "\"tenant_policy_hash\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",",
            "\"transport_identity\":\"transport-v1\",",
            "\"vector_space_id\":\"6666666666666666666666666666666666666666666666666666666666666666\"}"
        );
        assert_eq!(artifact.canonical_json, expected);
        assert_eq!(
            artifact.partition_hash,
            "d7e770601162faf7ae79df89585b79ae49a5dc3deae6258b54db8b18924734bc"
        );
        assert_eq!(
            serde_json::to_value(&artifact.partition).unwrap(),
            json!({
                "tenant_policy_hash": hash('a'),
                "agent_policy_hash": hash('b'),
                "policy_version_id": hash('c'),
                "learning_generation_id": "01890f31-a2b3-7c4d-8e5f-0123456789a7",
                "api_family": "openai_chat_completions",
                "transport_identity": "transport-v1",
                "anchor_model": "anchor-model",
                "anchor_revision": "anchor-r1",
                "candidate_id": "candidate-a",
                "candidate_model": "candidate-model",
                "candidate_model_revision": "candidate-r1",
                "decoding_fingerprint": hash('d'),
                "evaluator_version": hash('e'),
                "vector_space_id": hash('6'),
            })
        );
    }

    #[test]
    fn partition_base_excludes_only_candidate_specific_fields() {
        let partition = build(&reservation(), &hash('6')).partition;
        let base = RoutingPartitionBaseV1 {
            tenant_policy_hash: partition.tenant_policy_hash,
            agent_policy_hash: partition.agent_policy_hash,
            policy_version_id: partition.policy_version_id,
            learning_generation_id: partition.learning_generation_id,
            api_family: partition.api_family,
            transport_identity: partition.transport_identity,
            anchor_model: partition.anchor_model,
            anchor_revision: partition.anchor_revision,
            evaluator_version: partition.evaluator_version,
            vector_space_id: partition.vector_space_id,
        };
        let artifact = build_routing_partition_base_v1(&base).unwrap();
        assert_eq!(
            artifact.canonical_json,
            concat!(
                "{\"agent_policy_hash\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",",
                "\"anchor_model\":\"anchor-model\",\"anchor_revision\":\"anchor-r1\",",
                "\"api_family\":\"openai_chat_completions\",",
                "\"evaluator_version\":\"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee\",",
                "\"learning_generation_id\":\"01890f31-a2b3-7c4d-8e5f-0123456789a7\",",
                "\"policy_version_id\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",",
                "\"tenant_policy_hash\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",",
                "\"transport_identity\":\"transport-v1\",",
                "\"vector_space_id\":\"6666666666666666666666666666666666666666666666666666666666666666\"}"
            )
        );
        assert_eq!(
            artifact.partition_base_hash,
            "7a504ce74023271de2010dc5d43d0135dcdbd91d566958ab88fdaaf8210fd617"
        );

        let mut candidate_change = RoutingPartitionInputV1 {
            base,
            candidate_id: "candidate-a".to_string(),
            candidate_model: "candidate-model".to_string(),
            candidate_model_revision: "candidate-r1".to_string(),
            decoding_fingerprint: hash('d'),
        };
        let first = build_routing_partition_from_input_v1(&candidate_change).unwrap();
        candidate_change.candidate_id = "candidate-b".to_string();
        let second = build_routing_partition_from_input_v1(&candidate_change).unwrap();
        assert_ne!(first.partition_hash, second.partition_hash);
        assert_eq!(
            artifact,
            build_routing_partition_base_v1(&candidate_change.base).unwrap()
        );
    }

    #[test]
    fn every_partition_field_changes_the_full_identity() {
        let baseline_reservation = reservation();
        let baseline = build(&baseline_reservation, &hash('6'));
        let mut changed = Vec::new();

        macro_rules! mutate_attempt {
            ($mutation:expr) => {{
                let mut reservation = reservation();
                ($mutation)(&mut reservation.attempts[0]);
                refresh_projection(&mut reservation.attempts[0]);
                changed.push(build(&reservation, &hash('6')));
            }};
        }

        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.tenant_policy_hash = hash('1')
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.agent_policy_hash = hash('2')
        );
        let mut policy = reservation();
        policy.policy_version_id = hash('3');
        changed.push(build(&policy, &hash('6')));
        let mut learning = reservation();
        learning.learning_generation_id = uuid("01890f31-a2b3-7c4d-8e5f-0123456789b7");
        changed.push(build(&learning, &hash('6')));
        mutate_attempt!(|attempt: &mut ReservedShadowAttempt| {
            attempt.api_family = LlmApiFamily::OpenAIResponses;
        });
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.transport_identity =
                "transport-v2".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.anchor_model = "anchor-model-2".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.anchor_model_revision =
                "anchor-r2".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_id = "candidate-b".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_model =
                "candidate-model-2".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_model_revision =
                "candidate-r2".into()
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.decoding_fingerprint = hash('4')
        );
        mutate_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.evaluator_version = hash('5')
        );
        changed.push(build(&reservation(), &hash('7')));

        assert_eq!(changed.len(), 14);
        for artifact in changed {
            assert_ne!(artifact.partition_hash, baseline.partition_hash);
            assert_ne!(artifact.canonical_json, baseline.canonical_json);
        }
    }

    #[test]
    fn invalid_frozen_inputs_are_rejected() {
        let baseline_reservation = reservation();
        let mut unrelated = baseline_reservation.attempts[0].clone();
        unrelated.candidate_id = "candidate-other".into();
        assert_eq!(
            build_routing_partition_v1(&baseline_reservation, &unrelated, &hash('6')),
            Err(RoutingPartitionError::AttemptNotReserved)
        );

        macro_rules! assert_invalid_attempt {
            ($mutation:expr, $expected:expr) => {{
                let mut invalid = reservation();
                ($mutation)(&mut invalid.attempts[0]);
                assert_eq!(
                    build_routing_partition_v1(&invalid, &invalid.attempts[0], &hash('6')),
                    Err($expected)
                );
            }};
        }

        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.tenant_policy_hash = "A".repeat(64),
            RoutingPartitionError::InvalidHash("tenant_policy_hash")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.agent_policy_hash = "short".into(),
            RoutingPartitionError::InvalidHash("agent_policy_hash")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.decoding_fingerprint = hash('g'),
            RoutingPartitionError::InvalidHash("decoding_fingerprint")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.evaluator_version = hash('G'),
            RoutingPartitionError::InvalidHash("evaluator_version")
        );

        let mut invalid_policy = reservation();
        invalid_policy.policy_version_id = hash('z');
        assert_eq!(
            build_routing_partition_v1(&invalid_policy, &invalid_policy.attempts[0], &hash('6'),),
            Err(RoutingPartitionError::InvalidHash("policy_version_id"))
        );

        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_id = "-candidate".into(),
            RoutingPartitionError::InvalidText("candidate_id")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.anchor_model = "bad model".into(),
            RoutingPartitionError::InvalidText("anchor_model")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.anchor_model_revision = "latest".into(),
            RoutingPartitionError::InvalidText("anchor_revision")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_model = "bad model".into(),
            RoutingPartitionError::InvalidText("candidate_model")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.candidate_model_revision =
                "current".into(),
            RoutingPartitionError::InvalidText("candidate_model_revision")
        );
        assert_invalid_attempt!(
            |attempt: &mut ReservedShadowAttempt| attempt.transport_identity =
                "bad transport".into(),
            RoutingPartitionError::InvalidText("transport_identity")
        );

        let mut invalid_uuid = reservation();
        invalid_uuid.learning_generation_id = Uuid::nil();
        assert_eq!(
            build_routing_partition_v1(&invalid_uuid, &invalid_uuid.attempts[0], &hash('6'),),
            Err(RoutingPartitionError::InvalidUuid("learning_generation_id"))
        );
        let mut wrong_variant = reservation();
        let mut bytes = *wrong_variant.learning_generation_id.as_bytes();
        bytes[8] &= 0x3f;
        wrong_variant.learning_generation_id = Uuid::from_bytes(bytes);
        assert_eq!(wrong_variant.learning_generation_id.get_version_num(), 7);
        assert_ne!(
            wrong_variant.learning_generation_id.get_variant(),
            Variant::RFC4122
        );
        assert_eq!(
            build_routing_partition_v1(&wrong_variant, &wrong_variant.attempts[0], &hash('6'),),
            Err(RoutingPartitionError::InvalidUuid("learning_generation_id"))
        );

        let mut ineligible = reservation();
        ineligible.attempts[0].eligible = false;
        assert_eq!(
            build_routing_partition_v1(&ineligible, &ineligible.attempts[0], &hash('6')),
            Err(RoutingPartitionError::IneligibleAttempt)
        );

        let mut corrupt_projection = reservation();
        corrupt_projection.attempts[0]
            .request_projection
            .semantic_request_fingerprint = hash('0');
        assert_eq!(
            build_routing_partition_v1(
                &corrupt_projection,
                &corrupt_projection.attempts[0],
                &hash('6'),
            ),
            Err(RoutingPartitionError::InconsistentFrozenFact(
                "semantic_request_fingerprint"
            ))
        );

        let mut mismatched_family = reservation();
        mismatched_family.attempts[0].request_projection.family = LlmApiFamily::OpenAIResponses;
        assert_eq!(
            build_routing_partition_v1(
                &mismatched_family,
                &mismatched_family.attempts[0],
                &hash('6'),
            ),
            Err(RoutingPartitionError::InconsistentFrozenFact("api_family"))
        );

        let mut mismatched_model = reservation();
        mismatched_model.attempts[0]
            .request_projection
            .normalized_request
            .model = Some("other-model".into());
        assert_eq!(
            build_routing_partition_v1(
                &mismatched_model,
                &mismatched_model.attempts[0],
                &hash('6'),
            ),
            Err(RoutingPartitionError::InconsistentFrozenFact(
                "candidate_model"
            ))
        );

        assert_eq!(
            build_routing_partition_v1(
                &baseline_reservation,
                &baseline_reservation.attempts[0],
                "short",
            ),
            Err(RoutingPartitionError::InvalidHash("vector_space_id"))
        );
    }

    #[test]
    fn comparison_distinguishes_exact_difference_and_hash_collision() {
        let reservation = reservation();
        let artifact = build(&reservation, &hash('6'));
        assert_eq!(
            artifact.compare(&artifact.partition_hash, &artifact.canonical_json),
            RoutingPartitionComparison::Exact
        );
        assert_eq!(
            artifact.compare(&hash('9'), &artifact.canonical_json),
            RoutingPartitionComparison::Different
        );
        assert_eq!(
            artifact.compare(&artifact.partition_hash, "{}"),
            RoutingPartitionComparison::HashCollision
        );
    }
}
