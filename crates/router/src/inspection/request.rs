// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure request-file validation and exact frozen embedder authority selection.

use std::collections::BTreeMap;

use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, build_canonical_routing_query};
use crate::config::RouterConfig;
use crate::inspection::{
    InspectionError, ROUTING_INSPECTION_INPUT_SCHEMA_V1, RoutingInspectionInputV1,
};
use crate::ledger::repository::inspection::InspectionAuthoritySnapshot;
use crate::ledger::repository::vector_registry::{FrozenPoolVectorAuthority, VectorRegistryEnsure};
use crate::projection::validate_external_request_projection;
use crate::routing_partition::{RoutingPartitionArtifactV1, artifact_from_routing_partition_v1};
use crate::vector::VectorSpaceId;

/// Fully validated nonsecret request identity and its single referenced provider authority.
pub(crate) struct PreparedRequestInspection {
    pub(crate) pool_id: String,
    pub(crate) partition: RoutingPartitionArtifactV1,
    pub(crate) canonical_query: CanonicalRoutingQueryArtifactV1,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) registry: VectorRegistryEnsure,
}

pub(crate) fn prepare_request_inspection(
    config: &RouterConfig,
    authority: &InspectionAuthoritySnapshot,
    input: RoutingInspectionInputV1,
) -> Result<PreparedRequestInspection, InspectionError> {
    if input.schema != ROUTING_INSPECTION_INPUT_SCHEMA_V1 {
        return Err(InspectionError::InvalidArgument);
    }
    validate_external_request_projection(&input.request)
        .map_err(|_| InspectionError::InvalidArgument)?;
    let partition = artifact_from_routing_partition_v1(&input.partition)
        .map_err(|_| InspectionError::InvalidArgument)?;
    let pool = config
        .pools
        .iter()
        .find(|pool| pool.id == input.pool_id)
        .ok_or(InspectionError::InvalidArgument)?;
    let learning = pool
        .learning
        .as_ref()
        .and_then(|learning| learning.complete_policy())
        .ok_or(InspectionError::InvalidArgument)?;
    let expected_policy = authority
        .policy_version_ids
        .get(&pool.id)
        .ok_or(InspectionError::IntegrityError)?;
    let expected_learning = authority
        .learning_generation_ids
        .get(&pool.id)
        .ok_or(InspectionError::IntegrityError)?;
    let verified = match authority
        .vector_authorities
        .get(&pool.id)
        .ok_or(InspectionError::IntegrityError)?
    {
        FrozenPoolVectorAuthority::Enabled(verified) => verified,
        FrozenPoolVectorAuthority::Disabled => return Err(InspectionError::InvalidArgument),
    };
    let expected_evaluator = pool
        .judge
        .evaluator_version()
        .map_err(|_| InspectionError::IntegrityError)?;
    let candidate_matches = pool.candidates.iter().filter(|candidate| {
        candidate.id == input.partition.candidate_id
            && candidate.model == input.partition.candidate_model
            && candidate.model_revision == input.partition.candidate_model_revision
    });
    if candidate_matches.count() != 1
        || input.request.family != pool.api_family
        || input.request.family != input.partition.api_family
        || input.request.normalized_request.model.as_deref()
            != Some(input.partition.anchor_model.as_str())
        || !pool.anchor_models.contains(&input.partition.anchor_model)
        || input.partition.anchor_revision != pool.anchor_revision
        || input.partition.policy_version_id != *expected_policy
        || input.partition.learning_generation_id != *expected_learning
        || input.partition.tenant_policy_hash != input.routing_context.tenant_policy_hash
        || input.partition.agent_policy_hash != input.routing_context.agent_policy_hash
        || input.partition.evaluator_version != expected_evaluator
        || input.partition.vector_space_id != verified.mapping.vector_space_id.as_str()
    {
        return Err(InspectionError::InvalidArgument);
    }
    if verified.mapping.project_uuid != authority.project_uuid
        || verified.mapping.config_generation_id != authority.config_generation_id
        || verified.mapping.pool_id != pool.id
        || verified.mapping.policy_version_id != *expected_policy
        || verified.mapping.profile_id != learning.embedder
        || verified.profile.profile.profile_id != learning.embedder
        || verified.mapping.embedder_profile_version_id
            != verified.profile.profile.embedder_profile_version_id
        || verified.mapping.vector_space_id != verified.space.space.vector_space_id
        || verified.space.space.embedder_profile_version_id
            != verified.profile.profile.embedder_profile_version_id
        || verified.canonicalizer != pool.canonicalizer
    {
        return Err(InspectionError::IntegrityError);
    }

    let canonical_query = build_canonical_routing_query(
        &input.request,
        &input.routing_context,
        &verified.canonicalizer,
    )
    .map_err(|_| InspectionError::InvalidArgument)?;
    let vector_space_id = verified.mapping.vector_space_id.clone();
    let embedder_profile_version_id = verified.profile.profile.embedder_profile_version_id.clone();
    let mut spaces = BTreeMap::new();
    let mut mappings = BTreeMap::new();
    for (authority_pool_id, candidate) in &authority.vector_authorities {
        let FrozenPoolVectorAuthority::Enabled(candidate) = candidate else {
            continue;
        };
        if candidate.profile.profile.embedder_profile_version_id != embedder_profile_version_id {
            continue;
        }
        if candidate.profile.profile != verified.profile.profile
            || candidate.mapping.pool_id != *authority_pool_id
            || spaces
                .insert(
                    candidate.space.space.vector_space_id.clone(),
                    candidate.space.space.clone(),
                )
                .is_some_and(|existing| existing != candidate.space.space)
            || mappings
                .insert(authority_pool_id.clone(), candidate.mapping.clone())
                .is_some()
        {
            return Err(InspectionError::IntegrityError);
        }
    }
    if !spaces.contains_key(&vector_space_id) || !mappings.contains_key(&pool.id) {
        return Err(InspectionError::IntegrityError);
    }
    let registry = VectorRegistryEnsure {
        project_uuid: authority.project_uuid,
        config_generation_id: authority.config_generation_id.clone(),
        profiles: BTreeMap::from([(
            verified.profile.profile.embedder_profile_version_id.clone(),
            verified.profile.profile.clone(),
        )]),
        spaces,
        mappings,
        created_at_unix_ms: verified.created_at_unix_ms,
    };
    Ok(PreparedRequestInspection {
        pool_id: pool.id.clone(),
        partition,
        canonical_query,
        vector_space_id,
        embedder_profile_version_id,
        registry,
    })
}
