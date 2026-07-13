// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure embedding endpoint and vector-space identity construction.

use std::collections::{BTreeMap, BTreeSet};

use std::fmt;

use serde_json::{Value as Json, json};
use unicode_normalization::UnicodeNormalization;
use url::{Host, Url};
use uuid::{Uuid, Variant};

use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{
    CanonicalizerConfig, EMBEDDER_AGGREGATE_WORK_ITEMS_MAX, EmbedderConfig, RouterConfig,
};
use crate::fingerprint::sha256_hex;
use crate::vector::{VectorDimensions, VectorSpaceId};

pub(crate) const CANONICAL_ROUTING_QUERY_SCHEMA_V1: &str = "nemo.relay.router.routing-query@1";
pub(crate) const CANONICAL_ROUTING_QUERY_RULES_V1: &str = "latest-task-whole-context-v1";
pub(crate) const CANONICAL_TEXT_NORMALIZATION_V1: &str = "crlf-cr-lf-horizontal-space-v1";
const EMBEDDING_ENDPOINT_SCHEMA_V1: &str = "nemo.relay.router.embedding-endpoint@1";
const EMBEDDER_PROFILE_SCHEMA_V1: &str = "nemo.relay.router.embedder-profile@1";
const CANONICALIZER_IDENTITY_SCHEMA_V1: &str = "nemo.relay.router.canonicalizer-identity@1";
const VECTOR_SPACE_SCHEMA_V1: &str = "nemo.relay.router.vector-space@1";
const POOL_VECTOR_SPACE_MAPPING_SCHEMA_V1: &str = "nemo.relay.router.pool-vector-space-mapping@1";
const OPENAI_EMBEDDINGS_PROTOCOL_V1: &str = "openai-embeddings-v1";
const VECTOR_DISTANCE_METRIC_V1: &str = "cosine";
const VECTOR_NORMALIZATION_V1: &str = "l2_f32_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedderEgressClass {
    LoopbackHttp,
    LoopbackHttps,
    RemoteHttps,
}

impl EmbedderEgressClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::LoopbackHttp => "loopback_http",
            Self::LoopbackHttps => "loopback_https",
            Self::RemoteHttps => "remote_https",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalEmbedderEndpoint {
    pub(crate) request_url: Url,
    pub(crate) endpoint_identity_sha256: String,
    pub(crate) egress_class: EmbedderEgressClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbedderProfileVersion {
    pub(crate) profile_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) endpoint: CanonicalEmbedderEndpoint,
    pub(crate) canonical_identity_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalizerVersion {
    pub(crate) canonicalizer_version_id: String,
    pub(crate) canonical_identity_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProposedVectorSpace {
    pub(crate) vector_space_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) canonicalizer_version_id: String,
    pub(crate) canonical_identity_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProposedPoolVectorSpaceMapping {
    pub(crate) pool_id: String,
    pub(crate) profile_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) canonicalizer_version_id: String,
    pub(crate) vector_space_id: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProposedVectorSpaceRegistry {
    pub(crate) profiles: BTreeMap<String, EmbedderProfileVersion>,
    pub(crate) spaces: BTreeMap<String, ProposedVectorSpace>,
    pub(crate) pools: BTreeMap<String, ProposedPoolVectorSpaceMapping>,
    pub(crate) aggregate_profile_permits: usize,
    pub(crate) aggregate_work_items: usize,
}

/// Fully revalidated nonsecret profile row prepared for durable persistence.
///
/// This type intentionally implements neither serialization nor deserialization.
#[derive(Clone, PartialEq, Eq)]
#[allow(dead_code)] // The Task 7 repository slice consumes this prepared authority.
pub(crate) struct PreparedEmbedderProfile {
    pub(crate) profile_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) protocol: &'static str,
    pub(crate) endpoint_url: String,
    pub(crate) endpoint_identity_sha256: String,
    pub(crate) model: String,
    pub(crate) provider_revision: String,
    pub(crate) dimensions: VectorDimensions,
    pub(crate) credential_env_name_sha256: Option<String>,
    pub(crate) timeout_ms: u64,
    pub(crate) max_in_flight: usize,
    pub(crate) batch_size: usize,
    pub(crate) egress_class: EmbedderEgressClass,
    pub(crate) canonical_profile_json: String,
}

impl fmt::Debug for PreparedEmbedderProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedEmbedderProfile")
            .field("profile_id", &self.profile_id)
            .field(
                "embedder_profile_version_id",
                &self.embedder_profile_version_id,
            )
            .field("endpoint_url", &"<redacted>")
            .field("endpoint_identity_sha256", &self.endpoint_identity_sha256)
            .field("dimensions", &self.dimensions)
            .field(
                "credential_env_name_sha256",
                &self
                    .credential_env_name_sha256
                    .as_ref()
                    .map(|_| "<redacted>"),
            )
            .field("canonical_profile_json", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl PreparedEmbedderProfile {
    /// Recompute and freeze every nonsecret profile column from validated config.
    #[allow(dead_code)] // The Task 7 repository slice consumes this constructor.
    pub(crate) fn from_validated(
        config: &EmbedderConfig,
        allow_remote_https: bool,
        artifact: &EmbedderProfileVersion,
    ) -> Result<Self, String> {
        let expected = embedder_profile_version(config, allow_remote_https)?;
        if artifact != &expected {
            return Err("embedder profile artifact does not match validated config".to_string());
        }
        let identity = exact_canonical_value(
            &artifact.canonical_identity_json,
            "embedder profile identity",
        )?;
        if canonical_sha256(&identity)? != artifact.embedder_profile_version_id {
            return Err("embedder profile identity hash does not match its version".to_string());
        }
        let credential_env_name_sha256 = config
            .api_key_env
            .as_deref()
            .map(|value| protected_string_sha256("router-embedder-api-key-env-v1", value));
        let expected_identity = json!({
            "schema": EMBEDDER_PROFILE_SCHEMA_V1,
            "profile_id": config.id,
            "protocol": OPENAI_EMBEDDINGS_PROTOCOL_V1,
            "endpoint_identity_sha256": artifact.endpoint.endpoint_identity_sha256,
            "model": config.model,
            "provider_revision": config.provider_revision,
            "dimensions": config.dimensions,
            "api_key_env_sha256": credential_env_name_sha256,
            "timeout_ms": config.timeout_ms,
            "max_in_flight": config.max_in_flight,
            "batch_size": config.batch_size,
            "egress_class": artifact.endpoint.egress_class.as_str(),
        });
        if identity != expected_identity {
            return Err("embedder profile identity fields are inconsistent".to_string());
        }
        let dimensions = VectorDimensions::new(config.dimensions)
            .map_err(|_| "embedder profile dimensions are invalid".to_string())?;
        Ok(Self {
            profile_id: artifact.profile_id.clone(),
            embedder_profile_version_id: artifact.embedder_profile_version_id.clone(),
            protocol: OPENAI_EMBEDDINGS_PROTOCOL_V1,
            endpoint_url: artifact.endpoint.request_url.as_str().to_string(),
            endpoint_identity_sha256: artifact.endpoint.endpoint_identity_sha256.clone(),
            model: config.model.clone(),
            provider_revision: config.provider_revision.clone(),
            dimensions,
            credential_env_name_sha256,
            timeout_ms: config.timeout_ms,
            max_in_flight: config.max_in_flight,
            batch_size: config.batch_size,
            egress_class: artifact.endpoint.egress_class,
            canonical_profile_json: artifact.canonical_identity_json.clone(),
        })
    }
}

/// Fully revalidated immutable vector-space row prepared for persistence.
///
/// This type intentionally implements neither serialization nor deserialization.
#[derive(Clone, PartialEq, Eq)]
#[allow(dead_code)] // The Task 7 repository slice consumes this prepared authority.
pub(crate) struct PreparedVectorSpace {
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) canonicalizer_version_id: String,
    pub(crate) canonicalizer_identity_json: String,
    pub(crate) endpoint_identity_sha256: String,
    pub(crate) model: String,
    pub(crate) provider_revision: String,
    pub(crate) dimensions: VectorDimensions,
    pub(crate) metric: &'static str,
    pub(crate) normalization: &'static str,
    pub(crate) canonical_space_json: String,
}

impl fmt::Debug for PreparedVectorSpace {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedVectorSpace")
            .field("vector_space_id", &self.vector_space_id)
            .field(
                "embedder_profile_version_id",
                &self.embedder_profile_version_id,
            )
            .field("canonicalizer_version_id", &self.canonicalizer_version_id)
            .field("dimensions", &self.dimensions)
            .field("canonicalizer_identity_json", &"<redacted>")
            .field("canonical_space_json", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl PreparedVectorSpace {
    /// Recompute and freeze every vector-space column from validated artifacts.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // The Task 7 repository slice consumes this constructor.
    pub(crate) fn from_validated(
        profile_config: &EmbedderConfig,
        canonicalizer_config: &CanonicalizerConfig,
        allow_remote_https: bool,
        profile_artifact: &EmbedderProfileVersion,
        canonicalizer_artifact: &CanonicalizerVersion,
        space_artifact: &ProposedVectorSpace,
    ) -> Result<Self, String> {
        let prepared_profile = PreparedEmbedderProfile::from_validated(
            profile_config,
            allow_remote_https,
            profile_artifact,
        )?;
        let expected_canonicalizer = canonicalizer_version(canonicalizer_config)?;
        if canonicalizer_artifact != &expected_canonicalizer {
            return Err("canonicalizer artifact does not match validated config".to_string());
        }
        let canonicalizer_identity = exact_canonical_value(
            &canonicalizer_artifact.canonical_identity_json,
            "canonicalizer identity",
        )?;
        if canonical_sha256(&canonicalizer_identity)?
            != canonicalizer_artifact.canonicalizer_version_id
        {
            return Err("canonicalizer identity hash does not match its version".to_string());
        }
        let expected_space =
            proposed_vector_space(profile_artifact, profile_config, canonicalizer_artifact)?;
        if space_artifact != &expected_space {
            return Err("vector-space artifact does not match validated identities".to_string());
        }
        let space_identity = exact_canonical_value(
            &space_artifact.canonical_identity_json,
            "vector-space identity",
        )?;
        if canonical_sha256(&space_identity)? != space_artifact.vector_space_id {
            return Err("vector-space identity hash does not match its ID".to_string());
        }
        let expected_identity = json!({
            "schema": VECTOR_SPACE_SCHEMA_V1,
            "embedder_profile_version_id": prepared_profile.embedder_profile_version_id,
            "endpoint_identity_sha256": prepared_profile.endpoint_identity_sha256,
            "model": prepared_profile.model,
            "provider_revision": prepared_profile.provider_revision,
            "dimensions": prepared_profile.dimensions.value(),
            "distance_metric": VECTOR_DISTANCE_METRIC_V1,
            "normalization": VECTOR_NORMALIZATION_V1,
            "canonicalizer_version_id": canonicalizer_artifact.canonicalizer_version_id,
        });
        if space_identity != expected_identity {
            return Err("vector-space identity fields are inconsistent".to_string());
        }
        let vector_space_id = VectorSpaceId::new(space_artifact.vector_space_id.clone())
            .map_err(|_| "vector-space ID is invalid".to_string())?;
        Ok(Self {
            vector_space_id,
            embedder_profile_version_id: prepared_profile.embedder_profile_version_id,
            canonicalizer_version_id: canonicalizer_artifact.canonicalizer_version_id.clone(),
            canonicalizer_identity_json: canonicalizer_artifact.canonical_identity_json.clone(),
            endpoint_identity_sha256: prepared_profile.endpoint_identity_sha256,
            model: prepared_profile.model,
            provider_revision: prepared_profile.provider_revision,
            dimensions: prepared_profile.dimensions,
            metric: VECTOR_DISTANCE_METRIC_V1,
            normalization: VECTOR_NORMALIZATION_V1,
            canonical_space_json: space_artifact.canonical_identity_json.clone(),
        })
    }
}

/// Exact frozen pool-to-space mapping prepared for durable persistence.
///
/// This type intentionally implements neither serialization nor deserialization.
#[derive(Clone, PartialEq, Eq)]
#[allow(dead_code)] // The Task 7 repository slice consumes this prepared authority.
pub(crate) struct PreparedPoolVectorSpaceMapping {
    pub(crate) project_uuid: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) pool_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) profile_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) canonicalizer_version_id: String,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) canonical_mapping_json: String,
    pub(crate) canonical_payload_hash: String,
}

impl fmt::Debug for PreparedPoolVectorSpaceMapping {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedPoolVectorSpaceMapping")
            .field("project_uuid", &self.project_uuid)
            .field("config_generation_id", &self.config_generation_id)
            .field("pool_id", &self.pool_id)
            .field("policy_version_id", &self.policy_version_id)
            .field(
                "embedder_profile_version_id",
                &self.embedder_profile_version_id,
            )
            .field("canonicalizer_version_id", &self.canonicalizer_version_id)
            .field("vector_space_id", &self.vector_space_id)
            .field("canonical_mapping_json", &"<redacted>")
            .field("canonical_payload_hash", &self.canonical_payload_hash)
            .finish()
    }
}

impl PreparedPoolVectorSpaceMapping {
    /// Bind one validated proposed mapping to exact ledger configuration authority.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // The Task 7 repository slice consumes this constructor.
    pub(crate) fn from_validated(
        project_uuid: Uuid,
        config_generation_id: impl Into<String>,
        policy_version_id: impl Into<String>,
        artifact: &ProposedPoolVectorSpaceMapping,
        profile: &PreparedEmbedderProfile,
        space: &PreparedVectorSpace,
    ) -> Result<Self, String> {
        if project_uuid.get_version_num() != 7 || project_uuid.get_variant() != Variant::RFC4122 {
            return Err("mapping project UUID is invalid".to_string());
        }
        let config_generation_id = config_generation_id.into();
        let policy_version_id = policy_version_id.into();
        validate_sha256_identity(&config_generation_id, "configuration generation")?;
        validate_sha256_identity(&policy_version_id, "policy version")?;
        if artifact.pool_id.is_empty() || artifact.pool_id.len() > 128 {
            return Err("mapping pool ID is invalid".to_string());
        }
        if artifact.profile_id != profile.profile_id
            || artifact.embedder_profile_version_id != profile.embedder_profile_version_id
            || artifact.embedder_profile_version_id != space.embedder_profile_version_id
            || artifact.canonicalizer_version_id != space.canonicalizer_version_id
            || artifact.vector_space_id != space.vector_space_id.as_str()
        {
            return Err(
                "pool mapping artifact is inconsistent with prepared identities".to_string(),
            );
        }
        let mapping = json!({
            "schema": POOL_VECTOR_SPACE_MAPPING_SCHEMA_V1,
            "project_uuid": project_uuid,
            "config_generation_id": config_generation_id,
            "pool_id": artifact.pool_id,
            "policy_version_id": policy_version_id,
            "profile_id": artifact.profile_id,
            "embedder_profile_version_id": artifact.embedder_profile_version_id,
            "canonicalizer_version_id": artifact.canonicalizer_version_id,
            "vector_space_id": artifact.vector_space_id,
        });
        let canonical_mapping_json = canonical_json(&mapping)?;
        let canonical_payload_hash = canonical_sha256(&mapping)?;
        Ok(Self {
            project_uuid,
            config_generation_id,
            pool_id: artifact.pool_id.clone(),
            policy_version_id,
            profile_id: artifact.profile_id.clone(),
            embedder_profile_version_id: artifact.embedder_profile_version_id.clone(),
            canonicalizer_version_id: artifact.canonicalizer_version_id.clone(),
            vector_space_id: space.vector_space_id.clone(),
            canonical_mapping_json,
            canonical_payload_hash,
        })
    }
}

pub(crate) fn normalize_embedder_endpoint(
    value: &str,
    allow_remote_https: bool,
) -> Result<CanonicalEmbedderEndpoint, String> {
    if value.trim().is_empty()
        || value.chars().any(char::is_control)
        || !value.nfc().eq(value.chars())
    {
        return Err(
            "embedding endpoint must be nonblank, control-free, and NFC normalized".to_string(),
        );
    }
    if !has_unambiguous_http_authority(value) {
        return Err("embedding endpoint contains forbidden URL components".to_string());
    }

    let url = Url::parse(value)
        .map_err(|_| "embedding endpoint is not a valid absolute URL".to_string())?;
    if url.cannot_be_a_base() {
        return Err("embedding endpoint must be a hierarchical absolute URL".to_string());
    }
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("embedding endpoint contains forbidden URL components".to_string());
    }
    let host = url
        .host()
        .ok_or_else(|| "embedding endpoint must include a host".to_string())?;
    let loopback = match host {
        Host::Domain(domain) => domain.eq_ignore_ascii_case("localhost"),
        Host::Ipv4(address) => address.is_loopback(),
        Host::Ipv6(address) => address.is_loopback(),
    };
    let egress_class = match (url.scheme(), loopback, allow_remote_https) {
        ("http", true, _) => EmbedderEgressClass::LoopbackHttp,
        ("https", true, _) => EmbedderEgressClass::LoopbackHttps,
        ("https", false, true) => EmbedderEgressClass::RemoteHttps,
        _ => return Err("embedding endpoint violates the configured egress policy".to_string()),
    };

    if has_noncanonical_percent_encoding(url.path()) {
        return Err("embedding endpoint path contains noncanonical percent encoding".to_string());
    }
    let prefix_path = url.path().trim_end_matches('/');
    if prefix_path
        .rsplit('/')
        .next()
        .is_some_and(|segment| segment == "embeddings")
    {
        return Err("embedding base URL must not include the embeddings suffix".to_string());
    }

    let request_url = Url::parse(&format!(
        "{}/embeddings",
        url.as_str().trim_end_matches('/')
    ))
    .map_err(|_| "embedding endpoint could not be normalized".to_string())?;
    let identity = json!({
        "schema": EMBEDDING_ENDPOINT_SCHEMA_V1,
        "origin": request_url.origin().ascii_serialization(),
        "path": request_url.path(),
    });
    Ok(CanonicalEmbedderEndpoint {
        request_url,
        endpoint_identity_sha256: canonical_sha256(&identity)?,
        egress_class,
    })
}

fn has_unambiguous_http_authority(value: &str) -> bool {
    if value.contains('\\') {
        return false;
    }
    let Some((scheme, remainder)) = value.split_once(':') else {
        return false;
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return false;
    }
    let Some(remainder) = remainder.strip_prefix("//") else {
        return false;
    };
    if remainder.starts_with('/') {
        return false;
    }
    remainder
        .split(['/', '?', '#'])
        .next()
        .is_some_and(|authority| !authority.is_empty() && !authority.contains('@'))
}

fn has_noncanonical_percent_encoding(path: &str) -> bool {
    let bytes = path.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        let Some(high) = bytes.get(index + 1).and_then(|value| hex_value(*value)) else {
            return true;
        };
        let Some(low) = bytes.get(index + 2).and_then(|value| hex_value(*value)) else {
            return true;
        };
        let decoded = high * 16 + low;
        if decoded.is_ascii_alphanumeric()
            || matches!(decoded, b'-' | b'.' | b'_' | b'~' | b'/' | b'\\')
        {
            return true;
        }
        index += 3;
    }
    false
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn embedder_profile_version(
    config: &EmbedderConfig,
    allow_remote_https: bool,
) -> Result<EmbedderProfileVersion, String> {
    let endpoint = normalize_embedder_endpoint(&config.base_url, allow_remote_https)?;
    let identity = json!({
        "schema": EMBEDDER_PROFILE_SCHEMA_V1,
        "profile_id": config.id,
        "protocol": OPENAI_EMBEDDINGS_PROTOCOL_V1,
        "endpoint_identity_sha256": endpoint.endpoint_identity_sha256,
        "model": config.model,
        "provider_revision": config.provider_revision,
        "dimensions": config.dimensions,
        "api_key_env_sha256": config.api_key_env.as_deref().map(|value| {
            protected_string_sha256("router-embedder-api-key-env-v1", value)
        }),
        "timeout_ms": config.timeout_ms,
        "max_in_flight": config.max_in_flight,
        "batch_size": config.batch_size,
        "egress_class": endpoint.egress_class.as_str(),
    });
    Ok(EmbedderProfileVersion {
        profile_id: config.id.clone(),
        embedder_profile_version_id: canonical_sha256(&identity)?,
        endpoint,
        canonical_identity_json: canonical_json(&identity)?,
    })
}

pub(crate) fn canonicalizer_version(
    config: &CanonicalizerConfig,
) -> Result<CanonicalizerVersion, String> {
    let mut features = config.position_features.clone();
    features.sort();
    let identity = json!({
        "schema": CANONICALIZER_IDENTITY_SCHEMA_V1,
        "query_schema": CANONICAL_ROUTING_QUERY_SCHEMA_V1,
        "query_rules": CANONICAL_ROUTING_QUERY_RULES_V1,
        "text_normalization": CANONICAL_TEXT_NORMALIZATION_V1,
        "version": config.version,
        "max_instruction_bytes": config.max_instruction_bytes,
        "max_task_bytes": config.max_task_bytes,
        "max_context_messages": config.max_context_messages,
        "max_context_bytes": config.max_context_bytes,
        "max_position_features_bytes": config.max_position_features_bytes,
        "position_features": features,
    });
    Ok(CanonicalizerVersion {
        canonicalizer_version_id: canonical_sha256(&identity)?,
        canonical_identity_json: canonical_json(&identity)?,
    })
}

fn proposed_vector_space(
    profile: &EmbedderProfileVersion,
    profile_config: &EmbedderConfig,
    canonicalizer: &CanonicalizerVersion,
) -> Result<ProposedVectorSpace, String> {
    let identity = json!({
        "schema": VECTOR_SPACE_SCHEMA_V1,
        "embedder_profile_version_id": profile.embedder_profile_version_id,
        "endpoint_identity_sha256": profile.endpoint.endpoint_identity_sha256,
        "model": profile_config.model,
        "provider_revision": profile_config.provider_revision,
        "dimensions": profile_config.dimensions,
        "distance_metric": VECTOR_DISTANCE_METRIC_V1,
        "normalization": VECTOR_NORMALIZATION_V1,
        "canonicalizer_version_id": canonicalizer.canonicalizer_version_id,
    });
    Ok(ProposedVectorSpace {
        vector_space_id: canonical_sha256(&identity)?,
        embedder_profile_version_id: profile.embedder_profile_version_id.clone(),
        canonicalizer_version_id: canonicalizer.canonicalizer_version_id.clone(),
        canonical_identity_json: canonical_json(&identity)?,
    })
}

pub(crate) fn propose_vector_space_registry(
    config: &RouterConfig,
) -> Result<ProposedVectorSpaceRegistry, String> {
    let profiles_by_id = config
        .embedders
        .iter()
        .map(|profile| (profile.id.as_str(), profile))
        .collect::<BTreeMap<_, _>>();
    let mut registry = ProposedVectorSpaceRegistry::default();
    let mut counted_profile_versions = BTreeSet::new();

    for pool in &config.pools {
        let Some(learning) = &pool.learning else {
            continue;
        };
        if learning.version != 1 {
            return Err(format!(
                "pool '{}' has unsupported learning version {}",
                pool.id, learning.version
            ));
        }
        let profile_config = profiles_by_id
            .get(learning.embedder.as_str())
            .ok_or_else(|| {
                format!(
                    "pool '{}' references unknown embedder '{}'",
                    pool.id, learning.embedder
                )
            })?;
        let profile =
            embedder_profile_version(profile_config, config.allow_remote_embedding_egress)?;
        let canonicalizer = canonicalizer_version(&pool.canonicalizer)?;
        let space = proposed_vector_space(&profile, profile_config, &canonicalizer)?;
        let mapping = ProposedPoolVectorSpaceMapping {
            pool_id: pool.id.clone(),
            profile_id: profile.profile_id.clone(),
            embedder_profile_version_id: profile.embedder_profile_version_id.clone(),
            canonicalizer_version_id: canonicalizer.canonicalizer_version_id,
            vector_space_id: space.vector_space_id.clone(),
        };

        insert_or_verify(
            &mut registry.profiles,
            profile.embedder_profile_version_id.clone(),
            profile.clone(),
            "embedder profile",
        )?;
        insert_or_verify(
            &mut registry.spaces,
            space.vector_space_id.clone(),
            space,
            "vector space",
        )?;
        if registry.pools.insert(pool.id.clone(), mapping).is_some() {
            return Err(format!("duplicate proposed mapping for pool '{}'", pool.id));
        }

        if counted_profile_versions.insert(profile.embedder_profile_version_id) {
            registry.aggregate_profile_permits = registry
                .aggregate_profile_permits
                .checked_add(profile_config.max_in_flight)
                .ok_or_else(|| "aggregate embedder permits overflow usize".to_string())?;
            let work_items = profile_config
                .max_in_flight
                .checked_mul(profile_config.batch_size)
                .ok_or_else(|| "embedder work-item capacity overflow usize".to_string())?;
            registry.aggregate_work_items = registry
                .aggregate_work_items
                .checked_add(work_items)
                .ok_or_else(|| {
                "aggregate embedder work-item capacity overflow usize".to_string()
            })?;
        }
    }

    if registry.aggregate_profile_permits > tokio::sync::Semaphore::MAX_PERMITS {
        return Err(format!(
            "aggregate embedder permits {} exceed {}",
            registry.aggregate_profile_permits,
            tokio::sync::Semaphore::MAX_PERMITS
        ));
    }
    if registry.aggregate_work_items > EMBEDDER_AGGREGATE_WORK_ITEMS_MAX {
        return Err(format!(
            "aggregate embedder work items {} exceed {}",
            registry.aggregate_work_items, EMBEDDER_AGGREGATE_WORK_ITEMS_MAX
        ));
    }

    Ok(registry)
}

fn insert_or_verify<T: PartialEq>(
    entries: &mut BTreeMap<String, T>,
    identity: String,
    value: T,
    kind: &str,
) -> Result<(), String> {
    if let Some(existing) = entries.get(&identity) {
        if existing != &value {
            return Err(format!("{kind} identity collision for {identity}"));
        }
        return Ok(());
    }
    entries.insert(identity, value);
    Ok(())
}

fn protected_string_sha256(domain: &str, value: &str) -> String {
    let mut preimage = Vec::with_capacity(domain.len() + value.len() + 1);
    preimage.extend_from_slice(domain.as_bytes());
    preimage.push(0);
    preimage.extend_from_slice(value.as_bytes());
    sha256_hex(&preimage)
}

fn exact_canonical_value(value: &str, kind: &str) -> Result<Json, String> {
    let parsed =
        serde_json::from_str::<Json>(value).map_err(|_| format!("{kind} is not valid JSON"))?;
    if canonical_json(&parsed)? != value {
        return Err(format!("{kind} is not exact canonical JSON"));
    }
    Ok(parsed)
}

fn validate_sha256_identity(value: &str, kind: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{kind} identity is not lowercase SHA-256"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::config::{LearningConfig, RouterConfig};

    fn config() -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "embedders": [{
                "id": "embedding-main",
                "base_url": "http://localhost:8080/v1/",
                "model": "embed-model",
                "provider_revision": "revision-1",
                "dimensions": 1024,
                "api_key_env": "ROUTER_EMBEDDING_API_KEY",
                "timeout_ms": 10000,
                "max_in_flight": 4,
                "batch_size": 16
            }],
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-model"],
                "anchor_revision": "revision-1",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 1,
                "concurrency": {"shadow": 2, "judge": 1},
                "candidates": [{
                    "id": "candidate-a",
                    "model": "candidate-model",
                    "model_revision": "revision-1",
                    "cost_rank": 0
                }],
                "judge": {
                    "version": 1,
                    "model": "judge-model",
                    "model_revision": "revision-1",
                    "prompt_version": "pairwise-equivalence-v1",
                    "rubric_version": "response-trajectory-equivalence-v1",
                    "output_schema_version": 1,
                    "response_weight": 0.5,
                    "trajectory_weight": 0.5,
                    "response_floor": 0.8,
                    "trajectory_floor": 0.8,
                    "judge_confidence_floor": 0.7,
                    "pass_threshold": 0.85,
                    "max_rationale_bytes": 4096,
                    "base_cooloff_seconds": 10,
                    "max_cooloff_seconds": 300
                },
                "learning": {"version": 1, "embedder": "embedding-main"}
            }]
        }))
        .unwrap()
    }

    fn prepared_identities() -> (
        RouterConfig,
        EmbedderProfileVersion,
        CanonicalizerVersion,
        ProposedVectorSpace,
        PreparedEmbedderProfile,
        PreparedVectorSpace,
    ) {
        let config = config();
        let profile_artifact = embedder_profile_version(&config.embedders[0], false).unwrap();
        let canonicalizer_artifact = canonicalizer_version(&config.pools[0].canonicalizer).unwrap();
        let space_artifact = proposed_vector_space(
            &profile_artifact,
            &config.embedders[0],
            &canonicalizer_artifact,
        )
        .unwrap();
        let profile =
            PreparedEmbedderProfile::from_validated(&config.embedders[0], false, &profile_artifact)
                .unwrap();
        let space = PreparedVectorSpace::from_validated(
            &config.embedders[0],
            &config.pools[0].canonicalizer,
            false,
            &profile_artifact,
            &canonicalizer_artifact,
            &space_artifact,
        )
        .unwrap();
        (
            config,
            profile_artifact,
            canonicalizer_artifact,
            space_artifact,
            profile,
            space,
        )
    }

    #[test]
    fn prepared_rows_revalidate_every_durable_identity_field() {
        let (config, profile_artifact, canonicalizer_artifact, space_artifact, profile, space) =
            prepared_identities();
        let registry = propose_vector_space_registry(&config).unwrap();
        let mapping_artifact = &registry.pools["pool-a"];
        let project_uuid = Uuid::now_v7();
        let config_generation_id = config.generation_id().unwrap();
        let policy_value = &config.policy_generation_values().unwrap()["pool-a"];
        let policy_version_id = canonical_sha256(policy_value).unwrap();
        let mapping = PreparedPoolVectorSpaceMapping::from_validated(
            project_uuid,
            config_generation_id.clone(),
            policy_version_id.clone(),
            mapping_artifact,
            &profile,
            &space,
        )
        .unwrap();

        assert_eq!(profile.profile_id, config.embedders[0].id);
        assert_eq!(profile.protocol, OPENAI_EMBEDDINGS_PROTOCOL_V1);
        assert_eq!(profile.endpoint_url, "http://localhost:8080/v1/embeddings");
        assert_eq!(profile.model, config.embedders[0].model);
        assert_eq!(
            profile.provider_revision,
            config.embedders[0].provider_revision
        );
        assert_eq!(profile.dimensions.value(), config.embedders[0].dimensions);
        assert_eq!(profile.timeout_ms, config.embedders[0].timeout_ms);
        assert_eq!(profile.max_in_flight, config.embedders[0].max_in_flight);
        assert_eq!(profile.batch_size, config.embedders[0].batch_size);
        assert_eq!(profile.egress_class, EmbedderEgressClass::LoopbackHttp);
        assert_eq!(
            profile.credential_env_name_sha256.as_deref(),
            Some(
                protected_string_sha256(
                    "router-embedder-api-key-env-v1",
                    "ROUTER_EMBEDDING_API_KEY"
                )
                .as_str()
            )
        );
        assert_eq!(
            canonical_sha256(
                &serde_json::from_str::<Json>(&profile.canonical_profile_json).unwrap()
            )
            .unwrap(),
            profile_artifact.embedder_profile_version_id
        );

        assert_eq!(
            space.vector_space_id.as_str(),
            space_artifact.vector_space_id
        );
        assert_eq!(space.dimensions, profile.dimensions);
        assert_eq!(space.metric, VECTOR_DISTANCE_METRIC_V1);
        assert_eq!(space.normalization, VECTOR_NORMALIZATION_V1);
        assert_eq!(
            canonical_sha256(
                &serde_json::from_str::<Json>(&space.canonicalizer_identity_json).unwrap()
            )
            .unwrap(),
            canonicalizer_artifact.canonicalizer_version_id
        );
        assert_eq!(
            canonical_sha256(&serde_json::from_str::<Json>(&space.canonical_space_json).unwrap())
                .unwrap(),
            space.vector_space_id.as_str()
        );

        assert_eq!(mapping.project_uuid, project_uuid);
        assert_eq!(mapping.config_generation_id, config_generation_id);
        assert_eq!(mapping.policy_version_id, policy_version_id);
        assert_eq!(mapping.vector_space_id, space.vector_space_id);
        let mapping_value = serde_json::from_str::<Json>(&mapping.canonical_mapping_json).unwrap();
        assert_eq!(
            canonical_json(&mapping_value).unwrap(),
            mapping.canonical_mapping_json
        );
        assert_eq!(
            canonical_sha256(&mapping_value).unwrap(),
            mapping.canonical_payload_hash
        );
        assert_eq!(mapping_value["schema"], POOL_VECTOR_SPACE_MAPPING_SCHEMA_V1);
    }

    #[test]
    fn prepared_rows_reject_substituted_or_noncanonical_artifacts() {
        let (config, profile_artifact, canonicalizer_artifact, space_artifact, profile, space) =
            prepared_identities();

        let mut noncanonical_profile = profile_artifact.clone();
        noncanonical_profile.canonical_identity_json.push(' ');
        assert!(
            PreparedEmbedderProfile::from_validated(
                &config.embedders[0],
                false,
                &noncanonical_profile
            )
            .is_err()
        );

        let mut substituted_space = space_artifact.clone();
        substituted_space.vector_space_id = "a".repeat(64);
        assert!(
            PreparedVectorSpace::from_validated(
                &config.embedders[0],
                &config.pools[0].canonicalizer,
                false,
                &profile_artifact,
                &canonicalizer_artifact,
                &substituted_space,
            )
            .is_err()
        );

        let mut substituted_mapping =
            propose_vector_space_registry(&config).unwrap().pools["pool-a"].clone();
        substituted_mapping.profile_id = "other-profile".to_string();
        assert!(
            PreparedPoolVectorSpaceMapping::from_validated(
                Uuid::now_v7(),
                config.generation_id().unwrap(),
                canonical_sha256(&config.policy_generation_values().unwrap()["pool-a"]).unwrap(),
                &substituted_mapping,
                &profile,
                &space,
            )
            .is_err()
        );
        assert!(
            PreparedPoolVectorSpaceMapping::from_validated(
                Uuid::new_v4(),
                config.generation_id().unwrap(),
                canonical_sha256(&config.policy_generation_values().unwrap()["pool-a"]).unwrap(),
                &propose_vector_space_registry(&config).unwrap().pools["pool-a"],
                &profile,
                &space,
            )
            .is_err()
        );
    }

    #[test]
    fn prepared_rows_are_serde_free_and_debug_redacts_protected_documents() {
        let (config, _, _, _, profile, space) = prepared_identities();
        let proposal = &propose_vector_space_registry(&config).unwrap().pools["pool-a"];
        let mapping = PreparedPoolVectorSpaceMapping::from_validated(
            Uuid::now_v7(),
            config.generation_id().unwrap(),
            canonical_sha256(&config.policy_generation_values().unwrap()["pool-a"]).unwrap(),
            proposal,
            &profile,
            &space,
        )
        .unwrap();

        for debug in [
            format!("{profile:?}"),
            format!("{space:?}"),
            format!("{mapping:?}"),
        ] {
            assert!(!debug.contains("ROUTER_EMBEDDING_API_KEY"));
            assert!(!debug.contains("http://localhost"));
            assert!(!debug.contains("embed-model"));
            assert!(!debug.contains("nemo.relay.router.embedder-profile@1"));
        }
        assert!(format!("{profile:?}").contains("<redacted>"));
        assert!(format!("{space:?}").contains("<redacted>"));
        assert!(format!("{mapping:?}").contains("<redacted>"));
    }

    #[test]
    fn endpoint_normalization_is_canonical_and_rejects_ambiguous_paths() {
        let variants = [
            "http://LOCALHOST:80/v1",
            "http://localhost/v1/",
            "http://localhost/v1///",
        ];
        let endpoints = variants
            .into_iter()
            .map(|value| normalize_embedder_endpoint(value, false).unwrap())
            .collect::<Vec<_>>();
        assert!(endpoints.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(
            endpoints[0].request_url.as_str(),
            "http://localhost/v1/embeddings"
        );
        assert_eq!(
            normalize_embedder_endpoint("http://localhost", false)
                .unwrap()
                .request_url
                .as_str(),
            "http://localhost/embeddings"
        );

        for invalid in [
            "http://localhost/v1/embeddings",
            "http://localhost/v1/embeddings/",
            "http://localhost/v1/%65mbeddings",
            "http://localhost/v1%2Fprivate",
            "http://localhost/v1%5cprivate",
            "http://@localhost/v1",
            "http:\\@localhost\\v1",
            "http:/\\@localhost/v1",
            "http:////@localhost/v1",
            "http://user:secret@localhost/v1",
            "http://localhost/v1?secret=true",
            "http://localhost/v1#fragment",
            "http://localhost.example/v1",
            "http://api.example.com/v1",
        ] {
            assert!(
                normalize_embedder_endpoint(invalid, false).is_err(),
                "{invalid}"
            );
        }
        assert!(normalize_embedder_endpoint("http://api.example.com/v1", true).is_err());
        assert!(normalize_embedder_endpoint("https://api.example.com/v1", false).is_err());
        assert!(normalize_embedder_endpoint("https://api.example.com/v1", true).is_ok());
        assert!(normalize_embedder_endpoint("http://localhost/v1/%3F", false).is_ok());
    }

    #[test]
    fn identity_goldens_and_safe_preimages_are_stable() {
        let config = config();
        let profile = embedder_profile_version(&config.embedders[0], false).unwrap();
        let canonicalizer = canonicalizer_version(&config.pools[0].canonicalizer).unwrap();
        let space = proposed_vector_space(&profile, &config.embedders[0], &canonicalizer).unwrap();

        assert_eq!(
            (
                profile.embedder_profile_version_id.as_str(),
                canonicalizer.canonicalizer_version_id.as_str(),
                space.vector_space_id.as_str(),
            ),
            (
                "f3a32a3e5b56882ee3ef9295acc806b57df1d294367852573c1fb41e26b0b702",
                "b87665b2de1a6e7582ebba8a4935eb8d38f9a9501e83a6c589c8b30e1b08222e",
                "4cd71d724ff52cebf83767b0c0810ff670d2e4b17264cedaeae2cf1f3bf96d04",
            )
        );
        assert!(
            !profile
                .canonical_identity_json
                .contains("ROUTER_EMBEDDING_API_KEY")
        );
        assert!(!profile.canonical_identity_json.contains("localhost"));
        assert!(!profile.canonical_identity_json.contains("/v1"));
        assert!(
            profile
                .canonical_identity_json
                .contains("api_key_env_sha256")
        );
    }

    #[test]
    fn every_profile_and_canonicalizer_field_is_identity_sensitive() {
        let baseline_config = config();
        let baseline_profile =
            embedder_profile_version(&baseline_config.embedders[0], false).unwrap();
        let baseline_canonicalizer =
            canonicalizer_version(&baseline_config.pools[0].canonicalizer).unwrap();
        let baseline_space = proposed_vector_space(
            &baseline_profile,
            &baseline_config.embedders[0],
            &baseline_canonicalizer,
        )
        .unwrap();
        let mut profile_changes = Vec::new();
        for mutate in [
            |profile: &mut EmbedderConfig| profile.id = "embedding-other".into(),
            |profile: &mut EmbedderConfig| profile.base_url = "http://localhost:8080/v2".into(),
            |profile: &mut EmbedderConfig| profile.model = "other-model".into(),
            |profile: &mut EmbedderConfig| profile.provider_revision = "revision-2".into(),
            |profile: &mut EmbedderConfig| profile.dimensions = 2048,
            |profile: &mut EmbedderConfig| profile.api_key_env = Some("OTHER_KEY".into()),
            |profile: &mut EmbedderConfig| profile.timeout_ms = 11000,
            |profile: &mut EmbedderConfig| profile.max_in_flight = 5,
            |profile: &mut EmbedderConfig| profile.batch_size = 17,
        ] {
            let mut changed = baseline_config.embedders[0].clone();
            mutate(&mut changed);
            let version = embedder_profile_version(&changed, false).unwrap();
            profile_changes.push((changed, version));
        }
        let mut https = baseline_config.embedders[0].clone();
        https.base_url = "https://localhost:8080/v1".into();
        let https_version = embedder_profile_version(&https, false).unwrap();
        profile_changes.push((https, https_version));
        assert!(profile_changes.iter().all(|(_, changed)| {
            changed.embedder_profile_version_id != baseline_profile.embedder_profile_version_id
        }));
        assert!(profile_changes.iter().all(|(config, changed)| {
            proposed_vector_space(changed, config, &baseline_canonicalizer)
                .unwrap()
                .vector_space_id
                != baseline_space.vector_space_id
        }));

        let equivalent = {
            let mut config = baseline_config.embedders[0].clone();
            config.base_url = "http://LOCALHOST:8080/v1///".into();
            embedder_profile_version(&config, false).unwrap()
        };
        assert_eq!(
            equivalent.embedder_profile_version_id,
            baseline_profile.embedder_profile_version_id
        );

        let mut changes = Vec::new();
        for mutate in [
            |value: &mut CanonicalizerConfig| value.version = 2,
            |value: &mut CanonicalizerConfig| value.max_instruction_bytes += 1,
            |value: &mut CanonicalizerConfig| value.max_task_bytes += 1,
            |value: &mut CanonicalizerConfig| value.max_context_messages += 1,
            |value: &mut CanonicalizerConfig| value.max_context_bytes += 1,
            |value: &mut CanonicalizerConfig| value.max_position_features_bytes += 1,
        ] {
            let mut changed = baseline_config.pools[0].canonicalizer.clone();
            mutate(&mut changed);
            changes.push(canonicalizer_version(&changed).unwrap());
        }
        let mut with_position = baseline_config.pools[0].canonicalizer.clone();
        with_position.position_features.push("turn_index".into());
        changes.push(canonicalizer_version(&with_position).unwrap());
        assert!(changes.iter().all(|changed| {
            changed.canonicalizer_version_id != baseline_canonicalizer.canonicalizer_version_id
        }));
        assert!(changes.iter().all(|changed| {
            proposed_vector_space(&baseline_profile, &baseline_config.embedders[0], changed)
                .unwrap()
                .vector_space_id
                != baseline_space.vector_space_id
        }));
    }

    #[test]
    fn registry_is_referenced_only_and_deduplicates_spaces() {
        let mut config = config();
        let mut unused = config.embedders[0].clone();
        unused.id = "unused-profile".into();
        unused.api_key_env = Some("MISSING_UNUSED_SECRET".into());
        config.embedders.push(unused);

        let mut second = config.pools[0].clone();
        second.id = "pool-b".into();
        config.pools.push(second);
        let registry = propose_vector_space_registry(&config).unwrap();
        assert_eq!(registry.pools.len(), 2);
        assert_eq!(registry.profiles.len(), 1);
        assert_eq!(registry.spaces.len(), 1);
        assert_eq!(registry.aggregate_profile_permits, 4);
        assert_eq!(registry.aggregate_work_items, 64);
        assert!(
            registry
                .profiles
                .values()
                .all(|profile| profile.profile_id != "unused-profile")
        );

        let mut reordered = config.clone();
        reordered.embedders.reverse();
        reordered.pools.reverse();
        assert_eq!(propose_vector_space_registry(&reordered).unwrap(), registry);

        config.pools[1].canonicalizer.max_task_bytes += 1;
        let split = propose_vector_space_registry(&config).unwrap();
        assert_eq!(split.spaces.len(), 2);

        config.pools[1].learning = None;
        let disabled = propose_vector_space_registry(&config).unwrap();
        assert_eq!(disabled.pools.len(), 1);
        assert_eq!(disabled.spaces.len(), 1);
    }

    #[test]
    fn registry_rejects_aggregate_work_above_the_fixed_bound() {
        let mut config = config();
        config.embedders[0].max_in_flight = 256;
        config.embedders[0].batch_size = 128;
        for index in 1..3 {
            let mut profile = config.embedders[0].clone();
            profile.id = format!("embedding-{index}");
            config.embedders.push(profile);

            let mut pool = config.pools[0].clone();
            pool.id = format!("pool-{index}");
            pool.learning = Some(LearningConfig::minimal(format!("embedding-{index}")));
            config.pools.push(pool);
        }

        assert_eq!(
            propose_vector_space_registry(&config).unwrap_err(),
            "aggregate embedder work items 98304 exceed 65536"
        );
    }

    #[test]
    fn association_changes_policy_but_profile_details_do_not() {
        let mut config = config();
        let mut second = config.embedders[0].clone();
        second.id = "embedding-second".into();
        config.embedders.push(second);

        let baseline_config_id = config.generation_id().unwrap();
        let baseline_policy = config.policy_generation_values().unwrap();
        let generation = config.generation_value().unwrap();
        let canonicalizer = &generation["pools"][0]["canonicalizer"];
        assert_eq!(
            canonicalizer["identity"]["query_schema"],
            CANONICAL_ROUTING_QUERY_SCHEMA_V1
        );
        assert_eq!(
            canonicalizer["identity"]["query_rules"],
            CANONICAL_ROUTING_QUERY_RULES_V1
        );
        assert_eq!(
            canonicalizer["identity"]["text_normalization"],
            CANONICAL_TEXT_NORMALIZATION_V1
        );
        assert_eq!(
            canonicalizer["canonicalizer_version_id"]
                .as_str()
                .unwrap()
                .len(),
            64
        );
        let policy_json = canonical_json(&baseline_policy["pool-a"]).unwrap();
        assert!(policy_json.contains(CANONICAL_ROUTING_QUERY_RULES_V1));
        assert!(policy_json.contains(CANONICAL_TEXT_NORMALIZATION_V1));

        let mut profile_change = config.clone();
        profile_change.embedders[0].timeout_ms += 1;
        assert_ne!(profile_change.generation_id().unwrap(), baseline_config_id);
        assert_eq!(
            profile_change.policy_generation_values().unwrap(),
            baseline_policy
        );

        let mut association_change = config;
        association_change.pools[0].learning = Some(LearningConfig::minimal("embedding-second"));
        assert_ne!(
            association_change.policy_generation_values().unwrap(),
            baseline_policy
        );
    }
}
