// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded OpenAI-compatible embedding clients and provider decoding.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay::api::runtime::{TASK_SCOPE_STACK, create_scope_stack};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError, watch};
use unicode_normalization::UnicodeNormalization;
use url::{Host, Url};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::canonical_json::canonical_json;
use crate::config::{
    EMBEDDER_REQUEST_BYTES_MAX, EMBEDDER_RESPONSE_BYTES_MAX, ID_MAX_BYTES, RouterConfig,
    embedder_response_bytes_bound,
};
use crate::embedding_identity::{
    EmbedderEgressClass, PreparedEmbedderProfile, embedder_profile_version,
};
use crate::fingerprint::sha256_hex;
use crate::ledger::repository::vector_registry::VectorRegistryEnsure;
use crate::vector::{AuthoritativeVector, NormalizedVector, VectorError, VectorSpaceId};

pub(crate) const EMBEDDER_SCOPE_NAME: &str = "nemo_relay.router.embedder";
const JSON_CONTENT_TYPE: &str = "application/json";
const BEARER_PREFIX: &str = "Bearer ";
const CONNECT_TIMEOUT_MAX: Duration = Duration::from_secs(5);
const WORK_ID_MAX_BYTES: usize = 128;
pub(crate) const EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX: usize = 256 * 1024 * 1024;
const DECODED_VECTOR_COMPONENT_BYTES_MAX: usize =
    std::mem::size_of::<f64>() + 2 * std::mem::size_of::<f32>();

/// Stable client-construction failures without endpoint, resolver, or secret text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedderBuildFailure {
    RegistryMismatch,
    MissingCredential,
    InvalidCredential,
    LocalhostResolution,
    ClientConstruction,
}

impl EmbedderBuildFailure {
    pub(crate) const fn stable_class(self) -> &'static str {
        match self {
            Self::RegistryMismatch => "embedder_registry_mismatch",
            Self::MissingCredential => "embedder_missing_credential",
            Self::InvalidCredential => "embedder_invalid_credential",
            Self::LocalhostResolution => "embedder_localhost_resolution",
            Self::ClientConstruction => "embedder_client_construction",
        }
    }
}

impl fmt::Display for EmbedderBuildFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.stable_class())
    }
}

impl std::error::Error for EmbedderBuildFailure {}

/// Whether a durable owner may retry one stable provider failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedderFailureDisposition {
    Retryable,
    Quarantine,
}

/// Stable provider failure classes that never retain response or transport text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedderFailureClass {
    UnauthorizedSpace,
    Saturated,
    InvalidBatch,
    RequestBound,
    RequestEncoding,
    ScopeRuntime,
    Connect,
    Timeout,
    Cancelled,
    Transport,
    RequestTimeout,
    RateLimited,
    Server,
    Authentication,
    ClientStatus,
    UnexpectedStatus,
    ResponseBound,
    MalformedResponse,
    ResponseCount,
    ResponseIndex,
    DimensionMismatch,
    NonFiniteVector,
    ZeroVector,
    InvalidVector,
}

impl EmbedderFailureClass {
    pub(crate) const fn stable_class(self) -> &'static str {
        match self {
            Self::UnauthorizedSpace => "embedder_unauthorized_space",
            Self::Saturated => "embedder_saturated",
            Self::InvalidBatch => "embedder_invalid_batch",
            Self::RequestBound => "embedder_request_bound",
            Self::RequestEncoding => "embedder_request_encoding",
            Self::ScopeRuntime => "embedder_scope_runtime",
            Self::Connect => "embedder_connect",
            Self::Timeout => "embedder_timeout",
            Self::Cancelled => "embedder_cancelled",
            Self::Transport => "embedder_transport",
            Self::RequestTimeout => "embedder_http_request_timeout",
            Self::RateLimited => "embedder_rate_limited",
            Self::Server => "embedder_server",
            Self::Authentication => "embedder_authentication",
            Self::ClientStatus => "embedder_client_status",
            Self::UnexpectedStatus => "embedder_unexpected_status",
            Self::ResponseBound => "embedder_response_bound",
            Self::MalformedResponse => "embedder_malformed_response",
            Self::ResponseCount => "embedder_response_count",
            Self::ResponseIndex => "embedder_response_index",
            Self::DimensionMismatch => "embedder_dimension_mismatch",
            Self::NonFiniteVector => "embedder_nonfinite_vector",
            Self::ZeroVector => "embedder_zero_vector",
            Self::InvalidVector => "embedder_invalid_vector",
        }
    }

    pub(crate) const fn disposition(self) -> EmbedderFailureDisposition {
        match self {
            Self::Saturated
            | Self::Connect
            | Self::Timeout
            | Self::Cancelled
            | Self::Transport
            | Self::RequestTimeout
            | Self::RateLimited
            | Self::Server => EmbedderFailureDisposition::Retryable,
            Self::UnauthorizedSpace
            | Self::InvalidBatch
            | Self::RequestBound
            | Self::RequestEncoding
            | Self::ScopeRuntime
            | Self::Authentication
            | Self::ClientStatus
            | Self::UnexpectedStatus
            | Self::ResponseBound
            | Self::MalformedResponse
            | Self::ResponseCount
            | Self::ResponseIndex
            | Self::DimensionMismatch
            | Self::NonFiniteVector
            | Self::ZeroVector
            | Self::InvalidVector => EmbedderFailureDisposition::Quarantine,
        }
    }
}

/// One bounded, text-free failure returned by the provider layer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct EmbedderFailure {
    class: EmbedderFailureClass,
}

impl EmbedderFailure {
    const fn new(class: EmbedderFailureClass) -> Self {
        Self { class }
    }

    pub(crate) const fn class(self) -> EmbedderFailureClass {
        self.class
    }

    pub(crate) const fn disposition(self) -> EmbedderFailureDisposition {
        self.class.disposition()
    }
}

impl fmt::Debug for EmbedderFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbedderFailure")
            .field("class", &self.class)
            .finish()
    }
}

impl fmt::Display for EmbedderFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.class.stable_class())
    }
}

impl std::error::Error for EmbedderFailure {}

/// The durable authority correlated with one HTTP batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbedderWorkKind {
    EmbeddingJob,
    Decision,
    Inspection,
}

impl EmbedderWorkKind {
    const fn metadata_key(self) -> &'static str {
        match self {
            Self::EmbeddingJob => "embedding_job_ids",
            Self::Decision => "decision_ids",
            Self::Inspection => "inspection_request_ids",
        }
    }
}

/// One verified canonical query and its bounded durable work identity.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EmbedderBatchItem {
    canonical_query_hash: String,
    canonical_query: String,
    work_id: String,
}

impl EmbedderBatchItem {
    pub(crate) fn new(
        canonical_query_hash: impl Into<String>,
        canonical_query: impl Into<String>,
        work_id: impl Into<String>,
    ) -> Result<Self, EmbedderFailure> {
        let item = Self::from_verified_artifact(canonical_query_hash, canonical_query, work_id)?;
        let value = serde_json::from_str::<Json>(&item.canonical_query)
            .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))?;
        let reproduced = canonical_json(&value)
            .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))?;
        if reproduced != item.canonical_query {
            return Err(EmbedderFailure::new(EmbedderFailureClass::InvalidBatch));
        }
        Ok(item)
    }

    pub(crate) fn from_verified_artifact(
        canonical_query_hash: impl Into<String>,
        canonical_query: impl Into<String>,
        work_id: impl Into<String>,
    ) -> Result<Self, EmbedderFailure> {
        let canonical_query_hash = canonical_query_hash.into();
        let canonical_query = canonical_query.into();
        let work_id = work_id.into();
        if !is_sha256(&canonical_query_hash)
            || sha256_hex(canonical_query.as_bytes()) != canonical_query_hash
            || !is_stable_work_id(&work_id)
        {
            return Err(EmbedderFailure::new(EmbedderFailureClass::InvalidBatch));
        }
        Ok(Self {
            canonical_query_hash,
            canonical_query,
            work_id,
        })
    }

    pub(crate) fn canonical_query_hash(&self) -> &str {
        &self.canonical_query_hash
    }

    pub(crate) fn work_id(&self) -> &str {
        &self.work_id
    }
}

impl fmt::Debug for EmbedderBatchItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbedderBatchItem")
            .field("canonical_query_hash", &self.canonical_query_hash)
            .field("canonical_query", &"<redacted>")
            .field("work_id", &self.work_id)
            .finish()
    }
}

/// Frozen clients for exactly the profiles and spaces referenced by one activation.
pub(crate) struct FrozenEmbedderClients {
    profiles: BTreeMap<String, Arc<FrozenEmbedderClient>>,
    spaces: BTreeMap<VectorSpaceId, Arc<FrozenEmbedderClient>>,
    memory_bytes: Arc<Semaphore>,
}

impl FrozenEmbedderClients {
    pub(crate) fn try_acquire(
        &self,
        vector_space_id: &VectorSpaceId,
    ) -> Result<EmbedderPermit, EmbedderFailure> {
        let client = self
            .spaces
            .get(vector_space_id)
            .cloned()
            .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::UnauthorizedSpace))?;
        let profile_permit = client
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(saturated)?;
        let reserved_bytes = embedder_memory_reservation_bytes(&client.profile)?;
        let memory_permit = self
            .memory_bytes
            .clone()
            .try_acquire_many_owned(reserved_bytes)
            .map_err(saturated)?;
        Ok(EmbedderPermit {
            client,
            vector_space_id: vector_space_id.clone(),
            profile_permit,
            memory_permit,
        })
    }

    pub(crate) fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    pub(crate) fn space_count(&self) -> usize {
        self.spaces.len()
    }
}

impl fmt::Debug for FrozenEmbedderClients {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenEmbedderClients")
            .field("profile_count", &self.profiles.len())
            .field("space_count", &self.spaces.len())
            .finish()
    }
}

/// One pre-provider permit bound to an authorized vector space.
pub(crate) struct EmbedderPermit {
    client: Arc<FrozenEmbedderClient>,
    vector_space_id: VectorSpaceId,
    profile_permit: OwnedSemaphorePermit,
    memory_permit: OwnedSemaphorePermit,
}

impl EmbedderPermit {
    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) fn max_batch_size(&self) -> usize {
        self.client.profile.batch_size
    }

    pub(crate) fn operation_timeout(&self) -> Duration {
        Duration::from_millis(self.client.profile.timeout_ms)
    }

    pub(crate) async fn execute_batch(
        self,
        work_kind: EmbedderWorkKind,
        items: Vec<EmbedderBatchItem>,
    ) -> Result<Vec<AuthoritativeVector>, EmbedderFailure> {
        self.execute_batch_inner(work_kind, items, None).await
    }

    /// Execute one batch under a supervisor-owned deadline and cancellation signal.
    pub(crate) async fn execute_batch_until(
        self,
        work_kind: EmbedderWorkKind,
        items: Vec<EmbedderBatchItem>,
        deadline: Instant,
        cancellation: watch::Receiver<bool>,
    ) -> Result<Vec<AuthoritativeVector>, EmbedderFailure> {
        self.execute_batch_inner(
            work_kind,
            items,
            Some(EmbedderExecutionBound {
                deadline,
                cancellation,
            }),
        )
        .await
    }

    async fn execute_batch_inner(
        self,
        work_kind: EmbedderWorkKind,
        items: Vec<EmbedderBatchItem>,
        bound: Option<EmbedderExecutionBound>,
    ) -> Result<Vec<AuthoritativeVector>, EmbedderFailure> {
        let body = request_body(&self.client.profile, &items)?;
        let metadata = linkage_metadata(
            &self.client.profile.profile_id,
            &self.vector_space_id,
            work_kind,
            &items,
        );
        let expected_count = items.len();
        let client = self.client.clone();
        let vector_space_id = self.vector_space_id.clone();
        let response_bytes_max =
            embedder_response_bytes_bound(expected_count, client.profile.dimensions.value())
                .filter(|bytes| *bytes <= EMBEDDER_RESPONSE_BYTES_MAX)
                .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))?;
        let profile_permit = self.profile_permit;
        let memory_permit = self.memory_permit;
        let result = TASK_SCOPE_STACK
            .scope(create_scope_stack(), async move {
                let scope = push_scope(
                    PushScopeParams::builder()
                        .name(EMBEDDER_SCOPE_NAME)
                        .scope_type(ScopeType::Embedder)
                        .metadata(metadata)
                        .build(),
                )
                .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::ScopeRuntime))?;
                let scope = EmbedderScopeGuard::new(scope.uuid);
                let provider_result = match bound {
                    Some(mut bound) => {
                        tokio::select! {
                            biased;
                            _ = wait_for_cancellation(&mut bound.cancellation) => {
                                Err(EmbedderFailure::new(EmbedderFailureClass::Cancelled))
                            }
                            () = tokio::time::sleep_until(tokio::time::Instant::from_std(bound.deadline)) => {
                                Err(EmbedderFailure::new(EmbedderFailureClass::Timeout))
                            }
                            result = execute_http(&client, &vector_space_id, body, expected_count, response_bytes_max) => result,
                        }
                    }
                    None => execute_http(
                        &client,
                        &vector_space_id,
                        body,
                        expected_count,
                        response_bytes_max,
                    )
                    .await,
                };
                scope.finish()?;
                provider_result
            })
            .await;
        drop((profile_permit, memory_permit));
        result
    }
}

fn saturated(_: TryAcquireError) -> EmbedderFailure {
    EmbedderFailure::new(EmbedderFailureClass::Saturated)
}

pub(crate) fn embedder_memory_reservation_bytes(
    profile: &PreparedEmbedderProfile,
) -> Result<u32, EmbedderFailure> {
    let response_bytes =
        embedder_response_bytes_bound(profile.batch_size, profile.dimensions.value())
            .filter(|bytes| *bytes <= EMBEDDER_RESPONSE_BYTES_MAX)
            .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))?;
    let decoded_vector_bytes = profile
        .batch_size
        .checked_mul(profile.dimensions.as_usize())
        .and_then(|components| components.checked_mul(DECODED_VECTOR_COMPONENT_BYTES_MAX))
        .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))?;
    EMBEDDER_REQUEST_BYTES_MAX
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(response_bytes))
        .and_then(|bytes| bytes.checked_add(decoded_vector_bytes))
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::InvalidBatch))
}

struct EmbedderExecutionBound {
    deadline: Instant,
    cancellation: watch::Receiver<bool>,
}

struct EmbedderScopeGuard {
    scope_uuid: Option<Uuid>,
}

impl EmbedderScopeGuard {
    fn new(scope_uuid: Uuid) -> Self {
        Self {
            scope_uuid: Some(scope_uuid),
        }
    }

    fn finish(mut self) -> Result<(), EmbedderFailure> {
        let scope_uuid = self
            .scope_uuid
            .take()
            .expect("Embedder scope guard must be armed");
        pop_scope(PopScopeParams::builder().handle_uuid(&scope_uuid).build())
            .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::ScopeRuntime))?;
        Ok(())
    }
}

impl Drop for EmbedderScopeGuard {
    fn drop(&mut self) {
        if let Some(scope_uuid) = self.scope_uuid.take() {
            let _ = pop_scope(PopScopeParams::builder().handle_uuid(&scope_uuid).build());
        }
    }
}

async fn wait_for_cancellation(cancellation: &mut watch::Receiver<bool>) {
    if *cancellation.borrow_and_update() {
        return;
    }
    loop {
        if cancellation.changed().await.is_err() || *cancellation.borrow_and_update() {
            return;
        }
    }
}

impl fmt::Debug for EmbedderPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbedderPermit")
            .field("vector_space_id", &self.vector_space_id)
            .finish_non_exhaustive()
    }
}

struct EmbedderSecret(Zeroizing<String>);

impl EmbedderSecret {
    fn new(value: String) -> Result<Self, EmbedderBuildFailure> {
        let value = Zeroizing::new(value);
        if value.is_empty() {
            return Err(EmbedderBuildFailure::MissingCredential);
        }
        if value
            .as_bytes()
            .iter()
            .any(|byte| !(0x20..=0x7e).contains(byte))
        {
            return Err(EmbedderBuildFailure::InvalidCredential);
        }
        Ok(Self(value))
    }

    fn authorization(&self) -> Result<HeaderValue, EmbedderFailure> {
        let capacity = BEARER_PREFIX
            .len()
            .checked_add(self.0.len())
            .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::RequestEncoding))?;
        let mut wire = Zeroizing::new(String::with_capacity(capacity));
        wire.push_str(BEARER_PREFIX);
        wire.push_str(self.0.as_str());
        let mut header = HeaderValue::from_str(wire.as_str())
            .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::RequestEncoding))?;
        header.set_sensitive(true);
        Ok(header)
    }
}

struct FrozenEmbedderClient {
    profile: PreparedEmbedderProfile,
    endpoint: Url,
    credential: Option<EmbedderSecret>,
    http: reqwest::Client,
    permits: Arc<Semaphore>,
}

impl fmt::Debug for FrozenEmbedderClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenEmbedderClient")
            .field("profile_id", &self.profile.profile_id)
            .field(
                "embedder_profile_version_id",
                &self.profile.embedder_profile_version_id,
            )
            .field("endpoint", &"<redacted>")
            .field(
                "credential",
                &self.credential.as_ref().map(|_| "<redacted>"),
            )
            .field("available_permits", &self.permits.available_permits())
            .finish()
    }
}

/// Build clients for referenced profiles only, resolving no unused credential.
pub(crate) fn build_frozen_embedder_clients(
    config: &RouterConfig,
    registry: &VectorRegistryEnsure,
) -> Result<FrozenEmbedderClients, EmbedderBuildFailure> {
    build_frozen_embedder_clients_with_resolvers(
        config,
        registry,
        |name| std::env::var(name).map_err(|_| ()),
        |host, port| {
            (host, port)
                .to_socket_addrs()
                .map(|addresses| addresses.collect())
                .map_err(|_| ())
        },
    )
}

#[cfg(test)]
pub(crate) fn build_frozen_embedder_clients_for_test<C, L>(
    config: &RouterConfig,
    registry: &VectorRegistryEnsure,
    credential_resolver: C,
    localhost_resolver: L,
) -> Result<FrozenEmbedderClients, EmbedderBuildFailure>
where
    C: FnMut(&str) -> Result<String, ()>,
    L: FnMut(&str, u16) -> Result<Vec<SocketAddr>, ()>,
{
    build_frozen_embedder_clients_with_resolvers(
        config,
        registry,
        credential_resolver,
        localhost_resolver,
    )
}

fn build_frozen_embedder_clients_with_resolvers<C, L>(
    config: &RouterConfig,
    registry: &VectorRegistryEnsure,
    mut credential_resolver: C,
    mut localhost_resolver: L,
) -> Result<FrozenEmbedderClients, EmbedderBuildFailure>
where
    C: FnMut(&str) -> Result<String, ()>,
    L: FnMut(&str, u16) -> Result<Vec<SocketAddr>, ()>,
{
    if config.generation_id().ok().as_deref() != Some(&registry.config_generation_id) {
        return Err(EmbedderBuildFailure::RegistryMismatch);
    }
    let mut configs = BTreeMap::new();
    for profile in &config.embedders {
        if configs.insert(profile.id.as_str(), profile).is_some() {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
    }

    let mut profiles = BTreeMap::new();
    for (version_id, prepared) in &registry.profiles {
        if version_id != &prepared.embedder_profile_version_id {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
        let profile_config = configs
            .get(prepared.profile_id.as_str())
            .ok_or(EmbedderBuildFailure::RegistryMismatch)?;
        let artifact =
            embedder_profile_version(profile_config, config.allow_remote_embedding_egress)
                .map_err(|_| EmbedderBuildFailure::RegistryMismatch)?;
        let reproduced = PreparedEmbedderProfile::from_validated(
            profile_config,
            config.allow_remote_embedding_egress,
            &artifact,
        )
        .map_err(|_| EmbedderBuildFailure::RegistryMismatch)?;
        if &reproduced != prepared {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }

        let credential = match profile_config.api_key_env.as_deref() {
            Some(name) => Some(EmbedderSecret::new(
                credential_resolver(name).map_err(|_| EmbedderBuildFailure::MissingCredential)?,
            )?),
            None => None,
        };
        let endpoint = Url::parse(&prepared.endpoint_url)
            .map_err(|_| EmbedderBuildFailure::RegistryMismatch)?;
        let timeout = Duration::from_millis(prepared.timeout_ms);
        let mut builder = reqwest::Client::builder()
            .redirect(Policy::none())
            .no_proxy()
            .retry(reqwest::retry::never())
            .connect_timeout(timeout.min(CONNECT_TIMEOUT_MAX))
            .timeout(timeout)
            .https_only(prepared.egress_class != EmbedderEgressClass::LoopbackHttp)
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd();

        if let Some(Host::Domain(host)) = endpoint.host() {
            if host.eq_ignore_ascii_case("localhost") {
                let port = endpoint
                    .port_or_known_default()
                    .ok_or(EmbedderBuildFailure::RegistryMismatch)?;
                let mut addresses = localhost_resolver(host, port)
                    .map_err(|_| EmbedderBuildFailure::LocalhostResolution)?;
                addresses.sort_unstable();
                addresses.dedup();
                if addresses.is_empty()
                    || addresses.iter().any(|address| !address.ip().is_loopback())
                {
                    return Err(EmbedderBuildFailure::LocalhostResolution);
                }
                builder = builder.resolve_to_addrs(host, &addresses);
            } else if prepared.egress_class != EmbedderEgressClass::RemoteHttps {
                return Err(EmbedderBuildFailure::RegistryMismatch);
            }
        }

        let http = builder
            .build()
            .map_err(|_| EmbedderBuildFailure::ClientConstruction)?;
        let client = Arc::new(FrozenEmbedderClient {
            profile: prepared.clone(),
            endpoint,
            credential,
            http,
            permits: Arc::new(Semaphore::new(prepared.max_in_flight)),
        });
        if profiles.insert(version_id.clone(), client).is_some() {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
    }

    let mut spaces = BTreeMap::new();
    for (space_id, space) in &registry.spaces {
        if space_id != &space.vector_space_id {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
        let client = profiles
            .get(&space.embedder_profile_version_id)
            .cloned()
            .ok_or(EmbedderBuildFailure::RegistryMismatch)?;
        if space.dimensions != client.profile.dimensions
            || space.endpoint_identity_sha256 != client.profile.endpoint_identity_sha256
            || space.model != client.profile.model
            || space.provider_revision != client.profile.provider_revision
        {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
        if spaces.insert(space_id.clone(), client).is_some() {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
    }
    for mapping in registry.mappings.values() {
        let Some(client) = spaces.get(&mapping.vector_space_id) else {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        };
        if mapping.embedder_profile_version_id != client.profile.embedder_profile_version_id
            || mapping.profile_id != client.profile.profile_id
        {
            return Err(EmbedderBuildFailure::RegistryMismatch);
        }
    }

    Ok(FrozenEmbedderClients {
        profiles,
        spaces,
        memory_bytes: Arc::new(Semaphore::new(EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX)),
    })
}

pub(crate) fn request_body(
    profile: &PreparedEmbedderProfile,
    items: &[EmbedderBatchItem],
) -> Result<Vec<u8>, EmbedderFailure> {
    if items.is_empty() || items.len() > profile.batch_size {
        return Err(EmbedderFailure::new(EmbedderFailureClass::InvalidBatch));
    }
    let mut seen_work = BTreeSet::new();
    let mut raw_input_bytes = 0_usize;
    for item in items {
        if !seen_work.insert(item.work_id.as_str()) {
            return Err(EmbedderFailure::new(EmbedderFailureClass::InvalidBatch));
        }
        raw_input_bytes = raw_input_bytes
            .checked_add(item.canonical_query.len())
            .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::RequestBound))?;
        if raw_input_bytes > EMBEDDER_REQUEST_BYTES_MAX {
            return Err(EmbedderFailure::new(EmbedderFailureClass::RequestBound));
        }
    }
    // Struct fields are declared in lexicographic order, so ordinary JSON string
    // encoding is the exact JCS form without building an owned Value tree.
    #[derive(Serialize)]
    struct ProviderEmbeddingRequest<'a> {
        encoding_format: &'static str,
        input: Vec<&'a str>,
        model: &'a str,
    }
    let request = ProviderEmbeddingRequest {
        encoding_format: "float",
        input: items
            .iter()
            .map(|item| item.canonical_query.as_str())
            .collect(),
        model: &profile.model,
    };
    let mut writer = BoundedRequestWriter::new(EMBEDDER_REQUEST_BYTES_MAX);
    request
        .serialize(&mut serde_json::Serializer::new(&mut writer))
        .map_err(|_| {
            EmbedderFailure::new(if writer.exceeded {
                EmbedderFailureClass::RequestBound
            } else {
                EmbedderFailureClass::RequestEncoding
            })
        })?;
    Ok(writer.bytes)
}

struct BoundedRequestWriter {
    bytes: Vec<u8>,
    maximum: usize,
    exceeded: bool,
}

impl BoundedRequestWriter {
    fn new(maximum: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum,
            exceeded: false,
        }
    }
}

impl std::io::Write for BoundedRequestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let next = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|next| *next <= self.maximum);
        let Some(next) = next else {
            self.exceeded = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "embedding request exceeds its byte bound",
            ));
        };
        self.bytes.reserve(next - self.bytes.len());
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn linkage_metadata(
    profile_id: &str,
    vector_space_id: &VectorSpaceId,
    work_kind: EmbedderWorkKind,
    items: &[EmbedderBatchItem],
) -> Json {
    let query_hashes = items
        .iter()
        .map(|item| item.canonical_query_hash.clone())
        .collect::<Vec<_>>();
    let work_ids = items
        .iter()
        .map(|item| item.work_id.clone())
        .collect::<Vec<_>>();
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "profile_id".to_string(),
        Json::String(profile_id.to_string()),
    );
    metadata.insert(
        "vector_space_id".to_string(),
        Json::String(vector_space_id.as_str().to_string()),
    );
    metadata.insert("canonical_query_hashes".to_string(), json!(query_hashes));
    metadata.insert(work_kind.metadata_key().to_string(), json!(work_ids));
    Json::Object(metadata)
}

async fn execute_http(
    client: &FrozenEmbedderClient,
    vector_space_id: &VectorSpaceId,
    body: Vec<u8>,
    expected_count: usize,
    response_bytes_max: usize,
) -> Result<Vec<AuthoritativeVector>, EmbedderFailure> {
    let mut request = client
        .http
        .post(client.endpoint.clone())
        .header(ACCEPT, JSON_CONTENT_TYPE)
        .header(CONTENT_TYPE, JSON_CONTENT_TYPE)
        .body(body);
    if let Some(credential) = &client.credential {
        request = request.header(AUTHORIZATION, credential.authorization()?);
    }
    let mut response = request.send().await.map_err(classify_transport)?;
    let status = response.status();
    if !status.is_success() {
        return Err(EmbedderFailure::new(classify_status(status.as_u16())));
    }
    if response
        .content_length()
        .is_some_and(|length| length > response_bytes_max as u64)
    {
        return Err(EmbedderFailure::new(EmbedderFailureClass::ResponseBound));
    }

    let capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or_default()
        .min(response_bytes_max);
    let mut response_body = Vec::with_capacity(capacity);
    while let Some(chunk) = response.chunk().await.map_err(classify_transport)? {
        let next_len = response_body
            .len()
            .checked_add(chunk.len())
            .ok_or_else(|| EmbedderFailure::new(EmbedderFailureClass::ResponseBound))?;
        if next_len > response_bytes_max {
            return Err(EmbedderFailure::new(EmbedderFailureClass::ResponseBound));
        }
        response_body.extend_from_slice(&chunk);
    }
    decode_response(client, vector_space_id, expected_count, &response_body)
}

fn classify_transport(error: reqwest::Error) -> EmbedderFailure {
    let class = if error.is_timeout() {
        EmbedderFailureClass::Timeout
    } else if error.is_connect() {
        EmbedderFailureClass::Connect
    } else {
        EmbedderFailureClass::Transport
    };
    EmbedderFailure::new(class)
}

const fn classify_status(status: u16) -> EmbedderFailureClass {
    match status {
        401 | 403 => EmbedderFailureClass::Authentication,
        408 => EmbedderFailureClass::RequestTimeout,
        429 => EmbedderFailureClass::RateLimited,
        400..=499 => EmbedderFailureClass::ClientStatus,
        500..=599 => EmbedderFailureClass::Server,
        _ => EmbedderFailureClass::UnexpectedStatus,
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEmbeddingResponse {
    data: Vec<ProviderEmbeddingData>,
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage: Option<ProviderUsage>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderEmbeddingData {
    embedding: Vec<f64>,
    index: usize,
    #[serde(default)]
    object: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderUsage {
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens_details: Option<Json>,
    prompt_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<Json>,
    total_tokens: u64,
}

fn decode_response(
    client: &FrozenEmbedderClient,
    vector_space_id: &VectorSpaceId,
    expected_count: usize,
    response: &[u8],
) -> Result<Vec<AuthoritativeVector>, EmbedderFailure> {
    let decoded = serde_json::from_slice::<ProviderEmbeddingResponse>(response)
        .map_err(|_| EmbedderFailure::new(EmbedderFailureClass::MalformedResponse))?;
    if decoded
        .object
        .as_deref()
        .is_some_and(|value| value != "list")
        || decoded.model.as_deref().is_some_and(str::is_empty)
        || decoded.usage.as_ref().is_some_and(|usage| {
            usage.total_tokens < usage.prompt_tokens
                || usage
                    .completion_tokens
                    .is_some_and(|tokens| tokens > usage.total_tokens)
                || usage
                    .completion_tokens_details
                    .as_ref()
                    .is_some_and(|details| !details.is_object())
                || usage
                    .prompt_tokens_details
                    .as_ref()
                    .is_some_and(|details| !details.is_object())
        })
    {
        return Err(EmbedderFailure::new(
            EmbedderFailureClass::MalformedResponse,
        ));
    }
    if decoded.data.len() != expected_count {
        return Err(EmbedderFailure::new(EmbedderFailureClass::ResponseCount));
    }

    let mut vectors = Vec::with_capacity(decoded.data.len());
    for (expected_index, item) in decoded.data.into_iter().enumerate() {
        if item
            .object
            .as_deref()
            .is_some_and(|value| value != "embedding")
        {
            return Err(EmbedderFailure::new(
                EmbedderFailureClass::MalformedResponse,
            ));
        }
        if item.index != expected_index {
            return Err(EmbedderFailure::new(EmbedderFailureClass::ResponseIndex));
        }
        let normalized =
            NormalizedVector::from_provider_f64(&item.embedding, client.profile.dimensions)
                .map_err(classify_vector_error)?;
        vectors.push(
            AuthoritativeVector::from_normalized(vector_space_id, normalized)
                .map_err(classify_vector_error)?,
        );
    }
    Ok(vectors)
}

fn classify_vector_error(error: VectorError) -> EmbedderFailure {
    let class = match error {
        VectorError::DimensionMismatch | VectorError::InvalidDimensions => {
            EmbedderFailureClass::DimensionMismatch
        }
        VectorError::NonFiniteValue => EmbedderFailureClass::NonFiniteVector,
        VectorError::ZeroVector => EmbedderFailureClass::ZeroVector,
        VectorError::InvalidSpaceId
        | VectorError::InvalidRecordId
        | VectorError::InvalidPartitionId
        | VectorError::VectorSpaceMismatch
        | VectorError::InvalidBlobLength
        | VectorError::InvalidChecksum
        | VectorError::ChecksumMismatch
        | VectorError::DistanceOutOfRange => EmbedderFailureClass::InvalidVector,
    };
    EmbedderFailure::new(class)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn is_stable_work_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= WORK_ID_MAX_BYTES
        && value.len() <= ID_MAX_BYTES
        && value.nfc().eq(value.chars())
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':'))
}
