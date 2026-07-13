// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::json::Json;
use reqwest::redirect::Policy;
use serde::Deserialize;
use url::Url;
use zeroize::Zeroizing;

use super::*;
use crate::preflight::contains_sensitive_control_material;

const RUN_ENV: &str = "NEMO_RELAY_RUN_ROUTER_LIVE_PROVIDER_TESTS";
const CONFIG_ENV: &str = "NEMO_RELAY_ROUTER_LIVE_PROVIDER_CONFIG";
const PROVIDER_TIMEOUT: Duration = Duration::from_secs(120);
const PROVIDER_RESPONSE_LIMIT: usize = 1024 * 1024;
const PAID_CALL_LIMIT: usize = 2;
const STABLE_PROVIDER_ERROR: &str = "Router live-provider test transport failed";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveProviderConfigFile {
    version: u32,
    profiles: Vec<LiveProviderProfile>,
}

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LiveProviderAudience {
    #[default]
    Official,
    NvidiaInference,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveProviderProfile {
    #[serde(default)]
    provider: LiveProviderAudience,
    family: LlmApiFamily,
    endpoint: String,
    transport_identity: String,
    api_key_env: String,
    candidate_model: String,
    candidate_model_revision: String,
    judge_model: String,
    judge_model_revision: String,
    #[serde(default)]
    judge_temperature: Option<f64>,
    #[serde(default)]
    anthropic_version: Option<String>,
}

struct LiveReplayTransport {
    capability: LlmReplayCapability,
    endpoint: Url,
    credential: Zeroizing<String>,
    bearer_auth: bool,
    anthropic_version: Option<String>,
    client: reqwest::Client,
    starts: AtomicUsize,
    started_models: Mutex<Vec<String>>,
}

impl LiveReplayTransport {
    fn from_profile(profile: &LiveProviderProfile) -> Self {
        let endpoint = validated_endpoint(profile);
        let credential = read_credential(profile);
        let contract = provider_contract(profile.provider, profile.family);
        let client = reqwest::Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .timeout(PROVIDER_TIMEOUT)
            .build()
            .expect("live-provider HTTP client configuration should be valid");
        Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: profile.family,
                transport_identity: profile.transport_identity.clone(),
            },
            endpoint,
            credential,
            bearer_auth: contract.bearer_auth,
            anthropic_version: profile.anthropic_version.clone(),
            client,
            starts: AtomicUsize::new(0),
            started_models: Mutex::new(Vec::new()),
        }
    }

    fn started_models(&self) -> Vec<String> {
        self.started_models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn credential_probe(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(self.credential.as_bytes().to_vec())
    }
}

impl LlmReplayTransport for LiveReplayTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
        if !request.headers.is_empty() {
            return Err(stable_provider_error());
        }
        let model = request
            .content
            .get("model")
            .and_then(Json::as_str)
            .filter(|model| !model.is_empty())
            .ok_or_else(stable_provider_error)?
            .to_string();
        self.starts
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |starts| {
                (starts < PAID_CALL_LIMIT).then_some(starts + 1)
            })
            .map_err(|_| stable_provider_error())?;
        self.started_models
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(model);

        let client = self.client.clone();
        let endpoint = self.endpoint.clone();
        let family = self.capability.api_family;
        let credential = self.credential.clone();
        let bearer_auth = self.bearer_auth;
        let anthropic_version = self.anthropic_version.clone();
        let task = tokio::spawn(async move {
            let mut builder = client
                .post(endpoint)
                .header(reqwest::header::ACCEPT, "application/json")
                .json(&request.content);
            builder = if bearer_auth {
                let authorization = Zeroizing::new(format!("Bearer {}", credential.as_str()));
                builder.header(reqwest::header::AUTHORIZATION, authorization.as_str())
            } else {
                builder.header("x-api-key", credential.as_str())
            };
            if family == LlmApiFamily::AnthropicMessages {
                builder = builder.header(
                    "anthropic-version",
                    anthropic_version
                        .as_deref()
                        .ok_or_else(stable_provider_error)?,
                );
            }
            let mut response = builder.send().await.map_err(|_| stable_provider_error())?;
            if !response.status().is_success()
                || response
                    .content_length()
                    .is_some_and(|length| length > PROVIDER_RESPONSE_LIMIT as u64)
            {
                return Err(stable_provider_error());
            }

            let mut body = Vec::new();
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|_| stable_provider_error())?
            {
                let next_len = body
                    .len()
                    .checked_add(chunk.len())
                    .ok_or_else(stable_provider_error)?;
                if next_len > PROVIDER_RESPONSE_LIMIT {
                    return Err(stable_provider_error());
                }
                body.extend_from_slice(&chunk);
            }
            serde_json::from_slice(&body).map_err(|_| stable_provider_error())
        });
        let abort = task.abort_handle();
        Ok(LlmReplayCall::new(
            async move {
                match task.await {
                    Ok(result) => result,
                    Err(_) => Err(stable_provider_error()),
                }
            },
            move || abort.abort(),
        ))
    }
}

fn stable_provider_error() -> FlowError {
    FlowError::Internal(STABLE_PROVIDER_ERROR.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires explicit opt-in and configured paid provider services"]
async fn live_three_family_provider_evidence_is_correlated_and_secret_free() {
    assert_eq!(
        std::env::var(RUN_ENV).ok().as_deref(),
        Some("1"),
        "set {RUN_ENV}=1 to acknowledge paid live-provider traffic"
    );
    let profiles = load_profiles();
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;

    for profile in profiles {
        reset_runtime();
        let transport = Arc::new(LiveReplayTransport::from_profile(&profile));
        let credential = transport.credential_probe();
        let mut harness =
            Harness::with_config(|database_path| live_router_config(database_path, &profile));
        let anchor_id = harness.deliver(transport.clone()).await;
        let (scheduler, admissions) = harness.scheduler();
        let exit = tokio::time::timeout(PROVIDER_TIMEOUT.saturating_mul(2), scheduler.run())
            .await
            .expect("live candidate and judge calls exceeded the test deadline");
        let judge_max = exit
            .summary()
            .pool_gauges
            .get(POOL)
            .expect("live pool gauges should exist")
            .judge_max_in_flight;
        assert_eq!(
            judge_max,
            1,
            "live candidate did not reach Judge for {}",
            family_name(profile.family)
        );
        assert_successful_exit(&exit, 1, 1);
        drop(
            admissions
                .try_acquire(POOL, 1)
                .expect("live terminal evidence should release scheduler admission"),
        );
        harness.drain_writer().await;

        assert_eq!(transport.starts.load(Ordering::Acquire), 2);
        assert_eq!(
            transport.started_models(),
            [profile.candidate_model.clone(), profile.judge_model.clone()]
        );
        audit_durable_evidence(&harness, &profile, anchor_id, &credential);
    }
}

fn load_profiles() -> Vec<LiveProviderProfile> {
    let path = std::env::var_os(CONFIG_ENV)
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| panic!("{CONFIG_ENV} must name an absolute JSON configuration path"));
    let raw = fs::read_to_string(path).expect("live-provider config should be readable UTF-8");
    let mut config: LiveProviderConfigFile =
        serde_json::from_str(&raw).expect("live-provider config should match the strict schema");
    assert_eq!(config.version, 1, "live-provider config version must be 1");
    assert_eq!(
        config.profiles.len(),
        3,
        "live-provider config must contain exactly three profiles"
    );

    let mut families = HashSet::new();
    for profile in &config.profiles {
        assert!(
            families.insert(profile.family),
            "live-provider config contains a duplicate API family"
        );
        validate_profile(profile);
    }
    assert_eq!(families.len(), 3);
    config
        .profiles
        .sort_by_key(|profile| family_order(profile.family));
    config.profiles
}

fn validate_profile(profile: &LiveProviderProfile) {
    for value in [
        profile.transport_identity.as_str(),
        profile.candidate_model.as_str(),
        profile.candidate_model_revision.as_str(),
        profile.judge_model.as_str(),
        profile.judge_model_revision.as_str(),
    ] {
        assert!(
            !value.is_empty()
                && !value.chars().any(char::is_control)
                && !contains_sensitive_control_material(value),
            "live-provider identity and model fields must be nonempty, control-free, and credential-free"
        );
    }
    assert!(
        !profile.transport_identity.contains("://")
            && !profile.transport_identity.contains('@')
            && !profile.transport_identity.contains('/'),
        "live-provider transport identity must not contain an endpoint or credential"
    );
    assert!(
        !profile.api_key_env.is_empty()
            && profile
                .api_key_env
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_uppercase() || byte.is_ascii_digit()),
        "live-provider api_key_env must be an uppercase environment-variable name"
    );
    let contract = provider_contract(profile.provider, profile.family);
    assert_eq!(
        profile.api_key_env, contract.api_key_env,
        "live-provider credential environment variable does not match its provider audience"
    );
    match profile.family {
        LlmApiFamily::AnthropicMessages => assert!(
            profile.anthropic_version.as_deref().is_some_and(
                |version| !version.is_empty() && !version.chars().any(char::is_control)
            ),
            "Anthropic live-provider profile requires anthropic_version"
        ),
        LlmApiFamily::OpenAIChatCompletions | LlmApiFamily::OpenAIResponses => assert!(
            profile.anthropic_version.is_none(),
            "OpenAI live-provider profiles must omit anthropic_version"
        ),
    }
    let _ = validated_endpoint(profile);
}

fn validated_endpoint(profile: &LiveProviderProfile) -> Url {
    let endpoint =
        Url::parse(&profile.endpoint).expect("live-provider endpoint must be an absolute URL");
    assert_eq!(
        endpoint.scheme(),
        "https",
        "live-provider endpoint must use HTTPS"
    );
    let contract = provider_contract(profile.provider, profile.family);
    assert_eq!(
        endpoint.host_str(),
        Some(contract.host),
        "live-provider endpoint must use the selected provider host"
    );
    assert_eq!(
        endpoint.port_or_known_default(),
        Some(443),
        "live-provider endpoint must use the official HTTPS port"
    );
    assert!(
        endpoint.username().is_empty()
            && endpoint.password().is_none()
            && endpoint.query().is_none()
            && endpoint.fragment().is_none(),
        "live-provider endpoint must not contain userinfo, a query, or a fragment"
    );
    assert_eq!(
        endpoint.path(),
        contract.path,
        "live-provider endpoint must use the exact provider API-family path"
    );
    endpoint
}

fn read_credential(profile: &LiveProviderProfile) -> Zeroizing<String> {
    let raw = Zeroizing::new(std::env::var(&profile.api_key_env).unwrap_or_else(|_| {
        panic!(
            "credential environment variable {} is not set",
            profile.api_key_env
        )
    }));
    let value = raw.trim();
    assert!(
        value.len() >= 16 && !value.chars().any(char::is_control),
        "live-provider credential must be at least 16 control-free bytes"
    );
    for configured_value in [
        profile.endpoint.as_str(),
        profile.transport_identity.as_str(),
        profile.candidate_model.as_str(),
        profile.candidate_model_revision.as_str(),
        profile.judge_model.as_str(),
        profile.judge_model_revision.as_str(),
        profile.anthropic_version.as_deref().unwrap_or_default(),
    ] {
        assert!(
            !configured_value
                .as_bytes()
                .windows(value.len())
                .any(|bytes| bytes == value.as_bytes()),
            "live-provider profile fields must not contain their credential"
        );
    }
    Zeroizing::new(value.to_string())
}

fn live_router_config(path: &Path, profile: &LiveProviderProfile) -> RouterConfig {
    let mut config = config(path, 1, 1, 1, 1);
    config.project_id = Some(format!("scheduler-live-{}", family_name(profile.family)));
    let pool = &mut config.pools[0];
    pool.api_family = profile.family;
    pool.anchor_models = vec![format!("live-anchor-{}", family_name(profile.family))];
    pool.anchor_revision = "live-anchor-fixture-v1".to_string();
    pool.candidates[0].model = profile.candidate_model.clone();
    pool.candidates[0].model_revision = profile.candidate_model_revision.clone();
    pool.judge.model = profile.judge_model.clone();
    pool.judge.model_revision = profile.judge_model_revision.clone();
    pool.judge.temperature = profile.judge_temperature;
    assert!(
        config
            .validate()
            .iter()
            .all(|diagnostic| diagnostic.level != nemo_relay::plugin::DiagnosticLevel::Error),
        "live-provider profile produced an invalid Router configuration"
    );
    config
}

fn audit_durable_evidence(
    harness: &Harness,
    profile: &LiveProviderProfile,
    anchor_id: uuid::Uuid,
    credential: &[u8],
) {
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 1);
    assert_eq!(count(&connection, "judge_attempts"), 1);
    let judge_states = strings(
        &connection,
        "SELECT state FROM judge_attempt_state_events ORDER BY event_seq",
    );
    assert_eq!(
        count(&connection, "evaluations"),
        1,
        "live Judge produced no evaluation; states: {judge_states:?}"
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM shadow_attempt_state_events ORDER BY event_seq"
        ),
        ["reserved", "started", "completed"]
    );
    assert_eq!(judge_states, ["started", "valid"]);
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        [
            "candidate:admitted",
            "candidate:success",
            "judge:admitted",
            "judge:success",
        ]
    );

    let evidence = connection
        .query_row(
            "SELECT sb.sample_batch_id, sa.shadow_attempt_id, ja.judge_attempt_id,
                    e.evaluation_id, sr.shadow_result_id, e.label, sr.vector_source_hash,
                    ar.normalized_response_json
             FROM anchors AS a
             JOIN anchor_results AS ar USING (anchor_id)
             JOIN sample_batches AS sb USING (anchor_id)
             JOIN shadow_attempts AS sa ON sa.sample_batch_id = sb.sample_batch_id
             JOIN shadow_results AS sr USING (shadow_attempt_id)
             JOIN evaluations AS e ON e.evaluation_id = sr.evaluation_id
             JOIN judge_attempts AS ja ON ja.shadow_attempt_id = sa.shadow_attempt_id
             WHERE a.anchor_id = ?1 AND ja.attempt_ordinal = 0
               AND a.api_family = ?2 AND a.transport_identity = ?3
               AND sa.candidate_model = ?4 AND e.judge_model = ?5
               AND sr.terminal_class = 'completed' AND e.source = 'judge'
               AND sr.canonicalizable = 1 AND e.is_partial = 0",
            rusqlite::params![
                anchor_id.to_string(),
                family_name(profile.family),
                profile.transport_identity,
                profile.candidate_model,
                profile.judge_model,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                ))
            },
        )
        .expect("live-provider evidence should form one complete correlation chain");
    for id in [
        &evidence.0,
        &evidence.1,
        &evidence.2,
        &evidence.3,
        &evidence.4,
    ] {
        uuid::Uuid::parse_str(id).expect("live-provider evidence IDs should be UUIDs");
    }
    assert!(matches!(evidence.5.as_str(), "pass" | "fail" | "ambiguous"));
    assert_eq!(evidence.6.len(), 64);
    assert!(evidence.6.bytes().all(|byte| byte.is_ascii_hexdigit()));

    let pool = &harness.config.pools[0];
    let expected_anchor = project_anchor_response(
        profile.family,
        &family_response(profile.family, &pool.anchor_models[0], "anchor answer"),
        64 * 1024,
    )
    .expect("live-provider anchor fixture should project");
    assert_eq!(
        serde_json::from_str::<Json>(&evidence.7).expect("stored anchor evidence should be JSON"),
        serde_json::to_value(expected_anchor).expect("anchor projection should serialize")
    );

    let mut foreign_key_statement = connection
        .prepare("PRAGMA foreign_key_check")
        .expect("foreign-key audit should prepare");
    let mut foreign_keys = foreign_key_statement
        .query([])
        .expect("foreign-key audit should execute");
    assert!(
        foreign_keys
            .next()
            .expect("foreign-key audit should read")
            .is_none()
    );
    drop(foreign_keys);
    drop(foreign_key_statement);
    assert_eq!(
        connection
            .query_row("PRAGMA integrity_check(1)", [], |row| row
                .get::<_, String>(0))
            .expect("integrity audit should execute"),
        "ok"
    );
    connection
        .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
        .expect("live-provider evidence should checkpoint before leakage audit");
    drop(connection);

    for (index, path) in ledger_files(&harness.database_path).into_iter().enumerate() {
        if let Some(bytes) = read_ledger_artifact(&path, index == 0) {
            assert!(
                !bytes
                    .windows(credential.len())
                    .any(|window| window == credential),
                "credential bytes must not appear in the Router ledger or its sidecars"
            );
        }
    }
}

fn read_ledger_artifact(path: &Path, required: bool) -> Option<Vec<u8>> {
    match fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if required {
                panic!("required Router ledger artifact must exist for credential auditing");
            }
            match fs::symlink_metadata(path) {
                Err(metadata_error) if metadata_error.kind() == std::io::ErrorKind::NotFound => {
                    None
                }
                _ => panic!("Router ledger artifact must be readable for credential auditing"),
            }
        }
        Err(_) => panic!("Router ledger artifact must be readable for credential auditing"),
    }
}

fn ledger_files(database_path: &Path) -> [PathBuf; 4] {
    let sidecar = |suffix: &str| {
        let mut path = OsString::from(database_path.as_os_str());
        path.push(suffix);
        PathBuf::from(path)
    };
    [
        database_path.to_path_buf(),
        sidecar("-wal"),
        sidecar("-shm"),
        sidecar("-journal"),
    ]
}

const fn family_order(family: LlmApiFamily) -> u8 {
    match family {
        LlmApiFamily::OpenAIChatCompletions => 0,
        LlmApiFamily::OpenAIResponses => 1,
        LlmApiFamily::AnthropicMessages => 2,
    }
}

const fn family_name(family: LlmApiFamily) -> &'static str {
    match family {
        LlmApiFamily::OpenAIChatCompletions => "openai_chat_completions",
        LlmApiFamily::OpenAIResponses => "openai_responses",
        LlmApiFamily::AnthropicMessages => "anthropic_messages",
    }
}

#[derive(Clone, Copy)]
struct LiveProviderContract {
    host: &'static str,
    path: &'static str,
    api_key_env: &'static str,
    bearer_auth: bool,
}

const fn provider_contract(
    provider: LiveProviderAudience,
    family: LlmApiFamily,
) -> LiveProviderContract {
    match (provider, family) {
        (LiveProviderAudience::Official, LlmApiFamily::OpenAIChatCompletions) => {
            LiveProviderContract {
                host: "api.openai.com",
                path: "/v1/chat/completions",
                api_key_env: "OPENAI_API_KEY",
                bearer_auth: true,
            }
        }
        (LiveProviderAudience::Official, LlmApiFamily::OpenAIResponses) => LiveProviderContract {
            host: "api.openai.com",
            path: "/v1/responses",
            api_key_env: "OPENAI_API_KEY",
            bearer_auth: true,
        },
        (LiveProviderAudience::Official, LlmApiFamily::AnthropicMessages) => LiveProviderContract {
            host: "api.anthropic.com",
            path: "/v1/messages",
            api_key_env: "ANTHROPIC_API_KEY",
            bearer_auth: false,
        },
        (LiveProviderAudience::NvidiaInference, family) => LiveProviderContract {
            host: "inference-api.nvidia.com",
            path: match family {
                LlmApiFamily::OpenAIChatCompletions => "/v1/chat/completions",
                LlmApiFamily::OpenAIResponses => "/v1/responses",
                LlmApiFamily::AnthropicMessages => "/v1/messages",
            },
            api_key_env: "INFERENCE_API_KEY",
            bearer_auth: true,
        },
    }
}

fn test_profile(family: LlmApiFamily) -> LiveProviderProfile {
    let provider = LiveProviderAudience::Official;
    let contract = provider_contract(provider, family);
    LiveProviderProfile {
        provider,
        family,
        endpoint: format!("https://{}{}", contract.host, contract.path),
        transport_identity: format!("test-{}-v1", family_name(family)),
        api_key_env: contract.api_key_env.to_string(),
        candidate_model: "candidate-model".to_string(),
        candidate_model_revision: "candidate-revision".to_string(),
        judge_model: "judge-model".to_string(),
        judge_model_revision: "judge-revision".to_string(),
        judge_temperature: None,
        anthropic_version: (family == LlmApiFamily::AnthropicMessages)
            .then(|| "2023-06-01".to_string()),
    }
}

fn nvidia_test_profile(family: LlmApiFamily) -> LiveProviderProfile {
    let mut profile = test_profile(family);
    profile.provider = LiveProviderAudience::NvidiaInference;
    let contract = provider_contract(profile.provider, family);
    profile.endpoint = format!("https://{}{}", contract.host, contract.path);
    profile.api_key_env = contract.api_key_env.to_string();
    profile.judge_temperature = Some(1.0);
    profile
}

#[test]
fn live_profile_pins_supported_credential_audiences() {
    for family in [
        LlmApiFamily::OpenAIChatCompletions,
        LlmApiFamily::OpenAIResponses,
        LlmApiFamily::AnthropicMessages,
    ] {
        validate_profile(&test_profile(family));
        validate_profile(&nvidia_test_profile(family));
    }

    let mut wrong_host = test_profile(LlmApiFamily::OpenAIResponses);
    wrong_host.endpoint = "https://example.com/v1/responses".to_string();
    assert!(
        std::panic::catch_unwind(|| validate_profile(&wrong_host)).is_err(),
        "an arbitrary HTTPS credential destination must be rejected"
    );

    let mut wrong_port = test_profile(LlmApiFamily::OpenAIResponses);
    wrong_port.endpoint = "https://api.openai.com:8443/v1/responses".to_string();
    assert!(
        std::panic::catch_unwind(|| validate_profile(&wrong_port)).is_err(),
        "a nonstandard credential-destination port must be rejected"
    );

    let mut wrong_path = test_profile(LlmApiFamily::OpenAIResponses);
    wrong_path.endpoint = "https://api.openai.com/v1/chat/completions".to_string();
    assert!(
        std::panic::catch_unwind(|| validate_profile(&wrong_path)).is_err(),
        "a path for another API family must be rejected"
    );

    let mut trailing_slash = test_profile(LlmApiFamily::OpenAIResponses);
    trailing_slash.endpoint = "https://api.openai.com/v1/responses/".to_string();
    assert!(
        std::panic::catch_unwind(|| validate_profile(&trailing_slash)).is_err(),
        "a trailing slash must not weaken exact endpoint binding"
    );

    let mut wrong_credential = test_profile(LlmApiFamily::AnthropicMessages);
    wrong_credential.api_key_env = "UNRELATED_API_KEY".to_string();
    assert!(
        std::panic::catch_unwind(|| validate_profile(&wrong_credential)).is_err(),
        "a credential name for another audience must be rejected"
    );
}

#[test]
fn live_transport_refuses_a_third_start_before_network_work() {
    let profile = test_profile(LlmApiFamily::OpenAIChatCompletions);
    let transport = LiveReplayTransport {
        capability: LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: profile.family,
            transport_identity: profile.transport_identity.clone(),
        },
        endpoint: validated_endpoint(&profile),
        credential: Zeroizing::new("test-credential-never-sent".to_string()),
        bearer_auth: true,
        anthropic_version: None,
        client: reqwest::Client::builder()
            .https_only(true)
            .redirect(Policy::none())
            .build()
            .expect("test client should build"),
        starts: AtomicUsize::new(PAID_CALL_LIMIT),
        started_models: Mutex::new(Vec::new()),
    };
    let result = transport.start(LlmRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!({"model": "must-not-start"}),
    });
    assert!(result.is_err());
    assert_eq!(transport.starts.load(Ordering::Acquire), PAID_CALL_LIMIT);
    assert!(transport.started_models().is_empty());
}

#[test]
fn credential_audit_skips_only_absent_artifacts() {
    let temporary = tempfile::tempdir().expect("temporary directory should open");
    let missing = temporary.path().join("missing");
    assert!(read_ledger_artifact(&missing, false).is_none());
    assert!(
        std::panic::catch_unwind(|| read_ledger_artifact(&missing, true)).is_err(),
        "the required main database must not be treated as an absent sidecar"
    );
    assert!(
        std::panic::catch_unwind(|| read_ledger_artifact(temporary.path(), false)).is_err(),
        "an existing unreadable artifact shape must fail the audit"
    );

    #[cfg(unix)]
    {
        let dangling = temporary.path().join("dangling");
        std::os::unix::fs::symlink(temporary.path().join("absent-target"), &dangling)
            .expect("dangling test symlink should be created");
        assert!(
            std::panic::catch_unwind(|| read_ledger_artifact(&dangling, false)).is_err(),
            "a dangling sidecar symlink must not be treated as an absent artifact"
        );
    }
}
