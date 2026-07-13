// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen context capture for additive V2 managed LLM calls.

use std::collections::BTreeMap;
use std::sync::Arc;

use unicode_normalization::UnicodeNormalization;

use crate::api::llm::{
    LlmApiFamily, LlmCallRole, LlmExecutionContextSnapshot, LlmHandle, LlmTrajectoryScopeSnapshot,
};
use crate::api::runtime::{ScopeStack, current_scope_stack};
use crate::api::scope::ScopeType;
use crate::error::{FlowError, Result};
use crate::json::Json;

const MAX_ROUTING_IDENTITY_BYTES: usize = 256;

pub(crate) fn capture_llm_execution_context(
    handle: &LlmHandle,
    api_family: LlmApiFamily,
    call_role: LlmCallRole,
    tenant_id: Option<String>,
    agent_id: Option<String>,
    sanitized_metadata: BTreeMap<String, Json>,
) -> Result<Arc<LlmExecutionContextSnapshot>> {
    validate_routing_identity("tenant_id", tenant_id.as_deref())?;
    validate_routing_identity("agent_id", agent_id.as_deref())?;

    let scope_stack = current_scope_stack();
    let scope_guard = scope_stack
        .read()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let snapshot = capture_from_scope_stack(
        &scope_guard,
        handle,
        api_family,
        call_role,
        tenant_id,
        agent_id,
        sanitized_metadata,
    )?;
    validate_internal_call_context(&snapshot)?;
    Ok(Arc::new(snapshot))
}

fn capture_from_scope_stack(
    scope_stack: &ScopeStack,
    handle: &LlmHandle,
    api_family: LlmApiFamily,
    call_role: LlmCallRole,
    tenant_id: Option<String>,
    agent_id: Option<String>,
    sanitized_metadata: BTreeMap<String, Json>,
) -> Result<LlmExecutionContextSnapshot> {
    let parent_uuid = handle.parent_uuid.ok_or_else(|| {
        FlowError::InvalidArgument("V2 LLM call requires an active parent scope".to_string())
    })?;
    let root_uuid = scope_stack.root_uuid();
    let mut ancestry = Vec::new();
    let mut current_uuid = parent_uuid;
    loop {
        if ancestry
            .iter()
            .any(|scope: &&crate::api::scope::ScopeHandle| scope.uuid == current_uuid)
        {
            return Err(FlowError::InvalidArgument(
                "V2 LLM call parent ancestry contains a cycle".to_string(),
            ));
        }
        let scope = scope_stack.find(&current_uuid).ok_or_else(|| {
            FlowError::InvalidArgument(
                "V2 LLM call parent ancestry is not fully active on the current scope stack"
                    .to_string(),
            )
        })?;
        ancestry.push(scope);
        if current_uuid == root_uuid {
            break;
        }
        current_uuid = scope.parent_uuid.ok_or_else(|| {
            FlowError::InvalidArgument(
                "V2 LLM call parent ancestry is detached from the current root".to_string(),
            )
        })?;
    }
    ancestry.reverse();

    let owner_index = ancestry
        .iter()
        .enumerate()
        .skip(1)
        .rev()
        .find_map(|(index, scope)| (scope.scope_type == ScopeType::Agent).then_some(index))
        .unwrap_or(0);
    let trajectory_owner_path = ancestry[owner_index..]
        .iter()
        .map(|scope| LlmTrajectoryScopeSnapshot {
            uuid: scope.uuid,
            name: scope.name.clone(),
            scope_type: scope.scope_type,
        })
        .collect();

    Ok(LlmExecutionContextSnapshot {
        call_uuid: handle.uuid,
        root_uuid,
        parent_uuid,
        trajectory_owner_uuid: ancestry[owner_index].uuid,
        trajectory_owner_path,
        api_family,
        call_role,
        attributes: handle.attributes,
        tenant_id,
        agent_id,
        sanitized_metadata,
    })
}

pub(crate) fn validate_routing_identity(field: &str, value: Option<&str>) -> Result<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_empty() {
        return Err(invalid_identity(field, "must not be empty"));
    }
    if value.len() > MAX_ROUTING_IDENTITY_BYTES {
        return Err(invalid_identity(field, "must not exceed 256 UTF-8 bytes"));
    }
    if value.chars().any(char::is_control) {
        return Err(invalid_identity(
            field,
            "must not contain control characters",
        ));
    }
    if !value.nfc().eq(value.chars()) {
        return Err(invalid_identity(
            field,
            "must use Unicode NFC normalization",
        ));
    }
    if looks_like_credential(value) {
        return Err(invalid_identity(field, "must not contain a credential"));
    }
    Ok(())
}

fn validate_internal_call_context(snapshot: &LlmExecutionContextSnapshot) -> Result<()> {
    if snapshot.call_role == LlmCallRole::Primary {
        return Ok(());
    }
    let parent = snapshot
        .trajectory_owner_path
        .last()
        .filter(|scope| scope.uuid == snapshot.parent_uuid)
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Shadow and Judge calls require a captured immediate parent".to_string(),
            )
        })?;
    if parent.scope_type != ScopeType::Evaluator {
        return Err(FlowError::InvalidArgument(
            "Shadow and Judge calls require an Evaluator parent scope".to_string(),
        ));
    }
    let anchor_uuid = snapshot
        .sanitized_metadata
        .get("anchor_uuid")
        .and_then(Json::as_str)
        .ok_or_else(|| {
            FlowError::InvalidArgument(
                "Shadow and Judge calls require string anchor_uuid metadata".to_string(),
            )
        })?;
    uuid::Uuid::parse_str(anchor_uuid).map_err(|_| {
        FlowError::InvalidArgument(
            "Shadow and Judge calls require valid UUID anchor_uuid metadata".to_string(),
        )
    })?;
    Ok(())
}

fn invalid_identity(field: &str, reason: &str) -> FlowError {
    FlowError::InvalidArgument(format!("{field} {reason}"))
}

fn looks_like_credential(value: &str) -> bool {
    let value = value.trim();
    let lowercase = value.to_ascii_lowercase();
    const CREDENTIAL_MARKERS: [&str; 10] = [
        "authorization:",
        "bearer ",
        "basic ",
        "api_key=",
        "api-key=",
        "apikey=",
        "password=",
        "passwd=",
        "secret=",
        "token=",
    ];
    CREDENTIAL_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
        || has_known_secret_prefix(&lowercase)
        || looks_like_jwt(value)
        || looks_like_aws_access_key_id(value)
        || (lowercase.starts_with("-----begin ") && lowercase.contains("private key-----"))
        || credential_bearing_url(&lowercase)
}

fn has_known_secret_prefix(value: &str) -> bool {
    const PREFIXES: [&str; 13] = [
        "nvapi-",
        "sk-",
        "ghp_",
        "gho_",
        "ghu_",
        "ghs_",
        "ghr_",
        "github_pat_",
        "hf_",
        "xoxb-",
        "xoxp-",
        "xoxa-",
        "aiza",
    ];
    value.len() >= 20 && PREFIXES.iter().any(|prefix| value.starts_with(prefix))
}

fn looks_like_jwt(value: &str) -> bool {
    let mut segments = value.split('.');
    let Some(header) = segments.next() else {
        return false;
    };
    let Some(payload) = segments.next() else {
        return false;
    };
    let Some(signature) = segments.next() else {
        return false;
    };
    segments.next().is_none()
        && header.starts_with("eyJ")
        && [header, payload, signature]
            .iter()
            .all(|segment| !segment.is_empty() && segment.bytes().all(is_base64url_byte))
}

fn is_base64url_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

fn looks_like_aws_access_key_id(value: &str) -> bool {
    const PREFIXES: [&str; 8] = [
        "AKIA", "ASIA", "AIDA", "AROA", "AIPA", "ANPA", "ANVA", "ASCA",
    ];
    value.len() == 20
        && PREFIXES.iter().any(|prefix| value.starts_with(prefix))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
}

fn credential_bearing_url(value: &str) -> bool {
    let Some(authority) = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
        .and_then(|remainder| remainder.split('/').next())
    else {
        return false;
    };
    authority.contains('@')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::llm::LlmAttributes;
    use crate::api::runtime::with_scope_stack;
    use crate::api::scope::ScopeHandle;

    fn scope(name: &str, scope_type: ScopeType, parent_uuid: Option<uuid::Uuid>) -> ScopeHandle {
        ScopeHandle::builder()
            .name(name)
            .scope_type(scope_type)
            .parent_uuid_opt(parent_uuid)
            .build()
    }

    fn llm_handle(parent_uuid: uuid::Uuid) -> LlmHandle {
        LlmHandle::builder()
            .name("provider")
            .parent_uuid(parent_uuid)
            .attributes(LlmAttributes::STATEFUL)
            .build()
    }

    fn capture(
        stack: &ScopeStack,
        handle: &LlmHandle,
        tenant_id: Option<String>,
        agent_id: Option<String>,
    ) -> Result<LlmExecutionContextSnapshot> {
        validate_routing_identity("tenant_id", tenant_id.as_deref())?;
        validate_routing_identity("agent_id", agent_id.as_deref())?;
        capture_from_scope_stack(
            stack,
            handle,
            LlmApiFamily::OpenAIResponses,
            LlmCallRole::Primary,
            tenant_id,
            agent_id,
            BTreeMap::from([("region".to_string(), Json::String("us".to_string()))]),
        )
    }

    #[test]
    fn captures_root_fallback_and_owner_to_parent_path() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let turn = scope("turn", ScopeType::Function, Some(root_uuid));
        stack.push(turn.clone());
        let handle = llm_handle(turn.uuid);

        let snapshot = capture(&stack, &handle, Some("Tenant-A".into()), None).unwrap();

        assert_eq!(snapshot.call_uuid, handle.uuid);
        assert_eq!(snapshot.root_uuid, root_uuid);
        assert_eq!(snapshot.parent_uuid, turn.uuid);
        assert_eq!(snapshot.trajectory_owner_uuid, root_uuid);
        assert_eq!(
            snapshot
                .trajectory_owner_path
                .iter()
                .map(|scope| scope.name.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "turn"]
        );
        assert_eq!(snapshot.tenant_id.as_deref(), Some("Tenant-A"));
        assert_eq!(snapshot.attributes, LlmAttributes::STATEFUL);
    }

    #[test]
    fn deepest_explicit_agent_owns_nested_call() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let outer_agent = scope("outer", ScopeType::Agent, Some(root_uuid));
        stack.push(outer_agent.clone());
        let outer_step = scope("outer-step", ScopeType::Function, Some(outer_agent.uuid));
        stack.push(outer_step.clone());
        let subagent = scope("subagent", ScopeType::Agent, Some(outer_step.uuid));
        stack.push(subagent.clone());
        let inner_step = scope("inner-step", ScopeType::Function, Some(subagent.uuid));
        stack.push(inner_step.clone());

        let snapshot = capture(&stack, &llm_handle(inner_step.uuid), None, None).unwrap();

        assert_eq!(snapshot.trajectory_owner_uuid, subagent.uuid);
        assert_eq!(
            snapshot
                .trajectory_owner_path
                .iter()
                .map(|scope| scope.name.as_str())
                .collect::<Vec<_>>(),
            vec!["subagent", "inner-step"]
        );
    }

    #[test]
    fn explicit_active_ancestor_parent_limits_the_frozen_path() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let agent = scope("agent", ScopeType::Agent, Some(root_uuid));
        stack.push(agent.clone());
        let parent = scope("parent", ScopeType::Function, Some(agent.uuid));
        stack.push(parent.clone());
        let unrelated_descendant = scope("descendant", ScopeType::Function, Some(parent.uuid));
        stack.push(unrelated_descendant);

        let snapshot = capture(&stack, &llm_handle(parent.uuid), None, None).unwrap();

        assert_eq!(snapshot.parent_uuid, parent.uuid);
        assert_eq!(snapshot.trajectory_owner_uuid, agent.uuid);
        assert_eq!(snapshot.trajectory_owner_path.len(), 2);
        assert_eq!(snapshot.trajectory_owner_path[1].uuid, parent.uuid);
    }

    #[test]
    fn interleaved_active_branch_does_not_change_true_parent_ownership() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let unrelated_agent = scope("unrelated-agent", ScopeType::Agent, Some(root_uuid));
        stack.push(unrelated_agent.clone());
        let unrelated_child = scope(
            "unrelated-child",
            ScopeType::Function,
            Some(unrelated_agent.uuid),
        );
        stack.push(unrelated_child);
        let root_child = scope("root-child", ScopeType::Function, Some(root_uuid));
        stack.push(root_child.clone());

        let snapshot = capture(&stack, &llm_handle(root_child.uuid), None, None).unwrap();

        assert_eq!(snapshot.trajectory_owner_uuid, root_uuid);
        assert_eq!(
            snapshot
                .trajectory_owner_path
                .iter()
                .map(|scope| scope.name.as_str())
                .collect::<Vec<_>>(),
            vec!["root", "root-child"]
        );
    }

    #[test]
    fn frozen_path_survives_scope_removal() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let agent = scope("agent", ScopeType::Agent, Some(root_uuid));
        stack.push(agent.clone());
        let child = scope("child", ScopeType::Function, Some(agent.uuid));
        stack.push(child.clone());
        let snapshot = capture(&stack, &llm_handle(child.uuid), None, None).unwrap();
        let frozen_path = snapshot.trajectory_owner_path.clone();

        stack.remove(&child.uuid).unwrap();
        stack.remove(&agent.uuid).unwrap();

        assert_eq!(snapshot.trajectory_owner_path, frozen_path);
        assert_eq!(snapshot.trajectory_owner_path[0].name, "agent");
        assert_eq!(snapshot.trajectory_owner_path[1].name, "child");
    }

    #[test]
    fn detached_or_missing_parent_is_rejected() {
        let mut stack = ScopeStack::new();
        let detached = scope("detached", ScopeType::Function, None);
        stack.push(detached.clone());
        let error = capture(&stack, &llm_handle(detached.uuid), None, None).unwrap_err();
        assert!(matches!(error, FlowError::InvalidArgument(_)));

        let missing_parent_handle = LlmHandle::builder().name("provider").build();
        let error = capture(&stack, &missing_parent_handle, None, None).unwrap_err();
        assert!(matches!(error, FlowError::InvalidArgument(_)));
    }

    #[test]
    fn public_capture_entry_point_reads_the_visible_stack() {
        let mut stack = ScopeStack::new();
        let root_uuid = stack.root_uuid();
        let agent = scope("agent", ScopeType::Agent, Some(root_uuid));
        stack.push(agent.clone());
        let handle = llm_handle(agent.uuid);
        let stack = Arc::new(std::sync::RwLock::new(stack));

        let snapshot = with_scope_stack(stack, || {
            capture_llm_execution_context(
                &handle,
                LlmApiFamily::OpenAIResponses,
                LlmCallRole::Primary,
                Some("Tenant-A".to_string()),
                Some("Agent-A".to_string()),
                BTreeMap::new(),
            )
        })
        .unwrap();

        assert_eq!(snapshot.call_uuid, handle.uuid);
        assert_eq!(snapshot.trajectory_owner_uuid, agent.uuid);
        assert_eq!(snapshot.call_role, LlmCallRole::Primary);
        assert_eq!(snapshot.tenant_id.as_deref(), Some("Tenant-A"));
        assert_eq!(snapshot.agent_id.as_deref(), Some("Agent-A"));
    }

    #[test]
    fn identical_calls_keep_distinct_call_identity() {
        let stack = ScopeStack::new();
        let parent_uuid = stack.root_uuid();
        let first_task = std::thread::spawn(move || llm_handle(parent_uuid));
        let second_task = std::thread::spawn(move || llm_handle(parent_uuid));
        let first_handle = first_task.join().unwrap();
        let second_handle = second_task.join().unwrap();
        let first = capture(&stack, &first_handle, None, None).unwrap();
        let second = capture(&stack, &second_handle, None, None).unwrap();

        assert_ne!(first.call_uuid, second.call_uuid);
        assert_eq!(first.root_uuid, second.root_uuid);
        assert_eq!(first.trajectory_owner_path, second.trajectory_owner_path);
    }

    #[test]
    fn explicit_role_is_frozen_for_the_v2_event_path() {
        let stack = ScopeStack::new();
        let handle = LlmHandle::builder()
            .name("provider")
            .parent_uuid(stack.root_uuid())
            .build();

        let snapshot = capture_from_scope_stack(
            &stack,
            &handle,
            LlmApiFamily::OpenAIResponses,
            LlmCallRole::Shadow,
            None,
            None,
            BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(snapshot.call_role, LlmCallRole::Shadow);
    }

    #[test]
    fn routing_identities_are_strictly_validated() {
        for value in [
            "",
            "tenant\nother",
            "e\u{301}",
            "Bearer secret",
            "api_key=secret",
            "https://user:password@example.test/tenant",
            "nvapi-abcdefghijklmnopqrstuvwxyz0123456789",
            "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AKIAIOSFODNN7EXAMPLE",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZW5hbnQifQ.signature",
            "-----BEGIN PRIVATE KEY-----secret",
        ] {
            let error = validate_routing_identity("tenant_id", Some(value)).unwrap_err();
            assert!(matches!(error, FlowError::InvalidArgument(_)), "{value:?}");
        }

        let too_long = "x".repeat(MAX_ROUTING_IDENTITY_BYTES + 1);
        assert!(validate_routing_identity("agent_id", Some(&too_long)).is_err());
        assert!(validate_routing_identity("tenant_id", Some("tenant-A")).is_ok());
        assert!(validate_routing_identity("tenant_id", Some("sk-region")).is_ok());
        assert!(validate_routing_identity("agent_id", Some("caf\u{e9}")).is_ok());
        assert!(validate_routing_identity("tenant_id", None).is_ok());
    }
}
