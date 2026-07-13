// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization compatibility tests for shared NeMo Relay DTOs.

use std::collections::BTreeMap;
use std::sync::Arc;

use nemo_relay_types::Json;
use nemo_relay_types::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, PendingMarkSpec, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings,
};
use nemo_relay_types::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
    LlmRequestInterceptOutcome, LlmTrajectoryScopeSnapshot,
};
use nemo_relay_types::api::scope::ScopeType;
use nemo_relay_types::api::tool::ToolExecutionInterceptOutcome;
use nemo_relay_types::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, Message, MessageContent, StructuredResponseFormat,
    StructuredResponseFormatKind, ToolDefinition,
};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use serde_json::{Map, json};
use uuid::Uuid;

#[test]
fn llm_execution_context_snapshot_round_trips_with_stable_wire_values() {
    let root_uuid = Uuid::now_v7();
    let child_uuid = Uuid::now_v7();
    let snapshot = LlmExecutionContextSnapshot {
        call_uuid: Uuid::now_v7(),
        root_uuid,
        parent_uuid: child_uuid,
        trajectory_owner_uuid: root_uuid,
        trajectory_owner_path: vec![
            LlmTrajectoryScopeSnapshot {
                uuid: root_uuid,
                name: "root".to_string(),
                scope_type: ScopeType::Agent,
            },
            LlmTrajectoryScopeSnapshot {
                uuid: child_uuid,
                name: "turn".to_string(),
                scope_type: ScopeType::Function,
            },
        ],
        api_family: LlmApiFamily::OpenAIChatCompletions,
        call_role: LlmCallRole::Shadow,
        attributes: LlmAttributes::STATEFUL,
        tenant_id: Some("tenant-a".to_string()),
        agent_id: Some("agent-a".to_string()),
        sanitized_metadata: BTreeMap::from([("region".to_string(), json!("us"))]),
    };

    let encoded = serde_json::to_value(&snapshot).expect("snapshot should serialize");
    assert_eq!(encoded["api_family"], "openai_chat_completions");
    assert_eq!(encoded["call_role"], "shadow");
    let encoded_keys = encoded
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        encoded_keys,
        std::collections::BTreeSet::from([
            "agent_id",
            "api_family",
            "attributes",
            "call_role",
            "call_uuid",
            "parent_uuid",
            "root_uuid",
            "sanitized_metadata",
            "tenant_id",
            "trajectory_owner_path",
            "trajectory_owner_uuid",
        ])
    );

    let decoded: LlmExecutionContextSnapshot =
        serde_json::from_value(encoded).expect("snapshot should deserialize");
    assert_eq!(decoded, snapshot);
}

#[test]
fn llm_api_family_and_call_role_use_approved_wire_values() {
    let families = [
        (
            LlmApiFamily::OpenAIChatCompletions,
            "openai_chat_completions",
        ),
        (LlmApiFamily::OpenAIResponses, "openai_responses"),
        (LlmApiFamily::AnthropicMessages, "anthropic_messages"),
    ];
    for (family, expected) in families {
        assert_eq!(serde_json::to_value(family).unwrap(), json!(expected));
        assert_eq!(
            serde_json::from_value::<LlmApiFamily>(json!(expected)).unwrap(),
            family
        );
    }
    assert!(serde_json::from_value::<LlmApiFamily>(json!("unknown")).is_err());

    let roles = [
        (LlmCallRole::Primary, "primary"),
        (LlmCallRole::Shadow, "shadow"),
        (LlmCallRole::Judge, "judge"),
    ];
    for (role, expected) in roles {
        assert_eq!(serde_json::to_value(role).unwrap(), json!(expected));
        assert_eq!(
            serde_json::from_value::<LlmCallRole>(json!(expected)).unwrap(),
            role
        );
    }
    assert!(serde_json::from_value::<LlmCallRole>(json!("internal")).is_err());
}

#[test]
fn event_llm_call_role_handles_legacy_valid_and_malformed_profiles() {
    let make_llm_event = |profile| {
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name("llm").build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::llm(),
            profile,
        ))
    };

    assert_eq!(
        make_llm_event(None).llm_call_role(),
        Some(LlmCallRole::Primary)
    );
    assert_eq!(
        make_llm_event(Some(CategoryProfile::default())).llm_call_role(),
        Some(LlmCallRole::Primary)
    );

    for role in [
        LlmCallRole::Primary,
        LlmCallRole::Shadow,
        LlmCallRole::Judge,
    ] {
        let mut profile = CategoryProfile::default();
        profile
            .extra
            .insert("future_field".to_string(), json!(true));
        profile.set_llm_call_role(role);
        assert_eq!(profile.llm_call_role(), Some(role));
        assert_eq!(profile.extra.get("future_field"), Some(&json!(true)));
        assert_eq!(make_llm_event(Some(profile)).llm_call_role(), Some(role));
    }

    let malformed = CategoryProfile {
        extra: BTreeMap::from([("call_role".to_string(), json!("PRIMARY"))]),
        ..CategoryProfile::default()
    };
    assert_eq!(malformed.llm_call_role(), None);
    assert_eq!(make_llm_event(Some(malformed)).llm_call_role(), None);

    let tool_event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("tool").build(),
        ScopeCategory::Start,
        Vec::new(),
        EventCategory::tool(),
        None,
    ));
    assert_eq!(tool_event.llm_call_role(), None);
}

#[test]
fn event_round_trips_with_annotated_llm_profiles() {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: Map::new(),
    };
    let response = AnnotatedLlmResponse {
        id: Some("resp_1".into()),
        model: Some("model".into()),
        message: Some(MessageContent::Text("world".into())),
        tool_calls: None,
        finish_reason: None,
        usage: None,
        optimization_summary: None,
        api_specific: None,
        extra: Map::new(),
    };
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("llm")
            .data(json!(LlmRequest {
                headers: Map::new(),
                content: json!({ "prompt": "hello" }),
            }))
            .build(),
        ScopeCategory::Start,
        llm_attributes_to_strings(LlmAttributes::STATEFUL),
        EventCategory::llm(),
        Some(CategoryProfile {
            annotated_request: Some(Arc::new(request)),
            annotated_response: Some(Arc::new(response)),
            ..CategoryProfile::default()
        }),
    ));

    let encoded = serde_json::to_value(&event).expect("event should serialize");
    let decoded: Event = serde_json::from_value(encoded).expect("event should deserialize");
    assert_eq!(decoded.name(), "llm");
    assert_eq!(
        decoded
            .annotated_response()
            .and_then(|response| response.id.as_deref()),
        Some("resp_1")
    );
}

#[test]
fn annotated_request_preserves_legacy_wire_shape_without_response_format() {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: Map::new(),
    };

    assert_eq!(
        serde_json::to_string(&request).unwrap(),
        r#"{"messages":[{"role":"user","content":"hello"}],"model":"model"}"#
    );
}

#[test]
fn developer_and_structured_response_formats_round_trip() {
    let developer = Message::Developer {
        content: MessageContent::Text("Return machine-readable output.".into()),
        name: Some("policy".into()),
    };
    assert_eq!(
        serde_json::to_value(&developer).unwrap(),
        json!({
            "role": "developer",
            "content": "Return machine-readable output.",
            "name": "policy",
        })
    );
    assert_eq!(
        serde_json::from_value::<Message>(serde_json::to_value(&developer).unwrap()).unwrap(),
        developer
    );

    for response_format in [
        StructuredResponseFormat {
            kind: StructuredResponseFormatKind::JsonObject,
            name: None,
            schema: None,
            strict: None,
            extra: Map::new(),
        },
        StructuredResponseFormat {
            kind: StructuredResponseFormatKind::JsonSchema,
            name: Some("answer".into()),
            schema: Some(json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
            })),
            strict: Some(true),
            extra: Map::from_iter([(
                "native_format".into(),
                json!({"description": "Answer schema"}),
            )]),
        },
    ] {
        let encoded = serde_json::to_value(&response_format).unwrap();
        assert_eq!(
            serde_json::from_value::<StructuredResponseFormat>(encoded).unwrap(),
            response_format
        );
    }
}

#[test]
fn tool_definition_preserves_optional_strictness() {
    for strict in [Some(true), Some(false), None] {
        let tool = ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: "lookup".into(),
                description: Some("Look up data".into()),
                parameters: Some(json!({"type": "object"})),
                strict,
            },
        };

        let encoded = serde_json::to_value(&tool).unwrap();
        assert_eq!(
            encoded["function"].get("strict").and_then(Json::as_bool),
            strict
        );
        assert_eq!(
            serde_json::from_value::<ToolDefinition>(encoded).unwrap(),
            tool
        );
    }

    assert!(
        serde_json::from_value::<ToolDefinition>(json!({
            "type": "function",
            "function": {"name": "lookup", "strict": null}
        }))
        .is_err()
    );
}

#[cfg(feature = "schema")]
#[test]
fn function_definition_schema_advertises_boolean_or_omitted_strictness() {
    let schema = serde_json::to_value(schemars::schema_for!(FunctionDefinition)).unwrap();
    assert_eq!(schema["properties"]["strict"]["type"], json!("boolean"));
    assert!(
        !schema["required"]
            .as_array()
            .is_some_and(|required| required.iter().any(|field| field == "strict"))
    );
}

#[test]
fn annotated_request_distinguishes_typed_and_generic_response_formats() {
    let generic = json!({
        "messages": [],
        "response_format": {"type": "xml", "dialect": "provider-native"},
    });
    let decoded_generic: AnnotatedLlmRequest =
        serde_json::from_value(generic.clone()).expect("generic format should remain lossless");
    assert!(decoded_generic.response_format.is_none());
    assert_eq!(
        decoded_generic.extra.get("response_format"),
        generic.get("response_format")
    );
    assert_eq!(serde_json::to_value(decoded_generic).unwrap(), generic);

    let generic_kind = json!({
        "messages": [],
        "response_format": {"kind": "provider_native", "dialect": "vendor"},
    });
    let decoded_generic_kind: AnnotatedLlmRequest =
        serde_json::from_value(generic_kind.clone()).unwrap();
    assert!(decoded_generic_kind.response_format.is_none());
    assert_eq!(
        decoded_generic_kind.extra.get("response_format"),
        generic_kind.get("response_format")
    );
    assert_eq!(
        serde_json::to_value(decoded_generic_kind).unwrap(),
        generic_kind
    );

    let generic_null = json!({"messages": [], "response_format": null});
    let decoded_null: AnnotatedLlmRequest = serde_json::from_value(generic_null.clone()).unwrap();
    assert!(decoded_null.response_format.is_none());
    assert_eq!(decoded_null.extra.get("response_format"), Some(&Json::Null));
    assert_eq!(serde_json::to_value(decoded_null).unwrap(), generic_null);

    let typed = json!({
        "messages": [],
        "response_format": {
            "kind": "json_schema",
            "name": "answer",
            "schema": {"type": "object"},
            "strict": true,
        },
    });
    let decoded_typed: AnnotatedLlmRequest = serde_json::from_value(typed.clone()).unwrap();
    assert_eq!(
        decoded_typed
            .response_format
            .as_ref()
            .map(|format| format.kind),
        Some(StructuredResponseFormatKind::JsonSchema)
    );
    assert!(!decoded_typed.extra.contains_key("response_format"));
    assert_eq!(serde_json::to_value(&decoded_typed).unwrap(), typed);

    let mut conflicting = decoded_typed;
    conflicting
        .extra
        .insert("response_format".into(), json!({"type": "json_object"}));
    let error = serde_json::to_value(conflicting).unwrap_err().to_string();
    assert!(error.contains("typed and generic response_format representations conflict"));

    let duplicate = r#"{"messages":[],"response_format":{"kind":"json_object"},"response_format":{"type":"xml"}}"#;
    assert!(serde_json::from_str::<AnnotatedLlmRequest>(duplicate).is_err());
}

#[cfg(feature = "schema")]
#[test]
fn annotated_request_schema_includes_developer_and_structured_format_kinds() {
    let schema = serde_json::to_string(&schemars::schema_for!(AnnotatedLlmRequest)).unwrap();
    assert!(schema.contains("developer"));
    assert!(schema.contains("response_format"));
    assert!(schema.contains("json_object"));
    assert!(schema.contains("json_schema"));
}

#[test]
fn llm_request_intercept_outcome_round_trips_pending_marks() {
    let outcome = LlmRequestInterceptOutcome::new(
        LlmRequest {
            headers: Map::new(),
            content: json!({ "prompt": "hello" }),
        },
        None,
    )
    .with_pending_mark(
        PendingMarkSpec::builder()
            .name("request.optimized")
            .category(EventCategory::custom())
            .category_profile(
                CategoryProfile::builder()
                    .subtype("optimizer.saved_tokens")
                    .build(),
            )
            .data(json!({ "saved_tokens": 12 }))
            .metadata(json!({ "source": "test" }))
            .build(),
    );

    let encoded = serde_json::to_value(&outcome).expect("outcome should serialize");
    assert_eq!(encoded["pending_marks"][0]["name"], "request.optimized");
    assert_eq!(encoded["pending_marks"][0]["category"], "custom");
    assert!(encoded["annotated_request"].is_null());

    let mut encoded_without_pending_marks = encoded.clone();
    encoded_without_pending_marks
        .as_object_mut()
        .unwrap()
        .remove("pending_marks");
    let decoded_without_pending_marks: LlmRequestInterceptOutcome =
        serde_json::from_value(encoded_without_pending_marks)
            .expect("outcome without pending marks should deserialize");
    assert!(decoded_without_pending_marks.pending_marks.is_empty());

    let decoded_defaults: LlmRequestInterceptOutcome = serde_json::from_value(json!({
        "request": {"headers": {}, "content": {"prompt": "hello"}},
        "future_field": true
    }))
    .expect("omitted optional fields and unknown fields should be accepted");
    assert!(decoded_defaults.annotated_request.is_none());
    assert!(decoded_defaults.pending_marks.is_empty());

    assert!(
        serde_json::from_value::<LlmRequestInterceptOutcome>(json!({
            "annotated_request": null,
            "pending_marks": []
        }))
        .is_err(),
        "request is required"
    );

    let decoded: LlmRequestInterceptOutcome =
        serde_json::from_value(encoded).expect("outcome should deserialize");
    assert_eq!(decoded, outcome);
}

#[test]
fn llm_request_intercept_outcome_converts_from_request_inputs() {
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({ "prompt": "hello" }),
    };
    let annotated_request: AnnotatedLlmRequest = serde_json::from_value(json!({
        "messages": [],
        "model": "model"
    }))
    .expect("annotated request should deserialize");

    let request_only: LlmRequestInterceptOutcome = request.clone().into();
    assert_eq!(
        request_only,
        LlmRequestInterceptOutcome::new(request.clone(), None)
    );

    let required_annotation: LlmRequestInterceptOutcome =
        (request.clone(), annotated_request.clone()).into();
    assert_eq!(
        required_annotation,
        LlmRequestInterceptOutcome::new(request.clone(), Some(annotated_request.clone()))
    );

    let optional_annotation: LlmRequestInterceptOutcome =
        (request.clone(), Some(annotated_request.clone())).into();
    assert_eq!(
        optional_annotation,
        LlmRequestInterceptOutcome::new(request, Some(annotated_request))
    );
}

#[test]
fn tool_execution_intercept_outcome_round_trips_pending_marks() {
    let outcome = ToolExecutionInterceptOutcome::new(json!({"stdout": "compacted"}))
        .with_pending_mark(
            PendingMarkSpec::builder()
                .name("tool.output.compacted")
                .category(EventCategory::custom())
                .category_profile(
                    CategoryProfile::builder()
                        .subtype("optimizer.saved_tokens")
                        .build(),
                )
                .data(json!({"saved_tokens": 12}))
                .metadata(json!({"source": "test"}))
                .build(),
        );

    let encoded = serde_json::to_value(&outcome).expect("outcome should serialize");
    assert_eq!(encoded["result"]["stdout"], "compacted");
    assert_eq!(encoded["pending_marks"][0]["name"], "tool.output.compacted");
    assert_eq!(encoded["pending_marks"][0]["category"], "custom");

    let decoded: ToolExecutionInterceptOutcome =
        serde_json::from_value(encoded).expect("outcome should deserialize");
    assert_eq!(decoded, outcome);

    let defaults: ToolExecutionInterceptOutcome = serde_json::from_value(json!({
        "result": "plain",
        "future_field": true
    }))
    .expect("omitted pending marks and unknown fields should be accepted");
    assert!(defaults.pending_marks.is_empty());
    assert_eq!(defaults.result, json!("plain"));

    assert!(
        serde_json::from_value::<ToolExecutionInterceptOutcome>(json!({
            "pending_marks": []
        }))
        .is_err(),
        "result is required"
    );
}

#[test]
fn tool_execution_intercept_outcome_converts_from_json() {
    let result = json!({"value": 42});
    let outcome: ToolExecutionInterceptOutcome = result.clone().into();
    assert_eq!(outcome, ToolExecutionInterceptOutcome::new(result));
}
