// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for subscriber in the NeMo Relay adaptive crate.

use super::*;
use nemo_relay::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
};
use nemo_relay::api::llm::LlmCallRole;
use nemo_relay::api::scope::ScopeType;
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason};
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone, Copy)]
enum EventType {
    Start,
    End,
    Mark,
}

/// Helper to construct a minimal test [`Event`] with only the fields
/// relevant to subscriber/mapping logic populated.
fn make_test_event(
    event_type: EventType,
    scope_type: Option<ScopeType>,
    name: Option<&str>,
) -> Event {
    let event_name = name.unwrap_or("");
    match (event_type, scope_type) {
        (EventType::Start, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name(event_name).build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (EventType::End, Some(scope_type)) => Event::Scope(ScopeEvent::new(
            BaseEvent::builder().name(event_name).build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::from(scope_type),
            None,
        )),
        (EventType::Mark, _) | (_, None) => Event::Mark(MarkEvent::new(
            BaseEvent::builder().name(event_name).build(),
            None,
            None,
        )),
    }
}

fn with_llm_call_role_value(mut event: Event, value: serde_json::Value) -> Event {
    let Event::Scope(scope) = &mut event else {
        panic!("LLM role test fixture must be a scope event");
    };
    scope
        .category_profile
        .get_or_insert_with(CategoryProfile::default)
        .extra
        .insert("call_role".to_string(), value);
    event
}

fn make_evaluator_llm_event(event_type: EventType, role: LlmCallRole, name: &str) -> Event {
    let mut event = with_llm_call_role_value(
        make_test_event(event_type, Some(ScopeType::Llm), Some(name)),
        json!(role),
    );
    let Event::Scope(scope) = &mut event else {
        unreachable!("evaluator LLM fixture must be a scope event");
    };
    scope.base.metadata = Some(json!({
        "anchor_uuid": Uuid::now_v7().to_string(),
        "anchor_id": Uuid::now_v7().to_string(),
        "pool_id": "primary",
        "candidate_id": name,
        "config_generation_id": "primary-generation",
        "learning_generation_id": Uuid::now_v7().to_string(),
        "call_role": "primary",
        "name": "primary",
    }));
    scope
        .category_profile
        .get_or_insert_with(CategoryProfile::default)
        .model_name = Some("primary-model".to_string());
    event
}

// -----------------------------------------------------------------------
// create_subscriber tests
// -----------------------------------------------------------------------

#[test]
fn test_create_subscriber_sends_event() {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let subscriber = create_subscriber(tx);

    let event = make_test_event(EventType::Start, Some(ScopeType::Llm), Some("gpt-4"));
    subscriber(&event);

    let received = rx.try_recv().expect("should receive event");
    assert_eq!(received.uuid(), event.uuid());
    assert_eq!(received.name(), "gpt-4");
}

#[test]
fn test_subscriber_survives_dropped_receiver() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let subscriber = create_subscriber(tx);

    // Drop the receiver — subscriber must not panic
    drop(rx);

    let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
    subscriber(&event); // Must not panic
}

// -----------------------------------------------------------------------
// event_to_call_record tests
// -----------------------------------------------------------------------

#[test]
fn test_event_to_call_record_llm_start() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Llm), Some("gpt-4"));
    let record = event_to_call_record(&event).expect("should produce CallRecord for LLM start");

    assert_eq!(record.kind, CallKind::Llm);
    assert_eq!(record.name, "gpt-4");
    assert!(record.ended_at.is_none());
    assert!(record.metadata_snapshot.is_none());
}

#[test]
fn test_event_to_call_record_accepts_only_primary_llm_starts() {
    let explicit_primary = with_llm_call_role_value(
        make_test_event(EventType::Start, Some(ScopeType::Llm), Some("primary")),
        json!(LlmCallRole::Primary),
    );
    assert!(event_to_call_record(&explicit_primary).is_some());

    for role in [LlmCallRole::Shadow, LlmCallRole::Judge] {
        let event = with_llm_call_role_value(
            make_test_event(EventType::Start, Some(ScopeType::Llm), Some("internal")),
            json!(role),
        );
        assert!(event_to_call_record(&event).is_none());
    }

    let malformed = with_llm_call_role_value(
        make_test_event(EventType::Start, Some(ScopeType::Llm), Some("malformed")),
        json!("PRIMARY"),
    );
    assert!(event_to_call_record(&malformed).is_none());
}

#[test]
fn test_evaluator_llm_events_ignore_primary_looking_names_and_metadata() {
    for (role, name) in [
        (LlmCallRole::Shadow, "nemo_relay.router.shadow"),
        (LlmCallRole::Judge, "nemo_relay.router.judge"),
    ] {
        for event_type in [EventType::Start, EventType::End] {
            let event = make_evaluator_llm_event(event_type, role, name);
            assert!(is_non_primary_llm_event(&event));
            assert!(event_to_call_record(&event).is_none());
        }

        let primary = with_llm_call_role_value(
            make_test_event(EventType::Start, Some(ScopeType::Llm), Some(name)),
            json!(LlmCallRole::Primary),
        );
        assert!(event_to_call_record(&primary).is_some());
    }
}

#[test]
fn test_event_to_call_record_tool_start() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
    let record = event_to_call_record(&event).expect("should produce CallRecord for Tool start");

    assert_eq!(record.kind, CallKind::Tool);
    assert_eq!(record.name, "search");
    assert!(record.ended_at.is_none());
}

#[test]
fn test_event_to_call_record_end_event_returns_none() {
    let event = make_test_event(EventType::End, Some(ScopeType::Llm), Some("gpt-4"));
    assert!(
        event_to_call_record(&event).is_none(),
        "End events should not produce CallRecords"
    );
}

#[test]
fn test_event_to_call_record_llm_end_with_annotated_response_stays_observability_only() {
    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("gpt-4")
            .data(serde_json::json!({"response": "ok"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::llm(),
        Some(
            CategoryProfile::builder()
                .model_name("gpt-4")
                .annotated_response(Arc::new(AnnotatedLlmResponse {
                    id: Some("resp-1".to_string()),
                    model: Some("gpt-4".to_string()),
                    message: None,
                    tool_calls: None,
                    finish_reason: Some(FinishReason::Complete),
                    usage: None,
                    api_specific: None,
                    optimization_summary: None,
                    extra: serde_json::Map::new(),
                }))
                .build(),
        ),
    ));

    assert!(
        event_to_call_record(&event).is_none(),
        "annotated_response belongs to LLM end observability, not request/start call records",
    );
}

#[test]
fn test_event_to_call_record_agent_scope_returns_none() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Agent), Some("my-agent"));
    assert!(
        event_to_call_record(&event).is_none(),
        "Agent scope events are run boundaries, not call records"
    );
}

#[test]
fn test_event_to_call_record_no_name_defaults_to_empty() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Tool), None);
    let record = event_to_call_record(&event).expect("should produce CallRecord");
    assert_eq!(record.name, "");
}

// -----------------------------------------------------------------------
// is_run_boundary tests
// -----------------------------------------------------------------------

#[test]
fn test_is_run_boundary_agent_start() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Agent), Some("agent-1"));
    assert!(
        is_run_boundary(&event),
        "Agent Start should be a run boundary"
    );
}

#[test]
fn test_is_run_boundary_agent_end() {
    let event = make_test_event(EventType::End, Some(ScopeType::Agent), Some("agent-1"));
    assert!(
        is_run_boundary(&event),
        "Agent End should be a run boundary"
    );
}

#[test]
fn test_is_run_boundary_tool_start() {
    let event = make_test_event(EventType::Start, Some(ScopeType::Tool), Some("search"));
    assert!(
        !is_run_boundary(&event),
        "Tool Start should NOT be a run boundary"
    );
}

#[test]
fn test_is_run_boundary_agent_mark() {
    let event = make_test_event(EventType::Mark, Some(ScopeType::Agent), Some("agent-1"));
    assert!(
        !is_run_boundary(&event),
        "Agent Mark should NOT be a run boundary"
    );
}
