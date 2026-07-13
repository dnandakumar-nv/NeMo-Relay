// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use serde_json::json;

use super::*;

#[tokio::test]
async fn completed_replay_call_does_not_cancel() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(async { Ok(json!({"ok": true})) }, move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });

    assert_eq!(call.await.unwrap(), json!({"ok": true}));
    assert_eq!(cancellations.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn completion_disarms_every_cancellation_handle_clone() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(async { Ok(json!({"ok": true})) }, move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });
    let first = call.cancellation_handle();
    let second = first.clone();

    assert_eq!(call.await.unwrap(), json!({"ok": true}));
    assert!(!first.cancel());
    assert!(!second.cancel());
    assert_eq!(cancellations.load(Ordering::SeqCst), 0);
}

#[test]
fn dropping_incomplete_replay_call_cancels_exactly_once() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(std::future::pending(), move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });

    drop(call);

    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
}

#[test]
fn external_cancellation_and_drop_share_one_hook() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(std::future::pending(), move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });
    let cancellation = call.cancellation_handle();

    assert!(cancellation.cancel());
    assert!(!cancellation.cancel());
    drop(call);
    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
}

#[test]
fn concurrent_handle_clones_claim_the_hook_exactly_once() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(std::future::pending(), move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });
    let cancellation = call.cancellation_handle();
    let claimed = Arc::new(AtomicUsize::new(0));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let cancellation = cancellation.clone();
        let claimed = claimed.clone();
        workers.push(thread::spawn(move || {
            if cancellation.cancel() {
                claimed.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    drop(call);

    assert_eq!(claimed.load(Ordering::SeqCst), 1);
    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
}

#[test]
fn external_cancellation_races_drop_without_double_invocation() {
    let cancellations = Arc::new(AtomicUsize::new(0));
    let on_cancel = cancellations.clone();
    let call = LlmReplayCall::new(std::future::pending(), move || {
        on_cancel.fetch_add(1, Ordering::SeqCst);
    });
    let cancellation = call.cancellation_handle();
    let barrier = Arc::new(Barrier::new(2));
    let drop_barrier = barrier.clone();
    let dropper = thread::spawn(move || {
        drop_barrier.wait();
        drop(call);
    });

    barrier.wait();
    cancellation.cancel();
    dropper.join().unwrap();

    assert_eq!(cancellations.load(Ordering::SeqCst), 1);
}

#[test]
fn cancellation_panic_is_contained() {
    let call = LlmReplayCall::new(std::future::pending(), || panic!("cancel panic"));
    let cancellation = call.cancellation_handle();
    assert!(cancellation.cancel());
    assert!(!cancellation.cancel());
    drop(call);
}

#[tokio::test]
async fn sibling_calls_have_independent_completion_and_cancellation() {
    let first_cancellations = Arc::new(AtomicUsize::new(0));
    let second_cancellations = Arc::new(AtomicUsize::new(0));
    let first_on_cancel = first_cancellations.clone();
    let second_on_cancel = second_cancellations.clone();

    let first = LlmReplayCall::new(std::future::pending(), move || {
        first_on_cancel.fetch_add(1, Ordering::SeqCst);
    });
    let second = LlmReplayCall::new(async { Ok(json!(2)) }, move || {
        second_on_cancel.fetch_add(1, Ordering::SeqCst);
    });

    drop(first);
    assert_eq!(second.await.unwrap(), json!(2));
    assert_eq!(first_cancellations.load(Ordering::SeqCst), 1);
    assert_eq!(second_cancellations.load(Ordering::SeqCst), 0);
}

#[test]
fn capability_is_non_serializable_contract_data() {
    trait AmbiguousIfSerialize<Marker> {
        fn probe() {}
    }

    impl<T: ?Sized> AmbiguousIfSerialize<()> for T {}

    struct ImplementsSerialize;

    impl<T: ?Sized + serde::Serialize> AmbiguousIfSerialize<ImplementsSerialize> for T {}

    // This inference is unambiguous only while LlmReplayCapability does not
    // implement Serialize. Adding that implementation makes this test fail to
    // compile because both marker implementations apply.
    let _ = <LlmReplayCapability as AmbiguousIfSerialize<_>>::probe;

    let capability = LlmReplayCapability {
        contract_version: LLM_REPLAY_CONTRACT_VERSION,
        api_family: LlmApiFamily::OpenAIResponses,
        transport_identity: "gateway-A".to_string(),
    };

    // Exhaustive destructuring keeps endpoint/authentication policy and anchor
    // response side channels out of the public capability contract.
    let LlmReplayCapability {
        contract_version,
        api_family,
        transport_identity,
    } = capability;
    assert_eq!(contract_version, 1);
    assert_eq!(api_family, LlmApiFamily::OpenAIResponses);
    assert_eq!(transport_identity, "gateway-A");
}
