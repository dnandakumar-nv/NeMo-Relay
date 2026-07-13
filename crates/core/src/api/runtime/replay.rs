// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned replay capabilities for delayed non-streaming LLM calls.

use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};

use crate::api::llm::{LlmApiFamily, LlmExecutionContextSnapshot, LlmRequest};
use crate::error::Result;
use crate::json::Json;

/// Current replay transport contract version.
pub const LLM_REPLAY_CONTRACT_VERSION: u32 = 1;

/// Non-secret capabilities advertised by one replay transport.
///
/// Contract version 1 represents delayed, repeatable, non-streaming replay.
/// Endpoint, authentication, TLS, proxy, and transport policy remain immutable
/// host-owned implementation state. The capability exposes no anchor response
/// status, response headers, streaming writer, or other client side channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmReplayCapability {
    /// Replay contract implemented by the transport.
    pub contract_version: u32,
    /// Provider request/response family accepted by the transport.
    pub api_family: LlmApiFamily,
    /// Stable non-secret partition identity, never a URL credential or token.
    pub transport_identity: String,
}

/// Build one call-scoped replay transport from a frozen LLM context.
pub trait LlmReplayFactory: Send + Sync {
    /// Build a transport whose immutable host policy can outlive the anchor call.
    ///
    /// # Errors
    /// Returns an error when the host cannot construct the replay transport.
    fn build(&self, context: &LlmExecutionContextSnapshot) -> Result<Arc<dyn LlmReplayTransport>>;
}

/// Repeatable host-owned transport for delayed non-streaming LLM requests.
///
/// Implementations must ignore endpoint/authentication mutation in replay
/// request data and create independent cancellation state for every start.
pub trait LlmReplayTransport: Send + Sync {
    /// Return the immutable, non-secret replay capability.
    fn capability(&self) -> &LlmReplayCapability;

    /// Start one independent replay invocation.
    ///
    /// Implementations must return promptly without waiting for host I/O, an
    /// interpreter lock, or an event loop. Host work that can block must be
    /// dispatched asynchronously and represented by the returned call.
    ///
    /// # Errors
    /// Returns an error when the invocation cannot be started.
    fn start(&self, request: LlmRequest) -> Result<LlmReplayCall>;
}

type ReplayCancellationHook = Box<dyn FnOnce() + Send + 'static>;

struct ReplayCancellationState {
    hook: Mutex<Option<ReplayCancellationHook>>,
}

/// Cloneable exact-once cancellation authority for one replay invocation.
///
/// Every clone refers to the same one-shot host cancellation hook. The first
/// call to [`cancel`](Self::cancel), or dropping the associated incomplete
/// [`LlmReplayCall`], invokes that hook. Completing the call disarms all clones.
#[derive(Clone)]
pub struct LlmReplayCancellationHandle {
    state: Arc<ReplayCancellationState>,
}

impl LlmReplayCancellationHandle {
    fn new(cancel: ReplayCancellationHook) -> Self {
        Self {
            state: Arc::new(ReplayCancellationState {
                hook: Mutex::new(Some(cancel)),
            }),
        }
    }

    /// Invoke this replay invocation's host cancellation hook at most once.
    ///
    /// Returns `true` only for the caller that claimed the still-armed hook.
    /// Cancellation panics are contained and still count as having claimed it.
    pub fn cancel(&self) -> bool {
        let hook = lock_unpoisoned(&self.state.hook).take();
        let Some(hook) = hook else {
            return false;
        };
        let _ = catch_unwind(AssertUnwindSafe(hook));
        true
    }

    fn disarm(&self) {
        lock_unpoisoned(&self.state.hook).take();
    }
}

/// One cancellation-safe replay invocation.
///
/// Every transport start returns independent state. Dropping a call before it
/// resolves invokes the host cancellation callback exactly once. The callback
/// is discarded before a completed result is returned.
#[must_use = "replay calls must be awaited or explicitly dropped to cancel them"]
pub struct LlmReplayCall {
    future: Pin<Box<dyn Future<Output = Result<Json>> + Send + 'static>>,
    cancellation: LlmReplayCancellationHandle,
}

impl LlmReplayCall {
    /// Create a replay call from an owned future and one-shot cancellation hook.
    ///
    /// Hosts use this constructor when implementing [`LlmReplayTransport`].
    /// The cancellation hook must affect only this invocation and must tolerate
    /// the underlying host operation already having stopped. It must return
    /// promptly after signaling or dispatching cancellation and must not wait
    /// for host I/O, an interpreter lock, or task completion.
    pub fn new<F, C>(future: F, cancel: C) -> Self
    where
        F: Future<Output = Result<Json>> + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        Self {
            future: Box::pin(future),
            cancellation: LlmReplayCancellationHandle::new(Box::new(cancel)),
        }
    }

    /// Return cloneable exact-once cancellation authority for this invocation.
    pub fn cancellation_handle(&self) -> LlmReplayCancellationHandle {
        self.cancellation.clone()
    }
}

impl Future for LlmReplayCall {
    type Output = Result<Json>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.future.as_mut().poll(cx) {
            Poll::Ready(result) => {
                self.cancellation.disarm();
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for LlmReplayCall {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
#[path = "../../../tests/unit/replay_tests.rs"]
mod tests;
