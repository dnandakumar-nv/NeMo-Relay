// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, non-secret Router runtime health accounting.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Constant-space health state updated only by the coordinator task.
#[derive(Debug, Default)]
pub(crate) struct RouterHealth {
    state: Mutex<RouterHealthSnapshot>,
    rejected_records: AtomicU64,
}

/// Bounded health snapshot containing only stable reason codes and counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RouterHealthSnapshot {
    pub(crate) last_reason: Option<&'static str>,
    pub(crate) accepted_records: u64,
    pub(crate) rejected_records: u64,
}

impl RouterHealth {
    pub(crate) fn accept(&self, reason: &'static str) {
        if let Ok(mut state) = self.state.lock() {
            state.last_reason = Some(reason);
            state.accepted_records = state.accepted_records.saturating_add(1);
        } else {
            self.reject();
        }
    }

    pub(crate) fn reject(&self) {
        self.rejected_records.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> RouterHealthSnapshot {
        let mut snapshot = self.state.lock().map(|state| *state).unwrap_or_default();
        snapshot.rejected_records = self.rejected_records.load(Ordering::Relaxed);
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::RouterHealth;

    #[test]
    fn snapshot_contains_only_a_stable_reason_and_bounded_counters() {
        let health = RouterHealth::default();
        health.accept("router.runtime.failure");
        health.reject();
        let snapshot = health.snapshot();
        assert_eq!(snapshot.last_reason, Some("router.runtime.failure"));
        assert_eq!(snapshot.accepted_records, 1);
        assert_eq!(snapshot.rejected_records, 1);
    }
}
