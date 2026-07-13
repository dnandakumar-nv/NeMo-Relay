// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activation-fenced admission for Router-owned provider work.

#![allow(dead_code)] // The Task 10 supervisor integration consumes this primitive.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::Notify;

const PHASE_PENDING: u8 = 0;
const PHASE_OPEN: u8 = 1;
const PHASE_CLOSED: u8 = 2;

/// Monotonic paid-work admission phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderAdmissionPhase {
    Pending,
    Open,
    Closed,
}

/// Stable reason one provider start was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderStartRefusal {
    Pending,
    Closed,
}

/// Stable failure to issue the sole activation token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderActivationTokenError {
    AlreadyIssued,
    Closed,
}

struct ProviderAdmissionInner {
    phase: AtomicU8,
    token_issued: AtomicBool,
    transition: Mutex<()>,
    changed: Notify,
}

/// Cloneable authority for starting or permanently closing paid provider work.
///
/// The synchronous start closure runs while holding the same transition lock as
/// [`ProviderAdmissionGate::close`]. It must only create or spawn runtime-owned
/// work and return immediately; it must not await or call this gate recursively.
#[derive(Clone)]
pub(crate) struct ProviderAdmissionGate {
    inner: Arc<ProviderAdmissionInner>,
}

impl ProviderAdmissionGate {
    /// Create one pending gate without issuing activation authority.
    pub(crate) fn new_pending() -> Self {
        Self::with_phase(PHASE_PENDING)
    }

    #[cfg(test)]
    pub(crate) fn initially_open_for_test() -> Self {
        Self::with_phase(PHASE_OPEN)
    }

    fn with_phase(phase: u8) -> Self {
        Self {
            inner: Arc::new(ProviderAdmissionInner {
                phase: AtomicU8::new(phase),
                token_issued: AtomicBool::new(false),
                transition: Mutex::new(()),
                changed: Notify::new(),
            }),
        }
    }

    /// Issue the sole activation token at lifecycle-registration time.
    ///
    /// Concurrent and repeated calls deterministically return an error. Closing
    /// an unregistered gate also permanently prevents later token issuance.
    pub(crate) fn activation_token(
        &self,
    ) -> Result<ProviderActivationToken, ProviderActivationTokenError> {
        let _transition = lock_unpoisoned(&self.inner.transition);
        if self.inner.token_issued.load(Ordering::Acquire) {
            return Err(ProviderActivationTokenError::AlreadyIssued);
        }
        match self.phase() {
            ProviderAdmissionPhase::Closed => Err(ProviderActivationTokenError::Closed),
            ProviderAdmissionPhase::Open => Err(ProviderActivationTokenError::AlreadyIssued),
            ProviderAdmissionPhase::Pending => {
                self.inner.token_issued.store(true, Ordering::Release);
                Ok(ProviderActivationToken { gate: self.clone() })
            }
        }
    }

    /// Return the current monotonic phase.
    pub(crate) fn phase(&self) -> ProviderAdmissionPhase {
        decode_phase(self.inner.phase.load(Ordering::Acquire))
    }

    /// Synchronously linearize creation of one runtime-owned provider operation.
    ///
    /// Once this method invokes `start`, a concurrent close waits for the
    /// closure to return. If close linearizes first, `start` is never invoked.
    pub(crate) fn start_owned<T>(
        &self,
        start: impl FnOnce() -> T,
    ) -> Result<T, ProviderStartRefusal> {
        let _transition = lock_unpoisoned(&self.inner.transition);
        match self.phase() {
            ProviderAdmissionPhase::Pending => Err(ProviderStartRefusal::Pending),
            ProviderAdmissionPhase::Closed => Err(ProviderStartRefusal::Closed),
            ProviderAdmissionPhase::Open => Ok(start()),
        }
    }

    /// Permanently close admission, serialized with provider start creation.
    ///
    /// Returns `true` only for the caller that performs the transition.
    pub(crate) fn close(&self) -> bool {
        let changed = {
            let _transition = lock_unpoisoned(&self.inner.transition);
            if self.phase() == ProviderAdmissionPhase::Closed {
                false
            } else {
                self.inner.phase.store(PHASE_CLOSED, Ordering::Release);
                true
            }
        };
        if changed {
            self.inner.changed.notify_waiters();
        }
        changed
    }

    /// Wait until the gate no longer has `observed` as its phase.
    ///
    /// Registering the notification before re-reading the phase prevents a
    /// commit or close between those operations from being missed.
    pub(crate) async fn wait_for_change(
        &self,
        observed: ProviderAdmissionPhase,
    ) -> ProviderAdmissionPhase {
        loop {
            let changed = self.inner.changed.notified();
            let current = self.phase();
            if current != observed {
                return current;
            }
            changed.await;
        }
    }

    fn commit_pending(&self) -> bool {
        let changed = {
            let _transition = lock_unpoisoned(&self.inner.transition);
            self.inner
                .phase
                .compare_exchange(
                    PHASE_PENDING,
                    PHASE_OPEN,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
        };
        if changed {
            self.inner.changed.notify_waiters();
        }
        changed
    }
}

impl fmt::Debug for ProviderAdmissionGate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderAdmissionGate")
            .field("phase", &self.phase())
            .finish()
    }
}

/// Sole non-cloneable commit authority for one pending gate.
///
/// Core owns this token through the activation-rollback closure. Dropping that
/// closure on successful commit opens the gate. Invoked rollback closes the
/// gate first, and the subsequent token drop cannot reopen the absorbing
/// `Closed` phase.
#[must_use = "dropping the activation token commits provider admission"]
pub(crate) struct ProviderActivationToken {
    gate: ProviderAdmissionGate,
}

impl ProviderActivationToken {
    /// Close a failed activation before the token is dropped.
    pub(crate) fn rollback(self) -> bool {
        self.gate.close()
    }
}

impl Drop for ProviderActivationToken {
    fn drop(&mut self) {
        self.gate.commit_pending();
    }
}

impl fmt::Debug for ProviderActivationToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderActivationToken")
            .field("phase", &self.gate.phase())
            .finish_non_exhaustive()
    }
}

fn decode_phase(phase: u8) -> ProviderAdmissionPhase {
    match phase {
        PHASE_PENDING => ProviderAdmissionPhase::Pending,
        PHASE_OPEN => ProviderAdmissionPhase::Open,
        PHASE_CLOSED => ProviderAdmissionPhase::Closed,
        _ => unreachable!("provider admission phase is private and canonical"),
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn pending_refuses_start_without_invoking_the_factory() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();
        let invoked = AtomicBool::new(false);

        assert_eq!(
            gate.start_owned(|| invoked.store(true, Ordering::Release)),
            Err(ProviderStartRefusal::Pending)
        );
        assert!(!invoked.load(Ordering::Acquire));
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Pending);

        token.rollback();
    }

    #[tokio::test]
    async fn token_drop_opens_pending_and_wakes_waiters() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();
        let waiter_gate = gate.clone();
        let waiter = tokio::spawn(async move {
            waiter_gate
                .wait_for_change(ProviderAdmissionPhase::Pending)
                .await
        });
        tokio::task::yield_now().await;

        drop(token);

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), waiter)
                .await
                .unwrap()
                .unwrap(),
            ProviderAdmissionPhase::Open
        );
        assert_eq!(gate.start_owned(|| 17), Ok(17));
    }

    #[tokio::test]
    async fn token_rollback_closes_pending_and_drop_cannot_reopen() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();
        let waiter_gate = gate.clone();
        let waiter = tokio::spawn(async move {
            waiter_gate
                .wait_for_change(ProviderAdmissionPhase::Pending)
                .await
        });

        assert!(token.rollback());

        assert_eq!(waiter.await.unwrap(), ProviderAdmissionPhase::Closed);
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
        assert_eq!(
            gate.start_owned(|| unreachable!()),
            Err(ProviderStartRefusal::Closed)
        );
    }

    #[test]
    fn independent_close_before_token_drop_is_absorbing() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();

        assert!(gate.close());
        assert!(!gate.close());
        drop(token);

        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
        assert_eq!(
            gate.start_owned(|| unreachable!()),
            Err(ProviderStartRefusal::Closed)
        );
    }

    #[tokio::test]
    async fn open_close_wakes_waiters_and_is_absorbing() {
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let waiter_gate = gate.clone();
        let waiter = tokio::spawn(async move {
            waiter_gate
                .wait_for_change(ProviderAdmissionPhase::Open)
                .await
        });

        assert_eq!(gate.start_owned(|| "started"), Ok("started"));
        assert!(gate.close());

        assert_eq!(waiter.await.unwrap(), ProviderAdmissionPhase::Closed);
        assert!(!gate.close());
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
    }

    #[test]
    fn close_waits_for_a_start_that_linearized_first() {
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let start_gate = gate.clone();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let start_entered = entered.clone();
        let start_release = release.clone();
        let start = thread::spawn(move || {
            start_gate.start_owned(|| {
                start_entered.wait();
                start_release.wait();
                "owned-provider-work"
            })
        });

        entered.wait();
        let close_gate = gate.clone();
        let (attempting_tx, attempting_rx) = mpsc::sync_channel(1);
        let (closed_tx, closed_rx) = mpsc::sync_channel(1);
        let close = thread::spawn(move || {
            attempting_tx.send(()).unwrap();
            let result = close_gate.close();
            closed_tx.send(result).unwrap();
        });
        attempting_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(closed_rx.recv_timeout(Duration::from_millis(50)).is_err());

        release.wait();
        assert_eq!(start.join().unwrap(), Ok("owned-provider-work"));
        assert!(closed_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        close.join().unwrap();
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
    }

    #[test]
    fn close_and_start_races_have_one_valid_linearization() {
        for _ in 0..256 {
            let gate = ProviderAdmissionGate::initially_open_for_test();
            let ready = Arc::new(Barrier::new(3));
            let starts = Arc::new(AtomicUsize::new(0));

            let start_gate = gate.clone();
            let start_ready = ready.clone();
            let started = starts.clone();
            let start = thread::spawn(move || {
                start_ready.wait();
                start_gate.start_owned(|| started.fetch_add(1, Ordering::AcqRel))
            });
            let close_gate = gate.clone();
            let close_ready = ready.clone();
            let close = thread::spawn(move || {
                close_ready.wait();
                close_gate.close()
            });

            ready.wait();
            let start_result = start.join().unwrap();
            assert!(close.join().unwrap());
            match start_result {
                Ok(previous) => assert_eq!(previous, 0),
                Err(ProviderStartRefusal::Closed) => {}
                Err(ProviderStartRefusal::Pending) => {
                    panic!("an initially-open gate returned Pending")
                }
            }
            assert_eq!(
                starts.load(Ordering::Acquire),
                usize::from(start_result.is_ok())
            );
            assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
        }
    }

    #[test]
    fn concurrent_closers_publish_one_transition() {
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let ready = Arc::new(Barrier::new(9));
        let mut closers = Vec::new();
        for _ in 0..8 {
            let close_gate = gate.clone();
            let close_ready = ready.clone();
            closers.push(thread::spawn(move || {
                close_ready.wait();
                close_gate.close()
            }));
        }
        ready.wait();

        assert_eq!(
            closers
                .into_iter()
                .map(|closer| closer.join().unwrap())
                .filter(|closed| *closed)
                .count(),
            1
        );
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
    }

    #[test]
    fn a_panicking_start_releases_the_transition_lock() {
        let gate = ProviderAdmissionGate::initially_open_for_test();

        assert!(
            catch_unwind(AssertUnwindSafe(|| gate.start_owned(|| panic!("injected")))).is_err()
        );
        assert!(gate.close());
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
    }

    #[test]
    fn debug_output_exposes_only_the_phase() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();
        assert_eq!(
            format!("{gate:?}"),
            "ProviderAdmissionGate { phase: Pending }"
        );
        let token_debug = format!("{token:?}");
        assert!(token_debug.contains("phase: Pending"));
        assert!(!token_debug.contains("inner"));
        token.rollback();
    }

    #[test]
    fn unissued_pending_gate_stays_inert_when_dropped() {
        let gate = ProviderAdmissionGate::new_pending();
        let observer = gate.clone();

        drop(gate);

        assert_eq!(observer.phase(), ProviderAdmissionPhase::Pending);
        assert_eq!(
            observer.start_owned(|| unreachable!()),
            Err(ProviderStartRefusal::Pending)
        );
        assert!(observer.close());
    }

    #[test]
    fn repeated_token_issuance_fails_deterministically() {
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();

        assert_eq!(
            gate.activation_token().unwrap_err(),
            ProviderActivationTokenError::AlreadyIssued
        );
        drop(token);
        assert_eq!(
            gate.activation_token().unwrap_err(),
            ProviderActivationTokenError::AlreadyIssued
        );
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Open);
    }

    #[test]
    fn close_before_token_issuance_prevents_commit_authority() {
        let gate = ProviderAdmissionGate::new_pending();

        assert!(gate.close());
        assert_eq!(
            gate.activation_token().unwrap_err(),
            ProviderActivationTokenError::Closed
        );
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
    }

    #[test]
    fn concurrent_token_issuance_has_exactly_one_winner() {
        let gate = ProviderAdmissionGate::new_pending();
        let ready = Arc::new(Barrier::new(9));
        let mut issuers = Vec::new();
        for _ in 0..8 {
            let issue_gate = gate.clone();
            let issue_ready = ready.clone();
            issuers.push(thread::spawn(move || {
                issue_ready.wait();
                issue_gate.activation_token()
            }));
        }
        ready.wait();

        let mut winner = None;
        let mut already_issued = 0;
        for issuer in issuers {
            match issuer.join().unwrap() {
                Ok(token) => {
                    assert!(winner.replace(token).is_none());
                }
                Err(ProviderActivationTokenError::AlreadyIssued) => already_issued += 1,
                Err(ProviderActivationTokenError::Closed) => {
                    panic!("an open issuance race returned Closed")
                }
            }
        }
        assert_eq!(already_issued, 7);
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Pending);
        drop(winner.expect("one issuer must own the activation token"));
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Open);
    }
}
