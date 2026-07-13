// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded ownership shell for Router reconciliation work.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tokio::task::{AbortHandle, Id, JoinError, JoinHandle, JoinSet};
use tokio::time::{MissedTickBehavior, interval_at};
use uuid::Uuid;

use crate::embedder::FrozenEmbedderClients;
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::SchemaVerificationReport;
use crate::ledger::repository::vector_registry::VectorRegistryEnsure;
use crate::ledger::writer::LedgerWriterClient;
use crate::provider_admission::{
    ProviderAdmissionGate, ProviderAdmissionPhase, ProviderStartRefusal,
};

const RECONCILIATION_INTERVAL: Duration = Duration::from_secs(30);
const DRIVER_CONTRACT_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.driver_contract_failure");
const TASK_JOIN_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.task_join_failure");

pub(crate) type BackgroundPassFuture =
    Pin<Box<dyn Future<Output = Result<Vec<BackgroundWork>, BackgroundFailure>> + Send + 'static>>;
pub(crate) type BackgroundFailureNotifier = Arc<dyn Fn(BackgroundFailure) + Send + Sync + 'static>;
type BackgroundWorkFuture =
    Pin<Box<dyn Future<Output = Result<(), BackgroundFailure>> + Send + 'static>>;
type BackgroundWorkFactory =
    Box<dyn FnOnce(BackgroundCancellation) -> BackgroundWorkFuture + Send + 'static>;
type BackgroundJoinResult = Result<(Id, Result<(), BackgroundFailure>), JoinError>;

/// Why the supervisor is asking its driver to discover work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackgroundPass {
    Startup,
    Hint,
    Periodic,
}

/// Stable, non-secret failure returned by a driver or one owned work task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackgroundFailure {
    code: &'static str,
}

impl BackgroundFailure {
    pub(crate) const fn new(code: &'static str) -> Self {
        Self { code }
    }

    pub(crate) const fn code(self) -> &'static str {
        self.code
    }
}

/// Cancellation observed by reconciliation passes and every owned work task.
#[derive(Clone)]
pub(crate) struct BackgroundCancellation {
    inner: Arc<BackgroundCancellationInner>,
}

struct BackgroundCancellationInner {
    cancelled: AtomicBool,
    changed: watch::Sender<bool>,
    shutdown_deadline: Mutex<Option<Instant>>,
}

impl BackgroundCancellation {
    pub(crate) fn new() -> Self {
        let (changed, _) = watch::channel(false);
        Self {
            inner: Arc::new(BackgroundCancellationInner {
                cancelled: AtomicBool::new(false),
                changed,
                shutdown_deadline: Mutex::new(None),
            }),
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub(crate) async fn cancelled(&self) {
        let mut changed = self.subscribe();
        loop {
            if self.is_cancelled() {
                return;
            }
            if changed.changed().await.is_err() {
                return;
            }
        }
    }

    /// Return a fresh cancellation receiver for provider and writer operations.
    pub(crate) fn subscribe(&self) -> watch::Receiver<bool> {
        self.inner.changed.subscribe()
    }

    pub(crate) fn shutdown_deadline(&self) -> Option<Instant> {
        *self
            .inner
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    pub(crate) fn cancel_until(&self, deadline: Instant) -> bool {
        let mut current = self
            .inner
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if current.is_none_or(|current| deadline < current) {
            *current = Some(deadline);
        }
        drop(current);
        self.cancel()
    }

    fn cancel(&self) -> bool {
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            return false;
        }
        self.inner.changed.send_replace(true);
        true
    }
}

/// Per-pass limits and cancellation supplied by the supervisor.
#[derive(Clone)]
#[allow(dead_code)] // The concrete Task 10 driver consumes this context next.
pub(crate) struct BackgroundPassContext {
    cancellation: BackgroundCancellation,
    work_budget: usize,
}

#[allow(dead_code)] // The concrete Task 10 driver consumes these views next.
impl BackgroundPassContext {
    pub(crate) fn cancellation(&self) -> &BackgroundCancellation {
        &self.cancellation
    }

    pub(crate) const fn work_budget(&self) -> usize {
        self.work_budget
    }
}

/// One inert plan that becomes live only inside `ProviderAdmissionGate::start_owned`.
pub(crate) struct BackgroundWork {
    start: BackgroundWorkFactory,
}

impl BackgroundWork {
    #[allow(dead_code)] // The concrete Task 10 driver creates owned plans next.
    pub(crate) fn new<F, Fut>(start: F) -> Self
    where
        F: FnOnce(BackgroundCancellation) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), BackgroundFailure>> + Send + 'static,
    {
        Self {
            start: Box::new(move |cancellation| Box::pin(start(cancellation))),
        }
    }
}

/// Narrow discovery boundary implemented by concrete background workflows.
pub(crate) trait BackgroundDriver: Send + Sync {
    fn run_pass(
        &self,
        pass: BackgroundPass,
        context: BackgroundPassContext,
    ) -> BackgroundPassFuture;
}

/// Runtime resources retained for the concrete driver installed in the next step.
pub(crate) struct BackgroundResources {
    writer: LedgerWriterClient,
    read_pool: LedgerReadPool,
    process_instance_id: Uuid,
    vector_registry: Arc<VectorRegistryEnsure>,
    embedder_clients: Arc<FrozenEmbedderClients>,
    initial_schema_report: SchemaVerificationReport,
}

impl BackgroundResources {
    pub(crate) fn new(
        writer: LedgerWriterClient,
        read_pool: LedgerReadPool,
        process_instance_id: Uuid,
        vector_registry: Arc<VectorRegistryEnsure>,
        embedder_clients: Arc<FrozenEmbedderClients>,
        initial_schema_report: SchemaVerificationReport,
    ) -> Self {
        Self {
            writer,
            read_pool,
            process_instance_id,
            vector_registry,
            embedder_clients,
            initial_schema_report,
        }
    }

    #[allow(dead_code)] // Concrete Task 10 drivers consume these resource views next.
    pub(crate) fn writer(&self) -> &LedgerWriterClient {
        &self.writer
    }

    #[allow(dead_code)] // Concrete Task 10 drivers consume these resource views next.
    pub(crate) fn read_pool(&self) -> &LedgerReadPool {
        &self.read_pool
    }

    #[allow(dead_code)] // Concrete Task 10 selectors use current process ownership next.
    pub(crate) const fn process_instance_id(&self) -> Uuid {
        self.process_instance_id
    }

    #[allow(dead_code)] // Concrete Task 10 drivers consume these resource views next.
    pub(crate) fn embedder_clients(&self) -> &Arc<FrozenEmbedderClients> {
        &self.embedder_clients
    }

    #[allow(dead_code)] // Concrete Task 10 drivers consume frozen space authority next.
    pub(crate) fn vector_registry(&self) -> &Arc<VectorRegistryEnsure> {
        &self.vector_registry
    }

    #[allow(dead_code)] // Concrete Task 10 drivers consume these resource views next.
    pub(crate) fn initial_schema_report(&self) -> &SchemaVerificationReport {
        &self.initial_schema_report
    }
}

/// Result of a non-blocking hint submission to the capacity-one channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Producers are wired with the concrete Task 10 driver next.
pub(crate) enum BackgroundHintOutcome {
    Queued,
    Coalesced,
    Closed,
}

/// Cloneable producer for the bounded, coalescing reconciliation hint channel.
#[derive(Clone)]
#[allow(dead_code)] // Producers are wired with the concrete Task 10 driver next.
pub(crate) struct BackgroundHintSender {
    sender: mpsc::Sender<()>,
}

#[allow(dead_code)] // Producers are wired with the concrete Task 10 driver next.
impl BackgroundHintSender {
    pub(crate) fn notify(&self) -> BackgroundHintOutcome {
        match self.sender.try_send(()) {
            Ok(()) => BackgroundHintOutcome::Queued,
            Err(mpsc::error::TrySendError::Full(())) => BackgroundHintOutcome::Coalesced,
            Err(mpsc::error::TrySendError::Closed(())) => BackgroundHintOutcome::Closed,
        }
    }
}

#[derive(Default)]
struct OwnedTaskAborts {
    handles: Mutex<HashMap<Id, AbortHandle>>,
}

impl OwnedTaskAborts {
    fn insert(&self, handle: AbortHandle) {
        self.handles
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(handle.id(), handle);
    }

    fn remove(&self, id: Id) {
        self.handles
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&id);
    }

    fn abort_all(&self) {
        let handles = self
            .handles
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for handle in handles.values() {
            handle.abort();
        }
    }

    fn clear(&self) {
        self.handles
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }
}

/// Final accounting from the sole background supervisor task.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BackgroundExit {
    first_failure: Option<BackgroundFailure>,
    accepted_work: usize,
    completed_work: usize,
}

impl BackgroundExit {
    pub(crate) const fn first_failure(self) -> Option<BackgroundFailure> {
        self.first_failure
    }

    #[cfg(test)]
    const fn accepted_work(self) -> usize {
        self.accepted_work
    }

    #[cfg(test)]
    const fn completed_work(self) -> usize {
        self.completed_work
    }

    fn record_failure(&mut self, failure: BackgroundFailure) -> bool {
        if self.first_failure.is_some() {
            return false;
        }
        self.first_failure = Some(failure);
        true
    }
}

/// Owner of the supervisor, its cancellation authority, and synchronous abort handles.
pub(crate) struct BackgroundRuntime {
    gate: ProviderAdmissionGate,
    cancellation: BackgroundCancellation,
    hints: BackgroundHintSender,
    owned_task_aborts: Arc<OwnedTaskAborts>,
    supervisor: JoinHandle<BackgroundExit>,
}

/// Cloneable synchronous control retained outside a possibly in-flight drain.
#[derive(Clone)]
pub(crate) struct BackgroundAbortHandle {
    gate: ProviderAdmissionGate,
    cancellation: BackgroundCancellation,
    owned_task_aborts: Arc<OwnedTaskAborts>,
    supervisor: AbortHandle,
}

impl BackgroundAbortHandle {
    /// Close claims first and then request graceful cancellation.
    pub(crate) fn stop(&self) {
        self.gate.close();
        self.cancellation.cancel();
    }

    /// Install the shared shutdown deadline before requesting cleanup.
    pub(crate) fn stop_until(&self, deadline: Instant) {
        self.gate.close();
        self.cancellation.cancel_until(deadline);
    }

    /// Request cancellation of children before aborting the supervisor itself.
    pub(crate) fn abort(&self) {
        self.stop();
        self.owned_task_aborts.abort_all();
        self.supervisor.abort();
    }
}

impl BackgroundRuntime {
    pub(crate) fn start(
        runtime: &Handle,
        driver: Arc<dyn BackgroundDriver>,
        gate: ProviderAdmissionGate,
        failure_notifier: BackgroundFailureNotifier,
        max_owned_tasks: usize,
    ) -> Self {
        assert!(max_owned_tasks > 0, "background task bound must be nonzero");
        let cancellation = BackgroundCancellation::new();
        let (hint_tx, hint_rx) = mpsc::channel(1);
        let hints = BackgroundHintSender { sender: hint_tx };
        let owned_task_aborts = Arc::new(OwnedTaskAborts::default());
        let supervisor = runtime.spawn(run_supervisor(
            driver,
            gate.clone(),
            cancellation.clone(),
            failure_notifier,
            hint_rx,
            owned_task_aborts.clone(),
            max_owned_tasks,
        ));
        Self {
            gate,
            cancellation,
            hints,
            owned_task_aborts,
            supervisor,
        }
    }

    #[allow(dead_code)] // Concrete workflows retain this producer to request passes.
    pub(crate) fn hint_sender(&self) -> BackgroundHintSender {
        self.hints.clone()
    }

    pub(crate) fn abort_handle(&self) -> BackgroundAbortHandle {
        BackgroundAbortHandle {
            gate: self.gate.clone(),
            cancellation: self.cancellation.clone(),
            owned_task_aborts: self.owned_task_aborts.clone(),
            supervisor: self.supervisor.abort_handle(),
        }
    }

    /// Close all future starts first, then notify running work to finish gracefully.
    #[allow(dead_code)] // Deadline-free stop remains useful to focused supervisor tests.
    pub(crate) fn stop(&self) {
        self.abort_handle().stop();
    }

    pub(crate) fn stop_until(&self, deadline: Instant) {
        self.abort_handle().stop_until(deadline);
    }

    /// Signal cancellation and synchronously request abort of all owned task handles.
    pub(crate) fn abort(&self) {
        self.abort_handle().abort();
    }

    pub(crate) async fn join(&mut self) -> Result<BackgroundExit, JoinError> {
        (&mut self.supervisor).await
    }
}

impl Drop for BackgroundRuntime {
    fn drop(&mut self) {
        self.abort();
    }
}

async fn run_supervisor(
    driver: Arc<dyn BackgroundDriver>,
    gate: ProviderAdmissionGate,
    cancellation: BackgroundCancellation,
    failure_notifier: BackgroundFailureNotifier,
    mut hints: mpsc::Receiver<()>,
    owned_task_aborts: Arc<OwnedTaskAborts>,
    max_owned_tasks: usize,
) -> BackgroundExit {
    let mut exit = BackgroundExit::default();
    let mut tasks = JoinSet::new();

    if !wait_for_activation(&gate, &cancellation).await {
        return exit;
    }

    let mut periodic = interval_at(
        tokio::time::Instant::now() + RECONCILIATION_INTERVAL,
        RECONCILIATION_INTERVAL,
    );
    periodic.set_missed_tick_behavior(MissedTickBehavior::Skip);
    run_pass(
        driver.as_ref(),
        BackgroundPass::Startup,
        &gate,
        &cancellation,
        &failure_notifier,
        &owned_task_aborts,
        &mut tasks,
        max_owned_tasks,
        &mut exit,
    )
    .await;

    loop {
        if cancellation.is_cancelled() {
            break;
        }
        let gate_open = gate.phase() == ProviderAdmissionPhase::Open;
        if !gate_open && tasks.is_empty() {
            break;
        }

        tokio::select! {
            biased;
            _ = cancellation.cancelled() => {}
            _ = gate.wait_for_change(ProviderAdmissionPhase::Open), if gate_open => {}
            joined = tasks.join_next_with_id(), if !tasks.is_empty() => {
                if record_join(
                    joined,
                    &gate,
                    &cancellation,
                    &failure_notifier,
                    &owned_task_aborts,
                    &mut exit,
                )
                    && exit.first_failure.is_none()
                {
                    run_pass(
                        driver.as_ref(),
                        BackgroundPass::Hint,
                        &gate,
                        &cancellation,
                        &failure_notifier,
                        &owned_task_aborts,
                        &mut tasks,
                        max_owned_tasks,
                        &mut exit,
                    ).await;
                }
            }
            hint = hints.recv() => {
                if hint.is_some() {
                    run_pass(
                        driver.as_ref(),
                        BackgroundPass::Hint,
                        &gate,
                        &cancellation,
                        &failure_notifier,
                        &owned_task_aborts,
                        &mut tasks,
                        max_owned_tasks,
                        &mut exit,
                    ).await;
                }
            }
            _ = periodic.tick() => {
                run_pass(
                    driver.as_ref(),
                    BackgroundPass::Periodic,
                    &gate,
                    &cancellation,
                    &failure_notifier,
                    &owned_task_aborts,
                    &mut tasks,
                    max_owned_tasks,
                    &mut exit,
                ).await;
            }
        }
    }

    while let Some(joined) = tasks.join_next_with_id().await {
        let _ = record_join(
            Some(joined),
            &gate,
            &cancellation,
            &failure_notifier,
            &owned_task_aborts,
            &mut exit,
        );
    }
    owned_task_aborts.clear();
    exit
}

async fn wait_for_activation(
    gate: &ProviderAdmissionGate,
    cancellation: &BackgroundCancellation,
) -> bool {
    loop {
        match gate.phase() {
            ProviderAdmissionPhase::Open => return true,
            ProviderAdmissionPhase::Closed => return false,
            ProviderAdmissionPhase::Pending => {
                tokio::select! {
                    biased;
                    _ = cancellation.cancelled() => return false,
                    _ = gate.wait_for_change(ProviderAdmissionPhase::Pending) => {}
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_pass(
    driver: &dyn BackgroundDriver,
    pass: BackgroundPass,
    gate: &ProviderAdmissionGate,
    cancellation: &BackgroundCancellation,
    failure_notifier: &BackgroundFailureNotifier,
    owned_task_aborts: &OwnedTaskAborts,
    tasks: &mut JoinSet<Result<(), BackgroundFailure>>,
    max_owned_tasks: usize,
    exit: &mut BackgroundExit,
) {
    if cancellation.is_cancelled() || gate.phase() != ProviderAdmissionPhase::Open {
        return;
    }
    let work_budget = max_owned_tasks.saturating_sub(tasks.len());
    if work_budget == 0 {
        return;
    }
    let context = BackgroundPassContext {
        cancellation: cancellation.clone(),
        work_budget,
    };
    let planned = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return,
        planned = driver.run_pass(pass, context) => planned,
    };
    let planned = match planned {
        Ok(planned) if planned.len() <= work_budget => planned,
        Ok(_) => {
            publish_failure(
                exit,
                DRIVER_CONTRACT_FAILURE,
                gate,
                cancellation,
                failure_notifier,
            );
            return;
        }
        Err(failure) => {
            publish_failure(exit, failure, gate, cancellation, failure_notifier);
            return;
        }
    };

    for work in planned {
        if cancellation.is_cancelled() {
            break;
        }
        let task_cancellation = cancellation.clone();
        match gate.start_owned(|| {
            let future = (work.start)(task_cancellation);
            let abort = tasks.spawn(future);
            owned_task_aborts.insert(abort);
        }) {
            Ok(()) => exit.accepted_work = exit.accepted_work.saturating_add(1),
            Err(ProviderStartRefusal::Closed) => break,
            Err(ProviderStartRefusal::Pending) => {
                publish_failure(
                    exit,
                    DRIVER_CONTRACT_FAILURE,
                    gate,
                    cancellation,
                    failure_notifier,
                );
                break;
            }
        }
    }
}

fn record_join(
    joined: Option<BackgroundJoinResult>,
    gate: &ProviderAdmissionGate,
    cancellation: &BackgroundCancellation,
    failure_notifier: &BackgroundFailureNotifier,
    owned_task_aborts: &OwnedTaskAborts,
    exit: &mut BackgroundExit,
) -> bool {
    let Some(joined) = joined else {
        return false;
    };
    match joined {
        Ok((id, result)) => {
            owned_task_aborts.remove(id);
            exit.completed_work = exit.completed_work.saturating_add(1);
            match result {
                Ok(()) => true,
                Err(failure) => {
                    publish_failure(exit, failure, gate, cancellation, failure_notifier);
                    false
                }
            }
        }
        Err(error) => {
            owned_task_aborts.remove(error.id());
            exit.completed_work = exit.completed_work.saturating_add(1);
            publish_failure(
                exit,
                TASK_JOIN_FAILURE,
                gate,
                cancellation,
                failure_notifier,
            );
            false
        }
    }
}

fn publish_failure(
    exit: &mut BackgroundExit,
    failure: BackgroundFailure,
    gate: &ProviderAdmissionGate,
    cancellation: &BackgroundCancellation,
    failure_notifier: &BackgroundFailureNotifier,
) {
    if !exit.record_failure(failure) {
        return;
    }
    gate.close();
    cancellation.cancel();
    failure_notifier(failure);
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, mpsc as std_mpsc};
    use std::time::Duration;

    use tokio::sync::Notify;

    use super::*;

    #[test]
    fn shutdown_deadline_is_installed_after_an_earlier_stop_and_only_tightens() {
        let cancellation = BackgroundCancellation::new();
        assert!(cancellation.cancel());
        let later = Instant::now() + Duration::from_secs(2);
        assert!(!cancellation.cancel_until(later));
        assert_eq!(cancellation.shutdown_deadline(), Some(later));

        let earlier = later - Duration::from_secs(1);
        assert!(!cancellation.cancel_until(earlier));
        assert_eq!(cancellation.shutdown_deadline(), Some(earlier));
        assert!(!cancellation.cancel_until(later));
        assert_eq!(cancellation.shutdown_deadline(), Some(earlier));
    }

    fn ignore_failures() -> BackgroundFailureNotifier {
        Arc::new(|_| {})
    }

    #[derive(Default)]
    struct RecordingDriver {
        passes: Mutex<Vec<BackgroundPass>>,
        changed: Notify,
    }

    impl RecordingDriver {
        async fn wait_for_passes(&self, count: usize) {
            loop {
                let changed = self.changed.notified();
                if self
                    .passes
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .len()
                    >= count
                {
                    return;
                }
                changed.await;
            }
        }

        fn snapshot(&self) -> Vec<BackgroundPass> {
            self.passes
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        }
    }

    impl BackgroundDriver for RecordingDriver {
        fn run_pass(
            &self,
            pass: BackgroundPass,
            _context: BackgroundPassContext,
        ) -> BackgroundPassFuture {
            self.passes
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(pass);
            self.changed.notify_waiters();
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    struct SingleWorkDriver {
        work: Mutex<Option<BackgroundWork>>,
        pass_seen: Arc<AtomicBool>,
        pass_changed: Arc<Notify>,
    }

    impl SingleWorkDriver {
        fn new(work: BackgroundWork) -> Self {
            Self {
                work: Mutex::new(Some(work)),
                pass_seen: Arc::new(AtomicBool::new(false)),
                pass_changed: Arc::new(Notify::new()),
            }
        }

        async fn wait_for_pass(&self) {
            loop {
                let changed = self.pass_changed.notified();
                if self.pass_seen.load(Ordering::Acquire) {
                    return;
                }
                changed.await;
            }
        }
    }

    impl BackgroundDriver for SingleWorkDriver {
        fn run_pass(
            &self,
            pass: BackgroundPass,
            _context: BackgroundPassContext,
        ) -> BackgroundPassFuture {
            self.pass_seen.store(true, Ordering::Release);
            self.pass_changed.notify_waiters();
            let work = if pass == BackgroundPass::Startup {
                self.work
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .into_iter()
                    .collect()
            } else {
                Vec::new()
            };
            Box::pin(async move { Ok(work) })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn startup_and_periodic_pass_use_thirty_second_cadence() {
        let driver = Arc::new(RecordingDriver::default());
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver.clone(),
            ProviderAdmissionGate::initially_open_for_test(),
            ignore_failures(),
            1,
        );

        driver.wait_for_passes(1).await;
        assert_eq!(driver.snapshot(), vec![BackgroundPass::Startup]);
        tokio::time::advance(Duration::from_secs(29)).await;
        tokio::task::yield_now().await;
        assert_eq!(driver.snapshot(), vec![BackgroundPass::Startup]);
        tokio::time::advance(Duration::from_secs(1)).await;
        driver.wait_for_passes(2).await;
        assert_eq!(
            driver.snapshot(),
            vec![BackgroundPass::Startup, BackgroundPass::Periodic]
        );

        runtime.stop();
        assert!(runtime.join().await.unwrap().first_failure().is_none());
    }

    #[tokio::test]
    async fn capacity_one_hint_channel_coalesces_while_activation_is_pending() {
        let driver = Arc::new(RecordingDriver::default());
        let gate = ProviderAdmissionGate::new_pending();
        let token = gate.activation_token().unwrap();
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver.clone(),
            gate,
            ignore_failures(),
            1,
        );
        let hints = runtime.hint_sender();

        assert_eq!(hints.notify(), BackgroundHintOutcome::Queued);
        for _ in 0..100 {
            assert_eq!(hints.notify(), BackgroundHintOutcome::Coalesced);
        }
        drop(token);
        driver.wait_for_passes(2).await;
        assert_eq!(
            driver.snapshot(),
            vec![BackgroundPass::Startup, BackgroundPass::Hint]
        );

        runtime.stop();
        assert!(runtime.join().await.unwrap().first_failure().is_none());
    }

    #[tokio::test]
    async fn oversized_driver_pass_is_rejected_without_spawning_work() {
        struct OversizedDriver(Arc<AtomicUsize>);

        impl BackgroundDriver for OversizedDriver {
            fn run_pass(
                &self,
                _pass: BackgroundPass,
                context: BackgroundPassContext,
            ) -> BackgroundPassFuture {
                assert_eq!(context.work_budget(), 2);
                let started = self.0.clone();
                Box::pin(async move {
                    Ok((0..3)
                        .map(|_| {
                            let started = started.clone();
                            BackgroundWork::new(move |_| async move {
                                started.fetch_add(1, Ordering::AcqRel);
                                Ok(())
                            })
                        })
                        .collect())
                })
            }
        }

        let started = Arc::new(AtomicUsize::new(0));
        let driver = Arc::new(OversizedDriver(started.clone()));
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver,
            ProviderAdmissionGate::initially_open_for_test(),
            ignore_failures(),
            2,
        );
        tokio::task::yield_now().await;
        runtime.stop();
        let exit = runtime.join().await.unwrap();

        assert_eq!(started.load(Ordering::Acquire), 0);
        assert_eq!(exit.first_failure(), Some(DRIVER_CONTRACT_FAILURE));
        assert_eq!(exit.accepted_work(), 0);
    }

    #[tokio::test]
    async fn first_owned_failure_closes_gate_notifies_once_and_joins_all_work() {
        const TEST_FAILURE: BackgroundFailure =
            BackgroundFailure::new("router.test.background_failure");

        struct FailingWorkDriver;

        impl BackgroundDriver for FailingWorkDriver {
            fn run_pass(
                &self,
                _pass: BackgroundPass,
                _context: BackgroundPassContext,
            ) -> BackgroundPassFuture {
                Box::pin(async {
                    Ok((0..2)
                        .map(|_| BackgroundWork::new(|_| async { Err(TEST_FAILURE) }))
                        .collect())
                })
            }
        }

        let gate = ProviderAdmissionGate::initially_open_for_test();
        let notifications = Arc::new(Mutex::new(Vec::new()));
        let failure_notifier: BackgroundFailureNotifier = {
            let notifications = notifications.clone();
            Arc::new(move |failure| {
                notifications
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(failure.code());
            })
        };
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            Arc::new(FailingWorkDriver),
            gate.clone(),
            failure_notifier,
            2,
        );

        let exit = runtime.join().await.unwrap();

        assert_eq!(exit.first_failure(), Some(TEST_FAILURE));
        assert_eq!(exit.accepted_work(), 2);
        assert_eq!(exit.completed_work(), 2);
        assert_eq!(
            *notifications
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec![TEST_FAILURE.code()]
        );
        assert_eq!(gate.phase(), ProviderAdmissionPhase::Closed);
        let post_failure_starts = AtomicUsize::new(0);
        assert_eq!(
            gate.start_owned(|| post_failure_starts.fetch_add(1, Ordering::AcqRel)),
            Err(ProviderStartRefusal::Closed)
        );
        assert_eq!(post_failure_starts.load(Ordering::Acquire), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn accepted_spawn_linearizes_before_a_racing_close() {
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let (entered_tx, entered_rx) = std_mpsc::channel();
        let (release_tx, release_rx) = std_mpsc::channel();
        let work = BackgroundWork::new(move |cancellation| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            async move {
                cancellation.cancelled().await;
                Ok(())
            }
        });
        let driver = Arc::new(SingleWorkDriver::new(work));
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver,
            gate.clone(),
            ignore_failures(),
            1,
        );

        tokio::task::spawn_blocking(move || entered_rx.recv().unwrap())
            .await
            .unwrap();
        let (close_attempted_tx, close_attempted_rx) = std_mpsc::channel();
        let (closed_tx, closed_rx) = std_mpsc::channel();
        let close_thread = std::thread::spawn(move || {
            close_attempted_tx.send(()).unwrap();
            let closed = gate.close();
            closed_tx.send(closed).unwrap();
        });
        close_attempted_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        assert_eq!(
            closed_rx.recv_timeout(Duration::from_millis(50)),
            Err(std_mpsc::RecvTimeoutError::Timeout)
        );
        release_tx.send(()).unwrap();
        assert!(closed_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        close_thread.join().unwrap();
        assert!(!runtime.supervisor.is_finished());
        runtime.stop_until(Instant::now() + Duration::from_secs(1));

        let exit = runtime.join().await.unwrap();
        assert_eq!(exit.accepted_work(), 1);
        assert_eq!(exit.completed_work(), 1);
        assert!(exit.first_failure().is_none());
    }

    #[tokio::test]
    async fn graceful_stop_joins_owned_work_after_signalling_cancellation() {
        let probe = Arc::new(());
        let weak_probe = Arc::downgrade(&probe);
        let started = Arc::new(AtomicBool::new(false));
        let started_changed = Arc::new(Notify::new());
        let work = BackgroundWork::new({
            let started = started.clone();
            let started_changed = started_changed.clone();
            move |cancellation| {
                let probe = probe;
                let mut cancellation = cancellation.subscribe();
                async move {
                    let _probe = probe;
                    started.store(true, Ordering::Release);
                    started_changed.notify_waiters();
                    if !*cancellation.borrow() {
                        cancellation.changed().await.unwrap();
                    }
                    assert!(*cancellation.borrow());
                    Ok(())
                }
            }
        });
        let driver = Arc::new(SingleWorkDriver::new(work));
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver,
            ProviderAdmissionGate::initially_open_for_test(),
            ignore_failures(),
            1,
        );
        loop {
            let changed = started_changed.notified();
            if started.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }

        runtime.stop();
        let exit = runtime.join().await.unwrap();
        assert_eq!(exit.accepted_work(), 1);
        assert_eq!(exit.completed_work(), 1);
        assert!(weak_probe.upgrade().is_none());
    }

    #[tokio::test]
    async fn gate_close_preserves_accepted_work_until_timed_stop() {
        let started = Arc::new(AtomicBool::new(false));
        let started_changed = Arc::new(Notify::new());
        let cancelled = Arc::new(AtomicBool::new(false));
        let work = BackgroundWork::new({
            let started = started.clone();
            let started_changed = started_changed.clone();
            let cancelled = cancelled.clone();
            move |cancellation| async move {
                started.store(true, Ordering::Release);
                started_changed.notify_waiters();
                cancellation.cancelled().await;
                cancelled.store(true, Ordering::Release);
                Ok(())
            }
        });
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            Arc::new(SingleWorkDriver::new(work)),
            gate.clone(),
            ignore_failures(),
            1,
        );
        loop {
            let changed = started_changed.notified();
            if started.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }

        assert!(gate.close());
        tokio::task::yield_now().await;
        assert!(!cancelled.load(Ordering::Acquire));
        assert!(!runtime.supervisor.is_finished());

        runtime.stop_until(Instant::now() + Duration::from_secs(1));
        let exit = runtime.join().await.unwrap();
        assert!(cancelled.load(Ordering::Acquire));
        assert_eq!(exit.accepted_work(), 1);
        assert_eq!(exit.completed_work(), 1);
    }

    #[tokio::test]
    async fn abort_requests_owned_task_abort_before_supervisor_abort() {
        let probe = Arc::new(());
        let weak_probe = Arc::downgrade(&probe);
        let started = Arc::new(AtomicBool::new(false));
        let started_changed = Arc::new(Notify::new());
        let work = BackgroundWork::new({
            let started = started.clone();
            let started_changed = started_changed.clone();
            move |_| {
                let probe = probe;
                async move {
                    let _probe = probe;
                    started.store(true, Ordering::Release);
                    started_changed.notify_waiters();
                    std::future::pending::<()>().await;
                    Ok(())
                }
            }
        });
        let driver = Arc::new(SingleWorkDriver::new(work));
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver,
            ProviderAdmissionGate::initially_open_for_test(),
            ignore_failures(),
            1,
        );
        loop {
            let changed = started_changed.notified();
            if started.load(Ordering::Acquire) {
                break;
            }
            changed.await;
        }

        runtime.abort();
        assert!(runtime.join().await.unwrap_err().is_cancelled());
        assert!(weak_probe.upgrade().is_none());
    }

    #[tokio::test]
    async fn closed_gate_rejects_a_plan_before_its_work_factory_runs() {
        struct ClosingDriver {
            gate: ProviderAdmissionGate,
            started: Arc<AtomicUsize>,
        }

        impl BackgroundDriver for ClosingDriver {
            fn run_pass(
                &self,
                _pass: BackgroundPass,
                _context: BackgroundPassContext,
            ) -> BackgroundPassFuture {
                self.gate.close();
                let started = self.started.clone();
                Box::pin(async move {
                    Ok(vec![BackgroundWork::new(move |_| async move {
                        started.fetch_add(1, Ordering::AcqRel);
                        Ok(())
                    })])
                })
            }
        }

        let gate = ProviderAdmissionGate::initially_open_for_test();
        let started = Arc::new(AtomicUsize::new(0));
        let driver = Arc::new(ClosingDriver {
            gate: gate.clone(),
            started: started.clone(),
        });
        let mut runtime =
            BackgroundRuntime::start(&Handle::current(), driver, gate, ignore_failures(), 1);

        let exit = runtime.join().await.unwrap();
        assert_eq!(started.load(Ordering::Acquire), 0);
        assert_eq!(exit.accepted_work(), 0);
        assert!(exit.first_failure().is_none());
    }

    #[tokio::test]
    async fn single_work_driver_reports_its_startup_pass() {
        let driver = Arc::new(SingleWorkDriver::new(BackgroundWork::new(|_| async {
            Ok(())
        })));
        let mut runtime = BackgroundRuntime::start(
            &Handle::current(),
            driver.clone(),
            ProviderAdmissionGate::initially_open_for_test(),
            ignore_failures(),
            1,
        );

        driver.wait_for_pass().await;
        runtime.stop();
        assert!(runtime.join().await.unwrap().first_failure().is_none());
    }
}
