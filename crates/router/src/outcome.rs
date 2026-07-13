// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure bounded actual-outcome matching and root classification.

use std::collections::{BTreeSet, HashSet};
use std::fmt::{self, Write as _};
use std::sync::Arc;

use nemo_relay::api::event::ScopeCategory;
use nemo_relay::api::llm::LlmCallRole;
use nemo_relay::api::scope::ScopeType;
use serde::Serialize;
use serde_json::Value as Json;
use uuid::Uuid;

use crate::config::{
    OUTCOME_MATCHER_TEXT_MAX_BYTES, OUTCOME_MATCHERS_MAX, OUTCOME_METADATA_EQUALS_MAX,
    OutcomeConfig, OutcomeDisposition, OutcomeMatcher, OutcomeMatcherEventKind,
    OutcomeTerminalStatus, canonical_outcome_scalar_v1, outcome_matcher_hash_v1,
    outcome_metadata_key_hash_v1, outcome_observed_scalar_hash_v1,
};
use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
use crate::ledger::cohort::RootKey;
use crate::trajectory::{CapturedEventKind, CapturedTrajectoryEvent};

pub(crate) const OUTCOME_SIGNALS_MAX: usize = 64;
pub(crate) const OUTCOME_SIGNAL_FACTS_MAX_BYTES: usize = 64 * 1024;
pub(crate) const OUTCOME_OPEN_ROOT_WINDOWS_MAX: usize = 4_096;

const OUTCOME_EVENT_KEY_DOMAIN_V1: &[u8] = b"nemo-relay-router/outcome-event-key/v1\0";
const OUTCOME_SIGNAL_KEY_DOMAIN_V1: &[u8] = b"nemo-relay-router/outcome-signal-key/v1\0";
const OUTCOME_SIGNAL_AGGREGATE_SCHEMA_V1: &str = "nemo.relay.router.outcome-signals@1";

/// Failure to compile a complete, already config-validated outcome policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutcomeCompileErrorV1 {
    InvalidPolicy(&'static str),
    InvalidMatcherIdentity,
    DuplicateMatcher,
}

struct CompiledMetadataClauseV1 {
    key: String,
    canonical_expected: String,
}

struct CompiledMatcherV1 {
    matcher_hash: String,
    event_kind: OutcomeMatcherEventKind,
    category: String,
    name: String,
    terminal_status: OutcomeTerminalStatus,
    metadata_equals: Vec<CompiledMetadataClauseV1>,
}

/// Live-only compiled policy. Raw expected values are neither serializable nor debuggable.
pub(crate) struct CompiledOutcomePolicyV1 {
    success_matchers: Vec<CompiledMatcherV1>,
    failure_matchers: Vec<CompiledMatcherV1>,
    completion_disposition: OutcomeDisposition,
    error_disposition: OutcomeDisposition,
    tool_failure_disposition: OutcomeDisposition,
    end_of_run_disposition: OutcomeDisposition,
}

impl fmt::Debug for CompiledOutcomePolicyV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledOutcomePolicyV1")
            .field("success_matchers", &self.success_matchers.len())
            .field("failure_matchers", &self.failure_matchers.len())
            .field("completion_disposition", &self.completion_disposition)
            .field("error_disposition", &self.error_disposition)
            .field("tool_failure_disposition", &self.tool_failure_disposition)
            .field("end_of_run_disposition", &self.end_of_run_disposition)
            .finish()
    }
}

impl CompiledOutcomePolicyV1 {
    pub(crate) fn compile(config: &OutcomeConfig) -> Result<Self, OutcomeCompileErrorV1> {
        if config.version != 1 {
            return Err(OutcomeCompileErrorV1::InvalidPolicy("outcome version"));
        }
        if config.success_matchers.is_empty()
            || config.success_matchers.len() > OUTCOME_MATCHERS_MAX
            || config.failure_matchers.is_empty()
            || config.failure_matchers.len() > OUTCOME_MATCHERS_MAX
        {
            return Err(OutcomeCompileErrorV1::InvalidPolicy("matcher list length"));
        }
        if config.error_disposition != OutcomeDisposition::Failure
            || config.tool_failure_disposition != OutcomeDisposition::Failure
        {
            return Err(OutcomeCompileErrorV1::InvalidPolicy(
                "required failure disposition",
            ));
        }

        let mut seen = BTreeSet::new();
        let success_matchers = compile_matcher_list(&config.success_matchers, &mut seen)?;
        let failure_matchers = compile_matcher_list(&config.failure_matchers, &mut seen)?;
        Ok(Self {
            success_matchers,
            failure_matchers,
            completion_disposition: config.completion_disposition,
            error_disposition: config.error_disposition,
            tool_failure_disposition: config.tool_failure_disposition,
            end_of_run_disposition: config.end_of_run_disposition,
        })
    }
}

fn compile_matcher_list(
    matchers: &[OutcomeMatcher],
    seen: &mut BTreeSet<String>,
) -> Result<Vec<CompiledMatcherV1>, OutcomeCompileErrorV1> {
    let mut compiled = Vec::with_capacity(matchers.len());
    for matcher in matchers {
        if matcher.category.is_empty()
            || matcher.category.len() > OUTCOME_MATCHER_TEXT_MAX_BYTES
            || matcher.name.is_empty()
            || matcher.name.len() > OUTCOME_MATCHER_TEXT_MAX_BYTES
            || matcher.metadata_equals.len() > OUTCOME_METADATA_EQUALS_MAX
            || (matcher.event_kind == OutcomeMatcherEventKind::Mark
                && matcher.terminal_status != OutcomeTerminalStatus::Unset)
        {
            return Err(OutcomeCompileErrorV1::InvalidPolicy("matcher shape"));
        }
        if matcher
            .metadata_equals
            .keys()
            .any(|key| !is_outcome_metadata_key(key))
        {
            return Err(OutcomeCompileErrorV1::InvalidPolicy("metadata key"));
        }
        let matcher_hash = outcome_matcher_hash_v1(matcher)
            .map_err(|_| OutcomeCompileErrorV1::InvalidMatcherIdentity)?;
        if !seen.insert(matcher_hash.clone()) {
            return Err(OutcomeCompileErrorV1::DuplicateMatcher);
        }
        let metadata_equals = matcher
            .metadata_equals
            .iter()
            .map(|(key, value)| {
                Ok(CompiledMetadataClauseV1 {
                    key: key.clone(),
                    canonical_expected: canonical_outcome_scalar_v1(value)
                        .map_err(|_| OutcomeCompileErrorV1::InvalidMatcherIdentity)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        compiled.push(CompiledMatcherV1 {
            matcher_hash,
            event_kind: matcher.event_kind,
            category: matcher.category.clone(),
            name: matcher.name.clone(),
            terminal_status: matcher.terminal_status,
            metadata_equals,
        });
    }
    compiled.sort_by(|left, right| left.matcher_hash.cmp(&right.matcher_hash));
    Ok(compiled)
}

fn is_outcome_metadata_key(key: &str) -> bool {
    matches!(
        key,
        "error.type"
            | "outcome"
            | "outcome.label"
            | "outcome.success"
            | "result"
            | "status"
            | "success"
    )
}

/// Frozen ancestry result supplied by the existing trajectory coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PinnedOwnerRelationV1 {
    ExternalPinnedOwnerSubtree,
    InternalPinnedOwnerSubtree,
    Outside,
    Unreadable,
}

/// Correlation facts supplied with one already bounded Core event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutcomeEventContextV1 {
    pub(crate) owner_relation: PinnedOwnerRelationV1,
    pub(crate) exact_pinned_owner_end: bool,
}

/// Protected hashes for one equality clause that actually matched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ProtectedObservedClauseV1 {
    key_sha256: String,
    observed_scalar_sha256: String,
}

/// Protected signal source. It contains no configured or observed raw scalar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ProtectedOutcomeSignalSourceV1 {
    Matcher {
        matcher_sha256: String,
        observed: Vec<ProtectedObservedClauseV1>,
    },
    Completion,
    Error,
    ToolFailure,
    EndOfRun,
    PolicyFailure,
}

/// One persistence-safe signal fact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ProtectedOutcomeSignalV1 {
    signal_sha256: String,
    source_event_sha256: Option<String>,
    source: ProtectedOutcomeSignalSourceV1,
    disposition: OutcomeDisposition,
    terminal_status: Option<OutcomeTerminalStatus>,
    ingest_seq: u64,
}

impl ProtectedOutcomeSignalV1 {
    pub(crate) fn signal_sha256(&self) -> &str {
        &self.signal_sha256
    }

    pub(crate) const fn disposition(&self) -> OutcomeDisposition {
        self.disposition
    }
}

#[derive(Serialize)]
struct ProtectedSignalAggregateV1<'a> {
    schema: &'static str,
    signals: &'a [ProtectedOutcomeSignalV1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutcomeCollectionFaultV1 {
    Projection,
    EventLoss,
    OversizedEvent,
    UnreadableRootRelation,
    SignalCountExceeded,
    SignalBytesExceeded,
    ConflictingRepresentativeTerminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutcomeObservationV1 {
    Ignored,
    Duplicate,
    Added { signals: usize },
    Incomplete(OutcomeCollectionFaultV1),
}

/// Direct result of the one representative admitted Primary call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RepresentativeTerminalV1 {
    Completed,
    Error,
    Cancelled,
    Panicked,
    Aborted,
    Missing,
    UnknownAfterCrash,
}

/// Atomic direct-result and response-codec classification for the representative call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RepresentativeResultV1 {
    pub(crate) terminal: RepresentativeTerminalV1,
    pub(crate) codec_policy_failure: bool,
}

impl RepresentativeTerminalV1 {
    const fn is_coherent_direct_result(self) -> bool {
        matches!(self, Self::Completed | Self::Error)
    }

    const fn is_orphaned(self) -> bool {
        matches!(self, Self::UnknownAfterCrash)
    }

    const fn is_unattributed(self) -> bool {
        matches!(
            self,
            Self::Cancelled | Self::Panicked | Self::Aborted | Self::Missing
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EventBatchKeyV1 {
    event_uuid: Uuid,
    kind: &'static str,
    scope_phase: Option<&'static str>,
    canonical_payload_hash: String,
}

/// Bounded mutable signal prefix for one raw-root attribution window.
pub(crate) struct OutcomeSignalAccumulatorV1 {
    policy: Arc<CompiledOutcomePolicyV1>,
    opened_after_ingest_seq: u64,
    signals: Vec<ProtectedOutcomeSignalV1>,
    signal_ids: BTreeSet<String>,
    committed_event_batches: BTreeSet<EventBatchKeyV1>,
    canonical_signal_bytes: usize,
    faults: BTreeSet<OutcomeCollectionFaultV1>,
    representative_result: Option<RepresentativeResultV1>,
}

impl OutcomeSignalAccumulatorV1 {
    pub(crate) fn new(policy: Arc<CompiledOutcomePolicyV1>, opened_after_ingest_seq: u64) -> Self {
        let canonical_signal_bytes = canonical_signal_bytes(&[])
            .expect("the static empty protected signal aggregate is canonical");
        Self {
            policy,
            opened_after_ingest_seq,
            signals: Vec::new(),
            signal_ids: BTreeSet::new(),
            committed_event_batches: BTreeSet::new(),
            canonical_signal_bytes,
            faults: BTreeSet::new(),
            representative_result: None,
        }
    }

    pub(crate) fn signals(&self) -> &[ProtectedOutcomeSignalV1] {
        &self.signals
    }

    pub(crate) const fn canonical_signal_bytes(&self) -> usize {
        self.canonical_signal_bytes
    }

    pub(crate) fn label_complete(&self) -> bool {
        self.faults.is_empty()
    }

    pub(crate) fn faults(&self) -> &BTreeSet<OutcomeCollectionFaultV1> {
        &self.faults
    }

    pub(crate) const fn representative_terminal(&self) -> Option<RepresentativeTerminalV1> {
        match self.representative_result {
            Some(result) => Some(result.terminal),
            None => None,
        }
    }

    pub(crate) fn prefix_label(&self) -> RootOutcomeLabelV1 {
        reduce_signal_label_v1(&self.signals)
    }

    pub(crate) fn observe_event(
        &mut self,
        event: &CapturedTrajectoryEvent,
        context: &OutcomeEventContextV1,
    ) -> OutcomeObservationV1 {
        if let Some(fault) = self.faults.first().copied() {
            return OutcomeObservationV1::Incomplete(fault);
        }
        if event.ingest_seq <= self.opened_after_ingest_seq {
            return OutcomeObservationV1::Ignored;
        }

        // Exclusions use ancestry-derived context and typed call role before any
        // metadata/status inspection.
        if context.exact_pinned_owner_end
            && context.owner_relation != PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree
        {
            return self.fail(OutcomeCollectionFaultV1::UnreadableRootRelation);
        }
        if context.owner_relation == PinnedOwnerRelationV1::InternalPinnedOwnerSubtree
            || matches!(
                event.call_role,
                Some(LlmCallRole::Shadow | LlmCallRole::Judge)
            )
        {
            return OutcomeObservationV1::Ignored;
        }
        match context.owner_relation {
            PinnedOwnerRelationV1::Unreadable => {
                return self.fail(OutcomeCollectionFaultV1::UnreadableRootRelation);
            }
            PinnedOwnerRelationV1::Outside => return OutcomeObservationV1::Ignored,
            PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree => {}
            PinnedOwnerRelationV1::InternalPinnedOwnerSubtree => {
                unreachable!("internal pinned-owner subtrees return before relation processing")
            }
        }

        let event_kind = match projected_event_kind(event) {
            Ok(event_kind) => event_kind,
            Err(()) => return self.fail(OutcomeCollectionFaultV1::Projection),
        };
        if context.exact_pinned_owner_end && event_kind != Some(OutcomeMatcherEventKind::ScopeEnd) {
            return self.fail(OutcomeCollectionFaultV1::UnreadableRootRelation);
        }
        if event_kind.is_none() {
            return OutcomeObservationV1::Ignored;
        }
        if event.category.as_deref() == Some("llm") && event.call_role.is_none() {
            return self.fail(OutcomeCollectionFaultV1::Projection);
        }
        if !is_lower_hex_sha256(&event.canonical_payload_hash) {
            return self.fail(OutcomeCollectionFaultV1::Projection);
        }
        let status = match project_terminal_status(event.metadata.as_ref()) {
            Ok(status) => status,
            Err(()) => return self.fail(OutcomeCollectionFaultV1::Projection),
        };

        let category = event.category.as_deref();
        let coarse_match = self
            .policy
            .success_matchers
            .iter()
            .chain(&self.policy.failure_matchers)
            .any(|matcher| coarse_matcher_matches(matcher, event_kind, category, &event.name));
        let tool_end = event_kind == Some(OutcomeMatcherEventKind::ScopeEnd)
            && event.scope_type == Some(ScopeType::Tool);
        let exact_owner_end = context.exact_pinned_owner_end;
        if !coarse_match && !tool_end && !exact_owner_end {
            return OutcomeObservationV1::Ignored;
        }
        let event_key = EventBatchKeyV1 {
            event_uuid: event.event_uuid,
            kind: match event.kind {
                CapturedEventKind::Scope => "scope",
                CapturedEventKind::Mark => "mark",
            },
            scope_phase: match event.scope_phase {
                Some(ScopeCategory::Start) => Some("start"),
                Some(ScopeCategory::End) => Some("end"),
                None => None,
            },
            canonical_payload_hash: event.canonical_payload_hash.clone(),
        };
        if self.committed_event_batches.contains(&event_key) {
            return OutcomeObservationV1::Duplicate;
        }
        let source_event_sha256 = protected_event_key_sha256(&event_key);
        let metadata = event.metadata.as_ref().and_then(Json::as_object);
        let mut candidates = match event_signal_candidates(
            &self.policy,
            event_kind,
            category,
            &event.name,
            status,
            metadata,
            &source_event_sha256,
            event.ingest_seq,
            tool_end,
            exact_owner_end,
        ) {
            Ok(candidates) => candidates,
            Err(fault) => return self.fail(fault),
        };
        if candidates.is_empty() {
            return OutcomeObservationV1::Ignored;
        }
        candidates.sort_by(|left, right| left.signal_sha256.cmp(&right.signal_sha256));
        candidates.dedup_by(|left, right| left.signal_sha256 == right.signal_sha256);
        match self.commit_batch(candidates) {
            added @ OutcomeObservationV1::Added { .. } => {
                self.committed_event_batches.insert(event_key);
                added
            }
            other => other,
        }
    }

    pub(crate) fn record_representative_result(
        &mut self,
        result: RepresentativeResultV1,
        ingest_seq: u64,
    ) -> OutcomeObservationV1 {
        if let Some(existing) = self.representative_result {
            return if existing == result {
                OutcomeObservationV1::Duplicate
            } else {
                self.fail(OutcomeCollectionFaultV1::ConflictingRepresentativeTerminal)
            };
        }
        if result.codec_policy_failure && result.terminal != RepresentativeTerminalV1::Completed {
            return self.fail(OutcomeCollectionFaultV1::Projection);
        }
        self.representative_result = Some(result);
        if let Some(fault) = self.faults.first().copied() {
            return OutcomeObservationV1::Incomplete(fault);
        }
        let (source, disposition) = match result.terminal {
            RepresentativeTerminalV1::Completed => (
                ProtectedOutcomeSignalSourceV1::Completion,
                self.policy.completion_disposition,
            ),
            RepresentativeTerminalV1::Error => (
                ProtectedOutcomeSignalSourceV1::Error,
                self.policy.error_disposition,
            ),
            RepresentativeTerminalV1::Cancelled
            | RepresentativeTerminalV1::Panicked
            | RepresentativeTerminalV1::Aborted
            | RepresentativeTerminalV1::Missing
            | RepresentativeTerminalV1::UnknownAfterCrash => {
                return OutcomeObservationV1::Ignored;
            }
        };
        let mut candidates = vec![protected_signal(
            None,
            source,
            disposition,
            None,
            ingest_seq,
        )];
        if result.codec_policy_failure {
            candidates.push(protected_signal(
                None,
                ProtectedOutcomeSignalSourceV1::PolicyFailure,
                OutcomeDisposition::Failure,
                None,
                ingest_seq,
            ));
        }
        self.commit_batch(candidates)
    }

    pub(crate) fn mark_fault(&mut self, fault: OutcomeCollectionFaultV1) -> OutcomeObservationV1 {
        self.fail(fault)
    }

    pub(crate) fn classify(
        &self,
        closure: RootClosureV1,
        exposure_facts: &RootExposureFactsV1,
    ) -> RootOutcomeClassificationV1 {
        let scanned = scan_root_exposure_v1(exposure_facts);
        let representative_orphaned = self
            .representative_terminal()
            .is_some_and(RepresentativeTerminalV1::is_orphaned);
        let representative_unattributed = self
            .representative_terminal()
            .is_some_and(RepresentativeTerminalV1::is_unattributed);
        let deadline_missing_terminal = closure == RootClosureV1::AttributionDeadline
            && !self
                .representative_terminal()
                .is_some_and(RepresentativeTerminalV1::is_coherent_direct_result);

        let attribution = if closure == RootClosureV1::DeadProcessRecovery
            || representative_orphaned
            || scanned == RootExposureV1::Orphaned
        {
            RootExposureV1::Orphaned
        } else if closure == RootClosureV1::GracefulShutdown {
            RootExposureV1::ShutdownOrphaned
        } else if !self.label_complete()
            || representative_unattributed
            || deadline_missing_terminal
            || scanned == RootExposureV1::Unattributed
        {
            RootExposureV1::Unattributed
        } else if scanned == RootExposureV1::AmbiguousExposure
            || representative_dispatch_mismatch(self.representative_terminal(), exposure_facts)
        {
            RootExposureV1::AmbiguousExposure
        } else {
            scanned
        };
        RootOutcomeClassificationV1 {
            label: self.prefix_label(),
            label_complete: self.label_complete(),
            exposure: attribution,
            faults: self.faults.iter().copied().collect(),
            signal_count: self.signals.len(),
            canonical_signal_bytes: self.canonical_signal_bytes,
        }
    }

    fn commit_batch(&mut self, candidates: Vec<ProtectedOutcomeSignalV1>) -> OutcomeObservationV1 {
        let candidates = candidates
            .into_iter()
            .filter(|signal| !self.signal_ids.contains(&signal.signal_sha256))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return self.faults.first().copied().map_or(
                OutcomeObservationV1::Duplicate,
                OutcomeObservationV1::Incomplete,
            );
        }
        let Some(count) = self.signals.len().checked_add(candidates.len()) else {
            return self.fail(OutcomeCollectionFaultV1::SignalCountExceeded);
        };
        if count > OUTCOME_SIGNALS_MAX {
            return self.fail(OutcomeCollectionFaultV1::SignalCountExceeded);
        }
        let added_count = candidates.len();
        let mut combined = self.signals.clone();
        combined.extend(candidates.iter().cloned());
        combined.sort_by(|left, right| {
            left.ingest_seq
                .cmp(&right.ingest_seq)
                .then_with(|| left.signal_sha256.cmp(&right.signal_sha256))
        });
        let bytes = match canonical_signal_bytes(&combined) {
            Ok(bytes) => bytes,
            Err(()) => return self.fail(OutcomeCollectionFaultV1::Projection),
        };
        if bytes > OUTCOME_SIGNAL_FACTS_MAX_BYTES {
            return self.fail(OutcomeCollectionFaultV1::SignalBytesExceeded);
        }
        if let Some(fault) = self.faults.first().copied() {
            return OutcomeObservationV1::Incomplete(fault);
        }
        for signal in candidates {
            self.signal_ids.insert(signal.signal_sha256);
        }
        self.signals = combined;
        self.canonical_signal_bytes = bytes;
        OutcomeObservationV1::Added {
            signals: added_count,
        }
    }

    fn fail(&mut self, fault: OutcomeCollectionFaultV1) -> OutcomeObservationV1 {
        self.faults.insert(fault);
        OutcomeObservationV1::Incomplete(
            *self
                .faults
                .first()
                .expect("inserting a collection fault makes the set nonempty"),
        )
    }
}

fn representative_dispatch_mismatch(
    representative: Option<RepresentativeTerminalV1>,
    facts: &RootExposureFactsV1,
) -> bool {
    let Some(dispatch) = facts.dispatches.first() else {
        return false;
    };
    let Some(dispatch_terminal) = dispatch.terminals.first().copied() else {
        return false;
    };
    matches!(
        (representative, dispatch_terminal),
        (
            Some(RepresentativeTerminalV1::Completed),
            DispatchTerminalV1::ProviderError
        ) | (
            Some(RepresentativeTerminalV1::Error),
            DispatchTerminalV1::Completed
        )
    )
}

fn canonical_signal_bytes(signals: &[ProtectedOutcomeSignalV1]) -> Result<usize, ()> {
    canonical_serialize_bytes(&ProtectedSignalAggregateV1 {
        schema: OUTCOME_SIGNAL_AGGREGATE_SCHEMA_V1,
        signals,
    })
    .map(|bytes| bytes.len())
}

fn projected_event_kind(
    event: &CapturedTrajectoryEvent,
) -> Result<Option<OutcomeMatcherEventKind>, ()> {
    match (event.kind, event.scope_phase) {
        (CapturedEventKind::Scope, Some(ScopeCategory::End)) => {
            Ok(Some(OutcomeMatcherEventKind::ScopeEnd))
        }
        (CapturedEventKind::Scope, Some(ScopeCategory::Start)) => Ok(None),
        (CapturedEventKind::Mark, None) => Ok(Some(OutcomeMatcherEventKind::Mark)),
        (CapturedEventKind::Scope, None) | (CapturedEventKind::Mark, Some(_)) => Err(()),
    }
}

fn coarse_matcher_matches(
    matcher: &CompiledMatcherV1,
    event_kind: Option<OutcomeMatcherEventKind>,
    category: Option<&str>,
    name: &str,
) -> bool {
    event_kind == Some(matcher.event_kind)
        && category == Some(matcher.category.as_str())
        && name == matcher.name
}

fn project_terminal_status(metadata: Option<&Json>) -> Result<OutcomeTerminalStatus, ()> {
    let Some(metadata) = metadata else {
        return Ok(OutcomeTerminalStatus::Unset);
    };
    let object = metadata.as_object().ok_or(())?;
    match object.get("otel.status_code") {
        None => Ok(OutcomeTerminalStatus::Unset),
        Some(Json::String(value)) if value == "OK" => Ok(OutcomeTerminalStatus::Ok),
        Some(Json::String(value)) if value == "ERROR" => Ok(OutcomeTerminalStatus::Error),
        Some(_) => Err(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn event_signal_candidates(
    policy: &CompiledOutcomePolicyV1,
    event_kind: Option<OutcomeMatcherEventKind>,
    category: Option<&str>,
    name: &str,
    status: OutcomeTerminalStatus,
    metadata: Option<&serde_json::Map<String, Json>>,
    source_event_sha256: &str,
    ingest_seq: u64,
    tool_end: bool,
    exact_owner_end: bool,
) -> Result<Vec<ProtectedOutcomeSignalV1>, OutcomeCollectionFaultV1> {
    let mut candidates = Vec::new();
    append_matching_signals(
        &policy.success_matchers,
        OutcomeDisposition::Success,
        event_kind,
        category,
        name,
        status,
        metadata,
        source_event_sha256,
        ingest_seq,
        &mut candidates,
    )?;
    append_matching_signals(
        &policy.failure_matchers,
        OutcomeDisposition::Failure,
        event_kind,
        category,
        name,
        status,
        metadata,
        source_event_sha256,
        ingest_seq,
        &mut candidates,
    )?;
    if tool_end && status == OutcomeTerminalStatus::Error {
        candidates.push(protected_signal(
            Some(source_event_sha256.to_string()),
            ProtectedOutcomeSignalSourceV1::ToolFailure,
            policy.tool_failure_disposition,
            Some(status),
            ingest_seq,
        ));
    }
    if exact_owner_end {
        candidates.push(protected_signal(
            Some(source_event_sha256.to_string()),
            ProtectedOutcomeSignalSourceV1::EndOfRun,
            policy.end_of_run_disposition,
            Some(status),
            ingest_seq,
        ));
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
fn append_matching_signals(
    matchers: &[CompiledMatcherV1],
    disposition: OutcomeDisposition,
    event_kind: Option<OutcomeMatcherEventKind>,
    category: Option<&str>,
    name: &str,
    status: OutcomeTerminalStatus,
    metadata: Option<&serde_json::Map<String, Json>>,
    source_event_sha256: &str,
    ingest_seq: u64,
    signals: &mut Vec<ProtectedOutcomeSignalV1>,
) -> Result<(), OutcomeCollectionFaultV1> {
    for matcher in matchers {
        if !coarse_matcher_matches(matcher, event_kind, category, name)
            || matcher.terminal_status != status
        {
            continue;
        }
        let Some(observed) = match_metadata(&matcher.metadata_equals, metadata)? else {
            continue;
        };
        signals.push(protected_signal(
            Some(source_event_sha256.to_string()),
            ProtectedOutcomeSignalSourceV1::Matcher {
                matcher_sha256: matcher.matcher_hash.clone(),
                observed,
            },
            disposition,
            Some(status),
            ingest_seq,
        ));
    }
    Ok(())
}

fn match_metadata(
    clauses: &[CompiledMetadataClauseV1],
    metadata: Option<&serde_json::Map<String, Json>>,
) -> Result<Option<Vec<ProtectedObservedClauseV1>>, OutcomeCollectionFaultV1> {
    let mut protected = Vec::with_capacity(clauses.len());
    for clause in clauses {
        let Some(observed) = metadata.and_then(|metadata| metadata.get(&clause.key)) else {
            return Ok(None);
        };
        let Ok(canonical) = canonical_outcome_scalar_v1(observed) else {
            return Ok(None);
        };
        if canonical != clause.canonical_expected {
            return Ok(None);
        }
        protected.push(ProtectedObservedClauseV1 {
            key_sha256: outcome_metadata_key_hash_v1(&clause.key)
                .map_err(|_| OutcomeCollectionFaultV1::Projection)?,
            observed_scalar_sha256: outcome_observed_scalar_hash_v1(observed)
                .map_err(|_| OutcomeCollectionFaultV1::Projection)?,
        });
    }
    Ok(Some(protected))
}

fn protected_signal(
    source_event_sha256: Option<String>,
    source: ProtectedOutcomeSignalSourceV1,
    disposition: OutcomeDisposition,
    terminal_status: Option<OutcomeTerminalStatus>,
    ingest_seq: u64,
) -> ProtectedOutcomeSignalV1 {
    #[derive(Serialize)]
    struct SignalIdentity<'a> {
        domain: &'static str,
        source_event_sha256: &'a Option<String>,
        source: &'a ProtectedOutcomeSignalSourceV1,
        disposition: OutcomeDisposition,
        terminal_status: Option<OutcomeTerminalStatus>,
        ingest_seq: u64,
    }

    let identity = SignalIdentity {
        domain: std::str::from_utf8(OUTCOME_SIGNAL_KEY_DOMAIN_V1)
            .expect("outcome signal identity domain is ASCII"),
        source_event_sha256: &source_event_sha256,
        source: &source,
        disposition,
        terminal_status,
        ingest_seq,
    };
    let signal_sha256 = canonical_serialize_bytes(&identity)
        .map(|bytes| sha256_hex(&bytes))
        .expect("protected outcome signal identity is canonical");
    ProtectedOutcomeSignalV1 {
        signal_sha256,
        source_event_sha256,
        source,
        disposition,
        terminal_status,
        ingest_seq,
    }
}

fn protected_event_key_sha256(key: &EventBatchKeyV1) -> String {
    let mut preimage =
        Vec::with_capacity(OUTCOME_EVENT_KEY_DOMAIN_V1.len() + 16 + 4 + 5 + 4 + 5 + 32);
    preimage.extend_from_slice(OUTCOME_EVENT_KEY_DOMAIN_V1);
    preimage.extend_from_slice(key.event_uuid.as_bytes());
    preimage.extend_from_slice(&(key.kind.len() as u32).to_be_bytes());
    preimage.extend_from_slice(key.kind.as_bytes());
    let phase = key.scope_phase.unwrap_or("mark").as_bytes();
    preimage.extend_from_slice(&(phase.len() as u32).to_be_bytes());
    preimage.extend_from_slice(phase);
    preimage.extend_from_slice(
        &hex_sha256_bytes(&key.canonical_payload_hash)
            .expect("validated canonical payload hash is lowercase SHA-256"),
    );
    sha256_hex(&preimage)
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hex_sha256_bytes(value: &str) -> Option<[u8; 32]> {
    if !is_lower_hex_sha256(value) {
        return None;
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(chunk[0])? << 4) | hex_nibble(chunk[1])?;
    }
    Some(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Final independently reduced Bernoulli label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RootOutcomeLabelV1 {
    Success,
    Failure,
    Ignore,
}

pub(crate) fn reduce_signal_label_v1(signals: &[ProtectedOutcomeSignalV1]) -> RootOutcomeLabelV1 {
    if signals
        .iter()
        .any(|signal| signal.disposition == OutcomeDisposition::Failure)
    {
        RootOutcomeLabelV1::Failure
    } else if signals
        .iter()
        .any(|signal| signal.disposition == OutcomeDisposition::Success)
    {
        RootOutcomeLabelV1::Success
    } else {
        RootOutcomeLabelV1::Ignore
    }
}

/// Result of a pure process-local open-root budget admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenRootAdmissionV1 {
    Opened,
    AlreadyOpen,
    AtCapacity,
}

/// Exact process-local budget for root windows awaiting terminalization.
#[derive(Default)]
pub(crate) struct OpenRootWindowBudgetV1 {
    roots: HashSet<RootKey>,
}

impl OpenRootWindowBudgetV1 {
    pub(crate) fn try_open(&mut self, root_key: RootKey) -> OpenRootAdmissionV1 {
        if self.roots.contains(&root_key) {
            return OpenRootAdmissionV1::AlreadyOpen;
        }
        if self.roots.len() >= OUTCOME_OPEN_ROOT_WINDOWS_MAX {
            return OpenRootAdmissionV1::AtCapacity;
        }
        self.roots.insert(root_key);
        OpenRootAdmissionV1::Opened
    }

    pub(crate) fn close(&mut self, root_key: RootKey) -> bool {
        self.roots.remove(&root_key)
    }

    pub(crate) fn len(&self) -> usize {
        self.roots.len()
    }
}

/// Exact randomized arm persisted for one root assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootAssignmentArmV1 {
    ActiveCanary,
    AnchorControl,
    AnchorHoldout,
}

/// Fixed-width protected identity accepted by terminal rescan DTOs.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ProtectedIdentityV1([u8; 32]);

impl ProtectedIdentityV1 {
    pub(crate) fn from_sha256_hex(value: &str) -> Result<Self, ()> {
        hex_sha256_bytes(value).map(Self).ok_or(())
    }

    pub(crate) fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }
}

impl fmt::Debug for ProtectedIdentityV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ProtectedIdentityV1")
            .field(&self.to_hex())
            .finish()
    }
}

impl Serialize for ProtectedIdentityV1 {
    fn serialize<SerializerT>(
        &self,
        serializer: SerializerT,
    ) -> Result<SerializerT::Ok, SerializerT::Error>
    where
        SerializerT: serde::Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

/// One protected assignment/window relation from the terminal rescan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RootAssignmentFactV1 {
    pub(crate) window_sha256: ProtectedIdentityV1,
    pub(crate) pool_sha256: ProtectedIdentityV1,
    pub(crate) experiment_sha256: ProtectedIdentityV1,
    pub(crate) interval_sha256: ProtectedIdentityV1,
    pub(crate) assigned_candidate_sha256: ProtectedIdentityV1,
    pub(crate) arm: RootAssignmentArmV1,
}

/// Terminal fact attached to one candidate dispatch admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DispatchTerminalV1 {
    Completed,
    ProviderError,
    CancelledBeforeHandoff,
    CancelledAfterHandoff,
    PanickedAfterHandoff,
    AbortedBeforeHandoff,
    UnknownAfterCrash,
}

/// One protected dispatch admission and all terminal facts found during rescan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DispatchAdmissionFactV1 {
    pub(crate) admission_sha256: ProtectedIdentityV1,
    pub(crate) window_sha256: ProtectedIdentityV1,
    pub(crate) pool_sha256: ProtectedIdentityV1,
    pub(crate) experiment_sha256: ProtectedIdentityV1,
    pub(crate) interval_sha256: ProtectedIdentityV1,
    pub(crate) candidate_sha256: ProtectedIdentityV1,
    pub(crate) terminals: Vec<DispatchTerminalV1>,
}

/// Complete storage-independent fact set rescanned at terminalization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RootExposureFactsV1 {
    pub(crate) window_fact_count: usize,
    pub(crate) assignments: Vec<RootAssignmentFactV1>,
    pub(crate) dispatches: Vec<DispatchAdmissionFactV1>,
    pub(crate) conflicting_link: bool,
    pub(crate) overlapping_interval: bool,
}

/// Exposure/attribution result kept separate from the reduced label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum RootExposureV1 {
    Treatment {
        candidate_sha256: ProtectedIdentityV1,
    },
    AnchorControl,
    AnchorHoldout,
    AmbiguousExposure,
    Unattributed,
    Orphaned,
    ShutdownOrphaned,
}

/// Pure terminal rescan of assignment, window, and dispatch facts.
pub(crate) fn scan_root_exposure_v1(facts: &RootExposureFactsV1) -> RootExposureV1 {
    let has_unknown_crash = facts.dispatches.iter().any(|dispatch| {
        dispatch
            .terminals
            .contains(&DispatchTerminalV1::UnknownAfterCrash)
    });
    if has_unknown_crash {
        return RootExposureV1::Orphaned;
    }

    let missing_treatment_admission = facts
        .assignments
        .iter()
        .any(|assignment| assignment.arm == RootAssignmentArmV1::ActiveCanary)
        && facts.dispatches.is_empty();
    let has_missing_or_unattributed_terminal = facts.dispatches.iter().any(|dispatch| {
        dispatch.terminals.is_empty()
            || dispatch.terminals.iter().any(|terminal| {
                matches!(
                    terminal,
                    DispatchTerminalV1::CancelledBeforeHandoff
                        | DispatchTerminalV1::CancelledAfterHandoff
                        | DispatchTerminalV1::PanickedAfterHandoff
                        | DispatchTerminalV1::AbortedBeforeHandoff
                )
            })
    });
    if facts.window_fact_count == 0
        || facts.assignments.is_empty()
        || missing_treatment_admission
        || has_missing_or_unattributed_terminal
    {
        return RootExposureV1::Unattributed;
    }

    let structural_conflict = (facts.window_fact_count != 1 && facts.window_fact_count != 0)
        || facts.assignments.len() > 1
        || facts.conflicting_link
        || facts.overlapping_interval
        || facts.dispatches.len() > 1
        || facts
            .dispatches
            .iter()
            .any(|dispatch| dispatch.terminals.len() > 1);
    if structural_conflict {
        return RootExposureV1::AmbiguousExposure;
    }

    let assignment = &facts.assignments[0];
    if facts.dispatches.is_empty() {
        return match assignment.arm {
            RootAssignmentArmV1::AnchorControl => RootExposureV1::AnchorControl,
            RootAssignmentArmV1::AnchorHoldout => RootExposureV1::AnchorHoldout,
            RootAssignmentArmV1::ActiveCanary => RootExposureV1::Unattributed,
        };
    }
    let dispatch = &facts.dispatches[0];
    let relation_matches = dispatch.window_sha256 == assignment.window_sha256
        && dispatch.pool_sha256 == assignment.pool_sha256
        && dispatch.experiment_sha256 == assignment.experiment_sha256
        && dispatch.interval_sha256 == assignment.interval_sha256
        && dispatch.candidate_sha256 == assignment.assigned_candidate_sha256;
    if !relation_matches || assignment.arm != RootAssignmentArmV1::ActiveCanary {
        return RootExposureV1::AmbiguousExposure;
    }
    let Some(terminal) = dispatch.terminals.first().copied() else {
        return RootExposureV1::Unattributed;
    };
    match terminal {
        DispatchTerminalV1::Completed | DispatchTerminalV1::ProviderError => {
            RootExposureV1::Treatment {
                candidate_sha256: dispatch.candidate_sha256,
            }
        }
        DispatchTerminalV1::UnknownAfterCrash => RootExposureV1::Orphaned,
        DispatchTerminalV1::CancelledBeforeHandoff
        | DispatchTerminalV1::CancelledAfterHandoff
        | DispatchTerminalV1::PanickedAfterHandoff
        | DispatchTerminalV1::AbortedBeforeHandoff => RootExposureV1::Unattributed,
    }
}

/// How the bounded attribution interval was closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootClosureV1 {
    PinnedOwnerEnd,
    AttributionDeadline,
    GracefulShutdown,
    DeadProcessRecovery,
}

/// Complete pure classifier output for one terminal root.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct RootOutcomeClassificationV1 {
    pub(crate) label: RootOutcomeLabelV1,
    pub(crate) label_complete: bool,
    pub(crate) exposure: RootExposureV1,
    pub(crate) faults: Vec<OutcomeCollectionFaultV1>,
    pub(crate) signal_count: usize,
    pub(crate) canonical_signal_bytes: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    };
    use nemo_relay::api::llm::LlmCallRole;
    use serde_json::{Map, Value as Json, json};
    use static_assertions::assert_not_impl_any;
    use uuid::Uuid;

    use super::*;
    use crate::trajectory::{
        EventProjectionLimits, ProjectedTrajectoryEvent, project_captured_event,
    };

    const SUCCESS_EVENT_UUID: Uuid = Uuid::from_u128(0x018f_1234_5678_7000_8000_0000_0000_0001);
    const SECOND_EVENT_UUID: Uuid = Uuid::from_u128(0x018f_1234_5678_7000_8000_0000_0000_0002);

    fn matcher(
        event_kind: OutcomeMatcherEventKind,
        category: &str,
        name: &str,
        terminal_status: OutcomeTerminalStatus,
        metadata_equals: BTreeMap<String, Json>,
    ) -> OutcomeMatcher {
        OutcomeMatcher {
            event_kind,
            category: category.to_string(),
            name: name.to_string(),
            terminal_status,
            metadata_equals,
        }
    }

    fn mark_matcher(name: &str) -> OutcomeMatcher {
        matcher(
            OutcomeMatcherEventKind::Mark,
            "custom",
            name,
            OutcomeTerminalStatus::Unset,
            BTreeMap::new(),
        )
    }

    fn complete_outcome(
        success_matchers: Vec<OutcomeMatcher>,
        failure_matchers: Vec<OutcomeMatcher>,
    ) -> OutcomeConfig {
        OutcomeConfig {
            version: 1,
            success_matchers,
            failure_matchers,
            completion_disposition: OutcomeDisposition::Success,
            error_disposition: OutcomeDisposition::Failure,
            tool_failure_disposition: OutcomeDisposition::Failure,
            end_of_run_disposition: OutcomeDisposition::Ignore,
            max_attribution_seconds: 60,
            actual_outcome_half_life_seconds: 600,
            anchor_shadow_half_life_seconds: 600,
            relearning_cooloff_seconds: 60,
            min_treatment_roots: 32,
            min_control_roots: 32,
            min_treatment_effective_weight: 16.0,
            min_control_effective_weight: 16.0,
            noninferiority_margin: 0.05,
            noninferiority_probability: 0.995,
            rollback_probability: 0.99,
            outcome_evaluation_batch_size: 64,
            max_canary_roots: 64,
            authorization_ttl_seconds: 60,
        }
    }

    fn compiled_policy() -> Arc<CompiledOutcomePolicyV1> {
        Arc::new(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![mark_matcher("accepted")],
                vec![mark_matcher("rejected")],
            ))
            .unwrap(),
        )
    }

    fn mark_event(
        uuid: Uuid,
        name: &str,
        metadata: Option<Json>,
        data: Option<Json>,
    ) -> CapturedTrajectoryEvent {
        let mut base = BaseEvent::builder().uuid(uuid).name(name).build();
        base.metadata = metadata;
        base.data = data;
        let event = Event::Mark(MarkEvent::new(base, Some(EventCategory::custom()), None));
        captured_event(10, &event)
    }

    fn scope_end_event(
        uuid: Uuid,
        category: &str,
        name: &str,
        metadata: Option<Json>,
        profile: Option<CategoryProfile>,
    ) -> CapturedTrajectoryEvent {
        let mut base = BaseEvent::builder().uuid(uuid).name(name).build();
        base.metadata = metadata;
        let event = Event::Scope(ScopeEvent::new(
            base,
            ScopeCategory::End,
            Vec::new(),
            EventCategory::new(category),
            profile,
        ));
        captured_event(10, &event)
    }

    fn captured_event(ingest_seq: u64, event: &Event) -> CapturedTrajectoryEvent {
        match project_captured_event(
            ingest_seq,
            event,
            EventProjectionLimits::for_window_bytes(1024 * 1024),
        ) {
            ProjectedTrajectoryEvent::Captured(event) => Arc::unwrap_or_clone(event),
            ProjectedTrajectoryEvent::Oversized(_) => panic!("test event must fit projection"),
        }
    }

    fn context(
        _event: &CapturedTrajectoryEvent,
        _ingest_seq: u64,
        owner_relation: PinnedOwnerRelationV1,
        exact_pinned_owner_end: bool,
    ) -> OutcomeEventContextV1 {
        OutcomeEventContextV1 {
            owner_relation,
            exact_pinned_owner_end,
        }
    }

    fn external_context(event: &CapturedTrajectoryEvent, ingest_seq: u64) -> OutcomeEventContextV1 {
        context(
            event,
            ingest_seq,
            PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree,
            false,
        )
    }

    fn metadata(status: Option<Json>, pairs: &[(&str, Json)]) -> Option<Json> {
        let mut object = Map::new();
        if let Some(status) = status {
            object.insert("otel.status_code".to_string(), status);
        }
        for (key, value) in pairs {
            object.insert((*key).to_string(), value.clone());
        }
        Some(Json::Object(object))
    }

    fn protected(byte: u8) -> ProtectedIdentityV1 {
        ProtectedIdentityV1([byte; 32])
    }

    fn assignment(arm: RootAssignmentArmV1) -> RootAssignmentFactV1 {
        RootAssignmentFactV1 {
            window_sha256: protected(0x11),
            pool_sha256: protected(0x22),
            experiment_sha256: protected(0x33),
            interval_sha256: protected(0x44),
            assigned_candidate_sha256: protected(0x66),
            arm,
        }
    }

    fn dispatch(terminal: Option<DispatchTerminalV1>) -> DispatchAdmissionFactV1 {
        let assignment = assignment(RootAssignmentArmV1::ActiveCanary);
        DispatchAdmissionFactV1 {
            admission_sha256: protected(0x55),
            window_sha256: assignment.window_sha256,
            pool_sha256: assignment.pool_sha256,
            experiment_sha256: assignment.experiment_sha256,
            interval_sha256: assignment.interval_sha256,
            candidate_sha256: protected(0x66),
            terminals: terminal.into_iter().collect(),
        }
    }

    fn exposure_facts(
        arm: RootAssignmentArmV1,
        terminal: Option<DispatchTerminalV1>,
    ) -> RootExposureFactsV1 {
        RootExposureFactsV1 {
            window_fact_count: 1,
            assignments: vec![assignment(arm)],
            dispatches: if arm == RootAssignmentArmV1::ActiveCanary || terminal.is_some() {
                vec![dispatch(terminal)]
            } else {
                Vec::new()
            },
            conflicting_link: false,
            overlapping_interval: false,
        }
    }

    #[test]
    fn matcher_is_exact_conjunctive_and_numeric_equality_is_rfc8785() {
        let matcher = matcher(
            OutcomeMatcherEventKind::ScopeEnd,
            "function",
            "finished",
            OutcomeTerminalStatus::Ok,
            BTreeMap::from([
                ("outcome.success".to_string(), json!(true)),
                ("result".to_string(), json!(1)),
            ]),
        );
        let policy = Arc::new(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![matcher],
                vec![mark_matcher("failure")],
            ))
            .unwrap(),
        );
        let matching = scope_end_event(
            SUCCESS_EVENT_UUID,
            "function",
            "finished",
            metadata(
                Some(json!("OK")),
                &[("result", json!(1.0)), ("outcome.success", json!(true))],
            ),
            None,
        );
        let mut accumulator = OutcomeSignalAccumulatorV1::new(policy.clone(), 9);
        assert_eq!(
            accumulator.observe_event(&matching, &external_context(&matching, 10)),
            OutcomeObservationV1::Added { signals: 1 }
        );
        assert_eq!(accumulator.prefix_label(), RootOutcomeLabelV1::Success);

        for event in [
            scope_end_event(
                SECOND_EVENT_UUID,
                "custom",
                "finished",
                metadata(
                    Some(json!("OK")),
                    &[("result", json!(1)), ("outcome.success", json!(true))],
                ),
                None,
            ),
            scope_end_event(
                SECOND_EVENT_UUID,
                "function",
                "other",
                metadata(
                    Some(json!("OK")),
                    &[("result", json!(1)), ("outcome.success", json!(true))],
                ),
                None,
            ),
            scope_end_event(
                SECOND_EVENT_UUID,
                "function",
                "finished",
                metadata(
                    Some(json!("ERROR")),
                    &[("result", json!(1)), ("outcome.success", json!(true))],
                ),
                None,
            ),
            scope_end_event(
                SECOND_EVENT_UUID,
                "function",
                "finished",
                metadata(
                    Some(json!("OK")),
                    &[("result", json!(2)), ("outcome.success", json!(true))],
                ),
                None,
            ),
        ] {
            let mut candidate = OutcomeSignalAccumulatorV1::new(policy.clone(), 0);
            assert_eq!(
                candidate.observe_event(&event, &external_context(&event, 1)),
                OutcomeObservationV1::Ignored
            );
        }
    }

    #[test]
    fn compile_rechecks_version_allowlist_and_duplicate_identity_before_runtime_use() {
        let mut invalid_version = complete_outcome(
            vec![mark_matcher("accepted")],
            vec![mark_matcher("rejected")],
        );
        invalid_version.version = 2;
        assert_eq!(
            CompiledOutcomePolicyV1::compile(&invalid_version).unwrap_err(),
            OutcomeCompileErrorV1::InvalidPolicy("outcome version")
        );

        let mut unknown_key = mark_matcher("accepted");
        unknown_key
            .metadata_equals
            .insert("authorization".to_string(), json!("secret"));
        assert_eq!(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![unknown_key],
                vec![mark_matcher("rejected")],
            ))
            .unwrap_err(),
            OutcomeCompileErrorV1::InvalidPolicy("metadata key")
        );

        let duplicate = mark_matcher("same");
        assert_eq!(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![duplicate.clone()],
                vec![duplicate],
            ))
            .unwrap_err(),
            OutcomeCompileErrorV1::DuplicateMatcher
        );
    }

    #[test]
    fn unsafe_observed_values_are_nonmatches_and_are_never_hashed_or_retained() {
        let policy = Arc::new(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![matcher(
                    OutcomeMatcherEventKind::Mark,
                    "custom",
                    "accepted",
                    OutcomeTerminalStatus::Unset,
                    BTreeMap::from([("outcome.label".to_string(), json!("ok"))]),
                )],
                vec![mark_matcher("rejected")],
            ))
            .unwrap(),
        );
        for observed in [json!({"nested": "ok"}), json!("Bearer-secret"), json!(-0.0)] {
            let event = mark_event(
                SUCCESS_EVENT_UUID,
                "accepted",
                metadata(None, &[("outcome.label", observed)]),
                None,
            );
            let mut accumulator = OutcomeSignalAccumulatorV1::new(policy.clone(), 0);
            assert_eq!(
                accumulator.observe_event(&event, &external_context(&event, 1)),
                OutcomeObservationV1::Ignored
            );
            assert!(accumulator.label_complete());
            assert!(accumulator.signals().is_empty());
        }
    }

    #[test]
    fn opening_sequence_excludes_the_complete_pre_treatment_prefix() {
        let mut before = mark_event(SUCCESS_EVENT_UUID, "accepted", None, None);
        before.ingest_seq = 50;
        let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 50);
        assert_eq!(
            accumulator.observe_event(&before, &external_context(&before, 50)),
            OutcomeObservationV1::Ignored
        );
        let mut after = before.clone();
        after.ingest_seq = 51;
        assert_eq!(
            accumulator.observe_event(&after, &external_context(&after, 51)),
            OutcomeObservationV1::Added { signals: 1 }
        );
    }

    #[test]
    fn missing_status_is_unset_and_malformed_status_only_faults_relevant_external_events() {
        let event = mark_event(SUCCESS_EVENT_UUID, "accepted", None, None);
        let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            accumulator.observe_event(&event, &external_context(&event, 1)),
            OutcomeObservationV1::Added { signals: 1 }
        );

        let malformed = mark_event(
            SECOND_EVENT_UUID,
            "accepted",
            metadata(Some(json!("UNSET")), &[]),
            None,
        );
        let mut relevant = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            relevant.observe_event(&malformed, &external_context(&malformed, 1)),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::Projection)
        );

        let irrelevant = mark_event(
            SECOND_EVENT_UUID,
            "unmatched",
            metadata(Some(json!({"secret": true})), &[]),
            Some(json!({"secret": "never-read"})),
        );
        let mut ignored = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            ignored.observe_event(&irrelevant, &external_context(&irrelevant, 1)),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::Projection)
        );

        let mut internal = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let internal_context = context(
            &malformed,
            1,
            PinnedOwnerRelationV1::InternalPinnedOwnerSubtree,
            false,
        );
        assert_eq!(
            internal.observe_event(&malformed, &internal_context),
            OutcomeObservationV1::Ignored
        );
        assert!(internal.label_complete());
    }

    #[test]
    fn shadow_and_judge_llm_roles_are_excluded_before_status_projection() {
        for role in [LlmCallRole::Shadow, LlmCallRole::Judge] {
            let mut profile = CategoryProfile::default();
            profile.set_llm_call_role(role);
            let event = scope_end_event(
                SUCCESS_EVENT_UUID,
                "llm",
                "provider",
                metadata(Some(json!("INVALID")), &[]),
                Some(profile),
            );
            let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
            assert_eq!(
                accumulator.observe_event(&event, &external_context(&event, 1)),
                OutcomeObservationV1::Ignored
            );
            assert!(accumulator.label_complete());
        }
    }

    #[test]
    fn relation_context_distinguishes_nested_descendants_siblings_and_exact_owner_end() {
        let event = mark_event(SUCCESS_EVENT_UUID, "accepted", None, None);
        let mut descendant = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert!(matches!(
            descendant.observe_event(&event, &external_context(&event, 1)),
            OutcomeObservationV1::Added { .. }
        ));

        let mut outside = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let outside_context = context(&event, 1, PinnedOwnerRelationV1::Outside, false);
        assert_eq!(
            outside.observe_event(&event, &outside_context),
            OutcomeObservationV1::Ignored
        );

        let mut unreadable = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let unreadable_context = context(&event, 1, PinnedOwnerRelationV1::Unreadable, false);
        assert_eq!(
            unreadable.observe_event(&event, &unreadable_context),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::UnreadableRootRelation)
        );

        let owner_end = scope_end_event(
            SECOND_EVENT_UUID,
            "agent",
            "owner",
            metadata(Some(json!("OK")), &[]),
            None,
        );
        let mut child_owner_end = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            child_owner_end.observe_event(&owner_end, &external_context(&owner_end, 1)),
            OutcomeObservationV1::Ignored
        );
        let mut exact_owner_end = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let exact_context = context(
            &owner_end,
            1,
            PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree,
            true,
        );
        assert_eq!(
            exact_owner_end.observe_event(&owner_end, &exact_context),
            OutcomeObservationV1::Added { signals: 1 }
        );

        let mut contradiction = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let contradiction_context = context(&owner_end, 1, PinnedOwnerRelationV1::Outside, true);
        assert_eq!(
            contradiction.observe_event(&owner_end, &contradiction_context),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::UnreadableRootRelation)
        );

        let mut mark_as_owner_end = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let mark_context = context(
            &event,
            1,
            PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree,
            true,
        );
        assert_eq!(
            mark_as_owner_end.observe_event(&event, &mark_context),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::UnreadableRootRelation)
        );

        let mut malformed_kind = event.clone();
        malformed_kind.scope_phase = Some(ScopeCategory::End);
        let mut malformed = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            malformed.observe_event(&malformed_kind, &external_context(&malformed_kind, 1)),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::Projection)
        );
    }

    #[test]
    fn one_event_emits_each_matcher_and_builtin_then_replays_as_one_atomic_batch() {
        let common = matcher(
            OutcomeMatcherEventKind::ScopeEnd,
            "tool",
            "lookup",
            OutcomeTerminalStatus::Error,
            BTreeMap::new(),
        );
        let with_status = matcher(
            OutcomeMatcherEventKind::ScopeEnd,
            "tool",
            "lookup",
            OutcomeTerminalStatus::Error,
            BTreeMap::from([("status".to_string(), json!("failed"))]),
        );
        let mut config = complete_outcome(vec![common, with_status], vec![mark_matcher("nope")]);
        config.end_of_run_disposition = OutcomeDisposition::Success;
        let policy = Arc::new(CompiledOutcomePolicyV1::compile(&config).unwrap());
        let event = scope_end_event(
            SUCCESS_EVENT_UUID,
            "tool",
            "lookup",
            metadata(Some(json!("ERROR")), &[("status", json!("failed"))]),
            None,
        );
        let event_context = context(
            &event,
            1,
            PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree,
            true,
        );
        let mut accumulator = OutcomeSignalAccumulatorV1::new(policy, 0);
        assert_eq!(
            accumulator.observe_event(&event, &event_context),
            OutcomeObservationV1::Added { signals: 4 }
        );
        assert_eq!(accumulator.signals().len(), 4);
        assert_eq!(accumulator.prefix_label(), RootOutcomeLabelV1::Failure);
        assert_eq!(
            accumulator.observe_event(&event, &event_context),
            OutcomeObservationV1::Duplicate
        );
        assert_eq!(accumulator.signals().len(), 4);

        let mut changed_payload = event.clone();
        changed_payload.canonical_payload_hash = sha256_hex(b"different canonical payload");
        assert_eq!(
            accumulator.observe_event(&changed_payload, &event_context),
            OutcomeObservationV1::Added { signals: 4 }
        );
        assert_eq!(accumulator.signals().len(), 8);
    }

    #[test]
    fn tool_failure_requires_typed_tool_scope_and_rejects_category_spoofing() {
        let mut event = scope_end_event(
            SUCCESS_EVENT_UUID,
            "tool",
            "spoofed",
            metadata(Some(json!("ERROR")), &[]),
            None,
        );
        event.scope_type = Some(ScopeType::Function);
        let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            accumulator.observe_event(&event, &external_context(&event, 1)),
            OutcomeObservationV1::Ignored
        );
        assert!(accumulator.signals().is_empty());
    }

    #[test]
    fn representative_and_policy_builtins_are_idempotent_and_failure_dominant() {
        let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        let completed_with_policy_failure = RepresentativeResultV1 {
            terminal: RepresentativeTerminalV1::Completed,
            codec_policy_failure: true,
        };
        assert_eq!(
            accumulator.record_representative_result(completed_with_policy_failure, 4),
            OutcomeObservationV1::Added { signals: 2 }
        );
        assert_eq!(
            accumulator.record_representative_result(completed_with_policy_failure, 4),
            OutcomeObservationV1::Duplicate
        );
        assert_eq!(accumulator.prefix_label(), RootOutcomeLabelV1::Failure);

        assert_eq!(
            accumulator.record_representative_result(
                RepresentativeResultV1 {
                    terminal: RepresentativeTerminalV1::Error,
                    codec_policy_failure: false,
                },
                7,
            ),
            OutcomeObservationV1::Incomplete(
                OutcomeCollectionFaultV1::ConflictingRepresentativeTerminal
            )
        );
        assert!(!accumulator.label_complete());
        assert_eq!(accumulator.signals().len(), 2);

        let mut provider_error = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        assert_eq!(
            provider_error.record_representative_result(
                RepresentativeResultV1 {
                    terminal: RepresentativeTerminalV1::Error,
                    codec_policy_failure: false,
                },
                4,
            ),
            OutcomeObservationV1::Added { signals: 1 }
        );
        assert_eq!(provider_error.prefix_label(), RootOutcomeLabelV1::Failure);
    }

    #[test]
    fn event_batch_count_failure_is_atomic_and_retains_an_incomplete_prefix() {
        const KEYS: [&str; 7] = [
            "error.type",
            "outcome",
            "outcome.label",
            "outcome.success",
            "result",
            "status",
            "success",
        ];
        let mut matchers = Vec::new();
        for mask in 0_u8..64 {
            let metadata_equals = KEYS
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(_, key)| ((*key).to_string(), json!("ok")))
                .collect();
            matchers.push(matcher(
                OutcomeMatcherEventKind::ScopeEnd,
                "tool",
                "bounded",
                OutcomeTerminalStatus::Error,
                metadata_equals,
            ));
        }
        let policy = Arc::new(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                matchers,
                vec![mark_matcher("failure")],
            ))
            .unwrap(),
        );
        let event = scope_end_event(
            SUCCESS_EVENT_UUID,
            "tool",
            "bounded",
            metadata(Some(json!("ERROR")), &KEYS.map(|key| (key, json!("ok")))),
            None,
        );
        let mut accumulator = OutcomeSignalAccumulatorV1::new(policy, 0);
        assert!(matches!(
            accumulator.record_representative_result(
                RepresentativeResultV1 {
                    terminal: RepresentativeTerminalV1::Completed,
                    codec_policy_failure: false,
                },
                1,
            ),
            OutcomeObservationV1::Added { .. }
        ));
        assert_eq!(
            accumulator.observe_event(&event, &external_context(&event, 2)),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::SignalCountExceeded)
        );
        assert_eq!(accumulator.signals().len(), 1);
        assert_eq!(accumulator.prefix_label(), RootOutcomeLabelV1::Success);
        assert!(!accumulator.label_complete());
        let later = mark_event(SECOND_EVENT_UUID, "failure", None, None);
        assert!(matches!(
            accumulator.observe_event(&later, &external_context(&later, 3)),
            OutcomeObservationV1::Incomplete(_)
        ));
        assert_eq!(accumulator.signals().len(), 1);
    }

    #[test]
    fn exactly_64_signals_fit_and_the_65th_preserves_the_prefix() {
        let policy = compiled_policy();
        let mut accumulator = OutcomeSignalAccumulatorV1::new(policy, 0);
        for index in 0_u128..64 {
            let event = mark_event(Uuid::from_u128(index + 1), "accepted", None, None);
            assert_eq!(
                accumulator.observe_event(&event, &external_context(&event, index as u64 + 1)),
                OutcomeObservationV1::Added { signals: 1 }
            );
        }
        assert_eq!(accumulator.signals().len(), OUTCOME_SIGNALS_MAX);
        assert!(accumulator.canonical_signal_bytes() <= OUTCOME_SIGNAL_FACTS_MAX_BYTES);
        let overflow = mark_event(Uuid::from_u128(100), "accepted", None, None);
        assert_eq!(
            accumulator.observe_event(&overflow, &external_context(&overflow, 100)),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::SignalCountExceeded)
        );
        assert_eq!(accumulator.signals().len(), OUTCOME_SIGNALS_MAX);
    }

    #[test]
    fn completion_and_codec_failure_batch_never_partially_crosses_the_count_bound() {
        let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        for index in 0_u128..63 {
            let event = mark_event(Uuid::from_u128(index + 1), "accepted", None, None);
            assert!(matches!(
                accumulator.observe_event(&event, &external_context(&event, 10)),
                OutcomeObservationV1::Added { .. }
            ));
        }
        let before = canonical_serialize_bytes(&accumulator.signals()).unwrap();
        assert_eq!(
            accumulator.record_representative_result(
                RepresentativeResultV1 {
                    terminal: RepresentativeTerminalV1::Completed,
                    codec_policy_failure: true,
                },
                11,
            ),
            OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::SignalCountExceeded)
        );
        assert_eq!(accumulator.signals().len(), 63);
        assert_eq!(
            canonical_serialize_bytes(&accumulator.signals()).unwrap(),
            before
        );
        assert_eq!(accumulator.prefix_label(), RootOutcomeLabelV1::Success);
        assert!(!accumulator.label_complete());
    }

    #[test]
    fn first_fault_freezes_signal_prefix_while_explicit_causes_sort_independently() {
        fn build(order: [OutcomeCollectionFaultV1; 2]) -> OutcomeSignalAccumulatorV1 {
            let event = mark_event(SUCCESS_EVENT_UUID, "accepted", None, None);
            let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
            assert!(matches!(
                accumulator.observe_event(&event, &external_context(&event, 10)),
                OutcomeObservationV1::Added { .. }
            ));
            for fault in order {
                accumulator.mark_fault(fault);
            }
            accumulator
        }

        let mut left = build([
            OutcomeCollectionFaultV1::OversizedEvent,
            OutcomeCollectionFaultV1::EventLoss,
        ]);
        let right = build([
            OutcomeCollectionFaultV1::EventLoss,
            OutcomeCollectionFaultV1::OversizedEvent,
        ]);
        assert_eq!(left.faults(), right.faults());
        assert_eq!(
            left.faults().iter().copied().collect::<Vec<_>>(),
            vec![
                OutcomeCollectionFaultV1::EventLoss,
                OutcomeCollectionFaultV1::OversizedEvent,
            ]
        );
        let before = canonical_serialize_bytes(&left.signals()).unwrap();
        let before_bytes = left.canonical_signal_bytes();
        let before_ids = left
            .signals()
            .iter()
            .map(ProtectedOutcomeSignalV1::signal_sha256)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let failure = mark_event(SECOND_EVENT_UUID, "rejected", None, None);
        assert!(matches!(
            left.observe_event(&failure, &external_context(&failure, 11)),
            OutcomeObservationV1::Incomplete(_)
        ));
        assert_eq!(canonical_serialize_bytes(&left.signals()).unwrap(), before);
        assert_eq!(left.canonical_signal_bytes(), before_bytes);
        assert_eq!(
            left.signals()
                .iter()
                .map(ProtectedOutcomeSignalV1::signal_sha256)
                .map(str::to_string)
                .collect::<Vec<_>>(),
            before_ids
        );
    }

    #[test]
    fn canonical_64_kib_limit_has_a_passing_prefix_and_atomic_failing_batch() {
        const KEYS: [&str; 7] = [
            "error.type",
            "outcome",
            "outcome.label",
            "outcome.success",
            "result",
            "status",
            "success",
        ];
        let event_metadata = KEYS.map(|key| (key, json!("ok")));
        let event = mark_event(
            SUCCESS_EVENT_UUID,
            "wide",
            metadata(None, &event_metadata),
            None,
        );
        let mut ordered_masks = (1_u8..=127).collect::<Vec<_>>();
        ordered_masks.sort_by_key(|mask| std::cmp::Reverse(mask.count_ones()));
        ordered_masks.truncate(OUTCOME_MATCHERS_MAX);

        let all_matchers = ordered_masks
            .into_iter()
            .map(|mask| {
                matcher(
                    OutcomeMatcherEventKind::Mark,
                    "custom",
                    "wide",
                    OutcomeTerminalStatus::Unset,
                    KEYS.iter()
                        .enumerate()
                        .filter(|(index, _)| mask & (1 << index) != 0)
                        .map(|(_, key)| ((*key).to_string(), json!("ok")))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();

        let mut last_passing = None;
        let mut first_failing = None;
        for count in 1..=all_matchers.len() {
            let policy = Arc::new(
                CompiledOutcomePolicyV1::compile(&complete_outcome(
                    all_matchers[..count].to_vec(),
                    vec![mark_matcher("failure")],
                ))
                .unwrap(),
            );
            let mut accumulator = OutcomeSignalAccumulatorV1::new(policy, 0);
            match accumulator.observe_event(&event, &external_context(&event, 1)) {
                OutcomeObservationV1::Added { signals } => {
                    assert_eq!(signals, count);
                    assert!(accumulator.canonical_signal_bytes() <= OUTCOME_SIGNAL_FACTS_MAX_BYTES);
                    last_passing = Some((count, accumulator.canonical_signal_bytes()));
                }
                OutcomeObservationV1::Incomplete(OutcomeCollectionFaultV1::SignalBytesExceeded) => {
                    assert!(accumulator.signals().is_empty());
                    first_failing = Some(count);
                    break;
                }
                other => panic!("unexpected byte-bound result: {other:?}"),
            }
        }
        let (passing_count, passing_bytes) = last_passing.expect("at least one batch fits");
        let failing_count = first_failing.expect("wide protected facts exceed 64 KiB");
        assert_eq!(failing_count, passing_count + 1);
        assert!(passing_bytes <= OUTCOME_SIGNAL_FACTS_MAX_BYTES);
    }

    #[test]
    fn matcher_and_event_order_are_deterministic_and_replay_idempotent() {
        let first = mark_matcher("accepted");
        let second = matcher(
            OutcomeMatcherEventKind::Mark,
            "custom",
            "accepted",
            OutcomeTerminalStatus::Unset,
            BTreeMap::from([("status".to_string(), json!("ok"))]),
        );
        let event_a = mark_event(
            SUCCESS_EVENT_UUID,
            "accepted",
            metadata(None, &[("status", json!("ok"))]),
            None,
        );
        let event_b = mark_event(
            SECOND_EVENT_UUID,
            "accepted",
            metadata(None, &[("status", json!("ok"))]),
            None,
        );
        let build = |matchers: Vec<OutcomeMatcher>, reverse_events: bool| {
            let policy = Arc::new(
                CompiledOutcomePolicyV1::compile(&complete_outcome(
                    matchers,
                    vec![mark_matcher("failure")],
                ))
                .unwrap(),
            );
            let mut accumulator = OutcomeSignalAccumulatorV1::new(policy, 0);
            let observations = if reverse_events {
                [(&event_b, 2), (&event_a, 1)]
            } else {
                [(&event_a, 1), (&event_b, 2)]
            };
            for (event, ingest_seq) in observations {
                assert!(matches!(
                    accumulator.observe_event(event, &external_context(event, ingest_seq)),
                    OutcomeObservationV1::Added { .. }
                ));
            }
            canonical_serialize_bytes(&ProtectedSignalAggregateV1 {
                schema: OUTCOME_SIGNAL_AGGREGATE_SCHEMA_V1,
                signals: accumulator.signals(),
            })
            .unwrap()
        };
        assert_eq!(
            build(vec![first.clone(), second.clone()], false),
            build(vec![second, first], true)
        );
    }

    #[test]
    fn protected_signal_serde_and_compiled_debug_never_expose_raw_values_or_uuids() {
        assert_not_impl_any!(CompiledOutcomePolicyV1: Serialize, Clone);
        let protected_identity = ProtectedIdentityV1::from_sha256_hex(&"ab".repeat(32)).unwrap();
        assert_eq!(protected_identity.to_hex(), "ab".repeat(32));
        assert_eq!(
            serde_json::to_string(&protected_identity).unwrap(),
            format!("\"{}\"", "ab".repeat(32))
        );
        assert!(ProtectedIdentityV1::from_sha256_hex("raw-candidate-id").is_err());
        assert!(ProtectedIdentityV1::from_sha256_hex(&"AB".repeat(32)).is_err());
        let expected = "accepted-private-label";
        let policy = Arc::new(
            CompiledOutcomePolicyV1::compile(&complete_outcome(
                vec![matcher(
                    OutcomeMatcherEventKind::Mark,
                    "custom",
                    "private-event-name",
                    OutcomeTerminalStatus::Unset,
                    BTreeMap::from([("outcome.label".to_string(), json!(expected))]),
                )],
                vec![mark_matcher("failure")],
            ))
            .unwrap(),
        );
        let event = mark_event(
            SUCCESS_EVENT_UUID,
            "private-event-name",
            metadata(
                None,
                &[
                    ("outcome.label", json!(expected)),
                    ("authorization", json!("bearer-secret")),
                ],
            ),
            Some(json!({"prompt": "raw-secret-prompt"})),
        );
        let mut accumulator = OutcomeSignalAccumulatorV1::new(policy.clone(), 0);
        assert!(matches!(
            accumulator.observe_event(&event, &external_context(&event, 1)),
            OutcomeObservationV1::Added { .. }
        ));
        let serialized = serde_json::to_string(accumulator.signals()).unwrap();
        let debug = format!("{policy:?}");
        for forbidden in [
            expected,
            "private-event-name",
            "bearer-secret",
            "raw-secret-prompt",
            &SUCCESS_EVENT_UUID.to_string(),
        ] {
            assert!(!serialized.contains(forbidden), "{forbidden}");
            assert!(!debug.contains(forbidden), "{forbidden}");
        }
        assert!(serialized.contains("observed_scalar_sha256"));
        assert!(serialized.contains("source_event_sha256"));
    }

    #[test]
    fn exact_dispatch_terminal_table_is_fail_closed() {
        for terminal in [
            DispatchTerminalV1::Completed,
            DispatchTerminalV1::ProviderError,
        ] {
            assert!(matches!(
                scan_root_exposure_v1(&exposure_facts(
                    RootAssignmentArmV1::ActiveCanary,
                    Some(terminal)
                )),
                RootExposureV1::Treatment { .. }
            ));
        }
        for terminal in [
            DispatchTerminalV1::CancelledBeforeHandoff,
            DispatchTerminalV1::CancelledAfterHandoff,
            DispatchTerminalV1::PanickedAfterHandoff,
            DispatchTerminalV1::AbortedBeforeHandoff,
        ] {
            assert_eq!(
                scan_root_exposure_v1(&exposure_facts(
                    RootAssignmentArmV1::ActiveCanary,
                    Some(terminal)
                )),
                RootExposureV1::Unattributed
            );
        }
        assert_eq!(
            scan_root_exposure_v1(&exposure_facts(
                RootAssignmentArmV1::ActiveCanary,
                Some(DispatchTerminalV1::UnknownAfterCrash)
            )),
            RootExposureV1::Orphaned
        );
        assert_eq!(
            scan_root_exposure_v1(&exposure_facts(RootAssignmentArmV1::ActiveCanary, None)),
            RootExposureV1::Unattributed
        );
        assert_eq!(
            scan_root_exposure_v1(&exposure_facts(RootAssignmentArmV1::AnchorControl, None)),
            RootExposureV1::AnchorControl
        );
        assert_eq!(
            scan_root_exposure_v1(&exposure_facts(RootAssignmentArmV1::AnchorHoldout, None)),
            RootExposureV1::AnchorHoldout
        );
    }

    #[test]
    fn exposure_scan_rejects_repeated_mixed_pool_experiment_interval_and_control_dispatches() {
        let baseline = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::Completed),
        );
        let mut variants = Vec::new();

        let mut repeated_window = baseline.clone();
        repeated_window.window_fact_count = 2;
        variants.push(repeated_window);
        let mut repeated_assignment = baseline.clone();
        repeated_assignment
            .assignments
            .push(repeated_assignment.assignments[0].clone());
        variants.push(repeated_assignment);
        let mut repeated_admission = baseline.clone();
        repeated_admission
            .dispatches
            .push(repeated_admission.dispatches[0].clone());
        variants.push(repeated_admission);
        let mut repeated_terminal = baseline.clone();
        repeated_terminal.dispatches[0]
            .terminals
            .push(DispatchTerminalV1::Completed);
        variants.push(repeated_terminal);
        let mut mixed_candidate = baseline.clone();
        let mut second = mixed_candidate.dispatches[0].clone();
        second.admission_sha256 = protected(0x77);
        second.candidate_sha256 = protected(0x88);
        mixed_candidate.dispatches.push(second);
        variants.push(mixed_candidate);
        for field in ["candidate", "pool", "experiment", "interval", "window"] {
            let mut variant = baseline.clone();
            match field {
                "candidate" => variant.dispatches[0].candidate_sha256 = protected(0xaa),
                "pool" => variant.dispatches[0].pool_sha256 = protected(0xaa),
                "experiment" => variant.dispatches[0].experiment_sha256 = protected(0xaa),
                "interval" => variant.dispatches[0].interval_sha256 = protected(0xaa),
                "window" => variant.dispatches[0].window_sha256 = protected(0xaa),
                _ => unreachable!(),
            }
            variants.push(variant);
        }
        let mut conflicting = baseline.clone();
        conflicting.conflicting_link = true;
        variants.push(conflicting);
        let mut overlapping = baseline;
        overlapping.overlapping_interval = true;
        variants.push(overlapping);

        for variant in variants {
            assert_eq!(
                scan_root_exposure_v1(&variant),
                RootExposureV1::AmbiguousExposure
            );
        }

        let control_dispatch = exposure_facts(
            RootAssignmentArmV1::AnchorControl,
            Some(DispatchTerminalV1::Completed),
        );
        assert_eq!(
            scan_root_exposure_v1(&control_dispatch),
            RootExposureV1::AmbiguousExposure
        );
        let holdout_dispatch = exposure_facts(
            RootAssignmentArmV1::AnchorHoldout,
            Some(DispatchTerminalV1::ProviderError),
        );
        assert_eq!(
            scan_root_exposure_v1(&holdout_dispatch),
            RootExposureV1::AmbiguousExposure
        );

        let mut repeated_cancelled = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::CancelledAfterHandoff),
        );
        repeated_cancelled.window_fact_count = 2;
        assert_eq!(
            scan_root_exposure_v1(&repeated_cancelled),
            RootExposureV1::Unattributed
        );
    }

    #[test]
    fn terminal_precedence_and_deadline_evidence_are_exact() {
        let coherent = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::Completed),
        );
        let mut completed = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        completed.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::Completed,
                codec_policy_failure: false,
            },
            1,
        );
        assert!(matches!(
            completed
                .classify(RootClosureV1::AttributionDeadline, &coherent)
                .exposure,
            RootExposureV1::Treatment { .. }
        ));

        let custom = mark_event(SUCCESS_EVENT_UUID, "accepted", None, None);
        let mut matcher_only = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        matcher_only.observe_event(&custom, &external_context(&custom, 1));
        assert_eq!(
            matcher_only
                .classify(RootClosureV1::AttributionDeadline, &coherent)
                .exposure,
            RootExposureV1::Unattributed
        );
        assert!(matches!(
            matcher_only
                .classify(RootClosureV1::PinnedOwnerEnd, &coherent)
                .exposure,
            RootExposureV1::Treatment { .. }
        ));

        let mut shutdown = completed.classify(RootClosureV1::GracefulShutdown, &coherent);
        assert_eq!(shutdown.exposure, RootExposureV1::ShutdownOrphaned);
        shutdown = completed.classify(RootClosureV1::DeadProcessRecovery, &coherent);
        assert_eq!(shutdown.exposure, RootExposureV1::Orphaned);

        let mut incomplete = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        incomplete.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::Completed,
                codec_policy_failure: true,
            },
            1,
        );
        incomplete.mark_fault(OutcomeCollectionFaultV1::EventLoss);
        let mut ambiguous = coherent.clone();
        ambiguous.window_fact_count = 2;
        let classified = incomplete.classify(RootClosureV1::PinnedOwnerEnd, &ambiguous);
        assert_eq!(classified.label, RootOutcomeLabelV1::Failure);
        assert!(!classified.label_complete);
        assert_eq!(classified.exposure, RootExposureV1::Unattributed);

        let mut cancelled = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        cancelled.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::Cancelled,
                codec_policy_failure: false,
            },
            1,
        );
        assert_eq!(
            cancelled
                .classify(RootClosureV1::PinnedOwnerEnd, &ambiguous)
                .exposure,
            RootExposureV1::Unattributed
        );

        assert_eq!(
            matcher_only
                .classify(RootClosureV1::AttributionDeadline, &ambiguous)
                .exposure,
            RootExposureV1::Unattributed
        );

        let mut crash = ambiguous;
        crash.dispatches[0].terminals = vec![DispatchTerminalV1::UnknownAfterCrash];
        assert_eq!(
            incomplete
                .classify(RootClosureV1::GracefulShutdown, &crash)
                .exposure,
            RootExposureV1::Orphaned
        );
    }

    #[test]
    fn representative_and_dispatch_direct_terminals_must_agree() {
        let provider_error = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::ProviderError),
        );
        let mut representative_completed = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        representative_completed.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::Completed,
                codec_policy_failure: false,
            },
            1,
        );
        assert_eq!(
            representative_completed
                .classify(RootClosureV1::PinnedOwnerEnd, &provider_error)
                .exposure,
            RootExposureV1::AmbiguousExposure
        );
        assert_eq!(
            representative_completed
                .classify(RootClosureV1::GracefulShutdown, &provider_error)
                .exposure,
            RootExposureV1::ShutdownOrphaned
        );
        representative_completed.mark_fault(OutcomeCollectionFaultV1::EventLoss);
        assert_eq!(
            representative_completed
                .classify(RootClosureV1::PinnedOwnerEnd, &provider_error)
                .exposure,
            RootExposureV1::Unattributed
        );
        assert_eq!(
            representative_completed
                .classify(RootClosureV1::DeadProcessRecovery, &provider_error)
                .exposure,
            RootExposureV1::Orphaned
        );

        let completed = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::Completed),
        );
        let mut representative_error = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        representative_error.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::Error,
                codec_policy_failure: false,
            },
            1,
        );
        assert_eq!(
            representative_error
                .classify(RootClosureV1::PinnedOwnerEnd, &completed)
                .exposure,
            RootExposureV1::AmbiguousExposure
        );

        assert!(matches!(
            representative_error
                .classify(RootClosureV1::PinnedOwnerEnd, &provider_error)
                .exposure,
            RootExposureV1::Treatment { .. }
        ));
    }

    #[test]
    fn representative_uncertainty_rows_remain_nonlearning_without_corrupting_label_completeness() {
        let coherent = exposure_facts(
            RootAssignmentArmV1::ActiveCanary,
            Some(DispatchTerminalV1::Completed),
        );
        for terminal in [
            RepresentativeTerminalV1::Cancelled,
            RepresentativeTerminalV1::Panicked,
            RepresentativeTerminalV1::Aborted,
            RepresentativeTerminalV1::Missing,
        ] {
            let mut accumulator = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
            assert_eq!(
                accumulator.record_representative_result(
                    RepresentativeResultV1 {
                        terminal,
                        codec_policy_failure: false,
                    },
                    1,
                ),
                OutcomeObservationV1::Ignored
            );
            let result = accumulator.classify(RootClosureV1::PinnedOwnerEnd, &coherent);
            assert!(result.label_complete);
            assert_eq!(result.exposure, RootExposureV1::Unattributed);
        }
        let mut crash = OutcomeSignalAccumulatorV1::new(compiled_policy(), 0);
        crash.record_representative_result(
            RepresentativeResultV1 {
                terminal: RepresentativeTerminalV1::UnknownAfterCrash,
                codec_policy_failure: false,
            },
            1,
        );
        assert_eq!(
            crash
                .classify(RootClosureV1::PinnedOwnerEnd, &coherent)
                .exposure,
            RootExposureV1::Orphaned
        );
    }

    #[test]
    fn open_root_budget_has_an_exact_4096_window_boundary_and_releases_capacity() {
        let mut budget = OpenRootWindowBudgetV1::default();
        let key = |index: u64| {
            let mut bytes = [0_u8; 32];
            bytes[24..].copy_from_slice(&index.to_be_bytes());
            RootKey::from_test_bytes(bytes)
        };
        for index in 0..OUTCOME_OPEN_ROOT_WINDOWS_MAX as u64 {
            assert_eq!(budget.try_open(key(index)), OpenRootAdmissionV1::Opened);
        }
        assert_eq!(budget.len(), OUTCOME_OPEN_ROOT_WINDOWS_MAX);
        assert_eq!(budget.try_open(key(0)), OpenRootAdmissionV1::AlreadyOpen);
        assert_eq!(
            budget.try_open(key(OUTCOME_OPEN_ROOT_WINDOWS_MAX as u64)),
            OpenRootAdmissionV1::AtCapacity
        );
        assert!(budget.close(key(17)));
        assert!(!budget.close(key(17)));
        assert_eq!(
            budget.try_open(key(OUTCOME_OPEN_ROOT_WINDOWS_MAX as u64)),
            OpenRootAdmissionV1::Opened
        );
        assert_eq!(budget.len(), OUTCOME_OPEN_ROOT_WINDOWS_MAX);
    }
}
