// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use uuid::Uuid;

use super::{InspectionService, inspection_deadline, map_ledger_error, map_read_pool_error};
use crate::canonical_json::canonical_sha256;
use crate::config::{RouterConfig, RouterMode};
use crate::inspection::cursor::{CursorCodecV1, CursorEndpointV1, CursorSortV1};
use crate::inspection::types::validate_inspection_id;
use crate::inspection::{
    ControlStatusV1, DecisionDetailV1, DecisionExposureV1, DecisionFilterV1, DecisionSummaryV1,
    EffectiveRouterModeV1, EvidenceDetailV1, EvidenceFilterV1, EvidenceSummaryV1,
    FreshnessStatusV1, HealthEventV1, HealthSummaryV1, InspectionError, LeaseStatusV1,
    MigrationSummaryV1, OVERVIEW_REPORT_SCHEMA_V1, OVERVIEW_WINDOW_MS_V1, OperatorHistoryEntryV1,
    OutcomeFilterV1, OutcomeSummaryV1, OverviewReportV1, Page, PageRequest, PoolDetailV1,
    PoolSummaryV1, QueueStatusV1, STATUS_REPORT_SCHEMA_V1, StatusReportV1, VectorIndexStatusV1,
};
use crate::ledger::model::LedgerErrorClass;
use crate::ledger::repository::inspection::{
    InspectionAuthoritySnapshot, load_control_page, load_decision_detail, load_decision_exposure,
    load_decision_follow_page, load_decision_page, load_evidence_detail, load_evidence_page,
    load_health_page, load_migration_page, load_outcome_page, load_overview_snapshot,
    load_pool_detail, load_pool_page, load_status_snapshot,
};

impl InspectionService {
    /// Return one consistent non-secret Router status snapshot.
    pub async fn status(&self) -> Result<StatusReportV1, InspectionError> {
        let snapshot_time_unix_ms = now_unix_ms()?;
        let config = Arc::clone(&self.inner.config);
        let Some(current) = &self.inner.current else {
            return unavailable_status(&config, self.inner.database.clone(), snapshot_time_unix_ms);
        };
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_config = Arc::clone(&config);
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_status_snapshot(
                    connection,
                    &read_config,
                    snapshot_time_unix_ms,
                ))
            })
            .await
        {
            Ok(Ok(snapshot)) => Ok(current_status_report(
                &config,
                self.inner.database.clone(),
                snapshot_time_unix_ms,
                snapshot.authority,
                snapshot.queues,
                snapshot.leases,
                snapshot.vector_index,
                snapshot.freshness,
                snapshot.health,
            )),
            Ok(Err(error)) if is_authority_failure(error.class()) => {
                unavailable_status(&config, self.inner.database.clone(), snapshot_time_unix_ms)
            }
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    /// Return one current-state and exact preceding-24-hour overview snapshot.
    pub async fn overview(&self) -> Result<OverviewReportV1, InspectionError> {
        let snapshot_time_unix_ms = now_unix_ms()?;
        let window_start_unix_ms = snapshot_time_unix_ms
            .checked_sub(OVERVIEW_WINDOW_MS_V1)
            .ok_or(InspectionError::IntegrityError)?;
        let current = self.current()?;
        let config = Arc::clone(&self.inner.config);
        let read_config = Arc::clone(&config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let snapshot = match read_pool
            .run(deadline, move |connection| {
                Ok(load_overview_snapshot(
                    connection,
                    &read_config,
                    window_start_unix_ms,
                    snapshot_time_unix_ms,
                ))
            })
            .await
        {
            Ok(Ok(snapshot)) => snapshot,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let status = current_status_report(
            &config,
            self.inner.database.clone(),
            snapshot_time_unix_ms,
            snapshot.authority,
            snapshot.queues,
            snapshot.leases,
            snapshot.vector_index,
            snapshot.freshness,
            snapshot.health,
        );
        Ok(OverviewReportV1 {
            schema: OVERVIEW_REPORT_SCHEMA_V1.into(),
            window_start_unix_ms,
            snapshot_time_unix_ms,
            status,
            decisions: snapshot.decisions,
            exposures: snapshot.exposures,
            outcomes: snapshot.outcomes,
        })
    }

    /// List configured pools in stable lexical order.
    pub async fn list_pools(
        &self,
        page: PageRequest,
    ) -> Result<Page<PoolSummaryV1>, InspectionError> {
        page.validate()?;
        let current = self.current()?;
        let filter_hash = endpoint_filter_hash("pools")?;
        let (snapshot_time_unix_ms, after_pool_id, cursor_supplied) = match page.after.as_deref() {
            Some(encoded) => {
                let decoded =
                    current
                        .cursor
                        .decode(encoded, CursorEndpointV1::Pools, &filter_hash)?;
                if decoded.maximum_insertion_sequence != 1 {
                    return Err(InspectionError::InvalidCursor);
                }
                let CursorSortV1::Lexical { id } = decoded.sort else {
                    return Err(InspectionError::InvalidCursor);
                };
                (decoded.snapshot_time_unix_ms, Some(id), true)
            }
            None => (now_unix_ms()?, None, false),
        };
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_config = Arc::clone(&config);
        let after = after_pool_id.clone();
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_pool_page(
                    connection,
                    &read_config,
                    after.as_deref(),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        if cursor_supplied
            && read.authority.cohort_generation_id != current.authority.cohort_generation_id
        {
            return Err(InspectionError::InvalidCursor);
        }
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = if has_more {
            let last = read.items.last().ok_or(InspectionError::IntegrityError)?;
            Some(codec_for_authority(self, &read.authority)?.encode(
                CursorEndpointV1::Pools,
                &filter_hash,
                snapshot_time_unix_ms,
                1,
                CursorSortV1::Lexical {
                    id: last.id.clone(),
                },
            )?)
        } else {
            None
        };
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// Return complete stable detail for one exact configured pool.
    pub async fn get_pool(&self, pool_id: &str) -> Result<PoolDetailV1, InspectionError> {
        validate_inspection_id(pool_id)?;
        let current = self.current()?;
        let config = Arc::clone(&self.inner.config);
        let pool_id = pool_id.to_string();
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_pool_detail(connection, &config, &pool_id))
            })
            .await
        {
            Ok(Ok(Some(detail))) => Ok(detail),
            Ok(Ok(None)) => Err(InspectionError::NotFound),
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    /// List verified health events newest first.
    pub async fn list_health(
        &self,
        page: PageRequest,
    ) -> Result<Page<HealthEventV1>, InspectionError> {
        page.validate()?;
        let current = self.current()?;
        let filter_hash = endpoint_filter_hash("health")?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Health,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_health_page(
                    connection,
                    &config,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Health,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items
                .last()
                .map(|item| (item.created_at_unix_ms, item.health_event_id.to_string())),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// List verified embedded migration receipts newest first.
    pub async fn list_migrations(
        &self,
        page: PageRequest,
    ) -> Result<Page<MigrationSummaryV1>, InspectionError> {
        page.validate()?;
        let current = self.current()?;
        let filter_hash = endpoint_filter_hash("migrations")?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Migrations,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_migration_page(
                    connection,
                    &config,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Migrations,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items.last().and_then(|item| {
                item.applied_at_unix_ms
                    .map(|time| (time, format!("{:020}", item.version)))
            }),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// List verified live control mutation receipts newest first.
    pub async fn list_controls(
        &self,
        page: PageRequest,
    ) -> Result<Page<OperatorHistoryEntryV1>, InspectionError> {
        page.validate()?;
        let current = self.current()?;
        let filter_hash = endpoint_filter_hash("controls")?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Controls,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_control_page(
                    connection,
                    &config,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Controls,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items
                .last()
                .map(|item| (item.created_at_unix_ms, item.audit_id.to_string())),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// List verified immutable evidence links newest first.
    pub async fn list_evidence(
        &self,
        filter: EvidenceFilterV1,
        page: PageRequest,
    ) -> Result<Page<EvidenceSummaryV1>, InspectionError> {
        filter.validate()?;
        page.validate()?;
        let current = self.current()?;
        let filter_hash = evidence_filter_hash(&filter)?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Evidence,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let content_policy = self.content_policy();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_evidence_page(
                    connection,
                    &config,
                    content_policy,
                    &filter,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Evidence,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items
                .last()
                .map(|item| (item.created_at_unix_ms, item.evidence_id.to_string())),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy,
        })
    }

    /// Return one verified immutable evidence graph.
    pub async fn get_evidence(
        &self,
        evidence_id: Uuid,
    ) -> Result<EvidenceDetailV1, InspectionError> {
        validate_uuid_v7(evidence_id)?;
        let current = self.current()?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let content_policy = self.content_policy();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_evidence_detail(
                    connection,
                    &config,
                    content_policy,
                    evidence_id,
                ))
            })
            .await
        {
            Ok(Ok(Some(detail))) => Ok(detail),
            Ok(Ok(None)) => Err(InspectionError::NotFound),
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    /// List verified immutable routing decisions newest first.
    pub async fn list_decisions(
        &self,
        filter: DecisionFilterV1,
        page: PageRequest,
    ) -> Result<Page<DecisionSummaryV1>, InspectionError> {
        filter.validate()?;
        page.validate()?;
        let current = self.current()?;
        let filter_hash = decision_filter_hash(&filter)?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Decisions,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_decision_page(
                    connection,
                    &config,
                    &filter,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Decisions,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items
                .last()
                .map(|item| (item.created_at_unix_ms, item.decision_id.to_string())),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// Tail verified immutable decisions using a forward, resumable cursor.
    pub async fn tail_decisions(
        &self,
        filter: DecisionFilterV1,
        page: PageRequest,
    ) -> Result<Page<DecisionSummaryV1>, InspectionError> {
        filter.validate()?;
        page.validate()?;
        let current = self.current()?;
        let filter_hash = decision_filter_hash(&filter)?;
        let supplied_cursor = page.after.clone();
        let after_insertion_sequence = match page.after.as_deref() {
            Some(encoded) => {
                let decoded = current.cursor.decode(
                    encoded,
                    CursorEndpointV1::DecisionFollow,
                    &filter_hash,
                )?;
                let CursorSortV1::Follow { insertion_sequence } = decoded.sort else {
                    return Err(InspectionError::InvalidCursor);
                };
                Some(insertion_sequence)
            }
            None => None,
        };
        let snapshot_time_unix_ms = now_unix_ms()?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_decision_follow_page(
                    connection,
                    &config,
                    &filter,
                    after_insertion_sequence,
                    usize::from(page.limit),
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let next = if read.next_insertion_sequence == 0 {
            None
        } else if after_insertion_sequence == Some(read.next_insertion_sequence) {
            supplied_cursor
        } else {
            Some(current.cursor.encode(
                CursorEndpointV1::DecisionFollow,
                &filter_hash,
                snapshot_time_unix_ms,
                read.next_insertion_sequence,
                CursorSortV1::Follow {
                    insertion_sequence: read.next_insertion_sequence,
                },
            )?)
        };
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }

    /// Return one verified immutable decision aggregate.
    pub async fn get_decision(
        &self,
        decision_id: Uuid,
    ) -> Result<DecisionDetailV1, InspectionError> {
        validate_uuid_v7(decision_id)?;
        let current = self.current()?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_decision_detail(connection, &config, decision_id))
            })
            .await
        {
            Ok(Ok(Some(detail))) => Ok(detail),
            Ok(Ok(None)) => Err(InspectionError::NotFound),
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    /// Return persisted Active assignment and representative outcome facts for one decision.
    pub async fn get_decision_exposure(
        &self,
        decision_id: Uuid,
    ) -> Result<DecisionExposureV1, InspectionError> {
        validate_uuid_v7(decision_id)?;
        let current = self.current()?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_decision_exposure(connection, &config, decision_id))
            })
            .await
        {
            Ok(Ok(Some(exposure))) => Ok(exposure),
            Ok(Ok(None)) => Err(InspectionError::NotFound),
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    /// List verified immutable Active outcomes newest first.
    pub async fn list_outcomes(
        &self,
        filter: OutcomeFilterV1,
        page: PageRequest,
    ) -> Result<Page<OutcomeSummaryV1>, InspectionError> {
        filter.validate()?;
        page.validate()?;
        let current = self.current()?;
        let filter_hash = outcome_filter_hash(&filter)?;
        let decoded = decode_newest_page(
            &current.cursor,
            CursorEndpointV1::Outcomes,
            &filter_hash,
            page.after.as_deref(),
        )?;
        let snapshot_time_unix_ms = decoded.snapshot_time_unix_ms;
        let maximum = decoded.maximum_insertion_sequence;
        let after = decoded.after;
        let limit = usize::from(page.limit);
        let fetch_limit = limit
            .checked_add(1)
            .ok_or(InspectionError::InvalidArgument)?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let read_after = after.clone();
        let mut read = match read_pool
            .run(deadline, move |connection| {
                Ok(load_outcome_page(
                    connection,
                    &config,
                    &filter,
                    snapshot_time_unix_ms,
                    maximum,
                    read_after.as_ref().map(|(time, id)| (*time, id.as_str())),
                    fetch_limit,
                ))
            })
            .await
        {
            Ok(Ok(read)) => read,
            Ok(Err(error)) => return Err(map_ledger_error(error)),
            Err(error) => return Err(map_read_pool_error(error)),
        };
        let has_more = read.items.len() > limit;
        read.items.truncate(limit);
        let next = encode_newest_next(
            &current.cursor,
            CursorEndpointV1::Outcomes,
            &filter_hash,
            snapshot_time_unix_ms,
            read.maximum_insertion_sequence,
            has_more,
            read.items
                .last()
                .map(|item| (item.created_at_unix_ms, item.outcome_id.to_string())),
        )?;
        Ok(Page {
            items: read.items,
            next,
            snapshot_time_unix_ms,
            content_policy: self.content_policy(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn current_status_report(
    config: &RouterConfig,
    database: crate::inspection::DatabaseStatusV1,
    snapshot_time_unix_ms: u64,
    authority: InspectionAuthoritySnapshot,
    queues: QueueStatusV1,
    leases: LeaseStatusV1,
    vector_index: VectorIndexStatusV1,
    freshness: FreshnessStatusV1,
    health: HealthSummaryV1,
) -> StatusReportV1 {
    let controls = authority
        .control_snapshot
        .as_ref()
        .map(|control| ControlStatusV1 {
            control_generation: control.control_generation,
            all: control.all,
            effective: control.all,
        });
    StatusReportV1 {
        schema: STATUS_REPORT_SCHEMA_V1.into(),
        project_id: Some(authority.project_id),
        configured_mode: config.mode,
        effective_mode: effective_mode(config.mode, controls.as_ref(), true),
        config_generation_id: Some(authority.config_generation_id),
        cohort_generation_id: Some(authority.cohort_generation_id),
        controls,
        database,
        queues,
        leases,
        vector_index,
        freshness,
        health,
        snapshot_time_unix_ms,
    }
}

fn unavailable_status(
    config: &RouterConfig,
    database: crate::inspection::DatabaseStatusV1,
    snapshot_time_unix_ms: u64,
) -> Result<StatusReportV1, InspectionError> {
    let capacity = if config.mode == RouterMode::Off {
        0
    } else {
        u64::try_from(
            config
                .writer_command_capacity()
                .map_err(|_| InspectionError::InvalidArgument)?,
        )
        .map_err(|_| InspectionError::InvalidArgument)?
    };
    Ok(StatusReportV1 {
        schema: STATUS_REPORT_SCHEMA_V1.into(),
        project_id: config.project_id.clone(),
        configured_mode: config.mode,
        effective_mode: effective_mode(config.mode, None, false),
        config_generation_id: None,
        cohort_generation_id: None,
        controls: None,
        database,
        queues: QueueStatusV1 {
            capacity,
            ..QueueStatusV1::default()
        },
        leases: LeaseStatusV1::default(),
        vector_index: VectorIndexStatusV1::default(),
        freshness: FreshnessStatusV1::default(),
        health: HealthSummaryV1 {
            worst: "unavailable".into(),
            degraded: 1,
            latest_event_unix_ms: None,
        },
        snapshot_time_unix_ms,
    })
}

fn effective_mode(
    configured: RouterMode,
    controls: Option<&ControlStatusV1>,
    authority_available: bool,
) -> EffectiveRouterModeV1 {
    if configured == RouterMode::Off {
        return EffectiveRouterModeV1::Off;
    }
    if !authority_available {
        return EffectiveRouterModeV1::Unavailable;
    }
    let Some(controls) = controls else {
        return EffectiveRouterModeV1::Unavailable;
    };
    if controls.effective.paused {
        EffectiveRouterModeV1::Paused
    } else if controls.effective.force_anchor {
        EffectiveRouterModeV1::ForceAnchor
    } else {
        match configured {
            RouterMode::Off => EffectiveRouterModeV1::Off,
            RouterMode::Shadow => EffectiveRouterModeV1::Shadow,
            RouterMode::Recommend => EffectiveRouterModeV1::Recommend,
            RouterMode::Active => EffectiveRouterModeV1::Active,
        }
    }
}

struct DecodedNewestPage {
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, String)>,
}

fn decode_newest_page(
    codec: &CursorCodecV1,
    endpoint: CursorEndpointV1,
    filter_hash: &str,
    encoded: Option<&str>,
) -> Result<DecodedNewestPage, InspectionError> {
    let Some(encoded) = encoded else {
        return Ok(DecodedNewestPage {
            snapshot_time_unix_ms: now_unix_ms()?,
            maximum_insertion_sequence: None,
            after: None,
        });
    };
    let decoded = codec.decode(encoded, endpoint, filter_hash)?;
    let CursorSortV1::Newest {
        created_at_unix_ms,
        id,
    } = decoded.sort
    else {
        return Err(InspectionError::InvalidCursor);
    };
    Ok(DecodedNewestPage {
        snapshot_time_unix_ms: decoded.snapshot_time_unix_ms,
        maximum_insertion_sequence: Some(decoded.maximum_insertion_sequence),
        after: Some((created_at_unix_ms, id)),
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_newest_next(
    codec: &CursorCodecV1,
    endpoint: CursorEndpointV1,
    filter_hash: &str,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: u64,
    has_more: bool,
    last: Option<(u64, String)>,
) -> Result<Option<String>, InspectionError> {
    if !has_more {
        return Ok(None);
    }
    let (created_at_unix_ms, id) = last.ok_or(InspectionError::IntegrityError)?;
    codec
        .encode(
            endpoint,
            filter_hash,
            snapshot_time_unix_ms,
            maximum_insertion_sequence,
            CursorSortV1::Newest {
                created_at_unix_ms,
                id,
            },
        )
        .map(Some)
}

fn codec_for_authority(
    service: &InspectionService,
    authority: &InspectionAuthoritySnapshot,
) -> Result<CursorCodecV1, InspectionError> {
    CursorCodecV1::new(
        authority.cursor_key.clone(),
        service.inner.database.supported_schema_version,
        authority.project_uuid,
        authority.cohort_generation_id,
    )
}

fn endpoint_filter_hash(endpoint: &str) -> Result<String, InspectionError> {
    canonical_sha256(&json!({ "endpoint": endpoint })).map_err(|_| InspectionError::InvalidArgument)
}

fn evidence_filter_hash(filter: &EvidenceFilterV1) -> Result<String, InspectionError> {
    canonical_sha256(&json!({ "endpoint": "evidence", "filter": filter }))
        .map_err(|_| InspectionError::InvalidArgument)
}

fn decision_filter_hash(filter: &DecisionFilterV1) -> Result<String, InspectionError> {
    canonical_sha256(&json!({ "endpoint": "decisions", "filter": filter }))
        .map_err(|_| InspectionError::InvalidArgument)
}

fn outcome_filter_hash(filter: &OutcomeFilterV1) -> Result<String, InspectionError> {
    canonical_sha256(&json!({ "endpoint": "outcomes", "filter": filter }))
        .map_err(|_| InspectionError::InvalidArgument)
}

fn validate_uuid_v7(value: Uuid) -> Result<(), InspectionError> {
    if value.get_variant() != uuid::Variant::RFC4122 || value.get_version_num() != 7 {
        return Err(InspectionError::InvalidArgument);
    }
    Ok(())
}

pub(super) fn now_unix_ms() -> Result<u64, InspectionError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| InspectionError::StorageUnavailable)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| InspectionError::StorageUnavailable)
}

fn is_authority_failure(class: LedgerErrorClass) -> bool {
    matches!(
        class,
        LedgerErrorClass::CorruptDatabase
            | LedgerErrorClass::IdentityInvariant
            | LedgerErrorClass::CanonicalizationFailed
            | LedgerErrorClass::PragmaMismatch
            | LedgerErrorClass::SqliteVersionMismatch
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use tempfile::tempdir;
    use uuid::Uuid;

    use super::*;
    use crate::config::{
        CandidateCapabilities, CandidateConfig, ConcurrencyConfig, EmbedderConfig,
        JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig, LearningConfig, PoolConfig,
    };
    use crate::control::{
        ControlMutation, ControlOperation, ControlScope, ControlTransactionFence,
    };
    use crate::inspection::{
        CohortRotationRequestV1, ContentPolicy, InspectionDatabaseStateV1,
        InspectionServiceOptions, LearningResetRequestV1, LearningResetScopeV1,
        OperatorHistoryKindV1, OperatorMutationReceiptV1, PoolSupportV1,
    };
    use crate::ledger::migrations::{LEDGER_APPLICATION_ID, MIGRATIONS};
    use crate::ledger::repository::active::{
        ActiveAdmissionAck, ActiveAssignmentArm, ActiveDispatchTerminal, ActiveDispatchTerminalAck,
        ActiveDispatchTerminalState, ActiveRootAdmission, ActiveRootClosure, ActiveRootTerminalAck,
        ActiveSignalBatch, ActiveSignalBatchAck, ActiveSignalDisposition,
        tests as active_test_fixtures,
    };
    use crate::ledger::repository::control::prepare_control_mutation;
    use crate::ledger::repository::decision::{DecisionAuditAck, tests as decision_test_fixtures};
    use crate::ledger::repository::inspection::operator::{
        OperatorMutationTransactionAck, PreparedOperatorMutation, prepare_cohort_rotation,
        prepare_learning_reset,
    };
    use crate::ledger::repository::process::{LedgerHealthEvent, LedgerHealthSeverity};
    use crate::ledger::repository::vector_index::{
        GenerationAuthorizationAck, GenerationObjectCreationAck, RebuildFlipAck,
        RebuildLeaseClaimAck, RebuildStepAck, authorize_generation, catch_up_rebuild_changes,
        claim_rebuild_lease, create_generation_objects, flip_rebuild_generation,
        populate_rebuild_chunk, vector_source_sequence_payload_hash,
    };
    use crate::ledger::repository::{
        LedgerRepository, ready_evaluated_active_runtime_fixture, ready_evaluated_runtime_fixture,
    };
    use crate::vector::VectorDimensions;

    fn config(path: &Path, project_id: &str, mode: RouterMode) -> RouterConfig {
        RouterConfig {
            mode,
            project_id: Some(project_id.into()),
            database_path: path.to_string_lossy().into_owned(),
            ..RouterConfig::default()
        }
    }

    fn pool(id: &str) -> PoolConfig {
        PoolConfig {
            id: id.into(),
            api_family: nemo_relay_types::api::llm::LlmApiFamily::OpenAIChatCompletions,
            anchor_models: vec![format!("{id}-anchor")],
            anchor_revision: "anchor-r1".into(),
            sampling_probability: 1.0,
            max_candidates_per_sample: 1,
            selector: Default::default(),
            lookahead: Default::default(),
            concurrency: ConcurrencyConfig {
                shadow: 1,
                judge: 1,
                max_pending: 4,
                unknown_fields: BTreeMap::new(),
            },
            candidates: vec![
                CandidateConfig {
                    id: format!("{id}-expensive"),
                    model: "candidate-expensive".into(),
                    model_revision: "candidate-r1".into(),
                    cost_rank: 1,
                    max_context_tokens: None,
                    capabilities: CandidateCapabilities {
                        tools: true,
                        multimodal_input: true,
                        structured_output: true,
                        reasoning_controls: true,
                        unknown_fields: BTreeMap::new(),
                    },
                    unknown_fields: BTreeMap::new(),
                },
                CandidateConfig {
                    id: format!("{id}-cheap"),
                    model: "candidate-cheap".into(),
                    model_revision: "candidate-r1".into(),
                    cost_rank: 0,
                    max_context_tokens: None,
                    capabilities: CandidateCapabilities {
                        tools: true,
                        multimodal_input: true,
                        structured_output: true,
                        reasoning_controls: true,
                        unknown_fields: BTreeMap::new(),
                    },
                    unknown_fields: BTreeMap::new(),
                },
            ],
            canonicalizer: Default::default(),
            judge: JudgeConfig {
                version: 1,
                model: "judge-model".into(),
                model_revision: "judge-r1".into(),
                prompt_version: JUDGE_PROMPT_VERSION_V1.into(),
                rubric_version: JUDGE_RUBRIC_VERSION_V1.into(),
                output_schema_version: 1,
                temperature: None,
                response_weight: 0.5,
                trajectory_weight: 0.5,
                response_floor: 0.8,
                trajectory_floor: 0.8,
                judge_confidence_floor: 0.7,
                pass_threshold: 0.85,
                max_rationale_bytes: 4_096,
                base_cooloff_seconds: 10,
                max_cooloff_seconds: 300,
                unknown_fields: BTreeMap::new(),
            },
            learning: None,
            outcome: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        }
    }

    fn pooled_config(path: &Path, project_id: &str) -> RouterConfig {
        let mut config = config(path, project_id, RouterMode::Shadow);
        config.pools = vec![pool("pool-c"), pool("pool-a"), pool("pool-b")];
        assert!(config.validate().is_empty(), "{:?}", config.validate());
        config
    }

    fn learning_config(path: &Path, project_id: &str) -> RouterConfig {
        let mut config = config(path, project_id, RouterMode::Shadow);
        config.embedders = vec![EmbedderConfig {
            id: "embedding-main".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "embedding-model".into(),
            provider_revision: "embedding-r1".into(),
            dimensions: 2,
            api_key_env: None,
            timeout_ms: 1_000,
            max_in_flight: 1,
            batch_size: 1,
            unknown_fields: BTreeMap::new(),
        }];
        let mut pool = pool("pool-a");
        pool.learning = Some(LearningConfig::minimal("embedding-main"));
        config.pools = vec![pool];
        assert!(config.validate().is_empty(), "{:?}", config.validate());
        config
    }

    fn complete_learning(active: bool) -> LearningConfig {
        LearningConfig {
            version: 1,
            embedder: "embedding-main".into(),
            top_k: Some(8),
            radius: Some(0.25),
            min_points: Some(3),
            min_independent_roots: Some(2),
            min_effective_samples: Some(1.5),
            min_coverage: Some(0.8),
            time_decay_half_life_seconds: Some(3_600.0),
            prior_success: Some(1.0),
            prior_failure: Some(1.0),
            familywise_credible_level: Some(0.95),
            promotion_lower_bound: Some(0.9),
            retention_lower_bound: active.then_some(0.8),
            holdout_probability: active.then_some(0.2),
            active_canary_fraction: active.then_some(0.4),
        }
    }

    fn active_outcome_policy() -> BTreeMap<String, serde_json::Value> {
        serde_json::from_value(serde_json::json!({
            "version": 1,
            "success_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "ok",
                "metadata_equals": {"outcome.label": "passed"}
            }],
            "failure_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "error",
                "metadata_equals": {"outcome.label": "failed"}
            }],
            "completion_disposition": "success",
            "error_disposition": "failure",
            "tool_failure_disposition": "failure",
            "end_of_run_disposition": "ignore",
            "max_attribution_seconds": 600,
            "actual_outcome_half_life_seconds": 1800,
            "anchor_shadow_half_life_seconds": 1800,
            "relearning_cooloff_seconds": 300,
            "min_treatment_roots": 32,
            "min_control_roots": 32,
            "min_treatment_effective_weight": 16.0,
            "min_control_effective_weight": 16.0,
            "noninferiority_margin": 0.1,
            "noninferiority_probability": 0.99,
            "rollback_probability": 0.95,
            "outcome_evaluation_batch_size": 64,
            "max_canary_roots": 64,
            "authorization_ttl_seconds": 600
        }))
        .unwrap()
    }

    fn apply_control(
        repository: &mut LedgerRepository,
        mutation: ControlMutation,
    ) -> crate::ledger::repository::control::ControlMutationAck {
        let prepared = prepare_control_mutation(mutation).unwrap();
        let fence = Arc::new(ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(1),
        ));
        repository
            .apply_control_mutation_with_start_check(&prepared, &fence, || Some(()))
            .unwrap()
    }

    fn apply_operator(
        repository: &mut LedgerRepository,
        prepared: PreparedOperatorMutation,
    ) -> OperatorMutationReceiptV1 {
        let fence = Arc::new(ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(1),
        ));
        match repository
            .apply_operator_mutation_with_start_check(&prepared, &fence, || Some(()))
            .unwrap()
        {
            OperatorMutationTransactionAck::Completed(acknowledgement) => acknowledgement.receipt,
            OperatorMutationTransactionAck::TransactionNotStarted => {
                panic!("operator transaction did not start")
            }
        }
    }

    fn seed_active_outcome(
        fixture: &mut active_test_fixtures::Fixture,
        admission: ActiveRootAdmission,
        signal_seed: u64,
    ) -> Uuid {
        let receipt = match fixture
            .activated
            .repository
            .admit_active_root(&admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected admission acknowledgement: {acknowledgement:?}"),
        };
        assert!(matches!(
            fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: admission.active_root_window_id,
                    signals: vec![active_test_fixtures::protected_signal(
                        signal_seed,
                        ActiveSignalDisposition::Success,
                        receipt.admitted_at_unix_ms + 1,
                        "{}".into(),
                    )],
                })
                .unwrap(),
            ActiveSignalBatchAck::Applied { .. }
        ));
        if let Some(dispatch) = admission.dispatch.as_ref() {
            assert_eq!(
                fixture
                    .activated
                    .repository
                    .record_active_dispatch_terminal(&ActiveDispatchTerminal {
                        active_dispatch_terminal_event_id: Uuid::now_v7(),
                        active_dispatch_id: dispatch.active_dispatch_id,
                        active_assignment_id: admission.active_assignment_id,
                        terminal_state: ActiveDispatchTerminalState::Completed,
                        stable_error_class: None,
                        provider_receipt_hash: None,
                        handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                    })
                    .unwrap(),
                ActiveDispatchTerminalAck::Applied
            );
        }
        let terminal = active_test_fixtures::terminal_for(
            &admission,
            receipt,
            ActiveRootClosure::OwnerEnd,
            true,
            Some(crate::ledger::repository::active::ActiveRepresentativeStatus::Completed),
        );
        let outcome_id = terminal.outcome_id;
        assert_eq!(
            fixture
                .activated
                .repository
                .terminalize_active_root(&terminal)
                .unwrap(),
            ActiveRootTerminalAck::Applied
        );
        outcome_id
    }

    #[tokio::test]
    async fn status_only_reports_are_fail_closed_without_creating_storage() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("missing.sqlite3");
        let service = InspectionService::open(
            config(&path, "missing-status", RouterMode::Shadow),
            InspectionServiceOptions::default(),
        )
        .await
        .unwrap();
        let report = service.status().await.unwrap();
        assert_eq!(report.schema, STATUS_REPORT_SCHEMA_V1);
        assert_eq!(report.project_id.as_deref(), Some("missing-status"));
        assert_eq!(report.configured_mode, RouterMode::Shadow);
        assert_eq!(report.effective_mode, EffectiveRouterModeV1::Unavailable);
        assert_eq!(report.database.state, InspectionDatabaseStateV1::Missing);
        assert_eq!(report.health.worst, "unavailable");
        assert_eq!(report.health.degraded, 1);
        assert_eq!(
            service.overview().await.unwrap_err(),
            InspectionError::StorageUnavailable
        );
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn current_off_status_and_migrations_are_complete_and_paginated() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = config(&path, "off-status", RouterMode::Off);
        drop(LedgerRepository::activate(&config).unwrap());
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();

        let report = service.status().await.unwrap();
        assert_eq!(report.project_id.as_deref(), Some("off-status"));
        assert_eq!(report.effective_mode, EffectiveRouterModeV1::Off);
        assert_eq!(report.queues, QueueStatusV1::default());
        assert_eq!(report.leases, LeaseStatusV1::default());
        assert_eq!(report.vector_index, VectorIndexStatusV1::default());
        assert_eq!(report.freshness, FreshnessStatusV1::default());
        assert_eq!(report.health, HealthSummaryV1::default());
        let overview = service.overview().await.unwrap();
        assert_eq!(overview.schema, OVERVIEW_REPORT_SCHEMA_V1);
        assert_eq!(
            overview.snapshot_time_unix_ms - overview.window_start_unix_ms,
            OVERVIEW_WINDOW_MS_V1
        );
        assert_eq!(overview.decisions, Default::default());
        assert_eq!(overview.exposures, Default::default());
        assert_eq!(overview.outcomes, Default::default());
        assert_eq!(overview.status.effective_mode, EffectiveRouterModeV1::Off);
        let mut status_json = serde_json::to_value(&report).unwrap();
        status_json["config_generation_id"] = json!("CONFIG_GENERATION");
        status_json["cohort_generation_id"] = json!("COHORT_GENERATION");
        status_json["snapshot_time_unix_ms"] = json!(0);
        assert_eq!(
            status_json,
            json!({
                "schema": STATUS_REPORT_SCHEMA_V1,
                "project_id": "off-status",
                "configured_mode": "off",
                "effective_mode": "off",
                "config_generation_id": "CONFIG_GENERATION",
                "cohort_generation_id": "COHORT_GENERATION",
                "controls": null,
                "database": {
                    "state": "current",
                    "application_id": i64::from(LEDGER_APPLICATION_ID),
                    "schema_version": MIGRATIONS.last().unwrap().version,
                    "supported_schema_version": MIGRATIONS.last().unwrap().version,
                },
                "queues": { "pending": 0, "capacity": 0, "rejected": 0 },
                "leases": { "active": 0, "expired": 0 },
                "vector_index": {
                    "active_spaces": 0,
                    "rebuilding_spaces": 0,
                    "degraded_spaces": 0,
                    "stale_spaces": 0,
                },
                "freshness": {
                    "latest_evidence_unix_ms": null,
                    "latest_decision_unix_ms": null,
                    "latest_outcome_unix_ms": null,
                },
                "health": {
                    "worst": "clear",
                    "degraded": 0,
                    "latest_event_unix_ms": null,
                },
                "snapshot_time_unix_ms": 0,
            })
        );

        let mut request = PageRequest {
            limit: 2,
            after: None,
        };
        let mut versions = Vec::new();
        loop {
            let page = service.list_migrations(request.clone()).await.unwrap();
            versions.extend(page.items.iter().map(|migration| migration.version));
            let Some(next) = page.next else {
                break;
            };
            request.after = Some(next);
        }
        let expected = MIGRATIONS
            .iter()
            .rev()
            .map(|migration| migration.version)
            .collect::<Vec<_>>();
        assert_eq!(versions, expected);
        service.close().await.unwrap();
    }

    #[tokio::test]
    async fn controls_and_health_are_verified_and_paginated() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = pooled_config(&path, "history-status");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let initial_learning_generations = activated
            .identity
            .pools
            .iter()
            .map(|(pool_id, identity)| (pool_id.clone(), identity.learning_generation_id))
            .collect::<BTreeMap<_, _>>();
        let initial_cohort_generation = activated.identity.cohort_generation_id;

        let first_id = Uuid::now_v7();
        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: first_id,
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: true },
                expected_control_generation: 0,
                actor: "operator-a".into(),
                reason: "pause for inspection".into(),
            },
        );
        let second_id = Uuid::now_v7();
        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: second_id,
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: true },
                expected_control_generation: 1,
                actor: "operator-a".into(),
                reason: "repeat pause".into(),
            },
        );
        let third_id = Uuid::now_v7();
        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: third_id,
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: false },
                expected_control_generation: 0,
                actor: "operator-b".into(),
                reason: "stale resume".into(),
            },
        );

        let health_time = i64::try_from(now_unix_ms().unwrap()).unwrap();
        let mut health_ids = Vec::new();
        for (offset, severity) in [
            (0, LedgerHealthSeverity::Info),
            (1, LedgerHealthSeverity::Warning),
            (2, LedgerHealthSeverity::Degraded),
        ] {
            let event = LedgerHealthEvent::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                None,
                None,
                "router.writer.test",
                severity,
                health_time + offset,
            )
            .unwrap();
            health_ids.push(event.health_event_id);
            activated.repository.append_health_event(&event).unwrap();
        }
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        let status = service.status().await.unwrap();
        assert_eq!(status.effective_mode, EffectiveRouterModeV1::Paused);
        assert_eq!(status.health.worst, "degraded");
        assert_eq!(status.health.degraded, 1);

        let first = service
            .list_controls(PageRequest {
                limit: 2,
                after: None,
            })
            .await
            .unwrap();
        assert_eq!(first.items.len(), 2);
        let post_snapshot_control_id = Uuid::now_v7();
        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: post_snapshot_control_id,
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: false },
                expected_control_generation: 1,
                actor: "operator-c".into(),
                reason: "post-snapshot resume".into(),
            },
        );
        let reset_id = Uuid::now_v7();
        let reset_receipt = apply_operator(
            &mut activated.repository,
            prepare_learning_reset(LearningResetRequestV1 {
                mutation_id: reset_id,
                scope: LearningResetScopeV1::All {
                    expected_learning_generation_ids: initial_learning_generations.clone(),
                },
                confirm_project_id: "history-status".into(),
                actor: "operator-c".into(),
                reason: "reset learning after snapshot".into(),
            })
            .unwrap(),
        );
        let rotate_id = Uuid::now_v7();
        let rotate_receipt = apply_operator(
            &mut activated.repository,
            prepare_cohort_rotation(CohortRotationRequestV1 {
                mutation_id: rotate_id,
                expected_cohort_generation_id: initial_cohort_generation,
                confirm_project_id: "history-status".into(),
                actor: "operator-c".into(),
                reason: "rotate cohort after snapshot".into(),
            })
            .unwrap(),
        );
        let second = service
            .list_controls(PageRequest {
                limit: 2,
                after: first.next.clone(),
            })
            .await
            .unwrap();
        let controls = first
            .items
            .into_iter()
            .chain(second.items)
            .collect::<Vec<_>>();
        assert_eq!(controls.len(), 3);
        assert_eq!(
            controls
                .iter()
                .map(|entry| entry.audit_id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([first_id, second_id, third_id])
        );
        assert!(controls.iter().any(|entry| {
            entry.result == "applied"
                && entry.prior_value == Some(false)
                && entry.new_value == Some(true)
        }));
        assert!(controls.iter().any(|entry| {
            entry.result == "no_op"
                && entry.prior_value == Some(true)
                && entry.new_value == Some(true)
        }));
        assert!(controls.iter().any(|entry| {
            entry.result == "conflict" && entry.prior_value.is_none() && entry.new_value.is_none()
        }));

        let fresh_service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let fresh = fresh_service
            .list_controls(PageRequest {
                limit: 10,
                after: None,
            })
            .await
            .unwrap();
        assert_eq!(fresh.items.len(), 6);
        assert_eq!(
            fresh
                .items
                .iter()
                .map(|entry| entry.audit_id)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([
                first_id,
                second_id,
                third_id,
                post_snapshot_control_id,
                reset_id,
                rotate_id,
            ])
        );
        let reset = fresh
            .items
            .iter()
            .find(|entry| entry.audit_id == reset_id)
            .unwrap();
        assert_eq!(reset.kind, OperatorHistoryKindV1::ResetAll);
        assert_eq!(reset.result, "applied");
        assert_eq!(reset.prior_generations, initial_learning_generations);
        assert_eq!(reset.new_generations, reset_receipt.resulting_generations);
        assert_eq!(reset.superseded_ids.len(), 3);
        assert!(reset.control_generation.is_none() && reset.scope.is_none());
        let rotation = fresh
            .items
            .iter()
            .find(|entry| entry.audit_id == rotate_id)
            .unwrap();
        assert_eq!(rotation.kind, OperatorHistoryKindV1::RotateCohort);
        assert_eq!(rotation.result, "applied");
        assert_eq!(
            rotation.prior_generations,
            BTreeMap::from([("cohort".into(), initial_cohort_generation)])
        );
        assert_eq!(
            rotation.new_generations,
            rotate_receipt.resulting_generations
        );
        assert_eq!(rotation.superseded_ids, vec![initial_cohort_generation]);
        fresh_service.close().await.unwrap();

        let first = service
            .list_health(PageRequest {
                limit: 2,
                after: None,
            })
            .await
            .unwrap();
        assert_eq!(first.items.len(), 2);
        let post_snapshot_health = LedgerHealthEvent::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            None,
            None,
            "router.writer.test",
            LedgerHealthSeverity::Info,
            health_time + 3,
        )
        .unwrap();
        activated
            .repository
            .append_health_event(&post_snapshot_health)
            .unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "DELETE FROM health_events WHERE health_event_id = ?1",
                [health_ids[0].to_string()],
            )
            .unwrap();
        let second = service
            .list_health(PageRequest {
                limit: 2,
                after: first.next.clone(),
            })
            .await
            .unwrap();
        assert_eq!(first.items.len() + second.items.len(), 2);
        assert!(second.next.is_none());
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn pool_pages_are_lexical_and_observe_live_force_anchor_controls() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = pooled_config(&path, "pool-status");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        assert_eq!(
            service.status().await.unwrap().effective_mode,
            EffectiveRouterModeV1::Shadow
        );

        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: Uuid::now_v7(),
                scope: ControlScope::All,
                operation: ControlOperation::SetForceAnchor { value: true },
                expected_control_generation: 0,
                actor: "operator-a".into(),
                reason: "force anchor for maintenance".into(),
            },
        );
        assert_eq!(
            service.status().await.unwrap().effective_mode,
            EffectiveRouterModeV1::ForceAnchor
        );

        let first = service
            .list_pools(PageRequest {
                limit: 2,
                after: None,
            })
            .await
            .unwrap();
        assert_eq!(
            first
                .items
                .iter()
                .map(|pool| pool.id.as_str())
                .collect::<Vec<_>>(),
            vec!["pool-a", "pool-b"]
        );
        let second = service
            .list_pools(PageRequest {
                limit: 2,
                after: first.next,
            })
            .await
            .unwrap();
        assert_eq!(second.items.len(), 1);
        assert_eq!(second.items[0].id, "pool-c");
        assert!(second.next.is_none());
        for pool in first.items.iter().chain(&second.items) {
            assert_eq!(pool.support, PoolSupportV1::default());
            assert!(pool.policy_version_id.len() == 64);
            assert_eq!(pool.candidates[0].id, format!("{}-cheap", pool.id));
            let controls = pool.controls.as_ref().unwrap();
            assert!(controls.all.force_anchor);
            assert!(controls.effective.force_anchor);
        }
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn lost_control_or_learning_authority_is_reported_unavailable() {
        for table in ["controls", "learning_generation_state_events"] {
            let temporary = tempdir().unwrap();
            let path = temporary.path().join("router.sqlite3");
            let config = pooled_config(&path, &format!("lost-{table}"));
            drop(LedgerRepository::activate(&config).unwrap());
            let service =
                InspectionService::open(config.clone(), InspectionServiceOptions::default())
                    .await
                    .unwrap();
            let connection = rusqlite::Connection::open(&path).unwrap();
            connection
                .execute(&format!("DELETE FROM {table}"), [])
                .unwrap();
            drop(connection);

            let status = service.status().await.unwrap();
            assert_eq!(status.effective_mode, EffectiveRouterModeV1::Unavailable);
            assert!(status.controls.is_none());
            assert_eq!(status.health.worst, "unavailable");
            assert_eq!(
                service
                    .list_pools(PageRequest::default())
                    .await
                    .unwrap_err(),
                InspectionError::IntegrityError
            );
            service.close().await.unwrap();
        }
    }

    #[tokio::test]
    async fn vector_status_tracks_missing_rebuilding_active_and_stale_states() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = learning_config(&path, "vector-status");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let vector_space_id = activated.identity.pools["pool-a"]
            .vector_space
            .as_ref()
            .unwrap()
            .vector_space_id
            .clone();
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();

        let missing = service.status().await.unwrap().vector_index;
        assert_eq!(missing.active_spaces, 0);
        assert_eq!(missing.rebuilding_spaces, 0);
        assert_eq!(missing.degraded_spaces, 1);
        assert_eq!(missing.stale_spaces, 0);

        let now = i64::try_from(now_unix_ms().unwrap()).unwrap();
        let transaction = activated
            .repository
            .test_connection_mut()
            .transaction()
            .unwrap();
        let manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(2).unwrap(),
            now,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest)
            | GenerationAuthorizationAck::AlreadyExists(manifest) => manifest,
            GenerationAuthorizationAck::AuthorityMissing => panic!("vector authority missing"),
        };
        transaction.commit().unwrap();
        let rebuilding = service.status().await.unwrap().vector_index;
        assert_eq!(rebuilding.rebuilding_spaces, 1);
        assert_eq!(rebuilding.degraded_spaces, 1);

        let process_instance_id = activated.identity.process_instance_id;
        let transaction = activated
            .repository
            .test_connection_mut()
            .transaction()
            .unwrap();
        let fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, process_instance_id, now + 1)
                .unwrap()
            {
                RebuildLeaseClaimAck::Claimed(fence)
                | RebuildLeaseClaimAck::Reclaimed(fence)
                | RebuildLeaseClaimAck::AlreadyOwned(fence) => fence,
                acknowledgement => panic!("unexpected rebuild claim: {acknowledgement:?}"),
            };
        transaction.commit().unwrap();
        assert_eq!(service.status().await.unwrap().leases.active, 1);

        let transaction = activated
            .repository
            .test_connection_mut()
            .transaction()
            .unwrap();
        assert!(matches!(
            create_generation_objects(&transaction, &fence, now + 2).unwrap(),
            GenerationObjectCreationAck::Created | GenerationObjectCreationAck::AlreadyExists
        ));
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &fence, now + 3).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert!(matches!(
            catch_up_rebuild_changes(&transaction, &fence, now + 4).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &fence, now + 5).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();
        assert_eq!(manifest.vector_space_id(), &vector_space_id);

        let active = service.status().await.unwrap().vector_index;
        assert_eq!(active.active_spaces, 1);
        assert_eq!(active.rebuilding_spaces, 0);
        assert_eq!(active.degraded_spaces, 0);
        assert_eq!(active.stale_spaces, 0);
        assert_eq!(service.status().await.unwrap().leases.active, 0);

        let source_hash =
            vector_source_sequence_payload_hash(&vector_space_id, 1, now + 6).unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE vector_space_source_sequences
                 SET source_seq = 1, updated_at_unix_ms = ?1,
                     canonical_payload_hash = ?2
                 WHERE vector_space_id = ?3",
                rusqlite::params![now + 6, source_hash, vector_space_id.as_str()],
            )
            .unwrap();
        let stale = service.status().await.unwrap().vector_index;
        assert_eq!(stale.active_spaces, 1);
        assert_eq!(stale.degraded_spaces, 0);
        assert_eq!(stale.stale_spaces, 1);
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn complete_recommend_and_active_configs_report_their_real_modes() {
        for (mode, expected) in [
            (RouterMode::Recommend, EffectiveRouterModeV1::Recommend),
            (RouterMode::Active, EffectiveRouterModeV1::Active),
        ] {
            let temporary = tempdir().unwrap();
            let path = temporary.path().join("router.sqlite3");
            let project_id = match mode {
                RouterMode::Recommend => "recommend-status",
                RouterMode::Active => "active-status",
                RouterMode::Off | RouterMode::Shadow => unreachable!(),
            };
            let mut config = learning_config(&path, project_id);
            config.mode = mode;
            config.pools[0].learning = Some(complete_learning(mode == RouterMode::Active));
            config.pools[0].selector.tenant_ids = Some(vec!["tenant-a".into()]);
            config.pools[0].selector.agent_ids = Some(vec!["agent-a".into()]);
            config.pools[0].selector.owner_scope_types =
                Some(vec![nemo_relay_types::api::scope::ScopeType::Agent]);
            config.pools[0]
                .selector
                .metadata_equals
                .insert("tier".into(), json!("gold"));
            config.pools[0].selector.scope_path_patterns = Some(vec!["agent/llm/**".into()]);
            if mode == RouterMode::Active {
                config.pools[0].outcome = active_outcome_policy();
            }
            assert!(config.validate().is_empty(), "{:?}", config.validate());
            drop(LedgerRepository::activate(&config).unwrap());
            let service = InspectionService::open(config, InspectionServiceOptions::default())
                .await
                .unwrap();
            assert_eq!(service.status().await.unwrap().effective_mode, expected);
            let pools = service.list_pools(PageRequest::default()).await.unwrap();
            assert_eq!(pools.items.len(), 1);
            if mode == RouterMode::Active {
                assert_eq!(pools.items[0].active_canary_fraction, Some(0.4));
                assert_eq!(pools.items[0].holdout_probability, Some(0.2));
            } else {
                assert_eq!(pools.items[0].active_canary_fraction, None);
                assert_eq!(pools.items[0].holdout_probability, None);
            }
            let detail = service.get_pool("pool-a").await.unwrap();
            assert_eq!(detail.schema, crate::inspection::POOL_DETAIL_SCHEMA_V1);
            assert_eq!(detail.summary, pools.items[0]);
            assert_eq!(
                detail.selector.tenant_ids.as_deref(),
                Some(&["tenant-a".into()][..])
            );
            assert_eq!(detail.selector.metadata_equals["tier"], json!("gold"));
            assert_eq!(detail.concurrency.max_pending, 4);
            assert_eq!(detail.learning.as_ref().unwrap().top_k, Some(8));
            assert_eq!(detail.support_prerequisites.ready_evidence, Some(false));
            assert_eq!(detail.support_prerequisites.independent_roots, Some(false));
            assert_eq!(detail.support_prerequisites.coverage, Some(false));
            assert!(detail.support_prerequisites.query_local_evaluation_required);
            if mode == RouterMode::Recommend {
                assert_eq!(
                    service.get_pool("bad/id").await.unwrap_err(),
                    InspectionError::InvalidArgument
                );
                assert_eq!(
                    service.get_pool("missing").await.unwrap_err(),
                    InspectionError::NotFound
                );
            }
            service.close().await.unwrap();
        }
    }

    #[test]
    fn effective_mode_covers_every_configured_mode_and_control_precedence() {
        let clear = ControlStatusV1 {
            control_generation: 0,
            all: Default::default(),
            effective: Default::default(),
        };
        for (mode, expected) in [
            (RouterMode::Off, EffectiveRouterModeV1::Off),
            (RouterMode::Shadow, EffectiveRouterModeV1::Shadow),
            (RouterMode::Recommend, EffectiveRouterModeV1::Recommend),
            (RouterMode::Active, EffectiveRouterModeV1::Active),
        ] {
            assert_eq!(effective_mode(mode, Some(&clear), true), expected);
        }
        let mut force_anchor = clear.clone();
        force_anchor.effective.force_anchor = true;
        assert_eq!(
            effective_mode(RouterMode::Active, Some(&force_anchor), true),
            EffectiveRouterModeV1::ForceAnchor
        );
        force_anchor.effective.paused = true;
        assert_eq!(
            effective_mode(RouterMode::Active, Some(&force_anchor), true),
            EffectiveRouterModeV1::Paused
        );
        assert_eq!(
            effective_mode(RouterMode::Active, None, false),
            EffectiveRouterModeV1::Unavailable
        );
    }

    #[tokio::test]
    async fn tampered_control_and_health_rows_fail_deterministic_verification() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = config(&path, "tampered-reads", RouterMode::Shadow);
        let mut activated = LedgerRepository::activate(&config).unwrap();
        apply_control(
            &mut activated.repository,
            ControlMutation {
                mutation_id: Uuid::now_v7(),
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: true },
                expected_control_generation: 0,
                actor: "operator-a".into(),
                reason: "tamper fixture".into(),
            },
        );
        let event = LedgerHealthEvent::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            None,
            None,
            "router.writer.test",
            LedgerHealthSeverity::Warning,
            i64::try_from(now_unix_ms().unwrap()).unwrap(),
        )
        .unwrap();
        activated.repository.append_health_event(&event).unwrap();
        drop(activated);
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();

        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE control_mutation_receipts SET actor = 'operator-tampered'",
                [],
            )
            .unwrap();
        connection
            .execute(
                "UPDATE health_events
                 SET canonical_payload_hash =
                   '0000000000000000000000000000000000000000000000000000000000000000'",
                [],
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            service
                .list_controls(PageRequest::default())
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        assert_eq!(
            service
                .list_health(PageRequest::default())
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
    }

    #[tokio::test]
    async fn evidence_reads_verify_filters_cursors_ids_and_content_policy() {
        let (
            _temporary,
            config,
            activated,
            _vector_space_id,
            _vector_root,
            evidence_id,
            evaluation_id,
        ) = ready_evaluated_runtime_fixture();
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();

        let page = service
            .list_evidence(EvidenceFilterV1::default(), PageRequest::default())
            .await
            .unwrap();
        assert_eq!(page.content_policy, ContentPolicy::Redacted);
        assert_eq!(page.items.len(), 1);
        let summary = &page.items[0];
        assert_eq!(summary.evidence_id, evidence_id);
        assert_eq!(summary.terminal_class, "completed");
        assert_eq!(summary.quality_label.as_deref(), Some("pass"));
        assert_eq!(summary.vector_state, "ready");
        assert!(summary.content.value.is_none());
        assert!(
            summary
                .content
                .preview
                .as_deref()
                .is_some_and(|preview| preview.contains("route this request"))
        );

        let detail = service.get_evidence(evidence_id).await.unwrap();
        assert_eq!(detail.summary, *summary);
        assert_eq!(detail.evaluation_id, Some(evaluation_id));
        assert_eq!(detail.partition.candidate_id, summary.candidate_id);
        assert_eq!(
            detail.partition.learning_generation_id,
            summary.learning_generation_id
        );

        for filter in [
            EvidenceFilterV1 {
                pool_id: Some(summary.pool_id.clone()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                candidate_id: Some(summary.candidate_id.clone()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                terminal_class: Some(summary.terminal_class.clone()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                quality_label: summary.quality_label.clone(),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                learning_generation_id: Some(summary.learning_generation_id),
                ..EvidenceFilterV1::default()
            },
        ] {
            assert_eq!(
                service
                    .list_evidence(filter, PageRequest::default())
                    .await
                    .unwrap()
                    .items
                    .len(),
                1
            );
        }
        assert!(
            service
                .list_evidence(
                    EvidenceFilterV1 {
                        candidate_id: Some("not-this-candidate".into()),
                        ..EvidenceFilterV1::default()
                    },
                    PageRequest::default(),
                )
                .await
                .unwrap()
                .items
                .is_empty()
        );

        let default_filter = EvidenceFilterV1::default();
        let cursor = service
            .current()
            .unwrap()
            .cursor
            .encode(
                CursorEndpointV1::Evidence,
                &evidence_filter_hash(&default_filter).unwrap(),
                page.snapshot_time_unix_ms,
                1,
                CursorSortV1::Newest {
                    created_at_unix_ms: summary.created_at_unix_ms,
                    id: summary.evidence_id.to_string(),
                },
            )
            .unwrap();
        assert_eq!(
            service
                .list_evidence(
                    EvidenceFilterV1 {
                        candidate_id: Some(summary.candidate_id.clone()),
                        ..EvidenceFilterV1::default()
                    },
                    PageRequest {
                        limit: 1,
                        after: Some(cursor),
                    },
                )
                .await
                .unwrap_err(),
            InspectionError::InvalidCursor
        );
        assert_eq!(
            service.get_evidence(Uuid::nil()).await.unwrap_err(),
            InspectionError::InvalidArgument
        );
        assert_eq!(
            service.get_evidence(Uuid::now_v7()).await.unwrap_err(),
            InspectionError::NotFound
        );
        service.close().await.unwrap();

        let full = InspectionService::open(
            config,
            InspectionServiceOptions {
                content_policy: ContentPolicy::Full,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let detail = full.get_evidence(evidence_id).await.unwrap();
        assert!(detail.summary.content.value.is_some());
        assert_eq!(
            detail.summary.content.sha256, summary.content.sha256,
            "content policy must not change filtered content identity"
        );
        full.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn evidence_detail_rejects_a_corrupted_canonical_query_child() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _vector_root,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE canonical_routing_queries
                 SET canonical_query_json = '{}', canonical_size_bytes = 2",
                [],
            )
            .unwrap();
        assert_eq!(
            service.get_evidence(evidence_id).await.unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn retained_evidence_remains_inspectable_after_learning_reset() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _vector_root,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_runtime_fixture();
        let historical_generation = activated.identity.pools["pool-a"].learning_generation_id;
        let current_generation = activated
            .repository
            .reset_pool("pool-a", "operator", "inspection historical generation")
            .unwrap();
        assert_ne!(current_generation, historical_generation);
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let detail = service.get_evidence(evidence_id).await.unwrap();
        assert_eq!(detail.summary.learning_generation_id, historical_generation);
        assert_eq!(
            service
                .list_evidence(
                    EvidenceFilterV1 {
                        learning_generation_id: Some(historical_generation),
                        ..EvidenceFilterV1::default()
                    },
                    PageRequest::default(),
                )
                .await
                .unwrap()
                .items
                .len(),
            1
        );
        assert_eq!(
            service
                .list_pools(PageRequest::default())
                .await
                .unwrap()
                .items[0]
                .learning_generation_id,
            current_generation
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn decision_reads_verify_filters_details_and_frozen_pagination() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let first = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let second =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let mut original_ids = [first.parent.decision_id, second.parent.decision_id];
        original_ids.sort_by(|left, right| right.cmp(left));
        for audit in [&first, &second] {
            assert_eq!(
                activated
                    .repository
                    .record_decision_audit(audit, 10, Uuid::now_v7())
                    .unwrap(),
                DecisionAuditAck::Applied
            );
        }
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();

        let first_page = service
            .list_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(first_page.items.len(), 1);
        assert_eq!(first_page.items[0].decision_id, original_ids[0]);
        assert_eq!(first_page.items[0].mode, "recommend");
        assert_eq!(first_page.items[0].final_reason, "no_partition");
        let cursor = first_page.next.clone().unwrap();
        assert_eq!(
            service
                .list_decisions(
                    DecisionFilterV1 {
                        pool_id: Some(config.pools[0].id.clone()),
                        ..DecisionFilterV1::default()
                    },
                    PageRequest {
                        limit: 1,
                        after: Some(cursor.clone()),
                    },
                )
                .await
                .unwrap_err(),
            InspectionError::InvalidCursor
        );

        let appended =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&appended, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let second_page = service
            .list_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: Some(cursor),
                },
            )
            .await
            .unwrap();
        assert_eq!(second_page.items.len(), 1);
        assert_eq!(second_page.items[0].decision_id, original_ids[1]);
        assert!(second_page.next.is_none());

        let detail = service.get_decision(original_ids[0]).await.unwrap();
        assert_eq!(detail.summary.decision_id, original_ids[0]);
        assert_eq!(detail.candidates.len(), 1);
        assert_eq!(detail.candidates[0].rank_ordinal, 0);
        assert_eq!(detail.candidates[0].neighbor_count, 0);
        assert_eq!(detail.candidates[0].reason, "no_partition");
        assert!(detail.neighbors.is_empty());
        assert_eq!(detail.record_hash.len(), 64);

        for filter in [
            DecisionFilterV1 {
                pool_id: Some(config.pools[0].id.clone()),
                ..DecisionFilterV1::default()
            },
            DecisionFilterV1 {
                mode: Some("recommend".into()),
                ..DecisionFilterV1::default()
            },
            DecisionFilterV1 {
                final_reason: Some("no_partition".into()),
                ..DecisionFilterV1::default()
            },
        ] {
            assert_eq!(
                service
                    .list_decisions(filter, PageRequest::default())
                    .await
                    .unwrap()
                    .items
                    .len(),
                3
            );
        }
        assert!(
            service
                .list_decisions(
                    DecisionFilterV1 {
                        candidate_id: Some("not-selected".into()),
                        ..DecisionFilterV1::default()
                    },
                    PageRequest::default(),
                )
                .await
                .unwrap()
                .items
                .is_empty()
        );
        assert_eq!(
            service.get_decision(Uuid::nil()).await.unwrap_err(),
            InspectionError::InvalidArgument
        );
        assert_eq!(
            service.get_decision(Uuid::now_v7()).await.unwrap_err(),
            InspectionError::NotFound
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn overview_window_is_inclusive_and_recommend_exposure_is_explicitly_empty() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let audit = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let snapshot = 102_u64 + OVERVIEW_WINDOW_MS_V1;
        let inclusive = load_overview_snapshot(
            activated.repository.test_connection_mut(),
            &config,
            102,
            snapshot,
        )
        .unwrap();
        assert_eq!(inclusive.decisions.total, 1);
        assert_eq!(inclusive.decisions.recommend, 1);
        assert_eq!(inclusive.decisions.anchor_served, 1);
        assert_eq!(inclusive.exposures, Default::default());
        let excluded = load_overview_snapshot(
            activated.repository.test_connection_mut(),
            &config,
            103,
            snapshot,
        )
        .unwrap();
        assert_eq!(excluded.decisions, Default::default());

        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        let exposure = service.get_decision_exposure(decision_id).await.unwrap();
        assert_eq!(
            exposure.schema,
            crate::inspection::DECISION_EXPOSURE_SCHEMA_V1
        );
        assert_eq!(exposure.decision_id, decision_id);
        assert!(exposure.active.is_none());
        assert!(exposure.outcome.is_none());
        assert_eq!(
            service
                .get_decision_exposure(Uuid::nil())
                .await
                .unwrap_err(),
            InspectionError::InvalidArgument
        );
        assert_eq!(
            service
                .get_decision_exposure(Uuid::now_v7())
                .await
                .unwrap_err(),
            InspectionError::NotFound
        );

        activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE decisions
                 SET canonical_payload_hash =
                   '0000000000000000000000000000000000000000000000000000000000000000'
                 WHERE decision_id = ?1",
                [decision_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            service
                .get_decision_exposure(decision_id)
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn decision_tail_uses_a_distinct_forward_resumable_cursor() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let first = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let second =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        for audit in [&first, &second] {
            assert_eq!(
                activated
                    .repository
                    .record_decision_audit(audit, 10, Uuid::now_v7())
                    .unwrap(),
                DecisionAuditAck::Applied
            );
        }
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();

        let initial = service
            .tail_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(initial.items.len(), 1);
        assert_eq!(initial.items[0].decision_id, second.parent.decision_id);
        let initial_cursor = initial.next.unwrap();

        let newest_cursor = service
            .list_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: None,
                },
            )
            .await
            .unwrap()
            .next
            .unwrap();
        assert_eq!(
            service
                .tail_decisions(
                    DecisionFilterV1::default(),
                    PageRequest {
                        limit: 1,
                        after: Some(newest_cursor),
                    },
                )
                .await
                .unwrap_err(),
            InspectionError::InvalidCursor
        );
        assert_eq!(
            service
                .tail_decisions(
                    DecisionFilterV1 {
                        pool_id: Some(config.pools[0].id.clone()),
                        ..DecisionFilterV1::default()
                    },
                    PageRequest {
                        limit: 1,
                        after: Some(initial_cursor.clone()),
                    },
                )
                .await
                .unwrap_err(),
            InspectionError::InvalidCursor
        );

        let third = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let fourth =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        for audit in [&third, &fourth] {
            assert_eq!(
                activated
                    .repository
                    .record_decision_audit(audit, 10, Uuid::now_v7())
                    .unwrap(),
                DecisionAuditAck::Applied
            );
        }
        let resumed = service
            .tail_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: Some(initial_cursor),
                },
            )
            .await
            .unwrap();
        assert_eq!(resumed.items[0].decision_id, third.parent.decision_id);
        let resumed_cursor = resumed.next.unwrap();
        let final_page = service
            .tail_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: Some(resumed_cursor),
                },
            )
            .await
            .unwrap();
        assert_eq!(final_page.items[0].decision_id, fourth.parent.decision_id);
        let final_cursor = final_page.next.unwrap();
        let empty = service
            .tail_decisions(
                DecisionFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: Some(final_cursor.clone()),
                },
            )
            .await
            .unwrap();
        assert!(empty.items.is_empty());
        assert_eq!(empty.next.as_deref(), Some(final_cursor.as_str()));

        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn decision_detail_reconstructs_verified_neighbor_order() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _vector_root,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let audit = decision_test_fixtures::existing_neighbor_audit(
            &activated,
            &config,
            Uuid::now_v7(),
            evidence_id,
            50,
        );
        let decision_id = audit.parent.decision_id;
        decision_test_fixtures::insert_test_decision_graphs(
            activated.repository.test_connection_mut(),
            &[audit],
        );
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let detail = service.get_decision(decision_id).await.unwrap();
        assert_eq!(detail.candidates.len(), 1);
        assert_eq!(detail.candidates[0].neighbor_count, 1);
        assert_eq!(detail.neighbors.len(), 1);
        assert_eq!(detail.neighbors[0].neighbor_ordinal, 0);
        assert_eq!(detail.neighbors[0].evidence_id, evidence_id);
        assert_eq!(detail.neighbors[0].distance, 0.0);
        assert_eq!(detail.neighbors[0].exclusion_reason, "ineligible_quality");
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn decision_reads_reject_a_corrupted_candidate_child() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let audit = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE decision_candidate_summaries
                 SET candidate_model = 'tampered-model'
                 WHERE decision_id = ?1",
                [decision_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            service.get_decision(decision_id).await.unwrap_err(),
            InspectionError::IntegrityError
        );
        assert_eq!(
            service
                .list_decisions(DecisionFilterV1::default(), PageRequest::default())
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn outcome_reads_verify_active_joins_filters_and_frozen_pagination() {
        let mut fixture = active_test_fixtures::fixture();
        let candidate = fixture.admission.clone();
        let control =
            active_test_fixtures::new_admission(&fixture, ActiveAssignmentArm::AnchorControl);
        let candidate_outcome = seed_active_outcome(&mut fixture, candidate, 1);
        let control_outcome = seed_active_outcome(&mut fixture, control, 2);
        let original_ids = [candidate_outcome, control_outcome];
        let service =
            InspectionService::open(fixture.config.clone(), InspectionServiceOptions::default())
                .await
                .unwrap();

        let first_page = service
            .list_outcomes(
                OutcomeFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(first_page.items.len(), 1);
        assert!(original_ids.contains(&first_page.items[0].outcome_id));
        let first_seen = first_page.items[0].outcome_id;
        let cursor = first_page.next.clone().unwrap();
        assert_eq!(
            service
                .list_outcomes(
                    OutcomeFilterV1 {
                        arm: Some("candidate_treatment".into()),
                        ..OutcomeFilterV1::default()
                    },
                    PageRequest {
                        limit: 1,
                        after: Some(cursor.clone()),
                    },
                )
                .await
                .unwrap_err(),
            InspectionError::InvalidCursor
        );

        let holdout =
            active_test_fixtures::new_admission(&fixture, ActiveAssignmentArm::AnchorHoldout);
        let holdout_outcome = seed_active_outcome(&mut fixture, holdout, 3);
        let second_page = service
            .list_outcomes(
                OutcomeFilterV1::default(),
                PageRequest {
                    limit: 1,
                    after: Some(cursor),
                },
            )
            .await
            .unwrap();
        assert_eq!(second_page.items.len(), 1);
        assert_ne!(second_page.items[0].outcome_id, first_seen);
        assert!(original_ids.contains(&second_page.items[0].outcome_id));
        assert_ne!(second_page.items[0].outcome_id, holdout_outcome);
        assert!(second_page.next.is_none());

        let all = service
            .list_outcomes(OutcomeFilterV1::default(), PageRequest::default())
            .await
            .unwrap();
        assert_eq!(all.items.len(), 3);
        assert!(all.items.iter().all(|outcome| {
            outcome.label.as_deref() == Some("success") && outcome.latency_ms == 10.0
        }));
        assert_eq!(
            all.items
                .iter()
                .find(|outcome| outcome.outcome_id == candidate_outcome)
                .unwrap()
                .attribution_status,
            "eligible_treatment"
        );
        assert_eq!(
            all.items
                .iter()
                .find(|outcome| outcome.outcome_id == control_outcome)
                .unwrap()
                .attribution_status,
            "eligible_control"
        );
        assert_eq!(
            all.items
                .iter()
                .find(|outcome| outcome.outcome_id == holdout_outcome)
                .unwrap()
                .attribution_status,
            "monitoring_only"
        );

        for (filter, expected) in [
            (
                OutcomeFilterV1 {
                    pool_id: Some("pool-a".into()),
                    ..OutcomeFilterV1::default()
                },
                3,
            ),
            (
                OutcomeFilterV1 {
                    arm: Some("candidate_treatment".into()),
                    ..OutcomeFilterV1::default()
                },
                1,
            ),
            (
                OutcomeFilterV1 {
                    label: Some("success".into()),
                    ..OutcomeFilterV1::default()
                },
                3,
            ),
            (
                OutcomeFilterV1 {
                    attribution_status: Some("eligible_control".into()),
                    ..OutcomeFilterV1::default()
                },
                1,
            ),
        ] {
            assert_eq!(
                service
                    .list_outcomes(filter, PageRequest::default())
                    .await
                    .unwrap()
                    .items
                    .len(),
                expected
            );
        }
        service.close().await.unwrap();
        drop(fixture);
    }

    #[tokio::test]
    async fn outcome_reads_reject_a_corrupted_outcome_hash() {
        let mut fixture = active_test_fixtures::fixture();
        let admission = fixture.admission.clone();
        let outcome_id = seed_active_outcome(&mut fixture, admission, 1);
        let service =
            InspectionService::open(fixture.config.clone(), InspectionServiceOptions::default())
                .await
                .unwrap();
        fixture
            .activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE outcomes
                 SET canonical_payload_hash =
                   '0000000000000000000000000000000000000000000000000000000000000000'
                 WHERE outcome_id = ?1",
                [outcome_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            service
                .list_outcomes(OutcomeFilterV1::default(), PageRequest::default())
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
        drop(fixture);
    }
}
