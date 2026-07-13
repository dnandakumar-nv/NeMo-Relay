// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use uuid::Uuid;

use super::{InspectionService, inspection_deadline, map_ledger_error, map_read_pool_error};
use crate::embedder::{
    EmbedderBatchItem, EmbedderFailure, EmbedderFailureClass, EmbedderWorkKind,
    FrozenEmbedderClients, build_frozen_embedder_clients,
};
use crate::inspection::request::{PreparedRequestInspection, prepare_request_inspection};
use crate::inspection::{InspectionError, NeighborhoodReportV1, RoutingInspectionInputV1};
use crate::ledger::repository::inspection::{PersistedNeighborhoodRead, load_request_neighborhood};
use crate::vector::AuthoritativeVector;

impl InspectionService {
    pub(super) async fn inspect_request_neighborhood(
        &self,
        input: RoutingInspectionInputV1,
    ) -> Result<NeighborhoodReportV1, InspectionError> {
        let current = self.current()?;
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let snapshot_time_unix_ms = super::reads::now_unix_ms()?;
        let prepared = prepare_request_inspection(&self.inner.config, &current.authority, input)?;
        match self
            .load_prepared_request(&prepared, None, snapshot_time_unix_ms, deadline)
            .await?
        {
            PersistedNeighborhoodRead::Report(report) => return Ok(*report),
            PersistedNeighborhoodRead::InvalidArgument => {
                return Err(InspectionError::InvalidArgument);
            }
            PersistedNeighborhoodRead::NotFound => return Err(InspectionError::IntegrityError),
            PersistedNeighborhoodRead::NeedsEmbedding => {}
        }
        if !self.inner.options.allow_request_embedding {
            return Err(InspectionError::EgressDenied);
        }
        let cancellation = self.inner.cancellation.subscribe();
        if *cancellation.borrow() {
            return Err(InspectionError::StorageUnavailable);
        }
        let clients = self.request_embedder_clients(&prepared)?;
        let permit = clients
            .try_acquire(&prepared.vector_space_id)
            .map_err(|failure| map_embedder_failure(failure, deadline))?;
        let canonical_query = String::from_utf8(prepared.canonical_query.canonical_bytes.clone())
            .map_err(|_| InspectionError::IntegrityError)?;
        let item = EmbedderBatchItem::new(
            prepared.canonical_query.canonical_query_hash.clone(),
            canonical_query,
            Uuid::now_v7().to_string(),
        )
        .map_err(|_| InspectionError::IntegrityError)?;
        let mut vectors = permit
            .execute_batch_until(
                EmbedderWorkKind::Inspection,
                vec![item],
                deadline,
                cancellation,
            )
            .await
            .map_err(|failure| map_embedder_failure(failure, deadline))?;
        if vectors.len() != 1 {
            return Err(InspectionError::IntegrityError);
        }
        let vector = vectors.pop().ok_or(InspectionError::IntegrityError)?;
        match self
            .load_prepared_request(&prepared, Some(vector), snapshot_time_unix_ms, deadline)
            .await?
        {
            PersistedNeighborhoodRead::Report(report) => Ok(*report),
            PersistedNeighborhoodRead::InvalidArgument => Err(InspectionError::InvalidArgument),
            PersistedNeighborhoodRead::NotFound | PersistedNeighborhoodRead::NeedsEmbedding => {
                Err(InspectionError::IntegrityError)
            }
        }
    }

    async fn load_prepared_request(
        &self,
        prepared: &PreparedRequestInspection,
        provided_vector: Option<AuthoritativeVector>,
        snapshot_time_unix_ms: u64,
        deadline: Instant,
    ) -> Result<PersistedNeighborhoodRead, InspectionError> {
        let config = Arc::clone(&self.inner.config);
        let read_pool = self.current()?.read_pool.clone();
        let pool_id = prepared.pool_id.clone();
        let canonical_query_hash = prepared.canonical_query.canonical_query_hash.clone();
        let partition = prepared.partition.clone();
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_request_neighborhood(
                    connection,
                    &config,
                    pool_id,
                    canonical_query_hash,
                    partition,
                    provided_vector,
                    snapshot_time_unix_ms,
                ))
            })
            .await
        {
            Ok(Ok(read)) => Ok(read),
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }

    fn request_embedder_clients(
        &self,
        prepared: &PreparedRequestInspection,
    ) -> Result<Arc<FrozenEmbedderClients>, InspectionError> {
        let mut clients = self
            .inner
            .embedder_clients
            .lock()
            .map_err(|_| InspectionError::IntegrityError)?;
        if let Some(existing) = clients.get(&prepared.embedder_profile_version_id) {
            return Ok(Arc::clone(existing));
        }
        let built = Arc::new(
            build_frozen_embedder_clients(&self.inner.config, &prepared.registry)
                .map_err(|_| InspectionError::StorageUnavailable)?,
        );
        if built.profile_count() != 1 || built.space_count() != prepared.registry.spaces.len() {
            return Err(InspectionError::IntegrityError);
        }
        clients.insert(
            prepared.embedder_profile_version_id.clone(),
            Arc::clone(&built),
        );
        Ok(built)
    }
}

fn map_embedder_failure(failure: EmbedderFailure, deadline: Instant) -> InspectionError {
    if failure.class() == EmbedderFailureClass::Saturated || Instant::now() >= deadline {
        InspectionError::Busy
    } else {
        InspectionError::StorageUnavailable
    }
}
