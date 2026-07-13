// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded discovery of durable vector-index recovery work.

use rusqlite::{Transaction, params};
use uuid::Uuid;

use super::map_sqlite_error;
use super::process::{ProcessStatusAt, verified_process_status_at};
use super::vector_index::{
    GenerationObjectsStatus, RebuildLeaseFence, ValidatedVectorIndexManifest,
    VectorIndexManifestState, authoritative_source_fingerprint, generation_source_fingerprint,
    load_validated_manifest, load_validated_rebuild_lease, verify_generation_objects,
};
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::sqlite_vec_extension::{SqliteVecStatus, verify_connection as verify_sqlite_vec};
use crate::sqlite_vec_schema::VectorIndexGeneration;
use crate::vector::{VectorDimensions, VectorSpaceId};

pub(crate) const VECTOR_GENERATION_WORK_PAGE_MAX: usize = 256;

/// Stable keyset position for a project-wide vector-space inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorSpaceWorkCursor {
    pub(crate) vector_space_id: VectorSpaceId,
}

/// Why one space needs a new source-only generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorSpaceRecoveryReason {
    MissingCurrent,
    MissingObjects,
    PartialObjects,
    Unavailable,
    Corrupt,
    FingerprintMismatch,
}

/// One inspected space. Healthy and already-building entries carry no recovery reason
/// so callers can still advance a bounded cursor without scheduling work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorSpaceInspection {
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) dimensions: VectorDimensions,
    pub(crate) recovery: Option<VectorSpaceRecoveryReason>,
}

impl VectorSpaceInspection {
    pub(crate) fn cursor(&self) -> VectorSpaceWorkCursor {
        VectorSpaceWorkCursor {
            vector_space_id: self.vector_space_id.clone(),
        }
    }
}

/// Stable keyset position shared by building and retired generation scans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorGenerationWorkCursor {
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) generation: VectorIndexGeneration,
}

/// Whether a building generation can be claimed at the selector's frozen timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RebuildLeaseDisposition {
    Unclaimed,
    Owned,
    Held,
    Reclaimable,
}

/// One fully validated building generation and its current lease authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuildingGenerationWork {
    pub(crate) manifest: ValidatedVectorIndexManifest,
    pub(crate) objects: GenerationObjectsStatus,
    pub(crate) lease: Option<RebuildLeaseFence>,
    pub(crate) lease_disposition: RebuildLeaseDisposition,
}

impl BuildingGenerationWork {
    pub(crate) fn cursor(&self) -> VectorGenerationWorkCursor {
        cursor_for(&self.manifest)
    }
}

/// One fully validated retired generation awaiting exact-object cleanup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RetiredGenerationWork {
    pub(crate) manifest: ValidatedVectorIndexManifest,
    pub(crate) objects: GenerationObjectsStatus,
}

impl RetiredGenerationWork {
    pub(crate) fn cursor(&self) -> VectorGenerationWorkCursor {
        cursor_for(&self.manifest)
    }
}

/// Select one bounded keyset page of building generations across this project.
pub(crate) fn select_building_generations(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
    after: Option<&VectorGenerationWorkCursor>,
    limit: usize,
) -> Result<Vec<BuildingGenerationWork>, LedgerError> {
    validate_selection_authority(
        transaction,
        project_uuid,
        process_instance_id,
        observed_at_unix_ms,
        limit,
    )?;
    let keys = select_generation_keys(transaction, project_uuid, "building", after, limit)?;
    keys.into_iter()
        .map(|(vector_space_id, generation)| {
            let manifest = load_exact_manifest(
                transaction,
                &vector_space_id,
                generation,
                VectorIndexManifestState::Building,
            )?;
            let objects = verify_generation_objects(transaction, manifest.authority())?;
            let lease = load_validated_rebuild_lease(transaction, &vector_space_id)?;
            let lease_disposition = classify_lease(
                transaction,
                project_uuid,
                process_instance_id,
                observed_at_unix_ms,
                lease.as_ref(),
            )?;
            Ok(BuildingGenerationWork {
                manifest,
                objects,
                lease,
                lease_disposition,
            })
        })
        .collect()
}

/// Select one bounded keyset page of retired generations across this project.
pub(crate) fn select_retired_generations(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
    after: Option<&VectorGenerationWorkCursor>,
    limit: usize,
) -> Result<Vec<RetiredGenerationWork>, LedgerError> {
    validate_selection_authority(
        transaction,
        project_uuid,
        process_instance_id,
        observed_at_unix_ms,
        limit,
    )?;
    let keys = select_generation_keys(transaction, project_uuid, "retired", after, limit)?;
    keys.into_iter()
        .map(|(vector_space_id, generation)| {
            let manifest = load_exact_manifest(
                transaction,
                &vector_space_id,
                generation,
                VectorIndexManifestState::Retired,
            )?;
            let objects = verify_generation_objects(transaction, manifest.authority())?;
            Ok(RetiredGenerationWork { manifest, objects })
        })
        .collect()
}

/// Inspect one bounded keyset page of all project spaces for rebuild authority.
pub(crate) fn inspect_vector_spaces(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
    after: Option<&VectorSpaceWorkCursor>,
    limit: usize,
) -> Result<Vec<VectorSpaceInspection>, LedgerError> {
    validate_selection_authority(
        transaction,
        project_uuid,
        process_instance_id,
        observed_at_unix_ms,
        limit,
    )?;
    let mut statement = transaction
        .prepare(
            "SELECT vector_space_id, dimensions
             FROM vector_spaces
             WHERE project_uuid = ?1 AND vector_space_id > COALESCE(?2, '')
             ORDER BY vector_space_id
             LIMIT ?3",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(
            params![
                project_uuid.to_string(),
                after.map(|cursor| cursor.vector_space_id.as_str()),
                i64::try_from(limit)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);

    rows.into_iter()
        .map(|(vector_space_id, dimensions)| {
            let vector_space_id = VectorSpaceId::new(vector_space_id).map_err(|_| corrupt())?;
            let dimensions = u32::try_from(dimensions)
                .ok()
                .and_then(|value| VectorDimensions::new(value).ok())
                .ok_or_else(corrupt)?;
            let building = select_manifest_generations(
                transaction,
                &vector_space_id,
                &[VectorIndexManifestState::Building],
            )?;
            if building.len() > 1 {
                return Err(corrupt());
            }
            if !building.is_empty() {
                return Ok(VectorSpaceInspection {
                    vector_space_id,
                    dimensions,
                    recovery: None,
                });
            }
            let current = select_manifest_generations(
                transaction,
                &vector_space_id,
                &[
                    VectorIndexManifestState::Active,
                    VectorIndexManifestState::Unavailable,
                    VectorIndexManifestState::Corrupt,
                ],
            )?;
            if current.len() > 1 {
                return Err(corrupt());
            }
            let recovery = match current.first() {
                None => Some(VectorSpaceRecoveryReason::MissingCurrent),
                Some(manifest) => {
                    match verify_generation_objects(transaction, manifest.authority())? {
                        GenerationObjectsStatus::Missing => {
                            Some(VectorSpaceRecoveryReason::MissingObjects)
                        }
                        GenerationObjectsStatus::Partial => {
                            Some(VectorSpaceRecoveryReason::PartialObjects)
                        }
                        GenerationObjectsStatus::Complete => match manifest.state() {
                            VectorIndexManifestState::Unavailable => {
                                Some(VectorSpaceRecoveryReason::Unavailable)
                            }
                            VectorIndexManifestState::Corrupt => {
                                Some(VectorSpaceRecoveryReason::Corrupt)
                            }
                            VectorIndexManifestState::Active => {
                                if verify_sqlite_vec(transaction) != SqliteVecStatus::Available {
                                    Some(VectorSpaceRecoveryReason::Unavailable)
                                } else {
                                    let authoritative = authoritative_source_fingerprint(
                                        transaction,
                                        &vector_space_id,
                                    )?;
                                    let indexed =
                                        generation_source_fingerprint(transaction, manifest)?;
                                    (indexed.as_ref() != Some(&authoritative))
                                        .then_some(VectorSpaceRecoveryReason::FingerprintMismatch)
                                }
                            }
                            VectorIndexManifestState::Building
                            | VectorIndexManifestState::Retired
                            | VectorIndexManifestState::Dropped => return Err(corrupt()),
                        },
                    }
                }
            };
            Ok(VectorSpaceInspection {
                vector_space_id,
                dimensions,
                recovery,
            })
        })
        .collect()
}

fn select_manifest_generations(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    states: &[VectorIndexManifestState],
) -> Result<Vec<ValidatedVectorIndexManifest>, LedgerError> {
    let state_names = states
        .iter()
        .map(|state| state.as_str())
        .collect::<Vec<_>>();
    let mut statement = transaction
        .prepare(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1
               AND state IN (?2, ?3, ?4)
             ORDER BY generation",
        )
        .map_err(database_error)?;
    let names = [
        state_names.first().copied().unwrap_or("__none__"),
        state_names.get(1).copied().unwrap_or("__none__"),
        state_names.get(2).copied().unwrap_or("__none__"),
    ];
    let generations = statement
        .query_map(
            params![vector_space_id.as_str(), names[0], names[1], names[2]],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);
    generations
        .into_iter()
        .map(|generation| {
            let generation = VectorIndexGeneration::new(generation).map_err(|_| corrupt())?;
            load_validated_manifest(transaction, vector_space_id, generation)?.ok_or_else(corrupt)
        })
        .collect()
}

fn validate_selection_authority(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
    limit: usize,
) -> Result<(), LedgerError> {
    if !(1..=VECTOR_GENERATION_WORK_PAGE_MAX).contains(&limit)
        || verified_process_status_at(
            transaction,
            project_uuid,
            process_instance_id,
            observed_at_unix_ms,
        )? != ProcessStatusAt::Live
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn select_generation_keys(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    state: &str,
    after: Option<&VectorGenerationWorkCursor>,
    limit: usize,
) -> Result<Vec<(VectorSpaceId, VectorIndexGeneration)>, LedgerError> {
    let after_space = after.map(|cursor| cursor.vector_space_id.as_str());
    let after_generation = after.map(|cursor| cursor.generation.value());
    let limit =
        i64::try_from(limit).map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut statement = transaction
        .prepare(
            "SELECT manifest.vector_space_id, manifest.generation
             FROM vector_index_manifest AS manifest
             JOIN vector_spaces AS space
               ON space.vector_space_id = manifest.vector_space_id
             WHERE space.project_uuid = ?1 AND manifest.state = ?2
               AND (
                    ?3 IS NULL OR manifest.vector_space_id > ?3
                    OR (manifest.vector_space_id = ?3 AND manifest.generation > ?4)
               )
             ORDER BY manifest.vector_space_id, manifest.generation
             LIMIT ?5",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(
            params![
                project_uuid.to_string(),
                state,
                after_space,
                after_generation,
                limit,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?;
    rows.map(|row| {
        let (vector_space_id, generation) = row.map_err(database_error)?;
        let vector_space_id = VectorSpaceId::new(vector_space_id).map_err(|_| corrupt())?;
        let generation = VectorIndexGeneration::new(generation).map_err(|_| corrupt())?;
        Ok((vector_space_id, generation))
    })
    .collect()
}

fn load_exact_manifest(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    generation: VectorIndexGeneration,
    expected_state: VectorIndexManifestState,
) -> Result<ValidatedVectorIndexManifest, LedgerError> {
    let manifest =
        load_validated_manifest(transaction, vector_space_id, generation)?.ok_or_else(corrupt)?;
    if manifest.state() != expected_state {
        return Err(corrupt());
    }
    Ok(manifest)
}

fn classify_lease(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
    lease: Option<&RebuildLeaseFence>,
) -> Result<RebuildLeaseDisposition, LedgerError> {
    let Some(lease) = lease else {
        return Ok(RebuildLeaseDisposition::Unclaimed);
    };
    let owner_status = verified_process_status_at(
        transaction,
        project_uuid,
        lease.owner_process_instance_id(),
        observed_at_unix_ms,
    )?;
    if owner_status == ProcessStatusAt::Invalid {
        return Err(corrupt());
    }
    if lease.lease_expires_at_unix_ms() <= observed_at_unix_ms
        || matches!(
            owner_status,
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal
        )
    {
        return Ok(RebuildLeaseDisposition::Reclaimable);
    }
    Ok(
        if lease.owner_process_instance_id() == process_instance_id {
            RebuildLeaseDisposition::Owned
        } else {
            RebuildLeaseDisposition::Held
        },
    )
}

fn cursor_for(manifest: &ValidatedVectorIndexManifest) -> VectorGenerationWorkCursor {
    VectorGenerationWorkCursor {
        vector_space_id: manifest.vector_space_id().clone(),
        generation: manifest.generation(),
    }
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::config::{LearningConfig, RouterConfig};
    use crate::ledger::repository::process::{HeartbeatRenewal, ProcessStop};
    use crate::ledger::repository::tests::{config, database_path};
    use crate::ledger::repository::vector_index::{
        GenerationAuthorizationAck, GenerationObjectCreationAck, RebuildFlipAck,
        RebuildLeaseClaimAck, RebuildStepAck, authorize_generation, catch_up_rebuild_changes,
        claim_rebuild_lease, create_generation_objects, flip_rebuild_generation,
        populate_rebuild_chunk,
    };
    use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
    use crate::vector::VectorDimensions;

    fn learning_config(path: &Path, pool_count: usize) -> RouterConfig {
        let mut config = config(path, "vector-work-tests");
        config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        for index in 1..pool_count {
            let mut pool = config.pools[0].clone();
            pool.id = format!("pool-{index}");
            pool.anchor_models = vec![format!("anchor-{index}")];
            pool.candidates[0].id = format!("candidate-{index}");
            pool.candidates[0].model = format!("candidate-model-{index}");
            pool.canonicalizer.max_task_bytes += index;
            config.pools.push(pool);
        }
        config
    }

    fn activate(pool_count: usize, at: i64) -> (TempDir, RouterConfig, ActivatedLedger) {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = learning_config(&path, pool_count);
        let activated = LedgerRepository::activate_at(&config, at).unwrap();
        (temporary, config, activated)
    }

    fn vector_spaces(activated: &ActivatedLedger) -> Vec<VectorSpaceId> {
        let mut spaces = activated
            .identity
            .pools
            .values()
            .map(|pool| pool.vector_space.as_ref().unwrap().vector_space_id.clone())
            .collect::<Vec<_>>();
        spaces.sort();
        spaces.dedup();
        spaces
    }

    fn authorize(
        activated: &mut ActivatedLedger,
        vector_space_id: &VectorSpaceId,
        created_at_unix_ms: i64,
    ) -> ValidatedVectorIndexManifest {
        let transaction = activated.repository.connection.transaction().unwrap();
        let acknowledgement = authorize_generation(
            &transaction,
            vector_space_id,
            VectorDimensions::new(1024).unwrap(),
            created_at_unix_ms,
        )
        .unwrap();
        transaction.commit().unwrap();
        match acknowledgement {
            GenerationAuthorizationAck::Created(manifest)
            | GenerationAuthorizationAck::AlreadyExists(manifest) => manifest,
            GenerationAuthorizationAck::AuthorityMissing => {
                panic!("fixture space must have initial authorization authority")
            }
        }
    }

    fn select_building(
        activated: &ActivatedLedger,
        observed_at_unix_ms: i64,
        after: Option<&VectorGenerationWorkCursor>,
        limit: usize,
    ) -> Result<Vec<BuildingGenerationWork>, LedgerError> {
        let transaction = activated
            .repository
            .connection
            .unchecked_transaction()
            .unwrap();
        let result = select_building_generations(
            &transaction,
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            observed_at_unix_ms,
            after,
            limit,
        );
        transaction.commit().unwrap();
        result
    }

    fn select_retired(
        activated: &ActivatedLedger,
        observed_at_unix_ms: i64,
        after: Option<&VectorGenerationWorkCursor>,
        limit: usize,
    ) -> Result<Vec<RetiredGenerationWork>, LedgerError> {
        let transaction = activated
            .repository
            .connection
            .unchecked_transaction()
            .unwrap();
        let result = select_retired_generations(
            &transaction,
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            observed_at_unix_ms,
            after,
            limit,
        );
        transaction.commit().unwrap();
        result
    }

    fn inspect_spaces(
        activated: &ActivatedLedger,
        observed_at_unix_ms: i64,
        after: Option<&VectorSpaceWorkCursor>,
        limit: usize,
    ) -> Result<Vec<VectorSpaceInspection>, LedgerError> {
        let transaction = activated
            .repository
            .connection
            .unchecked_transaction()
            .unwrap();
        let result = inspect_vector_spaces(
            &transaction,
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            observed_at_unix_ms,
            after,
            limit,
        );
        transaction.commit().unwrap();
        result
    }

    fn finish_empty_rebuild(
        activated: &mut ActivatedLedger,
        vector_space_id: &VectorSpaceId,
        started_at_unix_ms: i64,
    ) {
        let process_instance_id = activated.identity.process_instance_id;
        let manifest = authorize(activated, vector_space_id, started_at_unix_ms);
        let transaction = activated.repository.connection.transaction().unwrap();
        let fence = match claim_rebuild_lease(
            &transaction,
            vector_space_id,
            process_instance_id,
            started_at_unix_ms + 1,
        )
        .unwrap()
        {
            RebuildLeaseClaimAck::Claimed(fence)
            | RebuildLeaseClaimAck::Reclaimed(fence)
            | RebuildLeaseClaimAck::AlreadyOwned(fence) => fence,
            acknowledgement => panic!("unexpected lease acknowledgement: {acknowledgement:?}"),
        };
        assert!(matches!(
            create_generation_objects(&transaction, &fence, started_at_unix_ms + 2).unwrap(),
            GenerationObjectCreationAck::Created | GenerationObjectCreationAck::AlreadyExists
        ));
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &fence, started_at_unix_ms + 3).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert!(matches!(
            catch_up_rebuild_changes(&transaction, &fence, started_at_unix_ms + 4).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &fence, started_at_unix_ms + 5).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();
        assert_eq!(manifest.vector_space_id(), vector_space_id);
    }

    #[test]
    fn building_selector_is_bounded_keyset_ordered_and_authorized() {
        let (_temporary, _config, mut activated) = activate(3, 0);
        let spaces = vector_spaces(&activated);
        assert_eq!(spaces.len(), 3);
        for (index, space) in spaces.iter().enumerate() {
            authorize(&mut activated, space, i64::try_from(index + 1).unwrap());
        }

        let first = select_building(&activated, 10, None, 2).unwrap();
        assert_eq!(first.len(), 2);
        assert!(first.iter().all(|work| {
            work.objects == GenerationObjectsStatus::Missing
                && work.lease.is_none()
                && work.lease_disposition == RebuildLeaseDisposition::Unclaimed
        }));
        let second = select_building(&activated, 10, Some(&first[1].cursor()), 2).unwrap();
        assert_eq!(second.len(), 1);
        assert!(first[1].cursor().vector_space_id < second[0].cursor().vector_space_id);

        assert_eq!(
            select_building(&activated, 10, None, 0)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        assert_eq!(
            select_building(&activated, 10, None, VECTOR_GENERATION_WORK_PAGE_MAX + 1,)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );

        let transaction = activated
            .repository
            .connection
            .unchecked_transaction()
            .unwrap();
        let error = select_building_generations(
            &transaction,
            Uuid::now_v7(),
            activated.identity.process_instance_id,
            10,
            None,
            1,
        )
        .unwrap_err();
        transaction.commit().unwrap();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
    }

    #[test]
    fn building_selector_classifies_owned_held_and_reclaimable_leases() {
        let (_temporary, config, mut first) = activate(1, 0);
        let space = vector_spaces(&first).pop().unwrap();
        authorize(&mut first, &space, 1);
        let first_process = first.identity.process_instance_id;
        let transaction = first.repository.connection.transaction().unwrap();
        let fence = match claim_rebuild_lease(&transaction, &space, first_process, 2).unwrap() {
            RebuildLeaseClaimAck::Claimed(fence) => fence,
            acknowledgement => panic!("unexpected lease acknowledgement: {acknowledgement:?}"),
        };
        transaction.commit().unwrap();
        assert_eq!(
            select_building(&first, 10, None, 1).unwrap()[0].lease_disposition,
            RebuildLeaseDisposition::Owned
        );

        let mut second = LedgerRepository::activate_at(&config, 10).unwrap();
        assert_eq!(
            select_building(&second, 10, None, 1).unwrap()[0].lease_disposition,
            RebuildLeaseDisposition::Held
        );
        second
            .repository
            .renew_heartbeat(HeartbeatRenewal::new(31_000).unwrap())
            .unwrap();
        let work = select_building(&second, 31_000, None, 1).unwrap();
        assert_eq!(work[0].lease.as_ref().unwrap(), &fence);
        assert_eq!(
            work[0].lease_disposition,
            RebuildLeaseDisposition::Reclaimable
        );
    }

    #[test]
    fn retired_selector_validates_manifest_and_generated_object_status() {
        let (_temporary, _config, mut activated) = activate(1, 0);
        let space = vector_spaces(&activated).pop().unwrap();
        finish_empty_rebuild(&mut activated, &space, 1);
        finish_empty_rebuild(&mut activated, &space, 10);
        finish_empty_rebuild(&mut activated, &space, 20);

        let first = select_retired(&activated, 29, None, 1).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].manifest.generation().value(), 1);
        assert_eq!(first[0].objects, GenerationObjectsStatus::Complete);
        assert_eq!(first[0].cursor().vector_space_id, space);
        let second = select_retired(&activated, 29, Some(&first[0].cursor()), 1).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].manifest.generation().value(), 2);

        activated
            .repository
            .connection
            .execute(
                "UPDATE vector_index_manifest SET canonical_payload_hash = ?1
                 WHERE vector_space_id = ?2 AND generation = 1",
                params!["f".repeat(64), space.as_str()],
            )
            .unwrap();
        assert_eq!(
            select_retired(&activated, 29, None, VECTOR_GENERATION_WORK_PAGE_MAX,)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn building_selector_reports_partial_authorized_objects() {
        let (_temporary, _config, mut activated) = activate(1, 0);
        let space = vector_spaces(&activated).pop().unwrap();
        let manifest = authorize(&mut activated, &space, 1);
        activated
            .repository
            .connection
            .execute_batch(manifest.authority().objects()[1].sql())
            .unwrap();
        assert_eq!(
            select_building(&activated, 10, None, 1).unwrap()[0].objects,
            GenerationObjectsStatus::Partial
        );
    }

    #[test]
    fn selector_rejects_a_terminal_current_process() {
        let (_temporary, _config, mut activated) = activate(1, 0);
        let space = vector_spaces(&activated).pop().unwrap();
        authorize(&mut activated, &space, 1);
        activated
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 2).unwrap())
            .unwrap();
        assert_eq!(
            select_building(&activated, 3, None, 1).unwrap_err().class(),
            LedgerErrorClass::IdentityInvariant
        );
    }

    #[test]
    fn space_inspection_bootstraps_then_ignores_building_and_healthy_generations() {
        let (_temporary, _config, mut activated) = activate(1, 0);
        let space = vector_spaces(&activated).pop().unwrap();
        let initial = inspect_spaces(&activated, 1, None, 1).unwrap();
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].vector_space_id, space);
        assert_eq!(initial[0].dimensions, VectorDimensions::new(1024).unwrap());
        assert_eq!(
            initial[0].recovery,
            Some(VectorSpaceRecoveryReason::MissingCurrent)
        );

        authorize(&mut activated, &space, 2);
        assert_eq!(
            inspect_spaces(&activated, 3, None, 1).unwrap()[0].recovery,
            None
        );

        finish_empty_rebuild(&mut activated, &space, 4);
        assert_eq!(
            inspect_spaces(&activated, 10, None, 1).unwrap()[0].recovery,
            None
        );
    }

    #[test]
    fn space_inspection_is_keyset_bounded_and_reports_active_object_loss() {
        let (_temporary, _config, mut activated) = activate(3, 0);
        let spaces = vector_spaces(&activated);
        let first = inspect_spaces(&activated, 1, None, 2).unwrap();
        assert_eq!(first.len(), 2);
        let second = inspect_spaces(&activated, 1, Some(&first[1].cursor()), 2).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].vector_space_id, spaces[2]);

        let lost = spaces[0].clone();
        finish_empty_rebuild(&mut activated, &lost, 2);
        let root = crate::sqlite_vec_schema::Vec0RootName::new(
            lost.clone(),
            VectorIndexGeneration::new(1).unwrap(),
        );
        activated
            .repository
            .connection
            .execute_batch(&format!("DROP TABLE \"{}\"", root.as_str()))
            .unwrap();
        let inspected = inspect_spaces(&activated, 10, None, 3).unwrap();
        let lost = inspected
            .iter()
            .find(|inspection| inspection.vector_space_id == lost)
            .unwrap();
        assert_eq!(
            lost.recovery,
            Some(VectorSpaceRecoveryReason::MissingObjects)
        );
    }
}
