// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only verified authority used by the public inspection service.

mod neighborhood;
pub(crate) mod operator;
mod read;

pub(crate) use neighborhood::{
    PersistedNeighborhoodRead, load_persisted_neighborhood, load_request_neighborhood,
};
pub(crate) use read::{
    load_control_page, load_decision_detail, load_decision_exposure, load_decision_follow_page,
    load_decision_page, load_evidence_detail, load_evidence_page, load_health_page,
    load_migration_page, load_outcome_page, load_overview_snapshot, load_pool_detail,
    load_pool_page, load_status_snapshot,
};

use std::collections::BTreeMap;

use rusqlite::{Connection, Transaction};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::control::{control_config_is_current, load_control_authority_snapshot};
use super::vector_registry::{
    FrozenMappingKey, FrozenPoolVectorAuthority, resolve_frozen_pool_vector_authority,
};
use super::{PreparedLedgerMaterial, initialize_project, inspect_existing_database};
use crate::config::{RouterConfig, RouterMode};
use crate::control::RouterControlSnapshot;
use crate::inspection::cursor::derive_cursor_key;
use crate::ledger::model::{LedgerError, LedgerErrorClass};

#[derive(Clone)]
pub(crate) struct InspectionAuthoritySnapshot {
    pub(crate) project_uuid: Uuid,
    pub(crate) project_id: String,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_ids: BTreeMap<String, String>,
    pub(crate) learning_generation_ids: BTreeMap<String, Uuid>,
    pub(crate) cohort_generation_id: Uuid,
    pub(crate) control_snapshot: Option<RouterControlSnapshot>,
    pub(crate) vector_authorities: BTreeMap<String, FrozenPoolVectorAuthority>,
    pub(crate) cursor_key: Zeroizing<[u8; 32]>,
}

pub(crate) fn load_inspection_authority_snapshot(
    connection: &Connection,
    config: &RouterConfig,
) -> Result<InspectionAuthoritySnapshot, LedgerError> {
    inspect_existing_database(connection)?;
    let transaction = connection
        .unchecked_transaction()
        .map_err(|_| LedgerError::new(LedgerErrorClass::DatabaseOperationFailed))?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    transaction
        .commit()
        .map_err(|_| LedgerError::new(LedgerErrorClass::DatabaseOperationFailed))?;
    Ok(authority)
}

fn load_inspection_authority_in_transaction(
    connection: &Transaction<'_>,
    config: &RouterConfig,
) -> Result<InspectionAuthoritySnapshot, LedgerError> {
    let expected_config_generation_id = config
        .config_generation_id()
        .map_err(|_| LedgerErrorClass::IdentityInvariant)?;
    let prepared = PreparedLedgerMaterial::from_config(config)?;
    if prepared.config_generation_id != expected_config_generation_id {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let (project_uuid, project_id, created) = initialize_project(connection, config, 0, false)?;
    if created {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    verify_config_and_policy_authority(connection, project_uuid, &prepared)?;
    if config.mode != RouterMode::Off
        && !control_config_is_current(connection, project_uuid, &expected_config_generation_id)?
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let policy_version_ids = prepared
        .policies
        .iter()
        .map(|(pool_id, policy)| (pool_id.clone(), policy.policy_version_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let pool_ids = policy_version_ids.keys().cloned().collect::<Vec<_>>();
    let operator::CurrentGenerationAuthority {
        learning_generation_ids,
        cohort_generation_id,
        cohort_salt,
    } = operator::load_current_generation_authority(connection, project_uuid, &pool_ids)?;
    let control_snapshot = if config.mode == RouterMode::Off {
        None
    } else {
        Some(
            load_control_authority_snapshot(
                connection,
                project_uuid,
                &expected_config_generation_id,
                &pool_ids,
            )?
            .snapshot,
        )
    };
    if learning_generation_ids.len() != pool_ids.len()
        || control_snapshot.as_ref().is_some_and(|snapshot| {
            snapshot.pools.len() != learning_generation_ids.len()
                || learning_generation_ids
                    .iter()
                    .any(|(pool_id, generation_id)| {
                        snapshot
                            .pools
                            .get(pool_id)
                            .is_none_or(|pool| pool.learning_generation_id != *generation_id)
                    })
        })
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut vector_authorities = BTreeMap::new();
    for (pool_id, policy_version_id) in &policy_version_ids {
        let key = FrozenMappingKey::new(
            project_uuid,
            expected_config_generation_id.clone(),
            pool_id.clone(),
            policy_version_id.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        vector_authorities.insert(
            pool_id.clone(),
            resolve_frozen_pool_vector_authority(connection, &key)?,
        );
    }

    let cursor_key = derive_cursor_key(cohort_salt.as_bytes(), project_uuid, cohort_generation_id)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(InspectionAuthoritySnapshot {
        project_uuid,
        project_id,
        config_generation_id: expected_config_generation_id,
        policy_version_ids,
        learning_generation_ids,
        cohort_generation_id,
        control_snapshot,
        vector_authorities,
        cursor_key,
    })
}

pub(super) fn verify_config_and_policy_authority(
    connection: &Connection,
    project_uuid: Uuid,
    prepared: &PreparedLedgerMaterial,
) -> Result<(), LedgerError> {
    let stored_config = connection
        .query_row(
            "SELECT project_uuid, canonical_config_json, canonical_payload_hash
             FROM config_generations WHERE config_generation_id = ?1",
            [prepared.config_generation_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if stored_config
        != (
            project_uuid.to_string(),
            prepared.canonical_config_json.clone(),
            prepared.config_generation_id.clone(),
        )
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    for (pool_id, policy) in &prepared.policies {
        let stored_policy = connection
            .query_row(
                "SELECT canonical_policy_json, canonical_payload_hash
                 FROM policy_versions
                 WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
                rusqlite::params![project_uuid.to_string(), pool_id, policy.policy_version_id,],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if stored_policy
            != (
                policy.canonical_policy_json.clone(),
                policy.policy_version_id.clone(),
            )
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}
