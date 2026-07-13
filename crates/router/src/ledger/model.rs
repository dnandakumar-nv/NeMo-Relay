// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable non-secret ledger identities and failure classes.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use uuid::Uuid;
use zeroize::Zeroizing;

use crate::vector::VectorSpaceId;

/// Exact size of protected cohort assignment material.
pub(super) const COHORT_SALT_BYTES: usize = 32;
/// Version-1 cohort assignment algorithm stored with each salt generation.
pub(super) const COHORT_ASSIGNMENT_ALGORITHM_V1: &str = "hmac-sha256-v1";

/// Stable non-secret class for one ledger failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LedgerErrorClass {
    /// The database path or an adjacent SQLite sidecar failed structural validation.
    InvalidFilesystem,
    /// Owner-only filesystem permissions could not be established or verified.
    InvalidPermissions,
    /// The target does not provide the required filesystem security primitives.
    UnsupportedFilesystemSecurity,
    /// The SQLite database could not be opened safely.
    OpenFailed,
    /// SQLite remained locked beyond the fixed busy timeout.
    Busy,
    /// SQLite reported corruption or the file was not a Router ledger.
    CorruptDatabase,
    /// A required SQLite PRAGMA did not retain its required value.
    PragmaMismatch,
    /// The linked SQLite runtime version did not match the supported version.
    SqliteVersionMismatch,
    /// The database was created by a newer Router schema.
    FutureSchema,
    /// A stored migration checksum did not match the compiled migration.
    MigrationChecksumMismatch,
    /// Stored migration versions were missing, duplicated, or out of order.
    InvalidMigrationHistory,
    /// A pending migration could not be applied atomically.
    MigrationFailed,
    /// The configured project identifier did not match the ledger project.
    ProjectIdMismatch,
    /// A durable identity row violated the ledger identity contract.
    IdentityInvariant,
    /// Cryptographically secure identity material could not be generated.
    RandomnessUnavailable,
    /// A safe ledger payload could not be serialized canonically.
    CanonicalizationFailed,
    /// A database operation failed without a more specific safe classification.
    DatabaseOperationFailed,
}

impl LedgerErrorClass {
    /// Return the stable machine-readable failure code.
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::InvalidFilesystem => "router.ledger.invalid_filesystem",
            Self::InvalidPermissions => "router.ledger.invalid_permissions",
            Self::UnsupportedFilesystemSecurity => "router.ledger.unsupported_filesystem_security",
            Self::OpenFailed => "router.ledger.open_failed",
            Self::Busy => "router.ledger.busy",
            Self::CorruptDatabase => "router.ledger.corrupt_database",
            Self::PragmaMismatch => "router.ledger.pragma_mismatch",
            Self::SqliteVersionMismatch => "router.ledger.sqlite_version_mismatch",
            Self::FutureSchema => "router.ledger.future_schema",
            Self::MigrationChecksumMismatch => "router.ledger.migration_checksum_mismatch",
            Self::InvalidMigrationHistory => "router.ledger.invalid_migration_history",
            Self::MigrationFailed => "router.ledger.migration_failed",
            Self::ProjectIdMismatch => "router.ledger.project_id_mismatch",
            Self::IdentityInvariant => "router.ledger.identity_invariant",
            Self::RandomnessUnavailable => "router.ledger.randomness_unavailable",
            Self::CanonicalizationFailed => "router.ledger.canonicalization_failed",
            Self::DatabaseOperationFailed => "router.ledger.database_operation_failed",
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::InvalidFilesystem => "Router ledger filesystem validation failed",
            Self::InvalidPermissions => "Router ledger permissions are not owner-only",
            Self::UnsupportedFilesystemSecurity => {
                "Router ledger filesystem security is unsupported"
            }
            Self::OpenFailed => "Router ledger could not be opened",
            Self::Busy => "Router ledger remained busy",
            Self::CorruptDatabase => "Router ledger is corrupt or has an invalid format",
            Self::PragmaMismatch => "Router ledger SQLite settings could not be verified",
            Self::SqliteVersionMismatch => "Router ledger SQLite runtime version is unsupported",
            Self::FutureSchema => "Router ledger schema is newer than this runtime",
            Self::MigrationChecksumMismatch => "Router ledger migration checksum mismatch",
            Self::InvalidMigrationHistory => "Router ledger migration history is invalid",
            Self::MigrationFailed => "Router ledger migration failed",
            Self::ProjectIdMismatch => "Router ledger project identifier mismatch",
            Self::IdentityInvariant => "Router ledger identity invariant failed",
            Self::RandomnessUnavailable => "Router ledger secure randomness was unavailable",
            Self::CanonicalizationFailed => "Router ledger canonicalization failed",
            Self::DatabaseOperationFailed => "Router ledger database operation failed",
        }
    }
}

/// Sanitized ledger error that cannot retain paths, SQL, secrets, or provider content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LedgerError {
    class: LedgerErrorClass,
}

impl LedgerError {
    /// Create a ledger error from a stable non-secret class.
    pub(super) const fn new(class: LedgerErrorClass) -> Self {
        Self { class }
    }

    /// Return this error's stable class.
    #[allow(dead_code)] // Repository tests and later health reporting consume the class directly.
    pub(crate) const fn class(&self) -> LedgerErrorClass {
        self.class
    }

    /// Return this error's stable machine-readable code.
    pub(crate) const fn code(&self) -> &'static str {
        self.class.code()
    }
}

impl From<LedgerErrorClass> for LedgerError {
    fn from(class: LedgerErrorClass) -> Self {
        Self::new(class)
    }
}

impl fmt::Display for LedgerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.class.message())
    }
}

impl Error for LedgerError {}

/// Ledger-derived identities for one configured routing pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PoolRuntimeIdentity {
    /// Deterministic version of the pool's routing and evaluation policy.
    pub(crate) policy_version_id: String,
    /// Current append-only learning generation selected for this pool.
    pub(crate) learning_generation_id: Uuid,
    /// Fully verified current vector-space mapping, when learning is enabled.
    pub(crate) vector_space: Option<PoolVectorSpaceRuntimeIdentity>,
}

/// Nonsecret ledger-derived vector authority for one configured pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PoolVectorSpaceRuntimeIdentity {
    pub(crate) profile_id: String,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) canonicalizer_version_id: String,
    pub(crate) vector_space_id: VectorSpaceId,
}

/// Ledger-derived identity snapshot owned by one Router process instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LedgerRuntimeIdentity {
    /// Fresh UUIDv7 identifying this process activation.
    pub(crate) process_instance_id: Uuid,
    /// Stable UUIDv7 identifying the project ledger.
    pub(crate) project_uuid: Uuid,
    /// Stable configured project identifier or the project UUID string.
    pub(crate) project_id: String,
    /// Current append-only cohort salt generation.
    pub(crate) cohort_generation_id: Uuid,
    /// Deterministic canonical Router configuration generation.
    pub(crate) config_generation_id: String,
    /// Current policy and learning identities keyed by configured pool ID.
    pub(crate) pools: BTreeMap<String, PoolRuntimeIdentity>,
}

impl LedgerRuntimeIdentity {
    /// Return the active identity for one configured pool.
    #[allow(dead_code)] // Task 6 writer commands use this checked pool lookup.
    pub(crate) fn pool(&self, pool_id: &str) -> Option<&PoolRuntimeIdentity> {
        self.pools.get(pool_id)
    }
}

/// Protected cohort material restricted to the private ledger module.
///
/// The wrapper deliberately implements neither `Debug`, `Clone`, nor serialization.
pub(super) struct CohortSalt(Zeroizing<[u8; COHORT_SALT_BYTES]>);

impl CohortSalt {
    /// Wrap newly generated protected material without copying it into a plain buffer.
    pub(super) fn new(bytes: Zeroizing<[u8; COHORT_SALT_BYTES]>) -> Self {
        Self(bytes)
    }

    /// Consume an exact-size SQLite buffer and zeroize both source and destination.
    pub(super) fn from_vec(bytes: Vec<u8>) -> Result<Self, LedgerError> {
        let bytes = Zeroizing::new(bytes);
        if bytes.len() != COHORT_SALT_BYTES {
            return Err(LedgerError::new(LedgerErrorClass::IdentityInvariant));
        }
        let mut protected = Zeroizing::new([0; COHORT_SALT_BYTES]);
        protected.as_mut().copy_from_slice(bytes.as_slice());
        Ok(Self::new(protected))
    }

    /// Borrow the protected bytes for a bound SQLite or HMAC operation.
    pub(super) fn as_bytes(&self) -> &[u8; COHORT_SALT_BYTES] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use uuid::Uuid;

    use super::{
        COHORT_ASSIGNMENT_ALGORITHM_V1, COHORT_SALT_BYTES, CohortSalt, LedgerError,
        LedgerErrorClass, LedgerRuntimeIdentity, PoolRuntimeIdentity,
    };

    #[test]
    fn errors_expose_only_stable_classes_and_messages() {
        for class in [
            LedgerErrorClass::InvalidFilesystem,
            LedgerErrorClass::InvalidPermissions,
            LedgerErrorClass::UnsupportedFilesystemSecurity,
            LedgerErrorClass::OpenFailed,
            LedgerErrorClass::Busy,
            LedgerErrorClass::CorruptDatabase,
            LedgerErrorClass::PragmaMismatch,
            LedgerErrorClass::SqliteVersionMismatch,
            LedgerErrorClass::FutureSchema,
            LedgerErrorClass::MigrationChecksumMismatch,
            LedgerErrorClass::InvalidMigrationHistory,
            LedgerErrorClass::MigrationFailed,
            LedgerErrorClass::ProjectIdMismatch,
            LedgerErrorClass::IdentityInvariant,
            LedgerErrorClass::RandomnessUnavailable,
            LedgerErrorClass::CanonicalizationFailed,
            LedgerErrorClass::DatabaseOperationFailed,
        ] {
            let error = LedgerError::new(class);
            assert_eq!(error.class(), class);
            assert_eq!(error.code(), class.code());
            assert!(!error.to_string().contains('/') && !error.to_string().contains('\\'));
        }
    }

    #[test]
    fn runtime_identity_is_pool_scoped() {
        let pool = PoolRuntimeIdentity {
            policy_version_id: "a".repeat(64),
            learning_generation_id: Uuid::now_v7(),
            vector_space: None,
        };
        let identity = LedgerRuntimeIdentity {
            process_instance_id: Uuid::now_v7(),
            project_uuid: Uuid::now_v7(),
            project_id: "project".into(),
            cohort_generation_id: Uuid::now_v7(),
            config_generation_id: "b".repeat(64),
            pools: BTreeMap::from([("pool".into(), pool.clone())]),
        };
        assert_eq!(identity.pool("pool"), Some(&pool));
        assert_eq!(identity.pool("other"), None);
    }

    #[test]
    fn cohort_salt_rejects_the_wrong_size() {
        assert_eq!(COHORT_ASSIGNMENT_ALGORITHM_V1, "hmac-sha256-v1");
        assert!(CohortSalt::from_vec(vec![0; COHORT_SALT_BYTES - 1]).is_err());
        let salt = CohortSalt::from_vec(vec![7; COHORT_SALT_BYTES]).unwrap();
        assert_eq!(salt.as_bytes(), &[7; COHORT_SALT_BYTES]);
    }
}
