// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-global registration and connection verification for pinned sqlite-vec.
//!
//! SQLite auto extensions affect every connection opened later in this process.
//! Router registers the statically linked extension once, only when a non-off
//! activation begins, and verifies each connection before vector operations.

use std::sync::OnceLock;

use rusqlite::{Connection, ffi};

pub(crate) const EXPECTED_VEC_VERSION: &str = env!("EXPECTED_VEC_VERSION");

static REGISTRATION: OnceLock<SqliteVecStatus> = OnceLock::new();

/// Stable sqlite-vec availability state without native or SQL error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SqliteVecStatus {
    Available,
    RegistrationFailed,
    VersionMissing,
    VersionMismatch,
}

/// Register the statically linked extension at most once for this process.
pub(crate) fn register() -> SqliteVecStatus {
    *REGISTRATION.get_or_init(register_once)
}

#[cfg(test)]
pub(crate) fn registration_status() -> Option<SqliteVecStatus> {
    REGISTRATION.get().copied()
}

/// Verify that one SQLite connection exposes the exact pinned runtime version.
pub(crate) fn verify_connection(connection: &Connection) -> SqliteVecStatus {
    let reported_version = connection
        .query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
        .ok();
    classify_reported_version(reported_version.as_deref())
}

fn register_once() -> SqliteVecStatus {
    // SAFETY: sqlite-vec 0.1.9's Rust declaration erases its C arguments, but
    // its pinned header defines the exact sqlite3 auto-extension callback ABI.
    // Registration is process-global, guarded by `REGISTRATION`, and the
    // callback only initializes the connection supplied by SQLite.
    let result = unsafe {
        ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            rusqlite::auto_extension::RawAutoExtension,
        >(::sqlite_vec::sqlite3_vec_init as *const ())))
    };
    if result == ffi::SQLITE_OK {
        SqliteVecStatus::Available
    } else {
        SqliteVecStatus::RegistrationFailed
    }
}

fn classify_reported_version(reported_version: Option<&str>) -> SqliteVecStatus {
    match reported_version {
        Some(EXPECTED_VEC_VERSION) => SqliteVecStatus::Available,
        Some(_) => SqliteVecStatus::VersionMismatch,
        None => SqliteVecStatus::VersionMissing,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use rusqlite::Connection;

    use super::{
        EXPECTED_VEC_VERSION, SqliteVecStatus, classify_reported_version, register,
        registration_status, verify_connection,
    };

    #[test]
    fn build_generated_version_matches_the_locked_contract() {
        assert_eq!(EXPECTED_VEC_VERSION, "v0.1.9");
    }

    #[test]
    fn missing_and_mismatched_versions_have_stable_classifications() {
        assert_eq!(
            classify_reported_version(None),
            SqliteVecStatus::VersionMissing
        );
        assert_eq!(
            classify_reported_version(Some("v0.1.8")),
            SqliteVecStatus::VersionMismatch
        );
        assert_eq!(
            classify_reported_version(Some(EXPECTED_VEC_VERSION)),
            SqliteVecStatus::Available
        );
    }

    #[test]
    fn an_unregistered_connection_degrades_without_breaking_relational_sql() {
        const CHILD_ENV: &str = "NEMO_RELAY_ROUTER_UNREGISTERED_VEC_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            assert_eq!(registration_status(), None);
            let connection = Connection::open_in_memory().unwrap();
            assert_eq!(
                verify_connection(&connection),
                SqliteVecStatus::VersionMissing
            );
            let answer: i64 = connection
                .query_row("SELECT 40 + 2", [], |row| row.get(0))
                .unwrap();
            assert_eq!(answer, 42);
            return;
        }

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "sqlite_vec_extension::tests::an_unregistered_connection_degrades_without_breaking_relational_sql",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[test]
    fn registration_is_concurrently_idempotent() {
        const THREADS: usize = 8;
        let barrier = Arc::new(Barrier::new(THREADS));
        let threads = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    register()
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            assert_eq!(thread.join().unwrap(), SqliteVecStatus::Available);
        }
    }

    #[test]
    fn fresh_connections_expose_the_exact_pinned_runtime() {
        assert_eq!(register(), SqliteVecStatus::Available);
        for _ in 0..2 {
            let connection = Connection::open_in_memory().unwrap();
            assert_eq!(verify_connection(&connection), SqliteVecStatus::Available);
            let runtime_version: String = connection
                .query_row("SELECT vec_version()", [], |row| row.get(0))
                .unwrap();
            assert_eq!(runtime_version, "v0.1.9");
        }
    }
}
