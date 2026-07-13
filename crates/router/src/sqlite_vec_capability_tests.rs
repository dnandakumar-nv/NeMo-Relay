// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use rusqlite::{Connection, params};

use crate::sqlite_vec_extension::{
    EXPECTED_VEC_VERSION, SqliteVecStatus, register, verify_connection,
};

const KNN_SQL: &str = "SELECT record_id, distance FROM capability_vectors \
     WHERE embedding MATCH ?1 AND k = ?2 AND partition_id = ?3 \
     ORDER BY distance";

fn registered_connection() -> Connection {
    assert_eq!(register(), SqliteVecStatus::Available);
    let connection = Connection::open_in_memory().expect("in-memory SQLite should open");
    assert_eq!(verify_connection(&connection), SqliteVecStatus::Available);
    connection
}

fn vector_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        blob.extend_from_slice(&value.to_ne_bytes());
    }
    blob
}

fn knn_rows(
    connection: &Connection,
    query: &[f32],
    k: i64,
    partition_id: i64,
) -> rusqlite::Result<Vec<(String, f64)>> {
    let query = vector_blob(query);
    connection
        .prepare(KNN_SQL)?
        .query_map(params![query, k, partition_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect()
}

#[test]
fn pinned_runtime_text_ids_knn_and_hard_boundaries_are_exact() {
    let connection = registered_connection();
    assert_eq!(EXPECTED_VEC_VERSION, "v0.1.9");
    let runtime: String = connection
        .query_row("SELECT vec_version()", [], |row| row.get(0))
        .expect("sqlite-vec should expose its runtime version");
    assert_eq!(runtime, EXPECTED_VEC_VERSION);

    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE capability_vectors USING vec0(
                record_id TEXT PRIMARY KEY,
                embedding FLOAT[3] distance_metric=cosine,
                partition_id INTEGER PARTITION KEY
             );",
        )
        .expect("the pinned vec0 schema should be accepted");
    let insert =
        "INSERT INTO capability_vectors(record_id, embedding, partition_id) VALUES (?1, ?2, ?3)";
    for (record_id, partition_id, vector) in [
        ("b", 7_i64, [1.0_f32, 0.0, 0.0]),
        ("a", 7_i64, [1.0_f32, 0.0, 0.0]),
        ("x", 8_i64, [1.0_f32, 0.0, 0.0]),
    ] {
        connection
            .execute(
                insert,
                params![record_id, vector_blob(&vector), partition_id],
            )
            .expect("text-key vector insertion should succeed");
    }

    let mut rows = knn_rows(&connection, &[1.0, 0.0, 0.0], 4_096, 7)
        .expect("the documented maximum KNN k should succeed");
    rows.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(rows, vec![("a".to_string(), 0.0), ("b".to_string(), 0.0)]);

    let secondary_order_error = connection
        .prepare(
            "SELECT record_id, distance FROM capability_vectors
             WHERE embedding MATCH ?1 AND k = ?2 AND partition_id = ?3
             ORDER BY distance, record_id",
        )
        .expect_err("vec0 KNN must reject a secondary ordering term");
    assert_eq!(
        secondary_order_error.to_string(),
        "Only a single 'ORDER BY distance' clause is allowed on vec0 KNN queries"
    );

    let k_error = knn_rows(&connection, &[1.0, 0.0, 0.0], 4_097, 7)
        .expect_err("k above the pinned vec0 limit must fail");
    assert_eq!(
        k_error.to_string(),
        "k value in knn query too large, provided 4097 and the limit is 4096"
    );

    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE dimension_8192 USING vec0(embedding FLOAT[8192]);
             DROP TABLE dimension_8192;",
        )
        .expect("the maximum vec0 dimension should succeed");
    let dimension_error = connection
        .execute_batch("CREATE VIRTUAL TABLE dimension_8193 USING vec0(embedding FLOAT[8193]);")
        .expect_err("a dimension above the pinned vec0 limit must fail");
    assert_eq!(
        dimension_error.to_string(),
        "vec0 constructor error: Dimension on vector column too large, provided 8193, maximum 8192"
    );
}

#[test]
fn more_than_4096_distinct_partition_values_remain_searchable() {
    const PARTITION_COUNT: i64 = 4_097;

    let mut connection = registered_connection();
    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE many_partitions USING vec0(
                record_id TEXT PRIMARY KEY,
                embedding FLOAT[1] distance_metric=cosine,
                partition_id INTEGER PARTITION KEY,
                chunk_size=8
             );",
        )
        .expect("partition capability table should be created");
    let vector = vector_blob(&[1.0]);
    let transaction = connection
        .transaction()
        .expect("partition inserts should be transactional");
    {
        let mut insert = transaction
            .prepare(
                "INSERT INTO many_partitions(record_id, embedding, partition_id)
                 VALUES (?1, ?2, ?3)",
            )
            .expect("partition insert should prepare");
        for partition_id in 1..=PARTITION_COUNT {
            insert
                .execute(params![
                    format!("record-{partition_id:04}"),
                    &vector,
                    partition_id
                ])
                .expect("every distinct partition should accept one vector");
        }
    }
    transaction
        .commit()
        .expect("partition inserts should commit");

    let query = vector_blob(&[1.0]);
    let rows = connection
        .prepare(
            "SELECT record_id, distance FROM many_partitions
             WHERE embedding MATCH ?1 AND k = 1 AND partition_id = ?2
             ORDER BY distance",
        )
        .expect("partition KNN should prepare")
        .query_map(params![query, PARTITION_COUNT], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .expect("partition KNN should execute")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("partition KNN rows should decode");
    assert_eq!(rows, vec![(format!("record-{PARTITION_COUNT:04}"), 0.0)]);
}

#[test]
fn renaming_a_vec0_root_leaves_shadow_names_and_breaks_drop() {
    let connection = registered_connection();
    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE rename_original USING vec0(
                record_id TEXT PRIMARY KEY,
                embedding FLOAT[2] distance_metric=cosine,
                partition_id INTEGER PARTITION KEY
             );
             ALTER TABLE rename_original RENAME TO rename_retired;",
        )
        .expect("the pinned extension permits renaming only the virtual root");

    let names = connection
        .prepare(
            "SELECT name FROM sqlite_schema
             WHERE name = 'rename_retired' OR name LIKE 'rename_original_%'
             ORDER BY name",
        )
        .expect("renamed schema lookup should prepare")
        .query_map([], |row| row.get::<_, String>(0))
        .expect("renamed schema lookup should execute")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("renamed schema names should decode");
    assert_eq!(
        names,
        vec![
            "rename_original_chunks".to_string(),
            "rename_original_info".to_string(),
            "rename_original_rowids".to_string(),
            "rename_original_vector_chunks00".to_string(),
            "rename_retired".to_string(),
        ]
    );
    assert!(
        connection
            .execute_batch("DROP TABLE rename_retired;")
            .is_err(),
        "the renamed root cannot drop shadow tables that kept their original names"
    );
}

#[derive(Debug)]
struct VdbeOperation {
    address: i64,
    opcode: String,
    p1: i64,
    p2: i64,
    p3: i64,
    p4: Option<String>,
}

#[test]
fn scalar_fallback_filters_partition_before_loading_or_ranking_vectors() {
    let connection = registered_connection();
    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE vdbe_vectors USING vec0(
                record_id TEXT PRIMARY KEY,
                embedding FLOAT[3] distance_metric=cosine,
                partition_id INTEGER PARTITION KEY
             );",
        )
        .expect("VDBE capability table should be created");

    let query = vector_blob(&[1.0, 0.0, 0.0]);
    let operations = connection
        .prepare(
            "EXPLAIN
             SELECT record_id, vec_distance_cosine(embedding, ?1) AS distance
             FROM vdbe_vectors
             WHERE partition_id = ?2
             ORDER BY distance, record_id
             LIMIT ?3",
        )
        .expect("scalar fallback EXPLAIN should prepare")
        .query_map(params![query, 7_i64, 10_i64], |row| {
            Ok(VdbeOperation {
                address: row.get(0)?,
                opcode: row.get(1)?,
                p1: row.get(2)?,
                p2: row.get(3)?,
                p3: row.get(4)?,
                p4: row.get(5)?,
            })
        })
        .expect("scalar fallback EXPLAIN should execute")
        .collect::<rusqlite::Result<Vec<_>>>()
        .expect("VDBE operations should decode");

    let partition_column = operations
        .iter()
        .find(|operation| operation.opcode == "VColumn" && operation.p2 == 2)
        .expect("VDBE must load the partition column");
    let partition_guard = operations
        .iter()
        .find(|operation| {
            operation.opcode == "Ne"
                && operation.address > partition_column.address
                && (operation.p1 == partition_column.p3 || operation.p3 == partition_column.p3)
        })
        .expect("VDBE must reject nonmatching partitions");
    let embedding_column = operations
        .iter()
        .find(|operation| {
            operation.opcode == "VColumn"
                && operation.p2 == 1
                && operation.address > partition_guard.address
        })
        .expect("VDBE must load an embedding after the partition guard");
    let distance = operations
        .iter()
        .find(|operation| {
            operation.opcode == "Function"
                && operation
                    .p4
                    .as_deref()
                    .is_some_and(|value| value.starts_with("vec_distance_cosine("))
        })
        .expect("VDBE must invoke scalar cosine distance");
    assert!(
        partition_column.address < partition_guard.address
            && partition_guard.address < embedding_column.address
            && embedding_column.address < distance.address
    );
    assert_eq!(
        distance.p2, embedding_column.p3,
        "the cosine function's first argument register must be the loaded embedding"
    );
    let guard_target = operations
        .iter()
        .find(|operation| operation.address == partition_guard.p2)
        .expect("the partition guard branch target must exist");
    assert_eq!(
        guard_target.opcode, "VNext",
        "a nonmatching partition must skip embedding load, distance, and ranking"
    );
}
