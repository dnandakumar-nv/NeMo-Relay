// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::reads::now_unix_ms;
use super::{InspectionService, inspection_deadline, map_ledger_error, map_read_pool_error};
use crate::inspection::{InspectionError, NeighborhoodLookupV1, NeighborhoodReportV1};
use crate::ledger::repository::inspection::{
    PersistedNeighborhoodRead, load_persisted_neighborhood,
};

impl InspectionService {
    /// Inspect one persisted evidence or query-hash neighborhood without provider calls.
    pub async fn inspect_neighborhood(
        &self,
        lookup: NeighborhoodLookupV1,
    ) -> Result<NeighborhoodReportV1, InspectionError> {
        lookup.validate()?;
        let lookup = match lookup {
            NeighborhoodLookupV1::Request { input } => {
                return self.inspect_request_neighborhood(*input).await;
            }
            lookup => lookup,
        };
        let current = self.current()?;
        let snapshot_time_unix_ms = now_unix_ms()?;
        let config = Arc::clone(&self.inner.config);
        let read_pool = current.read_pool.clone();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        match read_pool
            .run(deadline, move |connection| {
                Ok(load_persisted_neighborhood(
                    connection,
                    &config,
                    &lookup,
                    snapshot_time_unix_ms,
                ))
            })
            .await
        {
            Ok(Ok(PersistedNeighborhoodRead::Report(report))) => Ok(*report),
            Ok(Ok(PersistedNeighborhoodRead::NotFound)) => Err(InspectionError::NotFound),
            Ok(Ok(PersistedNeighborhoodRead::InvalidArgument)) => {
                Err(InspectionError::InvalidArgument)
            }
            Ok(Ok(PersistedNeighborhoodRead::NeedsEmbedding)) => {
                Err(InspectionError::IntegrityError)
            }
            Ok(Err(error)) => Err(map_ledger_error(error)),
            Err(error) => Err(map_read_pool_error(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rusqlite::params;
    use serde_json::json;
    use uuid::Uuid;

    use super::*;
    use crate::canonical_json::canonical_json;
    use crate::canonical_query::build_canonical_routing_query;
    use crate::embedder_tests::{RawHttpServer, json_response, parse_request};
    use crate::inspection::{
        DiagnosticPointKindV1, InspectionServiceOptions, NEIGHBORHOOD_REPORT_SCHEMA_V1,
    };
    use crate::ledger::repository::vector_index::{
        VectorIndexHealthMutationAck, VectorIndexHealthTarget, current_generation_manifest,
        mark_vector_index_health,
    };
    use crate::ledger::repository::{
        active_runtime_inspection_input, ready_evaluated_active_runtime_fixture,
        ready_evaluated_active_runtime_fixture_with_embedder,
    };
    use crate::projection::{
        SanitizedMessage, SanitizedMessageContent, SanitizedRouterInstructionFact,
        SanitizedRouterInstructionRole, projection_semantic_fingerprint,
    };
    use crate::routing_partition::artifact_from_routing_partition_v1;

    fn replace_current_task(input: &mut crate::inspection::RoutingInspectionInputV1, text: &str) {
        let content = input
            .request
            .normalized_request
            .messages
            .iter_mut()
            .rev()
            .find_map(|message| match message {
                SanitizedMessage::User { content, .. } => Some(content),
                _ => None,
            })
            .unwrap();
        *content = SanitizedMessageContent::Text(text.to_string());
        input.request.semantic_request_fingerprint =
            projection_semantic_fingerprint(&input.request).unwrap();
    }

    #[tokio::test]
    async fn persisted_neighborhoods_are_provider_free_exact_and_projected() {
        let (
            _temporary,
            config,
            activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            "http://127.0.0.1:1/v1",
            Some("NEMO_RELAY_INSPECTION_CACHE_MUST_NOT_RESOLVE"),
            1_000,
        );
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();

        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::Evidence { evidence_id })
            .await
            .unwrap();
        assert_eq!(report.schema, NEIGHBORHOOD_REPORT_SCHEMA_V1);
        assert_eq!(report.pool_id, evidence.summary.pool_id);
        assert_eq!(
            report.canonical_query_hash,
            evidence.summary.canonical_query_hash
        );
        assert_eq!(report.partition, evidence.partition);
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].evidence_id, evidence_id);
        assert_eq!(report.neighbors[0].ordinal, 0);
        assert_eq!(report.neighbors[0].binary_label.as_deref(), Some("pass"));
        assert_eq!(report.neighbors[0].inclusion, "included");
        assert_eq!(report.support.returned_neighbors, 1);
        assert_eq!(report.support.within_radius, 1);
        assert_eq!(report.support.attempted_roots, 1);
        assert_eq!(report.support.selected_roots, 1);
        assert_eq!(report.gates.partition, Some(true));
        assert_eq!(
            report.recommendation.candidate_id.as_deref(),
            Some("candidate-a")
        );
        assert!(!report.recommendation.anchor_fallback);
        let projection = report.projection.as_ref().unwrap();
        assert_eq!(projection.algorithm, "pca_2");
        assert_eq!(projection.points.len(), 2);
        assert_eq!(projection.points[0].kind, DiagnosticPointKindV1::Query);
        assert_eq!(projection.points[1].kind, DiagnosticPointKindV1::Evidence);
        assert_eq!(projection.points[1].record_id, evidence_id.to_string());

        let by_hash = service
            .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: evidence.summary.canonical_query_hash.clone(),
                partition: Box::new(evidence.partition.clone()),
            })
            .await
            .unwrap();
        assert_eq!(by_hash.pool_id, report.pool_id);
        assert_eq!(by_hash.partition, report.partition);
        assert_eq!(by_hash.neighbors.len(), report.neighbors.len());
        for (left, right) in by_hash.neighbors.iter().zip(&report.neighbors) {
            assert_eq!(left.evidence_id, right.evidence_id);
            assert_eq!(left.ordinal, right.ordinal);
            assert_eq!(left.distance, right.distance);
            assert_eq!(left.similarity_weight, right.similarity_weight);
            assert_eq!(left.binary_label, right.binary_label);
            assert_eq!(left.inclusion, right.inclusion);
        }
        assert_eq!(by_hash.recommendation, report.recommendation);
        assert_eq!(by_hash.projection, report.projection);

        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::Evidence {
                    evidence_id: Uuid::now_v7(),
                })
                .await
                .unwrap_err(),
            InspectionError::NotFound
        );
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                    canonical_query_hash: "not-a-hash".into(),
                    partition: Box::new(evidence.partition.clone()),
                })
                .await
                .unwrap_err(),
            InspectionError::InvalidArgument
        );
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                    canonical_query_hash:
                        "0000000000000000000000000000000000000000000000000000000000000000".into(),
                    partition: Box::new(evidence.partition.clone()),
                })
                .await
                .unwrap_err(),
            InspectionError::NotFound
        );

        let mut stale = evidence.partition.clone();
        stale.learning_generation_id = Uuid::now_v7();
        let stale = service
            .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: evidence.summary.canonical_query_hash,
                partition: Box::new(stale),
            })
            .await
            .unwrap();
        assert_eq!(stale.recommendation.reason, "version_mismatch");
        assert!(stale.recommendation.anchor_fallback);
        assert_eq!(stale.gates.partition, Some(false));
        assert!(stale.neighbors.is_empty());
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn persisted_neighborhood_requires_exact_partition_and_detects_hash_collision() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();

        let mut wrong_candidate = evidence.partition.clone();
        wrong_candidate.candidate_model.push_str("-wrong");
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                    canonical_query_hash: evidence.summary.canonical_query_hash.clone(),
                    partition: Box::new(wrong_candidate),
                })
                .await
                .unwrap_err(),
            InspectionError::InvalidArgument
        );

        let expected = artifact_from_routing_partition_v1(&evidence.partition).unwrap();
        let mut colliding = evidence.partition.clone();
        colliding.candidate_model_revision.push_str("-collision");
        let colliding = artifact_from_routing_partition_v1(&colliding).unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "UPDATE routing_partitions SET canonical_partition_json = ?1
                 WHERE partition_hash = ?2",
                params![colliding.canonical_json, expected.partition_hash],
            )
            .unwrap();
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                    canonical_query_hash: evidence.summary.canonical_query_hash,
                    partition: Box::new(evidence.partition),
                })
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn persisted_neighborhood_maps_missing_cache_without_provider_fallback() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let connection = activated.repository.test_connection_mut();
        connection
            .pragma_update(None, "foreign_keys", false)
            .unwrap();
        connection
            .execute(
                "DELETE FROM embeddings
                 WHERE vector_space_id = ?1 AND canonical_query_hash = ?2",
                params![
                    evidence.partition.vector_space_id,
                    evidence.summary.canonical_query_hash
                ],
            )
            .unwrap();
        connection
            .pragma_update(None, "foreign_keys", true)
            .unwrap();

        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: evidence.summary.canonical_query_hash,
                partition: Box::new(evidence.partition),
            })
            .await
            .unwrap();
        assert_eq!(report.recommendation.reason, "embedding_unavailable");
        assert!(report.recommendation.anchor_fallback);
        assert_eq!(report.gates.partition, Some(true));
        assert!(report.neighbors.is_empty());
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn persisted_neighborhood_maps_stale_index_and_missing_source_to_vector_health() {
        let (
            _temporary,
            config,
            mut activated,
            vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let root_table_name = current_generation_manifest(
            activated.repository.test_connection_mut(),
            &vector_space_id,
        )
        .unwrap()
        .unwrap()
        .authority()
        .root()
        .as_str()
        .to_string();
        activated
            .repository
            .test_connection_mut()
            .execute(
                &format!("DELETE FROM \"{root_table_name}\" WHERE record_id = ?1"),
                [evidence_id.to_string()],
            )
            .unwrap();
        let stale_index = service
            .inspect_neighborhood(NeighborhoodLookupV1::Evidence { evidence_id })
            .await
            .unwrap();
        assert_eq!(stale_index.recommendation.reason, "vector_unhealthy");
        assert_eq!(stale_index.gates.partition, Some(true));
        service.close().await.unwrap();
        drop(activated);

        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        activated
            .repository
            .test_connection_mut()
            .execute(
                "DELETE FROM evidence_vector_links WHERE evidence_vector_link_id = ?1",
                [evidence_id.to_string()],
            )
            .unwrap();
        let missing_source = service
            .inspect_neighborhood(NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: evidence.summary.canonical_query_hash,
                partition: Box::new(evidence.partition),
            })
            .await
            .unwrap();
        assert_eq!(missing_source.recommendation.reason, "vector_unhealthy");
        assert_eq!(missing_source.gates.partition, Some(true));
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::Evidence { evidence_id })
                .await
                .unwrap_err(),
            InspectionError::NotFound
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn persisted_neighborhood_maps_declared_unavailable_vector_generation() {
        let (
            _temporary,
            config,
            mut activated,
            vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture();
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let (generation, manifest_hash) = {
            let manifest = current_generation_manifest(
                activated.repository.test_connection_mut(),
                &vector_space_id,
            )
            .unwrap()
            .unwrap();
            (
                manifest.generation(),
                manifest.canonical_payload_hash().to_string(),
            )
        };
        let transaction = activated
            .repository
            .test_connection_mut()
            .transaction()
            .unwrap();
        assert!(matches!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                generation,
                &manifest_hash,
                VectorIndexHealthTarget::Unavailable,
                "router.vector.inspection_test",
                i64::MAX,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Applied { .. }
        ));
        transaction.commit().unwrap();

        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::Evidence { evidence_id })
            .await
            .unwrap();
        assert_eq!(report.recommendation.reason, "vector_unhealthy");
        assert_eq!(report.gates.partition, Some(true));
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn cached_request_neighborhood_uses_exact_projection_without_egress() {
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            "http://127.0.0.1:1/v1",
            Some("NEMO_RELAY_INSPECTION_CACHE_MUST_NOT_RESOLVE"),
            1_000,
        );
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        assert!(!service.request_embedding_enabled());
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let input = active_runtime_inspection_input(&config, evidence.partition.clone());
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::Request {
                input: Box::new(input),
            })
            .await
            .unwrap();
        assert_eq!(
            report.canonical_query_hash,
            evidence.summary.canonical_query_hash
        );
        assert_eq!(report.partition, evidence.partition);
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].evidence_id, evidence_id);
        assert_eq!(
            report.recommendation.candidate_id.as_deref(),
            Some("candidate-a")
        );
        assert!(report.projection.is_some());
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn provider_request_neighborhood_sends_one_exact_canonical_payload_without_writes() {
        const API_KEY_ENV: &str = "NEMO_RELAY_INSPECTION_TASK6_API_KEY";
        const API_KEY: &str = "inspection-task6-secret-value";
        let _context_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        unsafe {
            std::env::set_var(API_KEY_ENV, API_KEY);
        }
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": values, "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            Some(API_KEY_ENV),
            1_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition.clone());
        replace_current_task(&mut input, "inspect a novel provider query");
        let expected = build_canonical_routing_query(
            &input.request,
            &input.routing_context,
            &config.pools[0].canonicalizer,
        )
        .unwrap();
        assert_ne!(
            expected.canonical_query_hash,
            evidence.summary.canonical_query_hash
        );
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();

        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::Request {
                input: Box::new(input),
            })
            .await
            .unwrap();
        assert_eq!(report.canonical_query_hash, expected.canonical_query_hash);
        assert_eq!(report.partition, evidence.partition);
        assert_eq!(report.neighbors.len(), 1);
        assert_eq!(report.neighbors[0].evidence_id, evidence_id);
        assert_eq!(server.request_count(), 1);
        let request = parse_request(&server.wait_for_request());
        assert_eq!(request.request_line, "POST /v1/embeddings HTTP/1.1");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer inspection-task6-secret-value")
        );
        let canonical_query = String::from_utf8(expected.canonical_bytes).unwrap();
        assert_eq!(
            request.body,
            canonical_json(&json!({
                "model": "embedder-model",
                "input": [canonical_query],
                "encoding_format": "float"
            }))
            .unwrap()
            .into_bytes()
        );
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
        unsafe {
            std::env::remove_var(API_KEY_ENV);
        }
    }

    #[tokio::test]
    async fn request_cache_miss_respects_host_egress_capability_before_client_build() {
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": values, "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            Some("NEMO_RELAY_INSPECTION_MUST_NOT_RESOLVE"),
            1_000,
        );
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition);
        replace_current_task(&mut input, "cache miss with host egress disabled");
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::Request {
                    input: Box::new(input),
                })
                .await
                .unwrap_err(),
            InspectionError::EgressDenied
        );
        assert_eq!(server.request_count(), 0);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn malformed_request_inputs_fail_before_provider_start() {
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": values, "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            None,
            1_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let valid = active_runtime_inspection_input(&config, evidence.partition);
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();

        let mut unknown = serde_json::to_value(&valid).unwrap();
        unknown["request"]["normalized_request"]
            .as_object_mut()
            .unwrap()
            .insert("headers".into(), json!({"authorization": "Bearer secret"}));
        assert!(
            serde_json::from_value::<crate::inspection::RoutingInspectionInputV1>(unknown).is_err()
        );

        let mut malformed = Vec::new();
        let mut wrong_schema = valid.clone();
        wrong_schema.schema = "nemo.relay.router.inspection-input@2".into();
        malformed.push(wrong_schema);
        let mut wrong_pool = valid.clone();
        wrong_pool.pool_id = "missing-pool".into();
        malformed.push(wrong_pool);
        let mut wrong_family = valid.clone();
        wrong_family.request.family = nemo_relay_types::api::llm::LlmApiFamily::AnthropicMessages;
        wrong_family.request.semantic_request_fingerprint =
            projection_semantic_fingerprint(&wrong_family.request).unwrap();
        malformed.push(wrong_family);
        let mut wrong_partition = valid.clone();
        wrong_partition.partition.candidate_model.push_str("-wrong");
        malformed.push(wrong_partition);
        let mut mismatched_instructions = valid.clone();
        mismatched_instructions
            .request
            .ordered_instructions
            .push(SanitizedRouterInstructionFact {
                wire_ordinal: 0,
                role: SanitizedRouterInstructionRole::System,
                content: "not present in the sanitized messages".into(),
                name: None,
            });
        mismatched_instructions.request.semantic_request_fingerprint =
            projection_semantic_fingerprint(&mismatched_instructions.request).unwrap();
        malformed.push(mismatched_instructions);
        let mut credential = valid.clone();
        credential.request.normalized_request.reasoning =
            Some(json!({"api_key": "inspection-secret"}));
        credential.request.required_capabilities = vec!["reasoning_controls".into()];
        credential.request.semantic_request_fingerprint =
            projection_semantic_fingerprint(&credential.request).unwrap();
        malformed.push(credential);
        let mut metadata = valid;
        metadata.request.normalized_request.reasoning =
            Some(json!({"metadata": {"region": "untrusted"}}));
        metadata.request.required_capabilities = vec!["reasoning_controls".into()];
        metadata.request.semantic_request_fingerprint =
            projection_semantic_fingerprint(&metadata.request).unwrap();
        malformed.push(metadata);

        for input in malformed {
            assert_eq!(
                service
                    .inspect_neighborhood(NeighborhoodLookupV1::Request {
                        input: Box::new(input),
                    })
                    .await
                    .unwrap_err(),
                InspectionError::InvalidArgument
            );
        }
        assert_eq!(server.request_count(), 0);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn request_requires_an_existing_exact_partition_before_provider_start() {
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": values, "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            None,
            1_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition);
        replace_current_task(&mut input, "query under a nonexistent exact partition");
        input.partition.decoding_fingerprint =
            if input.partition.decoding_fingerprint == "a".repeat(64) {
                "b".repeat(64)
            } else {
                "a".repeat(64)
            };
        let expected_partition = input.partition.clone();
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        let report = service
            .inspect_neighborhood(NeighborhoodLookupV1::Request {
                input: Box::new(input),
            })
            .await
            .unwrap();
        assert_eq!(report.partition, expected_partition);
        assert_eq!(report.recommendation.reason, "no_partition");
        assert!(report.recommendation.anchor_fallback);
        assert_eq!(report.gates.partition, Some(false));
        assert_eq!(server.request_count(), 0);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn request_missing_credential_fails_before_provider_start() {
        const MISSING_ENV: &str = "NEMO_RELAY_INSPECTION_TASK6_MISSING_KEY";
        let _context_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        unsafe {
            std::env::remove_var(MISSING_ENV);
        }
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": values, "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            Some(MISSING_ENV),
            1_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition);
        replace_current_task(&mut input, "request requiring a missing credential");
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::Request {
                    input: Box::new(input),
                })
                .await
                .unwrap_err(),
            InspectionError::StorageUnavailable
        );
        assert_eq!(server.request_count(), 0);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn request_rejects_provider_dimension_mismatch_without_writes() {
        let server = RawHttpServer::start(json_response(json!({
            "data": [{"object": "embedding", "embedding": [1.0, 0.0], "index": 0}]
        })));
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            None,
            1_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition);
        replace_current_task(&mut input, "request with a wrong provider vector dimension");
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(
            service
                .inspect_neighborhood(NeighborhoodLookupV1::Request {
                    input: Box::new(input),
                })
                .await
                .unwrap_err(),
            InspectionError::StorageUnavailable
        );
        assert_eq!(server.request_count(), 1);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_cancels_an_in_flight_request_embedding_without_writes() {
        let mut values = vec![0.0; 16];
        values[0] = 1.0;
        let server = RawHttpServer::start_with_delay(
            json_response(json!({
                "data": [{"object": "embedding", "embedding": values, "index": 0}]
            })),
            std::time::Duration::from_secs(1),
        );
        let (
            _temporary,
            config,
            mut activated,
            _vector_space_id,
            _query_vector,
            evidence_id,
            _evaluation_id,
        ) = ready_evaluated_active_runtime_fixture_with_embedder(
            &server.base_url("127.0.0.1"),
            None,
            2_000,
        );
        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                allow_request_embedding: true,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let evidence = service.get_evidence(evidence_id).await.unwrap();
        let mut input = active_runtime_inspection_input(&config, evidence.partition);
        replace_current_task(&mut input, "cancel this in-flight inspection request");
        let before_data_version = activated
            .repository
            .test_connection_mut()
            .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
            .unwrap();
        let request_service = service.clone();
        let request = tokio::spawn(async move {
            request_service
                .inspect_neighborhood(NeighborhoodLookupV1::Request {
                    input: Box::new(input),
                })
                .await
        });
        let wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while server.request_count() == 0 {
            assert!(std::time::Instant::now() < wait_deadline);
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        service.abort();
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_millis(500), request)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err(),
            InspectionError::StorageUnavailable
        );
        assert_eq!(server.request_count(), 1);
        assert_eq!(
            activated
                .repository
                .test_connection_mut()
                .query_row("PRAGMA data_version", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            before_data_version
        );
        drop(service);
        drop(activated);
    }
}
