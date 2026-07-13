// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use nemo_relay_router::inspection::{
    DECISION_EXPOSURE_SCHEMA_V1, DecisionDetailV1, DecisionExposureV1, DecisionSummaryV1,
    EVIDENCE_EXPORT_RECORD_SCHEMA_V1, EvidenceDetailV1, EvidenceSummaryV1, HealthEventV1,
    MigrationSummaryV1, NEIGHBORHOOD_REPORT_SCHEMA_V1, NeighborhoodReportV1,
    OVERVIEW_REPORT_SCHEMA_V1, OVERVIEW_WINDOW_MS_V1, OperatorHistoryEntryV1, OutcomeSummaryV1,
    POOL_DETAIL_SCHEMA_V1, Page, PoolDetailV1, STATUS_REPORT_SCHEMA_V1, StatusReportV1,
};
use serde::Deserialize;
use serde_json::Value;

const FIXTURE_SCHEMA_V1: &str = "nemo.relay.router.dashboard-contract-fixture@1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DashboardContractFixtureV1 {
    schema: String,
    status_states: Vec<StatusStateFixtureV1>,
    overview: nemo_relay_router::inspection::OverviewReportV1,
    pool_detail: PoolDetailV1,
    evidence_page: Page<EvidenceSummaryV1>,
    evidence_detail: EvidenceDetailV1,
    neighborhood: NeighborhoodReportV1,
    decision_page: Page<DecisionSummaryV1>,
    decision_detail: DecisionDetailV1,
    decision_exposures: Vec<DecisionExposureV1>,
    outcome_page: Page<OutcomeSummaryV1>,
    controls: Page<OperatorHistoryEntryV1>,
    health: Page<HealthEventV1>,
    migrations: Page<MigrationSummaryV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StatusStateFixtureV1 {
    name: String,
    overview_available: bool,
    status: StatusReportV1,
}

fn fixture_bytes() -> &'static [u8] {
    include_bytes!("../fixtures/router-dashboard/contract-v1.json")
}

#[test]
fn dashboard_contract_fixture_strictly_deserializes_every_public_response() {
    let fixture: DashboardContractFixtureV1 = serde_json::from_slice(fixture_bytes()).unwrap();
    assert_eq!(fixture.schema, FIXTURE_SCHEMA_V1);

    let names = fixture
        .status_states
        .iter()
        .map(|state| state.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        BTreeSet::from([
            "active",
            "degraded",
            "empty",
            "force_anchor",
            "incompatible",
            "learning",
            "migration",
            "paused",
        ])
    );
    assert!(
        fixture
            .status_states
            .iter()
            .filter(|state| !state.overview_available)
            .all(|state| matches!(state.name.as_str(), "empty" | "migration" | "incompatible"))
    );
    assert!(
        fixture
            .status_states
            .iter()
            .all(|state| state.status.schema == STATUS_REPORT_SCHEMA_V1)
    );

    assert_eq!(fixture.overview.schema, OVERVIEW_REPORT_SCHEMA_V1);
    assert_eq!(
        fixture.overview.snapshot_time_unix_ms - fixture.overview.window_start_unix_ms,
        OVERVIEW_WINDOW_MS_V1
    );
    assert_eq!(
        fixture.overview.decisions.total,
        fixture.overview.decisions.recommend + fixture.overview.decisions.active
    );
    assert_eq!(
        fixture.overview.decisions.total,
        fixture.overview.decisions.candidate_served + fixture.overview.decisions.anchor_served
    );
    assert_eq!(
        fixture.overview.decisions.active,
        fixture.overview.exposures.candidate_treatment
            + fixture.overview.exposures.anchor_control
            + fixture.overview.exposures.anchor_holdout
            + fixture.overview.exposures.non_learning
    );
    for arm in [
        &fixture.overview.outcomes.candidate_treatment,
        &fixture.overview.outcomes.anchor_control,
        &fixture.overview.outcomes.anchor_holdout,
        &fixture.overview.outcomes.non_learning,
    ] {
        assert_eq!(arm.total, arm.success + arm.failure + arm.unlabeled);
    }
    assert_eq!(fixture.pool_detail.schema, POOL_DETAIL_SCHEMA_V1);
    assert_eq!(fixture.pool_detail.summary.id, "pool-a");
    let learning = fixture.pool_detail.learning.as_ref().unwrap();
    assert!(
        learning
            .radius
            .is_some_and(|value| (0.0..=2.0).contains(&value))
    );
    assert!(
        learning
            .min_coverage
            .is_some_and(|value| (0.0..=1.0).contains(&value))
    );
    assert_eq!(fixture.evidence_page.items.len(), 1);
    assert_eq!(
        fixture.evidence_detail.summary,
        fixture.evidence_page.items[0]
    );
    assert_eq!(fixture.neighborhood.schema, NEIGHBORHOOD_REPORT_SCHEMA_V1);
    let projection = fixture.neighborhood.projection.as_ref().unwrap();
    assert_eq!(projection.algorithm, "pca_2");
    assert_eq!(projection.algorithm_version, 1);
    assert!(projection.diagnostic_only);
    assert!(projection.points.len() >= 3);
    assert_eq!(fixture.decision_page.items.len(), 1);
    assert_eq!(
        fixture.decision_detail.summary,
        fixture.decision_page.items[0]
    );
    assert!(
        fixture
            .decision_exposures
            .iter()
            .all(|exposure| exposure.schema == DECISION_EXPOSURE_SCHEMA_V1)
    );
    assert!(
        fixture
            .decision_exposures
            .iter()
            .any(|exposure| exposure.active.is_none())
    );
    assert!(
        fixture
            .decision_exposures
            .iter()
            .any(|exposure| exposure.active.is_some() && exposure.outcome.is_some())
    );
    let active = fixture
        .decision_exposures
        .iter()
        .find_map(|exposure| exposure.active.as_ref())
        .unwrap();
    assert!((0.0..=1.0).contains(&active.propensity));
    assert_eq!(
        active.propensity,
        active.effective_arm_probability * active.conditional_selection_probability
    );
    assert_eq!(fixture.outcome_page.items.len(), 1);
    assert_eq!(fixture.controls.items.len(), 1);
    assert_eq!(fixture.health.items.len(), 1);
    assert_eq!(fixture.migrations.items.len(), 2);

    // Keep the export schema in the shared contract registry even though export
    // itself remains streamed and is not embedded in this JSON response fixture.
    assert_eq!(
        EVIDENCE_EXPORT_RECORD_SCHEMA_V1,
        "nemo.relay.router.evidence-export-record@1"
    );
}

#[test]
fn dashboard_response_contracts_reject_additive_fields_until_a_new_schema_is_used() {
    let fixture: Value = serde_json::from_slice(fixture_bytes()).unwrap();
    let mut overview = fixture["overview"].clone();
    overview
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), Value::Bool(true));
    assert!(
        serde_json::from_value::<nemo_relay_router::inspection::OverviewReportV1>(overview)
            .is_err()
    );

    let mut pool_detail = fixture["pool_detail"].clone();
    pool_detail
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), Value::Bool(true));
    assert!(serde_json::from_value::<PoolDetailV1>(pool_detail).is_err());

    let mut value = fixture["decision_exposures"][1].clone();
    value
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), Value::Bool(true));
    assert!(serde_json::from_value::<DecisionExposureV1>(value).is_err());
}
