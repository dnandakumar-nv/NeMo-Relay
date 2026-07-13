// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact capability-eligible candidate-set identity for recommendation.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
use crate::trajectory::PersistedCandidateFactV1;

pub(crate) const CANDIDATE_SET_SCHEMA_V1: &str = "nemo.relay.router.candidate-set@1";
pub(crate) const RECOMMEND_CANDIDATE_MAX: usize = crate::config::LEARNING_CANDIDATES_MAX;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CandidateSetMemberInputV1 {
    pub(crate) candidate_id: String,
    pub(crate) model: String,
    pub(crate) model_revision: String,
    pub(crate) cost_rank: u32,
}

/// Canonical membership document and hash for one exact live preflight set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateSetArtifactV1 {
    pub(crate) canonical_json: String,
    pub(crate) candidate_set_hash: String,
    pub(crate) candidate_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateSetError {
    Empty,
    TooMany,
    Duplicate,
    NotSorted,
    Canonicalization,
}

/// Hash the full live set in normative `(cost_rank, candidate_id)` order.
pub(crate) fn build_candidate_set_v1(
    candidates: &[PersistedCandidateFactV1],
) -> Result<CandidateSetArtifactV1, CandidateSetError> {
    let members = candidates
        .iter()
        .map(|candidate| CandidateSetMemberInputV1 {
            candidate_id: candidate.candidate_id.clone(),
            model: candidate.model.clone(),
            model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
        })
        .collect::<Vec<_>>();
    build_candidate_set_from_members_v1(&members)
}

pub(crate) fn build_candidate_set_from_members_v1(
    candidates: &[CandidateSetMemberInputV1],
) -> Result<CandidateSetArtifactV1, CandidateSetError> {
    if candidates.is_empty() {
        return Err(CandidateSetError::Empty);
    }
    if candidates.len() > RECOMMEND_CANDIDATE_MAX {
        return Err(CandidateSetError::TooMany);
    }
    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(CandidateSetError::Duplicate);
        }
    }
    for pair in candidates.windows(2) {
        let left = (pair[0].cost_rank, pair[0].candidate_id.as_str());
        let right = (pair[1].cost_rank, pair[1].candidate_id.as_str());
        if left == right {
            return Err(CandidateSetError::Duplicate);
        }
        if left > right {
            return Err(CandidateSetError::NotSorted);
        }
    }

    let document = serde_json::json!({
        "schema": CANDIDATE_SET_SCHEMA_V1,
        "candidates": candidates,
    });
    let canonical_bytes =
        canonical_serialize_bytes(&document).map_err(|_| CandidateSetError::Canonicalization)?;
    let canonical_json =
        String::from_utf8(canonical_bytes).map_err(|_| CandidateSetError::Canonicalization)?;
    let candidate_set_hash = sha256_hex(canonical_json.as_bytes());
    Ok(CandidateSetArtifactV1 {
        canonical_json,
        candidate_set_hash,
        candidate_count: candidates.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{CANDIDATE_FACT_SCHEMA_V1, PersistedCandidateCapabilitiesV1};

    fn candidate(id: &str, model: &str, rank: u32) -> PersistedCandidateFactV1 {
        PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: id.to_string(),
            model: model.to_string(),
            model_revision: "revision".to_string(),
            cost_rank: rank,
            capabilities: PersistedCandidateCapabilitiesV1 {
                tools: false,
                multimodal_input: false,
                structured_output: false,
                reasoning_controls: false,
            },
            decoding_fingerprint: "d".repeat(64),
        }
    }

    #[test]
    fn candidate_set_is_exact_ordered_membership_not_decoding_state() {
        let first = candidate("a", "small-a", 1);
        let mut second = candidate("b", "small-b", 2);
        let baseline = build_candidate_set_v1(&[first.clone(), second.clone()]).unwrap();
        second.decoding_fingerprint = "e".repeat(64);
        let decoding_change = build_candidate_set_v1(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(baseline, decoding_change);

        second.model_revision = "revision-2".to_string();
        assert_ne!(
            baseline.candidate_set_hash,
            build_candidate_set_v1(&[first, second])
                .unwrap()
                .candidate_set_hash
        );
        assert_eq!(baseline.candidate_count, 2);
        assert_eq!(
            baseline.candidate_set_hash,
            "1b69fd1a71603fca233bdbcec752f47f99ad0e6a93b4598bde0fa25804d7e2d5"
        );
        assert_eq!(
            baseline.canonical_json,
            "{\"candidates\":[{\"candidate_id\":\"a\",\"cost_rank\":1,\"model\":\"small-a\",\"model_revision\":\"revision\"},{\"candidate_id\":\"b\",\"cost_rank\":2,\"model\":\"small-b\",\"model_revision\":\"revision\"}],\"schema\":\"nemo.relay.router.candidate-set@1\"}"
        );
    }

    #[test]
    fn candidate_set_rejects_empty_duplicate_unsorted_and_oversized_inputs() {
        assert_eq!(build_candidate_set_v1(&[]), Err(CandidateSetError::Empty));
        let same = candidate("a", "small", 1);
        assert_eq!(
            build_candidate_set_v1(&[same.clone(), same]),
            Err(CandidateSetError::Duplicate)
        );
        assert_eq!(
            build_candidate_set_v1(&[candidate("b", "small-b", 2), candidate("a", "small-a", 1),]),
            Err(CandidateSetError::NotSorted)
        );
        let oversized = (0..=RECOMMEND_CANDIDATE_MAX)
            .map(|index| candidate(&format!("c{index:02}"), "small", index as u32))
            .collect::<Vec<_>>();
        assert_eq!(
            build_candidate_set_v1(&oversized),
            Err(CandidateSetError::TooMany)
        );
    }
}
