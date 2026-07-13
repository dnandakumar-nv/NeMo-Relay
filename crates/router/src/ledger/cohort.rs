// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protected version-1 cohort assignment and non-secret propensity facts.

use std::fmt;

use hmac::{Hmac, KeyInit, Mac};
use serde::{Serialize, Serializer};
use sha2::Sha256;
use uuid::Uuid;

use super::model::CohortSalt;
use crate::config::exact_probability_threshold;

const ROOT_KEY_DOMAIN_V1: &[u8] = b"nemo-relay-router/root-key/v1\0";
const HOLDOUT_DOMAIN_V1: &[u8] = b"nemo-relay-router/holdout/v1\0";
const CANARY_DOMAIN_V1: &[u8] = b"nemo-relay-router/canary/v1\0";
const PINNED_OWNER_RELATION_DOMAIN_V1: &[u8] = b"nemo-relay-router/pinned-owner-relation/v1\0";
const DIGEST_BYTES: usize = 32;
const INVERSE_TWO_POW_64: f64 = 5.421_010_862_427_522e-20;
const INVERSE_TWO_POW_128: f64 = 2.938_735_877_055_719e-39;
const DETERMINISTIC_PROBABILITY_BITS: u64 = 1.0_f64.to_bits();

type HmacSha256 = Hmac<Sha256>;

/// Opaque process-local authority for pseudonymizing raw Core roots.
///
/// The authority owns zeroizing key material and deliberately implements neither
/// `Clone`, `Debug`, nor serialization. Only safe derived facts leave it.
pub(crate) struct CohortAssignmentAuthority {
    cohort_generation_id: Uuid,
    salt: CohortSalt,
}

impl CohortAssignmentAuthority {
    /// Bind verified protected material to its append-only cohort generation.
    pub(super) const fn new(cohort_generation_id: Uuid, salt: CohortSalt) -> Self {
        Self {
            cohort_generation_id,
            salt,
        }
    }

    /// Return the non-secret generation that owns every derived assignment.
    pub(crate) const fn cohort_generation_id(&self) -> Uuid {
        self.cohort_generation_id
    }

    /// Pseudonymize and assign one raw root without exposing protected key material.
    pub(crate) fn assign(
        &self,
        request: CohortAssignmentRequest<'_>,
    ) -> Result<RandomizedCohortAssignment, CohortAssignmentError> {
        assign_randomized_cohort(self.cohort_generation_id, &self.salt, request)
    }

    /// Pseudonymize one raw root on a deterministic path excluded from learning.
    pub(crate) fn assign_non_learning(
        &self,
        raw_root_uuid: Uuid,
        cohort: NonLearningCohort,
    ) -> DeterministicCohortAssignment {
        DeterministicCohortAssignment {
            cohort_generation_id: self.cohort_generation_id,
            root_key: derive_root_key(&self.salt, raw_root_uuid),
            cohort,
            propensity: PropensityFacts::deterministic(),
        }
    }

    /// Protect the memory-only pinned owner in the scope of one pseudonymous root.
    pub(crate) fn protect_pinned_owner(
        &self,
        root_key: RootKey,
        pinned_owner_uuid: Uuid,
    ) -> PinnedOwnerRelationHash {
        PinnedOwnerRelationHash(hmac_digest(
            &self.salt,
            &[
                PINNED_OWNER_RELATION_DOMAIN_V1,
                root_key.as_bytes(),
                pinned_owner_uuid.as_bytes(),
            ],
        ))
    }
}

/// A safe pseudonymous identifier derived from one raw Core root.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RootKey([u8; DIGEST_BYTES]);

impl RootKey {
    /// Return the lowercase hexadecimal persistence representation.
    pub(crate) fn to_hex(self) -> String {
        lowercase_hex(&self.0)
    }

    fn as_bytes(&self) -> &[u8; DIGEST_BYTES] {
        &self.0
    }

    #[cfg(test)]
    pub(crate) const fn from_test_bytes(bytes: [u8; DIGEST_BYTES]) -> Self {
        Self(bytes)
    }
}

impl fmt::Debug for RootKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("RootKey")
            .field(&self.to_hex())
            .finish()
    }
}

impl Serialize for RootKey {
    fn serialize<SerializerT>(
        &self,
        serializer: SerializerT,
    ) -> Result<SerializerT::Ok, SerializerT::Error>
    where
        SerializerT: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

/// Safe protected relation between a pseudonymous root and its memory-only owner.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PinnedOwnerRelationHash([u8; DIGEST_BYTES]);

impl PinnedOwnerRelationHash {
    /// Return the lowercase hexadecimal persistence representation.
    pub(crate) fn to_hex(self) -> String {
        lowercase_hex(&self.0)
    }
}

impl fmt::Debug for PinnedOwnerRelationHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PinnedOwnerRelationHash")
            .field(&self.to_hex())
            .finish()
    }
}

impl Serialize for PinnedOwnerRelationHash {
    fn serialize<SerializerT>(
        &self,
        serializer: SerializerT,
    ) -> Result<SerializerT::Ok, SerializerT::Error>
    where
        SerializerT: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

/// One learning-eligible randomized arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RandomizedCohort {
    /// Monitoring-only anchor holdout.
    AnchorHoldout,
    /// Candidate treatment arm.
    ActiveCanary,
    /// Anchor control for the selected candidate.
    AnchorControl,
}

impl RandomizedCohort {
    /// Return the stable persistence code.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::AnchorHoldout => "anchor_holdout",
            Self::ActiveCanary => "active_canary",
            Self::AnchorControl => "anchor_control",
        }
    }
}

/// One distinct deterministic, non-learning cohort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum NonLearningCohort {
    /// An operator force-anchor control prevented randomization.
    ForcedAnchor,
    /// A pause control prevented randomization.
    Paused,
    /// The call did not pass the Active eligibility gate.
    Ineligible,
    /// The experiment had exhausted its bounded admission capacity.
    Exhausted,
    /// Durable storage was unavailable before randomization.
    StorageFallback,
}

impl NonLearningCohort {
    /// Return the stable persistence code.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ForcedAnchor => "forced_anchor",
            Self::Paused => "paused",
            Self::Ineligible => "ineligible",
            Self::Exhausted => "exhausted",
            Self::StorageFallback => "storage_fallback",
        }
    }
}

/// Binary64 audit facts for one assignment and deterministic candidate selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct PropensityFacts {
    effective_arm_probability_bits: u64,
    conditional_selection_probability_bits: u64,
    propensity_bits: u64,
}

impl PropensityFacts {
    /// Return the exact binary64 bits of the effective arm probability.
    pub(crate) const fn effective_arm_probability_bits(self) -> u64 {
        self.effective_arm_probability_bits
    }

    /// Return the exact binary64 bits of the conditional selection probability.
    pub(crate) const fn conditional_selection_probability_bits(self) -> u64 {
        self.conditional_selection_probability_bits
    }

    /// Return the exact binary64 bits of the complete propensity.
    pub(crate) const fn propensity_bits(self) -> u64 {
        self.propensity_bits
    }

    fn randomized(effective_arm_probability: f64) -> Self {
        let conditional_selection_probability = 1.0_f64;
        let propensity = effective_arm_probability * conditional_selection_probability;
        Self {
            effective_arm_probability_bits: effective_arm_probability.to_bits(),
            conditional_selection_probability_bits: conditional_selection_probability.to_bits(),
            propensity_bits: propensity.to_bits(),
        }
    }

    const fn deterministic() -> Self {
        Self {
            effective_arm_probability_bits: DETERMINISTIC_PROBABILITY_BITS,
            conditional_selection_probability_bits: DETERMINISTIC_PROBABILITY_BITS,
            propensity_bits: DETERMINISTIC_PROBABILITY_BITS,
        }
    }
}

/// Complete non-secret facts for one randomized raw-root assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct RandomizedCohortAssignment {
    cohort_generation_id: Uuid,
    root_key: RootKey,
    cohort: RandomizedCohort,
    holdout_draw: u64,
    canary_draw: u64,
    holdout_threshold_numerator: u128,
    canary_threshold_numerator: u128,
    configured_holdout_probability_bits: u64,
    configured_active_canary_fraction_bits: u64,
    propensity: PropensityFacts,
}

impl RandomizedCohortAssignment {
    /// Return the append-only cohort generation used for this assignment.
    pub(crate) const fn cohort_generation_id(&self) -> Uuid {
        self.cohort_generation_id
    }

    /// Return the pseudonymous raw-root identifier.
    pub(crate) const fn root_key(&self) -> RootKey {
        self.root_key
    }

    /// Return the randomized cohort.
    pub(crate) const fn cohort(&self) -> RandomizedCohort {
        self.cohort
    }

    /// Return the unsigned big-endian holdout draw.
    pub(crate) const fn holdout_draw(&self) -> u64 {
        self.holdout_draw
    }

    /// Return the unsigned big-endian conditional-canary draw.
    pub(crate) const fn canary_draw(&self) -> u64 {
        self.canary_draw
    }

    /// Return `floor(holdout_probability * 2^64)`.
    pub(crate) const fn holdout_threshold_numerator(&self) -> u128 {
        self.holdout_threshold_numerator
    }

    /// Return `floor(conditional_canary_probability * 2^64)`.
    pub(crate) const fn canary_threshold_numerator(&self) -> u128 {
        self.canary_threshold_numerator
    }

    /// Return the configured holdout probability's exact binary64 bits.
    pub(crate) const fn configured_holdout_probability_bits(&self) -> u64 {
        self.configured_holdout_probability_bits
    }

    /// Return the configured unconditional canary fraction's exact binary64 bits.
    pub(crate) const fn configured_active_canary_fraction_bits(&self) -> u64 {
        self.configured_active_canary_fraction_bits
    }

    /// Return the effective randomized arm and complete propensity facts.
    pub(crate) const fn propensity(&self) -> PropensityFacts {
        self.propensity
    }
}

/// Complete facts for a deterministic path that cannot enter randomized learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct DeterministicCohortAssignment {
    cohort_generation_id: Uuid,
    root_key: RootKey,
    cohort: NonLearningCohort,
    propensity: PropensityFacts,
}

impl DeterministicCohortAssignment {
    /// Return the append-only cohort generation used for this assignment.
    pub(crate) const fn cohort_generation_id(self) -> Uuid {
        self.cohort_generation_id
    }

    /// Return the pseudonymous raw-root identifier.
    pub(crate) const fn root_key(self) -> RootKey {
        self.root_key
    }

    /// Return the non-learning cohort.
    pub(crate) const fn cohort(self) -> NonLearningCohort {
        self.cohort
    }

    /// Return deterministic probability and propensity facts.
    pub(crate) const fn propensity(self) -> PropensityFacts {
        self.propensity
    }
}

/// Memory-only inputs accepted by the protected repository assignment boundary.
///
/// This type deliberately implements neither `Clone`, `Debug`, nor serialization
/// because it carries the raw root UUID before pseudonymization.
pub(crate) struct CohortAssignmentRequest<'a> {
    raw_root_uuid: Uuid,
    config_generation_id: &'a str,
    pool_id: &'a str,
    candidate_id: &'a str,
    holdout_probability: f64,
    active_canary_fraction: f64,
}

impl<'a> CohortAssignmentRequest<'a> {
    /// Bind one raw root and versioned candidate identity to configured probabilities.
    pub(crate) const fn new(
        raw_root_uuid: Uuid,
        config_generation_id: &'a str,
        pool_id: &'a str,
        candidate_id: &'a str,
        holdout_probability: f64,
        active_canary_fraction: f64,
    ) -> Self {
        Self {
            raw_root_uuid,
            config_generation_id,
            pool_id,
            candidate_id,
            holdout_probability,
            active_canary_fraction,
        }
    }
}

/// Safe class for an invalid internal assignment request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CohortAssignmentError {
    /// One configured probability was invalid or produced an unreachable threshold.
    InvalidProbability,
    /// One canonical length-prefixed HMAC input exceeded `u32::MAX` bytes.
    InputTooLong,
}

/// Derive one randomized assignment without consulting caps or mutable state.
pub(super) fn assign_randomized_cohort(
    cohort_generation_id: Uuid,
    salt: &CohortSalt,
    request: CohortAssignmentRequest<'_>,
) -> Result<RandomizedCohortAssignment, CohortAssignmentError> {
    let holdout_threshold = exact_probability_threshold(request.holdout_probability)
        .ok_or(CohortAssignmentError::InvalidProbability)?;
    let remaining_probability = 1.0_f64 - request.holdout_probability;
    let conditional_canary_probability = request.active_canary_fraction / remaining_probability;
    let canary_threshold = exact_probability_threshold(conditional_canary_probability)
        .ok_or(CohortAssignmentError::InvalidProbability)?;

    let root_key = derive_root_key(salt, request.raw_root_uuid);
    let holdout_digest = derive_holdout_digest(salt, &root_key);
    let canary_digest = derive_canary_digest(
        salt,
        &root_key,
        request.config_generation_id,
        request.pool_id,
        request.candidate_id,
    )?;
    let holdout_draw = digest_draw(&holdout_digest);
    let canary_draw = digest_draw(&canary_digest);
    let cohort = classify_draws(
        holdout_draw,
        canary_draw,
        holdout_threshold,
        canary_threshold,
    );
    let effective_arm_probability =
        effective_arm_probability(cohort, holdout_threshold, canary_threshold);

    Ok(RandomizedCohortAssignment {
        cohort_generation_id,
        root_key,
        cohort,
        holdout_draw,
        canary_draw,
        holdout_threshold_numerator: holdout_threshold,
        canary_threshold_numerator: canary_threshold,
        configured_holdout_probability_bits: request.holdout_probability.to_bits(),
        configured_active_canary_fraction_bits: request.active_canary_fraction.to_bits(),
        propensity: PropensityFacts::randomized(effective_arm_probability),
    })
}

fn derive_root_key(salt: &CohortSalt, raw_root_uuid: Uuid) -> RootKey {
    RootKey(hmac_digest(
        salt,
        &[ROOT_KEY_DOMAIN_V1, raw_root_uuid.as_bytes()],
    ))
}

fn derive_holdout_digest(salt: &CohortSalt, root_key: &RootKey) -> [u8; DIGEST_BYTES] {
    hmac_digest(salt, &[HOLDOUT_DOMAIN_V1, root_key.as_bytes()])
}

fn derive_canary_digest(
    salt: &CohortSalt,
    root_key: &RootKey,
    config_generation_id: &str,
    pool_id: &str,
    candidate_id: &str,
) -> Result<[u8; DIGEST_BYTES], CohortAssignmentError> {
    let config_length = encoded_length(config_generation_id)?;
    let pool_length = encoded_length(pool_id)?;
    let candidate_length = encoded_length(candidate_id)?;
    Ok(hmac_digest(
        salt,
        &[
            CANARY_DOMAIN_V1,
            root_key.as_bytes(),
            &config_length,
            config_generation_id.as_bytes(),
            &pool_length,
            pool_id.as_bytes(),
            &candidate_length,
            candidate_id.as_bytes(),
        ],
    ))
}

fn encoded_length(value: &str) -> Result<[u8; 4], CohortAssignmentError> {
    u32::try_from(value.len())
        .map(u32::to_be_bytes)
        .map_err(|_| CohortAssignmentError::InputTooLong)
}

fn hmac_digest(salt: &CohortSalt, parts: &[&[u8]]) -> [u8; DIGEST_BYTES] {
    let mut hmac = HmacSha256::new_from_slice(salt.as_bytes())
        .expect("HMAC-SHA256 accepts keys of every length");
    for part in parts {
        hmac.update(part);
    }
    hmac.finalize().into_bytes().into()
}

fn digest_draw(digest: &[u8; DIGEST_BYTES]) -> u64 {
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 always contains an eight-byte draw"),
    )
}

fn classify_draws(
    holdout_draw: u64,
    canary_draw: u64,
    holdout_threshold: u128,
    canary_threshold: u128,
) -> RandomizedCohort {
    if u128::from(holdout_draw) < holdout_threshold {
        RandomizedCohort::AnchorHoldout
    } else if u128::from(canary_draw) < canary_threshold {
        RandomizedCohort::ActiveCanary
    } else {
        RandomizedCohort::AnchorControl
    }
}

fn effective_arm_probability(
    cohort: RandomizedCohort,
    holdout_threshold: u128,
    canary_threshold: u128,
) -> f64 {
    let scale = 1_u128 << 64;
    match cohort {
        RandomizedCohort::AnchorHoldout => holdout_threshold as f64 * INVERSE_TWO_POW_64,
        RandomizedCohort::ActiveCanary => {
            ((scale - holdout_threshold) * canary_threshold) as f64 * INVERSE_TWO_POW_128
        }
        RandomizedCohort::AnchorControl => {
            ((scale - holdout_threshold) * (scale - canary_threshold)) as f64 * INVERSE_TWO_POW_128
        }
    }
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(DIGITS[usize::from(byte >> 4)]));
        encoded.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use static_assertions::assert_not_impl_any;
    use zeroize::Zeroizing;

    use super::*;
    use crate::config::{
        ACTIVE_COHORT_ASSIGNMENT_ID_V1, ACTIVE_COHORT_HMAC_ID_V1, ACTIVE_THRESHOLD_ID_V1,
    };
    use crate::ledger::model::{COHORT_ASSIGNMENT_ALGORITHM_V1, COHORT_SALT_BYTES};

    assert_not_impl_any!(CohortSalt: Clone, fmt::Debug, Serialize);
    assert_not_impl_any!(CohortAssignmentAuthority: Clone, fmt::Debug, Serialize);
    assert_not_impl_any!(CohortAssignmentRequest<'static>: Clone, fmt::Debug, Serialize);

    const CONFIG_GENERATION_ID: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const CHANGED_CONFIG_GENERATION_ID: &str =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const ROOT_UUID: &str = "00112233-4455-6677-8899-aabbccddeeff";

    fn cohort_generation_id() -> Uuid {
        Uuid::parse_str("01890f3e-2d4c-7abc-8def-0123456789ab").unwrap()
    }

    fn fixed_salt() -> CohortSalt {
        let mut bytes = Zeroizing::new([0_u8; COHORT_SALT_BYTES]);
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::try_from(index).unwrap();
        }
        CohortSalt::new(bytes)
    }

    fn request() -> CohortAssignmentRequest<'static> {
        CohortAssignmentRequest::new(
            Uuid::parse_str(ROOT_UUID).unwrap(),
            CONFIG_GENERATION_ID,
            "pool-a",
            "candidate-z",
            0.125,
            0.25,
        )
    }

    #[test]
    fn algorithm_and_domain_identities_are_pinned() {
        assert_eq!(ACTIVE_COHORT_ASSIGNMENT_ID_V1, "cohort_assignment_v1");
        assert_eq!(ACTIVE_COHORT_HMAC_ID_V1, "hmac-sha256-v1");
        assert_eq!(COHORT_ASSIGNMENT_ALGORITHM_V1, ACTIVE_COHORT_HMAC_ID_V1);
        assert_eq!(ACTIVE_THRESHOLD_ID_V1, "binary64_floor_u64_threshold_v1");
        assert_eq!(ROOT_KEY_DOMAIN_V1, b"nemo-relay-router/root-key/v1\0");
        assert_eq!(HOLDOUT_DOMAIN_V1, b"nemo-relay-router/holdout/v1\0");
        assert_eq!(CANARY_DOMAIN_V1, b"nemo-relay-router/canary/v1\0");
        assert_eq!(
            PINNED_OWNER_RELATION_DOMAIN_V1,
            b"nemo-relay-router/pinned-owner-relation/v1\0"
        );
    }

    #[test]
    fn fixed_hmac_bytes_draws_thresholds_and_propensity_match_golden() {
        let salt = fixed_salt();
        let root_uuid = Uuid::parse_str(ROOT_UUID).unwrap();
        let root_key = derive_root_key(&salt, root_uuid);
        let holdout_digest = derive_holdout_digest(&salt, &root_key);
        let canary_digest = derive_canary_digest(
            &salt,
            &root_key,
            CONFIG_GENERATION_ID,
            "pool-a",
            "candidate-z",
        )
        .unwrap();
        assert_eq!(
            root_key.to_hex(),
            "f2c0b48b840e09b1521e8810433ff78f8d7064abe5ecdd4212dd981f63b01e89"
        );
        assert_eq!(
            lowercase_hex(&holdout_digest),
            "f2ae7477767f1f5eaf3072a28d409f47440da9b656cc6dc03b3a7c6773424529"
        );
        assert_eq!(
            lowercase_hex(&canary_digest),
            "2bf301e4fb32cb620f4ecb141e77c0a24f66b486347744c812c0764a1a417449"
        );
        assert_eq!(
            CohortAssignmentAuthority::new(cohort_generation_id(), fixed_salt())
                .protect_pinned_owner(
                    root_key,
                    Uuid::parse_str("10213243-5465-7687-98a9-bacbdcedfe0f").unwrap(),
                )
                .to_hex(),
            "fbb17285c8173377e37120202ad02792c05cb931a93d1e9377bde6b2856035ef"
        );

        let assignment =
            assign_randomized_cohort(cohort_generation_id(), &salt, request()).unwrap();
        assert_eq!(assignment.cohort_generation_id(), cohort_generation_id());
        assert_eq!(assignment.root_key(), root_key);
        assert_eq!(assignment.cohort(), RandomizedCohort::ActiveCanary);
        assert_eq!(assignment.holdout_draw(), 0xf2ae_7477_767f_1f5e);
        assert_eq!(assignment.canary_draw(), 0x2bf3_01e4_fb32_cb62);
        assert_eq!(
            assignment.holdout_threshold_numerator(),
            0x2000_0000_0000_0000
        );
        assert_eq!(
            assignment.canary_threshold_numerator(),
            0x4924_9249_2492_4800
        );
        assert_eq!(
            assignment.configured_holdout_probability_bits(),
            0x3fc0_0000_0000_0000
        );
        assert_eq!(
            assignment.configured_active_canary_fraction_bits(),
            0x3fd0_0000_0000_0000
        );
        assert_eq!(
            assignment.propensity().effective_arm_probability_bits(),
            0x3fd0_0000_0000_0000
        );
        assert_eq!(
            assignment
                .propensity()
                .conditional_selection_probability_bits(),
            DETERMINISTIC_PROBABILITY_BITS
        );
        assert_eq!(
            assignment.propensity().propensity_bits(),
            0x3fd0_0000_0000_0000
        );
    }

    #[test]
    fn draw_comparison_is_integer_only_and_strict_at_each_boundary() {
        let holdout_threshold = 1_u128 << 63;
        let canary_threshold = 1_u128 << 62;
        assert_eq!(
            classify_draws(
                u64::try_from(holdout_threshold - 1).unwrap(),
                u64::MAX,
                holdout_threshold,
                canary_threshold,
            ),
            RandomizedCohort::AnchorHoldout
        );
        assert_eq!(
            classify_draws(
                u64::try_from(holdout_threshold).unwrap(),
                u64::try_from(canary_threshold - 1).unwrap(),
                holdout_threshold,
                canary_threshold,
            ),
            RandomizedCohort::ActiveCanary
        );
        assert_eq!(
            classify_draws(
                u64::try_from(holdout_threshold).unwrap(),
                u64::try_from(canary_threshold).unwrap(),
                holdout_threshold,
                canary_threshold,
            ),
            RandomizedCohort::AnchorControl
        );
    }

    #[test]
    fn exact_thresholds_are_monotone_across_representative_binary64_values() {
        let probabilities = [
            2.0_f64.powi(-64),
            2.0_f64.powi(-32),
            0.01,
            0.125,
            0.5,
            0.75,
            f64::from_bits(1.0_f64.to_bits() - 1),
        ];
        let thresholds =
            probabilities.map(|probability| exact_probability_threshold(probability).unwrap());
        assert!(thresholds.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(thresholds[0], 1);
        assert_eq!(thresholds[3], 1_u128 << 61);
        assert_eq!(thresholds[4], 1_u128 << 63);
        assert_eq!(thresholds[6], (1_u128 << 64) - 2_048);
    }

    #[test]
    fn effective_arm_goldens_use_one_exact_rational_conversion() {
        let holdout_probability = f64::from_bits(0x3fc1_9176_0634_7a8e);
        let active_canary_fraction = f64::from_bits(0x3faa_3814_a884_ca47);
        let holdout_threshold = exact_probability_threshold(holdout_probability).unwrap();
        let conditional_canary_probability =
            active_canary_fraction / (1.0_f64 - holdout_probability);
        let canary_threshold = exact_probability_threshold(conditional_canary_probability).unwrap();
        assert_eq!(holdout_threshold, 0x2322_ec0c_68f5_1c00);
        assert_eq!(canary_threshold, 0x0f31_f0e8_de63_0500);
        assert_eq!(
            effective_arm_probability(
                RandomizedCohort::AnchorHoldout,
                holdout_threshold,
                canary_threshold,
            )
            .to_bits(),
            0x3fc1_9176_0634_7a8e
        );
        assert_eq!(
            effective_arm_probability(
                RandomizedCohort::ActiveCanary,
                holdout_threshold,
                canary_threshold,
            )
            .to_bits(),
            0x3faa_3814_a884_ca48
        );
        assert_eq!(
            effective_arm_probability(
                RandomizedCohort::AnchorControl,
                holdout_threshold,
                canary_threshold,
            )
            .to_bits(),
            0x3fe9_f821_33ea_94b8
        );
    }

    #[test]
    fn root_holdout_and_canary_domains_change_only_for_their_semantic_inputs() {
        let authority = CohortAssignmentAuthority::new(cohort_generation_id(), fixed_salt());
        let raw_root_uuid = Uuid::parse_str(ROOT_UUID).unwrap();
        let base = authority.assign(request()).unwrap();
        for changed in [
            CohortAssignmentRequest::new(
                raw_root_uuid,
                CHANGED_CONFIG_GENERATION_ID,
                "pool-a",
                "candidate-z",
                0.125,
                0.25,
            ),
            CohortAssignmentRequest::new(
                raw_root_uuid,
                CONFIG_GENERATION_ID,
                "pool-b",
                "candidate-z",
                0.125,
                0.25,
            ),
            CohortAssignmentRequest::new(
                raw_root_uuid,
                CONFIG_GENERATION_ID,
                "pool-a",
                "candidate-y",
                0.125,
                0.25,
            ),
        ] {
            let changed = authority.assign(changed).unwrap();
            assert_eq!(changed.root_key(), base.root_key());
            assert_eq!(changed.holdout_draw(), base.holdout_draw());
            assert_ne!(changed.canary_draw(), base.canary_draw());
        }

        let changed_root = authority
            .assign(CohortAssignmentRequest::new(
                Uuid::parse_str("00112233-4455-6677-8899-aabbccddeefe").unwrap(),
                CONFIG_GENERATION_ID,
                "pool-a",
                "candidate-z",
                0.125,
                0.25,
            ))
            .unwrap();
        assert_ne!(changed_root.root_key(), base.root_key());
        assert_ne!(changed_root.holdout_draw(), base.holdout_draw());
        assert_ne!(changed_root.canary_draw(), base.canary_draw());
    }

    #[test]
    fn u32_big_endian_length_prefixes_are_exact_and_boundary_injective() {
        assert_eq!(encoded_length("").unwrap(), [0, 0, 0, 0]);
        assert_eq!(encoded_length("é").unwrap(), [0, 0, 0, 2]);
        assert_eq!(encoded_length(&"a".repeat(255)).unwrap(), [0, 0, 0, 255]);
        assert_eq!(encoded_length(&"a".repeat(256)).unwrap(), [0, 0, 1, 0]);
        assert_eq!(
            encoded_length(&"a".repeat(65_535)).unwrap(),
            [0, 0, 255, 255]
        );
        assert_eq!(encoded_length(&"a".repeat(65_536)).unwrap(), [0, 1, 0, 0]);

        let authority = CohortAssignmentAuthority::new(cohort_generation_id(), fixed_salt());
        let raw_root_uuid = Uuid::parse_str(ROOT_UUID).unwrap();
        let first = authority
            .assign(CohortAssignmentRequest::new(
                raw_root_uuid,
                "ab",
                "c",
                "",
                0.125,
                0.25,
            ))
            .unwrap();
        let second = authority
            .assign(CohortAssignmentRequest::new(
                raw_root_uuid,
                "a",
                "bc",
                "",
                0.125,
                0.25,
            ))
            .unwrap();
        assert_eq!(first.root_key(), second.root_key());
        assert_eq!(first.holdout_draw(), second.holdout_draw());
        assert_ne!(first.canary_draw(), second.canary_draw());
    }

    #[test]
    fn all_deterministic_fallbacks_are_distinct_non_learning_probability_one() {
        let cohorts = [
            NonLearningCohort::ForcedAnchor,
            NonLearningCohort::Paused,
            NonLearningCohort::Ineligible,
            NonLearningCohort::Exhausted,
            NonLearningCohort::StorageFallback,
        ];
        let mut codes = cohorts
            .map(NonLearningCohort::as_str)
            .into_iter()
            .collect::<Vec<_>>();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), cohorts.len());
        let authority = CohortAssignmentAuthority::new(cohort_generation_id(), fixed_salt());
        let raw_root_uuid = Uuid::parse_str(ROOT_UUID).unwrap();
        let expected_root_key = authority.assign(request()).unwrap().root_key();
        for cohort in cohorts {
            let assignment = authority.assign_non_learning(raw_root_uuid, cohort);
            assert_eq!(assignment.cohort_generation_id(), cohort_generation_id());
            assert_eq!(assignment.root_key(), expected_root_key);
            assert_eq!(assignment.cohort(), cohort);
            assert_eq!(
                assignment.propensity(),
                PropensityFacts {
                    effective_arm_probability_bits: DETERMINISTIC_PROBABILITY_BITS,
                    conditional_selection_probability_bits: DETERMINISTIC_PROBABILITY_BITS,
                    propensity_bits: DETERMINISTIC_PROBABILITY_BITS,
                }
            );
        }
    }

    #[test]
    fn salt_rotation_changes_only_protected_derived_identity() {
        let first =
            assign_randomized_cohort(cohort_generation_id(), &fixed_salt(), request()).unwrap();
        let second_salt = CohortSalt::new(Zeroizing::new([0xa5; COHORT_SALT_BYTES]));
        let second_generation = Uuid::parse_str("01890f3e-2d4c-7abc-8def-0123456789ac").unwrap();
        let second = assign_randomized_cohort(second_generation, &second_salt, request()).unwrap();
        assert_ne!(first.cohort_generation_id(), second.cohort_generation_id());
        assert_ne!(first.root_key(), second.root_key());
        assert_eq!(
            first.holdout_threshold_numerator(),
            second.holdout_threshold_numerator()
        );
        assert_eq!(
            first.canary_threshold_numerator(),
            second.canary_threshold_numerator()
        );
        assert_eq!(
            first.configured_holdout_probability_bits(),
            second.configured_holdout_probability_bits()
        );
        assert_eq!(
            first.configured_active_canary_fraction_bits(),
            second.configured_active_canary_fraction_bits()
        );
    }

    #[test]
    fn safe_debug_and_serde_views_never_contain_salt_or_raw_root() {
        let authority = CohortAssignmentAuthority::new(cohort_generation_id(), fixed_salt());
        let assignment = authority.assign(request()).unwrap();
        let deterministic = authority.assign_non_learning(
            Uuid::parse_str(ROOT_UUID).unwrap(),
            NonLearningCohort::Paused,
        );
        let debug = format!("{assignment:?}");
        let serialized = serde_json::to_string(&assignment).unwrap();
        let deterministic_debug = format!("{deterministic:?}");
        let deterministic_serialized = serde_json::to_string(&deterministic).unwrap();
        let salt_hex = lowercase_hex(&(0_u8..32).collect::<Vec<_>>());
        let raw_root_hex = ROOT_UUID.replace('-', "");
        for view in [
            &debug,
            &serialized,
            &deterministic_debug,
            &deterministic_serialized,
        ] {
            assert!(!view.contains(&salt_hex));
            assert!(!view.contains(ROOT_UUID));
            assert!(!view.contains(&raw_root_hex));
        }
        assert!(serialized.contains(&assignment.root_key().to_hex()));
        assert_eq!(assignment.root_key().to_hex().len(), DIGEST_BYTES * 2);
        assert!(
            assignment
                .root_key()
                .to_hex()
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        );
    }

    #[test]
    fn invalid_probability_shapes_fail_before_hmac_assignment() {
        for (holdout, canary) in [
            (0.0, 0.1),
            (1.0, 0.1),
            (f64::NAN, 0.1),
            (0.1, 0.0),
            (0.1, 0.9),
            (0.1, f64::INFINITY),
        ] {
            let invalid = CohortAssignmentRequest::new(
                Uuid::parse_str(ROOT_UUID).unwrap(),
                CONFIG_GENERATION_ID,
                "pool-a",
                "candidate-z",
                holdout,
                canary,
            );
            assert_eq!(
                assign_randomized_cohort(cohort_generation_id(), &fixed_salt(), invalid),
                Err(CohortAssignmentError::InvalidProbability)
            );
        }
    }
}
