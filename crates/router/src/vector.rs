// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked vector identities, normalization, canonical encoding, and cosine math.

use std::fmt;
use std::sync::Arc;

use uuid::{Uuid, Variant};

use crate::config::EMBEDDING_DIMENSIONS_MAX;
use crate::fingerprint::sha256_hex;

const VECTOR_CHECKSUM_DOMAIN_V1: &[u8] = b"nemo.relay.router.vector-checksum@1\0";
pub(crate) const COSINE_DISTANCE_CLAMP_TOLERANCE: f32 = 1.0e-5;

/// Stable vector validation failures without user-controlled text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorError {
    InvalidSpaceId,
    InvalidRecordId,
    InvalidPartitionId,
    InvalidDimensions,
    VectorSpaceMismatch,
    DimensionMismatch,
    NonFiniteValue,
    ZeroVector,
    InvalidBlobLength,
    InvalidChecksum,
    ChecksumMismatch,
    DistanceOutOfRange,
}

/// Validated lowercase SHA-256 vector-space identity.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VectorSpaceId(String);

impl VectorSpaceId {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, VectorError> {
        let value = value.into();
        if !is_sha256(&value) {
            return Err(VectorError::InvalidSpaceId);
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for VectorSpaceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("VectorSpaceId")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for VectorSpaceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Validated immutable evidence-link identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VectorRecordId(Uuid);

impl VectorRecordId {
    pub(crate) fn new(value: Uuid) -> Result<Self, VectorError> {
        if value.get_version_num() != 7 || value.get_variant() != Variant::RFC4122 {
            return Err(VectorError::InvalidRecordId);
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> Uuid {
        self.0
    }
}

impl fmt::Display for VectorRecordId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Positive SQLite partition key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PartitionId(i64);

impl PartitionId {
    pub(crate) fn new(value: i64) -> Result<Self, VectorError> {
        if value <= 0 {
            return Err(VectorError::InvalidPartitionId);
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> i64 {
        self.0
    }
}

/// Checked version-1 vector dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VectorDimensions(u32);

impl VectorDimensions {
    pub(crate) fn new(value: u32) -> Result<Self, VectorError> {
        if !(1..=EMBEDDING_DIMENSIONS_MAX).contains(&value) {
            return Err(VectorError::InvalidDimensions);
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> u32 {
        self.0
    }

    pub(crate) fn as_usize(self) -> usize {
        usize::try_from(self.0).expect("supported targets represent u32 dimensions as usize")
    }
}

/// Validated lowercase SHA-256 vector checksum.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VectorChecksum(String);

impl VectorChecksum {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, VectorError> {
        let value = value.into();
        if !is_sha256(&value) {
            return Err(VectorError::InvalidChecksum);
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for VectorChecksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("VectorChecksum")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for VectorChecksum {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Finite, nonzero f32 vector produced by the version-1 normalization path.
#[derive(Clone)]
pub(crate) struct NormalizedVector {
    dimensions: VectorDimensions,
    values: Arc<[f32]>,
}

impl PartialEq for NormalizedVector {
    fn eq(&self, other: &Self) -> bool {
        self.bitwise_eq(other)
    }
}

impl Eq for NormalizedVector {}

impl NormalizedVector {
    pub(crate) fn from_provider_f64(
        values: &[f64],
        dimensions: VectorDimensions,
    ) -> Result<Self, VectorError> {
        if values.len() != dimensions.as_usize() {
            return Err(VectorError::DimensionMismatch);
        }

        let mut converted = Vec::with_capacity(values.len());
        let mut norm_square = 0.0_f64;
        for value in values {
            if !value.is_finite() {
                return Err(VectorError::NonFiniteValue);
            }
            let value = *value as f32;
            if !value.is_finite() {
                return Err(VectorError::NonFiniteValue);
            }
            norm_square += f64::from(value) * f64::from(value);
            converted.push(value);
        }
        if norm_square == 0.0 {
            return Err(VectorError::ZeroVector);
        }
        if !norm_square.is_finite() {
            return Err(VectorError::NonFiniteValue);
        }
        let norm = norm_square.sqrt();
        let normalized = converted
            .into_iter()
            .map(|value| (f64::from(value) / norm) as f32)
            .collect::<Vec<_>>();
        Self::from_checked_f32(normalized, dimensions)
    }

    fn from_checked_f32(
        values: Vec<f32>,
        dimensions: VectorDimensions,
    ) -> Result<Self, VectorError> {
        if values.len() != dimensions.as_usize() {
            return Err(VectorError::DimensionMismatch);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(VectorError::NonFiniteValue);
        }
        if !values.iter().any(|value| *value != 0.0) {
            return Err(VectorError::ZeroVector);
        }
        Ok(Self {
            dimensions,
            values: values.into(),
        })
    }

    pub(crate) const fn dimensions(&self) -> VectorDimensions {
        self.dimensions
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn bitwise_eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

impl fmt::Debug for NormalizedVector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NormalizedVector")
            .field("dimensions", &self.dimensions)
            .field("values", &"<redacted>")
            .finish()
    }
}

/// Canonical little-endian vector bytes paired with their checksum.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct AuthoritativeVectorBlob {
    dimensions: VectorDimensions,
    bytes: Arc<[u8]>,
    checksum: VectorChecksum,
}

impl AuthoritativeVectorBlob {
    pub(crate) const fn dimensions(&self) -> VectorDimensions {
        self.dimensions
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn checksum(&self) -> &VectorChecksum {
        &self.checksum
    }

    pub(crate) fn native_endian_bytes(&self) -> Vec<u8> {
        let mut native = Vec::with_capacity(self.bytes.len());
        for component in self.bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(component.try_into().expect("four-byte chunk"));
            native.extend_from_slice(&value.to_ne_bytes());
        }
        native
    }
}

impl fmt::Debug for AuthoritativeVectorBlob {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthoritativeVectorBlob")
            .field("dimensions", &self.dimensions)
            .field("bytes", &"<redacted>")
            .field("checksum", &self.checksum)
            .finish()
    }
}

/// A checked vector and the only authoritative persistence representation of it.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct AuthoritativeVector {
    vector_space_id: VectorSpaceId,
    vector: NormalizedVector,
    blob: AuthoritativeVectorBlob,
}

impl AuthoritativeVector {
    pub(crate) fn from_normalized(
        vector_space_id: &VectorSpaceId,
        vector: NormalizedVector,
    ) -> Result<Self, VectorError> {
        let byte_len = vector
            .values()
            .len()
            .checked_mul(4)
            .ok_or(VectorError::InvalidBlobLength)?;
        let mut bytes = Vec::with_capacity(byte_len);
        for value in vector.values() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let checksum = checksum(vector_space_id, vector.dimensions(), &bytes)?;
        let blob = AuthoritativeVectorBlob {
            dimensions: vector.dimensions(),
            bytes: bytes.into(),
            checksum,
        };
        Ok(Self {
            vector_space_id: vector_space_id.clone(),
            vector,
            blob,
        })
    }

    pub(crate) fn from_blob_verified(
        vector_space_id: &VectorSpaceId,
        dimensions: VectorDimensions,
        bytes: Vec<u8>,
        checksum_value: VectorChecksum,
    ) -> Result<Self, VectorError> {
        let expected_len = dimensions
            .as_usize()
            .checked_mul(4)
            .ok_or(VectorError::InvalidBlobLength)?;
        if bytes.len() != expected_len {
            return Err(VectorError::InvalidBlobLength);
        }
        if checksum(vector_space_id, dimensions, &bytes)? != checksum_value {
            return Err(VectorError::ChecksumMismatch);
        }
        let values = bytes
            .chunks_exact(4)
            .map(|component| f32::from_le_bytes(component.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let vector = NormalizedVector::from_checked_f32(values, dimensions)?;
        let blob = AuthoritativeVectorBlob {
            dimensions,
            bytes: bytes.into(),
            checksum: checksum_value,
        };
        Ok(Self {
            vector_space_id: vector_space_id.clone(),
            vector,
            blob,
        })
    }

    /// Decode sqlite-vec's native-endian f32 storage into the canonical
    /// little-endian authoritative representation and derive its checksum.
    pub(crate) fn from_native_blob(
        vector_space_id: &VectorSpaceId,
        dimensions: VectorDimensions,
        bytes: &[u8],
    ) -> Result<Self, VectorError> {
        let expected_len = dimensions
            .as_usize()
            .checked_mul(4)
            .ok_or(VectorError::InvalidBlobLength)?;
        if bytes.len() != expected_len {
            return Err(VectorError::InvalidBlobLength);
        }
        let values = bytes
            .chunks_exact(4)
            .map(|component| f32::from_ne_bytes(component.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        let vector = NormalizedVector::from_checked_f32(values, dimensions)?;
        Self::from_normalized(vector_space_id, vector)
    }

    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) fn vector(&self) -> &NormalizedVector {
        &self.vector
    }

    pub(crate) fn blob(&self) -> &AuthoritativeVectorBlob {
        &self.blob
    }

    pub(crate) fn bitwise_eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl fmt::Debug for AuthoritativeVector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthoritativeVector")
            .field("vector_space_id", &self.vector_space_id)
            .field("vector", &self.vector)
            .field("blob", &self.blob)
            .finish()
    }
}

/// Compute f32 cosine distance with component-order accumulation.
pub(crate) fn cosine_distance(
    left: &NormalizedVector,
    right: &NormalizedVector,
) -> Result<f32, VectorError> {
    if left.dimensions() != right.dimensions() {
        return Err(VectorError::DimensionMismatch);
    }
    let mut dot = 0.0_f32;
    let mut left_magnitude = 0.0_f32;
    let mut right_magnitude = 0.0_f32;
    for (left, right) in left.values().iter().zip(right.values()) {
        dot += *left * *right;
        left_magnitude += *left * *left;
        right_magnitude += *right * *right;
    }
    let denominator = f64::from(left_magnitude).sqrt() * f64::from(right_magnitude).sqrt();
    if !dot.is_finite() || !denominator.is_finite() {
        return Err(VectorError::NonFiniteValue);
    }
    if denominator == 0.0 {
        return Err(VectorError::ZeroVector);
    }
    let distance = (1.0_f64 - f64::from(dot) / denominator) as f32;
    clamp_cosine_distance(distance)
}

pub(crate) fn clamp_cosine_distance(distance: f32) -> Result<f32, VectorError> {
    if !distance.is_finite() {
        return Err(VectorError::NonFiniteValue);
    }
    if distance < 0.0 {
        return (distance >= -COSINE_DISTANCE_CLAMP_TOLERANCE)
            .then_some(0.0)
            .ok_or(VectorError::DistanceOutOfRange);
    }
    if distance > 2.0 {
        return (distance <= 2.0 + COSINE_DISTANCE_CLAMP_TOLERANCE)
            .then_some(2.0)
            .ok_or(VectorError::DistanceOutOfRange);
    }
    Ok(distance)
}

fn checksum(
    vector_space_id: &VectorSpaceId,
    dimensions: VectorDimensions,
    bytes: &[u8],
) -> Result<VectorChecksum, VectorError> {
    let capacity = VECTOR_CHECKSUM_DOMAIN_V1
        .len()
        .checked_add(vector_space_id.as_str().len())
        .and_then(|value| value.checked_add(4))
        .and_then(|value| value.checked_add(bytes.len()))
        .ok_or(VectorError::InvalidBlobLength)?;
    let mut preimage = Vec::with_capacity(capacity);
    preimage.extend_from_slice(VECTOR_CHECKSUM_DOMAIN_V1);
    preimage.extend_from_slice(vector_space_id.as_str().as_bytes());
    preimage.extend_from_slice(&dimensions.value().to_le_bytes());
    preimage.extend_from_slice(bytes);
    VectorChecksum::new(sha256_hex(&preimage))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;

    fn space(byte: char) -> VectorSpaceId {
        VectorSpaceId::new(byte.to_string().repeat(64)).unwrap()
    }

    #[test]
    fn identifiers_and_dimensions_are_strict() {
        assert!(VectorSpaceId::new("a".repeat(64)).is_ok());
        for invalid in ["a".repeat(63), "A".repeat(64), "g".repeat(64)] {
            assert_eq!(
                VectorSpaceId::new(invalid),
                Err(VectorError::InvalidSpaceId)
            );
        }
        assert_eq!(
            VectorRecordId::new(Uuid::new_v4()),
            Err(VectorError::InvalidRecordId)
        );
        assert!(VectorRecordId::new(Uuid::now_v7()).is_ok());
        assert_eq!(PartitionId::new(0), Err(VectorError::InvalidPartitionId));
        assert_eq!(PartitionId::new(-1), Err(VectorError::InvalidPartitionId));
        assert!(PartitionId::new(1).is_ok());
        assert!(VectorDimensions::new(1).is_ok());
        assert!(VectorDimensions::new(EMBEDDING_DIMENSIONS_MAX).is_ok());
        assert_eq!(
            VectorDimensions::new(0),
            Err(VectorError::InvalidDimensions)
        );
        assert_eq!(
            VectorDimensions::new(EMBEDDING_DIMENSIONS_MAX + 1),
            Err(VectorError::InvalidDimensions)
        );
    }

    #[test]
    fn normalization_blob_endian_and_checksum_are_exact() {
        let dimensions = VectorDimensions::new(2).unwrap();
        let normalized = NormalizedVector::from_provider_f64(&[3.0, 4.0], dimensions).unwrap();
        assert_eq!(normalized.values(), &[0.6_f32, 0.8_f32]);
        let authoritative = AuthoritativeVector::from_normalized(&space('a'), normalized).unwrap();
        let expected = [0.6_f32.to_le_bytes(), 0.8_f32.to_le_bytes()].concat();
        assert_eq!(authoritative.blob().bytes(), expected);
        assert_eq!(
            authoritative.blob().native_endian_bytes(),
            [0.6_f32.to_ne_bytes(), 0.8_f32.to_ne_bytes()].concat()
        );
        assert_eq!(
            authoritative.blob().checksum().as_str(),
            "c2dc85de971bffc3b183209ff10575774cdb23dee907e8c125a3943a73a324e0"
        );

        let decoded = AuthoritativeVector::from_blob_verified(
            &space('a'),
            dimensions,
            expected,
            authoritative.blob().checksum().clone(),
        )
        .unwrap();
        assert!(authoritative.bitwise_eq(&decoded));
        let decoded_native = AuthoritativeVector::from_native_blob(
            &space('a'),
            dimensions,
            &authoritative.blob().native_endian_bytes(),
        )
        .unwrap();
        assert!(authoritative.bitwise_eq(&decoded_native));
    }

    #[test]
    fn checksum_is_sensitive_to_space_dimensions_and_every_blob_byte() {
        let dimensions = VectorDimensions::new(2).unwrap();
        let vector = NormalizedVector::from_provider_f64(&[1.0, 2.0], dimensions).unwrap();
        let first = AuthoritativeVector::from_normalized(&space('a'), vector.clone()).unwrap();
        let other_space = AuthoritativeVector::from_normalized(&space('b'), vector).unwrap();
        assert_ne!(first.blob().checksum(), other_space.blob().checksum());

        for index in 0..first.blob().bytes().len() {
            let mut changed = first.blob().bytes().to_vec();
            changed[index] ^= 1;
            let changed_checksum = checksum(&space('a'), dimensions, &changed).unwrap();
            assert_ne!(first.blob().checksum(), &changed_checksum);
        }

        let one_dimension = VectorDimensions::new(1).unwrap();
        let different_dimensions =
            checksum(&space('a'), one_dimension, first.blob().bytes()).unwrap();
        assert_ne!(first.blob().checksum(), &different_dimensions);
    }

    #[test]
    fn provider_conversion_rejects_nonfinite_overflow_and_zero() {
        let one = VectorDimensions::new(1).unwrap();
        assert_eq!(
            NormalizedVector::from_provider_f64(&[1.0, 2.0], one),
            Err(VectorError::DimensionMismatch)
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            assert_eq!(
                NormalizedVector::from_provider_f64(&[value], one),
                Err(VectorError::NonFiniteValue)
            );
        }
        for value in [0.0, -0.0, f64::MIN_POSITIVE] {
            assert_eq!(
                NormalizedVector::from_provider_f64(&[value], one),
                Err(VectorError::ZeroVector)
            );
        }
        assert!(NormalizedVector::from_provider_f64(&[f64::from(f32::from_bits(1))], one).is_ok());
        assert!(NormalizedVector::from_provider_f64(&[f64::from(f32::MAX)], one).is_ok());
    }

    #[test]
    fn provider_values_are_cast_to_f32_before_normalization() {
        let input = [
            f64::from_bits(0xc201_9207_6b2c_7a9f),
            f64::from_bits(0x41f9_0436_0be6_5bba),
            f64::from_bits(0xc1d4_0977_3479_8820),
        ];
        let normalized =
            NormalizedVector::from_provider_f64(&input, VectorDimensions::new(3).unwrap()).unwrap();
        assert_eq!(
            normalized
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0xbf4f_28db, 0x3f13_799a, 0xbdec_3da6]
        );
    }

    #[test]
    fn blob_verification_rejects_length_checksum_nonfinite_and_zero() {
        let id = space('a');
        let dimensions = VectorDimensions::new(1).unwrap();
        let valid = AuthoritativeVector::from_normalized(
            &id,
            NormalizedVector::from_provider_f64(&[1.0], dimensions).unwrap(),
        )
        .unwrap();
        assert_eq!(
            AuthoritativeVector::from_blob_verified(
                &id,
                dimensions,
                vec![0; 3],
                valid.blob().checksum().clone(),
            ),
            Err(VectorError::InvalidBlobLength)
        );
        assert_eq!(
            AuthoritativeVector::from_blob_verified(
                &id,
                dimensions,
                vec![0; 5],
                valid.blob().checksum().clone(),
            ),
            Err(VectorError::InvalidBlobLength)
        );
        assert_eq!(
            AuthoritativeVector::from_blob_verified(
                &id,
                dimensions,
                valid.blob().bytes().to_vec(),
                VectorChecksum::new("b".repeat(64)).unwrap(),
            ),
            Err(VectorError::ChecksumMismatch)
        );

        for value in [f32::NAN, f32::INFINITY, 0.0] {
            let bytes = value.to_le_bytes().to_vec();
            let checksum = checksum(&id, dimensions, &bytes).unwrap();
            let expected = if value == 0.0 {
                VectorError::ZeroVector
            } else {
                VectorError::NonFiniteValue
            };
            assert!(matches!(
                AuthoritativeVector::from_blob_verified(&id, dimensions, bytes, checksum),
                Err(error) if error == expected
            ));
        }
    }

    #[test]
    fn cosine_and_clamp_boundaries_are_exact() {
        let dimensions = VectorDimensions::new(2).unwrap();
        let x = NormalizedVector::from_provider_f64(&[1.0, 0.0], dimensions).unwrap();
        let same = NormalizedVector::from_provider_f64(&[2.0, 0.0], dimensions).unwrap();
        let opposite = NormalizedVector::from_provider_f64(&[-1.0, 0.0], dimensions).unwrap();
        let orthogonal = NormalizedVector::from_provider_f64(&[0.0, 1.0], dimensions).unwrap();
        assert_eq!(cosine_distance(&x, &same).unwrap(), 0.0);
        assert_eq!(cosine_distance(&x, &opposite).unwrap(), 2.0);
        assert_eq!(cosine_distance(&x, &orthogonal).unwrap(), 1.0);
        assert_eq!(
            clamp_cosine_distance(-COSINE_DISTANCE_CLAMP_TOLERANCE).unwrap(),
            0.0
        );
        assert_eq!(
            clamp_cosine_distance(2.0 + COSINE_DISTANCE_CLAMP_TOLERANCE).unwrap(),
            2.0
        );
        assert_eq!(
            clamp_cosine_distance(-COSINE_DISTANCE_CLAMP_TOLERANCE * 2.0),
            Err(VectorError::DistanceOutOfRange)
        );
        assert_eq!(
            clamp_cosine_distance(2.0 + COSINE_DISTANCE_CLAMP_TOLERANCE * 2.0),
            Err(VectorError::DistanceOutOfRange)
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                clamp_cosine_distance(value),
                Err(VectorError::NonFiniteValue)
            );
        }
        let one =
            NormalizedVector::from_provider_f64(&[1.0], VectorDimensions::new(1).unwrap()).unwrap();
        assert_eq!(
            cosine_distance(&x, &one),
            Err(VectorError::DimensionMismatch)
        );
    }

    #[test]
    fn cosine_uses_sqlite_vec_scalar_final_precision() {
        let dimensions = VectorDimensions::new(7).unwrap();
        let left = NormalizedVector::from_provider_f64(
            &[-923.0, 607.0, 882.0, -936.0, -752.0, -251.0, 21.0],
            dimensions,
        )
        .unwrap();
        let right = NormalizedVector::from_provider_f64(
            &[418.0, 160.0, 928.0, -357.0, -608.0, 906.0, -408.0],
            dimensions,
        )
        .unwrap();
        assert_eq!(
            cosine_distance(&left, &right).unwrap().to_bits(),
            0x3f22_fa71
        );
    }
}
