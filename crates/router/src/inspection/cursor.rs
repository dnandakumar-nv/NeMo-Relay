// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)] // Tasks 2-4 consume the codec through paginated repository reads.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::digest::KeyInit;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{INSPECTION_API_VERSION_V1, INSPECTION_CURSOR_MAX_BYTES, InspectionError};
use crate::canonical_json::canonical_json;

type HmacSha256 = Hmac<Sha256>;
const CURSOR_TAG_BYTES: usize = 32;
const CURSOR_DOMAIN: &[u8] = b"nemo-relay-router-inspection-cursor-v1\0";
const CURSOR_KEY_DOMAIN: &[u8] = b"nemo-relay-router-inspection-cursor-key-v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CursorEndpointV1 {
    Pools,
    Evidence,
    Decisions,
    DecisionFollow,
    Outcomes,
    Controls,
    Health,
    Migrations,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum CursorSortV1 {
    Lexical { id: String },
    Newest { created_at_unix_ms: u64, id: String },
    Follow { insertion_sequence: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CursorPayloadV1 {
    api_version: u32,
    schema_version: i64,
    project_uuid: Uuid,
    cohort_generation_id: Uuid,
    endpoint: CursorEndpointV1,
    filter_hash: String,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: u64,
    sort: CursorSortV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedCursorV1 {
    pub(crate) snapshot_time_unix_ms: u64,
    pub(crate) maximum_insertion_sequence: u64,
    pub(crate) sort: CursorSortV1,
}

#[derive(Clone)]
pub(crate) struct CursorCodecV1 {
    key: Zeroizing<[u8; 32]>,
    schema_version: i64,
    project_uuid: Uuid,
    cohort_generation_id: Uuid,
}

impl CursorCodecV1 {
    pub(crate) fn new(
        key: Zeroizing<[u8; 32]>,
        schema_version: i64,
        project_uuid: Uuid,
        cohort_generation_id: Uuid,
    ) -> Result<Self, InspectionError> {
        if schema_version <= 0
            || project_uuid.get_variant() != uuid::Variant::RFC4122
            || cohort_generation_id.get_variant() != uuid::Variant::RFC4122
            || cohort_generation_id.get_version_num() != 7
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(Self {
            key,
            schema_version,
            project_uuid,
            cohort_generation_id,
        })
    }

    pub(crate) fn encode(
        &self,
        endpoint: CursorEndpointV1,
        filter_hash: &str,
        snapshot_time_unix_ms: u64,
        maximum_insertion_sequence: u64,
        sort: CursorSortV1,
    ) -> Result<String, InspectionError> {
        if !is_sha256(filter_hash) || maximum_insertion_sequence == 0 {
            return Err(InspectionError::InvalidArgument);
        }
        validate_sort(&sort)?;
        let payload = CursorPayloadV1 {
            api_version: INSPECTION_API_VERSION_V1,
            schema_version: self.schema_version,
            project_uuid: self.project_uuid,
            cohort_generation_id: self.cohort_generation_id,
            endpoint,
            filter_hash: filter_hash.to_owned(),
            snapshot_time_unix_ms,
            maximum_insertion_sequence,
            sort,
        };
        let value = serde_json::to_value(&payload).map_err(|_| InspectionError::InvalidArgument)?;
        let canonical = canonical_json(&value).map_err(|_| InspectionError::InvalidArgument)?;
        let mut signed = canonical.into_bytes();
        let mut mac = <HmacSha256 as KeyInit>::new_from_slice(&self.key[..])
            .map_err(|_| InspectionError::InvalidArgument)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(&signed);
        signed.extend_from_slice(&mac.finalize().into_bytes());
        let encoded = URL_SAFE_NO_PAD.encode(signed);
        if encoded.len() > INSPECTION_CURSOR_MAX_BYTES {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(encoded)
    }

    pub(crate) fn decode(
        &self,
        encoded: &str,
        expected_endpoint: CursorEndpointV1,
        expected_filter_hash: &str,
    ) -> Result<DecodedCursorV1, InspectionError> {
        if encoded.is_empty()
            || encoded.len() > INSPECTION_CURSOR_MAX_BYTES
            || !is_sha256(expected_filter_hash)
        {
            return Err(InspectionError::InvalidCursor);
        }
        let signed = URL_SAFE_NO_PAD
            .decode(encoded)
            .map_err(|_| InspectionError::InvalidCursor)?;
        if signed.len() <= CURSOR_TAG_BYTES {
            return Err(InspectionError::InvalidCursor);
        }
        let (payload_bytes, tag) = signed.split_at(signed.len() - CURSOR_TAG_BYTES);
        let mut mac = <HmacSha256 as KeyInit>::new_from_slice(&self.key[..])
            .map_err(|_| InspectionError::InvalidCursor)?;
        mac.update(CURSOR_DOMAIN);
        mac.update(payload_bytes);
        mac.verify_slice(tag)
            .map_err(|_| InspectionError::InvalidCursor)?;
        let payload: CursorPayloadV1 =
            serde_json::from_slice(payload_bytes).map_err(|_| InspectionError::InvalidCursor)?;
        let value = serde_json::to_value(&payload).map_err(|_| InspectionError::InvalidCursor)?;
        let recanonicalized = canonical_json(&value).map_err(|_| InspectionError::InvalidCursor)?;
        if recanonicalized.as_bytes() != payload_bytes
            || payload.api_version != INSPECTION_API_VERSION_V1
            || payload.schema_version != self.schema_version
            || payload.project_uuid != self.project_uuid
            || payload.cohort_generation_id != self.cohort_generation_id
            || payload.endpoint != expected_endpoint
            || payload.filter_hash != expected_filter_hash
            || payload.maximum_insertion_sequence == 0
            || validate_sort(&payload.sort).is_err()
        {
            return Err(InspectionError::InvalidCursor);
        }
        Ok(DecodedCursorV1 {
            snapshot_time_unix_ms: payload.snapshot_time_unix_ms,
            maximum_insertion_sequence: payload.maximum_insertion_sequence,
            sort: payload.sort,
        })
    }
}

pub(crate) fn derive_cursor_key(
    cohort_salt: &[u8; 32],
    project_uuid: Uuid,
    cohort_generation_id: Uuid,
) -> Result<Zeroizing<[u8; 32]>, InspectionError> {
    if project_uuid.get_variant() != uuid::Variant::RFC4122
        || cohort_generation_id.get_variant() != uuid::Variant::RFC4122
        || cohort_generation_id.get_version_num() != 7
    {
        return Err(InspectionError::InvalidArgument);
    }
    let mut mac = <HmacSha256 as KeyInit>::new_from_slice(cohort_salt)
        .map_err(|_| InspectionError::InvalidArgument)?;
    mac.update(CURSOR_KEY_DOMAIN);
    mac.update(project_uuid.as_bytes());
    mac.update(cohort_generation_id.as_bytes());
    let digest = mac.finalize().into_bytes();
    let mut key = Zeroizing::new([0_u8; 32]);
    key.as_mut().copy_from_slice(&digest);
    Ok(key)
}

fn validate_sort(sort: &CursorSortV1) -> Result<(), InspectionError> {
    let valid = match sort {
        CursorSortV1::Lexical { id } => !id.is_empty() && id.len() <= 512,
        CursorSortV1::Newest {
            created_at_unix_ms: _,
            id,
        } => !id.is_empty() && id.len() <= 512,
        CursorSortV1::Follow { insertion_sequence } => *insertion_sequence > 0,
    };
    if valid {
        Ok(())
    } else {
        Err(InspectionError::InvalidArgument)
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uuid7(value: u128) -> Uuid {
        let mut bytes = value.to_be_bytes();
        bytes[6] = (bytes[6] & 0x0f) | 0x70;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Uuid::from_bytes(bytes)
    }

    fn codec() -> CursorCodecV1 {
        CursorCodecV1::new(Zeroizing::new([7; 32]), 7, uuid7(1), uuid7(2)).unwrap()
    }

    fn filter_hash() -> &'static str {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }

    #[test]
    fn cursor_round_trip_freezes_snapshot_and_sort() {
        let codec = codec();
        let encoded = codec
            .encode(
                CursorEndpointV1::Evidence,
                filter_hash(),
                1234,
                91,
                CursorSortV1::Newest {
                    created_at_unix_ms: 1200,
                    id: uuid7(3).to_string(),
                },
            )
            .unwrap();
        let decoded = codec
            .decode(&encoded, CursorEndpointV1::Evidence, filter_hash())
            .unwrap();
        assert_eq!(decoded.snapshot_time_unix_ms, 1234);
        assert_eq!(decoded.maximum_insertion_sequence, 91);
        assert_eq!(
            decoded.sort,
            CursorSortV1::Newest {
                created_at_unix_ms: 1200,
                id: uuid7(3).to_string(),
            }
        );
    }

    #[test]
    fn cursor_rejects_tampering_and_wrong_query_authority() {
        let codec = codec();
        let encoded = codec
            .encode(
                CursorEndpointV1::Pools,
                filter_hash(),
                5,
                1,
                CursorSortV1::Lexical {
                    id: "pool-a".into(),
                },
            )
            .unwrap();
        let mut bytes = encoded.into_bytes();
        let middle = bytes.len() / 2;
        bytes[middle] = if bytes[middle] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(bytes).unwrap();
        assert_eq!(
            codec.decode(&tampered, CursorEndpointV1::Pools, filter_hash()),
            Err(InspectionError::InvalidCursor)
        );

        let valid = codec
            .encode(
                CursorEndpointV1::Pools,
                filter_hash(),
                5,
                1,
                CursorSortV1::Lexical {
                    id: "pool-a".into(),
                },
            )
            .unwrap();
        assert_eq!(
            codec.decode(&valid, CursorEndpointV1::Evidence, filter_hash()),
            Err(InspectionError::InvalidCursor)
        );
        assert_eq!(
            codec.decode(
                &valid,
                CursorEndpointV1::Pools,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            Err(InspectionError::InvalidCursor)
        );
        let rotated = CursorCodecV1::new(Zeroizing::new([8; 32]), 7, uuid7(1), uuid7(4)).unwrap();
        assert_eq!(
            rotated.decode(&valid, CursorEndpointV1::Pools, filter_hash()),
            Err(InspectionError::InvalidCursor)
        );
    }

    #[test]
    fn cursor_rejects_noncanonical_or_unbounded_input() {
        assert_eq!(
            codec().decode("", CursorEndpointV1::Pools, filter_hash()),
            Err(InspectionError::InvalidCursor)
        );
        assert_eq!(
            codec().decode(
                &"A".repeat(INSPECTION_CURSOR_MAX_BYTES + 1),
                CursorEndpointV1::Pools,
                filter_hash(),
            ),
            Err(InspectionError::InvalidCursor)
        );
        assert_eq!(
            codec().encode(
                CursorEndpointV1::DecisionFollow,
                filter_hash(),
                1,
                1,
                CursorSortV1::Follow {
                    insertion_sequence: 0,
                },
            ),
            Err(InspectionError::InvalidArgument)
        );
    }

    #[test]
    fn cursor_key_derivation_is_domain_and_authority_bound() {
        let first = derive_cursor_key(&[1; 32], uuid7(1), uuid7(2)).unwrap();
        assert_eq!(
            first,
            derive_cursor_key(&[1; 32], uuid7(1), uuid7(2)).unwrap()
        );
        assert_ne!(
            first,
            derive_cursor_key(&[2; 32], uuid7(1), uuid7(2)).unwrap()
        );
        assert_ne!(
            first,
            derive_cursor_key(&[1; 32], uuid7(3), uuid7(2)).unwrap()
        );
        assert_ne!(
            first,
            derive_cursor_key(&[1; 32], uuid7(1), uuid7(4)).unwrap()
        );
    }
}
