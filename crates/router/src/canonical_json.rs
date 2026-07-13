// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RFC 8785 canonical JSON helpers shared by Router identities.

use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use crate::fingerprint::validate_canonical_json_domain;

pub(crate) fn canonical_json(value: &Json) -> Result<String, String> {
    validate_canonical_json_domain(value)
        .map_err(|()| "failed to canonicalize Router JSON: unsafe integer".to_string())?;
    serde_json_canonicalizer::to_string(value)
        .map_err(|err| format!("failed to canonicalize Router JSON: {err}"))
}

pub(crate) fn canonical_sha256(value: &Json) -> Result<String, String> {
    let canonical = canonical_json(value)?;
    Ok(Sha256::digest(canonical.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{canonical_json, canonical_sha256};

    #[test]
    fn unsafe_integer_json_is_never_canonicalized_or_hashed() {
        assert!(canonical_json(&json!((1_u64 << 53) - 1)).is_ok());
        assert!(canonical_sha256(&json!(9_007_199_254_740_991.0)).is_ok());
        for value in [
            json!(1_u64 << 53),
            json!((1_u64 << 53) + 1),
            json!(i64::MIN),
            json!(9_007_199_254_740_992.0),
            json!(-9_007_199_254_740_992.0),
        ] {
            assert!(canonical_json(&value).is_err());
            assert!(canonical_sha256(&value).is_err());
        }
    }
}
