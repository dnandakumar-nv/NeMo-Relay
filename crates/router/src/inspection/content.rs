// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic content filtering shared by inspection and export.

use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};

use super::{
    ContentPolicy, INSPECTION_REDACTED_PREVIEW_MAX_CHARS, InspectionContentV1, InspectionError,
};
use crate::canonical_json::canonical_json;

pub(crate) fn inspect_content(
    source: &Json,
    policy: ContentPolicy,
) -> Result<InspectionContentV1, InspectionError> {
    let filtered = filter_value(source);
    let canonical = canonical_json(&filtered).map_err(|_| InspectionError::IntegrityError)?;
    let preview = (!canonical.is_empty()).then(|| {
        canonical
            .chars()
            .take(INSPECTION_REDACTED_PREVIEW_MAX_CHARS)
            .collect()
    });
    Ok(InspectionContentV1 {
        sha256: hex_sha256(canonical.as_bytes()),
        byte_length: u64::try_from(canonical.len()).map_err(|_| InspectionError::IntegrityError)?,
        preview,
        value: (policy == ContentPolicy::Full).then_some(filtered),
    })
}

fn filter_value(value: &Json) -> Json {
    match value {
        Json::Object(object) => {
            let mut filtered = Map::new();
            for (key, value) in object {
                if secret_key(key) || schema_declares_secret(value) {
                    continue;
                }
                filtered.insert(key.clone(), filter_value(value));
            }
            Json::Object(filtered)
        }
        Json::Array(values) => Json::Array(values.iter().map(filter_value).collect()),
        Json::String(value) => Json::String(redact_secret_text(value)),
        Json::Null | Json::Bool(_) | Json::Number(_) => value.clone(),
    }
}

fn secret_key(key: &str) -> bool {
    let normalized = key.trim().to_ascii_lowercase().replace(['-', '.'], "_");
    matches!(
        normalized.as_str(),
        "authorization"
            | "proxy_authorization"
            | "cookie"
            | "set_cookie"
            | "api_key"
            | "apikey"
            | "access_token"
            | "refresh_token"
            | "id_token"
            | "client_secret"
            | "password"
            | "passwd"
            | "credential"
            | "credentials"
            | "secret"
            | "secret_key"
            | "private_key"
            | "session_token"
    ) || normalized.ends_with("_api_key")
        || normalized.ends_with("_password")
        || normalized.ends_with("_secret")
        || normalized.ends_with("_credential")
}

fn schema_declares_secret(value: &Json) -> bool {
    let Json::Object(object) = value else {
        return false;
    };
    object.get("writeOnly") == Some(&Json::Bool(true))
        || object.get("secret") == Some(&Json::Bool(true))
        || object.get("x-secret") == Some(&Json::Bool(true))
        || object.get("x_secret") == Some(&Json::Bool(true))
        || object.get("x-nemo-relay-secret") == Some(&Json::Bool(true))
        || object
            .get("format")
            .and_then(Json::as_str)
            .is_some_and(|format| matches!(format, "password" | "secret" | "credential"))
}

fn redact_secret_text(value: &str) -> String {
    const PREFIXES: &[&str] = &[
        "authorization:",
        "proxy-authorization:",
        "api_key=",
        "apikey=",
        "access_token=",
        "refresh_token=",
        "client_secret=",
        "password=",
        "passwd=",
        "secret=",
        "bearer ",
    ];
    let mut output = value.to_owned();
    for prefix in PREFIXES {
        loop {
            let lowercase = output.to_ascii_lowercase();
            let Some(start) = lowercase.find(prefix) else {
                break;
            };
            let mut value_start = start + prefix.len();
            while output[value_start..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
            {
                value_start += output[value_start..]
                    .chars()
                    .next()
                    .map(char::len_utf8)
                    .unwrap_or_default();
            }
            let header_value = prefix.ends_with(':');
            let value_end = output[value_start..]
                .char_indices()
                .find_map(|(offset, character)| {
                    ((!header_value && character.is_whitespace())
                        || matches!(character, '\n' | '\r' | ',' | ';' | '"' | '\'' | '}' | ']'))
                    .then_some(value_start + offset)
                })
                .unwrap_or(output.len());
            output.replace_range(start..value_end, "[REDACTED]");
        }
    }
    output
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn redacted_and_full_content_never_return_secret_fields_or_values() {
        let source = json!({
            "headers": {
                "Authorization": "Bearer header-secret",
                "x-api-key": "api-secret",
                "accept": "application/json"
            },
            "password": "password-secret",
            "nested": {
                "session_token": "session-secret",
                "safe": "authorization: Bearer inline-secret",
                "schema_only": {"type": "string", "writeOnly": true}
            },
            "ordinary": "visible"
        });
        for policy in [ContentPolicy::Redacted, ContentPolicy::Full] {
            let content = inspect_content(&source, policy).unwrap();
            let rendered = serde_json::to_string(&content).unwrap();
            for secret in [
                "header-secret",
                "api-secret",
                "password-secret",
                "session-secret",
                "inline-secret",
                "Authorization",
                "x-api-key",
                "session_token",
            ] {
                assert!(!rendered.contains(secret), "leaked {secret}: {rendered}");
            }
            assert!(rendered.contains("visible"));
            assert!(rendered.contains("[REDACTED]"));
            assert_eq!(content.value.is_some(), policy == ContentPolicy::Full);
        }
    }

    #[test]
    fn preview_is_unicode_scalar_bounded_and_hashes_filtered_canonical_bytes() {
        let source = json!({
            "text": "界".repeat(300),
            "client_secret": "never-hash-this"
        });
        let content = inspect_content(&source, ContentPolicy::Full).unwrap();
        let filtered = content.value.clone().unwrap();
        let canonical = canonical_json(&filtered).unwrap();
        assert_eq!(content.byte_length, canonical.len() as u64);
        assert_eq!(content.sha256, hex_sha256(canonical.as_bytes()));
        assert_eq!(
            content.preview.as_deref().unwrap().chars().count(),
            INSPECTION_REDACTED_PREVIEW_MAX_CHARS
        );
        assert!(!canonical.contains("never-hash-this"));
    }
}
