// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RFC 8785 and SHA-256 helpers shared by Router semantic contracts.

use std::fmt::Write;
use std::io;

use serde::Serialize;
use serde_json::Value as Json;
use serde_json::ser::{CharEscape, Formatter};
use sha2::{Digest, Sha256};

struct DigestWriter(Sha256);

struct BoundedDigestWriter {
    digest: Sha256,
    remaining: usize,
    exceeded: bool,
}

struct BoundedSizeWriter {
    remaining: usize,
    exceeded: bool,
}

/// RFC 8785 formatting without object sorting, used only for exact byte counts.
/// Object property order cannot change the serialized length.
struct JcsSizeFormatter {
    validate_integer_domain: bool,
}

const MAX_SAFE_JSON_INTEGER: u128 = (1_u128 << 53) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedFingerprintError {
    BoundExceeded,
    Serialization,
}

impl io::Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Write for BoundedDigestWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(io::Error::other("canonical JSON exceeds its byte bound"));
        }
        self.digest.update(bytes);
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl io::Write for BoundedSizeWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            self.exceeded = true;
            return Err(io::Error::other("canonical JSON exceeds its byte bound"));
        }
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_jcs_unsigned(value: u128) -> io::Result<()> {
    if value > MAX_SAFE_JSON_INTEGER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "integer is outside the interoperable RFC 8785 domain",
        ));
    }
    Ok(())
}

fn validate_jcs_signed(value: i128) -> io::Result<()> {
    let maximum = MAX_SAFE_JSON_INTEGER as i128;
    if !(-maximum..=maximum).contains(&value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "integer is outside the interoperable RFC 8785 domain",
        ));
    }
    Ok(())
}

fn validate_jcs_float(value: f64) -> io::Result<()> {
    if !value.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "non-finite number in canonical JSON",
        ));
    }
    let maximum = MAX_SAFE_JSON_INTEGER as f64;
    if value.fract() == 0.0 && !(-maximum..=maximum).contains(&value) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "integer is outside the interoperable RFC 8785 domain",
        ));
    }
    Ok(())
}

macro_rules! write_jcs_signed_integer {
    ($method:ident, $kind:ty) => {
        fn $method<W>(&mut self, writer: &mut W, value: $kind) -> io::Result<()>
        where
            W: ?Sized + io::Write,
        {
            if self.validate_integer_domain {
                validate_jcs_signed(value as i128)?;
            }
            self.write_f64(writer, value as f64)
        }
    };
}

macro_rules! write_jcs_unsigned_integer {
    ($method:ident, $kind:ty) => {
        fn $method<W>(&mut self, writer: &mut W, value: $kind) -> io::Result<()>
        where
            W: ?Sized + io::Write,
        {
            if self.validate_integer_domain {
                validate_jcs_unsigned(value as u128)?;
            }
            self.write_f64(writer, value as f64)
        }
    };
}

impl Formatter for JcsSizeFormatter {
    write_jcs_signed_integer!(write_i8, i8);
    write_jcs_signed_integer!(write_i16, i16);
    write_jcs_signed_integer!(write_i32, i32);
    write_jcs_signed_integer!(write_i64, i64);
    write_jcs_signed_integer!(write_i128, i128);
    write_jcs_unsigned_integer!(write_u8, u8);
    write_jcs_unsigned_integer!(write_u16, u16);
    write_jcs_unsigned_integer!(write_u32, u32);
    write_jcs_unsigned_integer!(write_u64, u64);
    write_jcs_unsigned_integer!(write_u128, u128);

    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.write_f64(writer, value as f64)
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.validate_integer_domain {
            validate_jcs_float(value)?;
        } else if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "non-finite number in canonical JSON",
            ));
        }
        let mut buffer = ryu_js::Buffer::new();
        writer.write_all(buffer.format_finite(value).as_bytes())
    }

    fn write_number_str<W>(&mut self, writer: &mut W, value: &str) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.validate_integer_domain {
            if let Ok(integer) = value.parse::<i128>() {
                validate_jcs_signed(integer)?;
            } else if let Ok(integer) = value.parse::<u128>() {
                validate_jcs_unsigned(integer)?;
            }
        }
        let value = value
            .parse::<f64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid JSON number"))?;
        self.write_f64(writer, value)
    }

    fn write_char_escape<W>(&mut self, writer: &mut W, char_escape: CharEscape) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        use CharEscape::*;

        let bytes: &[u8] = match char_escape {
            Quote => b"\\\"",
            ReverseSolidus => b"\\\\",
            Solidus => b"/",
            Backspace => b"\\b",
            FormFeed => b"\\f",
            LineFeed => b"\\n",
            CarriageReturn => b"\\r",
            Tab => b"\\t",
            AsciiControl(byte) => {
                static HEX: [u8; 16] = *b"0123456789abcdef";
                return writer.write_all(&[
                    b'\\',
                    b'u',
                    b'0',
                    b'0',
                    HEX[(byte >> 4) as usize],
                    HEX[(byte & 0x0f) as usize],
                ]);
            }
        };
        writer.write_all(bytes)
    }
}

fn preflight_canonical_size<T: Serialize>(
    value: &T,
    max_bytes: usize,
    validate_integer_domain: bool,
) -> Result<(), BoundedFingerprintError> {
    let mut writer = BoundedSizeWriter {
        remaining: max_bytes,
        exceeded: false,
    };
    let mut serializer = serde_json::Serializer::with_formatter(
        &mut writer,
        JcsSizeFormatter {
            validate_integer_domain,
        },
    );
    if value.serialize(&mut serializer).is_err() {
        return Err(if writer.exceeded {
            BoundedFingerprintError::BoundExceeded
        } else {
            BoundedFingerprintError::Serialization
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_serializable_canonical_bound<T: Serialize>(
    value: &T,
    max_bytes: usize,
) -> Result<(), BoundedFingerprintError> {
    preflight_canonical_size(value, max_bytes, true)
}

pub(crate) fn validate_serializable_size_bound<T: Serialize>(
    value: &T,
    max_bytes: usize,
) -> Result<(), BoundedFingerprintError> {
    preflight_canonical_size(value, max_bytes, false)
}

pub(crate) fn validate_canonical_json_domain(value: &Json) -> Result<(), ()> {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Json::Array(values) => pending.extend(values),
            Json::Object(values) => pending.extend(values.values()),
            Json::Number(number) => {
                if number
                    .as_i64()
                    .is_some_and(|value| validate_jcs_signed(value as i128).is_err())
                    || number
                        .as_u64()
                        .is_some_and(|value| validate_jcs_unsigned(value as u128).is_err())
                    || number
                        .as_f64()
                        .is_some_and(|value| validate_jcs_float(value).is_err())
                {
                    return Err(());
                }
            }
            Json::Null | Json::Bool(_) | Json::String(_) => {}
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonResourceUsage {
    pub(crate) bytes: usize,
    pub(crate) values: usize,
}

pub(crate) fn validate_json_resource_bounds(
    value: &Json,
    max_bytes: usize,
    max_values: usize,
    max_depth: usize,
) -> Result<JsonResourceUsage, ()> {
    let mut pending = vec![(value, 0usize)];
    let mut values = 0usize;
    let mut bytes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        values = values.checked_add(1).ok_or(())?;
        if values > max_values || depth > max_depth {
            return Err(());
        }
        let additional = match value {
            Json::Null => 4,
            Json::Bool(true) => 4,
            Json::Bool(false) => 5,
            Json::Number(number) => canonical_number_len(number)?,
            Json::String(value) => canonical_string_len(value)?,
            Json::Array(items) => {
                if values
                    .checked_add(pending.len())
                    .and_then(|count| count.checked_add(items.len()))
                    .is_none_or(|count| count > max_values)
                    || (depth == max_depth && !items.is_empty())
                {
                    return Err(());
                }
                pending.extend(items.iter().rev().map(|item| (item, depth + 1)));
                2usize
                    .checked_add(items.len().saturating_sub(1))
                    .ok_or(())?
            }
            Json::Object(object) => {
                if values
                    .checked_add(pending.len())
                    .and_then(|count| count.checked_add(object.len()))
                    .is_none_or(|count| count > max_values)
                    || (depth == max_depth && !object.is_empty())
                {
                    return Err(());
                }
                let mut object_bytes = 2usize
                    .checked_add(object.len().saturating_sub(1))
                    .ok_or(())?;
                for (key, value) in object.iter().rev() {
                    object_bytes = object_bytes
                        .checked_add(canonical_string_len(key)?)
                        .and_then(|bytes| bytes.checked_add(1))
                        .ok_or(())?;
                    pending.push((value, depth + 1));
                }
                object_bytes
            }
        };
        bytes = bytes.checked_add(additional).ok_or(())?;
        if bytes > max_bytes {
            return Err(());
        }
    }
    Ok(JsonResourceUsage { bytes, values })
}

fn canonical_number_len(number: &serde_json::Number) -> Result<usize, ()> {
    let value = if let Some(value) = number.as_i64() {
        value as f64
    } else if let Some(value) = number.as_u64() {
        value as f64
    } else {
        number.as_f64().ok_or(())?
    };
    if !value.is_finite() {
        return Err(());
    }
    let mut buffer = ryu_js::Buffer::new();
    Ok(buffer.format_finite(value).len())
}

fn canonical_string_len(value: &str) -> Result<usize, ()> {
    value.chars().try_fold(2usize, |bytes, character| {
        let additional = match character {
            '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        bytes.checked_add(additional).ok_or(())
    })
}

/// Canonicalize a JSON value using RFC 8785.
pub(crate) fn canonical_json_bytes(value: &Json) -> Result<Vec<u8>, ()> {
    validate_canonical_json_domain(value)?;
    serde_json_canonicalizer::to_vec(value).map_err(|_| ())
}

/// Canonicalize a serializable semantic DTO using RFC 8785.
pub(crate) fn canonical_serialize_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, ()> {
    let json = serde_json::to_value(value).map_err(|_| ())?;
    canonical_json_bytes(&json)
}

/// Return an unprefixed lowercase SHA-256 hexadecimal digest.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest_hex(&digest)
}

fn digest_hex(digest: &[u8]) -> String {
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Canonicalize and fingerprint one serializable value without retaining its bytes.
pub(crate) fn fingerprint_serializable<T: Serialize>(value: &T) -> Result<String, ()> {
    preflight_canonical_size(value, usize::MAX, true).map_err(|_| ())?;
    let mut writer = DigestWriter(Sha256::new());
    serde_json_canonicalizer::to_writer(value, &mut writer).map_err(|_| ())?;
    Ok(digest_hex(&writer.0.finalize()))
}

/// Canonicalize and fingerprint without writing more than `max_bytes`.
pub(crate) fn fingerprint_serializable_bounded<T: Serialize>(
    value: &T,
    max_bytes: usize,
) -> Result<String, BoundedFingerprintError> {
    // The JCS crate buffers complete objects before writing them. Count the
    // exact canonical length first so oversized values never reach that path.
    preflight_canonical_size(value, max_bytes, true)?;
    let mut writer = BoundedDigestWriter {
        digest: Sha256::new(),
        remaining: max_bytes,
        exceeded: false,
    };
    if serde_json_canonicalizer::to_writer(value, &mut writer).is_err() {
        return Err(if writer.exceeded {
            BoundedFingerprintError::BoundExceeded
        } else {
            BoundedFingerprintError::Serialization
        });
    }
    Ok(digest_hex(&writer.digest.finalize()))
}

/// Canonicalize and fingerprint one JSON value.
pub(crate) fn fingerprint_json(value: &Json) -> Result<String, ()> {
    canonical_json_bytes(value).map(|bytes| sha256_hex(&bytes))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use serde::ser::SerializeSeq;
    use serde::{Serialize, Serializer};
    use serde_json::json;

    use super::{
        BoundedFingerprintError, canonical_json_bytes, fingerprint_json, fingerprint_serializable,
        fingerprint_serializable_bounded, validate_serializable_canonical_bound,
    };

    #[test]
    fn rfc8785_and_sha256_match_external_golden() {
        let value = json!({"b": 2, "a": 1});
        assert_eq!(canonical_json_bytes(&value).unwrap(), br#"{"a":1,"b":2}"#);
        assert_eq!(
            fingerprint_json(&value).unwrap(),
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
        );
        assert_eq!(
            fingerprint_serializable(&value).unwrap(),
            fingerprint_json(&value).unwrap()
        );
    }

    #[test]
    fn bounded_fingerprint_accepts_exact_bytes_and_rejects_one_less() {
        for value in [
            json!({"b": 2, "a": "value"}),
            json!({"nested": {"z": "\u{0001}/\\\"", "a": [1.5e-7, -0.0]}}),
            json!({"integer": (1_u64 << 53) - 1, "small": 1.0e-7}),
        ] {
            let canonical = canonical_json_bytes(&value).unwrap();
            assert_eq!(
                fingerprint_serializable_bounded(&value, canonical.len()).unwrap(),
                fingerprint_json(&value).unwrap()
            );
            assert_eq!(
                fingerprint_serializable_bounded(&value, canonical.len() - 1),
                Err(BoundedFingerprintError::BoundExceeded)
            );
        }

        for unsafe_integer in [json!(1_u64 << 53), json!(u64::MAX), json!(i64::MIN)] {
            assert_eq!(
                fingerprint_serializable_bounded(&unsafe_integer, usize::MAX),
                Err(BoundedFingerprintError::Serialization)
            );
            assert!(canonical_json_bytes(&unsafe_integer).is_err());
        }
    }

    #[test]
    fn float_backed_integral_numbers_follow_safe_integer_domain() {
        let safe_integral_float = json!(9_007_199_254_740_991.0);
        assert_eq!(
            canonical_json_bytes(&safe_integral_float).unwrap(),
            b"9007199254740991"
        );
        assert_eq!(
            fingerprint_serializable(&safe_integral_float).unwrap(),
            fingerprint_json(&safe_integral_float).unwrap()
        );

        for safe_number in ["1e3", "9.007199254740991e15", "-9.007199254740991e15"] {
            let value: serde_json::Value = serde_json::from_str(safe_number).unwrap();
            assert!(canonical_json_bytes(&value).is_ok(), "{safe_number}");
            assert!(fingerprint_serializable(&value).is_ok(), "{safe_number}");
        }

        for unsafe_number in [
            "9007199254740992.0",
            "-9007199254740992.0",
            "9.007199254740992e15",
            "-9.007199254740992e15",
            "1e21",
        ] {
            let value: serde_json::Value = serde_json::from_str(unsafe_number).unwrap();
            assert!(canonical_json_bytes(&value).is_err(), "{unsafe_number}");
            assert!(fingerprint_json(&value).is_err(), "{unsafe_number}");
            assert!(fingerprint_serializable(&value).is_err(), "{unsafe_number}");
            assert_eq!(
                fingerprint_serializable_bounded(&value, usize::MAX),
                Err(BoundedFingerprintError::Serialization),
                "{unsafe_number}"
            );
        }

        for finite_fraction in [
            json!(-0.0),
            json!(0.0),
            json!(1.0),
            json!(0.9),
            json!(1.5),
            json!(-0.25),
            json!(1.25e-7),
            json!(4_503_599_627_370_495.5),
        ] {
            assert!(canonical_json_bytes(&finite_fraction).is_ok());
            assert_eq!(
                fingerprint_serializable(&finite_fraction).unwrap(),
                fingerprint_json(&finite_fraction).unwrap()
            );
        }
    }

    #[test]
    fn bounded_preflight_stops_before_traversing_an_oversized_sequence() {
        struct CountedSequence<'a> {
            visits: &'a Cell<usize>,
        }

        impl Serialize for CountedSequence<'_> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut sequence = serializer.serialize_seq(Some(1_000_000))?;
                for _ in 0..1_000_000 {
                    self.visits.set(self.visits.get() + 1);
                    sequence.serialize_element("xxxxxxxx")?;
                }
                sequence.end()
            }
        }

        let visits = Cell::new(0);
        assert_eq!(
            validate_serializable_canonical_bound(&CountedSequence { visits: &visits }, 64),
            Err(BoundedFingerprintError::BoundExceeded)
        );
        assert!(
            visits.get() < 10,
            "preflight visited {} entries",
            visits.get()
        );
    }
}
