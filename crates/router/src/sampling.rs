// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Secure, injectable sampling for shadow-routing decisions.

use crate::eligibility::IneligibilityReason;

const UNIT_DENOMINATOR: f64 = (1_u64 << 53) as f64;

/// One source of independent unit-interval draws.
pub(crate) trait Sampler: Send + Sync {
    /// Draw a finite value in the half-open interval `[0, 1)`.
    fn draw_unit(&self) -> Result<f64, IneligibilityReason>;
}

/// Production sampler backed directly by operating-system randomness.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct OsSampler;

impl Sampler for OsSampler {
    fn draw_unit(&self) -> Result<f64, IneligibilityReason> {
        let mut bytes = [0_u8; 8];
        getrandom::fill(&mut bytes).map_err(|_| IneligibilityReason::SamplingFailed)?;
        Ok(unit_from_u64(u64::from_le_bytes(bytes)))
    }
}

/// Decide whether one eligible request enters shadow sampling.
///
/// Exact probability boundaries avoid consuming randomness. Invalid values are
/// rejected defensively even though typed configuration validation excludes
/// them before runtime construction.
pub(crate) fn should_sample(
    sampler: &dyn Sampler,
    probability: f64,
) -> Result<bool, IneligibilityReason> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(IneligibilityReason::SamplingFailed);
    }
    if probability == 0.0 {
        return Ok(false);
    }
    if probability == 1.0 {
        return Ok(true);
    }

    let draw = sampler.draw_unit()?;
    if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
        return Err(IneligibilityReason::SamplingFailed);
    }
    Ok(draw < probability)
}

fn unit_from_u64(value: u64) -> f64 {
    ((value >> 11) as f64) / UNIT_DENOMINATOR
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{Sampler, should_sample, unit_from_u64};
    use crate::eligibility::IneligibilityReason;

    struct CountingSampler {
        draw: f64,
        calls: Mutex<usize>,
    }

    impl CountingSampler {
        fn new(draw: f64) -> Self {
            Self {
                draw,
                calls: Mutex::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    impl Sampler for CountingSampler {
        fn draw_unit(&self) -> Result<f64, IneligibilityReason> {
            *self.calls.lock().unwrap() += 1;
            Ok(self.draw)
        }
    }

    struct SeededSampler {
        state: Mutex<u64>,
    }

    impl SeededSampler {
        fn new(seed: u64) -> Self {
            Self {
                state: Mutex::new(seed),
            }
        }
    }

    impl Sampler for SeededSampler {
        fn draw_unit(&self) -> Result<f64, IneligibilityReason> {
            let mut state = self
                .state
                .lock()
                .map_err(|_| IneligibilityReason::SamplingFailed)?;
            *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = *state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            Ok(unit_from_u64(value))
        }
    }

    #[test]
    fn exact_boundaries_do_not_consume_randomness() {
        let sampler = CountingSampler::new(0.5);
        assert!(!should_sample(&sampler, 0.0).unwrap());
        assert!(should_sample(&sampler, 1.0).unwrap());
        assert_eq!(sampler.calls(), 0);
    }

    #[test]
    fn comparison_uses_a_half_open_unit_draw() {
        let below = CountingSampler::new(0.499_999);
        let equal = CountingSampler::new(0.5);
        assert!(should_sample(&below, 0.5).unwrap());
        assert!(!should_sample(&equal, 0.5).unwrap());
    }

    #[test]
    fn seeded_fixture_is_stable() {
        let sampler = SeededSampler::new(42);
        let draws = [
            sampler.draw_unit().unwrap(),
            sampler.draw_unit().unwrap(),
            sampler.draw_unit().unwrap(),
        ];
        assert_eq!(
            draws,
            [
                0.741_564_878_771_823_3,
                0.159_910_392_876_920_1,
                0.278_601_130_255_138_66,
            ]
        );
    }

    #[test]
    fn invalid_probability_or_draw_fails_closed() {
        let sampler = CountingSampler::new(f64::NAN);
        assert_eq!(
            should_sample(&sampler, 0.5),
            Err(IneligibilityReason::SamplingFailed)
        );
        assert_eq!(
            should_sample(&sampler, f64::INFINITY),
            Err(IneligibilityReason::SamplingFailed)
        );
        assert_eq!(
            should_sample(&CountingSampler::new(1.0), 0.5),
            Err(IneligibilityReason::SamplingFailed)
        );
    }
}
