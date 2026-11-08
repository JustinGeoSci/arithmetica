// tests/compare_distance.rs

//! Integration tests for the `compare_distance` predicate.
//!
//! The tests fall into seven groups.
//!
//! - An exact-sign oracle in `BigInt`, on coordinates scaled by `2^1074`.
//!   The polynomial `|q1 − p|² − |q2 − p|²` has degree 2, so the scaled
//!   value is the true one times `2^2148`, a positive factor.
//! - Fixed cases, each built to be answered at a particular stage, which
//!   the crate's unit tests check because they can see it. The cases cover
//!   the zero filter, the fast filter, the expansion stage, exact ties
//!   confirmed by the integer stage, and input wider than the exactness
//!   window.
//! - Uniform random cases, 100,000 of them, against the oracle.
//! - Near-degenerate cases, 100,000 of them. Each starts from an exactly
//!   equidistant configuration, built with exact operations, and then moves
//!   one coordinate by up to three floats.
//! - Non-finite input, which every argument position rejects, NaN and
//!   infinities alike.
//! - Boundary-seeking runs (`tests/support`) in the five magnitude
//!   profiles, and exact ties with their nearest neighbors.
//! - A cross-check of the `BigInt` oracle against an independent
//!   `Ratio<BigInt>` oracle on every fixed case plus 4,000 random ones.

// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp:          an exact-value assertion is a float comparison.
//   unreadable_literal: bit patterns are written as they are reported.
//   cast_*:             fixture construction, with ranges known by
//                       construction.
//   unwrap_used:        in a test, a panic is the failure signal.
#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unwrap_used
)]

mod support;

use std::cmp::Ordering;

use arithmetica::{Sign, compare_distance};
use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;
use support::{Pred, Profile, Report, SplitMix64, from_ordinal, ordinal, run_boundary};

// ===========================================================================
// The oracles: BigInt and the independent rational
// ===========================================================================

/// `x · 2^1074` as an exact integer, by bit decomposition.
///
/// The conversion is exact for every finite `f64`, since a normal value is
/// `sig · 2^(be − 1075)` and a subnormal is `field · 2^-1074`.
fn scaled_integer(x: f64) -> BigInt {
    let bits = x.to_bits();
    let negative = (bits >> 63) != 0;
    let biased = (bits >> 52) & 0x7FF;
    let field = bits & 0x000F_FFFF_FFFF_FFFF;
    let magnitude = if biased == 0 {
        BigInt::from(field)
    } else {
        BigInt::from(field | (1_u64 << 52)) << (biased - 1)
    };
    if negative { -magnitude } else { magnitude }
}

/// The exact sign of `|q1 − p|² − |q2 − p|²`, in integers scaled by
/// `2^1074` (the polynomial by `2^2148`).
fn oracle(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Sign {
    let s = |a: [f64; 2]| [scaled_integer(a[0]), scaled_integer(a[1])];
    let (p, q1, q2) = (s(p), s(q1), s(q2));
    let (ux, uy) = (&q1[0] - &p[0], &q1[1] - &p[1]);
    let (vx, vy) = (&q2[0] - &p[0], &q2[1] - &p[1]);
    let q = (&ux * &ux + &uy * &uy) - (&vx * &vx + &vy * &vy);
    match q.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// The same polynomial on `Ratio<BigInt>` values built by `from_float`, a
/// different representation with an independently derived conversion.
fn oracle_rational(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Sign {
    let r = |x: f64| Ratio::<BigInt>::from_float(x).expect("finite input");
    let (ux, uy) = (r(q1[0]) - r(p[0]), r(q1[1]) - r(p[1]));
    let (vx, vy) = (r(q2[0]) - r(p[0]), r(q2[1]) - r(p[1]));
    let q = (&ux * &ux + &uy * &uy) - (&vx * &vx + &vy * &vy);
    let zero = Ratio::from_integer(BigInt::from(0));
    match q.cmp(&zero) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

// ===========================================================================
// Fixed cases
// ===========================================================================

/// A fixed case: three points and the exact sign.
type Case = ([f64; 2], [f64; 2], [f64; 2], Sign);

/// The fixed cases with their exact signs. The same configurations as the
/// unit tests' stage table in `src/compare_distance.rs`, plus a few more.
fn fixed_cases() -> Vec<Case> {
    let h = 2.0_f64.powi(500);
    let d = 2.0_f64.powi(-1000);
    let s = 2.0_f64.powi(-600);
    vec![
        // The fast filter.
        ([0.0, 0.0], [3.0, 0.0], [1.0, 0.0], Sign::Positive),
        ([0.0, 0.0], [1.0, 0.0], [3.0, 0.0], Sign::Negative),
        ([0.0, 0.0], [3.0, 0.0], [0.0, 2.0], Sign::Positive),
        // The zero filter, and an integer tie it does not see.
        ([0.1, 0.2], [0.7, 0.3], [0.7, 0.3], Sign::Zero),
        ([0.0, 0.0], [3.0, 4.0], [5.0, 0.0], Sign::Zero),
        // The expansion stage: mirrored across the diagonal, p moved off it.
        (
            [1.0, 1.0 + f64::EPSILON],
            [3.0, 2.0],
            [2.0, 3.0],
            Sign::Positive,
        ),
        // An exact tie with inexact differences, confirmed by the integer stage.
        ([0.1, 0.1], [0.7, 0.3], [0.3, 0.7], Sign::Zero),
        ([-0.3, 0.3], [0.7, 0.1], [-0.1, -0.7], Sign::Zero),
        // Wider than the window.
        ([d, 0.0], [h, 0.0], [-h, 0.0], Sign::Negative),
        // Wider than the window, decided by a square that underflows.
        ([0.0, 0.0], [h, s], [h, 0.0], Sign::Positive),
        // Overflowing differences.
        (
            [-f64::MAX, 0.0],
            [f64::MAX, f64::MIN_POSITIVE],
            [f64::MAX, 0.0],
            Sign::Positive,
        ),
        // Underflowing squares, rescaled into the window.
        ([0.0, 0.0], [2.0 * s, 0.0], [s, s], Sign::Positive),
        // Subnormal coordinates: one unit in the last place apart.
        (
            [0.0, 0.0],
            [f64::from_bits(3), 0.0],
            [0.0, f64::from_bits(2)],
            Sign::Positive,
        ),
    ]
}

#[test]
fn compare_distance_fixed_cases_agree_with_the_oracle_and_the_expected_sign() {
    for (i, (p, q1, q2, sign)) in fixed_cases().into_iter().enumerate() {
        assert_eq!(
            oracle(p, q1, q2),
            sign,
            "case {i}: the oracle disagrees with the fixture"
        );
        assert_eq!(compare_distance(p, q1, q2), Ok(sign), "case {i}");
        assert_eq!(compare_distance(p, q2, q1), Ok(-sign), "case {i}, swapped");
    }
}

/// The anti-diagonal tie in the fixed cases, written out.
///
/// With `p = (−c, c)`, `q1 = (a, b)` and `q2 = (−b, −a)`, the squared
/// distances are `(a + c)² + (b − c)²` and `(c − b)² + (a + c)²`, which are
/// equal for any floats.
#[test]
fn compare_distance_reflections_across_both_diagonals_are_exact_ties() {
    let mut rng = SplitMix64::new(0x2026_1024_D157_0001);
    for _ in 0..2_000 {
        let e = rng.range(-400, 400);
        let (a, b, c) = (
            rng.float_with_exponent(e),
            near(&mut rng, e, -3, 3),
            near(&mut rng, e, -3, 3),
        );
        for (p, q1, q2) in [([c, c], [a, b], [b, a]), ([-c, c], [a, b], [-b, -a])] {
            assert_eq!(oracle(p, q1, q2), Sign::Zero);
            assert_eq!(compare_distance(p, q1, q2), Ok(Sign::Zero));
        }
    }
}

// ===========================================================================
// Uniform random cases
// ===========================================================================

/// Finite values: mostly normal numbers between about `2^-331` and `2^499`
/// with both signs, and some exact zeros and arbitrary finite bit patterns
/// (subnormals and the extremes included).
fn finite() -> impl Strategy<Value = f64> {
    prop_oneof![
        1 => Just(0.0_f64),
        90 => (0_u64..=1, 692_u64..=1521, 0_u64..(1_u64 << 52))
            .prop_map(|(sign, exp, field)| f64::from_bits((sign << 63) | (exp << 52) | field)),
        9 => any::<u64>()
            .prop_map(f64::from_bits)
            .prop_filter("finite", |x| x.is_finite()),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    fn compare_distance_uniform_cases_match_the_oracle(
        px in finite(), py in finite(),
        ax in finite(), ay in finite(),
        bx in finite(), by in finite(),
    ) {
        let (p, q1, q2) = ([px, py], [ax, ay], [bx, by]);
        prop_assert_eq!(compare_distance(p, q1, q2), Ok(oracle(p, q1, q2)));
    }
}

// ===========================================================================
// Near-degenerate cases
// ===========================================================================

/// An exactly equidistant configuration, then one coordinate moved by `k`
/// floats, `k` in `-3..=3` (zero keeps the exact tie).
///
/// The ties are built with exact operations only. A reflection across the
/// diagonal through `p = (c, c)` swaps the coordinates of `q1`, a reflection
/// across the anti-diagonal through `p = (−c, c)` maps `(a, b)` to
/// `(−b, −a)`, and a reflection across a vertical line through `p = (0, y)`
/// negates `x`. Magnitudes range from deep in the subnormals to near
/// overflow, so every stage is reached.
fn near_ties() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2])> {
    (any::<u64>(), 0_usize..3, 0_usize..6, -3_i64..=3).prop_map(|(seed, kind, index, k)| {
        let mut rng = SplitMix64::new(seed);
        let e = rng.range(-1060, 1000);
        let mut f = || {
            let shift = rng.range(-8, 2);
            rng.float_with_exponent(e + shift)
        };
        let (a, b, c) = (f(), f(), f());
        let mut config = match kind {
            0 => [c, c, a, b, b, a],
            1 => [-c, c, a, b, -b, -a],
            _ => [0.0, c, a, b, -a, b],
        };
        let moved = from_ordinal(ordinal(config[index]) + k);
        if moved.is_finite() {
            config[index] = moved;
        }
        (
            [config[0], config[1]],
            [config[2], config[3]],
            [config[4], config[5]],
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    fn compare_distance_near_degenerate_cases_match_the_oracle((p, q1, q2) in near_ties()) {
        prop_assert_eq!(compare_distance(p, q1, q2), Ok(oracle(p, q1, q2)));
    }
}

// ===========================================================================
// Non-finite input
// ===========================================================================

#[test]
fn compare_distance_every_argument_position_rejects_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..6 {
            let mut c = [0.5, -1.0, 2.0, 3.0, -4.0, 0.25];
            c[position] = bad;
            let result = compare_distance([c[0], c[1]], [c[2], c[3]], [c[4], c[5]]);
            assert!(
                result.is_err(),
                "position {position} with {bad} returned {result:?}"
            );
        }
    }
    // A coordinate equal to f64::MAX is finite, and answered.
    assert!(compare_distance([f64::MAX, 0.0], [0.0, 0.0], [1.0, 1.0]).is_ok());
}

// ===========================================================================
// Boundary-seeking runs and the neighbors of exact ties
// ===========================================================================

const BOUNDARY_CASES: u64 = 10_000;

/// A float with its leading bit at `2^(e + k)`, `k` drawn from `lo..=hi`.
fn near(rng: &mut SplitMix64, e: i64, lo: i64, hi: i64) -> f64 {
    let k = rng.range(lo, hi);
    rng.float_with_exponent(e + k)
}

fn seed(number: u64) -> u64 {
    0x2026_1024_C0D1_0000 + number
}

#[test]
fn compare_distance_boundary_runs_in_every_profile_match_the_oracle() {
    for (i, profile) in Profile::ALL.into_iter().enumerate() {
        run_boundary(
            Pred::CompareDistance,
            profile,
            BOUNDARY_CASES,
            seed(i as u64),
        )
        .assert_clean();
    }
}

/// Exact ties, each coordinate moved by one and two floats either way.
#[test]
fn compare_distance_exact_ties_and_their_neighbors_match_the_oracle() {
    let mut rng = SplitMix64::new(seed(10));
    let mut report = Report {
        label: "compare_distance/degenerate".into(),
        ..Report::default()
    };
    for _ in 0..2_000 {
        let e = rng.range(-1060, 1000);
        let (a, b, c) = (
            rng.float_with_exponent(e),
            near(&mut rng, e, -6, 0),
            near(&mut rng, e, -6, 0),
        );
        let config = [c, c, a, b, b, a];
        assert_eq!(support::oracle(Pred::CompareDistance, &config), Sign::Zero);
        report.check(Pred::CompareDistance, &config);
        for index in 0..6 {
            for step in [-2_i64, -1, 1, 2] {
                let mut neighbor = config;
                neighbor[index] = from_ordinal(ordinal(config[index]) + step);
                if neighbor[index].is_finite() {
                    report.check(Pred::CompareDistance, &neighbor);
                }
            }
        }
    }
    report.assert_clean();
}

// ===========================================================================
// The two oracles agree
// ===========================================================================

#[test]
fn compare_distance_the_two_oracles_agree_on_every_fixed_case() {
    for (i, (p, q1, q2, _)) in fixed_cases().into_iter().enumerate() {
        assert_eq!(oracle(p, q1, q2), oracle_rational(p, q1, q2), "case {i}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4_000))]

    #[test]
    fn compare_distance_the_two_oracles_agree_on_random_cases(
        uniform in (finite(), finite(), finite(), finite(), finite(), finite()),
        (p, q1, q2) in near_ties(),
    ) {
        let (a, b, c, d, e, f) = uniform;
        prop_assert_eq!(oracle([a, b], [c, d], [e, f]), oracle_rational([a, b], [c, d], [e, f]));
        prop_assert_eq!(oracle(p, q1, q2), oracle_rational(p, q1, q2));
    }
}
