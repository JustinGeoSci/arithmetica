// tests/dot_sign.rs

//! Integration tests for the `dot_sign` predicate.
//!
//! The tests fall into seven groups.
//!
//! - An exact-sign oracle in `BigInt`, on coordinates scaled by `2^1074`.
//!   The dot product `(a − p) · (b − p)` has degree 2, so the scaled value
//!   is the true one times `2^2148`, a positive factor.
//! - Fixed cases, each built to be answered at a particular stage, which
//!   the crate's unit tests check because they can see it. The cases cover
//!   the zero filter, the fast filter, the expansion stage, exact right
//!   angles confirmed by the integer stage, and input wider than the
//!   exactness window.
//! - Uniform random cases, 100,000 of them, against the oracle.
//! - Near-degenerate cases, 100,000 of them. Each starts from an exactly
//!   right angle, built with exact operations, and then moves one
//!   coordinate by up to three floats.
//! - Non-finite input, which every argument position rejects, NaN and
//!   infinities alike.
//! - Boundary-seeking runs (`tests/support`) in the five magnitude
//!   profiles, and exact right angles with their nearest neighbors.
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

use arithmetica::{Sign, dot_sign};
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

/// The exact sign of `(a − p) · (b − p)`, in integers scaled by `2^1074`
/// (the dot product by `2^2148`).
fn oracle(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Sign {
    let s = |q: [f64; 2]| [scaled_integer(q[0]), scaled_integer(q[1])];
    let (p, a, b) = (s(p), s(a), s(b));
    let dot = (&a[0] - &p[0]) * (&b[0] - &p[0]) + (&a[1] - &p[1]) * (&b[1] - &p[1]);
    match dot.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// The same dot product on `Ratio<BigInt>` values built by `from_float`, a
/// different representation with an independently derived conversion.
fn oracle_rational(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Sign {
    let r = |x: f64| Ratio::<BigInt>::from_float(x).expect("finite input");
    let dot = (r(a[0]) - r(p[0])) * (r(b[0]) - r(p[0])) + (r(a[1]) - r(p[1])) * (r(b[1]) - r(p[1]));
    let zero = Ratio::from_integer(BigInt::from(0));
    match dot.cmp(&zero) {
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
/// unit tests' stage table in `src/dot_sign.rs`, plus a few more.
fn fixed_cases() -> Vec<Case> {
    let h = 2.0_f64.powi(500);
    let t = 2.0_f64.powi(-600);
    vec![
        // The fast filter.
        ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], Sign::Positive),
        ([0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], Sign::Negative),
        // The zero filter: a == p, b == p, an axis-aligned right angle.
        ([0.3, 0.7], [0.3, 0.7], [5.0, 1.0], Sign::Zero),
        ([0.3, 0.7], [5.0, 1.0], [0.3, 0.7], Sign::Zero),
        ([0.1, 0.2], [0.7, 0.2], [0.1, 0.9], Sign::Zero),
        // An integer right angle the zero filter does not see.
        ([0.0, 0.0], [3.0, 4.0], [-4.0, 3.0], Sign::Zero),
        // The expansion stage.
        (
            [0.0, 0.0],
            [3.0, 4.0],
            [-4.0, 3.0 + 2.0_f64.powi(-50)],
            Sign::Positive,
        ),
        // An exact right angle with inexact differences.
        ([0.1, 0.1], [0.7, 0.3], [0.2 - 0.3, 0.7], Sign::Zero),
        // Wider than the window.
        (
            [0.0, 0.0],
            [h, t],
            [t, -h * (1.0 + f64::EPSILON)],
            Sign::Negative,
        ),
        // Wider than the window, decided by a product that underflows.
        ([0.0, 0.0], [h, t], [0.0, t], Sign::Positive),
        // Overflowing differences.
        (
            [-f64::MAX, 0.0],
            [f64::MAX, 0.0],
            [f64::MAX, 1.0],
            Sign::Positive,
        ),
        // Subnormal coordinates.
        (
            [0.0, 0.0],
            [f64::from_bits(3), f64::from_bits(1)],
            [f64::from_bits(1), -f64::from_bits(2)],
            Sign::Positive,
        ),
    ]
}

#[test]
fn dot_sign_fixed_cases_agree_with_the_oracle_and_the_expected_sign() {
    for (i, (p, a, b, sign)) in fixed_cases().into_iter().enumerate() {
        assert_eq!(
            oracle(p, a, b),
            sign,
            "case {i}: the oracle disagrees with the fixture"
        );
        assert_eq!(dot_sign(p, a, b), Ok(sign), "case {i}");
        assert_eq!(dot_sign(p, b, a), Ok(sign), "case {i}, swapped");
    }
}

/// Quarter turns about the origin are exact right angles.
///
/// The vectors `(x, y)` and `(−y, x)` are perpendicular for any floats, and
/// so are `(x, y)` and `(y, −x)`.
#[test]
fn dot_sign_quarter_turns_are_exact_right_angles() {
    let mut rng = SplitMix64::new(0x2026_1024_D075_0001);
    for _ in 0..2_000 {
        let e = rng.range(-500, 500);
        let x = rng.float_with_exponent(e);
        let y = near(&mut rng, e, -3, 3);
        for b in [[-y, x], [y, -x]] {
            assert_eq!(oracle([0.0, 0.0], [x, y], b), Sign::Zero);
            assert_eq!(dot_sign([0.0, 0.0], [x, y], b), Ok(Sign::Zero));
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
    fn dot_sign_uniform_cases_match_the_oracle(
        px in finite(), py in finite(),
        ax in finite(), ay in finite(),
        bx in finite(), by in finite(),
    ) {
        let (p, a, b) = ([px, py], [ax, ay], [bx, by]);
        prop_assert_eq!(dot_sign(p, a, b), Ok(oracle(p, a, b)));
    }
}

// ===========================================================================
// Near-degenerate cases
// ===========================================================================

/// An exactly right angle, then one coordinate moved by `k` floats, `k` in
/// `-3..=3` (zero keeps the exact right angle).
///
/// Two of the constructions are quarter turns about `p = (0, 0)`, pairing
/// `(x, y)` with `(−y, x)` and with `(y, −x)`. A third kind translates the
/// angle to `p = (c, c)` with `b = (fl(2c − t), s)` for `a = (s, t)`. That
/// angle is exactly right when the difference is exact, and within an ulp
/// of right otherwise, which is as useful a place to test. Magnitudes range
/// from deep in the subnormals to near overflow, so every stage is reached.
fn near_right_angles() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2])> {
    (any::<u64>(), 0_usize..3, 0_usize..6, -3_i64..=3).prop_map(|(seed, kind, index, k)| {
        let mut rng = SplitMix64::new(seed);
        let e = rng.range(-1060, 1000);
        let (x, y, c) = (
            near(&mut rng, e, -8, 2),
            near(&mut rng, e, -8, 2),
            near(&mut rng, e, -8, 2),
        );
        let mut config = match kind {
            0 => [0.0, 0.0, x, y, -y, x],
            1 => [0.0, 0.0, x, y, y, -x],
            _ => [c, c, x, y, 2.0 * c - y, x],
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
    fn dot_sign_near_degenerate_cases_match_the_oracle((p, a, b) in near_right_angles()) {
        // The translated kind can overflow `2c − t`. The non-finite test
        // covers such input rather than this one.
        if p.iter().chain(&a).chain(&b).all(|v| v.is_finite()) {
            prop_assert_eq!(dot_sign(p, a, b), Ok(oracle(p, a, b)));
        }
    }
}

// ===========================================================================
// Non-finite input
// ===========================================================================

#[test]
fn dot_sign_every_argument_position_rejects_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..6 {
            let mut c = [0.5, -1.0, 2.0, 3.0, -4.0, 0.25];
            c[position] = bad;
            let result = dot_sign([c[0], c[1]], [c[2], c[3]], [c[4], c[5]]);
            assert!(
                result.is_err(),
                "position {position} with {bad} returned {result:?}"
            );
        }
    }
    assert!(dot_sign([f64::MAX, 0.0], [0.0, 0.0], [1.0, 1.0]).is_ok());
}

// ===========================================================================
// Boundary-seeking runs and the neighbors of exact right angles
// ===========================================================================

const BOUNDARY_CASES: u64 = 10_000;

/// A float with its leading bit at `2^(e + k)`, `k` drawn from `lo..=hi`.
fn near(rng: &mut SplitMix64, e: i64, lo: i64, hi: i64) -> f64 {
    let k = rng.range(lo, hi);
    rng.float_with_exponent(e + k)
}

fn seed(number: u64) -> u64 {
    0x2026_1024_D075_0000 + number
}

#[test]
fn dot_sign_boundary_runs_in_every_profile_match_the_oracle() {
    for (i, profile) in Profile::ALL.into_iter().enumerate() {
        run_boundary(Pred::DotSign, profile, BOUNDARY_CASES, seed(16 + i as u64)).assert_clean();
    }
}

/// Exact right angles, each coordinate moved by one and two floats either
/// way.
#[test]
fn dot_sign_exact_right_angles_and_their_neighbors_match_the_oracle() {
    let mut rng = SplitMix64::new(seed(32));
    let mut report = Report {
        label: "dot_sign/degenerate".into(),
        ..Report::default()
    };
    for _ in 0..2_000 {
        let e = rng.range(-1060, 1000);
        let x = rng.float_with_exponent(e);
        let y = near(&mut rng, e, -6, 0);
        let config = [0.0, 0.0, x, y, -y, x];
        assert_eq!(support::oracle(Pred::DotSign, &config), Sign::Zero);
        report.check(Pred::DotSign, &config);
        for index in 0..6 {
            for step in [-2_i64, -1, 1, 2] {
                let mut neighbor = config;
                neighbor[index] = from_ordinal(ordinal(config[index]) + step);
                if neighbor[index].is_finite() {
                    report.check(Pred::DotSign, &neighbor);
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
fn dot_sign_the_two_oracles_agree_on_every_fixed_case() {
    for (i, (p, a, b, _)) in fixed_cases().into_iter().enumerate() {
        assert_eq!(oracle(p, a, b), oracle_rational(p, a, b), "case {i}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4_000))]

    #[test]
    fn dot_sign_the_two_oracles_agree_on_random_cases(
        uniform in (finite(), finite(), finite(), finite(), finite(), finite()),
        (p, a, b) in near_right_angles(),
    ) {
        let (u0, u1, u2, u3, u4, u5) = uniform;
        prop_assert_eq!(oracle([u0, u1], [u2, u3], [u4, u5]), oracle_rational([u0, u1], [u2, u3], [u4, u5]));
        if p.iter().chain(&a).chain(&b).all(|v| v.is_finite()) {
            prop_assert_eq!(oracle(p, a, b), oracle_rational(p, a, b));
        }
    }
}
