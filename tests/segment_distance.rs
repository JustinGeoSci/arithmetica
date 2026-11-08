// tests/segment_distance.rs

//! Integration tests for `compare_point_distance` and
//! `compare_segment_distance`.
//!
//! The tests fall into seven groups.
//!
//! - Exact-sign oracles in `BigInt`, on values scaled by `2^1074` (`d`
//!   scaled with the coordinates). The point polynomial has degree 2, so
//!   the scaled value is the true one times `2^2148`. The segment
//!   predicate's interior case has degree 4, scaled by `2^4296`, and its
//!   endpoint cases are the point polynomial. Each case's factor is
//!   positive and the case choice is exact integer arithmetic, so the
//!   oracle is exact.
//! - Fixed cases, each built to be answered at a particular stage and, for
//!   the segment form, in a particular case, both of which the crate's unit
//!   tests check because they can see them. The cases cover the zero
//!   filter, the fast filter, the filter's edge, the expansion stage, exact
//!   ties confirmed by the integer stage, degenerate segments, and input
//!   wider than the exactness window.
//! - Uniform random cases, 100,000 per predicate, against the oracle.
//! - Near-degenerate cases, 100,000 per predicate. Each starts from a
//!   configuration exactly at distance, built with exact operations
//!   (Pythagorean triples on a shared grid, or axis-aligned segments), and
//!   then moves one value by up to three floats.
//! - Non-finite input, which every argument position rejects, NaN and
//!   infinities alike.
//! - Boundary-seeking runs (`tests/support`) in the five magnitude
//!   profiles, and exact ties with their nearest neighbors.
//! - A cross-check of the `BigInt` oracles against independent
//!   `Ratio<BigInt>` oracles on every fixed case plus 4,000 random ones.
//!   For the segment form the second oracle *divides*, comparing
//!   `cross²/|u|²` with `d²` rather than clearing the denominator.

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

use arithmetica::{Sign, compare_point_distance, compare_segment_distance};
use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;
use support::{Pred, Profile, Report, SplitMix64, from_ordinal, ordinal, run_boundary};

type Q = Ratio<BigInt>;

// ===========================================================================
// The oracles: BigInt (through tests/support) and the independent rationals
// ===========================================================================

fn point_oracle(p: [f64; 2], q: [f64; 2], d: f64) -> Sign {
    support::oracle(Pred::ComparePointDistance, &[p[0], p[1], q[0], q[1], d])
}

fn segment_oracle(p: [f64; 2], a: [f64; 2], b: [f64; 2], d: f64) -> Sign {
    support::oracle(
        Pred::CompareSegmentDistance,
        &[p[0], p[1], a[0], a[1], b[0], b[1], d],
    )
}

fn rational(x: f64) -> Q {
    Q::from_float(x).expect("finite input")
}

fn sign_of_q(v: &Q) -> Sign {
    let zero = Q::from_integer(BigInt::from(0));
    match v.cmp(&zero) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// `|q − p|² − d²` on `Ratio<BigInt>`: a different representation and an
/// independently derived conversion.
fn point_oracle_rational(p: [f64; 2], q: [f64; 2], d: f64) -> Sign {
    let (ux, uy) = (
        rational(q[0]) - rational(p[0]),
        rational(q[1]) - rational(p[1]),
    );
    let dd = rational(d);
    sign_of_q(&(&ux * &ux + &uy * &uy - &dd * &dd))
}

/// `dist²(p, ab) − d²` on `Ratio<BigInt>`, with a division.
///
/// The interior case compares `cross²/|u|²` with `d²`, a quotient the
/// predicate never forms. The oracle is therefore an independent
/// formulation rather than only an independent representation.
fn segment_oracle_rational(p: [f64; 2], a: [f64; 2], b: [f64; 2], d: f64) -> Sign {
    let (ux, uy) = (
        rational(b[0]) - rational(a[0]),
        rational(b[1]) - rational(a[1]),
    );
    let (vx, vy) = (
        rational(p[0]) - rational(a[0]),
        rational(p[1]) - rational(a[1]),
    );
    let len2 = &ux * &ux + &uy * &uy;
    let zero = Q::from_integer(BigInt::from(0));
    let dist2 = if &vx * &ux + &vy * &uy <= zero {
        &vx * &vx + &vy * &vy
    } else {
        let (wx, wy) = (
            rational(p[0]) - rational(b[0]),
            rational(p[1]) - rational(b[1]),
        );
        if &wx * &ux + &wy * &uy >= zero {
            &wx * &wx + &wy * &wy
        } else {
            let cross = &ux * &vy - &uy * &vx;
            &cross * &cross / &len2
        }
    };
    let dd = rational(d);
    sign_of_q(&(dist2 - &dd * &dd))
}

// ===========================================================================
// Fixed cases
// ===========================================================================

/// A fixed point-form case: the points, `d`, the exact sign.
type PointCase = ([f64; 2], [f64; 2], f64, Sign);

fn point_fixed_cases() -> Vec<PointCase> {
    let h = 2.0_f64.powi(500);
    let t = 2.0_f64.powi(-600);
    let m = 268_435_457.0; // 2^28 + 1: squares round
    vec![
        // The zero filter.
        ([0.5, -0.25], [0.5, -0.25], 3.0, Sign::Negative),
        ([0.5, -0.25], [0.5, -0.25], 0.0, Sign::Zero),
        // The fast filter, both signs of d.
        ([0.0, 0.0], [3.0, 4.0], 1.0, Sign::Positive),
        ([0.0, 0.0], [3.0, 4.0], -6.0, Sign::Negative),
        ([0.0, 0.0], [3.0, 4.0], 5.0, Sign::Zero),
        // The expansion stage: one ulp of d off a tie whose squares round.
        (
            [0.0, 0.0],
            [3.0 * m, 4.0 * m],
            5.0 * m * (1.0 + f64::EPSILON),
            Sign::Negative,
        ),
        // An exact tie with inexact squares: the integer stage.
        ([0.0, 0.0], [3.0 * m, 4.0 * m], 5.0 * m, Sign::Zero),
        // The filter's edge (see the unit tests): between A/2 and A.
        (
            [0.0, -2.0],
            [1_316_160_597.000_001, 1_754_880_794.000_001],
            2_193_600_995.000_000_5,
            Sign::Positive,
        ),
        (
            [-3.0, -3.0],
            [579_255_474.000_001, 772_340_633.0],
            965_425_795.000_001,
            Sign::Negative,
        ),
        // Wider than the window.
        ([0.0, 0.0], [h, t], h, Sign::Positive),
        // Subnormal: one ulp apart in distance.
        (
            [0.0, 0.0],
            [f64::from_bits(3), 0.0],
            f64::from_bits(2),
            Sign::Positive,
        ),
    ]
}

/// A fixed segment-form case: the point, the segment, `d`, the exact sign.
type SegmentCase = ([f64; 2], [f64; 2], [f64; 2], f64, Sign);

fn segment_fixed_cases() -> Vec<SegmentCase> {
    let h = 2.0_f64.powi(500);
    let t = 2.0_f64.powi(-600);
    let a = [0.0, 0.0];
    let bx = [10.0, 0.0];
    vec![
        // The endpoint cases, exact ties and both sides of them.
        ([-3.0, 4.0], a, bx, 5.0, Sign::Zero),
        ([-3.0, 4.0], a, bx, 4.0, Sign::Positive),
        ([-3.0, 4.0], a, bx, -6.0, Sign::Negative),
        ([13.0, -4.0], a, bx, 5.0, Sign::Zero),
        // The foot exactly at an endpoint.
        ([0.0, 5.0], a, bx, 6.0, Sign::Negative),
        // Degenerate segments: the point a.
        ([3.0, 4.0], [1.0, 1.0], [1.0, 1.0], 1.0, Sign::Positive),
        ([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], 0.0, Sign::Zero),
        // Interior: the filter, the expansion stage, the exact tie.
        ([5.0, 3.0], a, bx, 1.0, Sign::Positive),
        (
            [5.0, 3.0],
            a,
            bx,
            3.0 * (1.0 + 2.0_f64.powi(-51)),
            Sign::Negative,
        ),
        ([5.0, 3.0], a, bx, 3.0, Sign::Zero),
        // The interior filter's edge (see the unit tests).
        (
            [21_330.0, 1_024.0],
            a,
            [32_768.0, 0.0],
            1_024.000_000_000_001_4,
            Sign::Negative,
        ),
        (
            [1_141.25, 16_384.0],
            a,
            [4_096.0, 0.0],
            16_384.000_000_000_025,
            Sign::Negative,
        ),
        // On the segment: distance zero.
        ([5.0, 0.0], a, bx, 0.0, Sign::Zero),
        ([5.0, 0.0], a, bx, 0.25, Sign::Negative),
        // Wider than the window.
        ([t, t], [h, -h], [h, h], 0.0, Sign::Positive),
    ]
}

#[test]
fn point_fixed_cases_agree_with_the_oracle_and_the_expected_sign() {
    for (i, (p, q, d, sign)) in point_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            point_oracle(p, q, d),
            sign,
            "case {i}: the oracle disagrees with the fixture"
        );
        assert_eq!(compare_point_distance(p, q, d), Ok(sign), "case {i}");
        assert_eq!(
            compare_point_distance(q, p, d),
            Ok(sign),
            "case {i}, points swapped"
        );
        assert_eq!(compare_point_distance(p, q, -d), Ok(sign), "case {i}, -d");
    }
}

#[test]
fn segment_fixed_cases_agree_with_the_oracle_and_the_expected_sign() {
    for (i, (p, a, b, d, sign)) in segment_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            segment_oracle(p, a, b, d),
            sign,
            "case {i}: the oracle disagrees with the fixture"
        );
        assert_eq!(compare_segment_distance(p, a, b, d), Ok(sign), "case {i}");
        assert_eq!(
            compare_segment_distance(p, b, a, d),
            Ok(sign),
            "case {i}, segment reversed"
        );
        assert_eq!(
            compare_segment_distance(p, a, b, -d),
            Ok(sign),
            "case {i}, -d"
        );
    }
}

/// The segment form agrees with the point form on degenerate segments, for
/// random values: `compare_segment_distance(p, a, a, d)` is
/// `compare_point_distance(p, a, d)`.
#[test]
fn a_degenerate_segment_is_its_point() {
    let mut rng = SplitMix64::new(0x2026_1030_D157_0001);
    for _ in 0..2_000 {
        let e = rng.range(-400, 400);
        let mut f = || rng.float_with_exponent(e);
        let (p, a, d) = ([f(), f()], [f(), f()], f());
        assert_eq!(
            compare_segment_distance(p, a, a, d),
            compare_point_distance(p, a, d)
        );
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
    fn point_uniform_cases_match_the_oracle(
        px in finite(), py in finite(),
        qx in finite(), qy in finite(),
        d in finite(),
    ) {
        let (p, q) = ([px, py], [qx, qy]);
        prop_assert_eq!(compare_point_distance(p, q, d), Ok(point_oracle(p, q, d)));
    }

    #[test]
    fn segment_uniform_cases_match_the_oracle(
        px in finite(), py in finite(),
        ax in finite(), ay in finite(),
        bx in finite(), by in finite(),
        d in finite(),
    ) {
        let (p, a, b) = ([px, py], [ax, ay], [bx, by]);
        prop_assert_eq!(compare_segment_distance(p, a, b, d), Ok(segment_oracle(p, a, b, d)));
    }
}

// ===========================================================================
// Near-degenerate cases
// ===========================================================================

/// A float with its high 26 bits kept: sums with another such value of a
/// nearby exponent are exact.
fn coarse(rng: &mut SplitMix64, e: i64) -> f64 {
    let v = rng.float_with_exponent(e);
    f64::from_bits(v.to_bits() & !0x0000_0007_FFFF_FFFF)
}

/// An exactly at-distance point configuration, then one value moved by `k`
/// floats, `k` in `-3..=3` (zero keeps the exact tie).
///
/// The ties are built with exact operations only. One kind sets
/// `q − p = (3t, 4t)` and `d = 5t` for a `t` that is a small odd integer
/// times a power of two, so that `3t`, `4t` and `5t` are exact, with `p`
/// coarse enough that the additions are exact. The other is an axis-aligned
/// tie, `q − p = (±d, 0)`. Magnitudes range from deep in the subnormals to
/// near overflow, so every stage is reached.
fn point_near_ties() -> impl Strategy<Value = ([f64; 2], [f64; 2], f64)> {
    (any::<u64>(), 0_usize..2, 0_usize..5, -3_i64..=3).prop_map(|(seed, kind, index, k)| {
        let mut rng = SplitMix64::new(seed);
        let e = rng.range(-1050, 1010);
        let odd = ((rng.next() % 1_023) | 1) as f64;
        let t = odd * 2.0_f64.powi(rng.range(e - 9, e - 8) as i32 - 10);
        let p = [coarse(&mut rng, e), coarse(&mut rng, e)];
        let (u, d) = if kind == 0 {
            ([3.0 * t, 4.0 * t], 5.0 * t)
        } else {
            ([if rng.coin() { 8.0 * t } else { -8.0 * t }, 0.0], 8.0 * t)
        };
        let mut config = [p[0], p[1], p[0] + u[0], p[1] + u[1], d];
        let moved = from_ordinal(ordinal(config[index]) + k);
        if moved.is_finite() {
            config[index] = moved;
        }
        ([config[0], config[1]], [config[2], config[3]], config[4])
    })
}

/// An exactly at-distance segment configuration, then one value moved by
/// `k` floats.
///
/// Three exact constructions, on coarse grids so every sum is exact:
///
/// - **interior**, an axis-aligned segment with `p` at height `±d` above a
///   point strictly inside it (and its form rotated by 90°);
/// - **endpoint**, where the nearest point is `a`, with `p − a = (3t, 4t)`,
///   `d = 5t`, and `b` on the far side of `a`;
/// - **on the segment**, with `p` inside it and `d = 0`.
fn segment_near_ties() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2], f64)> {
    (any::<u64>(), 0_usize..4, 0_usize..7, -3_i64..=3).prop_map(|(seed, kind, index, k)| {
        let mut rng = SplitMix64::new(seed);
        let e = rng.range(-1040, 1000);
        let odd = ((rng.next() % 1_023) | 1) as f64;
        let t = odd * 2.0_f64.powi(rng.range(e - 9, e - 8) as i32 - 10);
        let a = [coarse(&mut rng, e), coarse(&mut rng, e)];
        let w = coarse(&mut rng, e).abs();
        let inside = w * 0.5;
        let mut config = match kind {
            // Interior, horizontal, p above; d of either sign.
            0 => {
                let d = if rng.coin() { 4.0 * t } else { -4.0 * t };
                [a[0] + inside, a[1] + 4.0 * t, a[0], a[1], a[0] + w, a[1], d]
            }
            // Interior, vertical.
            1 => [
                a[0] + 4.0 * t,
                a[1] + inside,
                a[0],
                a[1],
                a[0],
                a[1] + w,
                4.0 * t,
            ],
            // Endpoint a nearest: b on the far side.
            2 => [
                a[0] + 3.0 * t,
                a[1] + 4.0 * t,
                a[0],
                a[1],
                a[0] - w,
                a[1],
                5.0 * t,
            ],
            // On the segment, d = 0.
            _ => [a[0] + inside, a[1], a[0], a[1], a[0] + w, a[1], 0.0],
        };
        let moved = from_ordinal(ordinal(config[index]) + k);
        if moved.is_finite() {
            config[index] = moved;
        }
        (
            [config[0], config[1]],
            [config[2], config[3]],
            [config[4], config[5]],
            config[6],
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    #[test]
    fn point_near_degenerate_cases_match_the_oracle((p, q, d) in point_near_ties()) {
        prop_assert_eq!(compare_point_distance(p, q, d), Ok(point_oracle(p, q, d)));
    }

    #[test]
    fn segment_near_degenerate_cases_match_the_oracle((p, a, b, d) in segment_near_ties()) {
        prop_assert_eq!(compare_segment_distance(p, a, b, d), Ok(segment_oracle(p, a, b, d)));
    }
}

// ===========================================================================
// Non-finite input
// ===========================================================================

#[test]
fn every_argument_position_rejects_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..5 {
            let mut c = [0.5, -1.0, 2.0, 3.0, 0.25];
            c[position] = bad;
            assert!(
                compare_point_distance([c[0], c[1]], [c[2], c[3]], c[4]).is_err(),
                "point form, position {position} with {bad}"
            );
        }
        for position in 0..7 {
            let mut c = [0.5, -1.0, 2.0, 3.0, -4.0, 0.25, 1.5];
            c[position] = bad;
            assert!(
                compare_segment_distance([c[0], c[1]], [c[2], c[3]], [c[4], c[5]], c[6]).is_err(),
                "segment form, position {position} with {bad}"
            );
        }
    }
    // A value equal to f64::MAX is finite, and answered.
    assert!(compare_point_distance([f64::MAX, 0.0], [0.0, 0.0], f64::MAX).is_ok());
    assert!(compare_segment_distance([f64::MAX, 0.0], [0.0, 0.0], [0.0, 1.0], f64::MAX).is_ok());
}

// ===========================================================================
// Boundary-seeking runs and the neighbors of exact ties
// ===========================================================================

const BOUNDARY_CASES: u64 = 10_000;

fn seed(number: u64) -> u64 {
    0x2026_1030_C0D1_0000 + number
}

#[test]
fn point_boundary_runs_in_every_profile_match_the_oracle() {
    for (i, profile) in Profile::ALL.into_iter().enumerate() {
        run_boundary(
            Pred::ComparePointDistance,
            profile,
            BOUNDARY_CASES,
            seed(i as u64),
        )
        .assert_clean();
    }
}

#[test]
fn segment_boundary_runs_in_every_profile_match_the_oracle() {
    for (i, profile) in Profile::ALL.into_iter().enumerate() {
        run_boundary(
            Pred::CompareSegmentDistance,
            profile,
            BOUNDARY_CASES,
            seed(10 + i as u64),
        )
        .assert_clean();
    }
}

/// Exact segment ties, each of the seven values moved by one and two floats
/// either way.
#[test]
fn segment_exact_ties_and_their_neighbors_match_the_oracle() {
    let mut rng = SplitMix64::new(seed(20));
    let mut report = Report {
        label: "compare_segment_distance/degenerate".into(),
        ..Report::default()
    };
    for _ in 0..1_000 {
        let e = rng.range(-1040, 1000);
        let odd = ((rng.next() % 1_023) | 1) as f64;
        let t = odd * 2.0_f64.powi(rng.range(e - 9, e - 8) as i32 - 10);
        let a = [coarse(&mut rng, e), coarse(&mut rng, e)];
        let w = coarse(&mut rng, e).abs();
        let config = [
            a[0] + w * 0.5,
            a[1] + 4.0 * t,
            a[0],
            a[1],
            a[0] + w,
            a[1],
            4.0 * t,
        ];
        if w == 0.0 {
            continue;
        }
        assert_eq!(
            support::oracle(Pred::CompareSegmentDistance, &config),
            Sign::Zero,
            "the construction must be an exact tie"
        );
        report.check(Pred::CompareSegmentDistance, &config);
        for index in 0..7 {
            for step in [-2_i64, -1, 1, 2] {
                let mut neighbor = config;
                neighbor[index] = from_ordinal(ordinal(config[index]) + step);
                if neighbor[index].is_finite() {
                    report.check(Pred::CompareSegmentDistance, &neighbor);
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
fn the_two_oracles_agree_on_every_fixed_case() {
    for (i, (p, q, d, _)) in point_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            point_oracle(p, q, d),
            point_oracle_rational(p, q, d),
            "point case {i}"
        );
    }
    for (i, (p, a, b, d, _)) in segment_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            segment_oracle(p, a, b, d),
            segment_oracle_rational(p, a, b, d),
            "segment case {i}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4_000))]

    #[test]
    fn the_two_oracles_agree_on_random_cases(
        uniform in (finite(), finite(), finite(), finite(), finite(), finite(), finite()),
        (p, q, d) in point_near_ties(),
        (sp, sa, sb, sd) in segment_near_ties(),
    ) {
        let (a, b, c, e, f, g, h) = uniform;
        prop_assert_eq!(
            point_oracle([a, b], [c, e], f),
            point_oracle_rational([a, b], [c, e], f)
        );
        prop_assert_eq!(
            segment_oracle([a, b], [c, e], [f, g], h),
            segment_oracle_rational([a, b], [c, e], [f, g], h)
        );
        prop_assert_eq!(point_oracle(p, q, d), point_oracle_rational(p, q, d));
        prop_assert_eq!(segment_oracle(sp, sa, sb, sd), segment_oracle_rational(sp, sa, sb, sd));
    }
}
