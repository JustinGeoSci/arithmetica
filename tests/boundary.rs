// tests/boundary.rs

//! Boundary-seeking correctness tests for the four floating-point predicates.
//! The method and the oracle are described in `tests/support/mod.rs`.
//!
//! # Three tiers
//!
//! - **Default**, 20 tests: each predicate in each of five magnitude profiles.
//!   `cargo test --test boundary`
//! - **Degenerate**, 4 tests and a fixture check: exactly degenerate input
//!   and its one- and two-float neighbors. Runs with the default tier.
//! - **Deep**, 4 tests, `#[ignore]`: each predicate over all five profiles,
//!   about 18 million calls in total.
//!   `cargo test --release --test boundary -- --ignored --nocapture`
//!
//! # What the profiles cover
//!
//! The `unit` profile checks ordinary magnitudes. The `small`, `mixed` and
//! `any` profiles place the boundary where products of differences are
//! subnormal or overflow, or where one call mixes huge and tiny coordinates;
//! the `edge` profile puts every call on both margins of the predicate's
//! exactness window at once. Every call is compared with the exact sign.
//!
//! # Reproducing a failure
//!
//! Every test is deterministic: a fixed seed, no clock, no environment. A
//! failure prints the first mismatching input as bit patterns, in a form
//! that pastes into `tests/regressions.rs`.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp          — an exact-value assertion IS a float comparison.
//   unreadable_literal — seeds and bit patterns are written as they are
//                        reported, so that they can be searched for.
//   cast_*             — fixture construction from small integers, where the
//                        ranges are known by construction.
//   unwrap_used        — in a test, a panic is the failure signal.
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

use arithmetica::Sign;
use support::{
    Pred, Profile, Report, SplitMix64, call, cocircular, collinear, coplanar, cospherical,
    from_ordinal, oracle, ordinal, power_of_two_scale, run_boundary,
};

// ===========================================================================
// Seeds
// ===========================================================================

/// Every test draws from its own stream: the tier and the test's number are
/// folded into one fixed base, so adding a test never shifts another's cases.
const fn seed(tier: u64, number: u64) -> u64 {
    0x2026_1010_A51C_0000 + (tier << 8) + number
}

const DEFAULT_TIER: u64 = 0;
const DEGENERATE_TIER: u64 = 1;
const DEEP_TIER: u64 = 2;

// ===========================================================================
// Default tier: twenty tests
// ===========================================================================

// Case counts are set by cost and by sensitivity. A case costs one boundary
// search (about 64 oracle evaluations) plus up to eight comparisons, and the
// oracle's cost grows with the predicate's degree and point count, so the
// higher-degree predicates get fewer cases. `orient2d` is the cheapest and
// gets the most, so that an error confined to one call in several thousand
// near the boundary would still show up as a mismatch in its run.

const ORIENT2D_CASES: u64 = 20_000;
const INCIRCLE_CASES: u64 = 4_000;
const ORIENT3D_CASES: u64 = 4_000;
const INSPHERE_CASES: u64 = 1_500;

macro_rules! boundary_tests {
    ($( $name:ident: $pred:ident, $profile:ident, $cases:expr, $number:expr; )*) => {
        $(
            #[test]
            fn $name() {
                run_boundary(
                    Pred::$pred,
                    Profile::$profile,
                    $cases,
                    seed(DEFAULT_TIER, $number),
                )
                .assert_clean();
            }
        )*
    };
}

// The `edge` profile puts every call on both of the window's margins at once.
boundary_tests! {
    orient2d_unit_boundary:  Orient2d, Unit,  ORIENT2D_CASES, 0;
    orient2d_small_boundary: Orient2d, Small, ORIENT2D_CASES, 1;
    orient2d_mixed_boundary: Orient2d, Mixed, ORIENT2D_CASES, 2;
    orient2d_any_boundary:   Orient2d, Any,   ORIENT2D_CASES, 3;
    incircle_unit_boundary:  Incircle, Unit,  INCIRCLE_CASES, 4;
    incircle_small_boundary: Incircle, Small, INCIRCLE_CASES, 5;
    incircle_mixed_boundary: Incircle, Mixed, INCIRCLE_CASES, 6;
    incircle_any_boundary:   Incircle, Any,   INCIRCLE_CASES, 7;
    orient3d_unit_boundary:  Orient3d, Unit,  ORIENT3D_CASES, 8;
    orient3d_small_boundary: Orient3d, Small, ORIENT3D_CASES, 9;
    orient3d_mixed_boundary: Orient3d, Mixed, ORIENT3D_CASES, 10;
    orient3d_any_boundary:   Orient3d, Any,   ORIENT3D_CASES, 11;
    insphere_unit_boundary:  Insphere, Unit,  INSPHERE_CASES, 12;
    insphere_small_boundary: Insphere, Small, INSPHERE_CASES, 13;
    insphere_mixed_boundary: Insphere, Mixed, INSPHERE_CASES, 14;
    insphere_any_boundary:   Insphere, Any,   INSPHERE_CASES, 15;
    orient2d_edge_boundary:  Orient2d, Edge,  ORIENT2D_CASES, 16;
    incircle_edge_boundary:  Incircle, Edge,  INCIRCLE_CASES, 17;
    orient3d_edge_boundary:  Orient3d, Edge,  ORIENT3D_CASES, 18;
    insphere_edge_boundary:  Insphere, Edge,  INSPHERE_CASES, 19;
}

// ===========================================================================
// Degenerate tier: exact zeros and their nearest neighbors
// ===========================================================================

// Each configuration is degenerate by construction, from small integers, so
// its determinant is zero as a statement about integers and not as an
// observation about floating-point arithmetic. The whole configuration is
// then multiplied by one power of two, which is exact and keeps it
// degenerate. Finally each coordinate in turn is moved one and two floats in
// each direction, and every result is compared with the oracle.
//
// These are the inputs on which a fast path is most tempted to answer
// `Zero`, and the neighbors are the inputs on which `Zero` is wrong by the
// smallest possible margin.

/// Runs the degenerate tier for one predicate.
fn run_degenerate(pred: Pred, build: fn(&mut SplitMix64) -> Vec<f64>, cases: u64, number: u64) {
    let mut rng = SplitMix64::new(seed(DEGENERATE_TIER, number));
    let mut report = Report {
        label: format!("{}/degenerate", pred.name()),
        ..Report::default()
    };
    for _ in 0..cases {
        let factor = power_of_two_scale(&mut rng);
        let config: Vec<f64> = build(&mut rng).iter().map(|c| c * factor).collect();

        // The construction itself is checked, not assumed.
        assert_eq!(
            oracle(pred, &config),
            Sign::Zero,
            "{}: the construction is not degenerate: {config:?}",
            pred.name()
        );
        report.check(pred, &config);

        for index in 0..config.len() {
            let center = ordinal(config[index]);
            for step in [-2_i64, -1, 1, 2] {
                let mut neighbor = config.clone();
                neighbor[index] = from_ordinal(center + step);
                report.check(pred, &neighbor);
            }
        }
    }
    report.assert_clean();
}

#[test]
fn orient2d_degenerate_neighbors() {
    run_degenerate(Pred::Orient2d, collinear, 2_000, 0);
}

#[test]
fn incircle_degenerate_neighbors() {
    run_degenerate(Pred::Incircle, cocircular, 400, 1);
}

#[test]
fn orient3d_degenerate_neighbors() {
    run_degenerate(Pred::Orient3d, coplanar, 400, 2);
}

#[test]
fn insphere_degenerate_neighbors() {
    run_degenerate(Pred::Insphere, cospherical, 60, 3);
}

/// The degenerate tier's own fixtures, checked without the predicate: a
/// generator that stopped producing degenerate input would turn the tier
/// into a test of nothing.
#[test]
fn degenerate_constructions_are_exactly_degenerate() {
    let mut rng = SplitMix64::new(seed(DEGENERATE_TIER, 4));
    for _ in 0..200 {
        assert_eq!(oracle(Pred::Orient2d, &collinear(&mut rng)), Sign::Zero);
        assert_eq!(oracle(Pred::Incircle, &cocircular(&mut rng)), Sign::Zero);
        assert_eq!(oracle(Pred::Orient3d, &coplanar(&mut rng)), Sign::Zero);
        assert_eq!(oracle(Pred::Insphere, &cospherical(&mut rng)), Sign::Zero);
    }
    // And the oracle is not constant: a right-handed unit triangle is
    // counterclockwise, so its orientation is positive.
    let triangle = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(oracle(Pred::Orient2d, &triangle), Sign::Positive);
    assert_eq!(call(Pred::Orient2d, &triangle), Sign::Positive);
}

// ===========================================================================
// Deep tier: about 18 million calls in total
// ===========================================================================

// Too slow for every `cargo test`, so ignored by default and run in release.
// Each test runs all five profiles and prints one line per profile before
// it asserts, so a failure in one profile does not hide the others.

fn run_deep(pred: Pred, cases: u64, number: u64) {
    let mut total = Report {
        label: format!("{}/all profiles", pred.name()),
        ..Report::default()
    };
    for (i, profile) in Profile::ALL.into_iter().enumerate() {
        let report = run_boundary(pred, profile, cases, seed(DEEP_TIER, number * 4 + i as u64));
        println!("{}", report.summary());
        total.absorb(report);
    }
    println!("{}", total.summary());
    total.assert_clean();
}

#[test]
#[ignore = "deep tier: minutes in release; run with --release -- --ignored --nocapture"]
fn orient2d_boundary_deep() {
    run_deep(Pred::Orient2d, 300_000, 0);
}

#[test]
#[ignore = "deep tier: minutes in release; run with --release -- --ignored --nocapture"]
fn incircle_boundary_deep() {
    run_deep(Pred::Incircle, 64_000, 1);
}

#[test]
#[ignore = "deep tier: minutes in release; run with --release -- --ignored --nocapture"]
fn orient3d_boundary_deep() {
    run_deep(Pred::Orient3d, 100_000, 2);
}

#[test]
#[ignore = "deep tier: minutes in release; run with --release -- --ignored --nocapture"]
fn insphere_boundary_deep() {
    run_deep(Pred::Insphere, 64_000, 3);
}
