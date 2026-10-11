// tests/window.rs

//! Scale invariance: multiplying every coordinate of a call by one power of
//! two must not change the answer, and inside the exactness window must not
//! change which stage gives it.
//!
//! # What this tests, and why it is the right test for the window
//!
//! Each predicate's determinant is homogeneous of degree `d` in the
//! coordinates, so scaling them all by `2^k` multiplies the determinant by
//! `2^(d·k)`, a positive number. The exact sign is therefore invariant; for
//! every `k`, in every model of arithmetic. That makes the **sign** half of
//! this file a plain correctness test. It is also where a rescale with the
//! wrong sign would show: `orient3d` and `insphere` are of odd degree, so a
//! negative factor would invert their sign.
//!
//! The **stage** half is a test of Lemma G, the claim behind the exactness
//! window (`docs/exactness_window.md`, and the module doc of the crate's
//! `range` module). The lemma says that inside the window every operation of
//! the adaptive stages returns the same value in binary64 as it would with an
//! unbounded exponent range. Arithmetic in that model is exactly scale
//! invariant: scaling the input by `2^k` scales every intermediate by a power
//! of two and leaves every significand alone, so each comparison against an
//! error bound falls the same way and the same stage answers. So if two
//! in-window scalings of one configuration were resolved by different stages,
//! some operation would have to have behaved differently at the two scales,
//! and the lemma would be false.
//!
//! The test cannot prove the lemma. What it can do is fail if the lemma is
//! wrong anywhere it looks, and it looks at the input that reaches the
//! deepest stages: exactly degenerate configurations and their one-float
//! neighbors.
//!
//! The stage half lives in `tests/window_counters.rs`, which needs the
//! `stage-counters` feature. It is a separate test binary, hence a separate
//! process, because the counters are process-wide: the tests in this file
//! call the predicates too, and run in parallel, so they would be counted
//! into a reading taken here.
//!
//! # Scalings outside the window
//!
//! A scaling that takes the input out of the window is still required to give
//! the same sign, but may legitimately be answered by a different stage: the
//! predicate then either rescales the input itself or goes to the exact
//! integer stage. Those cases are checked for the sign only.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp          — an exact-value assertion IS a float comparison.
//   unreadable_literal — seeds are written as they are reported.
//   cast_*             — fixture construction from small integers.
//   unwrap_used        — in a test, a panic is the failure signal.
#![allow(
    clippy::float_cmp,
    clippy::incompatible_msrv,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unwrap_used
)]

mod support;

use support::{
    Pred, SplitMix64, call, cocircular, collinear, config_span, coplanar, cospherical,
    from_ordinal, oracle, ordinal,
};

/// Each test draws from its own stream.
const fn seed(number: u64) -> u64 {
    0x2026_1010_D1D0_u64.wrapping_add(number)
}

/// `2^exponent` as an exact `f64`. The exponent stays inside the normal
/// range, so the result has a zero significand field.
fn power_of_two(exponent: i32) -> f64 {
    assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// The shifts `k` for which `2^k · config` is still inside the predicate's
/// window, as an inclusive range `(lowest, highest)`.
///
/// From the window's definition: the lowest set bit `q` must stay at or above
/// `q_min`, so `k >= q_min - q`; the leading bit `E` must stay at or below
/// `e_max`, so `k <= e_max - E`.
fn in_window_shifts(pred: Pred, config: &[f64]) -> (i32, i32) {
    let (q_min, e_max) = pred.window();
    let (highest, lowest) = config_span(config).expect("the fixtures are not all zero");
    (q_min - lowest, e_max - highest)
}

/// Scales every coordinate by `factor`, which must be a power of two so that
/// the multiplication is exact.
fn scaled(config: &[f64], factor: f64) -> Vec<f64> {
    assert_eq!(factor.to_bits() & 0x000F_FFFF_FFFF_FFFF, 0);
    config.iter().map(|&c| c * factor).collect()
}

/// The fixtures: exactly degenerate configurations, and their one- and
/// two-float neighbors.
///
/// Degenerate input reaches the exact integer stage, and its neighbors reach
/// the adaptive expansion stages, so between them they exercise the whole
/// ladder rather than only the fast filter.
fn fixtures(pred: Pred, rng: &mut SplitMix64, cases: usize) -> Vec<Vec<f64>> {
    let build = match pred {
        Pred::Orient2d => collinear,
        Pred::Incircle => cocircular,
        Pred::Orient3d => coplanar,
        Pred::Insphere => cospherical,
    };
    let mut out = Vec::new();
    for _ in 0..cases {
        let base = build(rng);
        out.push(base.clone());
        for _ in 0..3 {
            let index = rng.index(base.len());
            let step = if rng.coin() { 1 } else { -1 } * if rng.coin() { 1 } else { 2 };
            let mut neighbor = base.clone();
            neighbor[index] = from_ordinal(ordinal(base[index]) + i64::from(step));
            out.push(neighbor);
        }
    }
    out
}

/// Every in-window scaling of a configuration gives the same sign, and so
/// does every out-of-window one.
fn assert_sign_is_scale_invariant(pred: Pred, cases: usize, number: u64) {
    let mut rng = SplitMix64::new(seed(number));
    let mut checked = 0_u32;
    for config in fixtures(pred, &mut rng, cases) {
        let want = oracle(pred, &config);
        assert_eq!(call(pred, &config), want, "unscaled: {config:?}");

        let (lowest, highest) = in_window_shifts(pred, &config);
        // Both ends of the admissible range, the middle, and two shifts that
        // deliberately leave it.
        let shifts = [
            lowest,
            highest,
            i32::midpoint(lowest, highest),
            lowest - 40,
            highest + 40,
        ];
        for shift in shifts {
            if !(-1022..=1023).contains(&shift) || shift == 0 {
                continue;
            }
            let moved = scaled(&config, power_of_two(shift));
            // The rescale itself must be exact, or the test is comparing two
            // different configurations.
            for (original, scaled_value) in config.iter().zip(&moved) {
                assert!(
                    scaled_value.is_finite() && (*scaled_value == 0.0) == (*original == 0.0),
                    "shift {shift} was not exact for {original:e}"
                );
            }
            assert_eq!(
                call(pred, &moved),
                want,
                "{}: shift {shift} changed the sign of {config:?}",
                pred.name()
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no scaling was checked");
}

#[test]
fn orient2d_sign_is_scale_invariant() {
    assert_sign_is_scale_invariant(Pred::Orient2d, 300, 0);
}

#[test]
fn incircle_sign_is_scale_invariant() {
    assert_sign_is_scale_invariant(Pred::Incircle, 120, 1);
}

#[test]
fn orient3d_sign_is_scale_invariant() {
    assert_sign_is_scale_invariant(Pred::Orient3d, 120, 2);
}

#[test]
fn insphere_sign_is_scale_invariant() {
    assert_sign_is_scale_invariant(Pred::Insphere, 40, 3);
}
