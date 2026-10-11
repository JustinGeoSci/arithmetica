// tests/window_counters.rs

//! The stage half of the scale-invariance tests: inside the exactness window,
//! rescaling an exactly degenerate call by a power of two must not change
//! which stage answers it. Compiled only with `--features stage-counters`.
//!
//! `tests/window.rs` holds the sign half and the derivation of why scale
//! invariance is the right test for the window; this file assumes it.
//!
//! # Why only exactly degenerate input
//!
//! Lemma G is about the adaptive expansion stages, and those are scale
//! invariant inside the window. The fast filter in front of them is **not**,
//! deliberately: its underflow protection adds the absolute constants `u_N`
//! and `u_S` to the bound (Bartels et al. 2023, Theorem 1(2)), and an
//! absolute constant does not scale with the input. So for a configuration
//! whose determinant is small but non-zero, a one-float neighbor of a
//! degenerate one, the filter declines at small scales and certifies at
//! large ones. Both answers are correct, and the counters cannot tell the
//! filter's `Fast` from a later stage's `Fast`, so such a configuration
//! cannot be used to test the lemma. That is the price of a filter valid for
//! every finite input.
//!
//! An exactly degenerate configuration has no such ambiguity. The filter's
//! comparison is a strict `>` against a bound that is at least `u_S > 0`, so
//! a zero determinant can never satisfy it at any scale; and the zero filter
//! in `orient2d`, which is made of comparisons, gives the same answer at
//! every scale because scaling by a power of two preserves equality and
//! zeroness exactly. Whatever stage answers a degenerate call, therefore, the
//! same stage must answer every in-window rescaling of it. If one did not,
//! some operation of the expansion stages would have behaved differently at
//! the two scales, and the lemma would be false.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp           an exact-value assertion IS a float comparison.
//   unreadable_literal  seeds are written as they are reported.
//   cast_*              fixture construction from small integers.
//   unwrap_used         in a test, a panic is the failure signal.
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
#![cfg(feature = "stage-counters")]

mod support;

use std::sync::{Mutex, MutexGuard, PoisonError};

use arithmetica::counters::{Counts, Predicate, reset, snapshot};
use support::{
    Pred, SplitMix64, call, cocircular, collinear, config_span, coplanar, cospherical, oracle,
};

use arithmetica::Sign;

/// Each test draws from its own stream.
const fn seed(number: u64) -> u64 {
    0x2026_1010_D1D1_u64.wrapping_add(number)
}

/// `2^exponent` as an exact `f64`.
fn power_of_two(exponent: i32) -> f64 {
    assert!((-1022..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// The shifts `k` for which `2^k · config` is still inside the predicate's
/// window, as an inclusive range.
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

static SERIAL: Mutex<()> = Mutex::new(());

/// The counters are process-wide and the harness runs the tests of one file
/// in parallel, so each reset-call-read sequence takes the lock. A poisoned
/// lock from an earlier failure must not turn into a second failure here.
fn serialize() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn predicate_of(pred: Pred) -> Predicate {
    match pred {
        Pred::Orient2d => Predicate::Orient2d,
        Pred::Incircle => Predicate::Incircle,
        Pred::Orient3d => Predicate::Orient3d,
        Pred::Insphere => Predicate::Insphere,
    }
}

/// One call with the counters reset before it and read after it. The caller
/// holds the lock.
fn counted(pred: Pred, config: &[f64]) -> (Sign, Counts) {
    reset();
    let sign = call(pred, config);
    (sign, snapshot(predicate_of(pred)))
}

/// Every in-window rescaling of an exactly degenerate configuration is
/// answered by the same stage, and answered `Zero`.
fn assert_stage_is_scale_invariant(
    pred: Pred,
    build: fn(&mut SplitMix64) -> Vec<f64>,
    cases: usize,
    number: u64,
) {
    let _guard = serialize();
    let mut rng = SplitMix64::new(seed(number));
    let mut divergences = Vec::new();
    let mut checked = 0_u32;
    let mut reached_exact = 0_u32;

    for _ in 0..cases {
        let config = build(&mut rng);
        assert_eq!(
            oracle(pred, &config),
            Sign::Zero,
            "the fixture is not degenerate: {config:?}"
        );

        let (base_sign, base) = counted(pred, &config);
        assert_eq!(base_sign, Sign::Zero, "{config:?}");
        reached_exact += u32::from(base.exact > 0);

        let (lowest, highest) = in_window_shifts(pred, &config);
        for shift in [lowest, highest, i32::midpoint(lowest, highest)] {
            if !(-1022..=1023).contains(&shift) || shift == 0 {
                continue;
            }
            let moved = scaled(&config, power_of_two(shift));
            let (sign, after) = counted(pred, &moved);
            checked += 1;
            assert_eq!(sign, Sign::Zero, "shift {shift}: {config:?}");
            if after != base {
                divergences.push(format!(
                    "shift {shift}: {base:?} -> {after:?} for {config:?}"
                ));
            }
        }
    }

    assert!(checked > 0, "no rescaling was checked");
    // The fixtures must actually reach the exact stage, or the test is only
    // comparing calls answered by a filter.
    assert!(
        reached_exact > 0,
        "{}: no fixture reached the exact stage, so the deeper stages were not compared",
        pred.name()
    );
    assert!(
        divergences.is_empty(),
        "{}: {} of {checked} in-window rescalings changed which stage answered:\n{}",
        pred.name(),
        divergences.len(),
        divergences.join("\n")
    );
}

#[test]
fn orient2d_stage_is_scale_invariant() {
    assert_stage_is_scale_invariant(Pred::Orient2d, collinear, 400, 0);
}

#[test]
fn incircle_stage_is_scale_invariant() {
    assert_stage_is_scale_invariant(Pred::Incircle, cocircular, 150, 1);
}

#[test]
fn orient3d_stage_is_scale_invariant() {
    assert_stage_is_scale_invariant(Pred::Orient3d, coplanar, 150, 2);
}

#[test]
fn insphere_stage_is_scale_invariant() {
    assert_stage_is_scale_invariant(Pred::Insphere, cospherical, 50, 3);
}
