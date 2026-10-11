// tests/stage_counters.rs

//! Integration tests for the stage counters in `arithmetica::counters`,
//! compiled only with `--features stage-counters`.
//!
//! Functional check that the counters observe the real adaptive ladder, not
//! just their own arithmetic.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp           an exact-value assertion IS a float comparison. These
//                        tests assert bit-exact results against a rational
//                        oracle or a hand-computed value; a tolerance would
//                        defeat their purpose.
//   unreadable_literal  bit patterns and reference coordinates are
//                        transcribed as written in the source they come from,
//                        so that they can be checked against it.
//   cast_*              fixture construction, where the ranges are known by
//                        construction.
//   unwrap_used         in a test, a panic is the failure signal.
#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::no_effect_underscore_binding,
    clippy::if_not_else,
    clippy::comparison_chain
)]
#![cfg(feature = "stage-counters")]

use arithmetica::counters::{Predicate, reset, snapshot};
use arithmetica::{Sign, orient2d};

#[test]
fn ordinary_input_stays_on_the_fast_path_and_degenerate_input_reaches_stage_four() {
    reset();

    // Clearly non-degenerate: the fast path must resolve this.
    assert_eq!(
        orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
        Ok(Sign::Positive)
    );
    let after_easy = snapshot(Predicate::Orient2d);
    assert_eq!(after_easy.fast, 1, "{after_easy:?}");
    assert_eq!(after_easy.exact, 0, "{after_easy:?}");

    // Three distinct points on the line y = x, no two of which share an x or
    // a y coordinate, so the zero filter does not answer.
    // The exact determinant is zero. A zero is never certified by Stages 1-3
    // (the filter's strict comparison cannot accept it, and a zero from the
    // expansion stages is passed on), so Stage 4 must run to confirm it.
    reset();
    let e = 1.0_f64 / 3.0;
    assert_eq!(
        orient2d([0.5, 0.5], [12.0, 12.0], [24.0 + e, 24.0 + e]),
        Ok(Sign::Zero)
    );
    let after_hard = snapshot(Predicate::Orient2d);
    assert_eq!(
        after_hard.exact, 1,
        "Stage 4 must fire here: {after_hard:?}"
    );
    assert_eq!(after_hard.exact_rate(), Some(1.0));

    // Out-of-domain input is counted separately and excluded from the rate.
    reset();
    assert!(orient2d([f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]).is_err());
    let after_nan = snapshot(Predicate::Orient2d);
    assert_eq!(after_nan.non_finite, 1, "{after_nan:?}");
    assert_eq!(after_nan.exact_rate(), None);
}
