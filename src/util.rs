// src/util.rs

//! Shared private arithmetic helpers for predicate modules.
//!
//! These functions are used by both `orient2d` and `incircle`. Centralizing
//! them here avoids duplication and ensures both predicates use identical
//! implementations of the low-level building blocks.

use crate::Sign;
use crate::expansion::two_sum;

// ===========================================================================
// Sign extraction
// ===========================================================================

/// Returns the `Sign` of a single `f64` value.
///
/// Used as the final step in each predicate stage: once a determinant
/// estimate `det` is known to have the correct sign, this converts it to
/// the predicate's return type.
#[inline]
pub(crate) fn sign_of(det: f64) -> Sign {
    if det > 0.0 {
        Sign::Positive
    } else if det < 0.0 {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

// ===========================================================================
// Two-term difference arithmetic
// ===========================================================================

/// Computes the rounding error tail when subtracting two `f64` values.
///
/// Given `x = fl(a − b)`, returns `y` such that `x + y == a − b` exactly as
/// real numbers. This implements the `Two_Diff_Tail` macro from Shewchuk's
/// `predicates.c`.
///
/// # Precondition
///
/// `x` must equal `fl(a − b)` (the IEEE 754 rounded difference of `a` and `b`).
/// The caller is responsible for computing `x` before calling this function.
#[inline]
pub(crate) fn two_diff_tail(a: f64, b: f64, x: f64) -> f64 {
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    around + bround
}

/// Computes `(a1 + a0) − (b1 + b0)` as an exact four-component expansion.
///
/// Returns `[x0, x1, x2, x3]` in ascending magnitude order such that their
/// sum equals `(a1 + a0) − (b1 + b0)` exactly as real numbers.
///
/// Implements Shewchuk's `Two_Two_Diff` macro from `predicates.c`, composed of
/// two applications of `Two_One_Diff`. Uses the public `two_sum` kernel for
/// intermediate accumulation.
///
/// # Precondition
///
/// `(a1, a0)` and `(b1, b0)` must each be valid non-overlapping two-term
/// expansions (i.e., each pair is the output of `two_product` or `two_sum`).
#[inline]
pub(crate) fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> [f64; 4] {
    // Two_One_Diff(a1, a0, b0) → (s_hi, s_lo, x0):
    // s_lo = fl(a0 − b0); x0 = rounding tail of that subtraction.
    // s_hi = fl(a1 + s_lo); accumulates the carry.
    let s_lo_a = a0 - b0;
    let x0 = two_diff_tail(a0, b0, s_lo_a);
    let (s_hi, s_lo) = two_sum(a1, s_lo_a);

    // Two_One_Diff(s_hi, s_lo, b1) → (x3, x2, x1):
    let t_lo = s_lo - b1;
    let x1 = two_diff_tail(s_lo, b1, t_lo);
    let (x3, x2) = two_sum(s_hi, t_lo);

    [x0, x1, x2, x3]
}
