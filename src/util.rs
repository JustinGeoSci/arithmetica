// src/util.rs

//! Private arithmetic helpers shared by the predicate modules.
//!
//! Several predicates use these functions. Keeping them in one place avoids
//! duplication and establishes that every predicate operates on identical
//! implementations of the low-level building blocks.

use crate::Sign;
use crate::expansion::{expansion_sum, scale_expansion, two_sum};

// ===========================================================================
// Sign extraction
// ===========================================================================

/// Returns the `Sign` of a single `f64` value.
///
/// Each predicate stage ends here. Once a determinant estimate `det` is
/// known to have the correct sign, this converts it to the predicate's
/// return type.
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
/// real numbers. This is the `Two_Diff_Tail` macro of Shewchuk's
/// `predicates.c`.
///
/// # Precondition
///
/// `x` must equal `fl(a − b)`, the IEEE 754 rounded difference of `a` and
/// `b`. The caller computes `x` before the call.
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
/// This is Shewchuk's `Two_Two_Diff` macro from `predicates.c`, composed of
/// two applications of `Two_One_Diff`, with the public `two_sum` kernel for
/// the intermediate accumulation.
///
/// # Precondition
///
/// `(a1, a0)` and `(b1, b0)` must each be valid non-overlapping two-term
/// expansions, such as the output of `two_product` or `two_sum`.
#[inline]
pub(crate) fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> [f64; 4] {
    // Two_One_Diff(a1, a0, b0) → (s_hi, s_lo, x0).
    // s_lo = fl(a0 − b0), and x0 is the rounding tail of that subtraction.
    // s_hi = fl(a1 + s_lo) accumulates the carry.
    let s_lo_a = a0 - b0;
    let x0 = two_diff_tail(a0, b0, s_lo_a);
    let (s_hi, s_lo) = two_sum(a1, s_lo_a);

    // Two_One_Diff(s_hi, s_lo, b1) → (x3, x2, x1).
    let t_lo = s_lo - b1;
    let x1 = two_diff_tail(s_lo, b1, t_lo);
    let (x3, x2) = two_sum(s_hi, t_lo);

    [x0, x1, x2, x3]
}

// ===========================================================================
// Exact degree-2 expansions, for the degree-2 sign predicates
// ===========================================================================

/// `a − b` exactly, as the two-term expansion `[tail, head]` in ascending
/// magnitude.
///
/// `head = fl(a − b)` and `tail` is its rounding error, so
/// `head + tail == a − b` as real numbers (Shewchuk's `Two_Diff`). The
/// result is exact whenever `a − b` does not overflow, and the caller runs
/// it only on in-window coordinates, where nothing overflows.
#[inline]
pub(crate) fn two_diff(a: f64, b: f64) -> [f64; 2] {
    let head = a - b;
    [two_diff_tail(a, b, head), head]
}

/// The exact product of two two-term expansions, `(x1 + x0)(y1 + y0)`, as an
/// expansion of at most 8 components. Returns the buffer and its length.
///
/// `scale_expansion` multiplies the whole of `x` by each component of `y`
/// exactly, at most 4 components each, and `expansion_sum` adds the two
/// results exactly, at most 8, so the output sums to the product with no
/// rounding and all four partial products `x1·y1`, `x1·y0`, `x0·y1`,
/// `x0·y0` are present. The result is exact under the conditions of those
/// two primitives, which the exactness window establishes for degree-2
/// input.
#[inline]
pub(crate) fn two_by_two_product(x: [f64; 2], y: [f64; 2]) -> ([f64; 8], usize) {
    let mut by_high = [0.0_f64; 4];
    let mut by_low = [0.0_f64; 4];
    let high_len = scale_expansion(&x, y[1], &mut by_high);
    let low_len = scale_expansion(&x, y[0], &mut by_low);
    let mut out = [0.0_f64; 8];
    let len = expansion_sum(&by_high[..high_len], &by_low[..low_len], &mut out);
    (out, len)
}

/// Negates every component of an expansion in place.
///
/// Negation is exact, since it changes only the sign bit, and a
/// non-overlapping expansion stays non-overlapping.
#[inline]
pub(crate) fn negate(e: &mut [f64]) {
    for v in e {
        *v = -*v;
    }
}
