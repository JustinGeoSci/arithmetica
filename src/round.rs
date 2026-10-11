// src/round.rs

//! Exact bracketing of a rational value between its two nearest `f64`
//! neighbors.
//!
//! This is the one place in the project where an exact value becomes
//! inexact, and it is the reason the module exists on its own: the conversion
//! is not a convenience, it is a correctness boundary, and a caller that
//! builds on the rounded result should treat it as a new input, not as the
//! exact value it came from.
//!
//! # Why there is no division here
//!
//! Every predicate in this crate decides a *sign*, so nothing else needs a
//! quotient and no big-integer division exists. Rounding does need
//! `num / den`, but it does not need the quotient itself, only which of two
//! adjacent `f64` values the quotient lies closer to. That is a comparison,
//! and comparisons are what this crate already does exactly.
//!
//! So the strategy is the one the crate uses everywhere else, applied to a
//! new question:
//!
//! 1. **Estimate** in `f64`, from the top 64 bits of numerator and
//!    denominator and a single `u128` division. The estimate may be wrong.
//! 2. **Verify exactly.** Compare the true value against the two midpoints
//!    bracketing the candidate, using only `WideInt` multiplication, shifting
//!    and sign, the operations the predicates are built from.
//! 3. **Correct** by stepping one `f64` at a time, re-verifying each step.
//!
//! The estimate's quality affects only how many correction steps run. It
//! cannot affect the answer, because no path returns a value that has not
//! passed the exact comparison. An estimate that is wildly wrong costs time;
//! it does not cost correctness.
//!
//! # Rounding mode
//!
//! None is applied here. The result reports the value's exact position
//! relative to the two neighbors and their midpoint, so round-to-nearest with
//! ties-to-even (the IEEE 754 default) is a one-line selection for the
//! caller, and a coordinate that is exactly representable brackets to itself.
//! A tie is reported only when the exact comparison finds equality, never by
//! an epsilon.
//!
//! # Scaling
//!
//! A [`RationalCoord2d`](crate::RationalCoord2d)'s stored rational is `2^1074 ×`
//! the true coordinate. The true value of a stored `(num, den)` is therefore
//! `(num / den) · 2^−1074`, and that factor is carried explicitly through
//! every comparison below rather than canceled early, because it is the term
//! a transcription error would silently drop.

use core::cmp::Ordering;

use crate::exact_int::{RationalCoordInt, RoundingInt};

/// The `2^1074` scaling every stored rational carries.
///
/// Named rather than written inline at each use: the exponent appears in both
/// the comparison and the overflow test, and a mismatch between them would be
/// a wrong answer in the extreme range only, which is exactly where a test
/// suite is least likely to look.
const SCALE_EXP: i32 = 1074;

/// Iteration cap on the correction loop.
///
/// The estimate truncates numerator and denominator to 64 bits each and
/// divides exactly, so its relative error is bounded by roughly `2^−62`,
/// which is some four orders of magnitude finer than the `2^−53` spacing of
/// the result. One or two steps is the expectation and eight is already
/// unreachable; the cap exists so that a defect in the estimate shows up as a
/// panic in a debug build rather than as a hang.
const MAX_CORRECTIONS: u32 = 8;

/// Decomposes a positive finite `f64` into `(m, e)` with `x == m · 2^e`
/// exactly.
///
/// Normals get the implicit leading bit restored, so `m ∈ [2^52, 2^53)`.
/// Subnormals keep their stored mantissa at the fixed exponent `−1074`, so
/// `m ∈ [1, 2^52)`. Both are exact: no rounding happens here.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    if biased == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), biased - 1075)
    }
}

/// The exact midpoint of two adjacent positive `f64` values, as `(M, E)` with
/// the midpoint equal to `M · 2^E`.
///
/// Both inputs are decomposed and aligned to the smaller exponent, so no
/// binade-boundary case analysis is needed: when `x` and `y` straddle a power
/// of two their exponents differ by one and the alignment shift handles it.
/// `M < 2^55` by the argument in [`RoundingInt`]'s width proof.
fn midpoint(x: f64, y: f64) -> (u64, i32) {
    let (mx, ex) = decompose(x);
    let (my, ey) = decompose(y);
    // Only the larger value can need shifting, and never by more than one
    // place. For `x < y` adjacent, either they share a binade and `ex == ey`,
    // or `y` is the power of two starting the next binade and `ey == ex + 1`;
    // the subnormal-to-normal step shares an exponent as well. So `ex` is
    // always the smaller, and aligning `x` would always be a shift by zero —
    // a mutation setting that shift to zero was semantically equivalent,
    // which is how the line was found to be dead.
    debug_assert!(
        ey == ex || ey == ex + 1,
        "adjacent f64 exponents differ by ≤ 1"
    );
    let sy = u32::try_from(ey - ex).unwrap_or(0);
    let sum = mx + (my << sy);
    (sum, ex - 1)
}

/// Compares the true value `(num / den) · 2^−SCALE_EXP` against the dyadic
/// rational `m · 2^e`.
///
/// `num` must be non-negative and `den` strictly positive; the caller has
/// already split off the sign. Returns the ordering of the *true value*
/// relative to the dyadic.
///
/// Cross-multiplying by `den > 0` cannot change the ordering, which is the
/// whole reason `RationalCoord2d` normalizes its denominator positive.
fn cmp_value_to_dyadic(num: &RationalCoordInt, den: &RationalCoordInt, m: u64, e: i32) -> Ordering {
    debug_assert!(!num.is_negative(), "cmp_value_to_dyadic needs a magnitude");
    debug_assert!(den.is_positive(), "den > 0 is a RationalCoord2d invariant");

    // num / den · 2^−SCALE_EXP  ⋛  m · 2^e
    //   ⟺  num  ⋛  m · den · 2^(e + SCALE_EXP)
    let e_prime = e + SCALE_EXP;

    let left_shift = usize::try_from(-e_prime.min(0)).unwrap_or(0);
    let right_shift = usize::try_from(e_prime.max(0)).unwrap_or(0);

    let left: RoundingInt = num.widen::<100>() << left_shift;
    let right: RoundingInt = (RoundingInt::from_u64(m) * den.widen::<100>()) << right_shift;

    let diff = left - right;
    if diff.is_zero() {
        Ordering::Equal
    } else if diff.is_negative() {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// The next `f64` above a positive finite `x`, or infinity past the maximum.
///
/// Bit-pattern increment rather than arithmetic: for positive finite values
/// the IEEE encoding is monotone in its bit pattern, so this steps exactly
/// one representable value and crosses binade boundaries and the
/// subnormal/normal boundary without a special case.
fn next_up(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1)
}

/// The next `f64` below a positive `x`. At zero it stays at zero, which the
/// caller relies on to mean "the interval is bounded below by zero".
fn next_down(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// An `f64` estimate of `a / d · 2^−SCALE_EXP`, accurate to a few ulps.
///
/// Takes the top 64 bits of each magnitude with their shifts, divides exactly
/// in `u128`, and assembles the result from the quotient's top 53 bits and the
/// accumulated exponent. The estimate is allowed to be wrong (see the module
/// doc comment) so this function carries no correctness obligation beyond
/// landing close enough that the correction loop terminates inside
/// [`MAX_CORRECTIONS`].
fn estimate(a: &RationalCoordInt, d: &RationalCoordInt) -> f64 {
    let ba = a.bit_length();
    let bd = d.bit_length();

    let sa = ba.saturating_sub(64);
    let sd = bd.saturating_sub(64);
    let ta = a.bit_window(sa);
    let td = d.bit_window(sd);
    if td == 0 {
        // Unreachable for den > 0: bit_window at bit_length − 64 of a nonzero
        // magnitude always contains the leading bit. Guarded rather than
        // divided by.
        return 0.0;
    }

    // q ≈ (ta / td) · 2^64, exact on the truncated operands.
    let q = (u128::from(ta) << 64) / u128::from(td);
    if q == 0 {
        return 0.0;
    }
    // `leading_zeros` on a u128 is at most 128, so the difference is in
    // 0..=128 and the conversion cannot wrap.
    let bq = 128 - i32::try_from(q.leading_zeros()).unwrap_or(128);

    // a / d ≈ q · 2^(sa − sd − 64), and the true value divides that by
    // 2^SCALE_EXP.
    let mut exp =
        i32::try_from(sa).unwrap_or(i32::MAX / 4) - i32::try_from(sd).unwrap_or(0) - 64 - SCALE_EXP;

    // Reduce q to at most 53 significant bits, moving the discarded places
    // into the exponent.
    let drop = (bq - 53).max(0);
    // `drop` leaves at most 53 significant bits, so the truncation to u64
    // discards nothing.
    let m = u64::try_from(q >> drop).unwrap_or(u64::MAX);
    exp += drop;

    ldexp(m, exp)
}

/// Builds `m · 2^exp` as an `f64`, saturating to zero and infinity outside the
/// representable range.
///
/// Written out rather than taken from `libm` because this crate is `no_std`
/// with zero dependencies. Only the estimate calls it, so an inexactness here
/// costs a correction step and nothing else.
fn ldexp(m: u64, exp: i32) -> f64 {
    if m == 0 {
        return 0.0;
    }
    let mut m = m;
    let mut exp = exp;

    // Normalize to exactly 53 significant bits.
    let bits = 64 - i32::try_from(m.leading_zeros()).unwrap_or(64);
    if bits > 53 {
        let drop = bits - 53;
        m >>= drop;
        exp += drop;
    } else if bits < 53 {
        let gain = 53 - bits;
        m <<= gain;
        exp -= gain;
    }

    // Now m ∈ [2^52, 2^53), so the value is 1.f × 2^(exp + 52).
    let unbiased = exp + 52;
    if unbiased > 1023 {
        return f64::INFINITY;
    }
    if unbiased < -1074 {
        return 0.0;
    }
    if unbiased >= -1022 {
        // Guarded above: unbiased ∈ [−1022, 1023], so the sum is in
        // [1, 2046] and is non-negative.
        let biased = u64::try_from(unbiased + 1023).unwrap_or(0);
        return f64::from_bits((biased << 52) | (m & ((1u64 << 52) - 1)));
    }
    // Subnormal: shift the mantissa down to the fixed 2^−1074 exponent.
    let shift = (-1074 - exp).max(0);
    if shift >= 64 {
        return 0.0;
    }
    f64::from_bits(m >> shift)
}

// ===========================================================================
// The bracket
// ===========================================================================

/// Where an exact value sits relative to its two nearest representable
/// `f64` neighbors.
///
/// Every variant is an **exact statement**. None of them is the value
/// rounded; which neighbor to return is a rounding *policy*, and this crate
/// makes no such decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketPosition {
    /// The value is exactly representable: `lo == hi == value`.
    Exact,
    /// Strictly between `lo` and the midpoint of `lo` and `hi`.
    BelowMidpoint,
    /// Exactly the midpoint of `lo` and `hi`.
    AtMidpoint,
    /// Strictly between the midpoint of `lo` and `hi` and `hi`.
    AboveMidpoint,
}

/// The two representable `f64` values that bracket an exact value, and the
/// value's exact position between them.
///
/// `lo ≤ value ≤ hi`. When the value is representable, `lo == hi` and the
/// position is [`BracketPosition::Exact`]; otherwise `lo` and `hi` are
/// adjacent `f64` values. Round-to-nearest with any tie rule is then a
/// one-line selection on `position`, which is the caller's to make.
///
/// # Range extremes
///
/// A value beyond `f64::MAX` is bracketed by `lo = f64::MAX` and
/// `hi = +∞`, with `position` measured against the IEEE overflow threshold,
/// the midpoint of `MAX` and `2^1024`, so that a caller applying the IEEE
/// tie rule reproduces IEEE overflow behavior exactly. A value below half the
/// smallest subnormal is bracketed by `lo = 0.0` and `hi = 2^−1074`.
///
/// Both extremes **are reachable** from a [`RationalCoord2d`] built out of
/// finite inputs: two nearly parallel segments with coordinates near the top
/// of the `f64` range meet arbitrarily far away, and a tiny numerator over a
/// large denominator falls below the subnormal floor. Neither case is an
/// error; the bracket describes it.
///
/// # Sign
///
/// Negative values are bracketed in the natural order, `lo ≤ hi` with both
/// non-positive. The sign of zero is not carried: a `RationalCoord2d` has no
/// signed zero, so a value of zero brackets as `(+0.0, +0.0, Exact)`.
///
/// [`RationalCoord2d`]: crate::RationalCoord2d
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F64Bracket {
    /// The largest representable value `≤` the exact value.
    pub lo: f64,
    /// The smallest representable value `≥` the exact value. Equals `lo`
    /// when the value is representable.
    pub hi: f64,
    /// The exact position of the value between `lo` and `hi`.
    pub position: BracketPosition,
}

/// The overflow threshold, `(2^54 − 1) · 2^970`: the midpoint of `f64::MAX`
/// and `2^1024`. IEEE 754 rounds to infinity at or above it.
const OVERFLOW_MIDPOINT: (u64, i32) = ((1u64 << 54) - 1, 970);

/// Brackets the magnitude `(a / d) · 2^−SCALE_EXP`.
///
/// `a` must be non-negative and `d` strictly positive.
fn bracket_magnitude(a: &RationalCoordInt, d: &RationalCoordInt) -> F64Bracket {
    if a.is_zero() {
        return F64Bracket {
            lo: 0.0,
            hi: 0.0,
            position: BracketPosition::Exact,
        };
    }
    let mut c = estimate(a, d);
    if !c.is_finite() {
        c = f64::MAX;
    }
    correct_from(c, a, d)
}

/// Places the value relative to a candidate `c` already known to be the
/// nearest representable: the value lies strictly inside `c`'s rounding
/// interval, so it is either `c` itself, just below it, or just above it.
///
/// This is the one comparison the rounding did not need and the bracket
/// does: two midpoint tests establish that `c` is nearest, but not on which
/// side of `c` the value falls, nor whether it is `c` exactly.
fn place_within(c: f64, a: &RationalCoordInt, d: &RationalCoordInt) -> F64Bracket {
    let (mc, ec) = decompose(c);
    match cmp_value_to_dyadic(a, d, mc, ec) {
        Ordering::Equal => F64Bracket {
            lo: c,
            hi: c,
            position: BracketPosition::Exact,
        },
        // Below c but above the lower midpoint: the value is in the upper
        // half of the interval (next_down(c), c).
        Ordering::Less => F64Bracket {
            lo: next_down(c),
            hi: c,
            position: BracketPosition::AboveMidpoint,
        },
        // Above c but below the upper midpoint: the lower half of (c, next_up(c)).
        Ordering::Greater => F64Bracket {
            lo: c,
            hi: next_up(c),
            position: BracketPosition::BelowMidpoint,
        },
    }
}

/// The correction loop, entered from a given candidate.
///
/// Split out from [`bracket_magnitude`] so that a test can choose the
/// starting candidate. [`estimate`] truncates, so in practice it lands on or
/// below the true value and the lower-midpoint branch below is almost never
/// entered. A mutation to that branch survived the whole test suite until
/// this seam existed. Rather than argue the branch is unreachable, which
/// would be an argument about the estimate's rounding direction that no proof
/// in this crate supports, the branch is made reachable and tested.
///
/// The starting candidate affects only the number of iterations. Every return
/// path has passed an exact comparison, so a deliberately bad start yields
/// the same bracket as a good one, which is itself asserted in the tests.
fn correct_from(start: f64, a: &RationalCoordInt, d: &RationalCoordInt) -> F64Bracket {
    let mut c = start;

    for _ in 0..MAX_CORRECTIONS {
        // Lower bound of c's rounding interval. At c == 0 the interval starts
        // at zero, and midpoint(0, 0) is (0, −1075), which compares as zero —
        // correct, since no positive value is below it.
        let below = next_down(c);
        let (lo_m, lo_e) = midpoint(below, c);
        match cmp_value_to_dyadic(a, d, lo_m, lo_e) {
            Ordering::Less => {
                c = below;
                continue;
            }
            Ordering::Equal => {
                return F64Bracket {
                    lo: below,
                    hi: c,
                    position: BracketPosition::AtMidpoint,
                };
            }
            Ordering::Greater => {}
        }

        let above = next_up(c);
        if above.is_infinite() {
            // c is f64::MAX. The upper bound of its interval is the IEEE
            // overflow threshold, not a midpoint between two finite values.
            match cmp_value_to_dyadic(a, d, OVERFLOW_MIDPOINT.0, OVERFLOW_MIDPOINT.1) {
                Ordering::Less => return place_within(c, a, d),
                Ordering::Equal => {
                    return F64Bracket {
                        lo: c,
                        hi: f64::INFINITY,
                        position: BracketPosition::AtMidpoint,
                    };
                }
                Ordering::Greater => {
                    return F64Bracket {
                        lo: c,
                        hi: f64::INFINITY,
                        position: BracketPosition::AboveMidpoint,
                    };
                }
            }
        }
        let (hi_m, hi_e) = midpoint(c, above);
        match cmp_value_to_dyadic(a, d, hi_m, hi_e) {
            Ordering::Greater => c = above,
            Ordering::Equal => {
                return F64Bracket {
                    lo: c,
                    hi: above,
                    position: BracketPosition::AtMidpoint,
                };
            }
            Ordering::Less => return place_within(c, a, d),
        }
    }

    debug_assert!(
        false,
        "bracket correction did not converge within {MAX_CORRECTIONS} steps; \
         the f64 estimate is further off than its error bound allows"
    );
    place_within(c, a, d)
}

/// Brackets the exact rational `(num / den) · 2^−1074` between its two
/// nearest representable `f64` values.
///
/// `den` must be strictly positive, which every [`RationalCoord2d`]
/// guarantees. See [`F64Bracket`] for the contract at the range extremes and
/// at zero.
///
/// [`RationalCoord2d`]: crate::RationalCoord2d
pub(crate) fn rational_to_bracket(num: &RationalCoordInt, den: &RationalCoordInt) -> F64Bracket {
    debug_assert!(den.is_positive(), "den > 0 is a RationalCoord2d invariant");
    let b = bracket_magnitude(&num.wrapping_abs(), den);
    if !num.is_negative() {
        return b;
    }
    // Negation reverses order, so the neighbors swap, and a value in the
    // upper half of a positive interval is in the lower half of the negated
    // one. Exact and AtMidpoint are symmetric and unchanged.
    F64Bracket {
        lo: -b.hi,
        hi: -b.lo,
        position: match b.position {
            BracketPosition::BelowMidpoint => BracketPosition::AboveMidpoint,
            BracketPosition::AboveMidpoint => BracketPosition::BelowMidpoint,
            p @ (BracketPosition::Exact | BracketPosition::AtMidpoint) => p,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BracketPosition, F64Bracket, RationalCoordInt, bracket_magnitude, correct_from, decompose,
        rational_to_bracket,
    };

    /// Builds the exact tie value: the midpoint of `c` and its `f64`
    /// successor, as `(num, den)` in the crate's `2^1074`-scaled space.
    ///
    /// # Why this exists rather than a constructed intersection
    ///
    /// An exact tie cannot be reached through
    /// `RationalCoord2d::from_segment_intersection`: the midpoint of two
    /// adjacent `f64` values needs 54 significant bits, so it is not an `f64`
    /// and cannot be a segment endpoint, and the generators in
    /// `tests/round.rs` were measured not to produce one. A mutation that
    /// broke the tie branch passed every test in that file, which is why the
    /// tie path is tested here, at the unit, where the value can be
    /// constructed exactly.
    ///
    /// With `c = Mc · 2^Ec`, the midpoint is `(2·Mc + 1) · 2^(Ec − 1)`.
    /// Choosing `den = 2` and `num = (2·Mc + 1) · 2^(Ec + 1074)` gives
    /// `(num / den) · 2^−1074 = (2·Mc + 1) · 2^(Ec − 1)` exactly. The shift is
    /// never negative: `Ec ≥ −1074` for every finite `f64`.
    fn tie_above(c: f64) -> (RationalCoordInt, RationalCoordInt) {
        let (mc, ec) = decompose(c);
        let shift = usize::try_from(ec + 1074).expect("Ec ≥ −1074 for finite f64");
        let num = RationalCoordInt::from_u64(2 * mc + 1) << shift;
        (num, RationalCoordInt::from_u64(2))
    }

    fn at_midpoint(lo: f64, hi: f64) -> F64Bracket {
        F64Bracket {
            lo,
            hi,
            position: BracketPosition::AtMidpoint,
        }
    }

    /// Exact ties are reported as `AtMidpoint` with the right neighbors,
    /// from both mantissa parities at each exponent.
    ///
    /// Whichever neighbor the `f64` estimate lands on, both the
    /// lower-midpoint and upper-midpoint branches are exercised across the
    /// set, so a mutation to either fails here.
    #[test]
    fn exact_ties_bracket_as_at_midpoint() {
        let bases = [
            f64::from_bits(1),
            f64::from_bits(3),
            f64::from_bits((1u64 << 52) - 1),
            f64::from_bits(1u64 << 52),
            0.5f64,
            1.0f64,
            3.0f64,
            1_435_363.863_356_009_1f64,
            f64::from_bits(0x7FE0_0000_0000_0000),
        ];
        for base in bases {
            for c in [base, f64::from_bits(base.to_bits() + 1)] {
                let up = f64::from_bits(c.to_bits() + 1);
                let (num, den) = tie_above(c);
                assert_eq!(
                    bracket_magnitude(&num, &den),
                    at_midpoint(c, up),
                    "tie between {c:e} and {up:e} was not bracketed as AtMidpoint"
                );
            }
        }
    }

    /// Values one scaled unit off a tie land in the correct half.
    ///
    /// Pins that the midpoint comparisons are strict where they should be
    /// strict, and that `place_within` reports the correct half: a `<=`
    /// written for a `<` shows up here rather than as a wrong position in
    /// production.
    #[test]
    fn values_just_off_a_tie_report_the_correct_half() {
        for base in [0.5f64, 1.0, 3.0, 1_435_363.863_356_009_1] {
            let up = f64::from_bits(base.to_bits() + 1);
            let (tie, den) = tie_above(base);

            let below = tie - RationalCoordInt::one();
            assert_eq!(
                bracket_magnitude(&below, &den),
                F64Bracket {
                    lo: base,
                    hi: up,
                    position: BracketPosition::BelowMidpoint
                },
                "just below the tie between {base:e} and {up:e}"
            );

            let above = tie + RationalCoordInt::one();
            assert_eq!(
                bracket_magnitude(&above, &den),
                F64Bracket {
                    lo: base,
                    hi: up,
                    position: BracketPosition::AboveMidpoint
                },
                "just above the tie between {base:e} and {up:e}"
            );
        }
    }

    /// The lower-midpoint branch, reached by starting the correction above
    /// the true value.
    ///
    /// [`estimate`](super::estimate) truncates and so lands at or below the
    /// answer, leaving this branch unexercised by every other test in the
    /// crate. Entering [`correct_from`] at the upper neighbor forces it.
    #[test]
    fn the_lower_midpoint_branch_brackets_correctly() {
        for base in [0.5f64, 1.0, 3.0, 1_435_363.863_356_009_1, f64::from_bits(1)] {
            for c in [base, f64::from_bits(base.to_bits() + 1)] {
                let up = f64::from_bits(c.to_bits() + 1);
                let (num, den) = tie_above(c);
                assert_eq!(
                    correct_from(up, &num, &den),
                    at_midpoint(c, up),
                    "lower-branch tie between {c:e} and {up:e}"
                );
            }
        }
    }

    /// A deliberately bad starting candidate yields the same bracket.
    ///
    /// The estimate carries no correctness obligation, every return path has
    /// passed an exact comparison, and this is the assertion that says so.
    #[test]
    fn a_bad_starting_candidate_reaches_the_same_bracket() {
        for base in [0.5f64, 1.0, 1_435_363.863_356_009_1] {
            let up = f64::from_bits(base.to_bits() + 1);
            let (tie, den) = tie_above(base);
            let below = tie - RationalCoordInt::one();

            let truth = bracket_magnitude(&below, &den);
            assert_eq!(truth.lo.to_bits(), base.to_bits(), "baseline moved");

            for start in [base, up, f64::from_bits(base.to_bits() - 1)] {
                assert_eq!(
                    correct_from(start, &below, &den),
                    truth,
                    "starting at {start:e} gave a different bracket"
                );
            }
        }
    }

    /// A representable value brackets as `Exact` with `lo == hi`.
    ///
    /// Everything above depends on `decompose` being exact, and this is the
    /// check that is not a comparison against an oracle.
    #[test]
    fn representable_values_bracket_as_exact() {
        let cases = [
            f64::from_bits(1),
            f64::from_bits((1u64 << 52) - 1),
            f64::from_bits(1u64 << 52),
            0.5f64,
            1.0,
            f64::MAX,
            f64::MIN_POSITIVE,
        ];
        for x in cases {
            let (m, e) = decompose(x);
            let shift = usize::try_from(e + 1074).expect("e ≥ −1074");
            let scaled = RationalCoordInt::from_u64(m) << shift;
            assert_eq!(
                bracket_magnitude(&scaled, &RationalCoordInt::one()),
                F64Bracket {
                    lo: x,
                    hi: x,
                    position: BracketPosition::Exact
                },
                "{x:e} did not bracket as Exact (m = {m}, e = {e})"
            );
        }
    }

    /// Negation reverses the bracket and mirrors the position.
    #[test]
    fn negative_values_bracket_mirrored() {
        let base = 3.0f64;
        let up = f64::from_bits(base.to_bits() + 1);
        let (tie, den) = tie_above(base);
        let just_above = tie + RationalCoordInt::one();

        let pos = rational_to_bracket(&just_above, &den);
        assert_eq!(pos.position, BracketPosition::AboveMidpoint);
        assert_eq!((pos.lo, pos.hi), (base, up));

        let neg = rational_to_bracket(&-just_above, &den);
        assert_eq!(
            neg,
            F64Bracket {
                lo: -up,
                hi: -base,
                position: BracketPosition::BelowMidpoint
            },
            "negating a value in the upper half must land it in the lower half"
        );
    }

    /// Overflow and underflow are described, not errored or saturated silently.
    ///
    /// Both are reachable from finite inputs (see [`F64Bracket`]) so the
    /// contract is tested rather than left to a doc comment.
    #[test]
    fn range_extremes_are_bracketed_per_ieee() {
        // 2^1024 exactly: above the overflow threshold.
        let two_1024 = RationalCoordInt::one() << (1024 + 1074);
        let one = RationalCoordInt::one();
        assert_eq!(
            bracket_magnitude(&two_1024, &one),
            F64Bracket {
                lo: f64::MAX,
                hi: f64::INFINITY,
                position: BracketPosition::AboveMidpoint
            }
        );

        // Exactly the overflow threshold (2^54 − 1) · 2^970: a tie with ∞.
        let threshold = RationalCoordInt::from_u64((1u64 << 54) - 1) << (970 + 1074);
        assert_eq!(
            bracket_magnitude(&threshold, &one),
            F64Bracket {
                lo: f64::MAX,
                hi: f64::INFINITY,
                position: BracketPosition::AtMidpoint
            }
        );

        // Exactly 2^−1075: the tie between 0 and the smallest subnormal.
        // num = 1, den = 2 gives (1/2) · 2^−1074.
        let two = RationalCoordInt::from_u64(2);
        assert_eq!(
            bracket_magnitude(&one, &two),
            F64Bracket {
                lo: 0.0,
                hi: f64::from_bits(1),
                position: BracketPosition::AtMidpoint
            }
        );

        // 2^−1076: strictly below that tie.
        let four = RationalCoordInt::from_u64(4);
        assert_eq!(
            bracket_magnitude(&one, &four),
            F64Bracket {
                lo: 0.0,
                hi: f64::from_bits(1),
                position: BracketPosition::BelowMidpoint
            }
        );
    }
}
