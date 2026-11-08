// src/compare_distance.rs

//! The `compare_distance` predicate: which of two points is nearer to a third.

use crate::exact_int::f64_to_orient_int;
use crate::expansion::{CMP_DIST_ERR_BOUND_A, CMP_DIST_ERR_BOUND_ABS, expansion_sum};
use crate::range::{Prepared, Window, prepare};
use crate::util::{negate, sign_of, two_by_two_product, two_diff};
use crate::{NonFinite, Sign};

/// The stage that answered a call. This crate's own unit tests observe it
/// through [`compare_distance_staged`], and the public function discards it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// `q1 == q2`, certified by comparisons alone.
    ZeroFilter,
    /// The fast filter in native `f64`.
    Filter,
    /// The exact expansion stage, inside the exactness window.
    Expansion,
    /// The exact integer stage.
    Exact,
}

/// Compares the distances from `p` to `q1` and to `q2`.
///
/// Returns the sign of `|p − q1|² − |p − q2|²`: [`Sign::Positive`] if `q1` is
/// farther from `p` than `q2`, [`Sign::Negative`] if it is nearer, and
/// [`Sign::Zero`] if the two are exactly equidistant. The result is exact for
/// every finite input.
///
/// ```
/// use arithmetica::{Sign, compare_distance};
///
/// let p = [0.0, 0.0];
/// assert_eq!(compare_distance(p, [3.0, 0.0], [0.0, 2.0]), Ok(Sign::Positive));
/// assert_eq!(compare_distance(p, [3.0, 4.0], [5.0, 0.0]), Ok(Sign::Zero));
/// ```
///
/// # Mathematical definition
///
/// With `u = q1 − p` and `v = q2 − p`, the predicate is the sign of the
/// degree-2 polynomial `ux² + uy² − vx² − vy²`.
///
/// # Four stages
///
/// - **The zero filter**. `q1 == q2` is certified `Zero` by comparisons. Other
///   exact ties cannot be certified without arithmetic and are left to the
///   stages below.
/// - **Stage 1, the fast filter**. The polynomial is evaluated in native `f64`,
///   as
///   `fl(fl(fl(ux²) + fl(uy²)) − fl(fl(vx²) + fl(vy²)))`, and accepted when
///   its magnitude exceeds `A · fl(s1 + s2) + K`. The bound is derived for
///   this expression tree with the underflow error of each square included,
///   so the certificate holds for every finite input (the derivation is on
///   the constant in `expansion.rs`).
/// - **Stage 2, the exact expansion stage**. Inside the exactness window of
///   `orient2d`
///   (the predicate is degree 2, as `orient2d` is), each difference is
///   formed exactly as a two-term expansion, each square as the exact
///   product of two of them, and the four squares are summed exactly. The
///   sign of the largest component is the sign of the polynomial. These are
///   the operations of `orient2d`'s exact expansion stage (exact products of
///   exact differences, summed), so the window's argument that every
///   operation is exact in binary64 applies unchanged. Input that lies
///   outside the window is first moved inside by an exact power-of-two
///   rescale, which multiplies the polynomial by a positive power of two.
///   No error bound is consulted at this stage, so it needs none of the
///   window's margin for bound products.
/// - **Stage 4, the exact integer stage**. `WideInt<66>` arithmetic on the
///   caller's
///   coordinates answers input spread over more bits than the window is
///   wide, and it confirms every `Zero` the expansion stage reports.
///
/// # Non-finite inputs
///
/// A NaN or infinite coordinate returns [`Err(NonFinite)`](NonFinite) before
/// any stage runs. For finite inputs the result is always `Ok`.
///
/// # References
///
/// Shewchuk (1997), §§2–4, for the expansion arithmetic and the adaptive
/// pattern; Bartels, Fisikopoulos & Weiser (2023) for the underflow-protected
/// form of the filter.
///
/// # Errors
///
/// Returns [`NonFinite`] when any input coordinate is not finite.
#[must_use = "the comparison result is the return value; discarding it is almost certainly a bug"]
pub fn compare_distance(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Result<Sign, NonFinite> {
    compare_distance_staged(p, q1, q2).map(|(sign, _)| sign)
}

/// [`compare_distance`], reporting the stage that answered.
pub(crate) fn compare_distance_staged(
    p: [f64; 2],
    q1: [f64; 2],
    q2: [f64; 2],
) -> Result<(Sign, Stage), NonFinite> {
    if !p.iter().chain(&q1).chain(&q2).all(|v| v.is_finite()) {
        return Err(NonFinite::new());
    }
    if is_certainly_zero(q1, q2) {
        return Ok((Sign::Zero, Stage::ZeroFilter));
    }
    if let Some(sign) = filter(p, q1, q2) {
        return Ok((sign, Stage::Filter));
    }
    let raw = (p, q1, q2);
    let scaled = |a: [f64; 2], s: f64| [a[0] * s, a[1] * s];
    let (p, q1, q2) = match prepare(&[p[0], p[1], q1[0], q1[1], q2[0], q2[1]], Window::ORIENT2D) {
        Prepared::AsIs => (p, q1, q2),
        Prepared::Scaled(s) => (scaled(p, s), scaled(q1, s), scaled(q2, s)),
        Prepared::OutOfWindow => return Ok((exact_integer(raw.0, raw.1, raw.2), Stage::Exact)),
    };
    // A `Zero` from the expansion is not a certificate, so the exact integer
    // stage decides it.
    match in_window(p, q1, q2) {
        Sign::Zero => Ok((exact_integer(raw.0, raw.1, raw.2), Stage::Exact)),
        sign => Ok((sign, Stage::Expansion)),
    }
}

/// `q1 == q2` makes the two distances the same real number.
///
/// `==` is exact, and `-0.0 == 0.0` is true, which is correct because they
/// are the same real number. `clippy::float_cmp` is allowed because the
/// comparison asks exactly that question. A tolerance would turn a
/// certificate into a guess.
#[inline]
#[allow(clippy::float_cmp)]
fn is_certainly_zero(q1: [f64; 2], q2: [f64; 2]) -> bool {
    q1[0] == q2[0] && q1[1] == q2[1]
}

/// The fast filter, valid for every finite input (derivation on
/// [`CMP_DIST_ERR_BOUND_A`]).
///
/// Every term of `mag` is non-negative, so an overflow anywhere makes `mag`,
/// and with it the bound, infinite, and `det` is then infinite or NaN. The
/// strict comparison fails either way, so overflowing input is passed on
/// rather than certified.
#[inline]
fn filter(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Option<Sign> {
    let ux = q1[0] - p[0];
    let uy = q1[1] - p[1];
    let vx = q2[0] - p[0];
    let vy = q2[1] - p[1];
    let s1 = ux * ux + uy * uy;
    let s2 = vx * vx + vy * vy;
    let det = s1 - s2;
    let mag = s1 + s2;
    let bound = CMP_DIST_ERR_BOUND_A * mag + CMP_DIST_ERR_BOUND_ABS;
    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// The exact expansion stage, on coordinates the caller has placed inside
/// `Window::ORIENT2D`.
///
/// `ux = q1.x − p.x` and the other three differences are exact two-term
/// expansions. Each square is their exact product, of at most 8 components,
/// `S1 = ux² + uy²` and `S2 = vx² + vy²` have at most 16 each, and
/// `S1 − S2` at most 32. Every component is exact inside the window, so the
/// expansion sums to the polynomial, and its largest-magnitude component,
/// the last non-zero one of a non-overlapping expansion, carries its sign.
/// An empty expansion is a zero the caller confirms.
fn in_window(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Sign {
    let ux = two_diff(q1[0], p[0]);
    let uy = two_diff(q1[1], p[1]);
    let vx = two_diff(q2[0], p[0]);
    let vy = two_diff(q2[1], p[1]);

    let (ux2, ux2_len) = two_by_two_product(ux, ux);
    let (uy2, uy2_len) = two_by_two_product(uy, uy);
    let (mut vx2, vx2_len) = two_by_two_product(vx, vx);
    let (mut vy2, vy2_len) = two_by_two_product(vy, vy);
    negate(&mut vx2[..vx2_len]);
    negate(&mut vy2[..vy2_len]);

    let mut s1 = [0.0_f64; 16];
    let s1_len = expansion_sum(&ux2[..ux2_len], &uy2[..uy2_len], &mut s1);
    let mut s2 = [0.0_f64; 16];
    let s2_len = expansion_sum(&vx2[..vx2_len], &vy2[..vy2_len], &mut s2);
    let mut q = [0.0_f64; 32];
    let q_len = expansion_sum(&s1[..s1_len], &s2[..s2_len], &mut q);

    if q_len == 0 {
        Sign::Zero
    } else {
        sign_of(q[q_len - 1])
    }
}

/// The exact integer stage: `WideInt<66>`, 4,224 bits.
///
/// Each coordinate is converted to an integer scaled by `2^1074`, which is
/// exact for every finite `f64`. The polynomial is then multiplied by
/// `2^2148`, a positive factor, leaving its sign unchanged.
///
/// The bit-width argument is `orient2d`'s with one more term. A scaled
/// coordinate is below `2^2098`, a difference below `2^2099`, a square below
/// `2^4198`, a sum of two squares below `2^4199`, and the difference of two
/// such sums below `2^4200`, which fits in 4,224 bits with a sign (`4201`
/// bits are asserted).
///
/// The stage operates on the caller's coordinates, before any rescale, so it
/// decides exactly the caller's question.
fn exact_integer(p: [f64; 2], q1: [f64; 2], q2: [f64; 2]) -> Sign {
    let px = f64_to_orient_int(p[0]);
    let py = f64_to_orient_int(p[1]);
    let ux = f64_to_orient_int(q1[0]) - px;
    let uy = f64_to_orient_int(q1[1]) - py;
    let vx = f64_to_orient_int(q2[0]) - px;
    let vy = f64_to_orient_int(q2[1]) - py;
    let q = (ux * ux + uy * uy) - (vx * vx + vy * vy);
    debug_assert!(
        q.fits_in_signed_bits(4201),
        "compare_distance Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );
    if q.is_positive() {
        Sign::Positive
    } else if q.is_negative() {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
// test-lint-posture: see the lint table in Cargo.toml
#[allow(clippy::float_cmp, clippy::unreadable_literal)]
mod tests {
    use super::*;

    fn b(bits: u64) -> f64 {
        f64::from_bits(bits)
    }

    /// A fixed case: three points, the exact sign, and the stage it was
    /// built to be answered at. The integration tests check the signs
    /// against the oracles, and these unit tests check the stages.
    type Case = ([f64; 2], [f64; 2], [f64; 2], Sign, Stage);

    pub(crate) fn fixed_cases() -> [Case; 7] {
        let h = 2.0_f64.powi(500);
        let d = 2.0_f64.powi(-1000);
        let s = 2.0_f64.powi(-600);
        [
            // The fast filter: well separated.
            (
                [0.0, 0.0],
                [3.0, 0.0],
                [1.0, 0.0],
                Sign::Positive,
                Stage::Filter,
            ),
            // The zero filter: the same point twice.
            (
                [0.1, 0.2],
                [0.7, 0.3],
                [0.7, 0.3],
                Sign::Zero,
                Stage::ZeroFilter,
            ),
            // The expansion stage: q1 = (3, 2) and q2 = (2, 3) mirrored across
            // the diagonal, p moved off it by t = 2^-52. Exactly,
            // Q = (4 + (1 − t)²) − (1 + (2 − t)²) = 2t = 2^-51. In f64 the
            // filter's det is 2^-50, below its bound of about 4ε·10.
            (
                [1.0, 1.0 + f64::EPSILON],
                [3.0, 2.0],
                [2.0, 3.0],
                Sign::Positive,
                Stage::Expansion,
            ),
            // An exact tie with inexact differences, confirmed by the exact
            // integer stage. p = (c, c), q1 = (a, b), q2 = (b, a) are exactly
            // equidistant for any floats, but none of the differences is
            // representable.
            ([0.1, 0.1], [0.7, 0.3], [0.3, 0.7], Sign::Zero, Stage::Exact),
            // Wider than the window: 2^-1000 and 2^500 in one call. The exact
            // value is (h − d)² − (h + d)² = −4hd = −2^-498, invisible in f64.
            ([d, 0.0], [h, 0.0], [-h, 0.0], Sign::Negative, Stage::Exact),
            // Wider than the window, and decided by a square that
            // underflows: Q = d'² with d' = 2^-600. In f64 both sums are h²,
            // and an expansion run on these coordinates as given would lose
            // d'² and report a zero that is not one.
            ([0.0, 0.0], [h, s], [h, 0.0], Sign::Positive, Stage::Exact),
            // Overflowing differences: answered exactly, never `Zero`.
            (
                [-f64::MAX, 0.0],
                [f64::MAX, b(0x0010_0000_0000_0000)],
                [f64::MAX, 0.0],
                Sign::Positive,
                Stage::Exact,
            ),
        ]
    }

    #[test]
    fn each_fixed_case_is_answered_at_its_stage() {
        for (i, (p, q1, q2, sign, stage)) in fixed_cases().into_iter().enumerate() {
            assert_eq!(
                compare_distance_staged(p, q1, q2),
                Ok((sign, stage)),
                "fixed case {i}"
            );
        }
    }

    /// Swapping `q1` and `q2` negates the result, on every fixed case.
    #[test]
    fn swapping_the_two_points_negates_the_sign() {
        for (p, q1, q2, sign, _) in fixed_cases() {
            assert_eq!(compare_distance(p, q2, q1), Ok(-sign));
        }
    }

    /// The sign convention: positive means `q1` is the farther point.
    #[test]
    fn positive_means_the_first_point_is_farther() {
        assert_eq!(
            compare_distance([0.0, 0.0], [10.0, 0.0], [1.0, 0.0]),
            Ok(Sign::Positive)
        );
        assert_eq!(
            compare_distance([0.0, 0.0], [1.0, 0.0], [10.0, 0.0]),
            Ok(Sign::Negative)
        );
    }

    /// Input in the window's range but small enough that every square
    /// underflows. The filter's absolute term stops it from certifying, the
    /// rescale brings the input into the window, and the expansion answers.
    #[test]
    fn underflowing_squares_are_answered_by_the_expansion() {
        let s = 2.0_f64.powi(-600);
        assert_eq!(
            compare_distance_staged([0.0, 0.0], [2.0 * s, 0.0], [s, s]),
            Ok((Sign::Positive, Stage::Expansion))
        );
    }

    /// A tiny deterministic generator for the stage sweep (SplitMix64).
    struct Mix(u64);
    impl Mix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A float with a random sign and significand, leading bit at
        /// `2^e`, `e` in `-1022..=1000`.
        fn float(&mut self, e: i64) -> f64 {
            let e = e.clamp(-1022, 1000);
            let biased = u64::try_from(e + 1023).unwrap_or(1);
            f64::from_bits(((self.next() & 1) << 63) | (biased << 52) | (self.next() >> 12))
        }
    }

    /// Near-degenerate input of the kind the integration suite draws, across
    /// the exponent range, reaches every stage in the sequence. The property
    /// suites, whose results the oracles check, therefore exercise each one.
    #[test]
    fn a_near_degenerate_sweep_reaches_every_stage() {
        let mut rng = Mix(0x2026_1024_57A6_E000);
        let mut seen = [0_u32; 4];
        for i in 0..20_000_u32 {
            let e = i64::try_from(rng.next() % 2000).unwrap_or(0) - 1000;
            let (x, y, c) = (rng.float(e), rng.float(e - 1), rng.float(e - 2));
            let mut config = if i % 5 == 0 {
                [y, c, x, c, x, c]
            } else if i % 2 == 0 {
                [y, y, x, c, c, x]
            } else {
                [-y, y, x, c, -c, -x]
            };
            let index = (rng.next() % 6) as usize;
            let k = i64::try_from(rng.next() % 7).unwrap_or(0) - 3;
            // Moves the magnitude by `k` units in the last place.
            let moved = f64::from_bits(config[index].to_bits().wrapping_add_signed(k));
            if moved.is_finite() && k != 0 {
                config[index] = moved;
            }
            let (p, q, r) = (
                [config[0], config[1]],
                [config[2], config[3]],
                [config[4], config[5]],
            );
            if let Ok((_, stage)) = compare_distance_staged(p, q, r) {
                seen[stage as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "stage counts {seen:?} (zero filter, filter, expansion, exact)"
        );
    }
}
