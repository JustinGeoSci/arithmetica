// src/orient2d.rs

//! The `orient2d` geometric predicate.
//!
//! This module implements Shewchuk's three-stage adaptive `orient2d` predicate,
//! which determines the signed orientation (winding order) of three points in
//! the plane.

use crate::counters::{self, Outcome, Predicate};
use crate::expansion::{
    CCW_ERR_BOUND_A, CCW_ERR_BOUND_B, CCW_ERR_BOUND_C, RESULT_ERR_BOUND, expansion_sum, two_product,
};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{sign_of, two_diff_tail, two_two_diff};
use crate::{NonFinite, Sign};

#[cfg(test)]
use core::sync::atomic::{AtomicUsize, Ordering};

/// Counts entries to Stage 3. Present only in this crate's own unit-test
/// builds, so the unit tests can check that near-degenerate input reaches the
/// exact expansion; the public `stage-counters` feature is the supported way
/// to observe stages from outside.
#[cfg(test)]
pub(crate) static ORIENT2D_EXACT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Certifies `Sign::Zero` for the two kinds of degenerate triple that can be
/// recognized exactly, without any arithmetic beyond comparisons.
///
/// Returns `true` only when the exact determinant is zero.
///
/// # Which triples it catches
///
/// - **A repeated point.** Two equal points make two rows of the orientation
///   determinant equal, so the determinant is zero whatever the third point
///   is. The second clause catches `pa == pb`; the first catches `pa == pc`
///   and `pb == pc`, whose differences against `pc` are zero.
/// - **An axis-aligned triple.** If all three points share an x coordinate,
///   both x differences are zero; if they share a y coordinate, both y
///   differences are. Either way each of the two products in the determinant
///   has a factor that is exactly zero.
///
/// # Why it is exact
///
/// With gradual underflow, `fl(x - y) == 0.0` if and only if `x == y`: every
/// finite `f64` is a multiple of `2^-1074`, so a non-zero difference has
/// magnitude at least `2^-1074`, which is representable, and rounding is
/// monotone. (Rust never enables flush-to-zero, under which this would fail.)
/// An overflowing difference is infinite, not zero, so it cannot trigger the
/// filter either. So a zero difference here means the two coordinates are
/// equal, and the corresponding product is zero as a real number and not only
/// as a floating-point value.
///
/// Comparing coordinates with `==` is likewise exact. `-0.0 == 0.0` is true,
/// which is correct: they are the same real number.
///
/// # Why the form is symmetric
///
/// The two clauses together catch the same triples whatever order the three
/// arguments arrive in, so the filter's answer does not depend on the
/// caller's argument order or on the cyclic rotation below. The first clause
/// alone looks only at differences against `pc`, so it cannot see `pa == pb`;
/// on real workloads, where degenerate triples arrive in every order, that
/// would miss about half of them.
///
/// # Why it is worth having
///
/// Without it every exactly degenerate call runs the whole adaptive ladder
/// and then the exact integer stage, roughly two orders of magnitude slower
/// than the fast path. Exact degeneracy is common in real geometry, and most
/// of it is a repeated vertex or an axis-aligned edge, which this filter
/// certifies by comparisons alone. What it leaves is collinearity in a
/// general direction, which no comparison can certify.
///
/// Reference: Bartels, Fisikopoulos & Weiser 2023, Definition 6 (zero-filter)
/// and Example 7, which derives this condition for the 2D orientation
/// predicate.
///
/// `clippy::float_cmp` is allowed here, and only here, because the exactness
/// argument above is the whole point of the filter: these comparisons ask
/// whether two values are the same real number, which `==` answers exactly,
/// and a tolerance would turn an exact certificate into a guess.
#[inline]
#[allow(clippy::float_cmp)]
fn orient2d_is_certainly_zero(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> bool {
    let acx = pa[0] - pc[0];
    let acy = pa[1] - pc[1];
    let bcx = pb[0] - pc[0];
    let bcy = pb[1] - pc[1];
    ((acx == 0.0 || bcy == 0.0) && (acy == 0.0 || bcx == 0.0)) || (pa[0] == pb[0] && pa[1] == pb[1])
}

/// The fast filter, in the underflow-protected form of Bartels et al. 2023,
/// Theorem 1(2).
///
/// Returns `Some(sign)` when the sign of the floating-point determinant is
/// certified to be the sign of the exact determinant, and `None` when it is
/// not. **The certificate holds for every finite input**, with no assumption
/// that the evaluation is free of underflow or overflow.
///
/// # Why Shewchuk's form is not enough
///
/// Shewchuk's filter compares `|det|` with `A · (|detleft| + |detright|)`.
/// Its derivation gives every operation a small *relative* error, which is
/// what rounding in the normal range does. A product that lands among the
/// subnormals instead carries an *absolute* error of up to `2^-1075`, which
/// that bound does not cover; and when the two products are small enough, the
/// bound itself underflows to zero, at which point the comparison accepts any
/// non-zero `det`, including one whose sign is wrong. Bartels et al. 2023
/// give this as their Example 1, and the same failure is reachable in all
/// four predicates.
///
/// # The remedy
///
/// Add `U_N` after every product in the magnitude, and `U_S` to the final
/// bound; compare with a strict `>`. Each `U_N` dominates the absolute error
/// of the product it follows, and the `U_S` makes the bound non-zero no
/// matter how small the input, so a zero determinant can never be certified.
/// The cost is three additions and one comparison.
///
/// The magnitude is summed in the association order of the determinant, as
/// the paper's rules require: the error bound of a sum is derived from the
/// bounds of the two operands it is formed from.
///
/// # Overflow
///
/// If any intermediate overflows, the magnitude overflows with it: every term
/// of the magnitude is non-negative, so no cancellation can bring it back
/// into range. The bound is then infinite and both comparisons fail, so an
/// overflowing input is passed on rather than certified. (`det` itself cannot
/// be finite while the magnitude is infinite, since the magnitude dominates
/// it term for term.)
///
/// # Constants
///
/// Shewchuk's `CCW_ERR_BOUND_A`, `(3 + 16ε)ε`, satisfies the paper's
/// requirement for this expression tree, which is anything above
/// `(3 + 12ε)ε`. A constant as small as `(3 + 8ε)ε` would not.
#[inline]
fn orient2d_filter_raw(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Option<Sign> {
    let detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    let detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    let det = detleft - detright;

    let magnitude = (detleft.abs() + U_N) + (detright.abs() + U_N);
    let bound = CCW_ERR_BOUND_A * magnitude + U_S;

    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// Tests the orientation of the triangle formed by `pa`, `pb`, and `pc`.
///
/// Returns [`Sign::Positive`] if the triangle is oriented counterclockwise,
/// [`Sign::Negative`] if clockwise, and [`Sign::Zero`] if the three points
/// are collinear.
///
/// # Mathematical definition
///
/// The predicate evaluates the sign of the 3×3 determinant:
///
/// ```text
/// | pa.x  pa.y  1 |
/// | pb.x  pb.y  1 |
/// | pc.x  pc.y  1 |
/// ```
///
/// which equals `(pa.x − pc.x) · (pb.y − pc.y) − (pa.y − pc.y) · (pb.x − pc.x)`.
///
/// The sign encodes the winding order of the triangle: a positive value means
/// the triangle `pa → pb → pc` winds counterclockwise as viewed in the standard
/// orientation (y-axis up, x-axis right). This is the fundamental question for
/// point-in-triangle tests, convex-hull construction, and polygon orientation
/// checks: all of these reduce to knowing which side of a directed line a point
/// lies on.
///
/// # Robustness: four stages
///
/// 1. **The zero filter**, `orient2d_is_certainly_zero`: a repeated point
///    or an axis-aligned triple is certified `Zero` by comparisons alone.
/// 2. **The fast filter**, `orient2d_filter_raw`: the determinant in native
///    `f64`, with the underflow-protected error bound of Bartels et al. 2023.
///    Valid for every finite input, so a certificate here is final.
/// 3. **The adaptive expansion stages**, `orient2d_adapt`: exact products,
///    then the full exact expansion, guarded by Shewchuk's Stage B and C
///    bounds. These run only on input inside the exactness window of
///    `crate::range`, where binary64 arithmetic provably behaves as
///    Shewchuk's unbounded-exponent model does, so his bounds hold.
/// 4. **The exact integer stage**, `orient2d_exact_integer`: fixed-width
///    integer arithmetic on the original coordinates. It answers every input
///    the earlier stages leave undecided, and every input that cannot be
///    brought into the window at all.
///
/// Each stage either certifies a sign or passes the input on; no stage
/// returns an uncertified answer. Stage 4 costs about two orders of magnitude
/// more than the fast filter, and input that is neither exactly degenerate
/// nor spread over more bits than the window never reaches it.
///
/// # Non-finite inputs
///
/// If any input coordinate is NaN or infinite, the predicate returns
/// [`Err(NonFinite)`](NonFinite): a non-finite coordinate has no geometric
/// meaning, so there is no correct [`Sign`] to return. For *finite* inputs the
/// function always returns `Ok` with the exact sign.
///
/// # References
///
/// Shewchuk (1997), `orient2d` and `orient2dadapt`, predicates.c.
///
/// Bartels, Fisikopoulos & Weiser (2023), Theorem 1(2) and Definition 6.
///
/// # Errors
///
/// Returns [`NonFinite`] when any input coordinate is not finite. This is
/// the only error case: a NaN or infinite coordinate has no geometric
/// meaning, and it is rejected before the adaptive stages so that it can
/// never masquerade as a degenerate [`Sign::Zero`] result. A non-finite
/// *intermediate* is not an error; it is a recoverable artifact of finite
/// input, resolved by the exact integer stage.
#[must_use = "the orientation result is the return value; discarding it is almost certainly a bug"]
pub fn orient2d(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Result<Sign, NonFinite> {
    // Strict input-domain check: reject non-finite coordinates up front.
    // Every stage below may then assume finite input, so a non-finite
    // *intermediate* is a recoverable artifact of a finite input, never a
    // reason to return Zero.
    if !pa.iter().chain(&pb).chain(&pc).all(|v| v.is_finite()) {
        counters::record(Predicate::Orient2d, Outcome::NonFinite);
        return Err(NonFinite::new());
    }

    // The zero filter, before the rotation below.
    //
    // The filter is symmetric in its three arguments, so the rotation cannot
    // change what it catches; running it first also skips the rotation's
    // magnitude comparisons on the degenerate path. It must in any case run
    // before the fast filter, whose strict `>` can never accept a zero
    // determinant: an exact zero reaching that comparison would fall through
    // to the slow stages.
    if orient2d_is_certainly_zero(pa, pb, pc) {
        counters::record(Predicate::Orient2d, Outcome::Fast);
        return Ok(Sign::Zero);
    }

    // Keep the original coordinates for the exact integer stage.
    let raw = (pa, pb, pc);

    // Cyclic rotation: place the smallest-magnitude point last as the
    // reference.
    //
    // orient2d computes (pa − pc) × (pb − pc), using pc as the reference
    // point. When pc has much larger magnitude than pa and pb, both
    // differences are dominated by pc, and their cross product can overflow
    // f64 (degree-2 products of ~2^500+ values exceed 2^1023).
    //
    // A cyclic rotation (pa, pb, pc) → (pb, pc, pa) preserves the sign of the
    // 3×3 determinant (even permutations). By placing the smallest-magnitude
    // point last, the two differences become (large − small) ≈ large and
    // (medium − small) ≈ medium, so their cross product stays representable.
    //
    // The window check below does not depend on this for correctness: an
    // input whose products would overflow is either rescaled down or sent to
    // the exact stage. The rotation is a performance heuristic, keeping
    // mixed-magnitude input on the cheaper stages.
    //
    // We measure magnitude as max(|x|, |y|), a cheap proxy for the L∞ norm.
    let mag_a = pa[0].abs().max(pa[1].abs());
    let mag_b = pb[0].abs().max(pb[1].abs());
    let mag_c = pc[0].abs().max(pc[1].abs());
    let (pa, pb, pc) = if mag_a <= mag_b && mag_a <= mag_c {
        // pa is smallest — rotate (pa,pb,pc) → (pb,pc,pa): even permutation.
        (pb, pc, pa)
    } else if mag_b <= mag_c {
        // pb is smallest — rotate (pa,pb,pc) → (pc,pa,pb): even permutation.
        (pc, pa, pb)
    } else {
        // pc is already smallest — no rotation needed.
        (pa, pb, pc)
    };

    // The fast filter on the coordinates as given. Valid for every finite
    // input, so no guard on underflow or overflow is needed around it.
    if let Some(sign) = orient2d_filter_raw(pa, pb, pc) {
        counters::record(Predicate::Orient2d, Outcome::Fast);
        return Ok(sign);
    }

    // The expansion stages are exact only inside the window; see
    // `crate::range` for what that means and why.
    let (pa, pb, pc) = match prepare(
        &[pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]],
        Window::ORIENT2D,
    ) {
        Prepared::AsIs => (pa, pb, pc),
        Prepared::Scaled(s) => (
            [pa[0] * s, pa[1] * s],
            [pb[0] * s, pb[1] * s],
            [pc[0] * s, pc[1] * s],
        ),
        // Wider than the window: no single factor brings it inside.
        Prepared::OutOfWindow => return Ok(orient2d_exact_integer(raw.0, raw.1, raw.2)),
    };

    // Stages 2 and 3. A `Zero` from them is not a certificate: it means the
    // expansion came out zero, which the exact stage has to confirm.
    let result = orient2d_in_window(pa, pb, pc);
    Ok(if result == Sign::Zero {
        orient2d_exact_integer(raw.0, raw.1, raw.2)
    } else {
        counters::record(Predicate::Orient2d, Outcome::Fast);
        result
    })
}

/// Shewchuk's Stage A on in-window coordinates, then Stages 2 and 3.
///
/// The caller guarantees that the coordinates are inside
/// [`Window::ORIENT2D`], as given or after an exact rescale. Inside the
/// window no operation of these stages can underflow inexactly or overflow,
/// so Shewchuk's analysis applies to the binary64 execution and his
/// opposite-sign early returns and error bounds are valid as written. Outside
/// it they are not: an inexact underflow inside an expansion would make it
/// sum to the wrong value, with nothing to show for it.
///
/// This Stage A is Shewchuk's rather than the Bartels form: the raw filter
/// has already run, and inside the window the two agree about which inputs
/// they certify, so the cheaper one is used. Its purpose here is to catch the
/// input that the raw filter could not certify only because the rescale had
/// not happened yet.
///
/// A `Sign::Zero` return means no stage could certify a sign; the caller
/// resolves it exactly.
fn orient2d_in_window(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    let detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    let detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    let det = detleft - detright;

    // `detsum` bounds |detleft| + |detright| without cancellation. When the
    // two products have opposite signs, or one is zero, the determinant is
    // their sum of magnitudes and no bound is needed.
    let detsum = if detleft > 0.0 {
        if detright <= 0.0 {
            return sign_of(det);
        }
        detleft + detright
    } else if detleft < 0.0 {
        if detright >= 0.0 {
            return sign_of(det);
        }
        -detleft - detright
    } else {
        return sign_of(det);
    };

    let error_bound = CCW_ERR_BOUND_A * detsum;
    if det >= error_bound || -det >= error_bound {
        return sign_of(det);
    }

    orient2d_adapt(pa, pb, pc, detsum)
}

/// Stages 2 and 3 of the adaptive `orient2d` predicate.
///
/// Ports `orient2dadapt` from Shewchuk's `predicates.c` (1997).
///
/// ## Expansion sizes (from Shewchuk's analysis)
///
/// | Variable | Max components | Meaning |
/// |---|---|---|
/// | `B`  | 4  | Exact two-product expansion of the main determinant terms |
/// | `u`  | 4  | Correction term (each is a `two_two_diff` result) |
/// | `C1` | 8  | `B + u1` |
/// | `C2` | 12 | `C1 + u2` |
/// | `D`  | 16 | `C2 + u3` — the full exact expansion |
///
/// ## Stage 2
///
/// Uses `two_product` to compute `B = acx*bcy - acy*bcx` without rounding
/// error in the multiplications. The estimate of B is the sum of all four
/// expansion components (Shewchuk's `estimate()` function), NOT just `B[3]`.
/// Using only `B[3]` fails when `two_two_diff` produces an expansion like
/// `[0, -8.665e178, 0, 0]` where `B[3]=0` but the true B is large and
/// non-zero. Tests the sum against `CCW_ERR_BOUND_B * detsum`. If the bound
/// is satisfied, the sign of the sum is returned without entering Stage 3.
///
/// ## Stage 3 (exact path)
///
/// Fires only when Stage 2 is also insufficient. Adds three correction
/// expansions to `B`, one for each pair of subtraction-error tails:
///
/// - `u1 = acxtail*bcy − acytail*bcx`
/// - `u2 = acx*bcytail − acy*bcxtail`
/// - `u3 = acxtail*bcytail − acytail*bcxtail`
///
/// Each correction is computed with `two_product` + `two_two_diff` and merged
/// with `expansion_sum`. The sign of the last non-zero component of the final
/// 16-term expansion `D` is the sign of the exact determinant. The
/// non-overlapping property guarantees this component determines the sign.
///
/// The `#[cfg(test)] ORIENT2D_EXACT_COUNT` atomic counter is incremented at
/// the entry to Stage 3 so tests can verify the exact path fires on
/// near-degenerate input.
///
/// ## Difference from predicates.c
///
/// `fast_expansion_sum_zeroelim` (C) is `expansion_sum` here throughout.
/// Both produce the exact sum of two expansions. The fast variant
/// (Shewchuk 1997, Theorem 13) requires its inputs to be *strongly*
/// nonoverlapping and relies on round-to-even; `expansion_sum` (Theorem 12)
/// needs only nonoverlapping inputs, the weaker property and the one the
/// other expansion operations here produce, at a small cost in speed.
fn orient2d_adapt(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], detsum: f64) -> Sign {
    let acx = pa[0] - pc[0];
    let bcx = pb[0] - pc[0];
    let acy = pa[1] - pc[1];
    let bcy = pb[1] - pc[1];

    // ===========================================================================
    // Stage 2: Exact two-product expansion of the main determinant.
    // B = (acx * bcy) - (acy * bcx)  as an exact 4-component expansion.
    // ===========================================================================

    let (detleft1, detleft0) = two_product(acx, bcy);
    let (detright1, detright0) = two_product(acy, bcx);
    let b = two_two_diff(detleft1, detleft0, detright1, detright0);

    // `estimate` in Shewchuk's predicates.c sums all expansion components, not
    // just the last one. For a non-overlapping expansion, b[3] is the largest
    // component but can be zero when cancellation produces a result like
    // [0, -8.665e178, 0, 0]. Summing all four components gives the correct
    // approximation even in that case.
    let det = b[0] + b[1] + b[2] + b[3];
    let error_bound = CCW_ERR_BOUND_B * detsum;
    if det >= error_bound || -det >= error_bound {
        return sign_of(det);
    }

    // ===========================================================================
    // Compute subtraction error tails for the coordinate differences.
    // These are the rounding errors from fl(pa[i] - pc[i]) etc.
    // ===========================================================================

    let acxtail = two_diff_tail(pa[0], pc[0], acx);
    let bcxtail = two_diff_tail(pb[0], pc[0], bcx);
    let acytail = two_diff_tail(pa[1], pc[1], acy);
    let bcytail = two_diff_tail(pb[1], pc[1], bcy);

    // If all four tails are zero, B is already the exact determinant.
    if acxtail == 0.0 && bcxtail == 0.0 && acytail == 0.0 && bcytail == 0.0 {
        return sign_of(det);
    }

    // Refine the estimate using the tails (still in native f64; this is the
    // intermediate re-test from predicates.c, not yet Stage 3).
    // errbound uses the OLD det; det is updated after.
    let error_bound = CCW_ERR_BOUND_C * detsum + RESULT_ERR_BOUND * det.abs();
    let det = det + ((acx * bcytail + bcy * acxtail) - (acy * bcxtail + bcx * acytail));
    if det >= error_bound || -det >= error_bound {
        return sign_of(det);
    }

    // ===========================================================================
    // Stage 3: Full exact expansion. Counter incremented here only.
    // ===========================================================================

    #[cfg(test)]
    ORIENT2D_EXACT_COUNT.fetch_add(1, Ordering::Relaxed);

    // Buffers: sizes from Shewchuk's proven bounds.
    let mut c1_buf = [0.0_f64; 8]; // B (4) + u1 (4) ≤ 8
    let mut c2_buf = [0.0_f64; 12]; // C1 (8) + u2 (4) ≤ 12
    let mut d_buf = [0.0_f64; 16]; // C2 (12) + u3 (4) ≤ 16

    // u1 = acxtail*bcy − acytail*bcx
    let (s1, s0) = two_product(acxtail, bcy);
    let (t1, t0) = two_product(acytail, bcx);
    let u1 = two_two_diff(s1, s0, t1, t0);
    let c1_len = expansion_sum(&b, &u1, &mut c1_buf);

    // u2 = acx*bcytail − acy*bcxtail
    let (s1, s0) = two_product(acx, bcytail);
    let (t1, t0) = two_product(acy, bcxtail);
    let u2 = two_two_diff(s1, s0, t1, t0);
    let c2_len = expansion_sum(&c1_buf[..c1_len], &u2, &mut c2_buf);

    // u3 = acxtail*bcytail − acytail*bcxtail
    let (s1, s0) = two_product(acxtail, bcytail);
    let (t1, t0) = two_product(acytail, bcxtail);
    let u3 = two_two_diff(s1, s0, t1, t0);
    let d_len = expansion_sum(&c2_buf[..c2_len], &u3, &mut d_buf);

    // The sign of the exact determinant is the sign of the last non-zero
    // component of D. The non-overlapping invariant guarantees d_buf[d_len-1]
    // is the most-significant term.
    if d_len == 0 {
        return Sign::Zero;
    }
    sign_of(d_buf[d_len - 1])
}

// ===========================================================================
// Stage 4: Exact integer fallback
// ===========================================================================

/// Stage 4: exact integer arithmetic in `WideInt<66>`, 4,224 bits.
///
/// Reached in two cases: the input spans more bits than the exactness window
/// allows, so the expansion stages cannot run on it; or the expansion stages
/// returned `Zero`, which is not a certificate. In practice the second case is
/// almost always an exact zero, and this stage confirms it.
///
/// Each coordinate is converted to an integer scaled by `2^1074` using
/// [`crate::exact_int::f64_to_orient_int`], which is exact for every finite
/// `f64`, and the determinant is evaluated with no rounding.
///
/// Bit-width argument: every finite `f64` is below `2^1024` in magnitude, so a
/// scaled coordinate is below `2^2098`; a difference is below `2^2099`, a
/// product of two differences below `2^4198`, and the two-term determinant
/// below `2^4199`, which fits in 4,224 bits with a sign.
///
/// It is called with the caller's coordinates, before any rotation or
/// rescaling, so that what it decides is exactly the caller's question.
fn orient2d_exact_integer(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    counters::record(Predicate::Orient2d, Outcome::Exact);
    use crate::exact_int::f64_to_orient_int;

    let ax = f64_to_orient_int(pa[0]);
    let ay = f64_to_orient_int(pa[1]);
    let bx = f64_to_orient_int(pb[0]);
    let by = f64_to_orient_int(pb[1]);
    let cx = f64_to_orient_int(pc[0]);
    let cy = f64_to_orient_int(pc[1]);

    // det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    // All arithmetic is exact in the 4224-bit integer type.
    let acx = ax - cx;
    let acy = ay - cy;
    let bcx = bx - cx;
    let bcy = by - cy;
    let det = acx * bcy - acy * bcx;
    debug_assert!(
        det.fits_in_signed_bits(4200),
        "orient2d Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );

    if det.is_positive() {
        Sign::Positive
    } else if det.is_negative() {
        Sign::Negative
    } else {
        // Observability only (`stage-counters`): records how much of the exact
        // path is spent confirming exact zeros.
        counters::record(Predicate::Orient2d, Outcome::ExactZero);
        Sign::Zero
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
// test-lint-posture: see the workspace lint comment in Cargo.toml
#[allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::range_plus_one,
    clippy::no_effect_underscore_binding
)]
mod tests {
    use super::*;
    use core::sync::atomic::Ordering;

    fn reset_exact_count() {
        ORIENT2D_EXACT_COUNT.store(0, Ordering::SeqCst);
    }

    /// A counterclockwise triangle is `Positive`.
    ///
    /// pa=(0,0), pb=(1,0), pc=(0,1):
    /// det = (0−0)·(0−1) − (0−1)·(1−0) = 0·(−1) − (−1)·1 = 0 + 1 = 1 > 0
    #[test]
    fn orient2d_counterclockwise() {
        assert_eq!(
            orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
            Ok(Sign::Positive),
            "CCW triangle must return Positive"
        );
    }

    /// A clockwise triangle is `Negative`.
    ///
    /// pa=(0,0), pb=(0,1), pc=(1,0):
    /// det = (0−1)·(1−0) − (0−0)·(0−1) = (−1)·1 − 0·(−1) = −1 < 0
    #[test]
    fn orient2d_clockwise() {
        assert_eq!(
            orient2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]),
            Ok(Sign::Negative),
            "CW triangle must return Negative"
        );
    }

    /// Collinear points are `Zero`.
    ///
    /// pa=(0,0), pb=(1,1), pc=(2,2): all on y = x.
    /// det = (0−2)·(1−2) − (0−2)·(1−2) = (−2)·(−1) − (−2)·(−1) = 2 − 2 = 0
    #[test]
    fn orient2d_collinear() {
        assert_eq!(
            orient2d([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
            Ok(Sign::Zero),
            "Collinear points must return Zero"
        );
    }

    /// The exact path fires on a collinear triple whose coordinates are not
    /// representable, and returns `Zero`.
    ///
    /// pa = (1/3, 1/3), pb = (2/3, 2/3), pc = (1/7, 1/7).
    ///
    /// All three points lie on the line y = x (each point has equal x and y), so the
    /// exact determinant is zero and the correct result is `Sign::Zero`.
    ///
    /// Why Stage 3 fires:
    ///
    /// None of 1/3, 2/3, 1/7 are exactly representable in f64. The coordinate
    /// differences (acx = fl(1/3 − 1/7), bcy = fl(2/3 − 1/7), and so on) each carry
    /// a non-zero two_diff_tail, so the early-return guard
    /// `if acxtail == 0 && bcxtail == 0 && acytail == 0 && bcytail == 0` does NOT fire.
    ///
    /// However, because every point has x == y, we have acx == acy and bcy == bcx (as
    /// f64 values), so detleft == detright and Stage 2's B expansion is identically
    /// zero. The same symmetry makes the first-order tail correction in the
    /// CCW_ERR_BOUND_C re-test also vanish, so Stage 3 is reached.
    ///
    /// Stage 3 constructs the full D expansion, which also sums to zero (the triangle
    /// area is exactly zero). `expansion_sum` returns `d_len = 0`, and the predicate
    /// returns `Sign::Zero`.
    #[test]
    fn orient2d_exact_path_fires_on_collinear_with_non_representable_coords() {
        reset_exact_count();

        // All three points lie on y = x; exact determinant = 0.
        let pa = [1.0_f64 / 3.0, 1.0_f64 / 3.0];
        let pb = [2.0_f64 / 3.0, 2.0_f64 / 3.0];
        let pc = [1.0_f64 / 7.0, 1.0_f64 / 7.0];

        let result = orient2d(pa, pb, pc);
        assert_eq!(
            result,
            Ok(Sign::Zero),
            "points on y=x are collinear; exact path must return Zero"
        );

        let count = ORIENT2D_EXACT_COUNT.load(Ordering::SeqCst);
        assert!(
            count > 0,
            "Stage 3 must run for collinear non-representable input (count {count})"
        );
    }

    /// Non-finite inputs are rejected with `Err(NonFinite)`.
    #[test]
    fn orient2d_non_finite_inputs_return_err() {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        assert_eq!(
            orient2d([inf, 0.0], [1.0, 0.0], [0.0, 1.0]),
            Err(NonFinite::new())
        );
        assert_eq!(
            orient2d([0.0, 0.0], [nan, 0.0], [0.0, 1.0]),
            Err(NonFinite::new())
        );
        assert_eq!(
            orient2d([0.0, 0.0], [1.0, 0.0], [0.0, f64::NEG_INFINITY]),
            Err(NonFinite::new())
        );
    }

    /// All-finite inputs whose coordinate differences overflow f64 still return
    /// the exact sign (`Ok`), never a spurious `Sign::Zero`.
    #[test]
    fn orient2d_finite_overflow_is_not_zero() {
        let m = f64::MAX;
        // CCW triangle with opposite-sign extremes: pa-pc and pb-pc overflow.
        let r = orient2d([m, 0.0], [0.0, m], [-m, -m]);
        assert_eq!(r, Ok(Sign::Positive));
    }
}
