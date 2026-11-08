// src/incircle.rs

//! The adaptive in-circle predicate.
//!
//! The module ports `incircle` and `incircleadapt` from Shewchuk (1997)
//! `predicates.c`, with an underflow-protected fast filter in front and an
//! exact integer stage behind it for input the expansion stages cannot
//! certify.

use crate::counters::{self, Outcome, Predicate};
use crate::expansion::{
    ICC_ERR_BOUND_A, ICC_ERR_BOUND_B, ICC_ERR_BOUND_C, RESULT_ERR_BOUND, expansion_sum,
    scale_expansion, two_product,
};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{sign_of, two_diff_tail, two_two_diff};
use crate::{NonFinite, Sign};

#[cfg(test)]
use core::sync::atomic::{AtomicUsize, Ordering};

/// Counts the invocations of Stage 3, the exact expansion path. The counter
/// is present in test builds only.
///
/// Tests read it to establish that near-degenerate input reaches the exact
/// path (see `incircle_exact_path_fires`).
#[cfg(test)]
pub(crate) static INCIRCLE_EXACT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// The fast filter for `incircle`, in the underflow-protected form of
/// Bartels et al. 2023, Theorem 1(2).
///
/// The filter returns `Some(sign)` when the sign of the floating-point
/// determinant is certified to be the sign of the exact determinant, and
/// `None` when it is not. **The certificate holds for every finite input**, with
/// no assumption that the evaluation is free of underflow or overflow. The
/// filter of [`crate::orient2d`] explains why Shewchuk's form falls short and
/// what the protection adds. The same reasoning governs this one, one degree
/// higher.
///
/// The filter operates on the caller's coordinates exactly as given. Any
/// rescaling happens only after it, and only when it declines.
///
/// # The magnitude
///
/// The magnitude follows the association order of the determinant, which
/// sums `alift·(bdx·cdy − bdy·cdx)`, then the `blift` term, then the `clift`
/// term. Each row contributes
///
/// ```text
/// t = ((|dx·dx| + u_N) + (|dy·dy| + u_N)) · ((|p·q| + u_N) + (|r·s| + u_N)) + u_N
/// ```
///
/// with the first factor bounding the lift and the second the 2×2 minor. The
/// three rows are summed `(t_a + t_b) + t_c`. The squares need no `abs`,
/// since a square is never negative.
///
/// # Constants
///
/// Shewchuk's `ICC_ERR_BOUND_A`, `(10 + 96ε)ε`, satisfies the paper's
/// requirement for this expression tree, which admits anything above
/// `(10 + 75ε)ε`.
#[inline]
fn incircle_filter_raw(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Option<Sign> {
    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];

    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    let bc = bdx * cdy - bdy * cdx;
    let ca = cdx * ady - cdy * adx;
    let ab = adx * bdy - ady * bdx;

    let det = alift * bc + blift * ca + clift * ab;

    let lift_mag = |xx: f64, yy: f64| (xx + U_N) + (yy + U_N);
    let minor_mag = |p: f64, q: f64| (p.abs() + U_N) + (q.abs() + U_N);
    let ma = lift_mag(adx * adx, ady * ady) * minor_mag(bdx * cdy, bdy * cdx) + U_N;
    let mb = lift_mag(bdx * bdx, bdy * bdy) * minor_mag(cdx * ady, cdy * adx) + U_N;
    let mc = lift_mag(cdx * cdx, cdy * cdy) * minor_mag(adx * bdy, ady * bdx) + U_N;
    let bound = ICC_ERR_BOUND_A * ((ma + mb) + mc) + U_S;

    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// Tests whether point `pd` lies inside the circumcircle of triangle
/// `pa`, `pb`, `pc`.
///
/// The predicate returns [`Sign::Positive`] if `pd` is strictly inside the
/// circumcircle, [`Sign::Negative`] if it is strictly outside, and
/// [`Sign::Zero`] if it lies exactly on the circle.
///
/// # Mathematical definition
///
/// The predicate evaluates the sign of the 4×4 determinant
///
/// ```text
/// | pa[0]  pa[1]  pa[0]²+pa[1]²  1 |
/// | pb[0]  pb[1]  pb[0]²+pb[1]²  1 |
/// | pc[0]  pc[1]  pc[0]²+pc[1]²  1 |
/// | pd[0]  pd[1]  pd[0]²+pd[1]²  1 |
/// ```
///
/// Cofactor expansion along the last column reduces it to a 3×3 determinant
/// in the coordinate differences `(pa − pd)`, `(pb − pd)`, `(pc − pd)`,
/// augmented by the squared-distance terms. Concretely, let
///
/// ```text
/// adx = pa[0] − pd[0],  ady = pa[1] − pd[1],
/// bdx = pb[0] − pd[0],  bdy = pb[1] − pd[1],
/// cdx = pc[0] − pd[0],  cdy = pc[1] − pd[1],
/// alift = adx² + ady²,  blift = bdx² + bdy²,  clift = cdx² + cdy².
/// ```
///
/// The predicate is then the sign of
///
/// ```text
/// alift · (bdx·cdy − bdy·cdx)
/// + blift · (cdx·ady − cdy·adx)
/// + clift · (adx·bdy − ady·bdx).
/// ```
///
/// # Lifted-paraboloid interpretation
///
/// The determinant has a geometric reading on a paraboloid. Each 2D point
/// `p = (x, y)` lifts to the 3D paraboloid surface at `p̂ = (x, y, x² + y²)`,
/// and the circumcircle of a triangle in the plane corresponds to the plane
/// through its three lifted vertices. The in-circle predicate reduces to the
/// side of that plane on which the lifted fourth point falls, below (inside)
/// or above (outside). This picture is why the 4×4 determinant above encodes
/// the in-circle test exactly.
///
/// # Sign convention
///
/// The result assumes that `pa`, `pb`, `pc` are given in **counterclockwise**
/// order. Clockwise order negates the sign of the result.
///
/// # The four stages
///
/// 1. **Stage 1, the fast filter**, `incircle_filter_raw`, evaluates the
///    determinant in native `f64` with the underflow-protected error bound of
///    Bartels et al. 2023. The bound holds for every finite input, so a
///    certificate here is final.
/// 2. **Stage 2**, in `incircle_adapt`, forms exact two-product expansions of
///    the six cross-product minors, scales each by its lifted coordinate,
///    and tests the estimate against `ICC_ERR_BOUND_B`.
/// 3. **Stage 3**, also in `incircle_adapt`, adds the tail corrections for the
///    rounding error in the coordinate differences, in up to 1152
///    components. The sign of the last non-zero component is the sign of
///    the exact determinant, because `expansion_sum` and `scale_expansion`
///    keep the expansion non-overlapping.
/// 4. **Stage 4, the exact integer stage**, `incircle_exact_integer`, operates
///    on the original coordinates.
///
/// Stages 2 and 3 run only on input inside the exactness window of
/// `crate::range`. Inside it, binary64 arithmetic provably behaves as
/// Shewchuk's unbounded-exponent model does, so his bounds hold. Input that
/// one exact power-of-two rescale cannot bring into the window goes straight
/// to stage 4.
///
/// # Stack allocation
///
/// The stack cost is governed by the two largest arrays in `incircle_adapt`,
/// `fin1` and `fin2`, each of 1152 `f64` components, or 9,216 bytes. The
/// total stack per call is bounded by approximately 20 KiB, within any
/// normal thread stack budget. Stage 2 alone needs far less, an accumulator
/// of at most 96 components (768 bytes), and the 1152-component buffers are
/// sized for the worst case of Stage 3.
///
/// # Non-finite inputs
///
/// A NaN or infinite input coordinate makes the predicate return
/// [`Err(NonFinite)`](NonFinite). Such a coordinate has no geometric
/// meaning, so no [`Sign`] is correct for it. For *finite* inputs the function
/// always returns `Ok` with the exact sign.
///
/// # References
///
/// Shewchuk (1997), `incircle` and `incircleadapt`, predicates.c.
///
/// Bartels, Fisikopoulos & Weiser (2023), Theorem 1(2).
///
/// # Errors
///
/// Returns [`NonFinite`] when any input coordinate is not finite. This is
/// the only error case. A NaN or infinite coordinate has no geometric
/// meaning, and the predicate rejects it before the adaptive stages so that
/// it can never pass for a degenerate [`Sign::Zero`] result. A non-finite
/// *intermediate* is not an error, rather a recoverable artifact of finite
/// input that the exact integer stage resolves.
#[must_use = "the in-circle result is the return value; discarding it is almost certainly a bug"]
pub fn incircle(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Result<Sign, NonFinite> {
    // The strict input-domain check rejects non-finite coordinates up front,
    // so every stage below may assume finite input.
    if !pa
        .iter()
        .chain(&pb)
        .chain(&pc)
        .chain(&pd)
        .all(|v| v.is_finite())
    {
        counters::record(Predicate::Incircle, Outcome::NonFinite);
        return Err(NonFinite::new());
    }

    // The fast filter runs on the coordinates as given. Its bound holds for
    // every finite input, so it needs no guard on underflow or overflow. An
    // overflowing intermediate also overflows the magnitude, and the
    // comparison against an infinite bound fails.
    if let Some(sign) = incircle_filter_raw(pa, pb, pc, pd) {
        counters::record(Predicate::Incircle, Outcome::Fast);
        return Ok(sign);
    }

    // The expansion stages are exact only inside the window (see
    // `crate::range`). The rescale there is one exact power of two, so it
    // cannot zero a coordinate or drop low bits, and the stages always see
    // the caller's configuration, only scaled.
    let coords = [pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]];
    let (sa, sb, sc, sd) = match prepare(&coords, Window::INCIRCLE) {
        Prepared::AsIs => (pa, pb, pc, pd),
        Prepared::Scaled(s) => (
            [pa[0] * s, pa[1] * s],
            [pb[0] * s, pb[1] * s],
            [pc[0] * s, pc[1] * s],
            [pd[0] * s, pd[1] * s],
        ),
        Prepared::OutOfWindow => return Ok(incircle_exact_integer(pa, pb, pc, pd)),
    };

    // Stages 2 and 3. A `Zero` from them is not a certificate. It means only
    // that the expansion came out zero, which the exact stage must confirm.
    let result = incircle_in_window(sa, sb, sc, sd);
    Ok(if result == Sign::Zero {
        incircle_exact_integer(pa, pb, pc, pd)
    } else {
        counters::record(Predicate::Incircle, Outcome::Fast);
        result
    })
}

/// Shewchuk's Stage 1 on in-window coordinates, followed by Stages 2 and 3.
///
/// The caller guarantees that the coordinates lie inside
/// [`Window::INCIRCLE`], as given or after an exact rescale. Inside the
/// window no operation of these stages can underflow inexactly or overflow,
/// so Shewchuk's analysis governs the binary64 execution and his error
/// bounds hold as written. Outside it they do not. An inexact underflow
/// inside an expansion would make it sum to the wrong value, with nothing
/// to show for it.
///
/// This Stage 1 takes Shewchuk's form rather than the Bartels form. The raw
/// filter has already run, and inside the window the two certify the same
/// inputs, so the cheaper one is used. It catches input that the raw filter
/// declined only because the rescale had not yet happened.
///
/// A `Sign::Zero` return means that no stage could certify a sign. The
/// caller resolves it exactly.
fn incircle_in_window(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Sign {
    // Differences relative to pd eliminate the last column of the 4×4
    // determinant, and the reduced 3×3 form is better conditioned.
    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];

    // The lifted (paraboloid) coordinates, x² + y² for each shifted point.
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    // The determinant in native f64.
    let det = alift * (bdx * cdy - bdy * cdx)
        + blift * (cdx * ady - cdy * adx)
        + clift * (adx * bdy - ady * bdx);

    // `permanent` bounds from above the sum of the absolute values of each
    // 3×3 minor weighted by its lifted coordinate, with no catastrophic
    // cancellation inside the bound.
    //
    // The three terms are summed in the order `det` sums them, a, b, c.
    // Floating-point addition is not associative, so the order can move the
    // computed value by a unit in the last place. That cannot change a sign,
    // because every stage either certifies its result or passes the input
    // on. The magnitude of a determinant, however, is defined term for term
    // alongside it, and an error analysis that traces the two expressions
    // together needs them to associate the same way.
    let permanent = (bdx.abs() * cdy.abs() + bdy.abs() * cdx.abs()) * alift.abs()
        + (cdx.abs() * ady.abs() + cdy.abs() * adx.abs()) * blift.abs()
        + (adx.abs() * bdy.abs() + ady.abs() * bdx.abs()) * clift.abs();

    let error_bound = ICC_ERR_BOUND_A * permanent;
    if det > error_bound || -det > error_bound {
        return sign_of(det);
    }

    incircle_adapt(pa, pb, pc, pd, permanent)
}

// ===========================================================================
// Private: merging into the double-buffered accumulator
// ===========================================================================

/// Merges `correction` into the active `fin` buffer, writing the result into
/// the inactive one, and returns the new length. The caller must toggle
/// `active ^= 1` immediately after each call.
///
/// Taking both arrays as `&mut` lets the borrow checker establish statically
/// that each array is borrowed exactly once per call and that the two
/// borrows do not alias. Selecting the buffers with
/// `if active == 0 { &fin1 } else { &fin2 }` in both an immutable and a
/// mutable borrow within one expression would instead raise E0502.
///
/// Shewchuk's predicates.c swaps the pointers `finnow` and `finother`. This
/// function is the Rust equivalent.
#[inline(always)]
fn fin_merge(
    fin1: &mut [f64; 1152],
    fin2: &mut [f64; 1152],
    fin_len: usize,
    active: usize,
    correction: &[f64],
) -> usize {
    if active == 0 {
        expansion_sum(&fin1[..fin_len], correction, fin2)
    } else {
        expansion_sum(&fin2[..fin_len], correction, fin1)
    }
}

// ===========================================================================
// Private adaptive path
// ===========================================================================

/// Stages 2 and 3 of the adaptive `incircle` predicate.
///
/// The function ports `incircleadapt` from Shewchuk's predicates.c (1997).
///
/// ## Expansion sizes (Shewchuk's proven bounds, cited at each allocation)
///
/// | Variable      | Max components | Source |
/// |---------------|----------------|--------|
/// | `bc/ca/ab`    | 4              | `two_two_diff` of two `two_product` outputs |
/// | `axbc/aybc`   | 8              | `scale_expansion` of 4 by one f64 |
/// | `axxbc/ayybc` | 16             | `scale_expansion` of 8 by one f64 |
/// | `adet`        | 32             | `expansion_sum(16, 16)` |
/// | `bdet/cdet`   | 32 each        | same |
/// | `abdet`       | 64             | `expansion_sum(32, 32)` |
/// | `fin1/fin2`   | 1152           | accumulator after all Stage 3 corrections |
///
/// ## Stage 2
///
/// Stage 2 builds the exact 4-component cross-product expansions `bc`, `ca`
/// and `ab`, each `p*q - r*s` by `two_product` and `two_two_diff`. It scales
/// each twice by the matching coordinate differences and sums the results
/// into `adet`, `bdet` and `cdet` (32 components each), then into `abdet`
/// (64), and finally into a 96-component `fin` accumulator. The estimate
/// `estimate(fin) = sum(fin[..fin_len])` is tested against
/// `ICC_ERR_BOUND_B * permanent`.
///
/// ## Intermediate re-test
///
/// When all six coordinate-difference tails are zero, Stage 2 is exact and
/// returns at once. Otherwise the estimate is refined with the first-order
/// tail corrections, still in native f64, and tested against
/// `ICC_ERR_BOUND_C * permanent + RESULT_ERR_BOUND * |det|`.
///
/// ## Stage 3 (exact path)
///
/// Stage 3 adds up to six blocks of single-tail corrections, one per
/// non-zero tail (`adxtail`, `adytail`, `bdxtail`, `bdytail`, `cdxtail`,
/// `cdytail`), and up to three blocks of compound tail corrections, the
/// cross-products of pairs of tails. Each block enters `fin` through
/// `expansion_sum`, and the sign of the last non-zero component of the final
/// `fin` is returned.
///
/// The test-only counter `INCIRCLE_EXACT_COUNT` is incremented on entry to
/// Stage 3, after the intermediate re-test fails.
// The function exceeds the crate's 100-line function limit deliberately. It
// is Shewchuk's `incircleadapt`, and its later stages share many live
// expansion buffers and lengths. Splitting it would need a large state
// struct or very long parameter lists, and would obscure the correspondence
// with the published algorithm, which is how this function is checked.
#[allow(clippy::too_many_lines)]
fn incircle_adapt(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2], permanent: f64) -> Sign {
    // ===========================================================================
    // Stage 2: exact two-product expansions of the determinant.
    // ===========================================================================

    // The coordinate differences, recomputed here as Stage 1 computes them.
    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];

    // bc = bdx*cdy - bdy*cdx, 4 components (two_two_diff of two two_products)
    let (bdxcdy1, bdxcdy0) = two_product(bdx, cdy);
    let (cdxbdy1, cdxbdy0) = two_product(cdx, bdy);
    let bc = two_two_diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0);

    // ca = cdx*ady - cdy*adx  (4 components)
    let (cdxady1, cdxady0) = two_product(cdx, ady);
    let (adxcdy1, adxcdy0) = two_product(adx, cdy);
    let ca = two_two_diff(cdxady1, cdxady0, adxcdy1, adxcdy0);

    // ab = adx*bdy - ady*bdx  (4 components)
    let (adxbdy1, adxbdy0) = two_product(adx, bdy);
    let (bdxady1, bdxady0) = two_product(bdx, ady);
    let ab = two_two_diff(adxbdy1, adxbdy0, bdxady1, bdxady0);

    // The "a" row contribution scales bc by adx and ady, then combines.
    //   axbc = adx * bc  (8 components: scale_expansion of 4 by one f64)
    //   axxbc = adx * axbc  (16 components: scale_expansion of 8)
    //   aybc = ady * bc  (8 components)
    //   ayybc = ady * aybc  (16 components)
    //   adet = axxbc + ayybc  (32 components: expansion_sum of 16+16)
    let mut axbc = [0.0_f64; 8]; // scale(4) → 8
    let axbc_len = scale_expansion(&bc, adx, &mut axbc);
    let mut axxbc = [0.0_f64; 16]; // scale(8) → 16
    let axxbc_len = scale_expansion(&axbc[..axbc_len], adx, &mut axxbc);
    let mut aybc = [0.0_f64; 8];
    let aybc_len = scale_expansion(&bc, ady, &mut aybc);
    let mut ayybc = [0.0_f64; 16];
    let ayybc_len = scale_expansion(&aybc[..aybc_len], ady, &mut ayybc);
    let mut adet = [0.0_f64; 32]; // expansion_sum(16, 16) → 32
    let adet_len = expansion_sum(&axxbc[..axxbc_len], &ayybc[..ayybc_len], &mut adet);

    // The "b" row contribution scales ca by bdx and bdy.
    let mut bxca = [0.0_f64; 8];
    let bxca_len = scale_expansion(&ca, bdx, &mut bxca);
    let mut bxxca = [0.0_f64; 16];
    let bxxca_len = scale_expansion(&bxca[..bxca_len], bdx, &mut bxxca);
    let mut byca = [0.0_f64; 8];
    let byca_len = scale_expansion(&ca, bdy, &mut byca);
    let mut byyca = [0.0_f64; 16];
    let byyca_len = scale_expansion(&byca[..byca_len], bdy, &mut byyca);
    let mut bdet = [0.0_f64; 32];
    let bdet_len = expansion_sum(&bxxca[..bxxca_len], &byyca[..byyca_len], &mut bdet);

    // The "c" row contribution scales ab by cdx and cdy.
    let mut cxab = [0.0_f64; 8];
    let cxab_len = scale_expansion(&ab, cdx, &mut cxab);
    let mut cxxab = [0.0_f64; 16];
    let cxxab_len = scale_expansion(&cxab[..cxab_len], cdx, &mut cxxab);
    let mut cyab = [0.0_f64; 8];
    let cyab_len = scale_expansion(&ab, cdy, &mut cyab);
    let mut cyyab = [0.0_f64; 16];
    let cyyab_len = scale_expansion(&cyab[..cyab_len], cdy, &mut cyyab);
    let mut cdet = [0.0_f64; 32];
    let cdet_len = expansion_sum(&cxxab[..cxxab_len], &cyyab[..cyyab_len], &mut cdet);

    // The three row contributions combine.
    let mut abdet = [0.0_f64; 64]; // expansion_sum(32, 32) → 64
    let abdet_len = expansion_sum(&adet[..adet_len], &bdet[..bdet_len], &mut abdet);

    // fin, the running determinant accumulator, holds 1152 components in the
    // worst case. Stage 2 initializes it from abdet + cdet, at most
    // 64 + 32 = 96 components.
    let mut fin1 = [0.0_f64; 1152]; // proven Stage 3 worst-case bound (predicates.c)
    let mut fin2 = [0.0_f64; 1152];
    let mut fin_len = expansion_sum(&abdet[..abdet_len], &cdet[..cdet_len], &mut fin1);

    // The Stage 2 estimate sums every fin component, as Shewchuk's
    // estimate() does. Taking only the last component would be wrong when
    // zero elimination has left a smaller-indexed component carrying the
    // magnitude.
    let mut det: f64 = fin1[..fin_len].iter().sum();
    let error_bound = ICC_ERR_BOUND_B * permanent;
    if det >= error_bound || -det >= error_bound {
        return sign_of(det);
    }

    // ===========================================================================
    // The coordinate-difference tails.
    // ===========================================================================

    let adxtail = two_diff_tail(pa[0], pd[0], adx);
    let adytail = two_diff_tail(pa[1], pd[1], ady);
    let bdxtail = two_diff_tail(pb[0], pd[0], bdx);
    let bdytail = two_diff_tail(pb[1], pd[1], bdy);
    let cdxtail = two_diff_tail(pc[0], pd[0], cdx);
    let cdytail = two_diff_tail(pc[1], pd[1], cdy);

    // With every tail zero, the Stage 2 expansion is the exact determinant.
    if adxtail == 0.0
        && adytail == 0.0
        && bdxtail == 0.0
        && bdytail == 0.0
        && cdxtail == 0.0
        && cdytail == 0.0
    {
        return sign_of(det);
    }

    // The intermediate re-test refines det with first-order tail corrections
    // in f64. The bound is computed first, from the original det, before det
    // is updated.
    let error_bound = ICC_ERR_BOUND_C * permanent + RESULT_ERR_BOUND * det.abs();
    det += (adx * adx + ady * ady)
        * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail))
        + 2.0 * (adx * adxtail + ady * adytail) * (bdx * cdy - bdy * cdx)
        + (bdx * bdx + bdy * bdy)
            * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail))
        + 2.0 * (bdx * bdxtail + bdy * bdytail) * (cdx * ady - cdy * adx)
        + (cdx * cdx + cdy * cdy)
            * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail))
        + 2.0 * (cdx * cdxtail + cdy * cdytail) * (adx * bdy - ady * bdx);

    if det >= error_bound || -det >= error_bound {
        return sign_of(det);
    }

    // ===========================================================================
    // Stage 3: the full exact path.
    // INCIRCLE_EXACT_COUNT is incremented here (test builds only).
    // ===========================================================================

    #[cfg(test)]
    INCIRCLE_EXACT_COUNT.fetch_add(1, Ordering::Relaxed);

    // Scratch arrays reused across correction blocks, sized by Shewchuk's
    // proven analysis (each is a scale_expansion or expansion_sum output).
    // tmp4a and tmp4b are declared inside each compound block that uses them
    // rather than hoisted here, which avoids the `unused_assignments` warning
    // from an initializer that is immediately overwritten.
    let mut tmp8 = [0.0_f64; 8]; // scale_expansion of 4
    let mut tmp8b = [0.0_f64; 8];
    let mut tmp16 = [0.0_f64; 16]; // scale_expansion of 8
    let mut tmp16b = [0.0_f64; 16];
    // tmp32 holds the alift·bct compound terms (a scaling of up to 16
    // components).
    let mut tmp32 = [0.0_f64; 32];

    // The double-buffer index names the buffer holding the current expansion,
    // 0 for fin1 and 1 for fin2. fin_merge() reads the active buffer and
    // writes into the inactive one, and active ^= 1 then toggles it.
    let mut active: usize = 0;

    // ===========================================================================
    // Stage 3: single-tail correction blocks.
    //
    // Each non-zero coordinate-difference tail of point a, b or c adds its
    // correction terms to the fin accumulator.
    //
    // The pattern mirrors incircleadapt in predicates.c.
    // ===========================================================================

    // Single-tail correction blocks.
    //
    // Each non-zero coordinate-difference tail t of point p adds the two
    // primary correction terms arising from the partial derivative ∂D/∂coord,
    //
    //   (1) scale(base, tail) × (2·coord), the linear primary,
    //   (2) scale(base, tail) × tail, the quadratic primary,
    //
    // where `base` is the 3×3 minor cross-product of that row:
    //   adxtail/adytail → base = bc; bdxtail/bdytail → base = ca;
    //   cdxtail/cdytail → base = ab.
    //
    // The lift-correction terms (blift·(-cdy), clift·bdy, and so on) are not
    // handled here. The outer alift·bct, blift·cat and clift·abt terms of the
    // compound blocks capture them, accounting correctly for how the tails of
    // the other two points perturb each point's lift×cross contribution. The
    // compound block comments below trace the derivation.

    if adxtail != 0.0 {
        let tmp8_len = scale_expansion(&bc, adxtail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * adx, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], adxtail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    if adytail != 0.0 {
        let tmp8_len = scale_expansion(&bc, adytail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * ady, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], adytail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    if bdxtail != 0.0 {
        let tmp8_len = scale_expansion(&ca, bdxtail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * bdx, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], bdxtail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    if bdytail != 0.0 {
        let tmp8_len = scale_expansion(&ca, bdytail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * bdy, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], bdytail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    if cdxtail != 0.0 {
        let tmp8_len = scale_expansion(&ab, cdxtail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * cdx, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], cdxtail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    if cdytail != 0.0 {
        let tmp8_len = scale_expansion(&ab, cdytail, &mut tmp8);
        let tmp16_len = scale_expansion(&tmp8[..tmp8_len], 2.0 * cdy, &mut tmp16);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..tmp16_len]);
        active ^= 1;
        let tmp16b_len = scale_expansion(&tmp8[..tmp8_len], cdytail, &mut tmp16b);
        fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16b[..tmp16b_len]);
        active ^= 1;
    }

    // ===========================================================================
    // Stage 3: compound tail correction blocks.
    //
    // These blocks handle cross-products of tails from different points:
    //   (adxtail, adytail) × (bdy, cdy) and (bdx, cdx) cross terms
    //   (bdxtail, bdytail) × (cdy, ady) and (cdx, adx) cross terms
    //   (cdxtail, cdytail) × (ady, bdy) and (adx, bdx) cross terms
    //
    // bct = (bdxtail*cdy + cdxtail*(-bdy)) + (bdx*cdytail + cdx*(-bdytail))
    //       as a 4-component exact expansion (two_two_diff of two_product pairs).
    // bctt = bdxtail*cdytail - bdytail*cdxtail (4 components).
    // The contributions are then
    //   fin += scale(bct, adxtail) × (2*adx)
    //   fin += scale(bct, adxtail) × adxtail
    //   fin += scale(bctt, adxtail) × (2*adx)
    //   fin += scale(bctt, adxtail) × adxtail
    // (and likewise for adytail, and for the b and c point compound blocks)
    // ===========================================================================

    // The outer guard of each compound block checks whether the other two
    // points have non-zero tails. Since bct = cross(b_tails, c_coords /
    // c_tails), alift·bct is non-zero only when b or c has tails. The inner
    // `if adxtail` and `if adytail` sub-blocks add the tail×tail cross terms.
    if bdxtail != 0.0 || bdytail != 0.0 || cdxtail != 0.0 || cdytail != 0.0 {
        // bct and bctt. bct carries the cross-product of the tails of b and
        // c with the non-tail coordinates, and bctt the tail×tail
        // cross-product.
        let (bct, bct_len, bctt, bctt_len) =
            if bdxtail != 0.0 || bdytail != 0.0 || cdxtail != 0.0 || cdytail != 0.0 {
                // bct = (bdxtail*cdy - cdxtail*bdy) + (bdx*cdytail - cdx*bdytail)
                let (p1, p0) = two_product(bdxtail, cdy);
                let (q1, q0) = two_product(cdxtail, bdy);
                let tmp4a = two_two_diff(p1, p0, q1, q0); // bdxtail*cdy - cdxtail*bdy
                let (p1, p0) = two_product(bdx, cdytail);
                let (q1, q0) = two_product(cdx, bdytail);
                let tmp4b = two_two_diff(p1, p0, q1, q0); // bdx*cdytail - cdx*bdytail
                let bct_len = expansion_sum(&tmp4a, &tmp4b, &mut tmp8);
                // bctt = bdxtail*cdytail - bdytail*cdxtail
                let (p1, p0) = two_product(bdxtail, cdytail);
                let (q1, q0) = two_product(bdytail, cdxtail);
                let bctt = two_two_diff(p1, p0, q1, q0);
                (tmp8, bct_len, bctt, 4usize)
            } else {
                let mut bct_z = [0.0_f64; 8];
                bct_z[0] = 0.0;
                let bctt_z = [0.0_f64; 4];
                (bct_z, 1usize, bctt_z, 1usize)
            };

        // alift · bct, with bct·(adx²+ady²) split as
        // scale(scale(bct,adx),adx) + scale(scale(bct,ady),ady).
        // From bct_len ≤ 8, one scaling by adx gives ≤ 16 and the second ≤ 32.
        {
            let t16_len = scale_expansion(&bct[..bct_len], adx, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], adx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        {
            let t16_len = scale_expansion(&bct[..bct_len], ady, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], ady, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        // alift · bctt: bctt_len ≤ 4 → scale by adx → ≤ 8 → ≤ 16.
        {
            let t8_len = scale_expansion(&bctt[..bctt_len], adx, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], adx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }
        {
            let t8_len = scale_expansion(&bctt[..bctt_len], ady, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], ady, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }

        if adxtail != 0.0 {
            // fin += scale(bct, adxtail) × (2*adx)
            // bct_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*adx) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&bct[..bct_len], adxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * adx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // fin += scale(bctt, adxtail) × (2*adx)
            // bctt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 are sufficient.
            let t8_len = scale_expansion(&bctt[..bctt_len], adxtail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * adx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // fin += scale(scale(bctt, adxtail), adxtail): reuses bctt×adxtail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], adxtail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // fin += scale(scale(bct, adxtail), adxtail): bct_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&bct[..bct_len], adxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], adxtail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }

        if adytail != 0.0 {
            // bct_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*ady) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&bct[..bct_len], adytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * ady, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // bctt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 sufficient.
            let t8_len = scale_expansion(&bctt[..bctt_len], adytail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * ady, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(bctt, adytail), adytail): reuses bctt×adytail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], adytail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(bct, adytail), adytail): bct_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&bct[..bct_len], adytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], adytail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
    }

    // The compound block of point b (cat, catt). Its outer guard requires
    // tails on c or a, without which cat = cross(c_tails, a_coords/a_tails)
    // is zero and blift·cat contributes nothing.
    if cdxtail != 0.0 || cdytail != 0.0 || adxtail != 0.0 || adytail != 0.0 {
        let (cat, cat_len, catt, catt_len) =
            if cdxtail != 0.0 || cdytail != 0.0 || adxtail != 0.0 || adytail != 0.0 {
                // cat = (cdxtail*ady - adxtail*cdy) + (cdx*adytail - adx*cdytail)
                let (p1, p0) = two_product(cdxtail, ady);
                let (q1, q0) = two_product(adxtail, cdy);
                let tmp4a = two_two_diff(p1, p0, q1, q0);
                let (p1, p0) = two_product(cdx, adytail);
                let (q1, q0) = two_product(adx, cdytail);
                let tmp4b = two_two_diff(p1, p0, q1, q0);
                let cat_len = expansion_sum(&tmp4a, &tmp4b, &mut tmp8);
                let (p1, p0) = two_product(cdxtail, adytail);
                let (q1, q0) = two_product(cdytail, adxtail);
                let catt = two_two_diff(p1, p0, q1, q0);
                (tmp8, cat_len, catt, 4usize)
            } else {
                let mut cat_z = [0.0_f64; 8];
                cat_z[0] = 0.0;
                let catt_z = [0.0_f64; 4];
                (cat_z, 1usize, catt_z, 1usize)
            };

        // blift · cat, with cat·(bdx²+bdy²) split as
        // scale(scale(cat,bdx),bdx) + scale(scale(cat,bdy),bdy).
        // From cat_len ≤ 8, one scaling by bdx gives ≤ 16 and the second ≤ 32.
        {
            let t16_len = scale_expansion(&cat[..cat_len], bdx, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], bdx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        {
            let t16_len = scale_expansion(&cat[..cat_len], bdy, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], bdy, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        // blift · catt: catt_len ≤ 4 → scale by bdx → ≤ 8 → ≤ 16.
        {
            let t8_len = scale_expansion(&catt[..catt_len], bdx, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], bdx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }
        {
            let t8_len = scale_expansion(&catt[..catt_len], bdy, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], bdy, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }

        if bdxtail != 0.0 {
            // cat_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*bdx) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&cat[..cat_len], bdxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * bdx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // catt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 sufficient.
            let t8_len = scale_expansion(&catt[..catt_len], bdxtail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * bdx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(catt, bdxtail), bdxtail): reuses catt×bdxtail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], bdxtail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(cat, bdxtail), bdxtail): cat_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&cat[..cat_len], bdxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], bdxtail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }

        if bdytail != 0.0 {
            // cat_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*bdy) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&cat[..cat_len], bdytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * bdy, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // catt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 sufficient.
            let t8_len = scale_expansion(&catt[..catt_len], bdytail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * bdy, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(catt, bdytail), bdytail): reuses catt×bdytail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], bdytail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(cat, bdytail), bdytail): cat_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&cat[..cat_len], bdytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], bdytail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
    }

    // The compound block of point c (abt, abtt). Its outer guard requires
    // tails on a or b, without which abt = cross(a_tails, b_coords/b_tails)
    // is zero and clift·abt contributes nothing.
    if adxtail != 0.0 || adytail != 0.0 || bdxtail != 0.0 || bdytail != 0.0 {
        let (abt, abt_len, abtt, abtt_len) =
            if adxtail != 0.0 || adytail != 0.0 || bdxtail != 0.0 || bdytail != 0.0 {
                // abt = (adxtail*bdy - bdxtail*ady) + (adx*bdytail - bdx*adytail)
                let (p1, p0) = two_product(adxtail, bdy);
                let (q1, q0) = two_product(bdxtail, ady);
                let tmp4a = two_two_diff(p1, p0, q1, q0);
                let (p1, p0) = two_product(adx, bdytail);
                let (q1, q0) = two_product(bdx, adytail);
                let tmp4b = two_two_diff(p1, p0, q1, q0);
                let abt_len = expansion_sum(&tmp4a, &tmp4b, &mut tmp8);
                let (p1, p0) = two_product(adxtail, bdytail);
                let (q1, q0) = two_product(adytail, bdxtail);
                let abtt = two_two_diff(p1, p0, q1, q0);
                (tmp8, abt_len, abtt, 4usize)
            } else {
                let mut abt_z = [0.0_f64; 8];
                abt_z[0] = 0.0;
                let abtt_z = [0.0_f64; 4];
                (abt_z, 1usize, abtt_z, 1usize)
            };

        // clift · abt, with abt·(cdx²+cdy²) split as
        // scale(scale(abt,cdx),cdx) + scale(scale(abt,cdy),cdy).
        // From abt_len ≤ 8, one scaling by cdx gives ≤ 16 and the second ≤ 32.
        {
            let t16_len = scale_expansion(&abt[..abt_len], cdx, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], cdx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        {
            let t16_len = scale_expansion(&abt[..abt_len], cdy, &mut tmp16);
            let t32_len = scale_expansion(&tmp16[..t16_len], cdy, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
        // clift · abtt: abtt_len ≤ 4 → scale by cdx → ≤ 8 → ≤ 16.
        {
            let t8_len = scale_expansion(&abtt[..abtt_len], cdx, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], cdx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }
        {
            let t8_len = scale_expansion(&abtt[..abtt_len], cdy, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], cdy, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
        }

        if cdxtail != 0.0 {
            // abt_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*cdx) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&abt[..abt_len], cdxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * cdx, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // abtt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 sufficient.
            let t8_len = scale_expansion(&abtt[..abtt_len], cdxtail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * cdx, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(abtt, cdxtail), cdxtail): reuses abtt×cdxtail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], cdxtail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(abt, cdxtail), cdxtail): abt_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&abt[..abt_len], cdxtail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], cdxtail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }

        if cdytail != 0.0 {
            // abt_len ≤ 8 → output ≤ 16 (needs tmp16b); ×(2*cdy) → ≤ 32 (needs tmp32).
            let t16b_len = scale_expansion(&abt[..abt_len], cdytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], 2.0 * cdy, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
            // abtt_len ≤ 4 → ≤ 8 → ≤ 16: tmp8b and tmp16 sufficient.
            let t8_len = scale_expansion(&abtt[..abtt_len], cdytail, &mut tmp8b);
            let t16_len = scale_expansion(&tmp8b[..t8_len], 2.0 * cdy, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(abtt, cdytail), cdytail): reuses abtt×cdytail from above.
            let t16_len = scale_expansion(&tmp8b[..t8_len], cdytail, &mut tmp16);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp16[..t16_len]);
            active ^= 1;
            // scale(scale(abt, cdytail), cdytail): abt_len ≤ 8 → ≤ 16 → ≤ 32.
            let t16b_len = scale_expansion(&abt[..abt_len], cdytail, &mut tmp16b);
            let t32_len = scale_expansion(&tmp16b[..t16b_len], cdytail, &mut tmp32);
            fin_len = fin_merge(&mut fin1, &mut fin2, fin_len, active, &tmp32[..t32_len]);
            active ^= 1;
        }
    }

    // The sign of the exact determinant is the sign of the last non-zero
    // component of the active fin buffer, which the non-overlapping
    // invariant establishes as the most significant.
    if fin_len == 0 {
        return Sign::Zero;
    }
    let fin_slice = if active == 0 {
        &fin1[..fin_len]
    } else {
        &fin2[..fin_len]
    };
    sign_of(fin_slice[fin_len - 1])
}

// ===========================================================================
// Stage 4: the exact integer fallback
// ===========================================================================

/// Stage 4, exact integer arithmetic in `WideInt<132>` (8,448 bits).
///
/// Two cases reach this stage. Either the input spans more bits than the
/// exactness window allows, so the expansion stages cannot run on it, or the
/// expansion stages returned `Zero`, which is not a certificate. The second
/// case is in practice almost always an exact zero, and this stage confirms
/// it.
///
/// [`crate::exact_int::f64_to_incircle_int`] converts each coordinate to an
/// integer scaled by `2^1074`, exactly for every finite `f64`, and the
/// determinant is evaluated with no rounding.
///
/// The width of `WideInt<132>`, 8448 bits, rests on a short count. Each
/// scaled coordinate is at most 2099 bits, and so is each difference. A lift
/// term, the sum of two squared differences, is at most 4199 bits, and each
/// product of a lift with a 2×2 sub-determinant is below `2^8398` in
/// magnitude. The three-term sum is therefore below `2^8400`, well inside the
/// 8448-bit allocation.
///
/// The stage uses the coordinates from before normalization, because
/// normalization may underflow tiny values to zero and destroy the
/// information an exact computation needs.
fn incircle_exact_integer(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Sign {
    counters::record(Predicate::Incircle, Outcome::Exact);
    use crate::exact_int::f64_to_incircle_int;

    let ax = f64_to_incircle_int(pa[0]);
    let ay = f64_to_incircle_int(pa[1]);
    let bx = f64_to_incircle_int(pb[0]);
    let by = f64_to_incircle_int(pb[1]);
    let cx = f64_to_incircle_int(pc[0]);
    let cy = f64_to_incircle_int(pc[1]);
    let dx = f64_to_incircle_int(pd[0]);
    let dy = f64_to_incircle_int(pd[1]);

    // Coordinate differences relative to pd.
    let adx = ax - dx;
    let ady = ay - dy;
    let bdx = bx - dx;
    let bdy = by - dy;
    let cdx = cx - dx;
    let cdy = cy - dy;

    // The lifted paraboloid coordinates, adx² + ady², bdx² + bdy² and
    // cdx² + cdy². Each product is exact in the 8448-bit WideInt<132> type,
    // and since WideInt implements Copy, no clone() calls are needed.
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;

    // The 3×3 determinant expands as
    //   | adx  ady  alift |
    //   | bdx  bdy  blift |
    //   | cdx  cdy  clift |
    //
    //   = alift * (bdx * cdy - bdy * cdx)
    //   + blift * (cdx * ady - cdy * adx)
    //   + clift * (adx * bdy - ady * bdx)
    let det = alift * (bdx * cdy - bdy * cdx)
        + blift * (cdx * ady - cdy * adx)
        + clift * (adx * bdy - ady * bdx);

    // The width guard. The count above puts the magnitude below `2^8400`,
    // so the value fits in 8401 signed bits. If the guard fires, the width
    // analysis is wrong and the returned sign may be incorrect.
    debug_assert!(
        det.fits_in_signed_bits(8401),
        "incircle Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );

    if det.is_positive() {
        Sign::Positive
    } else if det.is_negative() {
        Sign::Negative
    } else {
        // Observability only (`stage-counters`), recording how much of the
        // exact path goes to confirming exact zeros.
        counters::record(Predicate::Incircle, Outcome::ExactZero);
        Sign::Zero
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
// test-lint-posture: see the lint table in Cargo.toml
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
        INCIRCLE_EXACT_COUNT.store(0, Ordering::SeqCst);
    }

    /// A point strictly inside the circumcircle of a counterclockwise triangle.
    ///
    /// The triangle pa=(0,0), pb=(1,0), pc=(1,1) is counterclockwise, with
    /// its right angle at pb, so its circumcenter is the midpoint (0.5, 0.5)
    /// of the hypotenuse pa–pc and its radius is sqrt(0.5). pd is that
    /// center, at distance 0 from it and strictly inside.
    #[test]
    fn incircle_point_inside() {
        assert_eq!(
            incircle([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]),
            Ok(Sign::Positive),
            "pd=(0.5,0.5) is inside the circumcircle of (0,0),(1,0),(1,1)"
        );
    }

    /// A point strictly outside the circumcircle.
    ///
    /// The triangle is the one above. pd=(2,2) lies at distance 1.5·sqrt(2)
    /// from the center (0.5, 0.5), against a radius of sqrt(0.5).
    #[test]
    fn incircle_point_outside() {
        assert_eq!(
            incircle([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
            Ok(Sign::Negative),
            "pd=(2,2) is outside the circumcircle of (0,0),(1,0),(1,1)"
        );
    }

    /// A point exactly on the circumcircle.
    ///
    /// The triangle is pa=(0,0), pb=(1,0), pc=(0,1), and pd=pa=(0,0) is a
    /// vertex, so it lies on the circle. The fourth row of the 4×4
    /// determinant equals the first, which makes the determinant exactly
    /// zero.
    #[test]
    fn incircle_point_on_circle() {
        // pd = pa = (0,0) lies exactly on the circumcircle of
        // (0,0),(1,0),(0,1), and the 4×4 det is zero because its fourth row
        // equals its first.
        assert_eq!(
            incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]),
            Ok(Sign::Zero),
            "pd = pa is on the circumcircle; expect Zero"
        );
    }

    /// The circumcenter lies inside the circle, not on it.
    ///
    /// The triangle pa=(0,0), pb=(1,0), pc=(0,1) has its right angle at pa,
    /// so its circumcenter is the midpoint (0.5, 0.5) of pb–pc. The center of
    /// a circle is at distance 0 from it, strictly less than the radius, so a
    /// plausible-looking `Zero` here would be wrong.
    #[test]
    fn incircle_circumcenter_is_inside() {
        // pd=(0.5,0.5) is the circumcenter, inside the circumcircle rather
        // than on it. The correct result is Sign::Positive.
        assert_eq!(
            incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
            Ok(Sign::Positive),
            "pd=(0.5,0.5) is the circumcenter of (0,0),(1,0),(0,1), strictly inside"
        );
    }

    /// Cocircular input whose coordinate differences are inexact reaches the
    /// exact path (Stage 3), which still returns `Zero`.
    ///
    /// The input, with `a = fl(3/5)` and `b = fl(4/5)`, is
    ///   pa = (a, b),  pb = (b, -a),  pc = (-b, a),  pd = (-a, -b).
    ///
    /// The exact answer is `Zero`. Each point is the previous one rotated by
    /// a quarter turn about the origin, (x, y) → (y, -x). Negation and
    /// swapping coordinates are exact in floating point, so the four `f64`
    /// points are exact quarter turns of each other, and all lie on the
    /// circle of radius sqrt(a² + b²) about the origin. This holds for any
    /// `a` and `b`, and does not rely on 3/5 and 4/5 being a Pythagorean
    /// pair.
    ///
    /// Stage 3 is reached because 3/5 and 4/5 are not representable.
    /// Differences such as bdx = b − (−a) = a + b round and leave non-zero
    /// tails, so the all-tails-zero early return inside `incircle_adapt`
    /// cannot fire, and the fast stages cannot certify a sign for a zero
    /// determinant.
    #[test]
    fn incircle_exact_path_fires() {
        reset_exact_count();

        let (a, b) = (3.0_f64 / 5.0, 4.0_f64 / 5.0);
        let pa = [a, b];
        let pb = [b, -a];
        let pc = [-b, a];
        let pd = [-a, -b];
        assert_eq!(incircle(pa, pb, pc, pd), Ok(Sign::Zero));

        let count = INCIRCLE_EXACT_COUNT.load(Ordering::SeqCst);
        assert!(
            count > 0,
            "Stage 3 must be reached on cocircular input with inexact differences (count {count})"
        );
    }

    /// Non-finite inputs are rejected with `Err(NonFinite)`.
    #[test]
    fn incircle_non_finite_inputs_return_err() {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        assert_eq!(
            incircle([inf, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]),
            Err(NonFinite::new())
        );
        assert_eq!(
            incircle([0.0, 0.0], [1.0, nan], [1.0, 1.0], [0.5, 0.5]),
            Err(NonFinite::new())
        );
        assert_eq!(
            incircle([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [f64::NEG_INFINITY, 0.5]),
            Err(NonFinite::new())
        );
    }
}
