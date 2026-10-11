// src/orient3d.rs

//! Adaptive `orient3d` predicate: orientation of four 3D points.
//!
//! Ports `orient3d` and `orient3dadapt` from Shewchuk (1997) `predicates.c`,
//! with an underflow-protected fast filter in front and an exact integer stage
//! behind, for input the expansion stages cannot certify.

#[cfg(test)]
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
pub(crate) static ORIENT3D_EXACT_COUNT: AtomicUsize = AtomicUsize::new(0);

use crate::counters::{self, Outcome, Predicate};
use crate::expansion::{
    O3D_ERR_BOUND_A, O3D_ERR_BOUND_B, O3D_ERR_BOUND_C, RESULT_ERR_BOUND, expansion_sum,
    scale_expansion, two_product,
};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{sign_of, two_diff_tail, two_two_diff};
use crate::{NonFinite, Sign};

// ===========================================================================
// Private: the fast filter
// ===========================================================================

/// The fast filter for `orient3d`, in the underflow-protected form of
/// Bartels et al. 2023, Theorem 1(2).
///
/// Returns `Some(sign)` when the sign of the floating-point determinant is
/// certified to be the sign of the exact determinant, and `None` when it is
/// not. **The certificate holds for every finite input**, with no assumption
/// that the evaluation is free of underflow or overflow. See
/// [`crate::orient2d`]'s filter for why Shewchuk's form is not enough and
/// what the protection does.
///
/// `orient3d` is the predicate where an unprotected filter fails most easily,
/// including at ordinary magnitudes when one coordinate is a subnormal next
/// to zero. The mechanism is visible in the shape of the determinant: an
/// inner 2×2 minor underflows to a subnormal, losing bits, and is then
/// multiplied by a `z` difference that may be large, scaling the absolute
/// error back up to the size of the result.
///
/// # The magnitude
///
/// Built in the association order of the determinant, which sums the `adz`
/// term, then the `bdz` term, then the `cdz` term. Each row contributes
///
/// ```text
/// t = |z| · ((|p·q| + u_N) + (|r·s| + u_N)) + u_N
/// ```
///
/// and the three are summed `(t1 + t2) + t3`. The `z` difference is a
/// coordinate difference, not a product, so it needs no `u_N` of its own: a
/// subtraction in binary64 with gradual underflow is exact when its result is
/// subnormal.
///
/// # Constants
///
/// Shewchuk's `O3D_ERR_BOUND_A`, `(7 + 56ε)ε`, satisfies the paper's
/// requirement for this expression tree, which is anything above
/// `(7 + 42ε)ε`.
#[inline]
fn orient3d_filter_raw(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> Option<Sign> {
    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let adz = pa[2] - pd[2];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let bdz = pb[2] - pd[2];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];
    let cdz = pc[2] - pd[2];

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    let row = |z: f64, p: f64, q: f64| z.abs() * ((p.abs() + U_N) + (q.abs() + U_N)) + U_N;
    let m1 = row(adz, bdxcdy, cdxbdy);
    let m2 = row(bdz, cdxady, adxcdy);
    let m3 = row(cdz, adxbdy, bdxady);
    let bound = O3D_ERR_BOUND_A * ((m1 + m2) + m3) + U_S;

    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

// ===========================================================================
// Public predicate
// ===========================================================================

/// Tests the orientation of the tetrahedron formed by four 3D points.
///
/// Returns [`Sign::Positive`] if `pd` lies below the plane through
/// `pa`, `pb`, `pc` (as seen from above in a right-handed coordinate
/// system), [`Sign::Negative`] if above, and [`Sign::Zero`] if coplanar.
///
/// # Mathematical definition
///
/// Evaluates the sign of the 4×4 determinant:
/// ```text
/// | pa.x  pa.y  pa.z  1 |
/// | pb.x  pb.y  pb.z  1 |
/// | pc.x  pc.y  pc.z  1 |
/// | pd.x  pd.y  pd.z  1 |
/// ```
/// which equals the sign of the 3×3 determinant of the relative coordinate
/// differences:
/// ```text
/// | adx  ady  adz |
/// | bdx  bdy  bdz |
/// | cdx  cdy  cdz |
/// ```
/// where `adx = pa.x - pd.x`, etc.
///
/// # Non-finite inputs
///
/// If any input coordinate is NaN or infinite, the predicate returns
/// [`Err(NonFinite)`](NonFinite): a non-finite coordinate has no geometric
/// meaning, so there is no correct [`Sign`] to return. It is reported as an
/// error rather than a plausible-looking `Sign::Zero`.
///
/// For *finite* inputs the function always returns `Ok` with the exact sign,
/// even when intermediate coordinate differences overflow `f64` to infinity:
/// such inputs are resolved by the exact Stage 4 integer path, never by a
/// premature `Sign::Zero`.
///
/// # Robustness: four stages
///
/// 1. **The fast filter**, `orient3d_filter_raw`: the determinant in native
///    `f64`, with the underflow-protected error bound of Bartels et al. 2023.
///    Valid for every finite input, so a certificate here is final.
/// 2. **Stage 2**, in `orient3d_adapt`: exact expansions of the three 2×2
///    minors, scaled by the `z` differences, tested against
///    `O3D_ERR_BOUND_B`.
/// 3. **Stage 3**, in `orient3d_adapt`: the full exact expansion with the
///    tail corrections, up to 192 components.
/// 4. **The exact integer stage**, `orient3d_exact_integer`, on the original
///    coordinates.
///
/// Stages 2 and 3 run only on input inside the exactness window of
/// `crate::range`, where binary64 arithmetic provably behaves as Shewchuk's
/// unbounded-exponent model does, so his bounds hold. Input that cannot be
/// brought into the window by one exact power-of-two rescale goes straight to
/// stage 4.
///
/// # References
///
/// Shewchuk (1997), `orient3d` and `orient3dadapt`, `predicates.c`.
///
/// Bartels, Fisikopoulos & Weiser (2023), Theorem 1(2).
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
pub fn orient3d(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> Result<Sign, NonFinite> {
    // Strict input-domain check. A non-finite coordinate has no geometric
    // meaning; reject it up front rather than let it reach the adaptive
    // stages, where it could masquerade as a degenerate (Zero) result.
    if !pa
        .iter()
        .chain(&pb)
        .chain(&pc)
        .chain(&pd)
        .all(|v| v.is_finite())
    {
        counters::record(Predicate::Orient3d, Outcome::NonFinite);
        return Err(NonFinite::new());
    }

    // The fast filter on the coordinates as given. Valid for every finite
    // input, so it needs no guard on underflow or overflow: an overflowing
    // intermediate also overflows the magnitude, and the comparison against
    // an infinite bound fails.
    if let Some(sign) = orient3d_filter_raw(pa, pb, pc, pd) {
        counters::record(Predicate::Orient3d, Outcome::Fast);
        return Ok(sign);
    }

    // The expansion stages are exact only inside the window; see
    // `crate::range`. The rescale there is one exact, positive power of two.
    // Positive matters: the determinant is homogeneous of odd degree 3, so a
    // negative factor would invert the result.
    let coords = [
        pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], pc[0], pc[1], pc[2], pd[0], pd[1], pd[2],
    ];
    let (sa, sb, sc, sd) = match prepare(&coords, Window::ORIENT3D) {
        Prepared::AsIs => (pa, pb, pc, pd),
        Prepared::Scaled(s) => {
            let scale = |p: [f64; 3]| [p[0] * s, p[1] * s, p[2] * s];
            (scale(pa), scale(pb), scale(pc), scale(pd))
        }
        Prepared::OutOfWindow => return Ok(orient3d_exact_integer(pa, pb, pc, pd)),
    };

    // Stages 2 and 3. A `Zero` from them is not a certificate: it means the
    // expansion came out zero, which the exact stage has to confirm.
    let result = orient3d_in_window(sa, sb, sc, sd);
    Ok(if result == Sign::Zero {
        orient3d_exact_integer(pa, pb, pc, pd)
    } else {
        counters::record(Predicate::Orient3d, Outcome::Fast);
        result
    })
}

/// Shewchuk's Stage 1 on in-window coordinates, then Stages 2 and 3.
///
/// The caller guarantees that the coordinates are inside
/// [`Window::ORIENT3D`], as given or after an exact rescale. Inside the
/// window no operation of these stages can underflow inexactly or overflow,
/// so Shewchuk's analysis applies to the binary64 execution and his error
/// bounds are valid as written.
///
/// This Stage 1 is Shewchuk's rather than the Bartels form: the raw filter
/// has already run, and inside the window the two certify the same inputs, so
/// the cheaper one is used. Its purpose here is to catch input that the raw
/// filter could not certify only because the rescale had not happened yet.
///
/// A `Sign::Zero` return means no stage could certify a sign; the caller
/// resolves it exactly.
fn orient3d_in_window(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> Sign {
    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let adz = pa[2] - pd[2];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let bdz = pb[2] - pd[2];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];
    let cdz = pc[2] - pd[2];

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    let permanent = ((bdxcdy.abs() + cdxbdy.abs()) * adz.abs())
        + ((cdxady.abs() + adxcdy.abs()) * bdz.abs())
        + ((adxbdy.abs() + bdxady.abs()) * cdz.abs());

    let errbound = O3D_ERR_BOUND_A * permanent;
    if det > errbound || -det > errbound {
        return sign_of(det);
    }

    orient3d_adapt(pa, pb, pc, pd, permanent)
}

// ===========================================================================
// Private: double-buffer merge helper
// ===========================================================================

/// Merges `correction` into the active fin buffer, writing the result into
/// the inactive buffer and returning the new length.
///
/// The caller must toggle `active ^= 1` after each call and before passing
/// `active` to the next call.
///
/// Buffer size: 192 components. The bound is tight:
/// - Stage 2 contributes at most 24 components.
/// - Stage 3 adds at most 56 components per sub-determinant group
///   (bc/ca/ab), giving 3 × 56 = 168.
/// - Total: 24 + 168 = 192.
#[inline(always)]
fn fin_merge_o3d(
    fin1: &mut [f64; 192],
    fin2: &mut [f64; 192],
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
// Private: Stages 2 and 3 (orient3dadapt)
// ===========================================================================

/// Stages 2 and 3 of the adaptive `orient3d` predicate.
///
/// Ports `orient3dadapt` from Shewchuk's `predicates.c` (1997).
///
/// ## Stage 2
///
/// Builds exact 4-component 2×2 sub-determinants:
/// - `bc = bdx·cdy − bdy·cdx`
/// - `ca = cdx·ady − cdy·adx`
/// - `ab = adx·bdy − ady·bdx`
///
/// (each via `two_product` + `two_two_diff`), scales each by the
/// corresponding primary z-difference (`adz`, `bdz`, `cdz`), and sums into
/// a ≤ 24-component fin accumulator. If `|estimate(fin)| ≥ O3D_ERR_BOUND_B ·
/// permanent`, the sign is returned immediately.
///
/// ## Stage 3 (full exact expansion)
///
/// For each of the nine coordinate-difference tails
/// (`adxtail`…`cdztail`), adds the exact correction terms to fin.
/// Each xy-tail perturbs exactly two of the three 2×2 sub-dets; each z-tail
/// scales its entire corresponding sub-det. Cross terms (products of two
/// xy-tails, and z-tail × xy-tail compounds) are also included so that the
/// fin contains the exact scalar determinant.
///
/// The counter `ORIENT3D_EXACT_COUNT` is incremented (test builds only) at
/// the entry to the adaptive path, i.e. whenever Stage 1's fast filter
/// was insufficient and higher-precision arithmetic was required.
// Longer than the crate's 100-line function limit, deliberately. This is
// Shewchuk's `orient3dadapt`, and its later stages share many live expansion
// buffers and lengths: splitting it would need a large state struct or very
// long parameter lists, and would make the correspondence with the published
// algorithm, which is how this function is checked, much harder to follow.
#[allow(clippy::too_many_lines)]
fn orient3d_adapt(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3], permanent: f64) -> Sign {
    // ===========================================================================
    // Stage 2: 2×2 sub-determinants scaled by z-diffs.
    // ===========================================================================

    #[cfg(test)]
    ORIENT3D_EXACT_COUNT.fetch_add(1, Ordering::Relaxed);

    let adx = pa[0] - pd[0];
    let ady = pa[1] - pd[1];
    let adz = pa[2] - pd[2];
    let bdx = pb[0] - pd[0];
    let bdy = pb[1] - pd[1];
    let bdz = pb[2] - pd[2];
    let cdx = pc[0] - pd[0];
    let cdy = pc[1] - pd[1];
    let cdz = pc[2] - pd[2];

    // bc = bdx*cdy − bdy*cdx  (4 components)
    let (bdxcdy1, bdxcdy0) = two_product(bdx, cdy);
    let (cdxbdy1, cdxbdy0) = two_product(cdx, bdy);
    let bc = two_two_diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0);

    // ca = cdx*ady − cdy*adx  (4 components)
    let (cdxady1, cdxady0) = two_product(cdx, ady);
    let (adxcdy1, adxcdy0) = two_product(adx, cdy);
    let ca = two_two_diff(cdxady1, cdxady0, adxcdy1, adxcdy0);

    // ab = adx*bdy − ady*bdx  (4 components)
    let (adxbdy1, adxbdy0) = two_product(adx, bdy);
    let (bdxady1, bdxady0) = two_product(bdx, ady);
    let ab = two_two_diff(adxbdy1, adxbdy0, bdxady1, bdxady0);

    // Scale each sub-det by the corresponding z-diff → at most 8 components each.
    let mut adet = [0.0_f64; 8];
    let adet_len = scale_expansion(&bc, adz, &mut adet);
    let mut bdet = [0.0_f64; 8];
    let bdet_len = scale_expansion(&ca, bdz, &mut bdet);
    let mut cdet = [0.0_f64; 8];
    let cdet_len = scale_expansion(&ab, cdz, &mut cdet);

    // Sum: adet + bdet → ≤ 16; then + cdet → ≤ 24.
    let mut abdet = [0.0_f64; 16];
    let abdet_len = expansion_sum(&adet[..adet_len], &bdet[..bdet_len], &mut abdet);
    let mut fin1 = [0.0_f64; 192];
    let mut fin2 = [0.0_f64; 192];
    let mut fin_len = expansion_sum(&abdet[..abdet_len], &cdet[..cdet_len], &mut fin1);
    let mut active: usize = 0; // 0 → fin1 is current; 1 → fin2 is current

    // Stage 2 estimate: sum of all fin components.
    let det: f64 = fin1[..fin_len].iter().sum();
    if det.abs() >= O3D_ERR_BOUND_B * permanent {
        return sign_of(det);
    }

    // ===========================================================================
    // Compute the nine two-diff tails.
    // ===========================================================================

    let adxtail = two_diff_tail(pa[0], pd[0], adx);
    let adytail = two_diff_tail(pa[1], pd[1], ady);
    let adztail = two_diff_tail(pa[2], pd[2], adz);
    let bdxtail = two_diff_tail(pb[0], pd[0], bdx);
    let bdytail = two_diff_tail(pb[1], pd[1], bdy);
    let bdztail = two_diff_tail(pb[2], pd[2], bdz);
    let cdxtail = two_diff_tail(pc[0], pd[0], cdx);
    let cdytail = two_diff_tail(pc[1], pd[1], cdy);
    let cdztail = two_diff_tail(pc[2], pd[2], cdz);

    // If all tails are zero, Stage 2 is exact.
    if adxtail == 0.0
        && adytail == 0.0
        && adztail == 0.0
        && bdxtail == 0.0
        && bdytail == 0.0
        && bdztail == 0.0
        && cdxtail == 0.0
        && cdytail == 0.0
        && cdztail == 0.0
    {
        return sign_of(det);
    }

    // First-order tail re-test (still native f64).
    // Incorporates all single-tail perturbations to the determinant.
    let errbound = O3D_ERR_BOUND_C * permanent + RESULT_ERR_BOUND * det.abs();
    let det2 = det
        + (adz * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail))
            + adztail * (bdx * cdy - bdy * cdx))
        + (bdz * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail))
            + bdztail * (cdx * ady - cdy * adx))
        + (cdz * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail))
            + cdztail * (adx * bdy - ady * bdx));
    if det2 >= errbound || -det2 >= errbound {
        return sign_of(det2);
    }

    // ===========================================================================
    // Stage 3: full exact fin accumulation.
    //
    // The exact 3×3 determinant can be written:
    //
    //   det = (adz + adztail) · bc_exact
    //       + (bdz + bdztail) · ca_exact
    //       + (cdz + cdztail) · ab_exact
    //
    // where bc_exact = bc + Δbc, etc. Stage 2 already contributed
    // adz·bc + bdz·ca + cdz·ab to fin. Stage 3 adds ALL remaining terms:
    //
    //   • z-tail × sub-det (adztail·bc, bdztail·ca, cdztail·ab)
    //   • xy-tail × primary-z (e.g., bdxtail·cdy·adz)
    //   • compound: z-tail × xy-tail (e.g., adztail·bdxtail·cdy)
    //   • cross-xy within sub-det × primary-z (e.g., bdxtail·cdytail·adz)
    //   • cross-xy × z-tail (e.g., bdxtail·cdytail·adztail)
    //
    // Each two_product term is a 2-component expansion; scale_expansion of
    // a 2-component term by one f64 gives at most 4 components. The buffer
    // holds 192 components total (24 from Stage 2 + 168 from Stage 3).
    // ===========================================================================

    // Reusable scratch buffers.
    let mut tmp8a = [0.0_f64; 8];
    let mut tmp8b = [0.0_f64; 8];

    // Inline helper to merge a scaled result into fin if non-empty.
    // Written as a macro to avoid borrow-checker issues with fin1/fin2/active.
    macro_rules! add_scaled {
        ($comp:expr, $scale:expr, $buf:expr) => {{
            let n = scale_expansion($comp, $scale, $buf);
            if n > 0 {
                fin_len = fin_merge_o3d(&mut fin1, &mut fin2, fin_len, active, &$buf[..n]);
                active ^= 1;
            }
        }};
    }

    // ===========================================================================
    // GROUP A: bc corrections → multiply by (adz + adztail)
    //
    // bc_exact = bc
    //          + bdxtail·cdy  + bdx·cdytail  + bdxtail·cdytail
    //          − bdytail·cdx  − bdy·cdxtail  − bdytail·cdxtail
    // ===========================================================================

    // adztail · bc (scale the entire 4-comp sub-det)
    if adztail != 0.0 {
        let n = scale_expansion(&bc, adztail, &mut tmp8a);
        if n > 0 {
            fin_len = fin_merge_o3d(&mut fin1, &mut fin2, fin_len, active, &tmp8a[..n]);
            active ^= 1;
        }
    }

    // + bdxtail · cdy  (∂bc/∂bdx = cdy)
    if bdxtail != 0.0 {
        let (s1, s0) = two_product(bdxtail, cdy);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, adz, &mut tmp8a);
        if adztail != 0.0 {
            add_scaled!(&tmp2, adztail, &mut tmp8b);
        }
    }

    // + bdx · cdytail  (∂bc/∂cdy = bdx)
    if cdytail != 0.0 {
        let (s1, s0) = two_product(bdx, cdytail);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, adz, &mut tmp8a);
        if adztail != 0.0 {
            add_scaled!(&tmp2, adztail, &mut tmp8b);
        }
        // cross: + bdxtail · cdytail
        if bdxtail != 0.0 {
            let (s1, s0) = two_product(bdxtail, cdytail);
            let tmp2c = [s0, s1];
            add_scaled!(&tmp2c, adz, &mut tmp8a);
            if adztail != 0.0 {
                add_scaled!(&tmp2c, adztail, &mut tmp8b);
            }
        }
    }

    // − bdytail · cdx  (∂bc/∂bdy = −cdx)
    if bdytail != 0.0 {
        let (s1, s0) = two_product(bdytail, cdx);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, adz, &mut tmp8a);
        if adztail != 0.0 {
            add_scaled!(&tmp2, adztail, &mut tmp8b);
        }
    }

    // − bdy · cdxtail  (∂bc/∂cdx = −bdy)
    if cdxtail != 0.0 {
        let (s1, s0) = two_product(bdy, cdxtail);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, adz, &mut tmp8a);
        if adztail != 0.0 {
            add_scaled!(&tmp2, adztail, &mut tmp8b);
        }
        // cross: − bdytail · cdxtail
        if bdytail != 0.0 {
            let (s1, s0) = two_product(bdytail, cdxtail);
            let tmp2c = [-s0, -s1];
            add_scaled!(&tmp2c, adz, &mut tmp8a);
            if adztail != 0.0 {
                add_scaled!(&tmp2c, adztail, &mut tmp8b);
            }
        }
    }

    // ===========================================================================
    // GROUP B: ca corrections → multiply by (bdz + bdztail)
    //
    // ca_exact = ca
    //          + cdxtail·ady  + cdx·adytail  + cdxtail·adytail
    //          − cdytail·adx  − cdy·adxtail  − cdytail·adxtail
    // ===========================================================================

    // bdztail · ca
    if bdztail != 0.0 {
        let n = scale_expansion(&ca, bdztail, &mut tmp8a);
        if n > 0 {
            fin_len = fin_merge_o3d(&mut fin1, &mut fin2, fin_len, active, &tmp8a[..n]);
            active ^= 1;
        }
    }

    // + cdxtail · ady  (∂ca/∂cdx = ady)
    if cdxtail != 0.0 {
        let (s1, s0) = two_product(cdxtail, ady);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, bdz, &mut tmp8a);
        if bdztail != 0.0 {
            add_scaled!(&tmp2, bdztail, &mut tmp8b);
        }
    }

    // + cdx · adytail  (∂ca/∂ady = cdx)
    if adytail != 0.0 {
        let (s1, s0) = two_product(cdx, adytail);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, bdz, &mut tmp8a);
        if bdztail != 0.0 {
            add_scaled!(&tmp2, bdztail, &mut tmp8b);
        }
        // cross: + cdxtail · adytail
        if cdxtail != 0.0 {
            let (s1, s0) = two_product(cdxtail, adytail);
            let tmp2c = [s0, s1];
            add_scaled!(&tmp2c, bdz, &mut tmp8a);
            if bdztail != 0.0 {
                add_scaled!(&tmp2c, bdztail, &mut tmp8b);
            }
        }
    }

    // − cdytail · adx  (∂ca/∂cdy = −adx)
    if cdytail != 0.0 {
        let (s1, s0) = two_product(cdytail, adx);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, bdz, &mut tmp8a);
        if bdztail != 0.0 {
            add_scaled!(&tmp2, bdztail, &mut tmp8b);
        }
    }

    // − cdy · adxtail  (∂ca/∂adx = −cdy)
    if adxtail != 0.0 {
        let (s1, s0) = two_product(cdy, adxtail);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, bdz, &mut tmp8a);
        if bdztail != 0.0 {
            add_scaled!(&tmp2, bdztail, &mut tmp8b);
        }
        // cross: − cdytail · adxtail
        if cdytail != 0.0 {
            let (s1, s0) = two_product(cdytail, adxtail);
            let tmp2c = [-s0, -s1];
            add_scaled!(&tmp2c, bdz, &mut tmp8a);
            if bdztail != 0.0 {
                add_scaled!(&tmp2c, bdztail, &mut tmp8b);
            }
        }
    }

    // ===========================================================================
    // GROUP C: ab corrections → multiply by (cdz + cdztail)
    //
    // ab_exact = ab
    //          + adxtail·bdy  + adx·bdytail  + adxtail·bdytail
    //          − adytail·bdx  − ady·bdxtail  − adytail·bdxtail
    // ===========================================================================

    // cdztail · ab
    if cdztail != 0.0 {
        let n = scale_expansion(&ab, cdztail, &mut tmp8a);
        if n > 0 {
            fin_len = fin_merge_o3d(&mut fin1, &mut fin2, fin_len, active, &tmp8a[..n]);
            active ^= 1;
        }
    }

    // + adxtail · bdy  (∂ab/∂adx = bdy)
    if adxtail != 0.0 {
        let (s1, s0) = two_product(adxtail, bdy);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, cdz, &mut tmp8a);
        if cdztail != 0.0 {
            add_scaled!(&tmp2, cdztail, &mut tmp8b);
        }
    }

    // + adx · bdytail  (∂ab/∂bdy = adx)
    if bdytail != 0.0 {
        let (s1, s0) = two_product(adx, bdytail);
        let tmp2 = [s0, s1];
        add_scaled!(&tmp2, cdz, &mut tmp8a);
        if cdztail != 0.0 {
            add_scaled!(&tmp2, cdztail, &mut tmp8b);
        }
        // cross: + adxtail · bdytail
        if adxtail != 0.0 {
            let (s1, s0) = two_product(adxtail, bdytail);
            let tmp2c = [s0, s1];
            add_scaled!(&tmp2c, cdz, &mut tmp8a);
            if cdztail != 0.0 {
                add_scaled!(&tmp2c, cdztail, &mut tmp8b);
            }
        }
    }

    // − adytail · bdx  (∂ab/∂ady = −bdx)
    if adytail != 0.0 {
        let (s1, s0) = two_product(adytail, bdx);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, cdz, &mut tmp8a);
        if cdztail != 0.0 {
            add_scaled!(&tmp2, cdztail, &mut tmp8b);
        }
    }

    // − ady · bdxtail  (∂ab/∂bdx = −ady)
    if bdxtail != 0.0 {
        let (s1, s0) = two_product(ady, bdxtail);
        let tmp2 = [-s0, -s1];
        add_scaled!(&tmp2, cdz, &mut tmp8a);
        if cdztail != 0.0 {
            add_scaled!(&tmp2, cdztail, &mut tmp8b);
        }
        // cross: − adytail · bdxtail
        if adytail != 0.0 {
            let (s1, s0) = two_product(adytail, bdxtail);
            let tmp2c = [-s0, -s1];
            add_scaled!(&tmp2c, cdz, &mut tmp8a);
            if cdztail != 0.0 {
                add_scaled!(&tmp2c, cdztail, &mut tmp8b);
            }
        }
    }

    // ===========================================================================
    // Return sign of the most-significant (last) component in the active fin.
    // The last component of a non-empty expansion has the same sign as the
    // total, since the components are sorted by magnitude in a zero-eliminated
    // expansion.
    // ===========================================================================
    let fin_slice = if active == 0 {
        &fin1[..fin_len]
    } else {
        &fin2[..fin_len]
    };
    if fin_len == 0 {
        return Sign::Zero;
    }
    sign_of(fin_slice[fin_len - 1])
}

// ===========================================================================
// Stage 4: Exact integer fallback
// ===========================================================================

/// Stage 4: exact integer arithmetic in `WideInt<99>`, 6,336 bits.
///
/// Reached in two cases: the input spans more bits than the exactness window
/// allows, so the expansion stages cannot run on it; or the expansion stages
/// returned `Zero`, which is not a certificate. In practice the second case is
/// almost always an exact zero, and this stage confirms it.
///
/// Converts all twelve coordinates to integers scaled by `2^1074`, which is
/// exact for every finite `f64`, and evaluates the 3×3 determinant with no
/// rounding. The bit-width argument is on the `Orient3dInt` type alias in
/// `exact_int.rs`.
fn orient3d_exact_integer(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> Sign {
    counters::record(Predicate::Orient3d, Outcome::Exact);
    use crate::exact_int::f64_to_orient3d_int;

    let ax = f64_to_orient3d_int(pa[0]);
    let ay = f64_to_orient3d_int(pa[1]);
    let az = f64_to_orient3d_int(pa[2]);
    let bx = f64_to_orient3d_int(pb[0]);
    let by_ = f64_to_orient3d_int(pb[1]);
    let bz = f64_to_orient3d_int(pb[2]);
    let cx = f64_to_orient3d_int(pc[0]);
    let cy = f64_to_orient3d_int(pc[1]);
    let cz = f64_to_orient3d_int(pc[2]);
    let dx = f64_to_orient3d_int(pd[0]);
    let dy = f64_to_orient3d_int(pd[1]);
    let dz = f64_to_orient3d_int(pd[2]);

    // Coordinate differences relative to pd.
    let adx = ax - dx;
    let ady = ay - dy;
    let adz = az - dz;
    let bdx = bx - dx;
    let bdy = by_ - dy;
    let bdz = bz - dz;
    let cdx = cx - dx;
    let cdy = cy - dy;
    let cdz = cz - dz;

    // 3×3 determinant:
    //   | adx  ady  adz |
    //   | bdx  bdy  bdz | = adz·bc + bdz·ca + cdz·ab
    //   | cdx  cdy  cdz |
    let bc = bdx * cdy - bdy * cdx;
    let ca = cdx * ady - cdy * adx;
    let ab = adx * bdy - ady * bdx;
    let det = adz * bc + bdz * ca + cdz * ab;

    if det.is_positive() {
        Sign::Positive
    } else if det.is_negative() {
        Sign::Negative
    } else {
        // Observability only (`stage-counters`): records how much of the exact
        // path is spent confirming exact zeros.
        counters::record(Predicate::Orient3d, Outcome::ExactZero);
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

    /// A clearly positive orientation: pd = (0,0,−1) lies below the plane
    /// through pa=(0,0,0), pb=(1,0,0), pc=(0,1,0), which appear
    /// counterclockwise seen from above.
    #[test]
    fn orient3d_clearly_positive() {
        let pa = [0.0_f64, 0.0, 0.0];
        let pb = [1.0_f64, 0.0, 0.0];
        let pc = [0.0_f64, 1.0, 0.0];
        let pd = [0.0_f64, 0.0, -1.0];
        assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
    }

    /// A clearly negative orientation: pd = (0,0,1) lies above the same plane.
    #[test]
    fn orient3d_clearly_negative() {
        let pa = [0.0_f64, 0.0, 0.0];
        let pb = [1.0_f64, 0.0, 0.0];
        let pc = [0.0_f64, 1.0, 0.0];
        let pd = [0.0_f64, 0.0, 1.0];
        assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Negative));
    }

    /// Exactly coplanar: all four points are in the plane z = 0, so every `z`
    /// difference is zero and the 3×3 determinant is zero exactly.
    #[test]
    fn orient3d_exactly_coplanar() {
        let pa = [0.0_f64, 0.0, 0.0];
        let pb = [1.0_f64, 0.0, 0.0];
        let pc = [0.5_f64, 1.0, 0.0];
        let pd = [0.25_f64, 0.5, 0.0];
        assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Zero));
    }

    /// The adaptive path fires on a nearly coplanar tetrahedron and returns
    /// the right sign.
    ///
    /// pa, pb, pc lie on the plane z = 1 + x + 2y, which at (1, 1) gives
    /// z = 4. pd is at z = 4 + 8ε with ε = 2^-52, exactly `2^-49` above the
    /// plane, so the exact determinant is −2^−49 ≈ −1.78e−15: negative,
    /// because pd is above. The fast filter's bound here is about 4.7e−15,
    /// more than twice |det|, so it cannot certify the sign and
    /// `orient3d_adapt` must run.
    #[test]
    fn orient3d_exact_count_increments_on_near_coplanar() {
        let pa = [0.0_f64, 0.0, 1.0];
        let pb = [1.0_f64, 0.0, 2.0];
        let pc = [0.0_f64, 1.0, 3.0];
        let pd = [1.0_f64, 1.0, 4.0 + 8.0 * f64::EPSILON];

        let before = ORIENT3D_EXACT_COUNT.load(Ordering::Relaxed);
        let result = orient3d(pa, pb, pc, pd);
        let after = ORIENT3D_EXACT_COUNT.load(Ordering::Relaxed);

        assert_eq!(result, Ok(Sign::Negative), "pd is above the plane, det < 0");
        assert!(
            after > before,
            "Stage 3 counter should have been incremented for near-coplanar input"
        );
    }

    /// Non-finite inputs are rejected with `Err(NonFinite)` (not a panic, and
    /// not a plausible-looking `Sign`).
    #[test]
    fn orient3d_non_finite_inputs_return_err() {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        assert_eq!(
            orient3d(
                [inf, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ),
            Err(NonFinite::new()),
            "a +inf coordinate must be rejected"
        );
        assert_eq!(
            orient3d(
                [nan, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ),
            Err(NonFinite::new()),
            "a NaN coordinate must be rejected"
        );
        assert_eq!(
            orient3d(
                [f64::NEG_INFINITY, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ),
            Err(NonFinite::new()),
            "a -inf coordinate must be rejected"
        );
    }

    /// All-finite inputs whose coordinate differences overflow f64 to infinity
    /// still return the exact sign (`Ok`), never a premature `Sign::Zero`.
    /// Here the four points are a non-degenerate tetrahedron near `f64::MAX`;
    /// the raw differences overflow, so the result must come from the normalized
    /// adaptive path or exact Stage 4.
    #[test]
    fn orient3d_finite_overflow_is_not_zero() {
        let m = f64::MAX;
        let pa = [m, 0.0, 0.0];
        let pb = [0.0, m, 0.0];
        let pc = [0.0, 0.0, m];
        let pd = [-m, -m, -m];
        assert!(
            pa[0] - pd[0] == f64::INFINITY,
            "precondition: difference overflows"
        );
        assert_eq!(
            orient3d(pa, pb, pc, pd),
            Ok(Sign::Positive),
            "finite non-degenerate input must not collapse to Zero on overflow"
        );
    }
}
