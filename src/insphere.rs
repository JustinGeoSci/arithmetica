// src/insphere.rs

//! Adaptive `insphere` predicate: circumsphere orientation test for five 3D points.
//!
//! Ports `insphere` and `insphereadapt` from Shewchuk (1997) `predicates.c`,
//! with an underflow-protected fast filter in front and an exact integer stage
//! behind, for input the expansion stages cannot certify.

#[cfg(test)]
use core::sync::atomic::{AtomicUsize, Ordering};

#[cfg(test)]
pub(crate) static INSPHERE_EXACT_COUNT: AtomicUsize = AtomicUsize::new(0);

use crate::counters::{self, Outcome, Predicate};
use crate::{NonFinite, Sign};

// `ISP_ERR_BOUND_C` and `RESULT_ERR_BOUND` are not imported: they belong to a
// first-order correction stage, and `insphere_adapt` has none (see its doc).
// A sound one would have to include the lift corrections, not only the
// coordinate tails.
use crate::expansion::{
    ISP_ERR_BOUND_A, ISP_ERR_BOUND_B, expansion_sum, scale_expansion, two_product,
};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{sign_of, two_diff_tail, two_two_diff};

type Point3 = [f64; 3];

// ===========================================================================
// Private: the fast filter
// ===========================================================================

/// The plain `f64` determinant together with the underflow-protected
/// magnitude of Bartels et al. 2023, rules `R3`, `R6,UFP` and `R7,UFP`.
///
/// Returns `(det, bound)`. The caller certifies the sign of `det` when
/// `|det| > bound` with a strict comparison, which is valid for **every**
/// finite input (Theorem 1(2)) and not only for an evaluation free of
/// underflow. See [`crate::orient2d`]'s filter for why Shewchuk's form is not
/// enough and what the protection does.
///
/// # The magnitude
///
/// Built term for term alongside the determinant, in the same association
/// order, because the paper's rule for a sum derives its bound from the
/// bounds of the two operands it is formed from. The determinant is
/// `(dlift·abc − clift·dab) + (blift·cda − alift·bcd)`, so the four row terms
/// are summed `(t_d + t_c) + (t_b + t_a)`, and inside each 3×3 minor the
/// three products are summed left to right, as `abc = aez·bc − bez·ac +
/// cez·ab` is.
///
/// Each product gets a `u_N`; each coordinate difference gets none, because a
/// binary64 subtraction with gradual underflow is exact when its result is
/// subnormal. A square needs no `abs`.
///
/// This pairwise sum of the four row terms is at least as conservative as a
/// left-to-right sum would be: it passes each term through two additions
/// rather than up to three, and a computed magnitude can only lose to
/// rounding at `(1 − ε)` per addition. The order matches this crate's own
/// determinant, which is what the filter's validity depends on; it does not
/// depend on the order Shewchuk's `insphere()` uses.
///
/// # Constants
///
/// Shewchuk's `ISP_ERR_BOUND_A`, `(16 + 224ε)ε`, satisfies the paper's
/// requirement for this expression tree, which is anything above
/// `(15 + 150ε)ε`.
fn insphere_filter_raw(pa: Point3, pb: Point3, pc: Point3, pd: Point3, pe: Point3) -> (f64, f64) {
    let aex = pa[0] - pe[0];
    let aey = pa[1] - pe[1];
    let aez = pa[2] - pe[2];
    let bex = pb[0] - pe[0];
    let bey = pb[1] - pe[1];
    let bez = pb[2] - pe[2];
    let cex = pc[0] - pe[0];
    let cey = pc[1] - pe[1];
    let cez = pc[2] - pe[2];
    let dex = pd[0] - pe[0];
    let dey = pd[1] - pe[1];
    let dez = pd[2] - pe[2];

    // The six 2x2 minors, and the magnitude of each.
    let minor = |p: f64, q: f64, r: f64, s: f64| p * q - r * s;
    let minor_mag = |p: f64, q: f64, r: f64, s: f64| ((p * q).abs() + U_N) + ((r * s).abs() + U_N);
    let ab = minor(aex, bey, bex, aey);
    let bc = minor(bex, cey, cex, bey);
    let cd = minor(cex, dey, dex, cey);
    let da = minor(dex, aey, aex, dey);
    let ac = minor(aex, cey, cex, aey);
    let bd = minor(bex, dey, dex, bey);
    let m_ab = minor_mag(aex, bey, bex, aey);
    let m_bc = minor_mag(bex, cey, cex, bey);
    let m_cd = minor_mag(cex, dey, dex, cey);
    let m_da = minor_mag(dex, aey, aex, dey);
    let m_ac = minor_mag(aex, cey, cex, aey);
    let m_bd = minor_mag(bex, dey, dex, bey);

    // The four 3x3 minors, and the magnitude of each: three products of a `z`
    // difference with a 2x2 minor, summed left to right.
    let three_mag = |z1: f64, n1: f64, z2: f64, n2: f64, z3: f64, n3: f64| {
        ((z1.abs() * n1 + U_N) + (z2.abs() * n2 + U_N)) + (z3.abs() * n3 + U_N)
    };
    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;
    let m_abc = three_mag(aez, m_bc, bez, m_ac, cez, m_ab);
    let m_bcd = three_mag(bez, m_cd, cez, m_bd, dez, m_bc);
    let m_cda = three_mag(cez, m_da, dez, m_ac, aez, m_cd);
    let m_dab = three_mag(dez, m_ab, aez, m_bd, bez, m_da);

    // The four lifts, and the magnitude of each.
    let lift = |x: f64, y: f64, z: f64| x * x + y * y + z * z;
    let lift_mag = |x: f64, y: f64, z: f64| ((x * x + U_N) + (y * y + U_N)) + (z * z + U_N);
    let alift = lift(aex, aey, aez);
    let blift = lift(bex, bey, bez);
    let clift = lift(cex, cey, cez);
    let dlift = lift(dex, dey, dez);

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    let t_a = lift_mag(aex, aey, aez) * m_bcd + U_N;
    let t_b = lift_mag(bex, bey, bez) * m_cda + U_N;
    let t_c = lift_mag(cex, cey, cez) * m_dab + U_N;
    let t_d = lift_mag(dex, dey, dez) * m_abc + U_N;
    let bound = ISP_ERR_BOUND_A * ((t_d + t_c) + (t_b + t_a)) + U_S;

    (det, bound)
}

// ===========================================================================
// Private: Stage 1 evaluation, then the public predicate
// ===========================================================================

/// Stage 1 of the adaptive evaluation: the plain `f64` determinant `det`
/// of the lifted 4x4 matrix (relative to `pe`) together with its permanent,
/// the sum of the absolute values of every term, which scales the error bound
/// `ISP_ERR_BOUND_A * permanent`.
///
/// This runs on in-window coordinates only, where Shewchuk's bound is valid.
/// On raw coordinates it would not be: a product that underflows carries an
/// absolute error the bound does not cover. The filter on the raw coordinates
/// is `insphere_filter_raw`, which uses the underflow-protected bound.
fn insphere_stage1(
    pa: [f64; 3],
    pb: [f64; 3],
    pc: [f64; 3],
    pd: [f64; 3],
    pe: [f64; 3],
) -> (f64, f64) {
    let aex = pa[0] - pe[0];
    let aey = pa[1] - pe[1];
    let aez = pa[2] - pe[2];
    let bex = pb[0] - pe[0];
    let bey = pb[1] - pe[1];
    let bez = pb[2] - pe[2];
    let cex = pc[0] - pe[0];
    let cey = pc[1] - pe[1];
    let cez = pc[2] - pe[2];
    let dex = pd[0] - pe[0];
    let dey = pd[1] - pe[1];
    let dez = pd[2] - pe[2];

    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;

    let abc = aez * bc - bez * ac + cez * ab;
    let bcd = bez * cd - cez * bd + dez * bc;
    let cda = cez * da + dez * ac + aez * cd;
    let dab = dez * ab + aez * bd + bez * da;

    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    let det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);

    let abc_p = aez.abs() * (bex.abs() * cey.abs() + cex.abs() * bey.abs())
        + bez.abs() * (aex.abs() * cey.abs() + cex.abs() * aey.abs())
        + cez.abs() * (aex.abs() * bey.abs() + bex.abs() * aey.abs());
    let bcd_p = bez.abs() * (cex.abs() * dey.abs() + dex.abs() * cey.abs())
        + cez.abs() * (bex.abs() * dey.abs() + dex.abs() * bey.abs())
        + dez.abs() * (bex.abs() * cey.abs() + cex.abs() * bey.abs());
    let cda_p = cez.abs() * (dex.abs() * aey.abs() + aex.abs() * dey.abs())
        + dez.abs() * (aex.abs() * cey.abs() + cex.abs() * aey.abs())
        + aez.abs() * (cex.abs() * dey.abs() + dex.abs() * cey.abs());
    let dab_p = dez.abs() * (aex.abs() * bey.abs() + bex.abs() * aey.abs())
        + aez.abs() * (bex.abs() * dey.abs() + dex.abs() * bey.abs())
        + bez.abs() * (dex.abs() * aey.abs() + aex.abs() * dey.abs());
    let permanent =
        alift.abs() * bcd_p + blift.abs() * cda_p + clift.abs() * dab_p + dlift.abs() * abc_p;

    (det, permanent)
}

/// Tests whether `pe` lies inside the circumsphere of `pa`, `pb`, `pc`, `pd`.
///
/// Returns [`Sign::Positive`] if `pe` is inside the circumsphere, when the
/// tetrahedron `(pa, pb, pc, pd)` has positive orientation as determined by
/// [`crate::orient3d`]. Returns [`Sign::Negative`] if `pe` is outside.
/// Returns [`Sign::Zero`] if `pe` lies exactly on the circumsphere, or if
/// the four reference points are coplanar.
///
/// # Mathematical definition
///
/// Evaluates the sign of the 5×5 determinant:
/// ```text
/// | pa.x  pa.y  pa.z  pa.x²+pa.y²+pa.z²  1 |
/// | pb.x  pb.y  pb.z  pb.x²+pb.y²+pb.z²  1 |
/// | pc.x  pc.y  pc.z  pc.x²+pc.y²+pc.z²  1 |
/// | pd.x  pd.y  pd.z  pd.x²+pd.y²+pd.z²  1 |
/// | pe.x  pe.y  pe.z  pe.x²+pe.y²+pe.z²  1 |
/// ```
/// which equals the sign of the 4×4 determinant of coordinate differences
/// relative to `pe` together with the lifted lift column:
/// ```text
/// | aex  aey  aez  aex²+aey²+aez² |
/// | bex  bey  bez  bex²+bey²+bez² |
/// | cex  cey  cez  cex²+cey²+cez² |
/// | dex  dey  dez  dex²+dey²+dez² |
/// ```
/// where `aex = pa.x - pe.x`, etc.
///
/// # Non-finite inputs
///
/// If any input coordinate is NaN or infinite, the predicate returns
/// [`Err(NonFinite)`](NonFinite): a non-finite coordinate has no geometric
/// meaning, so there is no correct [`Sign`] to return. It is reported as an
/// error rather than a plausible-looking `Sign::Zero`.
///
/// For *finite* inputs the function always returns `Ok` with the exact sign,
/// even when intermediate coordinate differences or the lift column overflow
/// `f64` to infinity: such inputs are resolved by the exact Stage 4 integer
/// path, never by a premature `Sign::Zero`.
///
/// # Robustness: four stages
///
/// 1. **The fast filter**, `insphere_filter_raw`: the determinant in native
///    `f64`, with the underflow-protected error bound of Bartels et al. 2023.
///    Valid for every finite input, so a certificate here is final.
/// 2. **Stage 2**, in `insphere_adapt`: exact 2×2 and 3×3 sub-determinants,
///    each scaled by its exact lift, summed into an expansion of up to 1152
///    components and tested against `ISP_ERR_BOUND_B`; plus the tails-zero
///    exit, which reads the sign of an exactly correct expansion.
/// 3. **No Stage 3.** Input that Stage 2 cannot certify goes to stage 4; see
///    `insphere_adapt` for why.
/// 4. **The exact integer stage**, `insphere_exact_integer`, in fixed-width
///    `WideInt` arithmetic on the original coordinates.
///
/// Stage 2 runs only on input inside the exactness window of
/// `crate::range`, where binary64 arithmetic provably behaves as Shewchuk's
/// unbounded-exponent model does, so his bounds hold. Input that cannot be
/// brought into the window by one exact power-of-two rescale goes straight to
/// stage 4.
///
/// # References
///
/// Shewchuk (1997), `insphere` and `insphereadapt`, `predicates.c`.
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
#[must_use = "the insphere result is the return value; discarding it is almost certainly a bug"]
pub fn insphere(
    pa: [f64; 3],
    pb: [f64; 3],
    pc: [f64; 3],
    pd: [f64; 3],
    pe: [f64; 3],
) -> Result<Sign, NonFinite> {
    // Strict input-domain check. A non-finite coordinate has no geometric
    // meaning, so there is no correct `Sign`; reject it up front. Every stage
    // below may then assume finite inputs, so a non-finite *intermediate* (an
    // overflowing coordinate difference or lift) is a recoverable artifact of
    // a finite input, resolved by the exact Stage 4 integer path, never a
    // reason to return Zero.
    if !pa
        .iter()
        .chain(&pb)
        .chain(&pc)
        .chain(&pd)
        .chain(&pe)
        .all(|v| v.is_finite())
    {
        counters::record(Predicate::Insphere, Outcome::NonFinite);
        return Err(NonFinite::new());
    }

    // The fast filter on the coordinates as given, with the
    // underflow-protected bound: valid for every finite input, so it needs no
    // guard on underflow or overflow. An overflowing intermediate also
    // overflows the magnitude, because every term of the magnitude is
    // non-negative and nothing can cancel it back into range, so the
    // comparison against an infinite bound fails and the input is passed on.
    let (det_raw, bound_raw) = insphere_filter_raw(pa, pb, pc, pd, pe);
    if det_raw > bound_raw || -det_raw > bound_raw {
        counters::record(Predicate::Insphere, Outcome::Fast);
        return Ok(sign_of(det_raw));
    }

    // The expansion stages are exact only inside the window; see
    // `crate::range`. The rescale there is one exact, positive power of two.
    // Positive matters: the determinant is homogeneous of odd degree 5, so a
    // negative factor would invert the result.
    let coords = [
        pa[0], pa[1], pa[2], pb[0], pb[1], pb[2], pc[0], pc[1], pc[2], pd[0], pd[1], pd[2], pe[0],
        pe[1], pe[2],
    ];
    let (sa, sb, sc, sd, se) = match prepare(&coords, Window::INSPHERE) {
        Prepared::AsIs => (pa, pb, pc, pd, pe),
        Prepared::Scaled(s) => {
            let scale = |p: [f64; 3]| [p[0] * s, p[1] * s, p[2] * s];
            (scale(pa), scale(pb), scale(pc), scale(pd), scale(pe))
        }
        Prepared::OutOfWindow => return Ok(insphere_exact_integer(pa, pb, pc, pd, pe)),
    };

    // Shewchuk's Stage 1 on the in-window coordinates. This catches input
    // that the raw filter could not certify only because the rescale had not
    // happened yet.
    let (det, permanent) = insphere_stage1(sa, sb, sc, sd, se);
    if det > ISP_ERR_BOUND_A * permanent || -det > ISP_ERR_BOUND_A * permanent {
        counters::record(Predicate::Insphere, Outcome::Fast);
        return Ok(sign_of(det));
    }

    // Stages 2 and 3. A `Zero` from them is not a certificate: it means the
    // expansion came out zero, which the exact stage has to confirm.
    let result = insphere_adapt(sa, sb, sc, sd, se, permanent);
    Ok(if result == Sign::Zero {
        insphere_exact_integer(pa, pb, pc, pd, pe)
    } else {
        counters::record(Predicate::Insphere, Outcome::Fast);
        result
    })
}

// ===========================================================================
// Private: exact lift scaling and expansion sign
// ===========================================================================

/// Multiplies a 3×3 sub-determinant expansion by the **exact** lift
/// `sign · (vx² + vy² + vz²)`.
///
/// # Why the lift is applied by double scaling
///
/// The obvious implementation rounds the lift to an `f64` first
/// (`let lift = vx*vx + vy*vy + vz*vz;`) and scales the expansion by it once.
/// That is wrong: `scale_expansion` is exact, so the product carries no
/// rounding of its own, but the *lift* was already rounded before it arrived.
/// The resulting expansion is exactly the determinant of a slightly different
/// problem, and nothing downstream can recover the difference.
///
/// Scaling twice by each component instead, `scale(scale(e, vx), vx)`, gives
/// exactly `vx²·e`, because each `scale_expansion` is exact and `vx` is an
/// exact `f64`. Summing the three exact products gives exactly
/// `(vx² + vy² + vz²)·e` with no rounding anywhere. This is what Shewchuk's
/// `insphereadapt` does, and what `incircle_adapt` in this crate does.
///
/// The distinction matters because `ISP_ERR_BOUND_B` and `ISP_ERR_BOUND_C` are
/// derived for the exact-lift formulation. Applied to a pre-rounded lift they
/// certify a sign that has not been established.
///
/// `sign` must be exactly `1.0` or `-1.0`; it is folded into the second scaling
/// so the negation is exact rather than a separate pass.
///
/// ## Buffer contract
///
/// `e` holds at most 24 components (a 3×3 sub-determinant), so each single
/// scaling yields ≤48, each double scaling ≤96, and the sum of the three
/// ≤288. `out` must therefore have length at least `12 * e.len()`.
fn scale_by_exact_lift(e: &[f64], vx: f64, vy: f64, vz: f64, sign: f64, out: &mut [f64]) -> usize {
    debug_assert!(e.len() <= 24, "lift scaling expects a 3x3 sub-determinant");
    // `float_cmp` is suppressed deliberately: `sign` is a caller-supplied
    // literal multiplier (+1.0 or -1.0), not a computed quantity, so an exact
    // comparison is the assertion being made. A tolerance would weaken it.
    #[allow(clippy::float_cmp)]
    {
        debug_assert!(sign == 1.0 || sign == -1.0, "sign must be exactly +/-1");
    }
    debug_assert!(
        out.len() >= 12 * e.len(),
        "lift scaling output buffer too small"
    );

    let mut half = [0.0_f64; 48];
    let mut sq_x = [0.0_f64; 96];
    let mut sq_y = [0.0_f64; 96];
    let mut sq_z = [0.0_f64; 96];

    // Each pair of scalings is exact, so sq_* holds exactly sign * v*^2 * e.
    let n = scale_expansion(e, vx, &mut half);
    let x_len = scale_expansion(&half[..n], sign * vx, &mut sq_x);
    let n = scale_expansion(e, vy, &mut half);
    let y_len = scale_expansion(&half[..n], sign * vy, &mut sq_y);
    let n = scale_expansion(e, vz, &mut half);
    let z_len = scale_expansion(&half[..n], sign * vz, &mut sq_z);

    let mut xy = [0.0_f64; 192];
    let xy_len = expansion_sum(&sq_x[..x_len], &sq_y[..y_len], &mut xy);
    expansion_sum(&xy[..xy_len], &sq_z[..z_len], out)
}

/// Returns the exact sign of a non-overlapping, zero-eliminated expansion.
///
/// Every operation in `expansion.rs` maintains the non-overlapping invariant
/// and compacts zeros out of its output, so the last live component is the
/// most significant and its sign is the sign of the whole sum. An empty
/// expansion is exactly zero.
///
/// This is exact and needs no error bound, unlike summing the components into
/// an `f64` and reading that, which is what `estimate()` does and why
/// `estimate()`'s result always has to be tested against a bound.
#[inline]
fn expansion_sign(e: &[f64]) -> Sign {
    match e.last() {
        None => Sign::Zero,
        Some(&top) => sign_of(top),
    }
}

// ===========================================================================
// Private: Stages 2 and 3 (insphereadapt)
// ===========================================================================

/// Builds the four exact 3×3 sub-determinants of the `insphere` matrix.
///
/// Takes the four coordinate-difference vectors relative to `pe` and returns
/// `[abc, bcd, cda, dab]`, each as a ≤24-component expansion with its live
/// length. Every step is exact: the six 2×2 sub-determinants come from
/// `two_product` / `two_two_diff`, and each 3×3 is an exact scaling and sum of
/// those.
///
/// Split out of `insphere_adapt` to keep it within the workspace's 100-line
/// function limit. The split is clean because these four expansions are the
/// only values that cross the boundary; nothing here reads an error bound or
/// decides a sign.
fn insphere_sub_determinants(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
) -> [([f64; 24], usize); 4] {
    let (aex, aey, aez) = (a[0], a[1], a[2]);
    let (bex, bey, bez) = (b[0], b[1], b[2]);
    let (cex, cey, cez) = (c[0], c[1], c[2]);
    let (dex, dey, dez) = (d[0], d[1], d[2]);

    // Exact 2×2 sub-dets (4 components each).
    let (axby1, axby0) = two_product(aex, bey);
    let (bxay1, bxay0) = two_product(bex, aey);
    let ab = two_two_diff(axby1, axby0, bxay1, bxay0);

    let (bxcy1, bxcy0) = two_product(bex, cey);
    let (cxby1, cxby0) = two_product(cex, bey);
    let bc = two_two_diff(bxcy1, bxcy0, cxby1, cxby0);

    let (cxdy1, cxdy0) = two_product(cex, dey);
    let (dxcy1, dxcy0) = two_product(dex, cey);
    let cd = two_two_diff(cxdy1, cxdy0, dxcy1, dxcy0);

    let (dxay1, dxay0) = two_product(dex, aey);
    let (axdy1, axdy0) = two_product(aex, dey);
    let da = two_two_diff(dxay1, dxay0, axdy1, axdy0);

    let (axcy1, axcy0) = two_product(aex, cey);
    let (cxay1, cxay0) = two_product(cex, aey);
    let ac = two_two_diff(axcy1, axcy0, cxay1, cxay0);

    let (bxdy1, bxdy0) = two_product(bex, dey);
    let (dxby1, dxby0) = two_product(dex, bey);
    let bd = two_two_diff(bxdy1, bxdy0, dxby1, dxby0);

    // Each 3×3 is `z·(2x2) ± z·(2x2) + z·(2x2)`, built exactly.
    let mut t8a = [0.0_f64; 8];
    let mut t8b = [0.0_f64; 8];
    let mut t16 = [0.0_f64; 16];

    let mut build = |p: &[f64; 4], sp: f64, q: &[f64; 4], sq: f64, r: &[f64; 4], sr: f64| {
        let mut out = [0.0_f64; 24];
        let n = scale_expansion(p, sp, &mut t8a);
        let m = scale_expansion(q, sq, &mut t8b);
        let nm = expansion_sum(&t8a[..n], &t8b[..m], &mut t16);
        let n = scale_expansion(r, sr, &mut t8a);
        let len = expansion_sum(&t16[..nm], &t8a[..n], &mut out);
        (out, len)
    };

    [
        //   abc = aez·bc − bez·ac + cez·ab
        build(&bc, aez, &ac, -bez, &ab, cez),
        //   bcd = bez·cd − cez·bd + dez·bc
        build(&cd, bez, &bd, -cez, &bc, dez),
        //   cda = cez·da + dez·ac + aez·cd
        build(&da, cez, &ac, dez, &cd, aez),
        //   dab = dez·ab + aez·bd + bez·da
        build(&ab, dez, &bd, aez, &da, bez),
    ]
}

/// Stages 2 and 3 of the adaptive `insphere` predicate.
///
/// ## Stage 2
///
/// Builds exact 4-component 2×2 sub-determinants for all six pairs of points
/// (ab, bc, cd, da, ac, bd), then exact 24-component 3×3 sub-determinants
/// (abc, bcd, cda, dab), scales each by its **exact** lift via
/// [`scale_by_exact_lift`], and sums the four into a ≤1152-component fin.
///
/// Because every step is exact, `fin` is exactly the determinant of the
/// *approximate* coordinate differences `aex…dez`. The only remaining error
/// relative to the true determinant comes from the differences themselves, and
/// that error is zero exactly when all twelve `two_diff_tail` values are zero.
///
/// `estimate(fin)` is then tested against `ISP_ERR_BOUND_B · permanent`. That
/// bound is Shewchuk's and is valid here, because it is derived for the
/// exact-lift formulation this function implements.
///
/// ## The tails-zero exit
///
/// When all twelve coordinate tails are zero the differences are exact, so
/// `fin` is exactly the true determinant and its sign can be read directly
/// from the expansion by [`expansion_sign`]; no error bound is involved, and
/// none is needed.
///
/// Reading the sign of the rounded `estimate(fin)` here instead would not be
/// sound: zero tails make the expansion exact, but its rounded sum is not.
///
/// ## No Stage 3
///
/// Shewchuk's `insphereadapt` follows Stage 2 with a first-order correction
/// for the coordinate-difference tails. A sound version must also correct for
/// the lift terms, and with exact lifts the Stage 2 expansion is about six
/// times larger than in the pre-rounded form, which makes that correction
/// expensive in buffer space as well as code. Input that Stage 2 cannot
/// certify therefore goes to the exact integer stage, which is always correct
/// and, for this predicate, rarely reached.
///
/// # Correctness guarantee
///
/// Every `Sign` this function returns is correct. There are exactly two
/// returns: the Stage 2 estimate guarded by Shewchuk's bound, and the
/// tails-zero exit which reads an exactly-correct expansion. Everything else
/// returns `Sign::Zero` and is resolved by the exact Stage 4 integer path.
fn insphere_adapt(
    pa: [f64; 3],
    pb: [f64; 3],
    pc: [f64; 3],
    pd: [f64; 3],
    pe: [f64; 3],
    permanent: f64,
) -> Sign {
    // ===========================================================================
    // Stage 2: exact 2×2 sub-dets, exact 3×3 sub-dets, exact lift scaling.
    // ===========================================================================

    let aex = pa[0] - pe[0];
    let aey = pa[1] - pe[1];
    let aez = pa[2] - pe[2];
    let bex = pb[0] - pe[0];
    let bey = pb[1] - pe[1];
    let bez = pb[2] - pe[2];
    let cex = pc[0] - pe[0];
    let cey = pc[1] - pe[1];
    let cez = pc[2] - pe[2];
    let dex = pd[0] - pe[0];
    let dey = pd[1] - pe[1];
    let dez = pd[2] - pe[2];

    let [
        (abc_exp, abc_len),
        (bcd_exp, bcd_len),
        (cda_exp, cda_len),
        (dab_exp, dab_len),
    ] = insphere_sub_determinants(
        [aex, aey, aez],
        [bex, bey, bez],
        [cex, cey, cez],
        [dex, dey, dez],
    );

    // Scale each 3×3 sub-det by its EXACT lift → ≤288 components each.
    // det = dlift·abc − clift·dab + blift·cda − alift·bcd   (Formula A)
    let mut a288 = [0.0_f64; 288];
    let mut b288 = [0.0_f64; 288];
    let mut c288 = [0.0_f64; 288];
    let mut d288 = [0.0_f64; 288];
    let a_len = scale_by_exact_lift(&abc_exp[..abc_len], dex, dey, dez, 1.0, &mut a288);
    let b_len = scale_by_exact_lift(&dab_exp[..dab_len], cex, cey, cez, -1.0, &mut b288);
    let c_len = scale_by_exact_lift(&cda_exp[..cda_len], bex, bey, bez, 1.0, &mut c288);
    let d_len = scale_by_exact_lift(&bcd_exp[..bcd_len], aex, aey, aez, -1.0, &mut d288);

    // Sum the four into fin: ≤1152 components.
    let mut fin = [0.0_f64; 1152];
    let mut t576a = [0.0_f64; 576];
    let mut t576b = [0.0_f64; 576];
    let ab_len = expansion_sum(&a288[..a_len], &b288[..b_len], &mut t576a);
    let cd_len = expansion_sum(&c288[..c_len], &d288[..d_len], &mut t576b);
    let fin_len = expansion_sum(&t576a[..ab_len], &t576b[..cd_len], &mut fin);

    debug_assert!(
        fin_len <= fin.len(),
        "Stage 2 expansion length must stay within its buffer"
    );

    // `fin` is now EXACTLY the determinant of the approximate differences.
    // Shewchuk's Stage 2 bound applies, because the lift is exact.
    let det: f64 = fin[..fin_len].iter().sum();
    if det.abs() >= ISP_ERR_BOUND_B * permanent {
        return sign_of(det);
    }

    // ===========================================================================
    // The twelve coordinate-difference tails.
    // ===========================================================================

    let aextail = two_diff_tail(pa[0], pe[0], aex);
    let aeytail = two_diff_tail(pa[1], pe[1], aey);
    let aeztail = two_diff_tail(pa[2], pe[2], aez);
    let bextail = two_diff_tail(pb[0], pe[0], bex);
    let beytail = two_diff_tail(pb[1], pe[1], bey);
    let beztail = two_diff_tail(pb[2], pe[2], bez);
    let cextail = two_diff_tail(pc[0], pe[0], cex);
    let ceytail = two_diff_tail(pc[1], pe[1], cey);
    let ceztail = two_diff_tail(pc[2], pe[2], cez);
    let dextail = two_diff_tail(pd[0], pe[0], dex);
    let deytail = two_diff_tail(pd[1], pe[1], dey);
    let deztail = two_diff_tail(pd[2], pe[2], dez);

    if aextail == 0.0
        && aeytail == 0.0
        && aeztail == 0.0
        && bextail == 0.0
        && beytail == 0.0
        && beztail == 0.0
        && cextail == 0.0
        && ceytail == 0.0
        && ceztail == 0.0
        && dextail == 0.0
        && deytail == 0.0
        && deztail == 0.0
    {
        // The differences are exact, and every operation building `fin` is
        // exact, so `fin` is exactly the true determinant. Read its sign from
        // the expansion rather than from the rounded `det` sum: the expansion
        // is exact, `det` is not.
        return expansion_sign(&fin[..fin_len]);
    }

    // Some coordinate difference is inexact. Correcting for that would need a
    // first-order correction including the lift terms; the exact Stage 4
    // integer path answers instead. Every input reaching here resolves there,
    // which is what the test counter below records.
    #[cfg(test)]
    INSPHERE_EXACT_COUNT.fetch_add(1, Ordering::Relaxed);

    Sign::Zero
}

// ===========================================================================
// Stage 4: Exact integer fallback
// ===========================================================================

/// Stage 4: exact integer arithmetic in `WideInt<165>`, 10,560 bits.
///
/// Reached in two cases: the input spans more bits than the exactness window
/// allows, so the expansion stages cannot run on it; or the expansion stages
/// returned `Zero`, which is not a certificate. In practice the second case is
/// almost always an exact zero, and this stage confirms it.
///
/// Converts all fifteen coordinates to integers scaled by `2^1074`, which is
/// exact for every finite `f64`, and evaluates the 4×4 lifted determinant with
/// no rounding. The bit-width argument is on the `InsphereInt` type alias in
/// `exact_int.rs`.
fn insphere_exact_integer(
    pa: [f64; 3],
    pb: [f64; 3],
    pc: [f64; 3],
    pd: [f64; 3],
    pe: [f64; 3],
) -> Sign {
    counters::record(Predicate::Insphere, Outcome::Exact);
    use crate::exact_int::f64_to_insphere_int;

    let ax = f64_to_insphere_int(pa[0]);
    let ay = f64_to_insphere_int(pa[1]);
    let az = f64_to_insphere_int(pa[2]);
    let bx = f64_to_insphere_int(pb[0]);
    let by_ = f64_to_insphere_int(pb[1]);
    let bz = f64_to_insphere_int(pb[2]);
    let cx = f64_to_insphere_int(pc[0]);
    let cy = f64_to_insphere_int(pc[1]);
    let cz = f64_to_insphere_int(pc[2]);
    let dx = f64_to_insphere_int(pd[0]);
    let dy = f64_to_insphere_int(pd[1]);
    let dz = f64_to_insphere_int(pd[2]);
    let ex = f64_to_insphere_int(pe[0]);
    let ey = f64_to_insphere_int(pe[1]);
    let ez = f64_to_insphere_int(pe[2]);

    // Coordinate differences relative to pe.
    let aex = ax - ex;
    let aey = ay - ey;
    let aez = az - ez;
    let bex = bx - ex;
    let bey = by_ - ey;
    let bez = bz - ez;
    let cex = cx - ex;
    let cey = cy - ey;
    let cez = cz - ez;
    let dex = dx - ex;
    let dey = dy - ey;
    let dez = dz - ez;

    // Exact lifts (integer squares; the scale factor 2^1074 is implicit so
    // each lift carries a factor of 2^2148 which cancels in the determinant).
    let alift = aex * aex + aey * aey + aez * aez;
    let blift = bex * bex + bey * bey + bez * bez;
    let clift = cex * cex + cey * cey + cez * cez;
    let dlift = dex * dex + dey * dey + dez * dez;

    // Exact 2×2 sub-dets.
    let ab = aex * bey - bex * aey;
    let bc = bex * cey - cex * bey;
    let cd = cex * dey - dex * cey;
    let da = dex * aey - aex * dey;
    let ac = aex * cey - cex * aey;
    let bd = bex * dey - dex * bey;

    // Exact 3×3 sub-dets.
    // abc = aez·bc − bez·ac + cez·ab
    let abc = aez * bc - bez * ac + cez * ab;
    // bcd = bez·cd − cez·bd + dez·bc
    let bcd = bez * cd - cez * bd + dez * bc;
    // cda = cez·da + dez·ac + aez·cd
    let cda = cez * da + dez * ac + aez * cd;
    // dab = dez·ab + aez·bd + bez·da
    let dab = dez * ab + aez * bd + bez * da;

    // det = dlift·abc − clift·dab + blift·cda − alift·bcd
    let det = dlift * abc - clift * dab + blift * cda - alift * bcd;

    if det.is_positive() {
        Sign::Positive
    } else if det.is_negative() {
        Sign::Negative
    } else {
        // Observability only (`stage-counters`): records how much of the exact
        // path is spent confirming exact zeros.
        counters::record(Predicate::Insphere, Outcome::ExactZero);
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
        INSPHERE_EXACT_COUNT.store(0, Ordering::SeqCst);
    }

    // Sign convention, for reading the expected values below: `insphere` is
    // `Positive` when `pe` is inside the sphere through `pa`, `pb`, `pc`,
    // `pd` and those four are positively oriented by `orient3d`. Each test
    // states the orientation of its tetrahedron, so the expected sign follows
    // from geometry rather than from running the predicate.

    /// A point inside the circumsphere of a positively oriented tetrahedron.
    ///
    /// The four vertices of a regular tetrahedron inscribed in the unit
    /// sphere, to within the rounding of `sqrt`. They are positively oriented
    /// (`orient3d` = `Positive`), and the origin is at distance 0 from the
    /// center of their circumsphere, whose radius is about 1. Rounding moves
    /// the vertices by about `1e-16`, far too little to change either fact.
    #[test]
    fn insphere_clearly_inside() {
        let pa = [1.0_f64, 0.0, 0.0];
        let pb = [-1.0_f64 / 3.0, 2.0 * (2.0_f64).sqrt() / 3.0, 0.0];
        let pc = [
            -1.0_f64 / 3.0,
            -(2.0_f64).sqrt() / 3.0,
            (2.0_f64 / 3.0_f64).sqrt(),
        ];
        let pd = [
            -1.0_f64 / 3.0,
            -(2.0_f64).sqrt() / 3.0,
            -(2.0_f64 / 3.0_f64).sqrt(),
        ];
        let pe = [0.0_f64, 0.0, 0.0];
        assert_eq!(crate::orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Positive));
    }

    /// A point outside the circumsphere of a positively oriented tetrahedron.
    ///
    /// `(1,0,0)`, `(0,1,0)`, `(0,0,1)` and `(−1,0,0)` lie on the unit sphere
    /// and are positively oriented. `(10,10,10)` is at distance `10·sqrt(3)`
    /// from the origin, far outside.
    #[test]
    fn insphere_clearly_outside() {
        let pa = [1.0_f64, 0.0, 0.0];
        let pb = [0.0_f64, 1.0, 0.0];
        let pc = [0.0_f64, 0.0, 1.0];
        let pd = [-1.0_f64, 0.0, 0.0];
        let pe = [10.0_f64, 10.0, 10.0];
        assert_eq!(crate::orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Negative));
    }

    /// A point exactly on the circumsphere.
    ///
    /// The same positively oriented tetrahedron on the unit sphere, and
    /// `pe = (0,−1,0)`, which is on the unit sphere too. All coordinates are
    /// small integers, so the determinant is zero exactly.
    #[test]
    fn insphere_exactly_on_circumsphere() {
        let pa = [1.0_f64, 0.0, 0.0];
        let pb = [0.0_f64, 1.0, 0.0];
        let pc = [0.0_f64, 0.0, 1.0];
        let pd = [-1.0_f64, 0.0, 0.0];
        let pe = [0.0_f64, -1.0, 0.0];
        assert_eq!(crate::orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Zero));
    }

    /// Coplanar reference points make the determinant zero, whatever `pe`
    /// is.
    ///
    /// `(±1,0,0)` and `(0,±1,0)` all lie in the plane `z = 0`, so they do not
    /// determine a sphere: the 5×5 determinant has the lifted coordinates of
    /// four coplanar points, and is zero.
    #[test]
    fn insphere_coplanar_reference_points_are_zero() {
        let pa = [1.0_f64, 0.0, 0.0];
        let pb = [-1.0_f64, 0.0, 0.0];
        let pc = [0.0_f64, 1.0, 0.0];
        let pd = [0.0_f64, -1.0, 0.0];
        let pe = [0.0_f64, 0.0, 1.0];
        assert_eq!(crate::orient3d(pa, pb, pc, pd), Ok(Sign::Zero));
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Zero));
    }

    /// A point just outside the circumsphere is resolved, with the right
    /// sign, by the fast stages.
    ///
    /// The tetrahedron of `insphere_clearly_outside` and `pe` at distance
    /// `1 + 1e-13` from the center: outside by a margin the fast filter can
    /// certify.
    #[test]
    fn insphere_point_just_outside() {
        let pa = [1.0_f64, 0.0, 0.0];
        let pb = [0.0_f64, 1.0, 0.0];
        let pc = [0.0_f64, 0.0, 1.0];
        let pd = [-1.0_f64, 0.0, 0.0];
        let pe = [1.0_f64 + 1e-13, 0.0, 0.0];
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Negative));
    }

    /// The exact path fires on cospherical input whose coordinate differences
    /// are inexact, and returns `Zero`.
    ///
    /// With `a = fl(1/3)`, `b = fl(1/7)`, `c = fl(1/11)`, every point below is
    /// a permutation of `(a, b, c)` with some signs flipped. Those operations
    /// are exact, so all five points have the same distance from the origin
    /// exactly, and lie on one sphere: the determinant is zero. The first four
    /// are not coplanar. At least one coordinate difference rounds
    /// (`pb.z − pe.z = a + c` is not representable), so the tails-zero exit
    /// in `insphere_adapt` cannot fire and the input reaches the exact stage.
    #[test]
    fn insphere_exact_path_fires_on_cospherical_input() {
        reset_exact_count();
        let (a, b, c) = (1.0_f64 / 3.0, 1.0_f64 / 7.0, 1.0_f64 / 11.0);
        let pa = [a, b, c];
        let pb = [b, c, a];
        let pc = [c, a, b];
        let pd = [-a, b, c];
        let pe = [a, -b, -c];
        assert_ne!(crate::orient3d(pa, pb, pc, pd), Ok(Sign::Zero));
        assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Zero));
        let count = INSPHERE_EXACT_COUNT.load(Ordering::SeqCst);
        assert!(count > 0, "the exact stage must be reached (count {count})");
    }

    /// Non-finite inputs are rejected with `Err(NonFinite)`.
    #[test]
    fn insphere_non_finite_inputs_return_err() {
        let inf = f64::INFINITY;
        let nan = f64::NAN;
        assert_eq!(
            insphere(
                [inf, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
            ),
            Err(NonFinite::new()),
            "a +inf coordinate must be rejected"
        );
        assert_eq!(
            insphere(
                [nan, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.5, 0.5],
            ),
            Err(NonFinite::new()),
            "a NaN coordinate must be rejected"
        );
    }

    /// All-finite inputs whose coordinate differences overflow f64 to infinity
    /// still return `Ok` with the exact sign, never a premature `Sign::Zero`.
    #[test]
    fn insphere_finite_overflow_is_not_zero() {
        let m = f64::MAX;
        // A non-cospherical configuration whose differences (pa - pe, etc.)
        // overflow: all coordinates finite, true sign is Negative.
        let result = insphere(
            [m, m, m],
            [m, 0.0, 0.0],
            [0.0, m, 0.0],
            [0.0, 0.0, m],
            [-m, -m, -m],
        );
        assert_eq!(
            result,
            Ok(Sign::Negative),
            "finite non-cospherical input must not collapse to Zero on overflow"
        );
    }
}
