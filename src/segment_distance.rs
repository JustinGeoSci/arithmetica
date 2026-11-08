// src/segment_distance.rs

//! The `compare_point_distance` and `compare_segment_distance` predicates:
//! a point's distance to a point, or to a closed segment, compared with a
//! given `d`, exactly.

use crate::dot_sign;
use crate::exact_int::{Incircle2dInt, f64_to_incircle_int, f64_to_orient_int};
use crate::expansion::{
    PT_DIST_ERR_BOUND_A, SEG_DIST_ERR_BOUND_A, expansion_sum, scale_expansion, two_product,
};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{negate, sign_of, two_by_two_product, two_diff};
use crate::{NonFinite, Sign};

/// The stage of the evaluation that answered a call. This crate's own unit
/// tests observe it through the `*_staged` functions, and the public
/// functions discard it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Certified by comparisons alone (`p == q` for the point form).
    ZeroFilter,
    /// The fast filter in native `f64`.
    Filter,
    /// The exact expansion stage, inside the exactness window.
    Expansion,
    /// The exact integer stage.
    Exact,
}

/// Which of the three cases of [`compare_segment_distance`] answered, that
/// is, the nearest point of the segment, decided exactly by [`dot_sign`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Case {
    /// `(p − a) · (b − a) ≤ 0`: the nearest point is `a`.
    NearA,
    /// `(p − b) · (a − b) ≤ 0`: the nearest point is `b`.
    NearB,
    /// Both dot products are positive: the perpendicular foot lies strictly
    /// inside the segment.
    Interior,
}

// ===========================================================================
// compare_point_distance
// ===========================================================================

/// Compares the distance from `p` to `q` with `d`.
///
/// Returns the sign of `|p − q|² − d²`: [`Sign::Negative`] if `p` is closer
/// to `q` than `|d|`, [`Sign::Zero`] if it is at exactly that distance, and
/// [`Sign::Positive`] if it is farther. The result is exact for every finite
/// input. Only `d²` enters, so the sign of `d` is irrelevant, and `d` and
/// `−d` ask the same question.
///
/// ```
/// use arithmetica::{Sign, compare_point_distance};
///
/// assert_eq!(compare_point_distance([0.0, 0.0], [3.0, 4.0], 5.0), Ok(Sign::Zero));
/// assert_eq!(compare_point_distance([0.0, 0.0], [3.0, 4.0], 6.0), Ok(Sign::Negative));
/// assert_eq!(compare_point_distance([0.0, 0.0], [3.0, 4.0], -4.0), Ok(Sign::Positive));
/// ```
///
/// # Mathematical definition
///
/// With `u = q − p`, the predicate is the sign of the degree-2 polynomial
/// `ux² + uy² − d²`, in which `d` takes part like a coordinate.
///
/// # Four stages
///
/// - **The zero filter**. `p == q` reduces the polynomial to `−d²`, whose sign
///   is read from `d == 0.0` by comparisons alone.
/// - **Stage 1, the fast filter**. The polynomial is evaluated in native `f64`
///   and
///   accepted when its magnitude exceeds `A · mag + u_S`, with `mag` the
///   same expression tree on magnitudes, protected against underflow by a
///   `u_N` term beside each product. The bound is derived on
///   `PT_DIST_ERR_BOUND_A` (in `expansion.rs`) and holds for every finite
///   input.
/// - **Stage 2, the exact expansion stage**. Inside the exactness window of
///   `orient2d`
///   (the polynomial is degree 2, and `d` is windowed with the
///   coordinates), each difference is an exact two-term expansion, each
///   square an exact product, `d²` an exact two-term product, and the sum
///   is exact. The largest component carries the sign. Out-of-window input
///   is moved inside by an exact power-of-two rescale when one exists,
///   which multiplies the polynomial by a positive `s²`.
/// - **Stage 4, the exact integer stage**. `WideInt<66>` operates on the
///   caller's
///   values scaled by `2^1074`, which multiplies the polynomial by
///   `2^2148`, a positive factor. It answers input spread over more bits
///   than the window is wide, and it confirms every `Zero` the expansion
///   stage reports.
///
/// # Non-finite inputs
///
/// A NaN or infinite coordinate, or a non-finite `d`, returns
/// [`Err(NonFinite)`](NonFinite) before any stage runs. For finite inputs
/// the result is always `Ok`.
///
/// # References
///
/// Shewchuk (1997), §§2–4, for the expansion arithmetic; Bartels,
/// Fisikopoulos & Weiser (2023) for the underflow-protected filter form.
///
/// # Errors
///
/// Returns [`NonFinite`] when any input value is not finite.
#[must_use = "the comparison result is the return value; discarding it is almost certainly a bug"]
pub fn compare_point_distance(p: [f64; 2], q: [f64; 2], d: f64) -> Result<Sign, NonFinite> {
    compare_point_distance_staged(p, q, d).map(|(sign, _)| sign)
}

/// [`compare_point_distance`], reporting the stage that answered.
pub(crate) fn compare_point_distance_staged(
    p: [f64; 2],
    q: [f64; 2],
    d: f64,
) -> Result<(Sign, Stage), NonFinite> {
    if !p.iter().chain(&q).chain(&[d]).all(|v| v.is_finite()) {
        return Err(NonFinite::new());
    }
    #[allow(clippy::float_cmp)] // the zero filter asks exactly this question
    if p == q {
        // The polynomial is −d², and `d == 0.0` reads its sign exactly
        // (`-0.0 == 0.0` is true, and they square to the same zero).
        let sign = if d == 0.0 { Sign::Zero } else { Sign::Negative };
        return Ok((sign, Stage::ZeroFilter));
    }
    if let Some(sign) = point_filter(p, q, d) {
        return Ok((sign, Stage::Filter));
    }
    let raw = (p, q, d);
    let (p, q, d) = match prepare(&[p[0], p[1], q[0], q[1], d], Window::ORIENT2D) {
        Prepared::AsIs => (p, q, d),
        Prepared::Scaled(s) => ([p[0] * s, p[1] * s], [q[0] * s, q[1] * s], d * s),
        Prepared::OutOfWindow => return Ok((point_exact(raw.0, raw.1, raw.2), Stage::Exact)),
    };
    // A `Zero` from the expansion is not a certificate, so the exact integer
    // stage decides it, as in `compare_distance`.
    match point_in_window(p, q, d) {
        Sign::Zero => Ok((point_exact(raw.0, raw.1, raw.2), Stage::Exact)),
        sign => Ok((sign, Stage::Expansion)),
    }
}

/// The fast filter for the point form, valid for every finite input
/// (derivation on [`PT_DIST_ERR_BOUND_A`]).
///
/// Every term of `mag` is non-negative, so an overflow anywhere makes `mag`,
/// and with it the bound, infinite, and `det` infinite or NaN. The strict
/// comparison fails either way, so overflowing input is passed on rather
/// than certified. The `u_S` term keeps the bound positive, so an exact zero
/// is never certified, and absorbs the rounding of the bound's own product
/// (see item (7) of the constant's derivation).
#[inline]
fn point_filter(p: [f64; 2], q: [f64; 2], d: f64) -> Option<Sign> {
    let ux = q[0] - p[0];
    let uy = q[1] - p[1];
    let x2 = ux * ux;
    let y2 = uy * uy;
    let dd = d * d;
    let det = (x2 + y2) - dd;
    let mag = ((x2 + U_N) + (y2 + U_N)) + (dd + U_N);
    let bound = PT_DIST_ERR_BOUND_A * mag + U_S;
    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// The exact expansion stage of the point form, on values the caller has
/// placed inside `Window::ORIENT2D` (`d` included).
///
/// The differences are exact two-term expansions, each square their exact
/// product (at most 8 components), `d²` an exact two-term product, and the
/// three are summed exactly (at most 18 components). Every operation is a
/// degree-2 product or a sum of windowed values, so the window's exactness
/// argument applies unchanged. The largest-magnitude component, the last
/// non-zero one of a non-overlapping expansion, carries the sign. An empty
/// expansion is a zero the caller confirms.
fn point_in_window(p: [f64; 2], q: [f64; 2], d: f64) -> Sign {
    let ux = two_diff(q[0], p[0]);
    let uy = two_diff(q[1], p[1]);
    let (x2, x2_len) = two_by_two_product(ux, ux);
    let (y2, y2_len) = two_by_two_product(uy, uy);
    // −d², zero-eliminated by hand. `two_product` returns (head, tail),
    // the expansion form is tail first, and either part can be zero.
    let (d_head, d_tail) = two_product(d, d);
    let mut dd = [0.0_f64; 2];
    let mut dd_len = 0;
    for part in [d_tail, d_head] {
        if part != 0.0 {
            dd[dd_len] = -part;
            dd_len += 1;
        }
    }

    let mut s = [0.0_f64; 16];
    let s_len = expansion_sum(&x2[..x2_len], &y2[..y2_len], &mut s);
    let mut out = [0.0_f64; 18];
    let out_len = expansion_sum(&s[..s_len], &dd[..dd_len], &mut out);
    if out_len == 0 {
        Sign::Zero
    } else {
        sign_of(out[out_len - 1])
    }
}

/// The exact integer stage of the point form: `WideInt<66>`, 4,224 bits.
///
/// Each value, `d` included, is converted to an integer scaled by `2^1074`,
/// which is exact for every finite `f64`. The polynomial is multiplied by
/// `2^2148`, a positive factor, leaving its sign unchanged.
///
/// The bit-width argument is `compare_distance`'s with `d` in the second
/// place. A scaled value is below `2^2098`, a difference below `2^2099`, a
/// square of a difference below `2^4198`, their sum below `2^4199`, `d²`
/// below `2^4196`, and the difference of the two below `2^4200`, which fits
/// in 4,224 bits with a sign (4,201 bits are asserted).
fn point_exact(p: [f64; 2], q: [f64; 2], d: f64) -> Sign {
    let ux = f64_to_orient_int(q[0]) - f64_to_orient_int(p[0]);
    let uy = f64_to_orient_int(q[1]) - f64_to_orient_int(p[1]);
    let dd = f64_to_orient_int(d);
    let out = (ux * ux + uy * uy) - dd * dd;
    debug_assert!(
        out.fits_in_signed_bits(4201),
        "compare_point_distance Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );
    if out.is_positive() {
        Sign::Positive
    } else if out.is_negative() {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

// ===========================================================================
// compare_segment_distance
// ===========================================================================

/// Compares the distance from `p` to the closed segment `ab` with `d`.
///
/// Returns the sign of `dist²(p, ab) − d²`: [`Sign::Negative`] if `p` is
/// closer to the segment than `|d|`, [`Sign::Zero`] if it is at exactly that
/// distance, and [`Sign::Positive`] if it is farther. The result is exact
/// for every finite input. Only `d²` enters, so the sign of `d` is
/// irrelevant. A degenerate segment (`a == b`) is the point `a`.
///
/// ```
/// use arithmetica::{Sign, compare_segment_distance};
///
/// let (a, b) = ([0.0, 0.0], [10.0, 0.0]);
/// assert_eq!(compare_segment_distance([5.0, 3.0], a, b, 3.0), Ok(Sign::Zero));
/// assert_eq!(compare_segment_distance([5.0, 3.0], a, b, 4.0), Ok(Sign::Negative));
/// assert_eq!(compare_segment_distance([-3.0, 4.0], a, b, 5.0), Ok(Sign::Zero));
/// ```
///
/// # Mathematical definition
///
/// With `u = b − a` and `v = p − a`, the nearest point of the closed segment
/// is `a` when `v·u ≤ 0`, `b` when `(p − b)·(a − b) ≤ 0`, and otherwise the
/// perpendicular foot, which then lies strictly inside. The three cases
/// cover every input. For `a == b` both dot products are zero, so the first
/// case applies and the segment is the point `a`. For `p` equal to an
/// endpoint, that endpoint's case applies and the distance is zero. In the
/// two endpoint cases the predicate is [`compare_point_distance`]. In the
/// interior case `dist² = cross² / |u|²` with `cross = u × v`, and since
/// `|u|² > 0` there (`a ≠ b`), the answer is the sign of the degree-4
/// polynomial
///
/// ```text
/// cross² − d² · |u|²
/// ```
///
/// with no division.
///
/// # Exactness
///
/// **The case choice is exact**, by two [`dot_sign`] calls (`≤ 0` is "not
/// positive"), and it is the only branch, so the predicate as a whole is
/// exact for every finite input. `≤` rather than `<` sends a foot that falls
/// exactly on an endpoint to that endpoint's case. The interior formula
/// would agree there (`cross² = |p − a|²·|u|²` when the foot is at `a`), but
/// only `≤` keeps the degenerate segment, where both dot products are zero
/// and `|u|² = 0`, out of the interior case.
///
/// The interior case is a sequenceof stages, as is every predicate in this
/// crate.
///
/// - **Stage 1, the fast filter**. The polynomial is evaluated in native `f64`
///   and
///   accepted when its magnitude exceeds `A · mag + u_S`, with `mag` the
///   same expression tree on magnitudes and a `u_N` term beside each
///   product. The bound is derived on `SEG_DIST_ERR_BOUND_A` (in
///   `expansion.rs`) and holds for every finite input.
/// - **Stage 2, the exact expansion stage**. It operates inside the exactness
///   window of
///   `incircle` (the polynomial is degree 4, and `d` is windowed with the
///   coordinates). `cross` is an exact expansion of at most 16 components,
///   its square an exact sum of `scale_expansion` products, `d² · |u|²`
///   likewise, and the difference exact, at most 576 components. Every
///   product is of two components of exact degree-2 expansions of windowed
///   values, so each factor is at least `2^-450` in magnitude when non-zero
///   and each product at least `2^-900`, inside the range where
///   `two_product` is exact and far above the `2^-970` floor. No sum can
///   overflow, since components stay below `2^958` and there are at most
///   576 of them. Out-of-window input is moved inside by an exact
///   power-of-two rescale when one exists, which multiplies the polynomial
///   by a positive `s⁴`.
/// - **Stage 4, the exact integer stage**. `WideInt<132>` (8,448 bits) operates
///   on the
///   caller's values scaled by `2^1074`, which multiplies the polynomial by
///   `2^4296`, a positive factor. It answers input wider than the window
///   and confirms every `Zero` the expansion stage reports.
///
/// A width argument governs the integer stage. A scaled value is below
/// `2^2098` and a difference below `2^2099`. `cross`, a sum of two products
/// of differences, is below `2^4199`, so `cross²` is below `2^8398`. The
/// scaled `d` is below `2^2098`, so `d²` is below `2^4196`, and `|u|²` is
/// below `2^4199`, so `d²·|u|²` is below `2^8395`. The difference is below
/// `2^8399`, which fits in 8,448 bits with a sign (8,400 bits are asserted).
///
/// # Non-finite inputs
///
/// A NaN or infinite coordinate, or a non-finite `d`, returns
/// [`Err(NonFinite)`](NonFinite) before any stage runs. For finite inputs
/// the result is always `Ok`.
///
/// # References
///
/// Shewchuk (1997), §§2–4; Bartels, Fisikopoulos & Weiser (2023) for the
/// underflow-protected filter form.
///
/// # Errors
///
/// Returns [`NonFinite`] when any input value is not finite.
#[must_use = "the comparison result is the return value; discarding it is almost certainly a bug"]
pub fn compare_segment_distance(
    p: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    d: f64,
) -> Result<Sign, NonFinite> {
    compare_segment_distance_staged(p, a, b, d).map(|(sign, _, _)| sign)
}

/// [`compare_segment_distance`], reporting the case and the stage that
/// answered.
pub(crate) fn compare_segment_distance_staged(
    p: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    d: f64,
) -> Result<(Sign, Case, Stage), NonFinite> {
    if !p
        .iter()
        .chain(&a)
        .chain(&b)
        .chain(&[d])
        .all(|v| v.is_finite())
    {
        return Err(NonFinite::new());
    }
    // The nearest point of the segment, decided exactly. `(p − a)·(b − a)`
    // is `dot_sign` at the vertex `a`, and `(p − b)·(a − b)` at `b`.
    if dot_sign(a, p, b)? != Sign::Positive {
        let (sign, stage) = compare_point_distance_staged(p, a, d)?;
        return Ok((sign, Case::NearA, stage));
    }
    if dot_sign(b, p, a)? != Sign::Positive {
        let (sign, stage) = compare_point_distance_staged(p, b, d)?;
        return Ok((sign, Case::NearB, stage));
    }
    if let Some(sign) = segment_filter(p, a, b, d) {
        return Ok((sign, Case::Interior, Stage::Filter));
    }
    let raw = (p, a, b, d);
    let scaled = |v: [f64; 2], s: f64| [v[0] * s, v[1] * s];
    let (p, a, b, d) = match prepare(&[p[0], p[1], a[0], a[1], b[0], b[1], d], Window::INCIRCLE) {
        Prepared::AsIs => (p, a, b, d),
        Prepared::Scaled(s) => (scaled(p, s), scaled(a, s), scaled(b, s), d * s),
        Prepared::OutOfWindow => {
            return Ok((
                segment_exact(raw.0, raw.1, raw.2, raw.3),
                Case::Interior,
                Stage::Exact,
            ));
        }
    };
    // A `Zero` from the expansion is not a certificate, so the exact integer
    // stage decides it, as in `compare_distance`.
    match segment_in_window(p, a, b, d) {
        Sign::Zero => Ok((
            segment_exact(raw.0, raw.1, raw.2, raw.3),
            Case::Interior,
            Stage::Exact,
        )),
        sign => Ok((sign, Case::Interior, Stage::Expansion)),
    }
}

/// The fast filter for the interior case, valid for every finite input
/// (derivation on [`SEG_DIST_ERR_BOUND_A`]).
///
/// Every term of `mag` is non-negative, so an overflow anywhere makes `mag`,
/// and with it the bound, infinite, and `det` infinite or NaN. The strict
/// comparison fails either way, so overflowing input is passed on rather
/// than certified. The `u_S` term keeps the bound positive, so an exact zero
/// is never certified, and absorbs the rounding of the bound's own product.
#[inline]
fn segment_filter(p: [f64; 2], a: [f64; 2], b: [f64; 2], d: f64) -> Option<Sign> {
    let ux = b[0] - a[0];
    let uy = b[1] - a[1];
    let vx = p[0] - a[0];
    let vy = p[1] - a[1];
    let t1 = ux * vy;
    let t2 = uy * vx;
    let cross = t1 - t2;
    let x2 = ux * ux;
    let y2 = uy * uy;
    let len2 = x2 + y2;
    let dd = d * d;
    let det = cross * cross - dd * len2;
    let cross_mag = (t1.abs() + U_N) + (t2.abs() + U_N);
    let len_mag = (x2 + U_N) + (y2 + U_N);
    let mag = (cross_mag * cross_mag + U_N) + ((dd + U_N) * len_mag + U_N);
    let bound = SEG_DIST_ERR_BOUND_A * mag + U_S;
    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// The exact sum of `acc[..acc_len]` and `e`, accumulated into `acc`
/// through the scratch buffer, returning the new length.
fn accumulate(acc: &mut [f64; 576], acc_len: usize, e: &[f64], scratch: &mut [f64; 576]) -> usize {
    let len = expansion_sum(&acc[..acc_len], e, &mut scratch[..acc_len + e.len()]);
    acc[..len].copy_from_slice(&scratch[..len]);
    len
}

/// The exact square of an expansion of at most 16 components, formed as the
/// sum of `scale_expansion(e, e[i])` over its components, at most 512 of
/// them.
fn expansion_square(e: &[f64], out: &mut [f64; 576], scratch: &mut [f64; 576]) -> usize {
    let mut len = 0;
    let mut term = [0.0_f64; 32];
    for &component in e {
        let term_len = scale_expansion(e, component, &mut term);
        len = accumulate(out, len, &term[..term_len], scratch);
    }
    len
}

/// The exact expansion stage of the interior case, on values the caller has
/// placed inside `Window::INCIRCLE` (`d` included).
///
/// `cross = (b − a) × (p − a)` is built exactly as in `orient2d`'s exact
/// stage, from two-term differences and their exact 8-component products,
/// summed to at most 16 components. `cross²` is its exact square (at most
/// 512 components), `|u|²` an exact sum of two squares (at most 16), `d²` an
/// exact two-term product, `d²·|u|²` their exact product (at most 64), and
/// the result their exact difference (at most 576). The window argument for
/// each product is on [`compare_segment_distance`]. The largest-magnitude
/// component carries the sign, and an empty expansion is a zero the caller
/// confirms.
fn segment_in_window(p: [f64; 2], a: [f64; 2], b: [f64; 2], d: f64) -> Sign {
    let ux = two_diff(b[0], a[0]);
    let uy = two_diff(b[1], a[1]);
    let vx = two_diff(p[0], a[0]);
    let vy = two_diff(p[1], a[1]);

    let (t1, t1_len) = two_by_two_product(ux, vy);
    let (mut t2, t2_len) = two_by_two_product(uy, vx);
    negate(&mut t2[..t2_len]);
    let mut cross = [0.0_f64; 16];
    let cross_len = expansion_sum(&t1[..t1_len], &t2[..t2_len], &mut cross);

    let mut out = [0.0_f64; 576];
    let mut scratch = [0.0_f64; 576];
    let mut out_len = expansion_square(&cross[..cross_len], &mut out, &mut scratch);

    // −d² · |u|², formed by scaling |u|² by the two components of d²,
    // negating, and adding.
    let (x2, x2_len) = two_by_two_product(ux, ux);
    let (y2, y2_len) = two_by_two_product(uy, uy);
    let mut len2 = [0.0_f64; 16];
    let len2_len = expansion_sum(&x2[..x2_len], &y2[..y2_len], &mut len2);
    let (d_head, d_tail) = two_product(d, d);
    let mut term = [0.0_f64; 32];
    for part in [d_tail, d_head] {
        if part == 0.0 {
            continue;
        }
        let term_len = scale_expansion(&len2[..len2_len], part, &mut term);
        negate(&mut term[..term_len]);
        out_len = accumulate(&mut out, out_len, &term[..term_len], &mut scratch);
    }

    if out_len == 0 {
        Sign::Zero
    } else {
        sign_of(out[out_len - 1])
    }
}

/// The exact integer stage of the interior case: `WideInt<132>`, 8,448
/// bits, on values scaled by `2^1074`, which multiplies the polynomial by
/// `2^4296`, a positive factor. The width argument is on
/// [`compare_segment_distance`].
fn segment_exact(p: [f64; 2], a: [f64; 2], b: [f64; 2], d: f64) -> Sign {
    let at = |x: f64| -> Incircle2dInt { f64_to_incircle_int(x) };
    let ux = at(b[0]) - at(a[0]);
    let uy = at(b[1]) - at(a[1]);
    let vx = at(p[0]) - at(a[0]);
    let vy = at(p[1]) - at(a[1]);
    let cross = ux * vy - uy * vx;
    let dd = at(d);
    let out = cross * cross - (dd * dd) * (ux * ux + uy * uy);
    debug_assert!(
        out.fits_in_signed_bits(8400),
        "compare_segment_distance Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );
    if out.is_positive() {
        Sign::Positive
    } else if out.is_negative() {
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
mod tests;
