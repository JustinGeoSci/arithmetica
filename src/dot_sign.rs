// src/dot_sign.rs

//! The `dot_sign` predicate: whether the angle at a point is acute, right or
//! obtuse.

use crate::exact_int::f64_to_orient_int;
use crate::expansion::{DOT_ERR_BOUND_A, expansion_sum};
use crate::range::{Prepared, U_N, U_S, Window, prepare};
use crate::util::{sign_of, two_by_two_product, two_diff};
use crate::{NonFinite, Sign};

/// The stage of the sequence that answered a call. This crate's own unit
/// tests observe it through [`dot_sign_staged`], and the public function
/// discards it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// A zero difference in each product, certified by comparisons alone.
    ZeroFilter,
    /// The fast filter in native `f64`.
    Filter,
    /// The exact expansion stage, inside the exactness window.
    Expansion,
    /// The exact integer stage.
    Exact,
}

/// The sign of the dot product `(a − p) · (b − p)`.
///
/// Returns [`Sign::Positive`] if the angle `a`–`p`–`b` is acute,
/// [`Sign::Zero`] if it is right (or if `a` or `b` equals `p`), and
/// [`Sign::Negative`] if it is obtuse. The result is exact for every finite
/// input.
///
/// ```
/// use arithmetica::{Sign, dot_sign};
///
/// let p = [0.0, 0.0];
/// assert_eq!(dot_sign(p, [1.0, 0.0], [1.0, 1.0]), Ok(Sign::Positive));
/// assert_eq!(dot_sign(p, [1.0, 0.0], [0.0, 1.0]), Ok(Sign::Zero));
/// assert_eq!(dot_sign(p, [1.0, 0.0], [-1.0, 1.0]), Ok(Sign::Negative));
/// ```
///
/// # Mathematical definition
///
/// With `u = a − p` and `v = b − p`, the predicate is the sign of
/// `ux·vx + uy·vy`. That is the `orient2d` determinant with `+` in place of
/// `−`. Writing it as `ux·vx − (−uy)·vy` makes it exactly the `orient2d`
/// expression on the four differences `ux`, `vx`, `p.y − a.y` and `vy`.
///
/// # Four stages
///
/// - **The zero filter**. When each of the two products has a factor that is
///   exactly zero (`a == p`, `b == p`, or an axis-aligned right angle), the
///   result is `Zero`. Under gradual underflow a difference `fl(x − y)` is
///   zero only when `x == y`, so this is a certificate.
/// - **Stage 1, the fast filter**. It is `orient2d`'s filter, the same
///   expression tree
///   on the four differences above, with the same constant and the same
///   underflow protection, so the certificate holds for every finite input
///   (the reduction is spelled out on the constant in `expansion.rs`).
/// - **Stage 2, the exact expansion stage**. Inside the exactness window of
///   `orient2d`,
///   each difference is an exact two-term expansion, each product their
///   exact product, and the two products are summed exactly. The sign of
///   the largest component is the sign of the dot product. These are the
///   operations of `orient2d`'s exact expansion stage (exact products of
///   exact differences, summed), so the window's exactness argument applies
///   unchanged. Out-of-window input is moved inside by an exact
///   power-of-two rescale when one exists.
/// - **Stage 4, the exact integer stage**. `WideInt<66>` arithmetic on the
///   caller's
///   coordinates answers input wider than the window and confirms every
///   `Zero` from the expansion stage.
///
/// # Non-finite inputs
///
/// A NaN or infinite coordinate returns [`Err(NonFinite)`](NonFinite) before
/// any stage runs. For finite inputs the result is always `Ok`.
///
/// # References
///
/// Shewchuk (1997), §§2–4; Bartels, Fisikopoulos & Weiser (2023),
/// Theorem 1(2) and Definition 6.
///
/// # Errors
///
/// Returns [`NonFinite`] when any input coordinate is not finite.
#[must_use = "the angle classification is the return value; discarding it is almost certainly a bug"]
pub fn dot_sign(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Result<Sign, NonFinite> {
    dot_sign_staged(p, a, b).map(|(sign, _)| sign)
}

/// [`dot_sign`], reporting the stage that answered.
pub(crate) fn dot_sign_staged(
    p: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
) -> Result<(Sign, Stage), NonFinite> {
    if !p.iter().chain(&a).chain(&b).all(|v| v.is_finite()) {
        return Err(NonFinite::new());
    }
    if is_certainly_zero(p, a, b) {
        return Ok((Sign::Zero, Stage::ZeroFilter));
    }
    if let Some(sign) = filter(p, a, b) {
        return Ok((sign, Stage::Filter));
    }
    let raw = (p, a, b);
    let scaled = |q: [f64; 2], s: f64| [q[0] * s, q[1] * s];
    let (p, a, b) = match prepare(&[p[0], p[1], a[0], a[1], b[0], b[1]], Window::ORIENT2D) {
        Prepared::AsIs => (p, a, b),
        Prepared::Scaled(s) => (scaled(p, s), scaled(a, s), scaled(b, s)),
        Prepared::OutOfWindow => return Ok((exact_integer(raw.0, raw.1, raw.2), Stage::Exact)),
    };
    // A `Zero` from the expansion is not a certificate, so the exact integer
    // stage decides it.
    match in_window(p, a, b) {
        Sign::Zero => Ok((exact_integer(raw.0, raw.1, raw.2), Stage::Exact)),
        sign => Ok((sign, Stage::Expansion)),
    }
}

/// Both products have a factor that is exactly zero.
///
/// With gradual underflow, `fl(x − y) == 0.0` if and only if `x == y` (the
/// argument is on `orient2d`'s zero filter), and an overflowing difference is
/// infinite, not zero. A zero difference here is therefore a zero real factor.
/// `clippy::float_cmp` is allowed because the comparisons ask exactly that.
#[inline]
#[allow(clippy::float_cmp)]
fn is_certainly_zero(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    let ux = a[0] - p[0];
    let uy = a[1] - p[1];
    let vx = b[0] - p[0];
    let vy = b[1] - p[1];
    (ux == 0.0 || vx == 0.0) && (uy == 0.0 || vy == 0.0)
}

/// `orient2d`'s underflow-protected filter on `ux·vx − (−uy)·vy`.
///
/// The names follow `orient2d`'s filter so the correspondence is visible:
/// `acx = ux`, `bcy = vx`, `acy = fl(p.y − a.y) = −fl(a.y − p.y)`,
/// `bcx = vy`. See [`DOT_ERR_BOUND_A`] for why the bound carries over.
#[inline]
fn filter(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Option<Sign> {
    let detleft = (a[0] - p[0]) * (b[0] - p[0]);
    let detright = (p[1] - a[1]) * (b[1] - p[1]);
    let det = detleft - detright;

    let magnitude = (detleft.abs() + U_N) + (detright.abs() + U_N);
    let bound = DOT_ERR_BOUND_A * magnitude + U_S;

    if det > bound || -det > bound {
        Some(sign_of(det))
    } else {
        None
    }
}

/// The exact expansion stage, on coordinates the caller has placed inside
/// `Window::ORIENT2D`.
///
/// `ux·vx` and `uy·vy` are formed as exact products of exact two-term
/// differences (at most 8 components each) and summed exactly (at most 16).
/// The last non-zero component carries the sign, and an empty expansion is
/// a zero the caller confirms.
fn in_window(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Sign {
    let ux = two_diff(a[0], p[0]);
    let uy = two_diff(a[1], p[1]);
    let vx = two_diff(b[0], p[0]);
    let vy = two_diff(b[1], p[1]);

    let (x, x_len) = two_by_two_product(ux, vx);
    let (y, y_len) = two_by_two_product(uy, vy);
    let mut dot = [0.0_f64; 16];
    let dot_len = expansion_sum(&x[..x_len], &y[..y_len], &mut dot);

    if dot_len == 0 {
        Sign::Zero
    } else {
        sign_of(dot[dot_len - 1])
    }
}

/// The exact integer stage: `WideInt<66>`, 4,224 bits.
///
/// The bit-width argument is exactly `orient2d`'s. A scaled coordinate is
/// below `2^2098`, a difference below `2^2099`, a product of two below
/// `2^4198`, and the two-term sum below `2^4199`, which fits in 4,224 bits
/// with a sign (`4200` bits are asserted, as for `orient2d`).
///
/// The stage operates on the caller's coordinates, before any rescale.
fn exact_integer(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> Sign {
    let px = f64_to_orient_int(p[0]);
    let py = f64_to_orient_int(p[1]);
    let ux = f64_to_orient_int(a[0]) - px;
    let uy = f64_to_orient_int(a[1]) - py;
    let vx = f64_to_orient_int(b[0]) - px;
    let vy = f64_to_orient_int(b[1]) - py;
    let dot = ux * vx + uy * vy;
    debug_assert!(
        dot.fits_in_signed_bits(4200),
        "dot_sign Stage 4 overflow: width analysis violated; result sign may be incorrect"
    );
    if dot.is_positive() {
        Sign::Positive
    } else if dot.is_negative() {
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
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// A fixed case: three points, the exact sign, and the stage it was
    /// built to be answered at. The integration tests check the signs
    /// against the oracles, and these unit tests check the stages.
    type Case = ([f64; 2], [f64; 2], [f64; 2], Sign, Stage);

    pub(crate) fn fixed_cases() -> [Case; 9] {
        let h = 2.0_f64.powi(500);
        let t = 2.0_f64.powi(-600);
        [
            // The fast filter: acute and obtuse, well separated.
            (
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                Sign::Positive,
                Stage::Filter,
            ),
            (
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 1.0],
                Sign::Negative,
                Stage::Filter,
            ),
            // The zero filter: a == p, and an axis-aligned right angle.
            (
                [0.3, 0.7],
                [0.3, 0.7],
                [5.0, 1.0],
                Sign::Zero,
                Stage::ZeroFilter,
            ),
            (
                [0.1, 0.2],
                [0.7, 0.2],
                [0.1, 0.9],
                Sign::Zero,
                Stage::ZeroFilter,
            ),
            // The expansion stage: (3, 4)·(−4, 3 + 2^-50) = 2^-48, below the
            // filter's bound of about 3ε·24.
            (
                [0.0, 0.0],
                [3.0, 4.0],
                [-4.0, 3.0 + 2.0_f64.powi(-50)],
                Sign::Positive,
                Stage::Expansion,
            ),
            // An exact right angle with inexact differences, confirmed by
            // the exact integer stage. With c = 0.1, s = 0.7 and t = 0.3,
            // u = (s − c, t − c) and v = (c − t, s − c), since b.x = 2c − t
            // is computed exactly (0.2 = 2·0.1 in f64, and 0.2 − 0.3 is exact
            // by Sterbenz).
            (
                [0.1, 0.1],
                [0.7, 0.3],
                [0.2 - 0.3, 0.7],
                Sign::Zero,
                Stage::Exact,
            ),
            // Wider than the window: 2^-600 and 2^500 in one call. Exactly,
            // h·t − t·h(1 + 2^-52) = −h·t·2^-52 < 0.
            (
                [0.0, 0.0],
                [h, t],
                [t, -h * (1.0 + f64::EPSILON)],
                Sign::Negative,
                Stage::Exact,
            ),
            // Wider than the window, and decided by a product that
            // underflows: (h, t)·(0, t) = t² with t = 2^-600. An expansion
            // run on these coordinates as given would lose t² and report a
            // zero that is not one.
            ([0.0, 0.0], [h, t], [0.0, t], Sign::Positive, Stage::Exact),
            // Overflowing differences: answered exactly, never `Zero`.
            (
                [-f64::MAX, 0.0],
                [f64::MAX, 0.0],
                [f64::MAX, 1.0],
                Sign::Positive,
                Stage::Exact,
            ),
        ]
    }

    #[test]
    fn each_fixed_case_is_answered_at_its_stage() {
        for (i, (p, a, b, sign, stage)) in fixed_cases().into_iter().enumerate() {
            assert_eq!(
                dot_sign_staged(p, a, b),
                Ok((sign, stage)),
                "fixed case {i}"
            );
        }
    }

    /// The dot product is symmetric in `a` and `b`.
    #[test]
    fn swapping_the_two_points_keeps_the_sign() {
        for (p, a, b, sign, _) in fixed_cases() {
            assert_eq!(dot_sign(p, b, a), Ok(sign));
        }
    }

    /// The filter is `orient2d`'s on the rearranged differences. On the same
    /// four differences the two filters agree, bit for bit, about whether
    /// and what they certify.
    #[test]
    fn the_filter_is_the_orient2d_filter_on_the_rearranged_differences() {
        let cases = [
            ([0.1, 0.2], [0.7, -0.3], [-0.4, 0.9]),
            ([1e-300, 3.0], [2e-300, -1.0], [5.0, 7e-310]),
            ([0.0, 0.0], [3.0, 4.0], [-4.0, 3.0 + 2.0_f64.powi(-50)]),
        ];
        for (p, a, b) in cases {
            // orient2d's filter expression, written out on the four
            // differences dot_sign uses.
            let acx = a[0] - p[0];
            let bcy = b[0] - p[0];
            let acy = p[1] - a[1];
            let bcx = b[1] - p[1];
            let detleft = acx * bcy;
            let detright = acy * bcx;
            let det = detleft - detright;
            let magnitude = (detleft.abs() + U_N) + (detright.abs() + U_N);
            let bound = crate::expansion::CCW_ERR_BOUND_A * magnitude + U_S;
            let orient = (det > bound || -det > bound).then(|| sign_of(det));
            assert_eq!(filter(p, a, b), orient);
            // The rearranged sum is also the dot product's, since
            // fl(p.y − a.y) is −fl(a.y − p.y) bit for bit.
            assert_eq!(acy.to_bits(), (-(a[1] - p[1])).to_bits());
        }
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
    /// the exponent range, reaches every stage of the sequence. The property
    /// suites, whose results the oracles check, therefore exercise each one.
    #[test]
    fn a_near_degenerate_sweep_reaches_every_stage() {
        let mut rng = Mix(0x2026_1024_57A6_E000);
        let mut seen = [0_u32; 4];
        for i in 0..20_000_u32 {
            let e = i64::try_from(rng.next() % 2000).unwrap_or(0) - 1000;
            let (x, y, c) = (rng.float(e), rng.float(e - 1), rng.float(e - 2));
            let mut config = if i % 3 == 0 {
                [x, x, x, x, x, x]
            } else if i % 2 == 0 {
                [0.0, 0.0, x, y, -y, x]
            } else {
                [c, c, x, y, 2.0 * c - y, x]
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
            if let Ok((_, stage)) = dot_sign_staged(p, q, r) {
                seen[stage as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "stage counts {seen:?} (zero filter, filter, expansion, exact)"
        );
    }
}
