// src/rational.rs

//! Exact predicates over *constructed* points.
//!
//! # Why this module exists
//!
//! Every other predicate in this crate answers a question about input
//! coordinates. An arrangement cannot be built that way. An arrangement *is* a
//! structure of segment-segment intersection points, and it must answer
//! questions **about** those points: are two of them the same point, in what
//! order do they lie along an edge, which side of a segment does one fall on.
//!
//! In `f64` those three questions are unreliable in exactly the configurations
//! where a planar overlay is hardest: nearly-parallel segments, three edges
//! meeting at nearly one point, or a vertex sitting nearly on an edge. A wrong
//! answer there is not a small coordinate error; it is a topologically
//! impossible arrangement, which is the silent-wrong-answer mode this crate
//! exists to eliminate.
//!
//! So a constructed point is represented **exactly**, as a rational over
//! fixed-width integers, and the three predicates are exact integer
//! computations. No division is ever performed; every comparison is
//! cross-multiplied.
//!
//! # The depth-1 property: every width rests on this
//!
//! A ratio of determinants cannot be bounded if the inputs are themselves
//! ratios of determinants: each level of chaining roughly doubles the width,
//! and an unbounded chain needs arbitrary precision.
//!
//! This stays bounded because **chaining depth is always 1**. A
//! [`RationalCoord2d`] is always the intersection of two segments whose four
//! endpoints are **input `f64` values**, never of two previously constructed
//! points. A caller that needs a constructed point as the input to a further
//! construction rounds it to `f64` first, using
//! [`RationalCoord2d::f64_bracket`], and the rounded point is then an
//! ordinary input.
//!
//! That is what makes every width in `exact_int` computable in
//! advance, by the same method already used for `Orient2dInt = WideInt<66>`
//! and `Incircle2dInt = WideInt<132>`.
//!
//! **If constructed points were ever chained, the output of one intersection
//! fed in as the endpoint of another segment without rounding to `f64` in
//! between, none of the width proofs in this module would hold, and the
//! `debug_assert!` guards would be the only thing to say so.** There is
//! deliberately no constructor that accepts a [`RationalCoord2d`] as a segment
//! endpoint, so the property is enforced by the absence of an API rather than
//! by a convention every maintainer has to know about.
//!
//! # Width proofs
//!
//! Each predicate's bound, and the `WideInt<N>` chosen for it, is derived in
//! the documentation of its type alias in the private `exact_int` module:
//! `RationalCoordInt` (storage), `RationalEqInt`, `RationalOrderInt`,
//! `RationalOrientInt`. Those aliases are `pub(crate)`, so the derivations
//! render under `cargo doc --document-private-items`. Each computation here closes
//! with a `debug_assert!(…fits_in_signed_bits(M))` using the `M` from its
//! proof. The guards cost nothing in release and turn a wrong proof into a
//! debug-mode panic rather than a wrong sign.

use core::cmp::Ordering;

use crate::exact_int::{
    RationalCoordInt, RationalEqInt, RationalOrderInt, RationalOrientInt, f64_to_rational_int,
};
use crate::round::F64Bracket;
use crate::{NonFinite, Sign};

// ===========================================================================
// Bit bounds from the width proofs
// ===========================================================================

/// Signed-bit bound on a stored numerator (`|num| < 2^6299`).
const NUM_BITS: usize = 6300;

/// Signed-bit bound on a stored denominator (`0 < den < 2^4199`).
const DEN_BITS: usize = 4200;

/// Signed-bit bound on a cross-multiplied coordinate difference
/// (`|ax·db − bx·da| < 2^10499`).
const CROSS_BITS: usize = 10500;

/// Signed-bit bound on the `rational_compare_along` determinant
/// (`< 2^12599`).
const ORDER_BITS: usize = 12600;

/// Signed-bit bound on a denominator-lifted `f64` difference in
/// `orient2d_rational` (`< 2^6300`).
const LIFT_BITS: usize = 6301;

/// Signed-bit bound on the `orient2d_rational` determinant (`< 2^8400`).
const ORIENT_BITS: usize = 8401;

// ===========================================================================
// RationalCoord2d
// ===========================================================================

/// A point whose coordinates are held **exactly**, as a rational with a
/// shared denominator.
///
/// The value represented is `(num_x / den, num_y / den)` in the scaled
/// integer space this crate uses throughout: the stored rational equals
/// `2^1074 ×` the true coordinate. The `2^1074` factor is common to every
/// quantity in every predicate below and so cancels out of every comparison;
/// it is what makes each finite `f64` an exact integer (the minimum positive
/// subnormal is `2^-1074`).
///
/// # Why a shared denominator
///
/// Both coordinates of a segment-segment intersection carry the *same*
/// determinant as denominator, so storing one denominator rather than two is
/// not a compression; it is the natural form. It also halves the width of
/// every downstream predicate, because a cross-multiplication involves one
/// denominator from each point rather than two.
///
/// # Invariants
///
/// - `den > 0`. Normalized at construction by negating all three components
///   when the determinant comes out negative. Every comparison below is then
///   free of a sign case split, which is where this class of code
///   characteristically goes wrong: a comparison that is correct for positive
///   denominators and inverted for negative ones is a predicate that works
///   until the segment order changes.
/// - `|num_x|, |num_y| < 2^6299` and `den < 2^4199`.
///
/// # Why the fields are private
///
/// A caller able to build a `RationalCoord2d` with `den < 0` could obtain a
/// wrong sign from a correct predicate. An invalid state that cannot be
/// constructed cannot be mishandled by any consumer. The only ways to obtain
/// a `RationalCoord2d` are [`RationalCoord2d::from_f64`] and
/// [`RationalCoord2d::from_segment_intersection`], both of which establish
/// every invariant above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalCoord2d {
    num_x: RationalCoordInt,
    num_y: RationalCoordInt,
    den: RationalCoordInt,
}

impl RationalCoord2d {
    /// Represents an exact `f64` coordinate pair as a `RationalCoord2d`.
    ///
    /// The denominator is `1`, so the representation is exact and the
    /// invariants hold trivially. This is how an input vertex enters a
    /// comparison against a constructed point, in an arrangement, the points
    /// to be ordered along an edge are a mixture of original vertices and
    /// crossings, and both kinds have to be orderable by the same predicate.
    ///
    /// # Errors
    ///
    /// [`NonFinite`] if either coordinate is NaN or infinite. A non-finite
    /// coordinate is rejected at the boundary rather than coerced into a
    /// geometric verdict.
    pub fn from_f64(p: [f64; 2]) -> Result<Self, NonFinite> {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(NonFinite::new());
        }
        Ok(Self {
            num_x: f64_to_rational_int(p[0]),
            num_y: f64_to_rational_int(p[1]),
            den: RationalCoordInt::one(),
        })
    }

    /// The exact intersection point of the line through `a0 a1` and the line
    /// through `b0 b1`.
    ///
    /// All four endpoints are input `f64` values. This is the **only**
    /// constructor that produces a nontrivial denominator, and it is the
    /// reason the depth-1 property holds: there is no overload taking a
    /// `RationalCoord2d` as an endpoint, so a constructed point can never become
    /// an input to another construction.
    ///
    /// # Return values
    ///
    /// - `Ok(Some(p))`: the two lines cross at exactly one point, and `p` is
    ///   that point with no rounding of any kind.
    /// - `Ok(None)`: the determinant is **exactly** zero, so the two lines
    ///   are parallel or identical and no unique intersection point exists.
    ///   This is a geometric fact about the input, not an error, and it is
    ///   reported through the type rather than through `Err` for that reason.
    ///   It is also why the predicates below may assume `den != 0`: a
    ///   zero-denominator `RationalCoord2d` is not constructible, so a zero
    ///   denominator is structurally impossible rather than a precondition
    ///   the caller must uphold.
    /// - `Err(NonFinite)`: some endpoint coordinate is NaN or infinite.
    ///
    /// # Note on segments versus lines
    ///
    /// This computes the **line-line** intersection. Whether that point lies
    /// within either segment is a separate question, decided exactly by
    /// `orient2d` on the input coordinates, and it is the caller's to ask.
    /// The width proofs deliberately do not assume it (see
    /// `RationalCoordInt`).
    ///
    /// # Errors
    ///
    /// [`NonFinite`] if any of the eight input coordinates is not finite.
    pub fn from_segment_intersection(
        a0: [f64; 2],
        a1: [f64; 2],
        b0: [f64; 2],
        b1: [f64; 2],
    ) -> Result<Option<Self>, NonFinite> {
        for p in [a0, a1, b0, b1] {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(NonFinite::new());
            }
        }

        let (a0x, a0y) = (f64_to_rational_int(a0[0]), f64_to_rational_int(a0[1]));
        let (a1x, a1y) = (f64_to_rational_int(a1[0]), f64_to_rational_int(a1[1]));
        let (b0x, b0y) = (f64_to_rational_int(b0[0]), f64_to_rational_int(b0[1]));
        let (b1x, b1y) = (f64_to_rational_int(b1[0]), f64_to_rational_int(b1[1]));

        // Direction vectors, exact in scaled integers.
        let rx = a1x - a0x;
        let ry = a1y - a0y;
        let sx = b1x - b0x;
        let sy = b1y - b0y;

        // den = r × s. Exactly zero iff the lines are parallel; there is no
        // tolerance here and none is wanted: a nearly-parallel pair has a
        // nonzero determinant and a genuine, if distant, crossing point, and
        // treating it as parallel is precisely the wrong answer an epsilon
        // test would give.
        let den = rx * sy - ry * sx;
        debug_assert!(
            den.fits_in_signed_bits(DEN_BITS),
            "RationalCoord2d denominator overflow: width analysis violated"
        );
        if den.is_zero() {
            return Ok(None);
        }

        // num_t = (b0 − a0) × s, the numerator of the parameter along a.
        let wx = b0x - a0x;
        let wy = b0y - a0y;
        let num_t = wx * sy - wy * sx;
        debug_assert!(
            num_t.fits_in_signed_bits(DEN_BITS),
            "RationalCoord2d parameter numerator overflow: width analysis violated"
        );

        // p = a0 + (num_t / den) · r, cleared of the division:
        //     num = a0·den + num_t·r,  with the same den.
        let mut num_x = a0x * den + num_t * rx;
        let mut num_y = a0y * den + num_t * ry;
        let mut den = den;

        debug_assert!(
            num_x.fits_in_signed_bits(NUM_BITS) && num_y.fits_in_signed_bits(NUM_BITS),
            "RationalCoord2d numerator overflow: width analysis violated"
        );

        // Normalize the sign onto the numerators so that `den > 0` holds for
        // every `RationalCoord2d` in existence.
        if den.is_negative() {
            num_x = -num_x;
            num_y = -num_y;
            den = -den;
        }

        debug_assert!(
            den.is_positive(),
            "RationalCoord2d sign normalization failed to make the denominator positive"
        );

        Ok(Some(Self { num_x, num_y, den }))
    }

    /// Brackets each coordinate between its two nearest representable `f64`
    /// values, exactly.
    ///
    /// # This is the boundary, and it does not cross it
    ///
    /// Every other operation on a `RationalCoord2d` is exact and so is this
    /// one: it returns two representable neighbors per coordinate and the
    /// coordinate's exact position between them. It never returns the
    /// coordinate rounded. Choosing a neighbor is a rounding *policy*, a
    /// decision this crate does not make, per the crate-level charter, and
    /// the caller that makes it is the one place an approximation enters.
    ///
    /// **A caller must re-establish, on the rounded output, any geometric
    /// property it needs.** Two distinct `RationalCoord2d`s can bracket to
    /// the same neighbors, and a set of coordinates that formed a simple
    /// polygon can round to one that does not; rounding reintroduces
    /// precisely the degeneracies an exact arrangement was built to resolve.
    /// Nothing about the rounded coordinates may be inferred from a property
    /// of the exact ones.
    ///
    /// Each coordinate is bracketed independently. See [`F64Bracket`] for the
    /// contract at the range extremes and at zero.
    #[must_use]
    pub fn f64_bracket(&self) -> [F64Bracket; 2] {
        [
            crate::round::rational_to_bracket(&self.num_x, &self.den),
            crate::round::rational_to_bracket(&self.num_y, &self.den),
        ]
    }
}

// ===========================================================================
// rational_coords_equal
// ===========================================================================

/// Cross-multiplies one coordinate pair: returns `an·bd − bn·ad`.
///
/// Both denominators are positive by the [`RationalCoord2d`] invariant, so the
/// sign of this value is the sign of `an/ad − bn/bd` with no case split.
fn cross(
    an: &RationalCoordInt,
    ad: &RationalCoordInt,
    bn: &RationalCoordInt,
    bd: &RationalCoordInt,
) -> RationalEqInt {
    let (an, ad): (RationalEqInt, RationalEqInt) = (an.widen(), ad.widen());
    let (bn, bd): (RationalEqInt, RationalEqInt) = (bn.widen(), bd.widen());
    let diff = an * bd - bn * ad;
    debug_assert!(
        diff.fits_in_signed_bits(CROSS_BITS),
        "rational cross-multiplication overflow: width analysis violated; \
         result sign may be incorrect"
    );
    diff
}

/// Returns `true` if `a` and `b` are the same point exactly.
///
/// Both coordinates are compared by cross-multiplication, `ax·db == bx·da`
/// and `ay·db == by·da`, so no division is performed and no tolerance is
/// involved. Two points constructed from different segment pairs, with
/// different denominators, compare equal if and only if they are
/// mathematically the same point.
///
/// This cannot fail: a `RationalCoord2d` is exact by construction and carries no
/// `f64` input to be checked, so there is no `Result`.
///
/// # Width
///
/// `RationalEqInt` = `WideInt<165>`; bound 10500 signed bits; guarded.
#[must_use]
pub fn rational_coords_equal(a: &RationalCoord2d, b: &RationalCoord2d) -> bool {
    cross(&a.num_x, &a.den, &b.num_x, &b.den).is_zero()
        && cross(&a.num_y, &a.den, &b.num_y, &b.den).is_zero()
}

// ===========================================================================
// rational_compare_along
// ===========================================================================

/// Orders two constructed points along the direction of a directed segment.
///
/// Returns the order of `a` and `b` by their projection parameter on the
/// direction `seg_end − seg_start`: `Less` if `a` comes first, `Greater` if
/// `b` does, `Equal` if they project to the same parameter.
///
/// # What is actually computed
///
/// The sign of `(a − b) · d`, where `d = seg_end − seg_start`. Writing
/// `a − b` over the common positive denominator `da·db` and dropping that
/// positive factor:
///
/// ```text
/// sign( proj(a) − proj(b) ) = sign( (ax·db − bx·da)·dx + (ay·db − by·da)·dy )
/// ```
///
/// **`seg_start` cancels algebraically**; only the direction survives. That
/// is not just an economy: it removes a multiplication level, and with it both
/// the width that level would have cost and the chance of an inconsistency
/// between the two sides of the comparison.
///
/// # Ordering guarantees
///
/// The result is a **strict weak ordering** on `RationalCoord2d` for any fixed
/// `seg_start`, `seg_end`:
///
/// - *Antisymmetric*: negating the determinant swaps `Less` and `Greater`.
/// - *Transitive*: because it is the sign of a difference of exact rational
///   numbers `proj(·)`, and `<` on the rationals is transitive. No
///   floating-point tolerance enters, so the comparator cannot be
///   intransitive. A tolerance-based comparator lacks this property, and a
///   sort given an intransitive comparator may panic or return an
///   inconsistent order.
/// - *Equivalence is transitive*: `Equal` means `proj(a) == proj(b)` exactly.
///
/// Note that `Equal` is **not** the same as [`rational_coords_equal`]: two
/// distinct points on a common perpendicular project to the same parameter.
/// That is correct for an ordering along a direction and is why the two
/// predicates are separate.
///
/// # Degenerate direction
///
/// If `seg_start == seg_end` the direction is zero, every determinant is zero,
/// and every pair compares `Equal`. That is still a valid strict weak ordering
/// (one equivalence class), but it carries no information; a caller that
/// reaches it is ordering points along a zero-length segment and has a problem
/// upstream.
///
/// # Errors
///
/// [`NonFinite`] if any coordinate of `seg_start` or `seg_end` is NaN or
/// infinite.
///
/// # Width
///
/// `RationalOrderInt` = `WideInt<198>`; bound 12600 signed bits; guarded.
pub fn rational_compare_along(
    seg_start: [f64; 2],
    seg_end: [f64; 2],
    a: &RationalCoord2d,
    b: &RationalCoord2d,
) -> Result<Ordering, NonFinite> {
    for p in [seg_start, seg_end] {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(NonFinite::new());
        }
    }

    // The direction, exact in scaled integers.
    let dx: RationalOrderInt =
        (f64_to_rational_int(seg_end[0]) - f64_to_rational_int(seg_start[0])).widen();
    let dy: RationalOrderInt =
        (f64_to_rational_int(seg_end[1]) - f64_to_rational_int(seg_start[1])).widen();

    // Coordinate differences of a − b, numerators over the dropped positive
    // denominator da·db. These are the `cross` quantities, re-widened: the
    // guard in `cross` has already checked them against CROSS_BITS.
    let ux: RationalOrderInt = cross(&a.num_x, &a.den, &b.num_x, &b.den).widen();
    let uy: RationalOrderInt = cross(&a.num_y, &a.den, &b.num_y, &b.den).widen();

    let det = ux * dx + uy * dy;
    debug_assert!(
        det.fits_in_signed_bits(ORDER_BITS),
        "rational_compare_along overflow: width analysis violated; \
         ordering may be incorrect"
    );

    Ok(if det.is_zero() {
        Ordering::Equal
    } else if det.is_negative() {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}

// ===========================================================================
// orient2d_rational
// ===========================================================================

/// The orientation of the triple (`pa`, `pb`, `c`), where `pa` and `pb` are
/// input `f64` points and `c` is a constructed point.
///
/// Returns the sign of the determinant `(pb − pa) × (c − pa)`:
/// `Positive` if `c` lies to the left of the directed line `pa → pb`,
/// `Negative` if to the right, `Zero` if exactly on it. The convention matches
/// [`crate::orient2d`] exactly, so a caller can mix the two without a mental
/// sign flip.
///
/// # The mixed-arity lift
///
/// The arguments are not the same kind of thing: `pa` and `pb` become exact
/// scaled integers with an implicit denominator of 1, while `c` carries a
/// denominator of up to 4199 bits. Multiplying the whole determinant through
/// by `c`'s denominator — which is positive, so the sign is untouched — lifts
/// the `f64` points onto `c`'s denominator:
///
/// ```text
/// det·dc = (pbx − pax)·(cy − pay·dc) − (pby − pay)·(cx − pax·dc)
/// ```
///
/// The `pay·dc` and `pax·dc` products are the lift. This is the step the width
/// grows across and the one where a transcription slip — lifting one point and
/// not the other, or omitting the lift on one axis — would produce a
/// well-formed sign that is simply wrong. The `rational_orient_fixtures` and
/// the 100,000-case oracle sweep both exist to catch exactly that.
///
/// # Errors
///
/// [`NonFinite`] if any coordinate of `pa` or `pb` is NaN or infinite.
///
/// # Width
///
/// `RationalOrientInt` = `WideInt<132>`; bound 8401 signed bits; guarded.
pub fn orient2d_rational(
    pa: [f64; 2],
    pb: [f64; 2],
    c: &RationalCoord2d,
) -> Result<Sign, NonFinite> {
    for p in [pa, pb] {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err(NonFinite::new());
        }
    }

    let pax: RationalOrientInt = f64_to_rational_int(pa[0]).widen();
    let pay: RationalOrientInt = f64_to_rational_int(pa[1]).widen();
    let pbx: RationalOrientInt = f64_to_rational_int(pb[0]).widen();
    let pby: RationalOrientInt = f64_to_rational_int(pb[1]).widen();

    let cx: RationalOrientInt = c.num_x.widen();
    let cy: RationalOrientInt = c.num_y.widen();
    let dc: RationalOrientInt = c.den.widen();

    // The lift: c's coordinates are already over `dc`, so pa's must be too.
    let lift_x = cx - pax * dc;
    let lift_y = cy - pay * dc;
    debug_assert!(
        lift_x.fits_in_signed_bits(LIFT_BITS) && lift_y.fits_in_signed_bits(LIFT_BITS),
        "orient2d_rational lift overflow: width analysis violated"
    );

    let det = (pbx - pax) * lift_y - (pby - pay) * lift_x;
    debug_assert!(
        det.fits_in_signed_bits(ORIENT_BITS),
        "orient2d_rational overflow: width analysis violated; \
         result sign may be incorrect"
    );

    Ok(if det.is_zero() {
        Sign::Zero
    } else if det.is_negative() {
        Sign::Negative
    } else {
        Sign::Positive
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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

    fn isect(a0: [f64; 2], a1: [f64; 2], b0: [f64; 2], b1: [f64; 2]) -> RationalCoord2d {
        RationalCoord2d::from_segment_intersection(a0, a1, b0, b1)
            .unwrap()
            .unwrap()
    }

    // ===========================================================================
    // representation
    // ===========================================================================

    #[test]
    fn denominator_is_normalized_positive_in_both_argument_orders() {
        // Swapping the two segments negates the determinant. Both orders must
        // still yield den > 0, and the same point.
        let p = isect([0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]);
        let q = isect([0.0, 4.0], [4.0, 0.0], [0.0, 0.0], [4.0, 4.0]);
        assert!(p.den.is_positive(), "den must be normalized positive");
        assert!(q.den.is_positive(), "den must be normalized positive");
        assert!(
            rational_coords_equal(&p, &q),
            "the crossing point does not depend on which segment is named first"
        );
    }

    #[test]
    fn exact_lattice_crossing_equals_the_f64_point_exactly() {
        // (0,0)–(4,4) and (0,4)–(4,0) cross at exactly (2,2), which is an
        // exactly representable f64 pair. The rational point must compare
        // equal to it with no tolerance.
        let p = isect([0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]);
        let two_two = RationalCoord2d::from_f64([2.0, 2.0]).unwrap();
        assert!(rational_coords_equal(&p, &two_two));
    }

    #[test]
    fn non_representable_crossing_is_not_equal_to_its_f64_rounding() {
        // x = 1/3 is not an f64. The exact crossing of (0,0)–(1,1) with
        // (1,0)–(0,2) is at (2/3, 2/3): solve t = 2 − 2t.
        let p = isect([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 2.0]);
        let rounded = RationalCoord2d::from_f64([2.0 / 3.0, 2.0 / 3.0]).unwrap();
        assert!(
            !rational_coords_equal(&p, &rounded),
            "the exact point and its f64 rounding are different points; \
             reporting them equal would mean the representation is not exact"
        );
    }

    #[test]
    fn parallel_lines_report_no_intersection() {
        assert_eq!(
            RationalCoord2d::from_segment_intersection(
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ),
            Ok(None)
        );
    }

    #[test]
    fn collinear_lines_report_no_intersection() {
        // Identical lines: infinitely many crossings, so no unique point.
        assert_eq!(
            RationalCoord2d::from_segment_intersection(
                [0.0, 0.0],
                [2.0, 2.0],
                [1.0, 1.0],
                [3.0, 3.0]
            ),
            Ok(None)
        );
    }

    #[test]
    fn non_finite_inputs_are_rejected() {
        assert_eq!(
            RationalCoord2d::from_f64([f64::NAN, 0.0]),
            Err(NonFinite::new())
        );
        assert_eq!(
            RationalCoord2d::from_f64([0.0, f64::INFINITY]),
            Err(NonFinite::new())
        );
        assert_eq!(
            RationalCoord2d::from_segment_intersection(
                [0.0, 0.0],
                [1.0, 1.0],
                [f64::NEG_INFINITY, 0.0],
                [1.0, 0.0]
            ),
            Err(NonFinite::new())
        );
        let p = RationalCoord2d::from_f64([0.0, 0.0]).unwrap();
        assert_eq!(
            rational_compare_along([0.0, 0.0], [f64::NAN, 1.0], &p, &p),
            Err(NonFinite::new())
        );
        assert_eq!(
            orient2d_rational([f64::NAN, 0.0], [1.0, 0.0], &p),
            Err(NonFinite::new())
        );
    }

    // ===========================================================================
    // rational_coords_equal
    // ===========================================================================

    #[test]
    fn equal_is_reflexive_and_distinguishes_distinct_points() {
        let p = isect([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 2.0]);
        let q = isect([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 3.0]);
        assert!(rational_coords_equal(&p, &p));
        assert!(!rational_coords_equal(&p, &q));
    }

    #[test]
    fn equal_sees_through_different_denominators() {
        // The same crossing found via two different segment pairs through it.
        // (2,2) is the crossing of the diagonals of the 0..4 square, and also
        // of the two axis-aligned mid-lines.
        let diag = isect([0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]);
        let mids = isect([2.0, 0.0], [2.0, 9.0], [0.0, 2.0], [7.0, 2.0]);
        assert!(
            rational_coords_equal(&diag, &mids),
            "equality must be a statement about the point, not about the \
             segment pair it was derived from"
        );
    }

    // ===========================================================================
    // rational_compare_along
    // ===========================================================================

    #[test]
    fn compare_along_orders_three_crossings_on_one_edge() {
        // Edge along y = 0 from (0,0) to (10,0), crossed by three verticals.
        let e0 = [0.0, 0.0];
        let e1 = [10.0, 0.0];
        let at = |x: f64| isect(e0, e1, [x, -1.0], [x, 1.0]);
        let (p1, p2, p3) = (at(1.0), at(5.0), at(9.0));

        assert_eq!(rational_compare_along(e0, e1, &p1, &p2), Ok(Ordering::Less));
        assert_eq!(
            rational_compare_along(e0, e1, &p3, &p2),
            Ok(Ordering::Greater)
        );
        assert_eq!(
            rational_compare_along(e0, e1, &p2, &p2),
            Ok(Ordering::Equal)
        );
    }

    #[test]
    fn compare_along_reverses_with_the_direction() {
        let e0 = [0.0, 0.0];
        let e1 = [10.0, 0.0];
        let at = |x: f64| isect(e0, e1, [x, -1.0], [x, 1.0]);
        let (p1, p2) = (at(1.0), at(9.0));
        assert_eq!(rational_compare_along(e0, e1, &p1, &p2), Ok(Ordering::Less));
        assert_eq!(
            rational_compare_along(e1, e0, &p1, &p2),
            Ok(Ordering::Greater),
            "reversing the directed segment must reverse the order"
        );
    }

    #[test]
    fn compare_along_equal_does_not_imply_same_point() {
        // Two distinct points on a common perpendicular to the direction.
        let a = RationalCoord2d::from_f64([3.0, 1.0]).unwrap();
        let b = RationalCoord2d::from_f64([3.0, 7.0]).unwrap();
        assert_eq!(
            rational_compare_along([0.0, 0.0], [1.0, 0.0], &a, &b),
            Ok(Ordering::Equal)
        );
        assert!(
            !rational_coords_equal(&a, &b),
            "Equal order and equal point are different questions"
        );
    }

    #[test]
    fn compare_along_zero_direction_is_all_equal() {
        let a = RationalCoord2d::from_f64([1.0, 2.0]).unwrap();
        let b = RationalCoord2d::from_f64([9.0, 9.0]).unwrap();
        assert_eq!(
            rational_compare_along([4.0, 4.0], [4.0, 4.0], &a, &b),
            Ok(Ordering::Equal)
        );
    }

    // ===========================================================================
    // orient2d_rational
    // ===========================================================================

    #[test]
    fn orient_rational_matches_orient2d_on_an_exact_point() {
        // For a point with den == 1 the two predicates answer the same
        // question, so they must agree, including the sign convention.
        let pa = [0.0, 0.0];
        let pb = [1.0, 0.0];
        for c in [[0.5, 1.0], [0.5, -1.0], [0.5, 0.0], [2.0, 0.0]] {
            let rp = RationalCoord2d::from_f64(c).unwrap();
            assert_eq!(
                orient2d_rational(pa, pb, &rp).unwrap(),
                crate::orient2d(pa, pb, c).unwrap(),
                "orient2d_rational must share orient2d's convention for c = {c:?}"
            );
        }
    }

    #[test]
    fn orient_rational_is_zero_on_the_generating_line() {
        // A crossing point lies exactly on both generating lines, so
        // orient2d_rational against either line's endpoints must be Zero.
        // This is the sharpest available test of the lift: a lift error on one
        // axis only, or on one point only, perturbs this off zero.
        let a0 = [0.0, 0.0];
        let a1 = [7.0, 3.0];
        let b0 = [1.0, 5.0];
        let b1 = [6.0, -2.0];
        let p = isect(a0, a1, b0, b1);
        assert_eq!(orient2d_rational(a0, a1, &p), Ok(Sign::Zero));
        assert_eq!(orient2d_rational(b0, b1, &p), Ok(Sign::Zero));
        assert_eq!(
            orient2d_rational(a1, a0, &p),
            Ok(Sign::Zero),
            "reversing the line must not disturb an exact zero"
        );
    }

    #[test]
    fn orient_rational_reverses_with_the_line() {
        let a0 = [0.0, 0.0];
        let a1 = [1.0, 0.0];
        let p = isect([0.0, 1.0], [2.0, 1.0], [1.0, 0.0], [1.0, 5.0]); // (1,1)
        let left = orient2d_rational(a0, a1, &p).unwrap();
        let right = orient2d_rational(a1, a0, &p).unwrap();
        assert_eq!(left, Sign::Positive);
        assert_eq!(right, Sign::Negative);
        assert_eq!(right, -left);
    }

    // ===========================================================================
    // the depth-1 / extreme-magnitude corner
    // ===========================================================================

    #[test]
    fn extreme_magnitude_endpoints_stay_within_the_proven_widths() {
        // The width proofs are stated for the full finite f64 range, so the
        // guards must survive coordinates at the top of that range. In debug
        // builds a violated proof panics here rather than returning a wrong
        // sign.
        let big = f64::MAX;
        let p = isect([-big, -big], [big, big], [-big, big], [big, -big]);
        // The two diagonals of a square centered at the origin cross at the
        // origin exactly.
        let origin = RationalCoord2d::from_f64([0.0, 0.0]).unwrap();
        assert!(rational_coords_equal(&p, &origin));
    }

    #[test]
    fn subnormal_endpoints_are_handled_exactly() {
        // The 2^1074 scaling exists so that the minimum subnormal is an
        // integer. A crossing built from subnormals must still be exact.
        let s = f64::from_bits(1); // 2^-1074
        let p = isect(
            [0.0, 0.0],
            [4.0 * s, 4.0 * s],
            [0.0, 4.0 * s],
            [4.0 * s, 0.0],
        );
        let mid = RationalCoord2d::from_f64([2.0 * s, 2.0 * s]).unwrap();
        assert!(rational_coords_equal(&p, &mid));
    }
}
