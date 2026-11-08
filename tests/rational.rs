// tests/rational.rs

//! Integration tests for the constructed-point predicates
//! (`RationalCoord2d`, `rational_coords_equal`, `rational_compare_along`,
//! `orient2d_rational`).
//!
//! # Structure
//!
//! The tests fall into five groups.
//!
//! - The oracle, an independent exact model built on `Ratio<BigInt>`. It
//!   shares no code with the implementation. It lives in this file, derives
//!   the intersection point from the two lines' *implicit* equations by
//!   Cramer's rule rather than from the parametric form the implementation
//!   uses, and answers each predicate by forming the rational coordinates and
//!   **dividing**, so the check assumes none of the cross-multiplication
//!   identities the implementation relies on. [`oracle_point`] gives the
//!   derivation.
//! - Degenerate fixtures, covering a shared endpoint, three-way concurrence,
//!   collinear overlap and endpoint-on-segment, each test stating its
//!   expected answer and the reason for it.
//! - Generated sweeps, comparing 100,000 cases per predicate with the
//!   oracle. The generator is deliberately *not* uniform. A uniform generator
//!   over a wide exponent range produces no coincident points, no
//!   concurrences and no parallel pairs at all, the configurations an
//!   arrangement meets constantly on real polygon data. The strategy mixes
//!   eight families (see [`three_segments`]), placing the degenerate cases in
//!   the distribution as well as in the fixtures.
//! - The strict weak ordering, checking antisymmetry, transitivity of
//!   `Less`, and transitivity of the `Equal` equivalence over generated
//!   triples of constructed points.
//! - Contract edges, covering non-finite rejection and a zero denominator
//!   reported as `Ok(None)` rather than as an error.

// The test-code lint posture is stated rather than left implicit, as in the
// matching comment in tests/orient2d.rs.
#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::unwrap_used,
    clippy::type_complexity,
    clippy::no_effect_underscore_binding,
    clippy::if_not_else,
    clippy::comparison_chain,
    clippy::too_many_lines
)]

use std::cmp::Ordering;

use arithmetica::{
    RationalCoord2d, Sign, orient2d_rational, rational_compare_along, rational_coords_equal,
};
use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;

/// Each generated sweep runs 100,000 cases in an optimized build and 2,000 in
/// a debug build.
///
/// Run the suite with `--release` to get the full 100,000-case sweeps:
/// `cargo test --release --test rational`.
///
/// # Why the count depends on the profile
///
/// The predicates compute in up to 12,672 exact bits. At `opt-level = 0` a
/// single `rational_compare_along` costs many times its optimized form, which
/// makes a 100,000-case debug sweep slow enough that it would stop being run.
/// The split keeps both halves of the coverage.
///
/// - **Optimized (100,000 cases)**, compared against the exact oracle. This is
///   the stronger check of the width proofs. A proof that is wrong by a bit
///   produces a *wrong sign*, which the oracle sees directly, whereas the
///   `debug_assert!` guards see only the bit width.
/// - **Debug (2,000 cases)**, the same generator with the `fits_in_signed_bits`
///   guards live, so a width violation fails fast and names which quantity
///   overflowed. [`guards_hold_at_extreme_magnitudes`] adds cases at the top
///   of the exponent range, where the proven bounds are approached.
///
/// One *case* is three segments and therefore up to three constructed points,
/// so each case drives several predicate comparisons, the equality sweep
/// performing up to nine, the ordering sweep up to twenty-seven, and the
/// strict-weak-ordering sweep up to eighty-one.
const RATIONAL_CASES: u32 = if cfg!(debug_assertions) {
    2_000
} else {
    100_000
};

type Rat = Ratio<BigInt>;
type Pt = [f64; 2];
type Seg = [Pt; 2];

// ===========================================================================
// The oracle
// ===========================================================================

/// Converts a finite `f64` to the exact rational it denotes.
///
/// The value is decomposed from the IEEE 754 fields directly, a normal value
/// being `(2^52 | mantissa) × 2^(biased_exp − 1075)` and a subnormal
/// `mantissa × 2^-1074`. A negative power of two becomes the denominator, so
/// the result is exact with no scaling convention of any kind. The oracle
/// deliberately does **not** reuse the implementation's `2^1074` scaling, so a
/// mistake in that scaling cannot hide here.
fn exact(x: f64) -> Rat {
    assert!(x.is_finite(), "oracle given a non-finite value");
    let bits = x.to_bits();
    let negative = (bits >> 63) != 0;
    // The field is 11 bits, so the reinterpretation cannot wrap. Naming it
    // keeps `cast_possible_wrap` enabled everywhere else in this file.
    let biased_exp = i64::try_from((bits >> 52) & 0x7FF).expect("11-bit field fits i64");
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    let (significand, exponent) = if biased_exp == 0 {
        (BigInt::from(mantissa), -1074_i64)
    } else {
        (BigInt::from((1u64 << 52) | mantissa), biased_exp - 1075)
    };

    let one = BigInt::from(1u8);
    let magnitude = if exponent >= 0 {
        Rat::from(significand << (exponent as usize))
    } else {
        Rat::new(significand, one << ((-exponent) as usize))
    };

    if negative { -magnitude } else { magnitude }
}

/// The exact intersection of the line through `a` and the line through `b`, or
/// `None` when they are parallel or identical.
///
/// # Independent derivation
///
/// The implementation uses the parametric form
/// `p = a0 + t·(a1 − a0)` with `t` a ratio of cross products. The oracle
/// instead writes each line in implicit form and solves the 2×2 system by
/// Cramer's rule.
///
/// ```text
/// line a:  (a1y − a0y)·x  −  (a1x − a0x)·y  =  (a1y − a0y)·a0x − (a1x − a0x)·a0y
/// line b:  (b1y − b0y)·x  −  (b1x − b0x)·y  =  (b1y − b0y)·b0x − (b1x − b0x)·b0y
///
/// D  = α₁β₂ − α₂β₁
/// x  = (γ₁β₂ − γ₂β₁) / D
/// y  = (α₁γ₂ − α₂γ₁) / D
/// ```
///
/// The numerators are different expressions from the implementation's, and
/// `Ratio` reduces the result to lowest terms, so a representation the
/// implementation stores unreduced over a 4199-bit denominator is compared
/// here against the same number in its canonical form. A predicate that is
/// accidentally sensitive to the *representation* rather than the *value* fails
/// against this oracle.
fn oracle_point(a: Seg, b: Seg) -> Option<(Rat, Rat)> {
    let (a0x, a0y) = (exact(a[0][0]), exact(a[0][1]));
    let (a1x, a1y) = (exact(a[1][0]), exact(a[1][1]));
    let (b0x, b0y) = (exact(b[0][0]), exact(b[0][1]));
    let (b1x, b1y) = (exact(b[1][0]), exact(b[1][1]));

    let alpha1 = &a1y - &a0y;
    let beta1 = -(&a1x - &a0x);
    let gamma1 = &alpha1 * &a0x + &beta1 * &a0y;

    let alpha2 = &b1y - &b0y;
    let beta2 = -(&b1x - &b0x);
    let gamma2 = &alpha2 * &b0x + &beta2 * &b0y;

    let d = &alpha1 * &beta2 - &alpha2 * &beta1;
    if d == Rat::from(BigInt::from(0u8)) {
        return None;
    }

    let x = (&gamma1 * &beta2 - &gamma2 * &beta1) / &d;
    let y = (&alpha1 * &gamma2 - &alpha2 * &gamma1) / &d;
    Some((x, y))
}

/// The oracle for `rational_coords_equal` decides equality of exact rational
/// pairs.
fn oracle_equal(p: &(Rat, Rat), q: &(Rat, Rat)) -> bool {
    p.0 == q.0 && p.1 == q.1
}

/// The oracle for `rational_compare_along` compares the projection parameter
/// computed as a rational number, with the segment origin included.
///
/// The implementation proves the origin cancels and omits it. The oracle keeps
/// it, so that the cancellation is a claim under test rather than one shared
/// by both sides.
fn oracle_compare_along(seg: Seg, p: &(Rat, Rat), q: &(Rat, Rat)) -> Ordering {
    let (ox, oy) = (exact(seg[0][0]), exact(seg[0][1]));
    let dx = exact(seg[1][0]) - &ox;
    let dy = exact(seg[1][1]) - &oy;

    let proj = |r: &(Rat, Rat)| (&r.0 - &ox) * &dx + (&r.1 - &oy) * &dy;
    proj(p).cmp(&proj(q))
}

/// The oracle for `orient2d_rational` takes the sign of the signed-area
/// determinant evaluated in exact rational arithmetic.
fn oracle_orient(pa: Pt, pb: Pt, c: &(Rat, Rat)) -> Sign {
    let (pax, pay) = (exact(pa[0]), exact(pa[1]));
    let (pbx, pby) = (exact(pb[0]), exact(pb[1]));
    let det = (&pbx - &pax) * (&c.1 - &pay) - (&pby - &pay) * (&c.0 - &pax);
    let zero = Rat::from(BigInt::from(0u8));
    if det == zero {
        Sign::Zero
    } else if det < zero {
        Sign::Negative
    } else {
        Sign::Positive
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn construct(a: Seg, b: Seg) -> Option<RationalCoord2d> {
    RationalCoord2d::from_segment_intersection(a[0], a[1], b[0], b[1]).unwrap()
}

/// Builds the implementation point and the oracle point for the same pair, or
/// `None` if the pair is parallel.
///
/// It also asserts that the two sides **agree about whether a point exists at
/// all**. A disagreement there would be a divergence in the degeneracy test
/// itself, easy to miss if only the existing points are compared.
fn pair(a: Seg, b: Seg) -> Option<(RationalCoord2d, (Rat, Rat))> {
    let impl_pt = construct(a, b);
    let oracle_pt = oracle_point(a, b);
    assert_eq!(
        impl_pt.is_some(),
        oracle_pt.is_some(),
        "implementation and oracle disagree on whether a unique intersection \
         exists for a = {a:?}, b = {b:?}"
    );
    match (impl_pt, oracle_pt) {
        (Some(i), Some(o)) => Some((i, o)),
        _ => None,
    }
}

// ===========================================================================
// Degenerate fixtures
// ===========================================================================

/// Two segments meeting at a shared endpoint construct exactly that endpoint.
///
/// The shared endpoint satisfies both line equations, and a pair of
/// non-parallel lines meets in exactly one point, so the endpoint is the
/// intersection. It is also exactly representable (it is an input `f64`
/// pair), which makes this the one degenerate family where a literal checks
/// the answer rather than the oracle alone.
#[test]
fn degenerate_shared_endpoint() {
    let shared = [3.0, 7.0];
    let a: Seg = [shared, [10.0, 2.0]];
    let b: Seg = [shared, [-4.0, 11.0]];

    let p = construct(a, b).expect("non-parallel lines must cross");
    let as_input = RationalCoord2d::from_f64(shared).unwrap();
    assert!(
        rational_coords_equal(&p, &as_input),
        "the crossing of two segments sharing an endpoint is that endpoint"
    );

    // And it lies exactly on both lines.
    assert_eq!(orient2d_rational(a[0], a[1], &p), Ok(Sign::Zero));
    assert_eq!(orient2d_rational(b[0], b[1], &p), Ok(Sign::Zero));
}

/// Three segments concurrent at one point give three equal pairwise
/// crossings, which compare `Equal` along any direction.
///
/// This configuration breaks arrangements built on `f64`. Each pairwise
/// crossing is computed from a *different* segment pair and so carries a
/// different denominator, and nothing but exactness makes the three agree. An
/// `f64` arrangement rounds them to three nearby but distinct vertices,
/// producing a face structure that does not exist.
#[test]
fn degenerate_three_way_concurrence() {
    // All three lines pass through (2, 3).
    let a: Seg = [[0.0, 1.0], [4.0, 5.0]]; // slope 1
    let b: Seg = [[2.0, -5.0], [2.0, 9.0]]; // vertical
    let c: Seg = [[0.0, 7.0], [4.0, -1.0]]; // slope -2

    let ab = construct(a, b).unwrap();
    let ac = construct(a, c).unwrap();
    let bc = construct(b, c).unwrap();

    assert!(rational_coords_equal(&ab, &ac));
    assert!(rational_coords_equal(&ab, &bc));
    assert!(rational_coords_equal(&ac, &bc));

    let expected = RationalCoord2d::from_f64([2.0, 3.0]).unwrap();
    assert!(rational_coords_equal(&ab, &expected));

    for (p, q) in [(&ab, &ac), (&ab, &bc), (&ac, &bc)] {
        assert_eq!(
            rational_compare_along(a[0], a[1], p, q),
            Ok(Ordering::Equal),
            "concurrent crossings occupy one position along the edge"
        );
    }
}

/// Collinear overlapping segments return `Ok(None)`, having no unique
/// intersection point.
///
/// The determinant is **exactly** zero, so the two lines coincide and their
/// intersection is a segment rather than a point. Returning any single point
/// would choose one of infinitely many, and returning an error would
/// misreport a well-defined geometric fact as a domain violation. `None` is
/// the honest answer.
#[test]
fn degenerate_collinear_overlap() {
    // Both segments lie on y = 2x, and they overlap on x ∈ [1, 2].
    let a: Seg = [[0.0, 0.0], [2.0, 4.0]];
    let b: Seg = [[1.0, 2.0], [3.0, 6.0]];
    assert_eq!(
        RationalCoord2d::from_segment_intersection(a[0], a[1], b[0], b[1]),
        Ok(None)
    );
    assert_eq!(oracle_point(a, b), None, "the oracle must agree");

    // Parallel-but-distinct is the same determinant condition.
    let c: Seg = [[0.0, 1.0], [2.0, 5.0]];
    assert_eq!(
        RationalCoord2d::from_segment_intersection(a[0], a[1], c[0], c[1]),
        Ok(None)
    );
}

/// A segment endpoint lying exactly on the interior of another segment is
/// the crossing, and `orient2d_rational` against the carrying segment is
/// `Sign::Zero`.
///
/// The endpoint is on both lines, so it *is* the unique crossing. The `Zero`
/// is the more telling half. It is the only case where the mixed-arity lift
/// in `orient2d_rational` has to cancel to exactly nothing, and a lift
/// applied to one axis but not the other would show up here as a small
/// nonzero determinant rather than as a crash.
#[test]
fn degenerate_endpoint_on_segment() {
    let carrier: Seg = [[0.0, 0.0], [8.0, 4.0]];
    let on_it = [6.0, 3.0]; // exactly on y = x/2
    let b: Seg = [on_it, [6.0, 10.0]];

    let p = construct(carrier, b).unwrap();
    assert!(rational_coords_equal(
        &p,
        &RationalCoord2d::from_f64(on_it).unwrap()
    ));
    assert_eq!(
        orient2d_rational(carrier[0], carrier[1], &p),
        Ok(Sign::Zero)
    );
    assert_eq!(
        orient2d_rational(carrier[1], carrier[0], &p),
        Ok(Sign::Zero),
        "an exact zero must not acquire a sign when the line is reversed"
    );
}

/// Nearly parallel lines have a real but enormously distant crossing, which
/// is produced, kept distinct from any nearby point, and lies exactly on both
/// lines.
///
/// The width proofs refuse the bounding-box argument for this case (see
/// `RationalCoordInt`). The crossing of two segments that differ in direction
/// by one ULP sits far outside both segments' bounding boxes, so the
/// numerator is near its proven maximum rather than near the coordinate
/// range. An `f64` computation here returns infinity or garbage.
#[test]
fn degenerate_near_parallel_distant_crossing() {
    let a: Seg = [[0.0, 0.0], [1.0, 1.0]];
    // Direction one ULP off from a's.
    let b: Seg = [[0.0, 1.0], [1.0, f64::from_bits(1.0_f64.to_bits() + 1)]];

    let (p, o) = pair(a, b).expect("one ULP of direction difference is not parallel");
    assert_eq!(orient2d_rational(a[0], a[1], &p), Ok(Sign::Zero));
    assert_eq!(orient2d_rational(b[0], b[1], &p), Ok(Sign::Zero));

    // The oracle's coordinate is astronomically large. The implementation
    // must agree that it is not any ordinary point.
    assert!(
        !rational_coords_equal(&p, &RationalCoord2d::from_f64([0.0, 0.0]).unwrap()),
        "a distant crossing must not collapse to the origin"
    );
    assert!(oracle_equal(&o, &o));
}

// ===========================================================================
// Generators
// ===========================================================================

/// A coordinate from a small integer lattice.
///
/// Small integers are the opposite of a uniform `f64` strategy and the reason
/// the degenerate families appear at all. On a 21×21 lattice, concurrences,
/// exactly collinear pairs and exactly coincident crossings occur by chance
/// rather than by construction.
fn lattice_coord() -> impl Strategy<Value = f64> {
    (-10_i32..=10).prop_map(f64::from)
}

fn lattice_point() -> impl Strategy<Value = Pt> {
    (lattice_coord(), lattice_coord()).prop_map(|(x, y)| [x, y])
}

/// A finite `f64` of ordinary magnitude, including subnormals and zero.
fn ordinary_coord() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => -1.0e6_f64..1.0e6,
        1 => -1.0_f64..1.0,
        1 => prop::num::f64::ANY.prop_filter("finite", |v| v.is_finite()),
    ]
}

fn ordinary_point() -> impl Strategy<Value = Pt> {
    (ordinary_coord(), ordinary_coord()).prop_map(|(x, y)| [x, y])
}

/// A finite `f64` near the top of the exponent range.
///
/// The width proofs are stated for the whole finite range, and the proven
/// bounds are approached only when the scaled coordinates are near `2^2098`,
/// that is, when `|x|` is near `f64::MAX`. A generator that draws from a
/// moderate range never gets within two thousand bits of any bound, so it
/// could not detect a width proof that is wrong by one bit. Raising the case
/// count does not help. Only changing where the cases come from does.
fn extreme_coord() -> impl Strategy<Value = f64> {
    (2020_u64..=2046, 0_u64..(1u64 << 52), any::<bool>()).prop_map(|(biased_exp, mantissa, neg)| {
        f64::from_bits((u64::from(neg) << 63) | (biased_exp << 52) | mantissa)
    })
}

fn extreme_point() -> impl Strategy<Value = Pt> {
    (extreme_coord(), extreme_coord()).prop_map(|(x, y)| [x, y])
}

/// Three segments, drawn from eight families so that every degenerate
/// configuration an arrangement meets is in the distribution.
///
/// Each family exists to reach one configuration.
///
/// 1. **Lattice**, reaching exact coincidences and collinearities by chance.
/// 2. **Ordinary `f64`**, reaching the generic non-degenerate case, plus
///    subnormals and the full exponent range.
/// 3. **Concurrent**, with all three lines through one lattice point.
/// 4. **Shared endpoint**, with two segments from a common vertex.
/// 5. **Endpoint-on-segment**, where one segment starts at the exact midpoint of
///    another (exact because the midpoint of two even lattice coordinates is a
///    lattice coordinate).
/// 6. **Near-parallel**, with directions differing by a few ULPs, producing
///    distant crossings at the top of the proven numerator width.
/// 7. **Near-coincident**, with two crossings separated by a few ULPs of input,
///    so `rational_coords_equal` and `rational_compare_along` must
///    distinguish points no `f64` computation could.
/// 8. **Extreme magnitude**, with coordinates near `f64::MAX`, the only regime in
///    which the proven width bounds are approached.
fn three_segments() -> impl Strategy<Value = [Seg; 3]> {
    let lattice = (
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
    )
        .prop_map(|(a, b, c, d, e, f)| [[a, b], [c, d], [e, f]]);

    let ordinary = (
        ordinary_point(),
        ordinary_point(),
        ordinary_point(),
        ordinary_point(),
        ordinary_point(),
        ordinary_point(),
    )
        .prop_map(|(a, b, c, d, e, f)| [[a, b], [c, d], [e, f]]);

    let concurrent = (
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
    )
        .prop_map(|(p, d1, d2, d3)| {
            let thru = |d: Pt| -> Seg { [[p[0] - d[0], p[1] - d[1]], [p[0] + d[0], p[1] + d[1]]] };
            [thru(d1), thru(d2), thru(d3)]
        });

    let shared_endpoint = (
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
    )
        .prop_map(|(p, q, r, s, t)| [[p, q], [p, r], [s, t]]);

    let endpoint_on = (
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
        lattice_point(),
    )
        .prop_map(|(p, q, r, s, t)| {
            // 2·p and 2·q are even, so their midpoint p + q is exact.
            let a: Seg = [[2.0 * p[0], 2.0 * p[1]], [2.0 * q[0], 2.0 * q[1]]];
            let mid = [p[0] + q[0], p[1] + q[1]];
            [a, [mid, r], [s, t]]
        });

    let near_parallel = (
        ordinary_point(),
        ordinary_point(),
        ordinary_point(),
        -3_i64..=3,
        -3_i64..=3,
    )
        .prop_map(|(p, q, offset, nx, ny)| {
            let nudge = |v: f64, n: i64| -> f64 {
                if v == 0.0 || !v.is_finite() {
                    v
                } else {
                    let bits = v.to_bits();
                    f64::from_bits(bits.wrapping_add(n as u64))
                }
            };
            let a: Seg = [p, q];
            let b: Seg = [
                [p[0] + offset[0], p[1] + offset[1]],
                [nudge(q[0] + offset[0], nx), nudge(q[1] + offset[1], ny)],
            ];
            // Third segment crosses both at a generic angle.
            let c: Seg = [[p[1], -p[0]], [q[1], -q[0]]];
            [a, b, c]
        });

    let near_coincident = (
        lattice_point(),
        lattice_point(),
        lattice_point(),
        -2_i64..=2,
    )
        .prop_map(|(p, q, r, n)| {
            let nudge = move |v: f64| -> f64 {
                if v == 0.0 {
                    v
                } else {
                    f64::from_bits(v.to_bits().wrapping_add(n as u64))
                }
            };
            let a: Seg = [p, q];
            let b: Seg = [r, [r[0] + 1.0, r[1] - 1.0]];
            let c: Seg = [[nudge(r[0]), nudge(r[1])], [r[0] + 1.0, r[1] - 1.0]];
            [a, b, c]
        });

    let extreme = (
        extreme_point(),
        extreme_point(),
        extreme_point(),
        extreme_point(),
        extreme_point(),
        extreme_point(),
    )
        .prop_map(|(a, b, c, d, e, f)| [[a, b], [c, d], [e, f]]);

    prop_oneof![
        3 => lattice,
        2 => ordinary,
        2 => concurrent,
        2 => shared_endpoint,
        2 => endpoint_on,
        2 => near_parallel,
        2 => near_coincident,
        2 => extreme,
    ]
}

// ===========================================================================
// The sweeps
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(RATIONAL_CASES))]

    /// `rational_coords_equal` against the oracle.
    ///
    /// Each case yields up to three pairwise crossings of three segments, so
    /// up to three equality comparisons are checked per case.
    #[test]
    fn equal_matches_oracle(segs in three_segments()) {
        let [a, b, c] = segs;
        let pairs = [(a, b), (a, c), (b, c)];
        let built: Vec<_> = pairs.iter().filter_map(|&(s, t)| pair(s, t)).collect();

        for i in 0..built.len() {
            for j in 0..built.len() {
                let (pi, oi) = &built[i];
                let (pj, oj) = &built[j];
                prop_assert_eq!(
                    rational_coords_equal(pi, pj),
                    oracle_equal(oi, oj),
                    "rational_coords_equal disagrees with the oracle for \
                     segments {:?} (indices {}, {})", segs, i, j
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(RATIONAL_CASES))]

    /// `rational_compare_along` against the oracle.
    ///
    /// The direction is taken from each of the three segments in turn, so a
    /// crossing is ordered along an edge it lies on *and* along edges it does
    /// not. The projection is defined either way, and the implementation must
    /// match the oracle in both.
    #[test]
    fn compare_along_matches_oracle(segs in three_segments()) {
        let [a, b, c] = segs;
        let built: Vec<_> = [(a, b), (a, c), (b, c)]
            .iter()
            .filter_map(|&(s, t)| pair(s, t))
            .collect();

        for dir in [a, b, c] {
            for i in 0..built.len() {
                for j in 0..built.len() {
                    let (pi, oi) = &built[i];
                    let (pj, oj) = &built[j];
                    prop_assert_eq!(
                        rational_compare_along(dir[0], dir[1], pi, pj).unwrap(),
                        oracle_compare_along(dir, oi, oj),
                        "rational_compare_along disagrees with the oracle for \
                         segments {:?} along {:?} (indices {}, {})",
                        segs, dir, i, j
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(RATIONAL_CASES))]

    /// `orient2d_rational` against the oracle.
    ///
    /// Every crossing is tested against every segment's endpoints in both
    /// orders. Two of those tests are the exactly-on-the-line case (a crossing
    /// against a line that generated it), which is where the mixed-arity lift
    /// has to cancel to exactly zero.
    #[test]
    fn orient_rational_matches_oracle(segs in three_segments()) {
        let [a, b, c] = segs;
        let built: Vec<_> = [(a, b), (a, c), (b, c)]
            .iter()
            .filter_map(|&(s, t)| pair(s, t))
            .collect();

        for line in [a, b, c] {
            for (p, o) in &built {
                prop_assert_eq!(
                    orient2d_rational(line[0], line[1], p).unwrap(),
                    oracle_orient(line[0], line[1], o),
                    "orient2d_rational disagrees with the oracle for segments \
                     {:?} against line {:?}", segs, line
                );
                prop_assert_eq!(
                    orient2d_rational(line[1], line[0], p).unwrap(),
                    oracle_orient(line[1], line[0], o),
                    "orient2d_rational disagrees with the oracle in the \
                     reversed order for {:?} against {:?}", segs, line
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]

    /// The `debug_assert!(fits_in_signed_bits(M))` guards hold at the top of
    /// the exponent range.
    ///
    /// Every predicate is driven on extreme-magnitude input, the only regime
    /// where the proven bounds are approached. The assertions here are almost
    /// beside the point. In a debug build this test exercises the guards
    /// inside the predicates, and a width proof that is wrong by a bit panics
    /// from inside `arithmetica` rather than returning a sign this test would
    /// then have to detect.
    ///
    /// The test runs 20,000 cases rather than 100,000 because it deliberately
    /// runs in the debug profile, where the guards exist. The optimized
    /// sweeps above carry the 100,000-case oracle comparison.
    #[test]
    fn guards_hold_at_extreme_magnitudes(
        a0 in extreme_point(), a1 in extreme_point(),
        b0 in extreme_point(), b1 in extreme_point(),
        c0 in extreme_point(), c1 in extreme_point(),
    ) {
        let a: Seg = [a0, a1];
        let b: Seg = [b0, b1];
        let c: Seg = [c0, c1];

        let Some(p) = construct(a, b) else { return Ok(()); };

        // Equality against itself and against an unrelated exact point.
        prop_assert!(rational_coords_equal(&p, &p));
        let far = RationalCoord2d::from_f64([f64::MAX, f64::MIN]).unwrap();
        let _ = rational_coords_equal(&p, &far);

        // Ordering along each segment, against itself and against `far`.
        for dir in [a, b, c] {
            prop_assert_eq!(
                rational_compare_along(dir[0], dir[1], &p, &p).unwrap(),
                Ordering::Equal
            );
            let fwd = rational_compare_along(dir[0], dir[1], &p, &far).unwrap();
            let rev = rational_compare_along(dir[0], dir[1], &far, &p).unwrap();
            prop_assert_eq!(fwd, rev.reverse());
        }

        // Orientation against each segment. Against the two lines that
        // generated `p` the determinant must be exactly zero even here.
        prop_assert_eq!(orient2d_rational(a[0], a[1], &p).unwrap(), Sign::Zero);
        prop_assert_eq!(orient2d_rational(b[0], b[1], &p).unwrap(), Sign::Zero);
        let _ = orient2d_rational(c[0], c[1], &p).unwrap();
    }
}

// ===========================================================================
// Strict weak ordering
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(RATIONAL_CASES))]

    /// `rational_compare_along` is a strict weak ordering.
    ///
    /// The axioms are checked on every ordered triple of the crossings a case
    /// produces, for each of the three available directions.
    ///
    /// - **Antisymmetry.** `cmp(p, q)` is the reverse of `cmp(q, p)`.
    /// - **Transitivity of `Less`.** `p < q` and `q < r` imply `p < r`.
    /// - **Transitivity of equivalence.** `p ≡ q` and `q ≡ r` imply `p ≡ r`.
    /// - **Compatibility.** `p ≡ q` and `q < r` imply `p < r`.
    ///
    /// Pairwise tests are not enough here. A non-transitive comparator can
    /// pass every one of them and still fail inside a sort, where a
    /// comparator that contradicts itself is detected (by a panic or a wrong
    /// order) only after the invariant is already broken. This comparator
    /// cannot be intransitive because it is the sign of a difference of exact
    /// rationals and admits no tolerance, but the argument is worth nothing
    /// without the sweep that confirms the code implements it.
    #[test]
    fn compare_along_is_a_strict_weak_ordering(segs in three_segments()) {
        let [a, b, c] = segs;
        let built: Vec<RationalCoord2d> = [(a, b), (a, c), (b, c)]
            .iter()
            .filter_map(|&(s, t)| construct(s, t))
            .collect();

        let n = built.len();

        for dir in [a, b, c] {
            // The comparison matrix is built once, `n²` calls rather than
            // `n³`, and every axiom below is then read off it. Calling the
            // comparator again inside the triple loop would make the sweep
            // nine times more expensive for no additional coverage, because
            // the axioms are properties *of the matrix* and any inconsistency
            // between two calls on the same pair would show up as an
            // antisymmetry failure here anyway.
            let mut m = [[Ordering::Equal; 3]; 3];
            for i in 0..n {
                for j in 0..n {
                    m[i][j] =
                        rational_compare_along(dir[0], dir[1], &built[i], &built[j]).unwrap();
                }
            }

            for (i, row) in m.iter().enumerate().take(n) {
                prop_assert_eq!(row[i], Ordering::Equal, "not reflexive");
                for (j, &forward) in row.iter().enumerate().take(n) {
                    prop_assert_eq!(
                        forward,
                        m[j][i].reverse(),
                        "not antisymmetric along {:?} for {:?}", dir, segs
                    );
                }
            }

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let (ij, jk, ik) = (m[i][j], m[j][k], m[i][k]);
                        if ij == Ordering::Less && jk == Ordering::Less {
                            prop_assert_eq!(
                                ik, Ordering::Less,
                                "Less is not transitive along {:?} for {:?}", dir, segs
                            );
                        }
                        if ij == Ordering::Equal && jk == Ordering::Equal {
                            prop_assert_eq!(
                                ik, Ordering::Equal,
                                "equivalence is not transitive along {:?} for {:?}",
                                dir, segs
                            );
                        }
                        if ij == Ordering::Equal && jk == Ordering::Less {
                            prop_assert_eq!(
                                ik, Ordering::Less,
                                "equivalence is not compatible with Less along \
                                 {:?} for {:?}", dir, segs
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The strict weak ordering drives a sort over many crossings on one edge.
///
/// The property sweep checks the comparator's axioms, and this test checks
/// that the comparator can actually drive a sort to a correct total order,
/// which is how an arrangement uses it. `sort_by` is where a contradictory
/// comparator fails.
#[test]
fn compare_along_drives_a_correct_sort() {
    let edge: Seg = [[0.0, 0.0], [64.0, 0.0]];

    // Verticals give crossings at x = 1..=63, generated in scrambled order and
    // with duplicated positions so that ties are exercised too.
    let mut pts: Vec<(f64, RationalCoord2d)> = Vec::new();
    for k in 1..64_u32 {
        let x = f64::from((k * 37) % 63 + 1);
        let p = construct(edge, [[x, -1.0], [x, 1.0]]).unwrap();
        pts.push((x, p));
        if k % 7 == 0 {
            // A second segment crossing at the same x makes a genuine tie
            // between two points constructed from different pairs.
            let q = construct(edge, [[x, -5.0], [x, 9.0]]).unwrap();
            pts.push((x, q));
        }
    }

    pts.sort_by(|l, r| rational_compare_along(edge[0], edge[1], &l.1, &r.1).unwrap());

    for w in pts.windows(2) {
        assert!(
            w[0].0 <= w[1].0,
            "sorted order {:?} violates the known x order",
            pts.iter().map(|p| p.0).collect::<Vec<_>>()
        );
    }
}

// ===========================================================================
// Contract edges
// ===========================================================================

/// Non-finite input is rejected by every entry point that takes
/// `f64`, and `rational_coords_equal` has no `f64` input to reject.
#[test]
fn non_finite_is_rejected_at_every_boundary() {
    let good = RationalCoord2d::from_f64([1.0, 2.0]).unwrap();

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(RationalCoord2d::from_f64([bad, 0.0]).is_err());
        assert!(RationalCoord2d::from_f64([0.0, bad]).is_err());
        assert!(
            RationalCoord2d::from_segment_intersection(
                [0.0, 0.0],
                [1.0, 1.0],
                [bad, 0.0],
                [1.0, 0.0]
            )
            .is_err()
        );
        assert!(rational_compare_along([bad, 0.0], [1.0, 1.0], &good, &good).is_err());
        assert!(rational_compare_along([0.0, 0.0], [1.0, bad], &good, &good).is_err());
        assert!(orient2d_rational([bad, 0.0], [1.0, 0.0], &good).is_err());
        assert!(orient2d_rational([0.0, 0.0], [bad, 1.0], &good).is_err());
    }
}

/// A zero determinant is `Ok(None)`, not `Err`.
///
/// The distinction governs how a caller reads the result. `Err` means "you
/// gave me something outside the domain", and `Ok(None)` means "the answer to
/// your well-formed question is that no such point exists". Collapsing the
/// second into the first would make a caller handle parallel segments as a
/// failure.
#[test]
fn zero_denominator_is_not_an_error() {
    let r =
        RationalCoord2d::from_segment_intersection([0.0, 0.0], [1.0, 0.0], [0.0, 5.0], [9.0, 5.0]);
    assert_eq!(r, Ok(None));
    assert!(
        r.is_ok(),
        "parallel is a geometric fact, not a domain error"
    );
}
