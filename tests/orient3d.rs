// tests/orient3d.rs

//! Integration tests for the `orient3d` predicate.
//!
//! These tests exercise the public API only, and check the *sign* returned.
//! Which stage produced that sign is not visible here; the unit tests inside
//! `src/orient3d.rs` check that near-coplanar input reaches the adaptive
//! stages.
//!
//! Test structure:
//!   - Shewchuk canonical cases (clearly positive, negative, coplanar)
//!   - Near-degenerate and extreme-magnitude cases
//!   - Rational oracle comparison (via `num-rational` / `num-bigint`)
//!   - Property-based tests (proptest, 100k random inputs)
//!   - Non-finite no-panic contract
//!   - Exactly-coplanar and one-ulp-off-coplanar generated cases

// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp          — an exact-value assertion IS a float comparison. These
//                        tests assert bit-exact results against a rational
//                        oracle or a hand-computed value; a tolerance would
//                        defeat their purpose.
//   unreadable_literal — bit patterns and reference coordinates are
//                        transcribed as written in the source they come from,
//                        so that they can be checked against it.
//   cast_*             — fixture construction, where the ranges are known by
//                        construction.
//   unwrap_used        — in a test, a panic is the failure signal.
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
    clippy::comparison_chain
)]

use arithmetica::{Sign, orient3d};

// ===========================================================================
// Helpers
// ===========================================================================

/// Compute the exact sign of the orient3d 3×3 determinant using rational
/// arithmetic via `num_rational::BigRational`.
///
/// Sign = sign of:
///   | adx  ady  adz |
///   | bdx  bdy  bdz |
///   | cdx  cdy  cdz |
/// where adx = pa.x - pd.x, etc.
///
/// The conversion from `f64` to `BigRational` is performed by bit-level
/// decomposition of the IEEE 754 representation rather than `from_float`,
/// so no `num-traits` dependency is needed and the result is exact for all
/// finite `f64`.
fn orient3d_rational(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3]) -> Sign {
    use num_bigint::BigInt;
    use num_rational::BigRational;

    /// Converts a finite `f64` to an exact `BigRational` by decomposing its
    /// IEEE 754 bits. The result equals `x` exactly for all finite inputs.
    fn r(x: f64) -> BigRational {
        let bits = x.to_bits();
        let negative = (bits >> 63) != 0;
        let biased_exp = ((bits >> 52) & 0x7FF) as i32;
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

        let (significand, exp): (BigInt, i32) = if biased_exp == 0 {
            // Subnormal: value = mantissa × 2^-1074
            (BigInt::from(mantissa), -1074)
        } else {
            // Normal: value = (2^52 + mantissa) × 2^(biased_exp - 1075)
            (BigInt::from((1u64 << 52) | mantissa), biased_exp - 1075)
        };

        let magnitude = if exp >= 0 {
            BigRational::from_integer(significand * BigInt::from(2u64).pow(exp as u32))
        } else {
            BigRational::new(significand, BigInt::from(2u64).pow((-exp) as u32))
        };

        if negative { -magnitude } else { magnitude }
    }

    let adx = r(pa[0]) - r(pd[0]);
    let ady = r(pa[1]) - r(pd[1]);
    let adz = r(pa[2]) - r(pd[2]);
    let bdx = r(pb[0]) - r(pd[0]);
    let bdy = r(pb[1]) - r(pd[1]);
    let bdz = r(pb[2]) - r(pd[2]);
    let cdx = r(pc[0]) - r(pd[0]);
    let cdy = r(pc[1]) - r(pd[1]);
    let cdz = r(pc[2]) - r(pd[2]);

    let bc = bdx.clone() * cdy.clone() - bdy.clone() * cdx.clone();
    let ca = cdx.clone() * ady.clone() - cdy.clone() * adx.clone();
    let ab = adx.clone() * bdy.clone() - ady.clone() * bdx.clone();
    let det = adz * bc + bdz * ca + cdz * ab;

    use std::cmp::Ordering;
    match det.cmp(&BigRational::from_integer(BigInt::ZERO)) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

// ===========================================================================
// Shewchuk canonical cases
// ===========================================================================

/// Clearly positive: pd lies below the plane through pa, pb, pc.
#[test]
fn orient3d_canonical_positive() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.0_f64, 0.0, -1.0];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
}

/// Clearly negative: pd lies above the plane through pa, pb, pc.
#[test]
fn orient3d_canonical_negative() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.0_f64, 0.0, 1.0];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Negative));
}

/// Exactly coplanar: all four points in the z = 0 plane.
#[test]
fn orient3d_exactly_coplanar() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [2.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 3.0, 0.0];
    let pd = [1.0_f64, 1.0, 0.0];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Zero));
}

/// Antipodal coplanar: two pairs of opposite points in the z = 0 plane.
#[test]
fn orient3d_antipodal_coplanar() {
    let pa = [1.0_f64, 0.0, 0.0];
    let pb = [-1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.0_f64, -1.0, 0.0];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Zero));
}

// ===========================================================================
// Near-degenerate cases
// ===========================================================================

/// Near-coplanar: pd is 1e-12 above the z = 0 plane.
///
/// Small in absolute terms but not degenerate: every `z` difference is
/// ±1e-12 and the determinant is −1e-12, far above Stage 1's error
/// bound (≈1e-27 here), so the fast filter certifies it. The test pins the
/// sign convention at small separation; the one-ulp-off-coplanar property
/// test below is the one that reaches the adaptive stages.
#[test]
fn orient3d_near_coplanar_above() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.5_f64, 0.25, 1e-12];
    // pd is above the plane → Negative.
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Negative));
}

/// Near-coplanar: pd is 1e-12 below the z = 0 plane.
#[test]
fn orient3d_near_coplanar_below() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.5_f64, 0.25, -1e-12];
    // pd is below the plane → Positive.
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
}

/// Extreme-scale positive: all coordinates near f64 MAX / 4.
#[test]
fn orient3d_large_scale_positive() {
    let s = f64::MAX / 4.0;
    let pa = [s, 0.0, 0.0];
    let pb = [0.0, s, 0.0];
    let pc = [0.0, 0.0, s];
    let pd = [s / 2.0, s / 2.0, -s];
    // pa, pb, pc define a plane and pd is well off it. Not a near-zero test:
    // the coordinate differences are finite, but their products overflow
    // binary64 if formed naively, and the predicate must still return a
    // nonzero sign.
    let result = orient3d(pa, pb, pc, pd).expect("finite inputs");
    assert_ne!(result, Sign::Zero);
}

/// Subnormal coordinates: all values near f64::MIN_POSITIVE.
#[test]
fn orient3d_subnormal_coordinates() {
    let tiny = f64::MIN_POSITIVE;
    let pa = [tiny, 0.0, 0.0];
    let pb = [0.0, tiny, 0.0];
    let pc = [0.0, 0.0, tiny];
    let pd = [tiny / 2.0, tiny / 2.0, -tiny];
    // Same topology as the large-scale case → not Zero.
    let result = orient3d(pa, pb, pc, pd).expect("finite inputs");
    assert_ne!(result, Sign::Zero);
}

// ===========================================================================
// Antisymmetry
// ===========================================================================

/// Swapping two rows negates the determinant.
#[test]
fn orient3d_swap_rows_negates() {
    let pa = [1.0_f64, 2.0, 3.0];
    let pb = [4.0_f64, 5.0, 6.0];
    let pc = [7.0_f64, 8.0, 10.0];
    let pd = [0.0_f64, 0.0, 0.0];
    let s1 = orient3d(pa, pb, pc, pd).expect("finite inputs");
    let s2 = orient3d(pb, pa, pc, pd).expect("finite inputs"); // swap pa ↔ pb → negate
    assert_ne!(s1, Sign::Zero, "test requires nonzero determinant");
    assert_ne!(s2, Sign::Zero);
    assert_ne!(s1, s2, "swapping two rows must negate the sign");
}

// ===========================================================================
// Non-finite no-panic contract
// ===========================================================================

#[test]
fn orient3d_infinity_returns_err() {
    assert!(
        orient3d(
            [f64::INFINITY, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .is_err()
    );
}

#[test]
fn orient3d_nan_returns_err() {
    assert!(
        orient3d(
            [f64::NAN, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .is_err()
    );
}

#[test]
fn orient3d_neg_infinity_returns_err() {
    assert!(
        orient3d(
            [f64::NEG_INFINITY, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        )
        .is_err()
    );
}

// ===========================================================================
// Property-based tests
// ===========================================================================

use proptest::prelude::*;

/// Finite f64 with unbiased exponent in [692, 1021].
///
/// `exp` is the unbiased exponent; `exp + 1023` is the 11-bit biased field
/// (finite range 0..=2046, 2047 = inf/NaN). The ceiling is 1021 so that every
/// generated value, and every difference `pa - pd` the oracle and the
/// predicate form from a pair of them, is finite:
///   * `exp + 1023 <= 2044 < 2047` ⇒ no coordinate is inf/NaN, and the
///     biased field never spills into the sign bit.
///   * A difference of two opposite-sign magnitudes `< 2^(exp+1)` is
///     `< 2^(exp+2)`; finiteness needs `exp + 2 <= 1023`, i.e. `exp <= 1021`.
///     At `exp >= 1022` a difference can overflow to inf, which the predicate
///     resolves exactly but which this comparison does not need to exercise.
///
/// The bounds focus this sweep on large magnitudes (about 2e208 to 4.5e307),
/// where the products of differences overflow binary64. They are a choice of
/// regime, not a limit of the predicate: the full finite range, subnormals
/// included, is covered by `tests/boundary.rs` and the fuzz corpus replay.
fn finite_f64_bounded_3d() -> impl Strategy<Value = f64> {
    (prop::bool::ANY, 692_i32..=1021_i32, 0_u64..(1_u64 << 52)).prop_map(|(sign, exp, mantissa)| {
        let bits = ((sign as u64) << 63) | (((exp + 1023) as u64) << 52) | mantissa;
        f64::from_bits(bits)
    })
}

fn point_3d() -> impl Strategy<Value = [f64; 3]> {
    [
        finite_f64_bounded_3d(),
        finite_f64_bounded_3d(),
        finite_f64_bounded_3d(),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// orient3d must match the rational oracle on all inputs.
    ///
    /// The rational oracle is exact for all finite f64 inputs (BigRational
    /// is constructed from the exact IEEE 754 bit decomposition, so it matches
    /// the mathematical value exactly).
    #[test]
    fn orient3d_matches_rational_oracle(
        pa in point_3d(),
        pb in point_3d(),
        pc in point_3d(),
        pd in point_3d(),
    ) {
        let expected = orient3d_rational(pa, pb, pc, pd);
        // The generator yields only finite coordinates, so the strict predicate
        // never returns Err here; unwrap to compare the sign against the oracle.
        let got = orient3d(pa, pb, pc, pd).expect("generator yields finite inputs");
        prop_assert_eq!(
            got, expected,
            "orient3d({:?}, {:?}, {:?}, {:?}): expected {:?}, got {:?}",
            pa, pb, pc, pd, expected, got
        );
    }

    /// orient3d is antisymmetric under row swap when the sign is nonzero.
    #[test]
    fn orient3d_antisymmetry(
        pa in point_3d(),
        pb in point_3d(),
        pc in point_3d(),
        pd in point_3d(),
    ) {
        let s1 = orient3d(pa, pb, pc, pd).expect("generator yields finite inputs");
        let s2 = orient3d(pb, pa, pc, pd).expect("generator yields finite inputs");
        // If nonzero, must be opposite signs.
        if s1 != Sign::Zero {
            prop_assert_ne!(s1, s2);
        } else {
            // If s1 is Zero, the determinant is exactly zero; s2 must also be Zero.
            prop_assert_eq!(s2, Sign::Zero);
        }
    }
}

// ===========================================================================
// Deep-subnormal coordinates
// ===========================================================================

// Every nonzero coordinate below is a small multiple of 2^-1074 (or, in the
// mixed case, a normal value beside such companions). Every product of two
// coordinate differences is far below the smallest subnormal and underflows to
// zero in native f64, so the sign has to come from exact arithmetic. The
// determinant is homogeneous of odd degree 3 in the coordinates, so an
// internal rescale by a negative factor would invert the result exactly.
//
// Each case is checked against the rational oracle rather than a baked
// expectation, so the test states the property (agree with exact arithmetic)
// rather than a remembered answer.
#[test]
fn subnormal_coordinates_agree_with_the_rational_oracle() {
    let s = f64::from_bits(1); // 2^-1074, the minimum subnormal
    let big = f64::from_bits(1 << 20); // a deeper-mantissa subnormal
    let cases: [([f64; 3], [f64; 3], [f64; 3], [f64; 3]); 6] = [
        ([s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, s], [0.0, s, 0.0], [s, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([3.0 * s, s, s], [s, 3.0 * s, s], [s, s, 3.0 * s], [s, s, s]),
        (
            [big, 0.0, 0.0],
            [0.0, big, 0.0],
            [0.0, 0.0, big],
            [0.0, 0.0, 0.0],
        ),
        // Mixed: one normal coordinate (1e-300) beside subnormal companions,
        // spanning bits 2^-997 down to 2^-1074 at the bottom of the range.
        (
            [1e-300, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, s],
            [0.0, 0.0, 0.0],
        ),
        // Exactly coplanar at subnormal scale must still read Zero.
        (
            [s, 0.0, 0.0],
            [2.0 * s, 0.0, 0.0],
            [3.0 * s, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
    ];
    for (i, (pa, pb, pc, pd)) in cases.iter().enumerate() {
        assert_eq!(
            orient3d(*pa, *pb, *pc, *pd),
            Ok(orient3d_rational(*pa, *pb, *pc, *pd)),
            "case {i}: orient3d disagrees with exact rational arithmetic"
        );
    }
}

// ===========================================================================
// Degenerate-plane cover
// ===========================================================================

// `finite_f64_bounded_3d` draws unbiased exponents in [692, 1021], so every
// coordinate it produces lies between about 2e208 and 4.5e307. Four points
// drawn independently from that range are essentially never close to
// coplanar, so the 100,000-case suite above says little about the degenerate
// regime, which is the regime the predicate exists to get right. Raising its
// case count would not change that; only changing where the cases come from
// does.
//
// The generators below build that regime directly: points exactly on a
// plane, and the same points with one coordinate moved by one ulp.

/// A dyadic value in [-4, 4) with a 1/256 step, exactly representable, so a
/// sum of two of them is exact.
fn dyadic(seed: u64) -> f64 {
    ((seed % 2048) as f64) / 256.0 - 4.0
}

/// Four points on the plane `z = x + y`, which is exact for dyadic `x` and `y`.
///
/// The determinant is therefore **exactly zero**, a configuration random
/// sampling essentially never produces, and the one where a predicate must
/// return `Sign::Zero` rather than a plausible nonzero sign.
fn exactly_coplanar_points() -> impl Strategy<Value = ([f64; 3], [f64; 3], [f64; 3], [f64; 3])> {
    [
        (0_u64..2048, 0_u64..2048),
        (0_u64..2048, 0_u64..2048),
        (0_u64..2048, 0_u64..2048),
        (0_u64..2048, 0_u64..2048),
    ]
    .prop_map(|pairs| {
        let pt = |(sx, sy): (u64, u64)| {
            let (x, y) = (dyadic(sx), dyadic(sy));
            [x, y, x + y]
        };
        (pt(pairs[0]), pt(pairs[1]), pt(pairs[2]), pt(pairs[3]))
    })
}

/// The next representable `f64` away from zero.
fn ulp_away_from_zero(x: f64) -> f64 {
    let bits = x.to_bits();
    f64::from_bits(if x >= 0.0 { bits + 1 } else { bits - 1 })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]

    /// Exactly-coplanar input must return exactly `Sign::Zero`.
    ///
    /// No earlier stage can return `Zero` on its own, so each of these inputs
    /// is confirmed by Stage 4. Verified against the exact oracle rather than
    /// against the construction, so a flaw in the construction shows up as an
    /// oracle disagreement rather than as a test that agrees with itself.
    #[test]
    fn orient3d_exactly_coplanar_returns_zero(
        (pa, pb, pc, pd) in exactly_coplanar_points()
    ) {
        let expected = orient3d_rational(pa, pb, pc, pd);
        prop_assume!(expected == Sign::Zero);
        prop_assert_eq!(
            orient3d(pa, pb, pc, pd), Ok(Sign::Zero),
            "exactly coplanar: {:?} {:?} {:?} {:?}", pa, pb, pc, pd
        );
    }

    /// One ulp off an exactly-coplanar configuration must resolve correctly.
    ///
    /// The determinant is as small as a one-ulp move of `pd` can make it
    /// without being zero, so Stage 1 almost never certifies these inputs and
    /// the expansion stages (and, for a small fraction, Stage 4) decide them.
    /// It is the case that distinguishes a sound adaptive ladder from one that
    /// returns a plausible sign.
    #[test]
    fn orient3d_one_ulp_off_coplanar_matches_oracle(
        (pa, pb, pc, mut pd) in exactly_coplanar_points()
    ) {
        pd[2] = ulp_away_from_zero(pd[2]);
        let expected = orient3d_rational(pa, pb, pc, pd);
        let got = orient3d(pa, pb, pc, pd).expect("finite inputs");
        prop_assert_eq!(
            got, expected,
            "one ulp off plane: {:?} {:?} {:?} {:?}", pa, pb, pc, pd
        );
    }
}
