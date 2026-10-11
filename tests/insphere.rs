// tests/insphere.rs

//! Integration tests for the `insphere` predicate.
//!
//! These tests exercise the public API only, and check the *sign* returned.
//! Which stage produced that sign is not visible here; the unit tests inside
//! `src/insphere.rs` check that cospherical input reaches the exact stage.
//!
//! Test structure:
//!   - Shewchuk canonical cases (clearly inside, clearly outside, exactly on sphere)
//!   - Near-degenerate cases
//!   - Exact integer oracle comparison (via `num-bigint`)
//!   - Property-based tests (proptest, 100k random inputs)
//!   - Non-finite no-panic contract
//!   - Near-cospherical fixed input and generated cases

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

use arithmetica::{Sign, insphere};
use proptest::prelude::*;

// ===========================================================================
// Helpers
// ===========================================================================

/// Compute the exact sign of the insphere 4×4 lifted-paraboloid determinant.
///
/// # Why integer, not rational
///
/// The predicate only needs the determinant's *sign*, and the sign is
/// invariant under multiplication by a positive constant. Scaling every
/// coordinate by `2^1074` turns each finite `f64` into an exact integer (the
/// minimum positive subnormal is `2^-1074`), so the whole determinant can be
/// evaluated in `BigInt` with no rational arithmetic at all.
///
/// Concretely, with `X_i = x_i · 2^1074`:
///   - each coordinate difference is scaled by `2^1074`,
///   - each squared lift by `2^2148`,
///   - each 3×3 minor by `2^3222`,
///   - each `lift · minor` term, and hence the determinant, by `2^5370`.
///
/// Because `2^5370 > 0`, `sign(det_int) == sign(det_rational)` for *all* finite
/// `f64` inputs, including the exactly-on-sphere case where both are zero.
/// This is an exact equality of signs, not an approximation.
///
/// `num-rational` normalizes (runs a Euclidean GCD) after every operation on
/// multi-thousand-bit terms; that work is pure overhead when only the sign is
/// wanted. Dropping it makes this oracle much faster, with provably
/// identical verdicts.
fn insphere_rational(pa: [f64; 3], pb: [f64; 3], pc: [f64; 3], pd: [f64; 3], pe: [f64; 3]) -> Sign {
    use num_bigint::BigInt;

    /// Converts a finite `f64` to the exact integer `x · 2^1074` by decomposing
    /// its IEEE 754 bits. Exact for all finite `f64`.
    fn s(x: f64) -> BigInt {
        let bits = x.to_bits();
        let negative = (bits >> 63) != 0;
        let biased_exp = ((bits >> 52) & 0x7FF) as i32;
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

        // Normal: value·2^1074 = (2^52 | mantissa) << (biased_exp - 1).
        // Subnormal (biased_exp == 0): value·2^1074 = mantissa  (no shift).
        let (significand, shift): (BigInt, usize) = if biased_exp == 0 {
            (BigInt::from(mantissa), 0)
        } else {
            (
                BigInt::from((1u64 << 52) | mantissa),
                (biased_exp - 1) as usize,
            )
        };

        let magnitude = significand << shift;
        if negative { -magnitude } else { magnitude }
    }

    // Coordinate differences relative to pe, each exact and scaled by 2^1074.
    let aex = s(pa[0]) - s(pe[0]);
    let aey = s(pa[1]) - s(pe[1]);
    let aez = s(pa[2]) - s(pe[2]);
    let bex = s(pb[0]) - s(pe[0]);
    let bey = s(pb[1]) - s(pe[1]);
    let bez = s(pb[2]) - s(pe[2]);
    let cex = s(pc[0]) - s(pe[0]);
    let cey = s(pc[1]) - s(pe[1]);
    let cez = s(pc[2]) - s(pe[2]);
    let dex = s(pd[0]) - s(pe[0]);
    let dey = s(pd[1]) - s(pe[1]);
    let dez = s(pd[2]) - s(pe[2]);

    // Squared lifts (scaled by 2^2148).
    let alift = &aex * &aex + &aey * &aey + &aez * &aez;
    let blift = &bex * &bex + &bey * &bey + &bez * &bez;
    let clift = &cex * &cex + &cey * &cey + &cez * &cez;
    let dlift = &dex * &dex + &dey * &dey + &dez * &dez;

    // 2×2 sub-determinants (scaled by 2^2148).
    let ab = &aex * &bey - &aey * &bex;
    let bc = &bex * &cey - &bey * &cex;
    let cd = &cex * &dey - &cey * &dex;
    let da = &dex * &aey - &dey * &aex;
    let ac = &aex * &cey - &aey * &cex;
    let bd = &bex * &dey - &bey * &dex;

    // 3×3 sub-determinants: abc, bcd, cda, dab (scaled by 2^3222).
    let abc = &aez * &bc - &bez * &ac + &cez * &ab;
    let bcd = &bez * &cd - &cez * &bd + &dez * &bc;
    let cda = &cez * &da + &dez * &ac + &aez * &cd;
    let dab = &dez * &ab + &aez * &bd + &bez * &da;

    // det = dlift·abc − clift·dab + blift·cda − alift·bcd   (scaled by 2^5370).
    // Same Formula-A pairing as the adaptive implementation under test.
    let det = &dlift * &abc - &clift * &dab + &blift * &cda - &alift * &bcd;

    use std::cmp::Ordering;
    match det.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

// ===========================================================================
// Shewchuk canonical cases
// ===========================================================================

/// Clearly inside: pe = origin is inside the circumsphere of a regular
/// tetrahedron inscribed in the unit sphere.
///
/// pa=(1,1,1)/√3, pb=(−1,1,−1)/√3, pc=(−1,−1,1)/√3, pd=(1,−1,−1)/√3
/// are the vertices of a regular tetrahedron on the unit sphere (to within
/// the rounding of `1/√3`), in positive `orient3d` order. The circumsphere is
/// the unit sphere; pe=(0,0,0) is clearly inside.
#[test]
fn insphere_canonical_inside() {
    let s = 1.0_f64 / 3.0_f64.sqrt();
    let pa = [s, s, s];
    let pb = [-s, s, -s];
    let pc = [-s, -s, s];
    let pd = [s, -s, -s];
    let pe = [0.0, 0.0, 0.0];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Positive));
}

/// Clearly outside: pe is far from the circumsphere.
#[test]
fn insphere_canonical_outside() {
    let s = 1.0_f64 / 3.0_f64.sqrt();
    let pa = [s, s, s];
    let pb = [-s, s, -s];
    let pc = [-s, -s, s];
    let pd = [s, -s, -s];
    let pe = [10.0, 10.0, 10.0];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Negative));
}

/// Exactly on circumsphere: five points all on the unit sphere.
///
/// The determinant is exactly zero when all five points lie on the same sphere.
/// We use axis-aligned unit-sphere points: pa=(1,0,0), pb=(0,1,0),
/// pc=(0,0,1), pd=(−1,0,0), pe=(0,−1,0).
#[test]
fn insphere_exactly_on_circumsphere() {
    let pa = [1.0_f64, 0.0, 0.0];
    let pb = [0.0_f64, 1.0, 0.0];
    let pc = [0.0_f64, 0.0, 1.0];
    let pd = [-1.0_f64, 0.0, 0.0];
    let pe = [0.0_f64, -1.0, 0.0];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Zero));
}

/// Unit cube vertices: pa=(0,0,0), pb=(1,0,0), pc=(0,1,0), pd=(0,0,1).
/// Circumsphere center = (0.5, 0.5, 0.5), radius = √3/2.
/// pe=(1,1,1): distance from center = √(0.25+0.25+0.25) = √0.75 = √3/2. → on sphere!
#[test]
fn insphere_unit_cube_on_circumsphere() {
    let pa = [0.0_f64, 0.0, 0.0];
    let pb = [1.0_f64, 0.0, 0.0];
    let pc = [0.0_f64, 1.0, 0.0];
    let pd = [0.0_f64, 0.0, 1.0];
    let pe = [1.0_f64, 1.0, 1.0];
    // All five on the circumsphere of the unit tetrahedron.
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Zero));
}

// ===========================================================================
// Near-degenerate cases
// ===========================================================================

/// Near-on-sphere: pe is slightly inside the circumsphere (distance < r by 1e-12).
///
/// Small in absolute terms but not degenerate: the determinant is about
/// 6.2e-12 against a Stage 1 error bound of about 2.5e-14, so the fast filter
/// certifies it. The test pins the sign convention close to the sphere; the
/// near-cospherical cases at the end of this file are the ones that reach the
/// adaptive stages.
#[test]
fn insphere_near_on_sphere_inside() {
    let s = 1.0_f64 / 3.0_f64.sqrt();
    let pa = [s, s, s];
    let pb = [-s, s, -s];
    let pc = [-s, -s, s];
    let pd = [s, -s, -s];
    let r = 1.0_f64 - 1e-12_f64;
    let pe = [r, 0.0, 0.0];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Positive));
}

/// Near-on-sphere: pe is slightly outside the circumsphere.
#[test]
fn insphere_near_on_sphere_outside() {
    let s = 1.0_f64 / 3.0_f64.sqrt();
    let pa = [s, s, s];
    let pb = [-s, s, -s];
    let pc = [-s, -s, s];
    let pd = [s, -s, -s];
    let r = 1.0_f64 + 1e-12_f64;
    let pe = [r, 0.0, 0.0];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Negative));
}

// ===========================================================================
// Antisymmetry
// ===========================================================================

/// Swapping two rows negates the determinant (swapping pa ↔ pb).
#[test]
fn insphere_swap_rows_negates() {
    let s = 1.0_f64 / 3.0_f64.sqrt();
    let pa = [-s, -s, s];
    let pb = [s, s, s];
    let pc = [-s, s, -s];
    let pd = [s, -s, -s];
    let pe = [0.1, 0.0, 0.0]; // Inside but not at center.
    let s1 = insphere(pa, pb, pc, pd, pe).expect("finite inputs");
    let s2 = insphere(pb, pa, pc, pd, pe).expect("finite inputs"); // Swap pa ↔ pb.
    assert_ne!(s1, Sign::Zero, "test requires nonzero determinant");
    assert_ne!(s2, Sign::Zero);
    assert_ne!(s1, s2, "swapping two rows must negate the sign");
}

// ===========================================================================
// Non-finite no-panic contract
// ===========================================================================

#[test]
fn insphere_infinity_returns_err() {
    assert!(
        insphere(
            [f64::INFINITY, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .is_err()
    );
}

#[test]
fn insphere_nan_returns_err() {
    assert!(
        insphere(
            [f64::NAN, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .is_err()
    );
}

#[test]
fn insphere_neg_infinity_returns_err() {
    assert!(
        insphere(
            [f64::NEG_INFINITY, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .is_err()
    );
}

/// `f64::MAX` is finite, so this input is in-domain and must return `Ok`
/// with its exact sign, not an error, even though the lifted products
/// overflow binary64 if formed naively.
#[test]
fn insphere_max_finite_is_ok() {
    assert!(
        insphere(
            [f64::MAX, f64::MAX, f64::MAX],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        )
        .is_ok()
    );
}

// ===========================================================================
// Property-based tests
// ===========================================================================

/// Finite f64 with unbiased exponent in [512, 1021].
///
/// Magnitudes run from about 1.3e154 to 4.5e307. The ceiling 1021 keeps every
/// coordinate and every coordinate difference finite (a difference of two
/// opposite-sign magnitudes `< 2^1022` is `< 2^1023`); the lifts and the
/// higher-degree products overflow binary64 throughout this range.
///
/// The bounds focus this sweep on large magnitudes. They are a choice of
/// regime, not a limit of the predicate: the full finite range, subnormals
/// included, is covered by `tests/boundary.rs` and the fuzz corpus replay.
fn finite_f64_bounded_isp() -> impl Strategy<Value = f64> {
    (prop::bool::ANY, 512_i32..=1021_i32, 0_u64..(1_u64 << 52)).prop_map(|(sign, exp, mantissa)| {
        let bits = ((sign as u64) << 63) | (((exp + 1023) as u64) << 52) | mantissa;
        f64::from_bits(bits)
    })
}

fn point_3d_isp() -> impl Strategy<Value = [f64; 3]> {
    [
        finite_f64_bounded_isp(),
        finite_f64_bounded_isp(),
        finite_f64_bounded_isp(),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// insphere must match the exact oracle on all inputs.
    ///
    /// The oracle is exact for all finite f64 inputs: each coordinate is
    /// converted from its IEEE 754 bits to the exact integer `x · 2^1074`,
    /// and the scaling does not change the determinant's sign.
    #[test]
    fn insphere_matches_rational_oracle(
        pa in point_3d_isp(),
        pb in point_3d_isp(),
        pc in point_3d_isp(),
        pd in point_3d_isp(),
        pe in point_3d_isp(),
    ) {
        let expected = insphere_rational(pa, pb, pc, pd, pe);
        let got = insphere(pa, pb, pc, pd, pe).expect("generator yields finite inputs");
        prop_assert_eq!(
            got, expected,
            "insphere({:?}, {:?}, {:?}, {:?}, {:?}): expected {:?}, got {:?}",
            pa, pb, pc, pd, pe, expected, got
        );
    }

    /// insphere is antisymmetric under row swap when the sign is nonzero.
    #[test]
    fn insphere_antisymmetry(
        pa in point_3d_isp(),
        pb in point_3d_isp(),
        pc in point_3d_isp(),
        pd in point_3d_isp(),
        pe in point_3d_isp(),
    ) {
        let s1 = insphere(pa, pb, pc, pd, pe).expect("finite inputs");
        let s2 = insphere(pb, pa, pc, pd, pe).expect("finite inputs"); // Swap pa ↔ pb.
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

// See the companion test in `tests/orient3d.rs`. Every product of coordinate
// differences underflows to zero in native f64, so the sign has to come from
// exact arithmetic. The insphere determinant carries the lifted column, making
// it homogeneous of odd degree 5 in the coordinates, so an internal rescale by
// a negative factor would invert the result exactly. Each case is checked
// against the exact oracle.
#[test]
fn subnormal_coordinates_agree_with_the_rational_oracle() {
    let s = f64::from_bits(1);
    let cases: [([f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3]); 3] = [
        (
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, s],
            [-s, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        (
            [2.0 * s, s, s],
            [s, 2.0 * s, s],
            [s, s, 2.0 * s],
            [0.0, s, s],
            [s, s, s],
        ),
        (
            [1e-300, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, s],
            [-s, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
    ];
    for (i, (pa, pb, pc, pd, pe)) in cases.iter().enumerate() {
        assert_eq!(
            insphere(*pa, *pb, *pc, *pd, *pe),
            Ok(insphere_rational(*pa, *pb, *pc, *pd, *pe)),
            "case {i}: insphere disagrees with exact rational arithmetic"
        );
    }
}

// ===========================================================================
// Near-cospherical cover
// ===========================================================================

// The generators above draw coordinates with unbiased exponents in [512, 1021],
// i.e. magnitudes from ~1e154 to ~1e307. Five points drawn independently from
// that range are essentially never close to cospherical, so that suite rarely
// visits the narrow band just above the stage error bounds, where an early
// exit has the least margin. Raising its case count would not change that;
// the distribution simply does not reach the regime.
//
// These tests target it directly: five points placed on a common sphere and
// rounded to f64 are *nearly* cospherical, with a determinant tiny relative to
// the permanent but almost never exactly zero, and with full mantissas that
// produce nonzero coordinate tails.

/// A fixed near-cospherical input at ordinary magnitudes.
///
/// The exact determinant is approximately −8.436e-15, so the answer is
/// `Negative`. What makes it hard: a plain `f64` evaluation of the cofactor
/// expansion comes out *positive* (about +7.1e-15), and the Stage 1 error
/// bound (about 3.3e-13) is roughly forty times `|det|`, so the sign has to
/// come from a later stage.
///
/// The expected sign here is not transcribed from a run of the code under
/// test: it is confirmed independently in exact rational arithmetic, by two
/// different expansions of the determinant (the Formula-A cofactor expansion
/// this crate uses, and a full Laplace expansion of the lifted determinant),
/// which agree. The second assertion re-checks it against this file's oracle.
#[test]
fn near_cospherical_fixed_input() {
    let pa = [-1.3759880214153173, -3.1771908316298934, -3.703429122290807];
    let pb = [-3.595324011544156, -2.4622333350978582, -4.962567840975432];
    let pc = [-1.698571545344452, -2.4406069178786627, -3.8825767460797467];
    let pd = [-4.097790528551583, -5.033985288133055, -2.6923846234816025];
    let pe = [-2.1072731929324653, -4.768891783251616, -4.162170191368165];

    assert_eq!(
        insphere(pa, pb, pc, pd, pe),
        Ok(Sign::Negative),
        "the exact determinant is negative (about -8.436e-15)"
    );
    assert_eq!(
        insphere_rational(pa, pb, pc, pd, pe),
        Sign::Negative,
        "oracle must agree; if this fails the oracle changed, not the predicate"
    );
}

/// Five points on a common sphere, rounded to `f64`.
///
/// The rounding is the point: the five points are near-cospherical rather than
/// exactly cospherical, which puts the determinant in the band just above or
/// below the stage error bounds, where an unsound early exit would show
/// itself.
///
/// Radii and centers are kept at ordinary magnitudes (|center| ≤ 4, radius in
/// [0.5, 4.5]) so that no intermediate overflows and the regime stays the one
/// real geometry actually produces.
fn near_cospherical_five()
-> impl Strategy<Value = ([f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3])> {
    // Directions are built from spherical angles rather than drawn from a cube
    // and filtered to the unit ball. Rejection sampling would discard about
    // half of each direction, and five independent directions compound to a
    // ~4% acceptance rate, which exhausts proptest's local rejection limit
    // long before the case count is reached. Constructing only valid values
    // avoids filtering altogether.
    //
    // `u` is the cosine of the polar angle, so (sqrt(1-u^2)*cos(t),
    // sqrt(1-u^2)*sin(t), u) is a unit vector for every u in [-1, 1] and every
    // t, with no value to reject.
    let dir = (-1.0_f64..=1.0, 0.0_f64..core::f64::consts::TAU).prop_map(|(u, t)| {
        let r = (1.0 - u * u).max(0.0).sqrt();
        (r * t.cos(), r * t.sin(), u)
    });

    (
        (-4.0_f64..4.0, -4.0_f64..4.0, -4.0_f64..4.0),
        0.5_f64..4.5,
        [dir.clone(), dir.clone(), dir.clone(), dir.clone(), dir],
    )
        .prop_map(|(center, r, dirs)| {
            let place =
                |(x, y, z): (f64, f64, f64)| [center.0 + r * x, center.1 + r * y, center.2 + r * z];
            (
                place(dirs[0]),
                place(dirs[1]),
                place(dirs[2]),
                place(dirs[3]),
                place(dirs[4]),
            )
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]

    /// `insphere` must match the exact oracle on near-cospherical input.
    ///
    /// These inputs sit close to the error bounds of Stages 1-3, so this
    /// checks every early exit against exact arithmetic where the bounds have
    /// the least margin.
    #[test]
    fn insphere_near_cospherical_matches_oracle(
        (pa, pb, pc, pd, pe) in near_cospherical_five()
    ) {
        let got = insphere(pa, pb, pc, pd, pe).expect("generator yields finite inputs");
        let expected = insphere_rational(pa, pb, pc, pd, pe);
        prop_assert_eq!(
            got, expected,
            "insphere({:?}, {:?}, {:?}, {:?}, {:?}): expected {:?}, got {:?}",
            pa, pb, pc, pd, pe, expected, got
        );
    }
}
