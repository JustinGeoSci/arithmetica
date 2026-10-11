// tests/regressions.rs

//! Fixed-input tests at the extremes of the `f64` range.
//!
//! Each test pins one input whose difficulty comes from its magnitudes rather
//! than from near-degeneracy at ordinary scale:
//!
//! - all coordinates small enough that the products of differences inside
//!   the determinant are subnormal, or the exact determinant is below the
//!   smallest positive `f64`;
//! - huge and tiny coordinates in one call, with a bit span far wider than
//!   any predicate's exactness window;
//! - a single subnormal coordinate among coordinates of ordinary magnitude.
//!
//! The first five inputs are given as bit patterns, the only form that
//! reproduces them exactly. The sixth is built by hand from small dyadic
//! numbers, and its doc comment derives the expected sign.
//!
//! The two `orient2d` tests also compare `orient2d` with `orient2d_rational`,
//! the crate's pure-integer evaluation of the same orientation. Every
//! expected sign has been checked against an exact rational evaluation of the
//! determinant; the oracles in `tests/support/mod.rs`, `tests/incircle.rs`,
//! `tests/orient3d.rs` and `tests/insphere.rs` can be pointed at the same
//! inputs to confirm them.

use arithmetica::{
    RationalCoord2d, Sign, incircle, insphere, orient2d, orient2d_rational, orient3d,
};

fn b(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// `orient2d(pa, pb, pc)` against the exact integer evaluation of the same
/// orientation.
fn orient2d_exact_twin(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    let c = RationalCoord2d::from_f64(pc).expect("finite");
    orient2d_rational(pa, pb, &c).expect("finite")
}

// ===========================================================================
// Subnormal products in the filter
// ===========================================================================

/// Coordinates near 1e-161, 1e-148 and 1e-172. Both cross products are about
/// 5.6e-309, below the smallest normal `f64` (about 2.2e-308), so they carry
/// an absolute rounding error that a purely relative error bound does not
/// cover. The two products agree in their leading digits, and the exact
/// determinant, about 2e-325, is smaller than the smallest positive
/// subnormal: no `f64` can hold it, only its sign. The exact sign is
/// positive.
#[test]
fn orient2d_subnormal_products_agree_with_exact_twin() {
    let pa = [b(0x9eb1_78c1_9490_cf8a), b(0x9ea4_c03b_fc8e_e143)];
    let pb = [b(0xa138_d983_652a_ca3c), b(0xa12d_837c_8e5d_3fc4)];
    let pc = [b(0x9c40_abcb_dfc8_9ad9), b(0x1c4a_6c2b_9d10_a362)];
    assert_eq!(orient2d_exact_twin(pa, pb, pc), Sign::Positive);
    assert_eq!(orient2d(pa, pb, pc), Ok(Sign::Positive));
}

/// Four points with coordinates between about 2e-108 and 2e-105. The triple
/// products of differences fall in the subnormal range, and the exact
/// determinant, about 3e-333, is smaller than the smallest positive
/// subnormal. The exact sign is positive.
#[test]
fn orient3d_small_magnitudes_expected_positive() {
    let pa = [
        b(0xa9e6_c00c_925a_b2b7),
        b(0xa9d4_cc93_35d8_6d0d),
        b(0xa9d2_861d_7df0_5b59),
    ];
    let pb = [
        b(0xa994_be4a_e13c_e6ee),
        b(0xa99e_660a_5320_bf5c),
        b(0xa99d_7d0d_9f3d_4534),
    ];
    let pc = [
        b(0x2a35_0d70_f3f5_91fb),
        b(0xaa31_54bb_10fc_2914),
        b(0xaa1a_5a2f_03be_f099),
    ];
    let pd = [
        b(0xa9ff_0094_ab49_249f),
        b(0x2a1c_f9ad_7405_e55b),
        b(0x2a0d_b0e2_f61b_4b00),
    ];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
}

/// Five points with coordinates between about 3e-65 and 7e-63, two of them
/// exactly zero. The fifth-degree products of differences fall in the
/// subnormal range, and the exact determinant, about 8e-330, is smaller than
/// the smallest positive subnormal. The exact sign is positive.
#[test]
fn insphere_small_magnitudes_expected_positive() {
    let pa = [
        b(0xb2dd_ff30_70b7_acac),
        b(0x32d9_0ccc_1e0b_e7cf),
        b(0xb2b4_e1d5_60ea_7ed7),
    ];
    let pb = [
        b(0x328a_35c5_44eb_c9a2),
        b(0xb2a3_1fe6_8ad4_9736),
        b(0x328d_948e_a3dd_a3b6),
    ];
    let pc = [
        b(0x32a6_06b0_a03b_c832),
        b(0xb29c_49ca_a000_0f1d),
        b(0x329b_ef71_5627_3bb0),
    ];
    let pd = [0.0, b(0x3306_0917_a788_8839), 0.0];
    let pe = [
        b(0xb2f3_1e72_46c0_5987),
        b(0xb2d6_9478_2ea4_09fc),
        b(0xb2e5_8d88_67e4_c1d0),
    ];
    assert_eq!(insphere(pa, pb, pc, pd, pe), Ok(Sign::Positive));
}

// ===========================================================================
// Huge and tiny coordinates in one call
// ===========================================================================

/// One point near 1e226, two near 1e-240: the coordinates span about 467
/// orders of magnitude (over 1,500 bits), far wider than the `orient2d`
/// exactness window, so no single power-of-two rescale brings the input
/// inside it. Each cross product is about 8.0e-13 and the two agree in their
/// leading digits; the exact determinant is about 5e-28. The exact sign is
/// negative.
#[test]
fn orient2d_mixed_magnitudes_agree_with_exact_twin() {
    let pa = [b(0x6ef1_7ebb_5adc_2a94), b(0xef06_7f65_958f_a1b3)];
    let pb = [b(0x0e13_fc22_82d5_c45f), b(0x0e03_ff16_9c0b_59f5)];
    let pc = [b(0x0e55_4aad_b967_1574), b(0x8e69_7616_a820_5300)];
    assert_eq!(orient2d_exact_twin(pa, pb, pc), Sign::Negative);
    assert_eq!(orient2d(pa, pb, pc), Ok(Sign::Negative));
}

/// Two points near 1e-181, two near 1e212: the coordinates span about 394
/// orders of magnitude (over 1,300 bits), far wider than the `incircle`
/// exactness window. The exact sign is positive.
#[test]
fn incircle_mixed_magnitudes_expected_positive() {
    let pa = [b(0x1a88_26fd_bca7_5e80), b(0x1a4f_f7ea_1c03_6677)];
    let pb = [b(0x1a6c_f73a_8a9c_a68a), b(0x9a53_eea7_c6b2_e80d)];
    let pc = [b(0x6c03_220c_0950_38fd), b(0x6c0c_11f9_1421_cdc1)];
    let pd = [b(0xec10_0c9b_637a_77a6), b(0x6c03_df5c_a36b_9a75)];
    assert_eq!(incircle(pa, pb, pc, pd), Ok(Sign::Positive));
}

// ===========================================================================
// One subnormal coordinate at ordinary magnitudes
// ===========================================================================

/// A triangle at unit scale in the plane `z = 0`, and a fourth point one
/// subnormal unit `u = 2^-1074` below the origin. Every other coordinate is
/// 0, 1, -1, 0.375 or 0.625.
///
/// Why the exact sign is positive. Relative to `pd = (0, 0, -u)` the x and y
/// differences are the coordinates themselves, and each z difference is
/// `0 - (-u) = u` exactly. The three 2x2 minors are
///
/// ```text
/// bdx*cdy - cdx*bdy = 1*0.375 - 0*0      =  0.375
/// cdx*ady - adx*cdy = 0*0.625 + 1*0.375  =  0.375
/// adx*bdy - bdx*ady = -1*0    - 1*0.625  = -0.625
/// ```
///
/// so the determinant is `u * (0.375 + 0.375 - 0.625) = u / 8 > 0`.
///
/// What makes it hard. In `f64`, `u * 0.375` rounds to zero twice and
/// `u * -0.625` rounds to `-u`, so a plain floating-point evaluation of the
/// determinant gives `-u`, the opposite sign. The magnitude sum is `u`, and a
/// purely relative error bound, about `7.8e-16 * u`, underflows to zero, so
/// it would accept that value. The underflow-protected filter adds an
/// absolute term to its bound and so leaves the sign to the later stages.
/// The input's bits run from `2^-1074` to `2^0`, wider than the `orient3d`
/// exactness window, so the sign comes from exact integer arithmetic.
///
/// Swapping `pa` and `pb` negates the determinant, so the second call is
/// `Negative`.
#[test]
fn orient3d_unit_magnitudes_with_a_subnormal_coordinate() {
    let u = f64::from_bits(1);
    let pa = [-1.0, 0.625, 0.0];
    let pb = [1.0, 0.0, 0.0];
    let pc = [0.0, 0.375, 0.0];
    let pd = [0.0, 0.0, -u];
    assert_eq!(orient3d(pa, pb, pc, pd), Ok(Sign::Positive));
    assert_eq!(orient3d(pb, pa, pc, pd), Ok(Sign::Negative));
}
