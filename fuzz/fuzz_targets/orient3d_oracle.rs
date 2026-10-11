// fuzz/fuzz_targets/orient3d_oracle.rs

#![no_main]

//! Correctness fuzz target for `orient3d`.
//!
//! Parses 96 bytes as four 3D points (`pa`, `pb`, `pc`, `pd`) and checks that
//! `orient3d` returns the sign an exact integer evaluation of the determinant
//! gives.
//!
//! Inputs with a non-finite coordinate are skipped; the contract for those is
//! `Err(NonFinite)`, which the unit tests cover.
//!
//! The byte-to-`f64` decoding reaches the whole value space, subnormals and
//! mixed magnitudes included, which is where a predicate that assumes an
//! unbounded exponent range goes wrong. The oracle is the same integer
//! evaluation as in `orient2d_oracle`; see that target for why.

use std::cmp::Ordering;

use arithmetica::{Sign, orient3d};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;

/// `x * 2^1074` as an exact integer.
///
/// Every finite `f64` is an integer multiple of `2^-1074`: a normal value is
/// `sig * 2^(be - 1075)` with `sig` the 53-bit significand and `be` the biased
/// exponent, and a subnormal is its 52-bit field times `2^-1074`. Scaling all
/// coordinates by the same positive power of two multiplies the determinant
/// by a positive number, so the sign of the integer determinant is the sign
/// of the true one. No floating-point operation takes part in the oracle.
fn exact_int(x: f64) -> BigInt {
    let bits = x.to_bits();
    let negative = (bits >> 63) != 0;
    let biased = (bits >> 52) & 0x7FF;
    let field = bits & 0x000F_FFFF_FFFF_FFFF;
    let magnitude = if biased == 0 {
        BigInt::from(field)
    } else {
        BigInt::from(field | (1_u64 << 52)) << (biased - 1)
    };
    if negative { -magnitude } else { magnitude }
}

fn sign_of(det: &BigInt) -> Sign {
    match det.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// Reads coordinate `index` of the input as a little-endian `f64`.
fn coordinate(data: &[u8], index: usize) -> f64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&data[8 * index..8 * index + 8]);
    f64::from_le_bytes(buf)
}

/// The input as bit patterns, one point per line: the only form that
/// reproduces a failure exactly.
fn describe(points: &[[f64; 3]]) -> String {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            format!(
                "  p{i} = [{:#018x}, {:#018x}, {:#018x}]  // [{:e}, {:e}, {:e}]",
                p[0].to_bits(),
                p[1].to_bits(),
                p[2].to_bits(),
                p[0],
                p[1],
                p[2]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The exact sign of the 3x3 determinant of the differences relative to `pd`.
fn oracle(p: &[[f64; 3]; 4]) -> Sign {
    let d = |point: usize, axis: usize| exact_int(p[point][axis]) - exact_int(p[3][axis]);
    let (adx, ady, adz) = (d(0, 0), d(0, 1), d(0, 2));
    let (bdx, bdy, bdz) = (d(1, 0), d(1, 1), d(1, 2));
    let (cdx, cdy, cdz) = (d(2, 0), d(2, 1), d(2, 2));
    let det = adz * (&bdx * &cdy - &bdy * &cdx)
        + bdz * (&cdx * &ady - &cdy * &adx)
        + cdz * (&adx * &bdy - &ady * &bdx);
    sign_of(&det)
}

fuzz_target!(|data: &[u8]| {
    // Need 96 bytes: 12 coordinates (4 points of 3) of 8 bytes each.
    if data.len() < 96 {
        return;
    }
    let mut points = [[0.0_f64; 3]; 4];
    for (i, point) in points.iter_mut().enumerate() {
        for (axis, value) in point.iter_mut().enumerate() {
            *value = coordinate(data, 3 * i + axis);
        }
    }
    if points.iter().flatten().any(|c| !c.is_finite()) {
        return;
    }

    let result = orient3d(points[0], points[1], points[2], points[3]);
    let expected = oracle(&points);
    assert_eq!(
        result,
        Ok(expected),
        "orient3d disagreed with the exact oracle\n{}",
        describe(&points)
    );
});
