// fuzz_targets/compare_distance.rs

#![no_main]

//! Correctness fuzz target for `compare_distance`.
//!
//! Parses 48 bytes as three 2D points (`p`, `q1`, `q2`) and checks that
//! `compare_distance` returns the sign an exact integer evaluation of
//! `|q1 - p|^2 - |q2 - p|^2` gives.
//!
//! Inputs with a non-finite coordinate are skipped rather than checked. Their
//! contract is `Err(NonFinite)`, which the integration tests check in every
//! argument position.
//!
//! The byte-to-`f64` decoding reaches the whole value space, subnormals and
//! mixed magnitudes included. That is where a predicate that assumes an
//! unbounded exponent range goes wrong.
//!
//! # Why the oracle is integer arithmetic
//!
//! The oracle operates on integers rather than rationals. Every finite `f64`
//! is an integer multiple of `2^-1074`, so scaling every coordinate by
//! `2^1074` turns the polynomial into an integer polynomial with the same
//! sign. Evaluating that in `BigInt` is exact and needs none of the GCD
//! reductions an exact rational type performs after every operation. Both
//! forms are exact, and the integer form runs one to two orders of magnitude
//! more executions per second, covering that much more of the input space in
//! a given time.

use std::cmp::Ordering;

use arithmetica::{Sign, compare_distance};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;

/// `x * 2^1074` as an exact integer.
///
/// Every finite `f64` is an integer multiple of `2^-1074`. A normal value is
/// `sig * 2^(be - 1075)` with `sig` the 53-bit significand and `be` the biased
/// exponent, and a subnormal is its 52-bit field times `2^-1074`. Scaling all
/// coordinates by the same positive power of two multiplies the polynomial
/// by a positive number, so the sign of the integer value is the sign of
/// the true one. No floating-point operation takes part in the oracle.
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

/// The input as bit patterns, one point per line, the only form that
/// reproduces a failure exactly.
fn describe(points: &[[f64; 2]]) -> String {
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            format!(
                "  p{i} = [{:#018x}, {:#018x}]  // [{:e}, {:e}]",
                p[0].to_bits(),
                p[1].to_bits(),
                p[0],
                p[1]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The exact sign of `|q1 - p|^2 - |q2 - p|^2`, with `p` first.
fn oracle(p: &[[f64; 2]; 3]) -> Sign {
    let d = |point: usize, axis: usize| exact_int(p[point][axis]) - exact_int(p[0][axis]);
    let (ux, uy, vx, vy) = (d(1, 0), d(1, 1), d(2, 0), d(2, 1));
    let q = (&ux * &ux + &uy * &uy) - (&vx * &vx + &vy * &vy);
    sign_of(&q)
}

fuzz_target!(|data: &[u8]| {
    // Need 48 bytes: 6 coordinates (3 points of 2) of 8 bytes each.
    if data.len() < 48 {
        return;
    }
    let mut points = [[0.0_f64; 2]; 3];
    for (i, point) in points.iter_mut().enumerate() {
        for (axis, value) in point.iter_mut().enumerate() {
            *value = coordinate(data, 2 * i + axis);
        }
    }
    if points.iter().flatten().any(|c| !c.is_finite()) {
        return;
    }

    let result = compare_distance(points[0], points[1], points[2]);
    let expected = oracle(&points);
    assert_eq!(
        result,
        Ok(expected),
        "compare_distance disagreed with the exact oracle\n{}",
        describe(&points)
    );
});
