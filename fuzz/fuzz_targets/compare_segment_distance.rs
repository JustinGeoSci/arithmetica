// fuzz_targets/compare_segment_distance.rs

#![no_main]

//! Correctness fuzz target for `compare_segment_distance`.
//!
//! Parses 56 bytes as three 2D points (`p`, `a`, `b`) and `d`, and checks
//! that `compare_segment_distance` returns the sign an exact integer
//! evaluation of `dist^2(p, ab) - d^2` gives. Inputs with a non-finite value
//! are skipped rather than checked. Their contract is `Err(NonFinite)`, which
//! the integration tests check in every argument position.
//!
//! The oracle scales every value, `d` included, by `2^1074`, exact for every
//! finite `f64`. It makes the same exact three-way case choice as the
//! predicate, since the dot products are exact integers. Each endpoint case
//! is the degree-2 point polynomial, scaled by `2^2148`, and the interior
//! case is `cross^2 - d^2·|b - a|^2`, degree 4, scaled by `2^4296`. Both
//! factors are positive, so every case's sign is the true one.

use std::cmp::Ordering;

use arithmetica::{Sign, compare_segment_distance};
use libfuzzer_sys::fuzz_target;
use num_bigint::BigInt;

/// `x * 2^1074` as an exact integer.
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

fn sign_of(v: &BigInt) -> Sign {
    match v.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// Reads value `index` of the input as a little-endian `f64`.
fn value(data: &[u8], index: usize) -> f64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&data[8 * index..8 * index + 8]);
    f64::from_le_bytes(buf)
}

/// The input as bit patterns, the only form that reproduces a failure
/// exactly.
fn describe(v: &[f64; 7]) -> String {
    v.iter()
        .enumerate()
        .map(|(i, x)| format!("  v{i} = {:#018x}  // {:e}", x.to_bits(), x))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The exact sign of `dist^2(p, ab) - d^2`, with `p` first and `d` last.
fn oracle(v: &[f64; 7]) -> Sign {
    let s: Vec<BigInt> = v.iter().map(|&x| exact_int(x)).collect();
    let (px, py, ax, ay, bx, by, d) = (&s[0], &s[1], &s[2], &s[3], &s[4], &s[5], &s[6]);
    let (ux, uy) = (bx - ax, by - ay);
    let (vx, vy) = (px - ax, py - ay);
    let q = if &vx * &ux + &vy * &uy <= BigInt::ZERO {
        (&vx * &vx + &vy * &vy) - d * d
    } else {
        let (wx, wy) = (px - bx, py - by);
        if &wx * &ux + &wy * &uy >= BigInt::ZERO {
            (&wx * &wx + &wy * &wy) - d * d
        } else {
            let cross = &ux * &vy - &uy * &vx;
            &cross * &cross - d * d * (&ux * &ux + &uy * &uy)
        }
    };
    sign_of(&q)
}

fuzz_target!(|data: &[u8]| {
    // Need 56 bytes: 7 values of 8 bytes each.
    if data.len() < 56 {
        return;
    }
    let mut v = [0.0_f64; 7];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = value(data, i);
    }
    if v.iter().any(|x| !x.is_finite()) {
        return;
    }

    let result = compare_segment_distance([v[0], v[1]], [v[2], v[3]], [v[4], v[5]], v[6]);
    let expected = oracle(&v);
    assert_eq!(
        result,
        Ok(expected),
        "compare_segment_distance disagreed with the exact oracle\n{}",
        describe(&v)
    );
});
