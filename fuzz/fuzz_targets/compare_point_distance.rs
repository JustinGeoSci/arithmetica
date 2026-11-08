// fuzz_targets/compare_point_distance.rs

#![no_main]

//! Correctness fuzz target for `compare_point_distance`.
//!
//! Parses 40 bytes as two 2D points and `d`, and checks that
//! `compare_point_distance` returns the sign an exact integer evaluation of
//! `|q - p|^2 - d^2` gives. Inputs with a non-finite value are skipped rather
//! than checked. Their contract is `Err(NonFinite)`, which the integration
//! tests check in every argument position.
//!
//! The oracle scales every value, `d` included, by `2^1074`. The scaling is
//! exact for every finite `f64` and multiplies the degree-2 polynomial by
//! `2^2148`, a positive factor. The `compare_distance` target explains why
//! integer arithmetic is the oracle of choice here.

use std::cmp::Ordering;

use arithmetica::{Sign, compare_point_distance};
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
fn describe(v: &[f64; 5]) -> String {
    v.iter()
        .enumerate()
        .map(|(i, x)| format!("  v{i} = {:#018x}  // {:e}", x.to_bits(), x))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The exact sign of `|q - p|^2 - d^2`, with `p` first and `d` last.
fn oracle(v: &[f64; 5]) -> Sign {
    let s: Vec<BigInt> = v.iter().map(|&x| exact_int(x)).collect();
    let (ux, uy) = (&s[2] - &s[0], &s[3] - &s[1]);
    let q = (&ux * &ux + &uy * &uy) - &s[4] * &s[4];
    sign_of(&q)
}

fuzz_target!(|data: &[u8]| {
    // Need 40 bytes: 5 values of 8 bytes each.
    if data.len() < 40 {
        return;
    }
    let mut v = [0.0_f64; 5];
    for (i, slot) in v.iter_mut().enumerate() {
        *slot = value(data, i);
    }
    if v.iter().any(|x| !x.is_finite()) {
        return;
    }

    let result = compare_point_distance([v[0], v[1]], [v[2], v[3]], v[4]);
    let expected = oracle(&v);
    assert_eq!(
        result,
        Ok(expected),
        "compare_point_distance disagreed with the exact oracle\n{}",
        describe(&v)
    );
});
