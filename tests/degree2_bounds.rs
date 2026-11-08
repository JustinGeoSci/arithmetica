// tests/degree2_bounds.rs

//! The fast-filter constants of the degree-2 predicates, checked against
//! their derivations in exact rational arithmetic.
//!
//! The constants are crate-private, so their bit patterns are written out
//! here, as `tests/support` writes out the window limits. A test taken from
//! the implementation would follow the implementation if it changed, and
//! these cannot. The unit test `err_bound_constants_have_their_table_values`
//! pins the same bit patterns from the other side.
//!
//! The derivation of `compare_distance`'s bound, on `CMP_DIST_ERR_BOUND_A`
//! in `src/expansion.rs`, closes on two inequalities, with `ε = 2^-53`,
//! `e4 = (1 + ε)⁴ − 1` and `u = 2^-1074`:
//!
//! - `A(1 − ε)² ≥ e4 / ((1 − e4)(1 − 2ε))`, for the relative part;
//! - `(K − u/2)(1 − ε) ≥ c·u` with `c = 2(1 + ε)(1 − ε)/((1 − e4)(1 − 2ε))`,
//!   for the absolute part.
//!
//! `dot_sign`'s constant is `orient2d`'s, by the reduction on
//! `DOT_ERR_BOUND_A`, and the test checks that the two are the same value.

// Test-code lint posture, stated rather than left implicit:
//
//   unreadable_literal: bit patterns are written as the unit test pins them.
#![allow(clippy::unreadable_literal)]

use num_bigint::BigInt;
use num_rational::Ratio;

type Q = Ratio<BigInt>;

fn q(x: f64) -> Q {
    Q::from_float(x).expect("finite")
}

fn int(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

/// `2^-k` exactly.
fn pow2_neg(k: u32) -> Q {
    Q::new(BigInt::from(1), BigInt::from(1) << k)
}

const CMP_DIST_ERR_BOUND_A: u64 = 0x3cc0_0000_0000_0006; // (4 + 48ε)ε
const CMP_DIST_ERR_BOUND_ABS: u64 = 0x0000_0000_0000_0004; // 4 · 2^-1074
const DOT_ERR_BOUND_A: u64 = 0x3cb8_0000_0000_0004; // (3 + 16ε)ε
const CCW_ERR_BOUND_A: u64 = 0x3cb8_0000_0000_0004; // (3 + 16ε)ε

#[test]
fn the_compare_distance_constant_is_its_formula() {
    let eps = pow2_neg(53);
    let a = (int(4) + int(48) * &eps) * &eps;
    assert_eq!(q(f64::from_bits(CMP_DIST_ERR_BOUND_A)), a);
    assert_eq!(
        q(f64::from_bits(CMP_DIST_ERR_BOUND_ABS)),
        int(4) * pow2_neg(1074)
    );
}

#[test]
fn the_compare_distance_relative_bound_holds() {
    let eps = pow2_neg(53);
    let one = int(1);
    let e4 = (&one + &eps).pow(4) - &one;
    let need = &e4 / ((&one - &e4) * (&one - int(2) * &eps));
    let a = q(f64::from_bits(CMP_DIST_ERR_BOUND_A));
    let have = &a * (&one - &eps).pow(2);
    assert!(have >= need, "A(1 − ε)² is below the derived requirement");
    // The margin is real but small, so A is not wildly larger than needed.
    // The least (4 + kε)ε that suffices has k = 39.
    let least = (int(4) + int(39) * &eps) * &eps;
    assert!(&least * (&one - &eps).pow(2) >= need);
    let short = (int(4) + int(38) * &eps) * &eps;
    assert!(&short * (&one - &eps).pow(2) < need);
}

#[test]
fn the_compare_distance_absolute_bound_holds() {
    let eps = pow2_neg(53);
    let one = int(1);
    let u = pow2_neg(1074);
    let e4 = (&one + &eps).pow(4) - &one;
    let c = int(2) * (&one + &eps) * (&one - &eps) / ((&one - &e4) * (&one - int(2) * &eps));
    let k = q(f64::from_bits(CMP_DIST_ERR_BOUND_ABS));
    assert!((&k - &u / int(2)) * (&one - &eps) >= &c * &u);
}

#[test]
fn the_dot_sign_constant_is_the_orient2d_constant() {
    assert_eq!(DOT_ERR_BOUND_A, CCW_ERR_BOUND_A);
    let eps = pow2_neg(53);
    assert_eq!(
        q(f64::from_bits(DOT_ERR_BOUND_A)),
        (int(3) + int(16) * &eps) * &eps
    );
}
