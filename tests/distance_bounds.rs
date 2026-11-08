// tests/distance_bounds.rs

//! The fast-filter constants of `compare_point_distance` and
//! `compare_segment_distance`, checked against their derivations in exact
//! rational arithmetic.
//!
//! The constants are crate-private, so their bit patterns are written out
//! here, as `tests/support` writes out the window limits. A test taken from
//! the implementation would follow the implementation if it changed, and
//! these cannot. The unit test `err_bound_constants_have_their_table_values`
//! pins the same bit patterns from the other side.
//!
//! Each derivation (on the constants in `src/expansion.rs`) is a numbered
//! chain of absorptions in Higham's `γk = kε/(1 − kε)` notation, ending in
//! one closing condition on the constant. Every link that is a statement
//! about rationals is verified here exactly. That covers the absorptions
//! with their named coefficients and the closing condition, together with
//! the margin, the least constant of the stated form that satisfies the
//! closing condition.

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

/// `γk = kε/(1 − kε)`.
fn gamma(k: i64) -> Q {
    let eps = pow2_neg(53);
    let ke = int(k) * &eps;
    &ke / (int(1) - &ke)
}

const PT_DIST_ERR_BOUND_A: u64 = 0x3cc4_0000_0000_0009; // (5 + 72ε)ε
const SEG_DIST_ERR_BOUND_A: u64 = 0x3ce1_0000_0000_0001; // (17 + 32ε)ε

#[test]
fn the_constants_are_their_formulas() {
    let eps = pow2_neg(53);
    assert_eq!(
        q(f64::from_bits(PT_DIST_ERR_BOUND_A)),
        (int(5) + int(72) * &eps) * &eps
    );
    assert_eq!(
        q(f64::from_bits(SEG_DIST_ERR_BOUND_A)),
        (int(17) + int(32) * &eps) * &eps
    );
}

/// The point form's absorptions, numbered as in the derivation on
/// `PT_DIST_ERR_BOUND_A`, as rational inequalities.
#[test]
fn the_point_derivations_absorptions_hold() {
    let eps = pow2_neg(53);
    let one = int(1);
    // (4) (1 + γ1)γ4 ≤ γ5, (1 + γ1)γ1 ≤ γ2, (1 + γ1)(3 + 2ε)ε ≤ 4ε.
    assert!((&one + gamma(1)) * gamma(4) <= gamma(5));
    assert!((&one + gamma(1)) * gamma(1) <= gamma(2));
    assert!((&one + gamma(1)) * (int(3) + int(2) * &eps) * &eps <= int(4) * &eps);
    // (5) Each mag term keeps (1 − γ3) of its value and (1 − ε) of a u_N,
    // and the two outer additions cost (1 − ε)². Three u_N(1 − ε)³
    // dominate 2u_N.
    assert!((&one - &eps).pow(2) * (&one - gamma(3)) >= &one - gamma(5));
    assert!(int(3) * (&one - &eps).pow(3) >= int(2));
    // (6) γ2 ≤ γ5 and 4ε ≤ 2γ5.
    assert!(gamma(2) <= gamma(5));
    assert!(int(4) * &eps <= int(2) * gamma(5));
}

/// The point form's closing condition (7),
/// `A(1 − ε)² ≥ γ5/((1 − γ5)(1 − γ1))`. The bound computation's own
/// roundings are covered because `u_S ≥ ε·u_N`.
#[test]
fn the_point_closing_condition_holds_with_its_stated_margin() {
    let eps = pow2_neg(53);
    let one = int(1);
    let need = gamma(5) / ((&one - gamma(5)) * (&one - gamma(1)));
    let a = q(f64::from_bits(PT_DIST_ERR_BOUND_A));
    assert!(
        &a * (&one - &eps).pow(2) >= need,
        "A(1 − ε)² is below the requirement"
    );
    // The margin: the least (5 + kε)ε that suffices has k = 66.
    let least = (int(5) + int(66) * &eps) * &eps;
    assert!(&least * (&one - &eps).pow(2) >= need);
    let short = (int(5) + int(65) * &eps) * &eps;
    assert!(&short * (&one - &eps).pow(2) < need);
    // u_S = 2^-1074 ≥ ε·u_N = 2^-1075.
    assert!(pow2_neg(1074) >= pow2_neg(53) * pow2_neg(1022));
}

/// The segment form's absorptions, numbered as in the derivation on
/// `SEG_DIST_ERR_BOUND_A`, as rational inequalities.
#[test]
fn the_segment_derivations_absorptions_hold() {
    let eps = pow2_neg(53);
    let one = int(1);
    // (1) γ4·P₀ + 3ε·u_N ≤ γ5·(P₀ + 2u_N), coefficientwise.
    assert!(gamma(4) <= gamma(5));
    assert!(int(3) * &eps <= int(2) * gamma(5));
    // (2) As the point form's (5), with two inner terms.
    assert!((&one - &eps).pow(2) * (&one - gamma(3)) >= &one - gamma(5));
    assert!(int(2) * (&one - &eps).pow(3) >= int(2) * (&one - gamma(5)));
    // (3) (2 + γ5)γ5 + γ1(1 + γ5)² ≤ γ12.
    assert!((int(2) + gamma(5)) * gamma(5) + gamma(1) * (&one + gamma(5)).pow(2) <= gamma(12));
    // (4) ma ≥ (1 − γ2)·cross_mag² and ma ≥ u_N(1 − γ2). The two roundings
    // and the η of the square cost at most (1 − ε)² on the square (the η is
    // dominated by the u_N term) and (1 − ε)² ≥ (1 − γ2) on u_N.
    assert!((&one - &eps).pow(2) >= &one - gamma(2));
    // (5) The coefficients, from the product expansion with absorptions of
    // the kind in (1). The relative ones are γ1(1 + γ4) + γ4 ≤ γ6, then
    // ×(1 + γ1) ≤ γ7. The absolute coefficients are 1·(1 + γ4) ≤ 2 (on
    // u_N·L, then 3 after the final ×(1 + γ1) and the trailing η), and
    // 3(1 + γ1) + 2 ≤ 5 on u_N·D2, 2 on u_N.
    assert!(gamma(1) * (&one + gamma(4)) + gamma(4) <= gamma(6));
    assert!(gamma(6) * (&one + gamma(1)) <= gamma(7));
    assert!((&one + gamma(4)) * (&one + gamma(1)) + &eps <= int(3));
    assert!(int(3) * (&one + gamma(1)) + int(1) <= int(5));
    // (6) The product's factors are (1 − γ2)(1 − γ5) ≥ 1 − γ7, the
    // doubled u_N term 2(1 − γ2)(1 − γ3) ≥ 2(1 − γ5), and the single
    // (1 − ε)²(1 − γ5) ≥ 1 − γ7. The outer roundings (1 − ε)² keep
    // (1 − γ9) of all of it, and 7.5ε covers 3ε + 2.5ε + 2ε.
    assert!((&one - gamma(2)) * (&one - gamma(5)) >= &one - gamma(7));
    assert!(int(2) * (&one - gamma(2)) * (&one - gamma(3)) >= int(2) * (&one - gamma(5)));
    assert!((&one - &eps).pow(2) * (&one - gamma(5)) >= &one - gamma(7));
    assert!((&one - gamma(7)) * (&one - &eps).pow(2) >= &one - gamma(9));
    let half = Q::new(BigInt::from(1), BigInt::from(2));
    assert!(
        int(3) + int(5) * &half
            >= int(3) + int(2) + &half
    );
    assert!(int(7) + &half >= int(3) + (int(2) + &half) + int(2));
}

/// The segment form's closing condition (7),
/// `A(1 − ε)² ≥ (1 + γ1)·max(C_a, C_b)/((1 − ε)(1 − γ1))`, with
/// `C_a = γ12/((1 − γ5)²(1 − γ2)) + ε/(1 − γ2)` and
/// `C_b = (γ7 + 7.5ε)/(1 − γ9) + 2ε/(1 − γ2)`.
#[test]
fn the_segment_closing_condition_holds_with_its_stated_margin() {
    let eps = pow2_neg(53);
    let one = int(1);
    let half = Q::new(BigInt::from(1), BigInt::from(2));
    let c_a = gamma(12) / ((&one - gamma(5)).pow(2) * (&one - gamma(2))) + &eps / (&one - gamma(2));
    let c_b = (gamma(7) + (int(7) + &half) * &eps) / (&one - gamma(9))
        + int(2) * &eps / (&one - gamma(2));
    let worst = if c_a > c_b { c_a } else { c_b };
    let need = (&one + gamma(1)) * worst / ((&one - &eps) * (&one - gamma(1)));
    let a = q(f64::from_bits(SEG_DIST_ERR_BOUND_A));
    assert!(
        &a * (&one - &eps).pow(2) >= need,
        "A(1 − ε)² is below the requirement"
    );
    // The margin: base 17 suffices with no ε term, and base 16 cannot be
    // rescued by one (the requirement is just over 16.5ε).
    let least = int(17) * &eps;
    assert!(&least * (&one - &eps).pow(2) >= need);
    let short = (int(16) + int(4096) * &eps) * &eps;
    assert!(&short * (&one - &eps).pow(2) < need);
}
