// tests/zero_filter.rs

//! The signs the `orient2d` zero filter must not change.
//!
//! The zero filter is a shortcut: it answers `Zero` for a repeated point or
//! an axis-aligned triple without reaching the exact integer stage. A
//! shortcut can go wrong in two directions, and this file pins both:
//!
//! - where the filter is meant to fire, the answer must be `Zero` in every
//!   argument order, because the filter may not depend on the order;
//! - where it must not fire (a line in a general direction, input one float
//!   away from degenerate, subnormal differences that are not zero), the
//!   answer must stay what exact arithmetic says it is.
//!
//! Every test here checks the answer, not the stage that gives it, so none
//! depends on the filter being present: each would pass the same way with
//! every call going through the later stages. Whether the filter actually
//! saves the exact stage is a separate question, answered by
//! `tests/zero_filter_counters.rs`.
//!
//! The reference for the perturbed cases is `orient2d_rational`, the crate's
//! pure-integer evaluation of the same orientation. It shares no code with
//! the floating-point stages of `orient2d`.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp          — an exact-value assertion IS a float comparison.
//   unreadable_literal — coordinates are written as the values they are.
//   unwrap_used        — in a test, a panic is the failure signal.
#![allow(clippy::float_cmp, clippy::unreadable_literal, clippy::unwrap_used)]

mod support;

use arithmetica::{RationalCoord2d, Sign, orient2d, orient2d_rational};
use support::{
    Triple, all_orders, axis_aligned_triples, duplicate_triples, from_ordinal,
    general_collinear_triple, ordinal,
};

/// `orient2d` through the adaptive floating-point stages.
fn float_sign(t: Triple) -> Sign {
    orient2d(t[0], t[1], t[2]).expect("the fixtures are finite")
}

/// The same orientation in exact integer arithmetic.
fn exact_sign(t: Triple) -> Sign {
    let c = RationalCoord2d::from_f64(t[2]).expect("the fixtures are finite");
    orient2d_rational(t[0], t[1], &c).expect("the fixtures are finite")
}

/// The float next to `x`, `steps` floats up (or down, if negative).
fn nudge(x: f64, steps: i64) -> f64 {
    from_ordinal(ordinal(x) + steps)
}

// ===========================================================================
// Where the filter is meant to fire
// ===========================================================================

/// Two equal points make two rows of the determinant equal, so it is zero.
/// The six orders cover each duplicate position rotated and swapped.
#[test]
fn duplicate_points_are_zero_in_every_argument_order() {
    for (name, triple) in duplicate_triples() {
        for (i, order) in all_orders(triple).into_iter().enumerate() {
            assert_eq!(float_sign(order), Sign::Zero, "{name}, order {i}");
            assert_eq!(exact_sign(order), Sign::Zero, "{name}, order {i} (twin)");
        }
    }
}

/// On a horizontal line every y-difference is exactly zero, so each product
/// in the determinant has a zero factor. A vertical line does the same with
/// the x-differences.
#[test]
fn axis_aligned_triples_are_zero_in_every_argument_order() {
    for (name, triple) in axis_aligned_triples() {
        for (i, order) in all_orders(triple).into_iter().enumerate() {
            assert_eq!(float_sign(order), Sign::Zero, "{name}, order {i}");
            assert_eq!(exact_sign(order), Sign::Zero, "{name}, order {i} (twin)");
        }
    }
}

// ===========================================================================
// Where the filter must not fire
// ===========================================================================

/// `p(5) - p(0) = 5·(p(1) - p(0))` exactly, so the triple is collinear, but
/// no difference is zero and no two points are equal. Only exact arithmetic
/// can answer it.
#[test]
fn general_collinear_triple_is_zero() {
    for (i, order) in all_orders(general_collinear_triple())
        .into_iter()
        .enumerate()
    {
        assert_eq!(float_sign(order), Sign::Zero, "order {i}");
        assert_eq!(exact_sign(order), Sign::Zero, "order {i} (twin)");
    }
}

/// Every fixture with one coordinate moved to the next float, in each
/// direction and in every argument order: the result must equal the exact
/// twin's.
///
/// Most of these are no longer degenerate, and a filter that fired on them
/// would return a wrong `Zero`. Some still are (moving `Q` in `[P, P, Q]`
/// leaves two points equal), which is why the reference is the twin and not
/// a blanket "non-zero". The count at the end keeps the test from passing
/// on an all-zero run.
#[test]
fn one_ulp_perturbations_match_the_rational_twin() {
    let mut bases: Vec<Triple> = Vec::new();
    bases.extend(duplicate_triples().into_iter().map(|(_, t)| t));
    bases.extend(axis_aligned_triples().into_iter().map(|(_, t)| t));
    bases.push(general_collinear_triple());

    let (mut compared, mut nonzero) = (0_u32, 0_u32);
    for base in bases {
        for point in 0..3 {
            for axis in 0..2 {
                for steps in [-1_i64, 1] {
                    let mut moved = base;
                    moved[point][axis] = nudge(base[point][axis], steps);
                    for order in all_orders(moved) {
                        let want = exact_sign(order);
                        assert_eq!(
                            float_sign(order),
                            want,
                            "base {base:?}, point {point}, axis {axis}, steps {steps}"
                        );
                        compared += 1;
                        nonzero += u32::from(want != Sign::Zero);
                    }
                }
            }
        }
    }
    // 7 bases × 3 points × 2 axes × 2 directions × 6 orders.
    assert_eq!(compared, 504);
    assert!(
        nonzero > compared / 2,
        "only {nonzero} of {compared} perturbed cases were non-degenerate"
    );
}

/// The same thing for one case worked by hand, so that the expected sign
/// does not rest on the twin alone.
///
/// `a = (1, 1)`, `b = (2, 1)`, `c = (3, 1)` lie on the line `y = 1`. Raise
/// `b` by one float, to `(2, 1 + δ)` with `δ = 2^-52`. Then
///
/// ```text
/// (ax - cx)(by - cy) - (ay - cy)(bx - cx) = (-2)(δ) - (0)(-1) = -2δ < 0,
/// ```
///
/// so the orientation is negative: walking from `a` to `b` to `c` turns
/// clockwise over the raised middle point. Lowering `b` instead gives `+2δ`.
#[test]
fn a_point_one_float_off_a_horizontal_line_is_not_collinear() {
    let (a, c) = ([1.0, 1.0], [3.0, 1.0]);
    let raised = [2.0, nudge(1.0, 1)];
    let lowered = [2.0, nudge(1.0, -1)];
    assert_eq!(
        raised[1] - 1.0,
        f64::EPSILON,
        "one float above 1.0 is 1 + 2^-52"
    );

    assert_eq!(float_sign([a, [2.0, 1.0], c]), Sign::Zero);
    assert_eq!(float_sign([a, raised, c]), Sign::Negative);
    assert_eq!(float_sign([a, lowered, c]), Sign::Positive);
    // Swapping two arguments negates the orientation.
    assert_eq!(float_sign([raised, a, c]), Sign::Positive);
    assert_eq!(float_sign([lowered, a, c]), Sign::Negative);
}

/// Differences of one subnormal unit are not zero.
///
/// With `u = 2^-1074`, the smallest positive `f64`:
/// `pa = (u, 0)`, `pb = (0, u)`, `pc = (0, 0)` has determinant
/// `u·u - 0·0 = 2^-2148 > 0`. Both products underflow to zero in `f64`, so a
/// floating-point evaluation sees `0 - 0`; the differences themselves are
/// `u`, not zero, so the zero filter must not fire, and the answer has to
/// come from the stages after the fast filter, which cannot certify `0 - 0`.
/// Swapping `pa` and `pb` negates it.
///
/// The second triple is the first translated by `(16u, 16u)`: the same
/// determinant with no coordinate equal to zero.
#[test]
fn nonzero_subnormal_differences_are_not_zero() {
    let u = f64::from_bits(1);
    assert_eq!(u, 5e-324, "the smallest positive subnormal");

    let (pa, pb, pc) = ([u, 0.0], [0.0, u], [0.0, 0.0]);
    assert_eq!(float_sign([pa, pb, pc]), Sign::Positive);
    assert_eq!(float_sign([pb, pa, pc]), Sign::Negative);
    assert_eq!(exact_sign([pa, pb, pc]), Sign::Positive);

    let o = 16.0 * u;
    let (qa, qb, qc) = ([o + u, o], [o, o + u], [o, o]);
    assert_eq!(qa[0] - qc[0], u, "subnormal subtraction is exact");
    assert_eq!(float_sign([qa, qb, qc]), Sign::Positive);
    assert_eq!(float_sign([qb, qa, qc]), Sign::Negative);
    assert_eq!(exact_sign([qa, qb, qc]), Sign::Positive);
}
