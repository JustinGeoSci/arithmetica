// tests/round.rs

//! Tests for `RationalCoord2d::f64_bracket` — the exact bracket of a
//! rational coordinate between its two nearest representable `f64` values.
//!
//! # The oracle is a different algorithm, not a second copy
//!
//! The implementation estimates in `f64` and then verifies the candidate by
//! exact comparison against dyadic rationals held as `(u64, i32)` pairs in
//! fixed-width integer arithmetic. The oracle here never touches a dyadic or
//! a `WideInt`: it finds the lower neighbor by `BigInt` long division, then
//! settles every question — is this the lower neighbor, is the value exact,
//! which side of the midpoint — by comparing `Ratio<BigInt>` values directly.
//! The midpoint is formed as a `Ratio` and compared as a `Ratio`. A
//! transcription error shared by both derivations is not available.
//!
//! The segment-intersection oracle is Cramer's rule on implicit line
//! equations, as in `tests/rational.rs`, for the same reason.
// Test-code lint posture, stated rather than left implicit:
//
//   unwrap_used        — in a test, a panic is the failure signal.
//   float_cmp          — an exact-value assertion IS a float comparison.
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use arithmetica::{BracketPosition, F64Bracket, RationalCoord2d};
use num_bigint::{BigInt, Sign as BigSign};
use num_rational::Ratio;
use proptest::prelude::*;

type Rat = Ratio<BigInt>;
type Pt = [f64; 2];
type Seg = (Pt, Pt);

const CASES: u32 = if cfg!(debug_assertions) {
    2_000
} else {
    100_000
};

// ===========================================================================
// The oracle
// ===========================================================================

fn rat(x: f64) -> Rat {
    Ratio::<BigInt>::from_float(x).unwrap()
}

fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        -f64::from_bits((-x).to_bits() - 1)
    }
}

fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// A finite `f64` candidate near `v`, from `BigInt` long division.
///
/// Only a starting point: `oracle_bracket` corrects it by exact comparison,
/// so this need not be accurate; it must merely be finite and within a few
/// steps of the truth. Subnormal and overflow handling is therefore left to
/// the exact correction rather than attempted here.
fn candidate_near(v: &Rat) -> f64 {
    let num = BigInt::from_biguint(BigSign::Plus, v.numer().magnitude().clone());
    let den = BigInt::from_biguint(BigSign::Plus, v.denom().magnitude().clone());
    let bn = i64::try_from(num.bits()).unwrap();
    let bd = i64::try_from(den.bits()).unwrap();
    let k = 60 - (bn - bd);
    let (n, d) = if k >= 0 {
        (&num << u64::try_from(k).unwrap(), den)
    } else {
        (num, &den << u64::try_from(-k).unwrap())
    };
    let q = n / d;
    let qb = i64::try_from(q.bits()).unwrap();
    let drop = (qb - 53).max(0);
    let m = u64::try_from(&q >> u64::try_from(drop).unwrap()).unwrap();
    let exp = i32::try_from(-k + drop).unwrap_or(i32::MAX / 2);
    // `m` has at most 53 bits after the shift, so the conversion is exact;
    // and this is only a candidate the exact comparison corrects anyway.
    #[allow(clippy::cast_precision_loss)]
    let approx = (m as f64) * 2f64.powi(exp);
    let approx = if approx.is_finite() { approx } else { f64::MAX };
    if v.numer().sign() == BigSign::Minus {
        -approx
    } else {
        approx
    }
}

/// The bracket of an exact value, by exact `Ratio` comparison.
fn oracle_bracket(v: &Rat) -> F64Bracket {
    let zero = Ratio::from_integer(BigInt::from(0));
    if *v == zero {
        return F64Bracket {
            lo: 0.0,
            hi: 0.0,
            position: BracketPosition::Exact,
        };
    }

    // Settle `lo` as the largest finite f64 ≤ v, by stepping from the
    // candidate in whichever direction the exact comparison says.
    let mut lo = candidate_near(v);
    while rat(lo) > *v {
        lo = next_down(lo);
    }
    loop {
        let up = next_up(lo);
        if up.is_infinite() || rat(up) > *v {
            break;
        }
        lo = up;
    }

    if rat(lo) == *v {
        return F64Bracket {
            lo,
            hi: lo,
            position: BracketPosition::Exact,
        };
    }

    let hi = next_up(lo);
    // The upper bound of MAX's interval is the IEEE overflow threshold,
    // 2^1024 − 2^970, not a midpoint between two finite values.
    let mid = if hi.is_infinite() {
        let two_1024 = Ratio::from_integer(BigInt::from(1) << 1024u32);
        let two_970 = Ratio::from_integer(BigInt::from(1) << 970u32);
        two_1024 - two_970
    } else {
        (rat(lo) + rat(hi)) / Ratio::from_integer(BigInt::from(2))
    };

    let position = match v.cmp(&mid) {
        core::cmp::Ordering::Less => BracketPosition::BelowMidpoint,
        core::cmp::Ordering::Equal => BracketPosition::AtMidpoint,
        core::cmp::Ordering::Greater => BracketPosition::AboveMidpoint,
    };
    F64Bracket { lo, hi, position }
}

fn oracle_point(a: Seg, b: Seg) -> Option<(Rat, Rat)> {
    let (a0, a1) = a;
    let (b0, b1) = b;
    let a1c = rat(a1[1]) - rat(a0[1]);
    let b1c = rat(a0[0]) - rat(a1[0]);
    let c1 = a1c.clone() * rat(a0[0]) + b1c.clone() * rat(a0[1]);
    let a2c = rat(b1[1]) - rat(b0[1]);
    let b2c = rat(b0[0]) - rat(b1[0]);
    let c2 = a2c.clone() * rat(b0[0]) + b2c.clone() * rat(b0[1]);
    let det = a1c.clone() * b2c.clone() - a2c.clone() * b1c.clone();
    if det == Ratio::from_integer(BigInt::from(0)) {
        return None;
    }
    let x = (c1.clone() * b2c - c2.clone() * b1c) / det.clone();
    let y = (a1c * c2 - a2c * c1) / det;
    Some((x, y))
}

fn check(got: F64Bracket, want: F64Bracket, what: &str) -> Result<(), TestCaseError> {
    prop_assert!(
        got.lo.to_bits() == want.lo.to_bits()
            && got.hi.to_bits() == want.hi.to_bits()
            && got.position == want.position,
        "{what}: got ({:e}, {:e}, {:?}), oracle ({:e}, {:e}, {:?})",
        got.lo,
        got.hi,
        got.position,
        want.lo,
        want.hi,
        want.position
    );
    Ok(())
}

// ===========================================================================
// The property that needs no oracle
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// `from_f64` then `f64_bracket` must report `Exact` at the input, bit
    /// for bit.
    ///
    /// `from_f64` stores the coordinate as an exact integer over 1, so the
    /// bracket has a representable answer and must find it; any failure is
    /// the bracket's. The one exception is `−0.0`: `RationalCoord2d` has no
    /// signed zero, so it brackets as `+0.0`, which is the representation
    /// being faithful to the geometry rather than an error.
    #[test]
    fn from_f64_then_bracket_is_exact(x in any::<f64>(), y in any::<f64>()) {
        prop_assume!(x.is_finite() && y.is_finite());
        let b = RationalCoord2d::from_f64([x, y]).unwrap().f64_bracket();
        for (orig, got) in [(x, b[0]), (y, b[1])] {
            prop_assert_eq!(got.position, BracketPosition::Exact, "{:e} not Exact", orig);
            if orig == 0.0 {
                prop_assert!(got.lo == 0.0 && got.lo.is_sign_positive());
            } else {
                prop_assert!(
                    got.lo.to_bits() == orig.to_bits() && got.hi.to_bits() == orig.to_bits(),
                    "{:e} bracketed as ({:e}, {:e})", orig, got.lo, got.hi
                );
            }
        }
    }
}

// ===========================================================================
// Against the oracle
// ===========================================================================

fn lattice_coord() -> impl Strategy<Value = f64> {
    (-8i32..=8).prop_map(f64::from)
}

fn lattice_point() -> impl Strategy<Value = Pt> {
    (lattice_coord(), lattice_coord()).prop_map(|(x, y)| [x, y])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    /// A constructed intersection brackets exactly as the `Ratio` oracle
    /// says it does.
    ///
    /// The lattice is deliberate: small-denominator rationals are where
    /// exactly-representable results and near-ties are common.
    #[test]
    fn constructed_points_bracket_like_the_oracle(
        a0 in lattice_point(), a1 in lattice_point(),
        b0 in lattice_point(), b1 in lattice_point(),
    ) {
        let Ok(Some(p)) = RationalCoord2d::from_segment_intersection(a0, a1, b0, b1) else {
            return Ok(());
        };
        let Some((ox, oy)) = oracle_point((a0, a1), (b0, b1)) else { return Ok(()); };
        let got = p.f64_bracket();
        check(got[0], oracle_bracket(&ox), "x")?;
        check(got[1], oracle_bracket(&oy), "y")?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES / 4))]

    /// The same at projected-coordinate magnitudes (around `10^6`), where the
    /// ulp is `2^−33` to `2^−32` (about `1.2–2.3 × 10^−10`) and the exponent
    /// range is entirely different.
    #[test]
    fn large_magnitude_points_bracket_like_the_oracle(
        dx0 in -64i32..=64, dy0 in -64i32..=64,
        dx1 in -64i32..=64, dy1 in -64i32..=64,
    ) {
        let base = [1_435_363.863_356_009_1f64, 630_961.550_231_009_7f64];
        let off = |dx: i32, dy: i32| -> Pt {
            [base[0] + f64::from(dx) * 0.25, base[1] + f64::from(dy) * 0.25]
        };
        let (a0, a1) = (off(dx0, dy0), off(dx1, dy1));
        let (b0, b1) = (off(dx0, dy1), off(dx1, dy0));
        let Ok(Some(p)) = RationalCoord2d::from_segment_intersection(a0, a1, b0, b1) else {
            return Ok(());
        };
        let Some((ox, oy)) = oracle_point((a0, a1), (b0, b1)) else { return Ok(()); };
        let got = p.f64_bracket();
        check(got[0], oracle_bracket(&ox), "x")?;
        check(got[1], oracle_bracket(&oy), "y")?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES / 10))]

    /// Nearly parallel segments at large magnitude: intersections far outside
    /// the inputs' bounding box, reaching into and past the top of the `f64`
    /// range. Overflow is reachable from finite inputs, and this is the test
    /// that exercises it against the oracle's exact IEEE threshold.
    #[test]
    fn far_intersections_bracket_like_the_oracle(
        scale_exp in 900i32..=1023,
        slope_num in 1i32..=4,
        slope_den in 1i32..=4,
        sep in 1i32..=3,
    ) {
        let s = 2f64.powi(scale_exp);
        let slope = f64::from(slope_num) / f64::from(slope_den);
        let a0 = [0.0, 0.0];
        let a1 = [s, s * slope];
        let b0 = [0.0, s * f64::from(sep) * 2f64.powi(-40)];
        let b1 = [s, s * slope + s * 2f64.powi(-50)];
        prop_assume!([a1, b0, b1].iter().all(|p| p[0].is_finite() && p[1].is_finite()));

        let Ok(Some(p)) = RationalCoord2d::from_segment_intersection(a0, a1, b0, b1) else {
            return Ok(());
        };
        let Some((ox, oy)) = oracle_point((a0, a1), (b0, b1)) else { return Ok(()); };
        let got = p.f64_bracket();
        check(got[0], oracle_bracket(&ox), "x")?;
        check(got[1], oracle_bracket(&oy), "y")?;
    }
}

// ===========================================================================
// Hand-built cases
// ===========================================================================

/// Subnormal and extreme coordinates bracket as `Exact` at themselves.
#[test]
fn subnormal_and_extreme_coordinates_are_exact() {
    let cases = [
        f64::from_bits(1),
        f64::from_bits(2),
        f64::from_bits((1u64 << 52) - 1),
        f64::from_bits(1u64 << 52),
        f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
    ];
    for x in cases {
        let b = RationalCoord2d::from_f64([x, 1.0]).unwrap().f64_bracket()[0];
        assert_eq!(b.position, BracketPosition::Exact, "{x:e}");
        assert_eq!(b.lo.to_bits(), x.to_bits(), "{x:e}");
        assert_eq!(b.hi.to_bits(), x.to_bits(), "{x:e}");
    }
}

/// `2/3` is not representable; the bracket must say so and place it.
#[test]
fn two_thirds_is_bracketed_not_truncated() {
    let q =
        RationalCoord2d::from_segment_intersection([0.0, 0.0], [2.0, 2.0], [0.0, 1.0], [2.0, 0.0])
            .unwrap()
            .unwrap();
    let b = q.f64_bracket()[0];
    assert_ne!(b.position, BracketPosition::Exact);
    assert_eq!(
        next_up(b.lo).to_bits(),
        b.hi.to_bits(),
        "neighbors not adjacent"
    );
    let two_thirds = 2.0f64 / 3.0;
    let near = match b.position {
        BracketPosition::BelowMidpoint => b.lo,
        BracketPosition::AboveMidpoint => b.hi,
        other => panic!("2/3 is not a tie or exact, got {other:?}"),
    };
    assert_eq!(
        near.to_bits(),
        two_thirds.to_bits(),
        "nearest neighbor is not IEEE's 2/3"
    );
}

/// `3/4` is representable and must bracket as `Exact`.
#[test]
fn three_quarters_is_exact() {
    let p =
        RationalCoord2d::from_segment_intersection([0.0, 0.0], [3.0, 3.0], [0.0, 1.0], [3.0, 0.0])
            .unwrap()
            .unwrap();
    assert_eq!(
        p.f64_bracket()[0],
        F64Bracket {
            lo: 0.75,
            hi: 0.75,
            position: BracketPosition::Exact
        }
    );
}
