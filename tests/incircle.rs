// tests/incircle.rs

//! Integration tests for the `incircle` predicate.
//!
//! Tests the public `incircle` API against:
//!
//! - **Component 1**: An exact-sign oracle using arbitrary-precision integer
//!   arithmetic (ground truth against which all other components compare).
//!   Coordinates are scaled by `2^1074` so the determinant is exact in
//!   `BigInt`; the determinant carries a positive factor of `2^4296`, which
//!   leaves the sign unchanged. See `oracle_incircle` for the derivation.
//! - **Component 2**: Shewchuk-style fixed test cases, each built to be
//!   answered at a particular point of the adaptive ladder (Stage 1 fast
//!   filter, Stage 2, exact zero confirmed by Stage 4).
//! - **Component 3**: Proptest uniform strategy: 100,000 random finite inputs
//!   compared to the oracle.
//! - **Component 4**: Proptest near-degenerate strategy: 100,000 cases whose
//!   exact determinant falls below `2^-1074`, so no native-`f64` evaluation can
//!   represent it, compared to the oracle.
//! - **Component 5**: Stage observation: this file checks only the returned
//!   sign. Which stage answers a call is observed through the `stage-counters`
//!   feature (`arithmetica::counters`), exercised by `tests/stage_counters.rs`.
//! - **Component 6**: Non-finite inputs return `Err(NonFinite)`.
//! - **Component 7**: Dual-oracle agreement: the `BigInt` oracle of Component
//!   1 is checked against an independent second oracle, built on
//!   `Ratio<BigInt>`, on every canonical fixture plus 4,000 random cases.

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

use std::cmp::Ordering;

use arithmetica::{Sign, incircle};
use num_bigint::BigInt;
use num_rational::Ratio;
use proptest::prelude::*;

// ===========================================================================
// Component 1: Exact-sign oracle
// ===========================================================================

/// Converts a finite `f64` to the exact integer `x · 2^1074` by decomposing its
/// IEEE 754 bits.
///
/// Exact for every finite `f64`. The scale factor is `2^1074` because the
/// smallest positive subnormal is `2^-1074`; multiplying by that power of two
/// clears the fractional part of every representable value at once, so no
/// rounding or denominator is ever needed.
///
/// # Panics
///
/// Does not panic. Non-finite inputs decode to meaningless magnitudes, so
/// callers must screen for finiteness first.
fn scaled_integer(x: f64) -> BigInt {
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

/// Computes the exact in-circle sign of the lifted 3×3 determinant.
///
/// This is the test oracle for `incircle`:
///
/// ```text
/// det = adlift · (bdx·cdy − bdy·cdx)
///     + bdlift · (cdx·ady − cdy·adx)
///     + cdlift · (adx·bdy − ady·bdx)
/// ```
///
/// where `adx = ax−dx`, `adlift = adx²+ady²`, etc. This is the expansion of the
/// 4×4 incircle determinant by cofactors along the last column, as described in
/// Shewchuk (1997), and the term pairing matches the adaptive implementation
/// under test.
///
/// # Why integer, not rational
///
/// The predicate needs only the determinant's *sign*, and the sign is invariant
/// under multiplication by a positive constant. Scaling every coordinate by
/// `2^1074` turns each finite `f64` into an exact integer, so the determinant
/// evaluates in pure integer arithmetic with no rational normalization.
///
/// The scaling exponent is derived for this predicate rather than carried over
/// from another. With `X_i = x_i · 2^1074`:
///   - each coordinate difference is scaled by `2^1074`,
///   - each squared lift by `2^2148`,
///   - each 2×2 minor of differences by `2^2148`,
///   - each `lift · minor` term, and hence the determinant, by
///     `2^2148 · 2^2148 = ` **`2^4296`**.
///
/// (For contrast, `orient2d` lands on `2^2148` and `insphere` on `2^5370`; the
/// exponent tracks the predicate's degree in the coordinates, so it must be
/// re-derived per predicate. The same accounting reproduces the `insphere`
/// figure: `lift · (3×3 minor)` = `2^2148 · 2^3222` = `2^5370`.)
///
/// Because `2^4296 > 0`, `sign(det_int) == sign(det_rational)` for *all* finite
/// `f64` inputs, including the exactly-cocircular case where both are zero.
/// This is an exact equality of signs, not an approximation.
///
/// `num-rational` runs a Euclidean GCD normalization after every operation on
/// multi-hundred-bit terms; that work is pure overhead when only the sign is
/// wanted. Dropping it is what makes the 100,000-case proptests affordable in
/// debug builds, where `cargo test` runs.
///
/// Cross-checked against [`oracle_incircle_bigrational`] by the
/// `dual_oracle_agreement_incircle_*` tests on every run.
///
/// # Panics
///
/// Does not panic. The returned sign is meaningless for non-finite input;
/// callers must pass finite coordinates.
fn oracle_incircle(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Sign {
    let ax = scaled_integer(pa[0]);
    let ay = scaled_integer(pa[1]);
    let bx = scaled_integer(pb[0]);
    let by = scaled_integer(pb[1]);
    let cx = scaled_integer(pc[0]);
    let cy = scaled_integer(pc[1]);
    let dx = scaled_integer(pd[0]);
    let dy = scaled_integer(pd[1]);

    // Coordinate differences (pa−pd, etc.), each scaled by 2^1074.
    let adx = &ax - &dx;
    let ady = &ay - &dy;
    let bdx = &bx - &dx;
    let bdy = &by - &dy;
    let cdx = &cx - &dx;
    let cdy = &cy - &dy;

    // Squared magnitudes of the differences (the "lift" terms), scaled by 2^2148.
    let adlift = &adx * &adx + &ady * &ady;
    let bdlift = &bdx * &bdx + &bdy * &bdy;
    let cdlift = &cdx * &cdx + &cdy * &cdy;

    // 3×3 determinant (cofactor expansion along the last column), scaled by 2^4296.
    let det = &adlift * (&bdx * &cdy - &bdy * &cdx)
        + &bdlift * (&cdx * &ady - &cdy * &adx)
        + &cdlift * (&adx * &bdy - &ady * &bdx);

    match det.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// An independent second oracle, built on `Ratio<BigInt>`.
///
/// This evaluates the same determinant in a different number representation —
/// exact rationals built by `Ratio::<BigInt>::from_float` rather than integers
/// built by bit decomposition — and its conversion from `f64` is derived
/// independently of [`scaled_integer`]. Agreement between the two means an
/// error in either conversion would have to be matched by the same error in
/// the other.
///
/// It is deliberately *not* used by the 100,000-case proptests; it is called
/// only by the `dual_oracle_agreement_incircle_*` tests, which run it on the
/// canonical fixtures and a bounded random sample.
///
/// # Panics
///
/// Panics if any coordinate is NaN or infinite (`from_float` returns `None`
/// for those values). Only call this function with finite inputs.
fn oracle_incircle_bigrational(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Sign {
    let to_ratio = |x: f64| {
        Ratio::<BigInt>::from_float(x)
            .expect("oracle_incircle_bigrational: input must be finite (not NaN or infinity)")
    };

    let ax = to_ratio(pa[0]);
    let ay = to_ratio(pa[1]);
    let bx = to_ratio(pb[0]);
    let by = to_ratio(pb[1]);
    let cx = to_ratio(pc[0]);
    let cy = to_ratio(pc[1]);
    let dx = to_ratio(pd[0]);
    let dy = to_ratio(pd[1]);

    // Coordinate differences (pa−pd, etc.)
    let adx = &ax - &dx;
    let ady = &ay - &dy;
    let bdx = &bx - &dx;
    let bdy = &by - &dy;
    let cdx = &cx - &dx;
    let cdy = &cy - &dy;

    // Squared magnitudes of the differences (the "lift" terms)
    let adlift = &adx * &adx + &ady * &ady;
    let bdlift = &bdx * &bdx + &bdy * &bdy;
    let cdlift = &cdx * &cdx + &cdy * &cdy;

    // 3×3 determinant (cofactor expansion of the 4×4 matrix along last column)
    let det = &adlift * (&bdx * &cdy - &bdy * &cdx)
        + &bdlift * (&cdx * &ady - &cdy * &adx)
        + &cdlift * (&adx * &bdy - &ady * &bdx);

    let zero = Ratio::from_integer(BigInt::from(0));
    if det > zero {
        Sign::Positive
    } else if det < zero {
        Sign::Negative
    } else {
        Sign::Zero
    }
}

// ===========================================================================
// Component 2: Fixed test cases (one per adaptive stage)
// ===========================================================================

/// Stage 1 fast-path case: `pd` clearly inside the circumcircle.
///
/// `pa=(0,0)`, `pb=(1,0)`, `pc=(1,1)`, `pd=(0.5,0.5)`.
///
/// `pa → pb → pc` is counterclockwise, so a point strictly inside the
/// circumcircle gives `Positive`. The triangle has its right angle at (1,0),
/// so the circumcircle has the hypotenuse from (0,0) to (1,1) as a diameter:
/// center (0.5, 0.5), radius² = 0.5. `pd` is the center itself, so it is
/// strictly inside.
///
/// Every operation is exact in `f64` and the determinant is far from zero
/// relative to the error bound, so the Stage 1 fast filter certifies it.
///
/// Expected: `Sign::Positive`.
#[test]
fn incircle_stage1_inside() {
    let pa = [0.0, 0.0];
    let pb = [1.0, 0.0];
    let pc = [1.0, 1.0];
    let pd = [0.5, 0.5];
    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    assert_eq!(
        result,
        Sign::Positive,
        "pd=(0.5,0.5) is inside circumcircle of (0,0),(1,0),(1,1); oracle confirms Positive"
    );
    assert_eq!(result, oracle_incircle(pa, pb, pc, pd));
}

/// Stage 1 fast-path case: `pd` clearly outside the circumcircle.
///
/// `pa=(0,0)`, `pb=(1,0)`, `pc=(1,1)`, `pd=(2,2)`.
///
/// Distance² from the circumcenter (0.5, 0.5): 1.5² + 1.5² = 4.5 > 0.5
/// → pd strictly outside. The Stage 1 fast filter certifies it.
///
/// Expected: `Sign::Negative`.
#[test]
fn incircle_stage1_outside() {
    let pa = [0.0, 0.0];
    let pb = [1.0, 0.0];
    let pc = [1.0, 1.0];
    let pd = [2.0, 2.0];
    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    assert_eq!(
        result,
        Sign::Negative,
        "pd=(2,2) is outside circumcircle of (0,0),(1,0),(1,1); oracle confirms Negative"
    );
    assert_eq!(result, oracle_incircle(pa, pb, pc, pd));
}

/// Stage 2 intermediate-path case: Stage 1 cannot certify, Stage 2 resolves.
///
/// `pa=(0,0)`, `pb=(2,0)`, `pc=(0,2)`, `pd=(2 + delta, 2)` with
/// `delta = 25·f64::EPSILON ≈ 5.55e-15`.
///
/// ## Numerical analysis
///
/// The circumcircle of (0,0),(2,0),(0,2) has center (1,1) and radius² = 2.
/// All four points (0,0),(2,0),(0,2),(2,2) lie on this circle. Moving
/// pd to `(2+e, 2)` with `e > 0` moves it outside the circle.
///
/// The sum `2 + delta` rounds: the spacing of `f64` near 2 is `2^-51`, and
/// `delta` is 12.5 of those spacings, so it rounds to even, giving
/// `pd.x = 2 + e` with `e = 24·2^-52 = 3·2^-49 ≈ 5.33e-15`. The exact
/// determinant is `−4·e·(2+e) ≈ −4.26e-14`.
///
/// The permanent is approximately 64 (independent of `e` for small `e`).
///
/// - `ICC_ERR_BOUND_A ≈ 1.11e-15`; Stage 1 threshold ≈ `1.11e-15 × 64 = 7.11e-14`.
///   Since `|det| ≈ 4.26e-14 < 7.11e-14`, Stage 1 cannot certify.
/// - `ICC_ERR_BOUND_B ≈ 4.44e-16`; Stage 2 threshold ≈ `4.44e-16 × 64 = 2.84e-14`.
///   Since `|det| ≈ 4.26e-14 > 2.84e-14`, Stage 2 resolves.
///
/// Expected: `Sign::Negative` (pd outside; oracle confirms).
#[test]
fn incircle_stage2_near_circle() {
    let delta = 25.0 * f64::EPSILON; // ≈ 5.55e-15; in Stage-1/Stage-2 crossover band
    let pa = [0.0, 0.0];
    let pb = [2.0, 0.0];
    let pc = [0.0, 2.0];
    let pd = [2.0 + delta, 2.0];
    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    let expected = oracle_incircle(pa, pb, pc, pd);
    assert_eq!(
        result, expected,
        "Stage 2 case: pa=(0,0), pb=(2,0), pc=(0,2), pd=(2+25ε, 2); oracle={expected:?}"
    );
}

/// Exact-path case: four exactly cocircular points whose coordinate
/// differences have non-zero rounding tails.
///
/// `pa=(a, b)`, `pb=(b, −a)`, `pc=(−b, a)`, `pd=(−a, −b)` with
/// `a = fl(3/5)`, `b = fl(4/5)`.
///
/// The four points are the images of `(a, b)` under repeated quarter turns
/// about the origin, `(x, y) → (y, −x)`: `pa → pb → pd → pc`. Negating and
/// swapping coordinates is exact, so the four
/// `f64` points lie exactly on the circle of radius `sqrt(a² + b²)` about the
/// origin. (That radius is not exactly 1, since `a` and `b` are rounded; the
/// argument does not need it to be.) The exact determinant is zero.
///
/// The f64 coordinate differences such as `bdx = b − (−a) = a + b` round, so
/// they have non-zero tails and the all-tails-zero early return inside
/// `incircle_adapt` does not apply. The floating-point determinant is about
/// `4.4e-16` against a Stage 1 threshold of about `1.9e-14`, so no fast stage
/// certifies a sign; Stage 3 builds the full expansion, which is zero, and
/// Stage 4 confirms the exact zero.
///
/// Expected: `Sign::Zero` (the oracle confirms).
#[test]
fn incircle_stage3_cocircular() {
    let pa = [3.0_f64 / 5.0, 4.0_f64 / 5.0];
    let pb = [4.0_f64 / 5.0, -3.0_f64 / 5.0];
    let pc = [-4.0_f64 / 5.0, 3.0_f64 / 5.0];
    let pd = [-3.0_f64 / 5.0, -4.0_f64 / 5.0];
    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    let expected = oracle_incircle(pa, pb, pc, pd);
    assert_eq!(
        result, expected,
        "Stage 3 case: quarter-turn cocircular points; oracle={expected:?}"
    );
}

// ===========================================================================
// Component 3: Proptest uniform strategy
// ===========================================================================

/// Generates zero or a finite `f64` with `|x| ∈ [2^-233, 2^233)`
/// (about `[7.2e-71, 1.4e70)`).
///
/// The bound focuses this strategy on ordinary magnitudes. It is not a limit
/// of the predicate: `incircle` returns the exact sign for every finite
/// input, and the full range, subnormals and near-overflow values included,
/// is covered by `tests/boundary.rs` and the fuzz corpus replay.
///
/// Values are built directly from bit patterns (sign, biased exponent,
/// mantissa) to avoid proptest's rejection limit. Exact zero is included
/// with low weight.
fn finite_f64_bounded() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Exact zero: coordinate differences collapse to simple floats.
        1 => Just(0.0_f64),
        // Finite non-zero values with |x| ∈ [2^-233, 2^233).
        // Sign bit uniform; exponent field in [790, 1255]; mantissa uniform.
        99 => (0_u64..=1_u64, 790_u64..=1255_u64, 0_u64..(1_u64 << 52))
            .prop_map(|(sign, exp, mantissa)| {
                f64::from_bits((sign << 63) | (exp << 52) | mantissa)
            }),
    ]
}

proptest! {
    // 100,000 cases. The count is affordable in debug builds because the
    // sign-only BigInt oracle does no rational normalization; a
    // Ratio<BigInt> oracle runs a Euclidean GCD after every operation on
    // multi-hundred-bit terms, which would dominate wall time.
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// Verifies `incircle` against the exact oracle on 100,000 uniformly
    /// sampled finite `f64` inputs.
    ///
    /// Coordinates are zero or `|x| ∈ [2^-233, 2^233)` (see
    /// `finite_f64_bounded`). This exercises primarily the Stage 1 fast filter
    /// (most random inputs produce a determinant well above its error bound)
    /// and some Stage 2 cases. Near-degenerate inputs are
    /// rare under uniform sampling; they are covered explicitly by Component 4.
    ///
    /// Any divergence from the oracle is a test failure.
    #[test]
    fn incircle_uniform_matches_oracle(
        ax in finite_f64_bounded(), ay in finite_f64_bounded(),
        bx in finite_f64_bounded(), by in finite_f64_bounded(),
        cx in finite_f64_bounded(), cy in finite_f64_bounded(),
        dx in finite_f64_bounded(), dy in finite_f64_bounded(),
    ) {
        let pa = [ax, ay];
        let pb = [bx, by];
        let pc = [cx, cy];
        let pd = [dx, dy];
        let result   = incircle(pa, pb, pc, pd).expect("finite inputs");
        let expected = oracle_incircle(pa, pb, pc, pd);
        prop_assert_eq!(result, expected);
    }
}

// ===========================================================================
// Component 4: Proptest near-degenerate strategy
// ===========================================================================

/// Generates incircle cases whose exact determinant is nonzero but below the
/// smallest positive subnormal, `2^-1074`.
///
/// Each case is the right triangle `(0,0)`, `(2s,0)`, `(0,2s)` with
/// `s = 2^e`, `e ∈ [-600, -500]`, and `pd = (2s + delta, 2s)` just outside
/// its circumcircle. Every coordinate is exactly representable. The exact
/// determinant is `−4·delta·(2s + delta)·s²`, which with `delta = s` is
/// `−12·s⁴ = −12·2^(4e)`, at most about `2^-1996`. A native-`f64` evaluation
/// rounds it to zero, so the fast filter cannot certify a sign. The input
/// spans few bits, so an exact power-of-two rescale (see `src/range.rs`)
/// moves it into the exactness window, where the in-window stages resolve it
/// without Stage 4.
///
/// The strategy has 101 distinct values of `e`, so the 100,000 cases below
/// repeat each input many times.
fn near_circle_inputs() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2], [f64; 2])> {
    // Powers chosen so the coordinates remain representable normal values
    // while the determinant is far below the smallest subnormal.
    let exponent = -600i32..=-500i32;

    exponent.prop_map(|e| {
        let s = 2.0_f64.powi(e);

        let pa = [0.0, 0.0];
        let pb = [2.0 * s, 0.0];
        let pc = [0.0, 2.0 * s];

        // Choose a perturbation that is itself exactly representable.
        // The determinant is -4 * delta * (2s + delta) * s².
        let delta = 2.0_f64.powi(e);

        let pd = [2.0 * s + delta, 2.0 * s];

        (pa, pb, pc, pd)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// Verifies `incircle` against the exact oracle on 100,000 cases drawn
    /// from `near_circle_inputs`.
    ///
    /// `near_circle_inputs` builds a right triangle scaled by a power of two
    /// between `2^-600` and `2^-500` and places `pd` just outside the
    /// circumcircle, so the exact determinant is proportional to `scale⁴` and
    /// falls far below `2^-1074`; a native-`f64` evaluation rounds it to zero.
    /// The oracle is the authoritative expected value.
    #[test]
    fn incircle_near_degenerate_matches_oracle(
        (pa, pb, pc, pd) in near_circle_inputs()
    ) {
        let result   = incircle(pa, pb, pc, pd).expect("finite inputs");
        let expected = oracle_incircle(pa, pb, pc, pd);
        prop_assert_eq!(result, expected);
    }
}

// ===========================================================================
// Component 5: Stage observation
// ===========================================================================

// The tests in this file check the returned sign, which is the same whichever
// stage produces it, so they cannot observe which stage answered. That is
// observed through the `stage-counters` feature (`arithmetica::counters`),
// exercised by `tests/stage_counters.rs`.

// ===========================================================================
// Component 6: Non-finite inputs return Err
// ===========================================================================

/// Verifies that `incircle` returns `Err(NonFinite)` when any coordinate is
/// NaN, infinity, or negative infinity.
///
/// Tiny finite values such as `f64::MIN_POSITIVE` (the smallest normal) are
/// in-domain and must return `Ok`. The correctness of the sign across the
/// full finite range, subnormals included, is checked by `tests/boundary.rs`
/// and the fuzz corpus replay.
#[test]
fn incircle_non_finite_returns_err() {
    for call in [
        incircle([f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, f64::NAN], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [f64::NAN, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [f64::NAN, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, f64::NAN], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [f64::NAN, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, f64::NAN]),
        incircle([f64::INFINITY, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [f64::INFINITY, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [f64::INFINITY, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [f64::INFINITY, 0.5]),
        incircle([f64::NEG_INFINITY, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [f64::NEG_INFINITY, 0.0], [0.0, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [f64::NEG_INFINITY, 1.0], [0.5, 0.5]),
        incircle([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [f64::NEG_INFINITY, 0.5]),
    ] {
        assert!(call.is_err(), "non-finite input must return Err");
    }

    // The smallest normal value is finite and in-domain: it must return Ok.
    assert!(incircle([f64::MIN_POSITIVE, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]).is_ok());
    assert!(
        incircle(
            [f64::MIN_POSITIVE, f64::MIN_POSITIVE],
            [f64::MIN_POSITIVE, 0.0],
            [0.0, f64::MIN_POSITIVE],
            [0.5, 0.5],
        )
        .is_ok()
    );
}

// ===========================================================================
// Fixed inputs: large and mixed magnitude ranges
// ===========================================================================

/// Fixed input: large coordinates whose degree-4 products overflow binary64.
///
/// ## Input
///
/// ```text
/// pa = [-2.2899894549926956e226, -2.289989454992704e226]
/// pb = [-2.289989454992704e226,  -2.2333103652358466e226]
/// pc = [-2.289989454992704e226,  -2.289989454992704e226]
/// pd = [-2.289989454992704e226,  -2.2899888151770415e226]
/// ```
///
/// ## What makes it hard
///
/// Every coordinate is about `−2.29e226`, with leading bit `2^751`. The
/// differences between them are smaller but still huge, and the determinant
/// is of degree 4 in the differences, so evaluated in place its products
/// overflow binary64. The input's bits span from `2^700` to `2^751`, well
/// within the width of `incircle`'s exactness window, so an exact
/// power-of-two rescale moves it inside, where the expansion stages run
/// without overflow.
///
/// ## Why the sign is Positive
///
/// Checked against the exact oracle, which evaluates the determinant in
/// unbounded integer arithmetic.
#[test]
fn incircle_large_coordinates_overflowing_products() {
    let pa = [-2.2899894549926956e226_f64, -2.289989454992704e226_f64];
    let pb = [-2.289989454992704e226_f64, -2.2333103652358466e226_f64];
    let pc = [-2.289989454992704e226_f64, -2.289989454992704e226_f64];
    let pd = [-2.289989454992704e226_f64, -2.2899888151770415e226_f64];

    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    let expected = oracle_incircle(pa, pb, pc, pd);

    assert_eq!(
        result, expected,
        "incircle([{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}]): \
         got {result:?}, oracle says {expected:?}",
        pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]
    );
    assert_eq!(result, Sign::Positive, "this input must return Positive");
}

/// Fixed input: one tiny coordinate among large ones.
///
/// ## Input
///
/// ```text
/// pa = [-2.162818157176e222,      -2.289989454992704e226]
/// pb = [-2.289989454992704e226,   -2.289989454992704e226]
/// pc = [-2.289988572488342e226,   -2.289989454992704e226]
/// pd = [-2.289989454992704e226,    5.150393605287852e-256]
/// ```
///
/// ## What makes it hard
///
/// Seven of the eight coordinates have leading bits between `2^738` and
/// `2^751`, but `pd.y ≈ 5.15e-256` has leading bit `2^-849`. Evaluated in
/// place, the degree-4 products of the large differences overflow binary64.
/// The input's bits span from `2^-900` to `2^751`, far wider than
/// `incircle`'s exactness window, so no single power-of-two rescale can
/// bring it inside, and the exact integer stage answers it. Every point has
/// at least one large coordinate, so no choice of reference point makes the
/// differences small.
///
/// ## Why the sign is Positive
///
/// Checked against the exact oracle, which evaluates the determinant in
/// unbounded integer arithmetic.
#[test]
fn incircle_one_tiny_coordinate_among_large() {
    let pa = [-2.162818157176e222_f64, -2.289989454992704e226_f64];
    let pb = [-2.289989454992704e226_f64, -2.289989454992704e226_f64];
    let pc = [-2.289988572488342e226_f64, -2.289989454992704e226_f64];
    let pd = [-2.289989454992704e226_f64, 5.150393605287852e-256_f64];

    let result = incircle(pa, pb, pc, pd).expect("finite inputs");
    let expected = oracle_incircle(pa, pb, pc, pd);

    assert_eq!(
        result, expected,
        "incircle([{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}]): \
         got {result:?}, oracle says {expected:?}",
        pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]
    );
    assert_eq!(result, Sign::Positive, "this input must return Positive");
}

/// A determinant far below the smallest subnormal, on input the exactness
/// window can absorb.
///
/// `s = 1.875·2^-1020`: the exact determinant is `−12·s⁴ ≈ −2^-4072.8`, far
/// below `2^-1074` (the smallest subnormal), so a native-`f64` evaluation
/// rounds it to zero. The coordinates span only a few bits, so an exact
/// power-of-two rescale moves the input into the exactness window, where the
/// expansion stages return the sign. `pd = (3s, 2s)` is at squared distance
/// `5s²` from the circumcenter `(s, s)` of `(0,0)`, `(2s,0)`, `(0,2s)`, whose
/// squared radius is `2s²`; it is outside, so the expected sign is
/// `Negative`, which the oracle confirms.
#[test]
fn incircle_determinant_below_subnormal_range() {
    let s = f64::from_bits(0x003E_0000_0000_0000u64); // 1.875·2^-1020, a normal f64
    let pa = [0.0f64, 0.0];
    let pb = [2.0 * s, 0.0];
    let pc = [0.0, 2.0 * s];
    let pd = [2.0 * s + s, 2.0 * s]; // slightly outside
    assert_eq!(incircle(pa, pb, pc, pd), Ok(Sign::Negative));
    assert_eq!(oracle_incircle(pa, pb, pc, pd), Sign::Negative);
}

// ===========================================================================
// Component 7: Dual-oracle agreement
// ===========================================================================

/// Every fixed-coordinate case this file exercises elsewhere, in one table.
///
/// The dual-oracle check must cover the canonical fixtures, not just random
/// input: the fixtures are where the exactly-zero determinants and the extreme
/// dynamic ranges live, and those are precisely the cases where a
/// representation change is most likely to diverge. Random sampling almost
/// never produces an exact zero.
fn canonical_incircle_cases() -> Vec<([f64; 2], [f64; 2], [f64; 2], [f64; 2])> {
    let delta = 25.0 * f64::EPSILON;
    let s_tiny = f64::from_bits(0x003E_0000_0000_0000u64); // 1.875·2^-1020

    vec![
        // Stage 1 fast filter: clearly inside, clearly outside.
        ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]),
        ([0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
        // Stage 2: determinant in the Stage-1/Stage-2 crossover band.
        ([0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0 + delta, 2.0]),
        // Exactly cocircular, determinant exactly zero.
        (
            [3.0 / 5.0, 4.0 / 5.0],
            [4.0 / 5.0, -3.0 / 5.0],
            [-4.0 / 5.0, 3.0 / 5.0],
            [-3.0 / 5.0, -4.0 / 5.0],
        ),
        // Exact determinant below the smallest subnormal (Component 4).
        (
            [0.0, 0.0],
            [2.0 * s_tiny, 0.0],
            [0.0, 2.0 * s_tiny],
            [2.0 * s_tiny + s_tiny, 2.0 * s_tiny],
        ),
        // Fixed inputs: large coordinates, and one tiny coordinate among large.
        (
            [-2.289_989_454_992_695_6e226, -2.289_989_454_992_704e226],
            [-2.289_989_454_992_704e226, -2.233_310_365_235_846_6e226],
            [-2.289_989_454_992_704e226, -2.289_989_454_992_704e226],
            [-2.289_989_454_992_704e226, -2.289_988_815_177_041_5e226],
        ),
        (
            [-2.162_818_157_176e222, -2.289_989_454_992_704e226],
            [-2.289_989_454_992_704e226, -2.289_989_454_992_704e226],
            [-2.289_988_572_488_342e226, -2.289_989_454_992_704e226],
            [-2.289_989_454_992_704e226, 5.150_393_605_287_852e-256],
        ),
    ]
}

/// The two oracles must agree on every canonical fixture.
///
/// This is the fixed half of the dual-oracle check. It runs the `BigRational`
/// oracle, which is otherwise unused, against the `BigInt` oracle that the
/// proptests depend on, so a divergence between the two representations is
/// caught on every run.
#[test]
fn dual_oracle_agreement_incircle_canonical() {
    let cases = canonical_incircle_cases();
    assert!(
        cases.len() >= 7,
        "canonical table shrank: expected at least 7 cases, found {}",
        cases.len()
    );
    for (i, (pa, pb, pc, pd)) in cases.iter().enumerate() {
        assert_eq!(
            oracle_incircle(*pa, *pb, *pc, *pd),
            oracle_incircle_bigrational(*pa, *pb, *pc, *pd),
            "canonical case {i}: BigInt oracle disagrees with BigRational oracle \
             on pa={pa:?} pb={pb:?} pc={pc:?} pd={pd:?}"
        );
    }
}

proptest! {
    // 4,000 cases: a broad random sample for the dual-oracle check while
    // keeping the slow BigRational oracle off the 100,000-case path. The BigInt oracle is
    // what the uniform and near-degenerate suites use; this check exists to
    // prove the two representations agree, not to re-cover the input space.
    #![proptest_config(ProptestConfig::with_cases(4_000))]

    /// The two oracles must agree on randomly sampled input.
    ///
    /// A disagreement here means one of the two exact evaluations is wrong,
    /// either the bit-decomposition scaling or the `from_float` rational
    /// conversion. Because both are exact by construction, any divergence is a
    /// defect in this test file rather than in `incircle`.
    #[test]
    fn dual_oracle_agreement_incircle_random(
        ax in finite_f64_bounded(), ay in finite_f64_bounded(),
        bx in finite_f64_bounded(), by in finite_f64_bounded(),
        cx in finite_f64_bounded(), cy in finite_f64_bounded(),
        dx in finite_f64_bounded(), dy in finite_f64_bounded(),
    ) {
        let pa = [ax, ay];
        let pb = [bx, by];
        let pc = [cx, cy];
        let pd = [dx, dy];
        prop_assert_eq!(
            oracle_incircle(pa, pb, pc, pd),
            oracle_incircle_bigrational(pa, pb, pc, pd)
        );
    }
}
