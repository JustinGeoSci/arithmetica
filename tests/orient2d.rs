// tests/orient2d.rs

//! Integration tests for the `orient2d` predicate.
//!
//! Tests the public `orient2d` API against:
//!
//! - **Component 1**: An exact-sign oracle using arbitrary-precision integer
//!   arithmetic (ground truth against which all other components compare).
//!   Coordinates are scaled by `2^1074` so the determinant is exact in
//!   `BigInt`; the determinant carries a positive factor of `2^2148`, which
//!   leaves the sign unchanged. See `oracle_orient2d` for the derivation.
//! - **Component 2**: Shewchuk-style fixed test cases, each built to be
//!   answered at a particular point of the adaptive ladder (fast filter,
//!   Stage 2 exit, exact zero confirmed by Stage 4).
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

use arithmetica::{Sign, orient2d};
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

/// Computes the exact orientation sign of the `orient2d` 2×2 determinant.
///
/// This is the test oracle for `orient2d`:
///
/// ```text
/// det = (ax − cx)·(by − cy) − (ay − cy)·(bx − cx)
/// ```
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
///   - every term of the 2×2 determinant is a product of two differences, hence
///     scaled by `2^1074 · 2^1074 = 2^2148`,
///   - so the determinant is scaled by **`2^2148`**.
///
/// (For contrast, `incircle` lands on `2^4296` and `insphere` on `2^5370`; the
/// exponent tracks the predicate's degree in the coordinates, so it must be
/// re-derived per predicate.)
///
/// Because `2^2148 > 0`, `sign(det_int) == sign(det_rational)` for *all* finite
/// `f64` inputs, including the exactly-collinear case where both are zero.
/// This is an exact equality of signs, not an approximation.
///
/// `num-rational` runs a Euclidean GCD normalization after every operation on
/// multi-hundred-bit terms; that work is pure overhead when only the sign is
/// wanted. Dropping it is what makes the 100,000-case proptests affordable in
/// debug builds, where `cargo test` runs.
///
/// Cross-checked against [`oracle_orient2d_bigrational`] by the
/// `dual_oracle_agreement_orient2d_*` tests on every run.
///
/// # Panics
///
/// Does not panic. The returned sign is meaningless for non-finite input;
/// callers must pass finite coordinates.
fn oracle_orient2d(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    let ax = scaled_integer(pa[0]);
    let ay = scaled_integer(pa[1]);
    let bx = scaled_integer(pb[0]);
    let by = scaled_integer(pb[1]);
    let cx = scaled_integer(pc[0]);
    let cy = scaled_integer(pc[1]);

    // Exact determinant, scaled by 2^2148: (ax−cx)·(by−cy) − (ay−cy)·(bx−cx)
    let det = (&ax - &cx) * (&by - &cy) - (&ay - &cy) * (&bx - &cx);

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
/// only by the `dual_oracle_agreement_orient2d_*` tests, which run it on the
/// canonical fixtures and a bounded random sample.
///
/// # Panics
///
/// Panics if any coordinate is NaN or infinite (`from_float` returns `None`
/// for those values). Only call this function with finite inputs.
fn oracle_orient2d_bigrational(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    let to_ratio = |x: f64| {
        Ratio::<BigInt>::from_float(x)
            .expect("oracle_orient2d_bigrational: input must be finite (not NaN or infinity)")
    };

    let ax = to_ratio(pa[0]);
    let ay = to_ratio(pa[1]);
    let bx = to_ratio(pb[0]);
    let by = to_ratio(pb[1]);
    let cx = to_ratio(pc[0]);
    let cy = to_ratio(pc[1]);

    // Exact determinant: (ax−cx)·(by−cy) − (ay−cy)·(bx−cx)
    let det = (&ax - &cx) * (&by - &cy) - (&ay - &cy) * (&bx - &cx);
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
// Component 2: Shewchuk fixed test cases
// ===========================================================================

/// Fast-path case: clearly counterclockwise triangle.
///
/// `pa=(0,0)`, `pb=(1,0)`, `pc=(0,1)`.
///
/// **Exact determinant:** `(0−0)·(0−1) − (0−1)·(1−0) = 1 > 0` → `Sign::Positive`.
///
/// Every coordinate difference and product is exact in `f64`, and `|det| = 1`
/// is far above the fast filter's error bound, so Stage 1 certifies the sign.
#[test]
fn shewchuk_fast_path_ccw() {
    let pa = [0.0, 0.0];
    let pb = [1.0, 0.0];
    let pc = [0.0, 1.0];
    let result = orient2d(pa, pb, pc).expect("finite inputs");
    assert_eq!(result, Sign::Positive, "CCW triangle must return Positive");
    let expected = oracle_orient2d(pa, pb, pc);
    assert_eq!(result, expected, "fast-path result must match oracle");
}

/// Fast-path case: clearly clockwise triangle.
///
/// `pa=(0,0)`, `pb=(0,1)`, `pc=(1,0)`.
///
/// **Exact determinant:** `(0−1)·(1−0) − (0−0)·(0−1) = −1 < 0` → `Sign::Negative`.
///
/// As in the counterclockwise case, the evaluation is exact and `|det| = 1`
/// is far above the error bound, so Stage 1 certifies the sign.
#[test]
fn shewchuk_fast_path_cw() {
    let pa = [0.0, 0.0];
    let pb = [0.0, 1.0];
    let pc = [1.0, 0.0];
    let result = orient2d(pa, pb, pc).expect("finite inputs");
    assert_eq!(result, Sign::Negative, "CW triangle must return Negative");
    let expected = oracle_orient2d(pa, pb, pc);
    assert_eq!(result, expected, "fast-path result must match oracle");
}

/// Collinear case with exact integer coordinates.
///
/// `pa=(0,0)`, `pb=(1,1)`, `pc=(2,2)`.
///
/// All three points lie on `y = x`, so the exact determinant is 0. The triple
/// is neither repeated nor axis-aligned, so the zero filter does not apply,
/// and the fast filter's strict comparison never certifies a zero. Every
/// difference is exact, so the expansion stages compute the determinant as
/// exactly zero; a `Zero` from them is not a certificate, and Stage 4
/// confirms it.
#[test]
fn collinear_integer_coordinates() {
    let pa = [0.0, 0.0];
    let pb = [1.0, 1.0];
    let pc = [2.0, 2.0];
    let result = orient2d(pa, pb, pc).expect("finite inputs");
    assert_eq!(
        result,
        Sign::Zero,
        "integer-coordinate collinear triple must return Zero"
    );
    let expected = oracle_orient2d(pa, pb, pc);
    assert_eq!(result, expected, "collinear result must match oracle");
}

/// Intermediate-path case: Stage 1 cannot certify; Stage 2 resolves.
///
/// `pa=(0,0)`, `pb=(1,1)`, `pc=(2, 2+1e-15)`.
///
/// `pc[1] = fl(2 + 1e-15) = 2 + 2^-50`, since the spacing of `f64` near 2 is
/// `2^-51`. The exact determinant is
/// `(0−2)·(1−pc[1]) − (0−pc[1])·(1−2) = pc[1] − 2 = 2^-50 > 0`
/// → `Sign::Positive`.
///
/// **Why Stage 1 cannot certify:** the two products have magnitude about 2
/// each, so the error bound is about `3ε · 4 ≈ 1.3e-15` (with `ε = 2^-53`),
/// above `|det| = 2^-50 ≈ 8.9e-16`.
///
/// **Why Stage 2 resolves:** `orient2d` rotates the triple so the point of
/// smallest magnitude, `(0,0)`, is the reference. Every coordinate difference
/// is then a subtraction of zero, which is exact, so all four subtraction
/// tails are zero and the Stage 2 expansion is already the exact determinant.
#[test]
fn shewchuk_intermediate_path() {
    let pa = [0.0, 0.0_f64];
    let pb = [1.0, 1.0_f64];
    let pc = [2.0, 2.0_f64 + 1e-15];
    let result = orient2d(pa, pb, pc).expect("finite inputs");
    assert_eq!(
        result,
        Sign::Positive,
        "near-collinear triple with small positive displacement: expected Positive"
    );
    let expected = oracle_orient2d(pa, pb, pc);
    assert_eq!(
        result, expected,
        "intermediate-path result must match oracle"
    );
}

/// Exact-path case: perfectly collinear, non-representable coordinates.
///
/// `pa=(1/3, 1/3)`, `pb=(2/3, 2/3)`, `pc=(1/7, 1/7)`.
///
/// Each point has `x == y` as `f64` values (both coordinates are the same
/// rounded quotient), so all three lie exactly on `y = x` and the exact
/// determinant is 0. Expected: `Sign::Zero`.
///
/// **Why the full expansion is reached:**
/// 1/3, 2/3, and 1/7 are not exactly representable as `f64`, so the coordinate
/// subtractions carry non-zero `two_diff_tail` values and the all-tails-zero
/// early return in Stage 2 does not apply.
///
/// Because each point has `x == y`, `acx == acy` and `bcy == bcx` as `f64`
/// values. This makes Stage 2's `B` expansion identically zero, and the
/// first-order tail correction in the `CCW_ERR_BOUND_C` re-test also
/// vanishes, so Stage 3 builds the full 16-term exact `D` expansion. It sums
/// to zero, Stages 2-3 return `Zero`, and Stage 4 confirms the exact zero.
#[test]
fn shewchuk_exact_path_collinear() {
    let pa = [1.0_f64 / 3.0, 1.0_f64 / 3.0];
    let pb = [2.0_f64 / 3.0, 2.0_f64 / 3.0];
    let pc = [1.0_f64 / 7.0, 1.0_f64 / 7.0];
    let result = orient2d(pa, pb, pc).expect("finite inputs");
    assert_eq!(
        result,
        Sign::Zero,
        "collinear triple on y = x with non-representable coords: expected Zero"
    );
    let expected = oracle_orient2d(pa, pb, pc);
    assert_eq!(result, expected, "exact-path result must match oracle");
}

// ===========================================================================
// Component 3: Proptest uniform strategy
// ===========================================================================

/// Generates zero or a finite `f64` with `|x| ∈ [2^-331, 2^499)`.
///
/// The bound focuses this strategy on ordinary magnitudes. It is not a
/// limit of the predicate: `orient2d`
/// returns the exact sign for every finite input, and the full range,
/// subnormals and near-overflow values included, is covered by
/// `tests/boundary.rs` and the fuzz corpus replay.
///
/// This function builds values directly from bit patterns (sign, biased
/// exponent, mantissa) to avoid proptest's rejection limit. Exact zero is
/// included with low weight.
fn finite_f64_bounded() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Exact zero: a common degenerate coordinate (collinear-with-axis cases).
        1 => proptest::strategy::Just(0.0_f64),
        // Normal f64 with biased exponent in [692, 1521], so
        // |x| ∈ [2^-331, 2^499) ≈ [2.3e-100, 1.6e150). The sign bit is
        // sampled uniformly so both positive and negative values are
        // generated.
        99 => (0_u64..=1_u64, 692_u64..=1521_u64, 0_u64..(1_u64 << 52))
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

    /// Verifies `orient2d` against the exact oracle on 100,000 uniformly
    /// sampled finite `f64` inputs.
    ///
    /// Inputs are 0.0 or |coordinate| ∈ \[2^-331, 2^499) (see
    /// `finite_f64_bounded`). This strategy exercises primarily the Stage 1
    /// fast filter (most random inputs have determinants well above its error
    /// bound) and some Stage 2 cases. Near-degenerate
    /// inputs are rare in uniform sampling; the near-degenerate strategy
    /// (Component 4) covers those specifically.
    ///
    /// Any divergence from the oracle is a test failure.
    #[test]
    fn orient2d_uniform_matches_oracle(
        ax in finite_f64_bounded(), ay in finite_f64_bounded(),
        bx in finite_f64_bounded(), by in finite_f64_bounded(),
        cx in finite_f64_bounded(), cy in finite_f64_bounded(),
    ) {
        let pa = [ax, ay];
        let pb = [bx, by];
        let pc = [cx, cy];
        let result   = orient2d(pa, pb, pc).expect("finite inputs");
        let expected = oracle_orient2d(pa, pb, pc);
        prop_assert_eq!(result, expected);
    }
}

// ===========================================================================
// Component 4: Proptest near-degenerate strategy
// ===========================================================================

/// Generates orient2d cases whose exact determinant is nonzero but below the
/// smallest positive subnormal, `2^-1074`.
///
/// Each case is the right triangle `(0,0)`, `(s,0)`, `(0,s)` with
/// `s = 2^e`, `e ∈ [-600, -538]`. All coordinates are exact powers of two, so
/// the oracle can determine the expected sign without relying on
/// floating-point arithmetic. The exact determinant is `s² = 2^(2e)`, at most
/// `2^-1076`, so every product in a native-`f64` evaluation rounds to zero
/// and the fast filter cannot certify a sign. The input spans few bits, so an
/// exact power-of-two rescale (see `src/range.rs`) moves it into the
/// exactness window, where the in-window stages resolve it without Stage 4.
///
/// The strategy has 63 distinct values of `e`, so the 100,000 cases below
/// repeat each input many times.
fn near_collinear_inputs() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2], Sign)> {
    (-600i32..=-538i32).prop_map(|e| {
        let s = 2.0_f64.powi(e);

        let pa = [0.0, 0.0];
        let pb = [s, 0.0];
        let pc = [0.0, s];

        // Exact determinant = s², which is below 2^-1074 for e < -537.
        let expected = Sign::Positive;

        (pa, pb, pc, expected)
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100_000))]

    /// Verifies `orient2d` against the exact oracle on 100,000 cases drawn
    /// from `near_collinear_inputs`.
    ///
    /// `near_collinear_inputs` builds axis-aligned right triangles scaled by a
    /// power of two small enough that the exact determinant `s²` falls below
    /// `2^-1074`, so a native-`f64` evaluation rounds it to zero. Uniform
    /// sampling almost never produces such input.
    ///
    /// The oracle is the authoritative expected value; `_expected` (from the
    /// construction formula) is provided for debugging but is not the assertion
    /// target.
    #[test]
    fn orient2d_near_degenerate_matches_oracle(
        (pa, pb, pc, _expected) in near_collinear_inputs()
    ) {
        let result   = orient2d(pa, pb, pc).expect("finite inputs");
        let expected = oracle_orient2d(pa, pb, pc);
        prop_assert_eq!(result, expected);
    }
}

// ===========================================================================
// Component 6: Non-finite inputs return Err
// ===========================================================================

/// Verifies that `orient2d` returns `Err(NonFinite)` when any coordinate is
/// NaN, infinity, or negative infinity.
///
/// Tiny finite values such as `f64::MIN_POSITIVE` (the smallest normal) are
/// in-domain and must return `Ok`. The correctness of the sign for subnormal
/// coordinates is checked by `subnormal_coordinates_agree_with_the_oracle`
/// below, `tests/boundary.rs`, and the fuzz corpus replay.
#[test]
fn non_finite_inputs_return_err() {
    // Non-finite coordinates are rejected with Err.
    for call in [
        orient2d([f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]),
        orient2d([0.0, f64::NAN], [1.0, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [f64::NAN, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [1.0, f64::NAN], [0.0, 1.0]),
        orient2d([0.0, 0.0], [1.0, 0.0], [f64::NAN, 1.0]),
        orient2d([0.0, 0.0], [1.0, 0.0], [0.0, f64::NAN]),
        orient2d([f64::INFINITY, 0.0], [1.0, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [f64::INFINITY, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [1.0, 0.0], [f64::INFINITY, 1.0]),
        orient2d([f64::NEG_INFINITY, 0.0], [1.0, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [f64::NEG_INFINITY, 0.0], [0.0, 1.0]),
        orient2d([0.0, 0.0], [1.0, 0.0], [f64::NEG_INFINITY, 1.0]),
    ] {
        assert!(call.is_err(), "non-finite input must return Err");
    }

    // The smallest normal value is finite and in-domain: it must return Ok.
    assert!(orient2d([f64::MIN_POSITIVE, 0.0], [1.0, 0.0], [0.0, 1.0]).is_ok());
    assert!(
        orient2d(
            [f64::MIN_POSITIVE, f64::MIN_POSITIVE],
            [1.0, 0.0],
            [0.0, 1.0],
        )
        .is_ok()
    );
}

// ===========================================================================
// Fixed inputs: mixed and extreme magnitude ranges
// ===========================================================================

/// Fixed input: subnormal and `~2^101` coordinates with massive cancellation.
///
/// ## Input
///
/// ```text
/// pa = [3.81752867881136e-310,  3.486039150627631e30]
/// pb = [2.2754243e-317,         3.486039150627631e30]   (subnormal x)
/// pc = [3.5295369653413445e30,  6.681002276805634e-240]
/// ```
///
/// ## What makes it hard
///
/// The coordinates run from a subnormal (`pb.x`, lowest set bit `2^-1073`) to
/// `~2^101`, a span of more than 1,100 bits, wider than `orient2d`'s
/// exactness window, so no single power-of-two rescale brings the input into
/// it. The two products of the determinant are each about `-1.2e61` and
/// cancel to about `1.3e-279`.
///
/// ## Why the sign is Positive
///
/// `pa` and `pb` share the y coordinate `Y = 3.486…e30`, so the determinant
/// factors exactly as `(Y − pc.y)·(pa.x − pb.x)`. `Y > pc.y` and
/// `pa.x > pb.x`, so both factors are positive. The oracle agrees.
#[test]
fn mixed_subnormal_and_large_coordinates() {
    let pa = [3.81752867881136e-310_f64, 3.486039150627631e30_f64];
    let pb = [2.2754243e-317_f64, 3.486039150627631e30_f64]; // subnormal x
    let pc = [3.5295369653413445e30_f64, 6.681002276805634e-240_f64];

    let result = orient2d(pa, pb, pc).expect("finite inputs");
    let expected = oracle_orient2d(pa, pb, pc);

    assert_eq!(
        result, expected,
        "orient2d([{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}]): \
         got {result:?}, oracle says {expected:?}",
        pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]
    );
    assert_eq!(result, Sign::Positive, "this input must return Positive");
}

/// Fixed input: a determinant far below the smallest subnormal.
///
/// ## Input
///
/// ```text
/// pa = [3.785767986820787e-270, 0.0]
/// pb = [8.289046e-317, 0.0]           (subnormal)
/// pc = [3.529536301617365e30, 3.621778591446635e-259]
/// ```
///
/// ## What makes it hard
///
/// The two products of the determinant are each about `1.3e-228` and cancel
/// to an exact determinant of about `-1.4e-528`, hundreds of orders of
/// magnitude below the smallest subnormal (`2^-1074 ≈ 4.9e-324`). The
/// coordinates span from a subnormal to `~2^101`, wider than the exactness
/// window, so only the exact integer stage can answer it.
///
/// ## Why the sign is Negative
///
/// `pa.y = pb.y = 0`, so the determinant factors exactly as
/// `−pc.y·(pa.x − pb.x)`. `pc.y > 0` and `pa.x > pb.x`, so it is negative.
/// The oracle agrees.
#[test]
fn determinant_far_below_the_smallest_subnormal() {
    let pa = [3.785767986820787e-270_f64, 0.0_f64];
    let pb = [8.289046e-317_f64, 0.0_f64]; // subnormal
    let pc = [3.529536301617365e30_f64, 3.621778591446635e-259_f64];

    let result = orient2d(pa, pb, pc).expect("finite inputs");
    let expected = oracle_orient2d(pa, pb, pc);

    assert_eq!(
        result, expected,
        "orient2d([{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}]): \
         got {result:?}, oracle says {expected:?}",
        pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]
    );
    assert_eq!(result, Sign::Negative, "this input must return Negative");
}

/// Fixed input: a dynamic range of about 590 orders of magnitude.
///
/// ## Input
///
/// ```text
/// pa = [3.81752867881136e-310,    3.033653279310457e-294]
/// pb = [4.778309726736481e-299,   2.4688758965194162e-307]
/// pc = [-4.539427514930182e279,   1.4102612927432247e31]
/// ```
///
/// ## What makes it hard
///
/// `pa` and `pb` are tiny (down to a subnormal), while `pc` reaches
/// `~2^929`. With `pc` as the reference point, each product of differences
/// is about `6.4e310`, beyond the binary64 range, and the two cancel to a
/// determinant of about `-1.4e-14`. With the tiny point `pa` as the
/// reference, which is how `orient2d` arranges the evaluation, each product
/// pairs a small difference with a large one and stays representable.
///
/// ## Why the sign is Negative
///
/// With `pa` as the reference, the determinant is
/// `(pb.x − pa.x)·(pc.y − pa.y) − (pb.y − pa.y)·(pc.x − pa.x)`. The first
/// product is about `6.7e-268`; the second is the product of two negative
/// factors, about `+1.4e-14`, and dominates. The determinant is therefore
/// negative. The oracle agrees.
#[test]
fn extreme_range_overflowing_products() {
    let pa = [3.81752867881136e-310_f64, 3.033653279310457e-294_f64];
    let pb = [4.778309726736481e-299_f64, 2.4688758965194162e-307_f64];
    let pc = [-4.539427514930182e279_f64, 1.4102612927432247e31_f64];

    let result = orient2d(pa, pb, pc).expect("finite inputs");
    let expected = oracle_orient2d(pa, pb, pc);

    assert_eq!(
        result, expected,
        "orient2d([{:e}, {:e}], [{:e}, {:e}], [{:e}, {:e}]): \
         got {result:?}, oracle says {expected:?}",
        pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]
    );
    assert_eq!(result, Sign::Negative, "this input must return Negative");
}

/// A determinant far below the smallest subnormal, on input the exactness
/// window can absorb.
///
/// `s = 2^-1019`: the exact determinant is `s² = 2^-2038`, far below `2^-1074`
/// (the smallest subnormal), so a native-`f64` evaluation rounds it to zero.
/// Every non-zero coordinate is `s`, so the input spans a single bit and an
/// exact power-of-two rescale moves it into the exactness window, where the
/// expansion stages return the sign. The triangle is counterclockwise, so the
/// expected sign is `Positive`, which the oracle confirms.
#[test]
fn orient2d_determinant_below_subnormal_range() {
    let s = f64::from_bits(0x0040_0000_0000_0000u64); // 2^-1019, a normal f64
    let pa = [0.0f64, 0.0];
    let pb = [s, 0.0];
    let pc = [0.0, s];
    // det = s² = 2^-2038, positive: counterclockwise triangle
    assert_eq!(orient2d(pa, pb, pc), Ok(Sign::Positive));
    assert_eq!(oracle_orient2d(pa, pb, pc), Sign::Positive);
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
fn canonical_orient2d_cases() -> Vec<([f64; 2], [f64; 2], [f64; 2])> {
    let tiny = f64::from_bits(1); // 2^-1074, the minimum positive subnormal
    let s1019 = f64::from_bits(0x0040_0000_0000_0000u64); // 2^-1019

    vec![
        // Shewchuk-style fixed cases (Component 2).
        ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
        ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]),
        ([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]),
        ([0.0, 0.0], [1.0, 1.0], [2.0, 2.0 + 1e-15]),
        (
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
            [1.0 / 7.0, 1.0 / 7.0],
        ),
        // Exact determinant below the smallest subnormal (Component 4).
        ([0.0, 0.0], [s1019, 0.0], [0.0, s1019]),
        // Fixed inputs: mixed and extreme magnitude ranges, subnormal coordinates.
        (
            [3.817_528_678_811_36e-310, 3.486_039_150_627_631e30],
            [2.275_424_3e-317, 3.486_039_150_627_631e30],
            [3.529_536_965_341_344_5e30, 6.681_002_276_805_634e-240],
        ),
        (
            [3.785_767_986_820_787e-270, 0.0],
            [8.289_046e-317, 0.0],
            [3.529_536_301_617_365e30, 3.621_778_591_446_635e-259],
        ),
        (
            [3.817_528_678_811_36e-310, 3.033_653_279_310_457e-294],
            [4.778_309_726_736_481e-299, 2.468_875_896_519_416_2e-307],
            [-4.539_427_514_930_182e279, 1.410_261_292_743_224_7e31],
        ),
        // Coordinates at or near the smallest subnormal.
        ([tiny, 0.0], [0.0, tiny], [0.0, 0.0]),
        ([0.0, 0.0], [tiny, 0.0], [0.0, tiny]),
        ([3.0 * tiny, tiny], [tiny, 3.0 * tiny], [tiny, tiny]),
        ([1e-300, 0.0], [0.0, tiny], [0.0, 0.0]),
    ]
}

/// The two oracles must agree on every canonical fixture.
///
/// This is the fixed half of the dual-oracle check. It runs the `BigRational`
/// oracle, which is otherwise unused, against the `BigInt` oracle that the
/// proptests depend on, so a divergence between the two representations is
/// caught on every run.
#[test]
fn dual_oracle_agreement_orient2d_canonical() {
    let cases = canonical_orient2d_cases();
    assert!(
        cases.len() >= 13,
        "canonical table shrank: expected at least 13 cases, found {}",
        cases.len()
    );
    for (i, (pa, pb, pc)) in cases.iter().enumerate() {
        assert_eq!(
            oracle_orient2d(*pa, *pb, *pc),
            oracle_orient2d_bigrational(*pa, *pb, *pc),
            "canonical case {i}: BigInt oracle disagrees with BigRational oracle \
             on pa={pa:?} pb={pb:?} pc={pc:?}"
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
    /// defect in this test file rather than in `orient2d`.
    #[test]
    fn dual_oracle_agreement_orient2d_random(
        ax in finite_f64_bounded(), ay in finite_f64_bounded(),
        bx in finite_f64_bounded(), by in finite_f64_bounded(),
        cx in finite_f64_bounded(), cy in finite_f64_bounded(),
    ) {
        let pa = [ax, ay];
        let pb = [bx, by];
        let pc = [cx, cy];
        prop_assert_eq!(
            oracle_orient2d(pa, pb, pc),
            oracle_orient2d_bigrational(pa, pb, pc)
        );
    }
}

// ===========================================================================
// Subnormal coordinates
// ===========================================================================

// Coordinates at the smallest subnormal, `2^-1074`, give exact determinants as
// small as `2^-2148`, which no native-`f64` product can represent. Each case is
// checked against the exact oracle.
#[test]
fn subnormal_coordinates_agree_with_the_oracle() {
    let s = f64::from_bits(1);
    let cases = [
        ([s, 0.0], [0.0, s], [0.0, 0.0]),
        ([0.0, 0.0], [s, 0.0], [0.0, s]),
        ([3.0 * s, s], [s, 3.0 * s], [s, s]),
        ([1e-300, 0.0], [0.0, s], [0.0, 0.0]),
    ];
    for (i, (pa, pb, pc)) in cases.iter().enumerate() {
        assert_eq!(
            orient2d(*pa, *pb, *pc),
            Ok(oracle_orient2d(*pa, *pb, *pc)),
            "case {i}: orient2d disagrees with the exact oracle"
        );
    }
}
