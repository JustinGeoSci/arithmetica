// src/expansion.rs

//! Low-level floating-point expansion arithmetic primitives.
//!
//! This module provides the foundational two-term expansion operations that
//! underpin the adaptive predicates in this crate. Each operation produces an
//! *exact* result represented as a pair `(x, y)` of `f64` values whose sum
//! equals the true mathematical result.
//!
//! ## Background
//!
//! A **floating-point expansion** is a sequence of non-overlapping `f64`
//! values whose exact real-number sum equals a mathematical quantity that
//! cannot be represented in a single `f64`. The two-term case `(x, y)` with
//! `|x| >= |y|` and no overlap between the significands covers all results of
//! exact addition or multiplication of two `f64` values.
//!
//! The operations here are Algorithms 1–4 from Shewchuk (1997), §3. They are
//! the building blocks for the multi-term expansion arithmetic used by
//! [`crate::orient2d`] and [`crate::incircle`].
//!
//! ## Exactness is conditional, and the condition is the caller's
//!
//! "Exact" above means exact in the model Shewchuk's proofs assume: an
//! unbounded exponent range. His §2.1 says so plainly: *"This article does
//! not address issues of overflow and underflow, so I allow the exponent to
//! be an integer in the range ..."*. Binary64's exponent range is bounded,
//! and at both ends these functions stop being exact:
//!
//! - **Overflow.** A value the function forms is infinite, and the returned
//!   pair does not sum to the true result. The result is non-finite rather
//!   than a plausible finite number, because an infinity can only return to
//!   the finite range through `inf - inf`, which is NaN.
//! - **Inexact underflow.** A value the function forms is too small to be
//!   represented exactly. The returned pair is finite, sums to something
//!   other than the true result, and carries **no indication that it is
//!   wrong**. This is the dangerous end.
//!
//! The condition each function needs is on its own doc. In short: the
//! addition primitives need only that nothing overflows, while the
//! multiplication primitives also need their product to stay above about
//! `2^-970`, and anything built on them inherits both.
//!
//! **The predicates in this crate guarantee the condition** before they use
//! these primitives, with the exactness window of the private `range` module;
//! the derivation is in `docs/exactness_window.md`. A caller who uses these
//! functions directly takes that duty on. If you cannot establish it for your
//! own inputs, use the predicates instead: they resolve every finite input
//! exactly, by falling back to integer arithmetic where floating-point
//! expansions cannot reach.
//!
//! ## IEEE 754 dependence
//!
//! All correctness proofs assume IEEE 754 round-to-nearest-even arithmetic
//! with no excess precision. On x86 with the x87 FPU in 80-bit mode these
//! functions are incorrect; SSE2 (`-C target-feature=+sse2`) or `x86-64`
//! (which uses SSE2 by default) must be used. All modern Rust targets for
//! x86-64 satisfy this requirement.
//!
//! ## References
//!
//! Shewchuk, J.R. (1997). Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates. *Discrete & Computational Geometry*,
//! 18(3), 305–363. <https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf>

// ===========================================================================
// Error bound constants
// ===========================================================================

/// Machine epsilon as defined in Shewchuk's predicates.c: the unit roundoff
/// `u = 2^-53` for IEEE 754 double precision.
///
/// Shewchuk's predicates.c computes this dynamically by halving 1.0 until
/// `1.0 + epsilon == 1.0` under round-to-nearest-even, which yields `2^-53`.
/// This equals `f64::EPSILON / 2.0` in Rust (Rust's `f64::EPSILON` is `2^-52`,
/// the machine epsilon in the sense of the smallest value where `1.0 + e ≠ 1.0`,
/// which is one ULP of 1.0; the predicates.c convention is the unit roundoff,
/// one half of that). The error bound constants below use this convention to
/// match `predicates.c` exactly.
const EPSILON: f64 = f64::EPSILON / 2.0; // 2^-53, the unit roundoff

/// Error bound for the first fast-path stage of [`crate::orient2d`].
///
/// If `|det| > CCW_ERR_BOUND_A * (|adx * bdy| + |bdx * ady| + ...)`, the
/// fast-path floating-point sign is provably correct and the exact fallback
/// need not fire.
///
/// Formula: `(3 + 16ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// The second-order coefficient is 16, not the 8 of [`RESULT_ERR_BOUND`]:
/// the two constants share their leading coefficient and nothing else. The
/// difference matters beyond matching the source. The underflow-protected
/// filter in `orient2d` is valid only with a constant above `(3 + 12ε) · ε`,
/// which `(3 + 16ε) · ε` is and `(3 + 8ε) · ε` is not.
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundA`; Shewchuk 1997,
/// Table 1.
pub(crate) const CCW_ERR_BOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;

/// Error bound for the second adaptive stage of [`crate::orient2d`].
///
/// Applied after one round of expansion arithmetic. If the adaptive result
/// satisfies this bound, it is returned; otherwise the full exact computation
/// fires.
///
/// Formula: `(2 + 12ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundB`; Shewchuk 1997,
/// Table 1.
pub(crate) const CCW_ERR_BOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;

/// Error bound for the third exact stage of [`crate::orient2d`].
///
/// Applied to the final multi-term expansion result. A nonzero value larger
/// than this bound has the correct sign; a value within this bound of zero
/// requires the most expensive exact-arithmetic stage.
///
/// Formula: `(9 + 64ε) · ε²` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundC`.
pub(crate) const CCW_ERR_BOUND_C: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

/// Error bound for the first fast-path stage of [`crate::incircle`].
///
/// Analogous to [`CCW_ERR_BOUND_A`] but for the `incircle` determinant, which
/// involves 3×3 minors and has a larger coefficient.
///
/// Formula: `(10 + 96ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `iccerrboundA`.
pub(crate) const ICC_ERR_BOUND_A: f64 = (10.0 + 96.0 * EPSILON) * EPSILON;

/// Error bound for the second adaptive stage of [`crate::incircle`].
///
/// Formula: `(4 + 48ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `iccerrboundB`.
pub(crate) const ICC_ERR_BOUND_B: f64 = (4.0 + 48.0 * EPSILON) * EPSILON;

/// Error bound for the third exact stage of [`crate::incircle`].
///
/// Formula: `(44 + 576ε) · ε²` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `iccerrboundC`.
pub(crate) const ICC_ERR_BOUND_C: f64 = (44.0 + 576.0 * EPSILON) * EPSILON * EPSILON;

/// Error bound applied to the magnitude of the running determinant estimate
/// in the Stage 3 exact fallback of both `orient2d` and `incircle`.
///
/// Represents the relative rounding error accumulated when summing an
/// expansion using `estimate()`. Shewchuk derives this as `(3 + 8ε) · ε` and
/// uses it as a coefficient on `|det|` in the final adaptive stage guard.
///
/// It is its own constant. `ccwerrboundA` is `(3 + 16ε) · ε`: the same leading
/// coefficient, a different second-order one.
///
/// Formula: `(3 + 8ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `resulterrbound`.
pub(crate) const RESULT_ERR_BOUND: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;

// ===========================================================================
// Error-bound constants for orient3d: Shewchuk predicates.c `o3derrboundA/B/C`
// ===========================================================================

/// Stage 1 fast-path error bound for `orient3d`.
///
/// If `|det| ≥ O3D_ERR_BOUND_A · permanent`, the sign of `det` is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundA`.
/// Formula: (7 + 56ε)ε where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_A: f64 = (7.0 + 56.0 * EPSILON) * EPSILON;

/// Stage 2 adaptive-path error bound for `orient3d`.
///
/// If `|estimate(fin)| ≥ O3D_ERR_BOUND_B · permanent` after Stage 2, the sign
/// of the estimate is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundB`.
/// Formula: (3 + 28ε)ε where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_B: f64 = (3.0 + 28.0 * EPSILON) * EPSILON;

/// Stage 2 intermediate re-test error bound for `orient3d`.
///
/// Used in combination with `RESULT_ERR_BOUND · |det|` for the first-order
/// tail correction re-test before committing to Stage 3.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundC`.
/// Formula: (26 + 288ε)ε² where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_C: f64 = (26.0 + 288.0 * EPSILON) * EPSILON * EPSILON;

// ===========================================================================
// Error-bound constants for insphere: Shewchuk predicates.c `isperrboundA/B/C`
// ===========================================================================

/// Stage 1 fast-path error bound for `insphere`.
///
/// If `|det| ≥ ISP_ERR_BOUND_A · permanent`, the sign of `det` is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `isperrboundA`.
/// Formula: (16 + 224ε)ε where ε = 2^-53.
pub(crate) const ISP_ERR_BOUND_A: f64 = (16.0 + 224.0 * EPSILON) * EPSILON;

/// Stage 2 adaptive-path error bound for `insphere`.
///
/// If `|estimate(fin)| ≥ ISP_ERR_BOUND_B · permanent` after Stage 2, the sign
/// of the estimate is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `isperrboundB`.
/// Formula: (5 + 72ε)ε where ε = 2^-53.
pub(crate) const ISP_ERR_BOUND_B: f64 = (5.0 + 72.0 * EPSILON) * EPSILON;

/// Stage 2 intermediate re-test error bound for `insphere`.
///
/// Used in combination with `RESULT_ERR_BOUND · |det|` for the first-order
/// tail correction re-test before committing to Stage 3.
///
/// **Not used.** `insphere` has no Stage 3: input that its Stage 2 cannot
/// certify goes straight to the exact integer stage, because a sound
/// first-order correction would have to include the lift terms. The constant
/// is kept so that the table of Shewchuk's error bounds is complete and every
/// entry is pinned by `err_bound_constants_have_their_table_values`.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `isperrboundC`.
/// Formula: (71 + 1408ε)ε² where ε = 2^-53.
#[allow(dead_code)]
pub(crate) const ISP_ERR_BOUND_C: f64 = (71.0 + 1408.0 * EPSILON) * EPSILON * EPSILON;

// ===========================================================================
// Exact two-term arithmetic
// ===========================================================================

/// Computes the exact sum of two `f64` values as a two-term expansion.
///
/// Returns `(x, y)` where `x = fl(a + b)` is the standard floating-point
/// rounded sum and `y` is the rounding error, satisfying `x + y == a + b`
/// exactly as real numbers.
///
/// ## How it works
///
/// After computing `x = fl(a + b)`, the subtraction `bv = x - a` recovers
/// what `b` *would have been* if the addition had been performed in the
/// opposite order; it is the part of `b` that survived into `x`. The
/// difference `b - bv` is then the part of `b` that did **not** make it into
/// `x` (it was lost to rounding). Similarly, `a - (x - bv)` recovers the
/// part of `a` that was lost. Summing these two residuals yields `y`, the
/// exact rounding error.
///
/// This six-operation sequence works for **any** relative magnitudes of `a`
/// and `b`. When `|a| >= |b|` is guaranteed, [`fast_two_sum`] achieves the
/// same result with four operations.
///
/// ## Exactness
///
/// Exact whenever `a + b` does not overflow. Underflow is harmless here: if
/// the rounded sum is subnormal then `a + b` is a multiple of `2^-1074` below
/// `2^-1022`, which is representable, so the addition was exact and `y` is
/// zero. See the module doc for what the module-wide condition is and who
/// owes it.
///
/// This is Algorithm 2 (TwoSum) from Shewchuk (1997), p. 8.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bv = x - a;
    let y = (a - (x - bv)) + (b - bv);
    (x, y)
}

/// Computes the exact sum of two `f64` values as a two-term expansion.
///
/// Returns `(x, y)` where `x = fl(a + b)` and `y` is the rounding error,
/// satisfying `x + y == a + b` exactly as real numbers.
///
/// ## Precondition
///
/// `|a| >= |b|` must hold before calling this function. In debug builds this
/// is checked with [`debug_assert!`] and will panic if violated. In release
/// builds the check is elided; a violation silently produces an incorrect `y`
/// that does **not** satisfy the exact-sum property, corrupting any subsequent
/// adaptive computation. The precondition is always established by the calling
/// context.
///
/// ## Why the precondition is necessary
///
/// When `|a| < |b|`, the bits of `a` may fall below the rounding point of `b`
/// and be absorbed completely, so `x - a` no longer recovers the part of `b`
/// that survived; it recovers a quantity that mixes contributions from both
/// operands in a way that does not decompose cleanly into the rounding errors
/// of `a` and `b` separately. The exact-sum identity `x + y == a + b` fails
/// in this case.
///
/// ## Exactness
///
/// As for [`two_sum`]: exact whenever `a + b` does not overflow, with
/// underflow harmless for the same reason. This is in addition to the
/// `|a| >= |b|` precondition above, which is about the algorithm rather than
/// about the exponent range.
///
/// This is Algorithm 1 (FastTwoSum) from Shewchuk (1997), p. 7.
#[inline]
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    debug_assert!(
        a.abs() >= b.abs(),
        "fast_two_sum precondition violated: |a| < |b| (a={a}, b={b})"
    );
    let x = a + b;
    let y = b - (x - a);
    (x, y)
}

/// Splits `a` into two non-overlapping `f64` values `(hi, lo)` such that
/// `hi + lo == a` exactly and `|hi| >= |lo|`.
///
/// Uses Veltkamp's splitting constant `2^27 + 1 = 134_217_729`. Multiplying
/// by this constant shifts the significand of `a` so that the high 27 bits
/// are isolated in the rounded product `c`; subtracting `c - a` then leaves
/// the high 27 bits of `a` in `hi` and the remaining 26 bits in `lo =
/// a - hi`. Each half fits exactly in `f64`'s 53-bit significand, and their
/// product with a corresponding split of another value can be computed without
/// further rounding error.
///
/// **Precondition:** `a` must be finite. Non-finite inputs produce unspecified
/// output; the caller must ensure finiteness before calling.
///
/// Algorithm 3 (Split) from Shewchuk (1997), p. 9.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = 134_217_729.0_f64 * a; // 2^27 + 1
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    (a_hi, a_lo)
}

/// Computes the exact product of two `f64` values as a two-term expansion.
///
/// Returns `(x, y)` where `x = fl(a * b)` is the standard floating-point
/// rounded product and `y` is the rounding error, satisfying
/// `x + y == a * b` exactly as real numbers.
///
/// ## Why Veltkamp-Dekker splitting works
///
/// After splitting `a` into `(a_hi, a_lo)` and `b` into `(b_hi, b_lo)` via
/// Veltkamp splitting (the private `split` helper), each half occupies at most
/// 27 significant bits. A product of two
/// 27-bit values has at most 54 significant bits, just at the edge of `f64`'s
/// 53-bit significand, but crucially, the leading bit of `a_hi * b_hi`
/// cancels exactly with `x` (the rounded full product), leaving the residual
/// expressible in `f64`. The cross products `a_hi * b_lo` and `a_lo * b_hi`
/// each fit in 54 bits, so no precision is lost when they are computed. The
/// sum of all four partial products minus `x` gives `y` exactly.
///
/// ## Why not FMA?
///
/// An FMA-based `TwoProduct` (`x = a * b; y = fma(a, b, -x)`) is simpler and
/// faster but depends on hardware FMA support. Veltkamp-Dekker is correct on
/// every IEEE 754-compliant platform without any hardware feature dependency
/// and without a runtime capability check. See Decision 7 in the project
/// design record.
///
/// ## Exactness
///
/// Exact when both of these hold:
///
/// 1. **No overflow.** The split forms `(2^27 + 1) * a`, so `|a|` and `|b|`
///    must stay below about `2^995`, and the product `a * b` itself must be
///    finite.
/// 2. **The product is not too small.** Writing `a = m_a * 2^(e_a - 52)` and
///    `b = m_b * 2^(e_b - 52)` with integer significands, the exact product
///    `a * b` is an integer multiple of `2^(e_a + e_b - 104)`, and so is every
///    partial product the algorithm forms. Those are representable in binary64
///    exactly when that granule is at least `2^-1074`, which is
///    `e_a + e_b >= -970`: the product's magnitude must be roughly `2^-970`
///    (about `1e-292`) or more. Below it, `y` loses bits and `x + y` is no
///    longer `a * b`, with nothing in the return value to say so.
///
/// Each partial product also fits in 53 bits under the same condition: it is
/// a multiple of that granule and bounded by `|a * b|`, so it needs at most 52
/// bits.
///
/// Algorithm 4 (TwoProduct) from Shewchuk (1997), p. 9, using Split (p. 9)
/// rather than the FMA variant.
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    let y = ((a_hi * b_hi - x) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (x, y)
}

// ===========================================================================
// Higher-order expansion operations
// ===========================================================================

/// Computes `two_product(a, b)` where `b` has already been split into `(b_hi, b_lo)`.
///
/// Returns `(x, y)` where `x = fl(a * b)` and `x + y == a * b` exactly as
/// real numbers. Equivalent to `two_product(a, b)` but avoids re-splitting `b`
/// on every call. Used by `scale_expansion` to amortize the cost of splitting
/// `b` across all components of the expansion.
///
/// `b_hi` and `b_lo` must satisfy `b_hi + b_lo == b` exactly with the
/// properties guaranteed by `split(b)`.
///
/// Corresponds to the `Two_Product_Presplit` macro in Shewchuk's predicates.c.
#[inline]
fn two_product_presplit(a: f64, b: f64, b_hi: f64, b_lo: f64) -> (f64, f64) {
    let x = a * b;
    let (a_hi, a_lo) = split(a);
    let y = ((a_hi * b_hi - x) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (x, y)
}

/// Adds a scalar `b` to expansion `e`, producing expansion `h`.
///
/// Returns the number of live components written into `h`. The output
/// satisfies `sum(h[..len]) == sum(e) + b` exactly as real numbers.
///
/// ## Caller-owned buffer contract
///
/// `h` must have length at least `e.len() + 1`. A `debug_assert!` verifies
/// this in debug builds. The first `len` elements of `h` contain the result;
/// elements beyond `len` are unspecified.
///
/// ## Exactness
///
/// Built from [`two_sum`], so exact whenever no addition along the way
/// overflows. The running sum passes through every component of `e`, so the
/// value to watch is the total, not the individual terms. See the module doc
/// for the module-wide condition.
///
/// ## References
///
/// Shewchuk (1997), `grow_expansion_zeroelim`, predicates.c.
pub fn grow_expansion(e: &[f64], b: f64, h: &mut [f64]) -> usize {
    debug_assert!(
        h.len() > e.len(),
        "grow_expansion: h too small (len={}, need {})",
        h.len(),
        e.len() + 1
    );
    let mut q = b;
    let mut h_len = 0usize;
    for &e_i in e {
        let (q_new, h_i) = two_sum(q, e_i);
        if h_i != 0.0 {
            h[h_len] = h_i;
            h_len += 1;
        }
        q = q_new;
    }
    if q != 0.0 {
        h[h_len] = q;
        h_len += 1;
    }
    h_len
}

/// Multiplies expansion `e` by scalar `b`, producing expansion `h`.
///
/// Returns the number of live components written into `h`. The output
/// satisfies `sum(h[..len]) == sum(e) * b` exactly as real numbers.
///
/// ## Caller-owned buffer contract
///
/// `h` must have length at least `2 * e.len()`. A `debug_assert!` verifies
/// this in debug builds.
///
/// ## Precondition
///
/// `e` must be a valid non-overlapping expansion (the non-overlapping
/// property is maintained by all operations in this module; callers that
/// construct expansions manually are responsible for the invariant).
///
/// ## Exactness
///
/// Built from [`two_product`] and [`two_sum`], so it needs both of their
/// conditions: no overflow, and every product `e[i] * b` above about
/// `2^-970`. The smallest component of a non-overlapping expansion can be
/// many orders below its largest, so this is the operation where an unwary
/// caller is most likely to lose exactness silently; the leading term can
/// look comfortably scaled while the trailing ones do not. See the module doc
/// for the module-wide condition.
///
/// ## References
///
/// Shewchuk (1997), `scale_expansion_zeroelim`, predicates.c.
pub fn scale_expansion(e: &[f64], b: f64, h: &mut [f64]) -> usize {
    debug_assert!(
        h.len() >= 2 * e.len(),
        "scale_expansion: h too small (len={}, need {})",
        h.len(),
        2 * e.len()
    );
    if e.is_empty() {
        return 0;
    }
    // Split b once here; all per-component products reuse (b_hi, b_lo).
    let (b_hi, b_lo) = split(b);
    let mut h_len = 0usize;

    // Bootstrap: first component produces only a single (q, hh) pair.
    let (mut q, hh) = two_product_presplit(e[0], b, b_hi, b_lo);
    if hh != 0.0 {
        h[h_len] = hh;
        h_len += 1;
    }

    for &e_i in &e[1..] {
        let (product1, product0) = two_product_presplit(e_i, b, b_hi, b_lo);

        // Accumulate the low term of the product with the running carry.
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h[h_len] = hh;
            h_len += 1;
        }

        // Merge the high term of the product with the updated carry.
        // Shewchuk uses Fast_Two_Sum here, which is exact only when
        // |product1| >= |sum|. That ordering does hold for valid nonoverlapping
        // expansions, but the argument is not short, and two_sum is exact with
        // no ordering condition at all, for the price of two more operations.
        let (q_new, hh) = two_sum(product1, sum);
        if hh != 0.0 {
            h[h_len] = hh;
            h_len += 1;
        }
        q = q_new;
    }

    if q != 0.0 {
        h[h_len] = q;
        h_len += 1;
    }
    h_len
}

/// Adds expansions `e` and `f`, producing expansion `h`.
///
/// Returns the number of live components in `h`. The output satisfies
/// `sum(h[..len]) == sum(e) + sum(f)` exactly as real numbers.
///
/// ## Algorithm
///
/// The algorithm processes each component of `f` in turn, growing the
/// intermediate result stored in `h` by that component via repeated application
/// of the `two_sum` kernel. The passes maintain the non-overlapping invariant
/// throughout, so the intermediate result is always a valid expansion. A final
/// zero-elimination pass compacts the output.
///
/// ## Caller-owned buffer contract
///
/// `h` must have length at least `e.len() + f.len()`. A `debug_assert!`
/// verifies this in debug builds.
///
/// ## Exactness
///
/// Built from [`two_sum`], so exact whenever no addition along the way
/// overflows. Both inputs must already be valid non-overlapping expansions;
/// this function preserves that property but cannot establish it. See the
/// module doc for the module-wide condition.
///
/// ## References
///
/// Shewchuk (1997), `expansion_sum_zeroelim2`, predicates.c.
pub fn expansion_sum(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    debug_assert!(
        h.len() >= e.len() + f.len(),
        "expansion_sum: h too small (len={}, need {})",
        h.len(),
        e.len() + f.len()
    );

    // Degenerate cases: one operand is empty. Copy the other, zero-eliminating.
    if f.is_empty() {
        let mut h_len = 0usize;
        for &v in e {
            if v != 0.0 {
                h[h_len] = v;
                h_len += 1;
            }
        }
        return h_len;
    }
    if e.is_empty() {
        let mut h_len = 0usize;
        for &v in f {
            if v != 0.0 {
                h[h_len] = v;
                h_len += 1;
            }
        }
        return h_len;
    }

    // Phase 1: grow f[0] through all of e, writing every intermediate value
    // (including zeros) into h[0..e.len()], and the final carry into h[e.len()].
    // Zeros must be written because phase 2 reads back h[findex..=hlast].
    let mut q = f[0];
    for (hindex, &e_i) in e.iter().enumerate() {
        let (q_new, h_i) = two_sum(q, e_i);
        h[hindex] = h_i; // Written even when zero; phase 2 reads this slot.
        q = q_new;
    }
    let mut hlast = e.len();
    h[hlast] = q; // Final carry from f[0] grow.

    // Phase 2: for each remaining f[i], grow through h[findex..=hlast] in place.
    // Slots h[0..findex] are already finalized (their components are too small
    // to interact with f[i] without violating the non-overlapping invariant).
    for (fi, &f_i) in f[1..].iter().enumerate() {
        let findex = fi + 1;
        q = f_i;
        for h_slot in h.iter_mut().take(hlast + 1).skip(findex) {
            let h_now = *h_slot; // Read before write; they alias the same slot.
            let (q_new, h_h) = two_sum(q, h_now);
            *h_slot = h_h;
            q = q_new;
        }
        hlast += 1;
        h[hlast] = q;
    }

    // Zero-elimination pass: compact h[0..=hlast], removing exactly-0.0 entries.
    let total = hlast + 1;
    let mut h_len = 0usize;
    for i in 0..total {
        let v = h[i];
        if v != 0.0 {
            h[h_len] = v;
            h_len += 1;
        }
    }
    h_len
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
// test-lint-posture: see the workspace lint comment in Cargo.toml
#[allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::range_plus_one,
    clippy::no_effect_underscore_binding
)]
mod tests {
    use super::*;

    // ===========================================================================
    // two_sum
    // ===========================================================================

    /// Constructs a case where naive f64 addition loses the low bits of b.
    ///
    /// a = 2^53 (the largest f64 integer, i.e. 9007199254740992)
    /// b = 1.0
    ///
    /// fl(a + b) = a because 1.0 is below the ULP of a (which is 2.0).
    /// two_sum must recover b = 1.0 in the error term y.
    #[test]
    fn two_sum_recovers_exact_sum() {
        let a = (1_u64 << 53) as f64; // 2^53
        let b = 1.0_f64;

        // Demonstrate that naive addition loses the bit.
        assert_eq!(a + b, a, "precondition: naive addition absorbs b");

        let (x, y) = two_sum(a, b);
        // x is the rounded sum (equals a due to rounding)
        assert_eq!(x, a);
        // y must recover the lost bit
        assert_eq!(y, b);
        // The fundamental property: x + y == a + b exactly
        // Here x + y = 2^53 + 1.0 which is NOT representable in one f64,
        // so we verify the individual components rather than x + y directly.
        assert_eq!(x + y, a); // loses y again, proving y is below ULP of x
        // But the *exact* sum is a + b = 2^53 + 1; we verify x and y encode it.
        let exact: i64 = (a as i64) + (b as i64);
        assert_eq!((x as i64) + (y as i64), exact);
    }

    /// A case where both a and b are in range for exact recovery.
    ///
    /// a = 1.0, b = 2^-53 (half of machine epsilon; below the ULP of 1.0)
    /// fl(a + b) = 1.0 (b is rounded away)
    /// two_sum should recover b in y.
    #[test]
    fn two_sum_small_b_is_recovered() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0; // 2^-53, half ULP of 1.0

        assert_eq!(a + b, a, "precondition: naive addition loses b");

        let (x, y) = two_sum(a, b);
        assert_eq!(x, 1.0);
        assert_eq!(y, b, "error term must recover b exactly");
    }

    /// Verify that two_sum is correct when b > a (no magnitude precondition).
    #[test]
    fn two_sum_no_precondition_on_magnitude() {
        // b >> a; ensure the result is still exact
        let a = 1.0_f64;
        let b = (1_u64 << 53) as f64;
        let (x, y) = two_sum(a, b);
        // x = fl(a + b) = b (a is below the ULP of b)
        assert_eq!(x, b);
        assert_eq!(y, a);
        assert_eq!((x as i64) + (y as i64), (a as i64) + (b as i64));
    }

    // ===========================================================================
    // fast_two_sum
    // ===========================================================================

    /// Same scenario as two_sum_recovers_exact_sum, but with |a| >= |b|.
    #[test]
    fn fast_two_sum_recovers_exact_sum() {
        let a = (1_u64 << 53) as f64; // 2^53; |a| >> |b|
        let b = 1.0_f64;

        assert!(a.abs() >= b.abs(), "precondition satisfied");
        assert_eq!(a + b, a, "precondition: naive addition absorbs b");

        let (x, y) = fast_two_sum(a, b);
        assert_eq!(x, a);
        assert_eq!(y, b);
        let exact: i64 = (a as i64) + (b as i64);
        assert_eq!((x as i64) + (y as i64), exact);
    }

    /// fast_two_sum with a = 1.0 and b = 2^-53 (same as two_sum companion).
    #[test]
    fn fast_two_sum_small_b_is_recovered() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0;

        assert!(a.abs() >= b.abs());
        assert_eq!(a + b, a);

        let (x, y) = fast_two_sum(a, b);
        assert_eq!(x, 1.0);
        assert_eq!(y, b);
    }

    /// Precondition violation must panic in debug builds.
    ///
    /// fast_two_sum(1.0, 2.0) violates |a| >= |b| since 1.0 < 2.0.
    /// The debug_assert! must fire.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "fast_two_sum precondition violated")]
    fn fast_two_sum_panics_on_precondition_violation() {
        let _ = fast_two_sum(1.0, 2.0);
    }

    // ===========================================================================
    // split (private — tested indirectly and directly via module visibility)
    // ===========================================================================

    /// Split must satisfy hi + lo == a exactly for a representative value.
    /// We use a value with all 53 mantissa bits set.
    #[test]
    fn split_exact_reconstruction() {
        // 2^53 - 1: all 53 significand bits set, largest odd integer in f64
        let a = ((1_u64 << 53) - 1) as f64;
        let (hi, lo) = split(a);
        // hi + lo must equal a exactly (this is the core Split guarantee)
        assert_eq!(hi + lo, a);
        // hi and lo must not overlap: |hi| >= |lo|
        assert!(hi.abs() >= lo.abs());
    }

    // ===========================================================================
    // two_product
    // ===========================================================================

    /// Constructs a case where fl(a * b) loses the low-order bits.
    ///
    /// a = 2^27 + 1  (fits exactly in the high split-half of a 53-bit f64)
    /// b = 2^27 + 1
    ///
    /// a * b = 2^54 + 2^28 + 1
    ///
    /// 2^54 has more than 53 significant bits in this product (the leading bit
    /// is at position 54, and there is a bit at position 28 and position 0),
    /// so fl(a * b) rounds away the low bits. two_product must recover them.
    #[test]
    fn two_product_recovers_exact_product() {
        let a = (1_u64 << 27) as f64 + 1.0; // 2^27 + 1 = 134_217_729
        let b = a; // same value

        // True product: (2^27 + 1)^2 = 2^54 + 2*2^27 + 1 = 2^54 + 2^28 + 1
        // 2^54 + 2^28 + 1 requires 55 bits, so fl(a*b) must lose at least
        // the low-order bit.
        let naive = a * b;

        let (x, y) = two_product(a, b);
        assert_eq!(x, naive, "x must equal the rounded product");

        // Verify x + y equals the exact product using i64 arithmetic.
        // (2^27 + 1)^2 = 2^54 + 2^28 + 1 = 18_014_398_777_917_441
        let exact: i64 = 18_014_398_777_917_441;
        let reconstructed: i64 = (x as i64) + (y as i64);
        assert_eq!(reconstructed, exact, "x + y must equal exact product");

        // The naive product must differ from the exact value, confirming the
        // test case actually demonstrates a precision loss.
        assert_ne!(naive as i64, exact, "naive f64 product must lose bits");
    }

    /// two_product must be exact for inputs whose product fits in 53 bits.
    #[test]
    fn two_product_exact_when_no_rounding() {
        let a = 3.0_f64;
        let b = 4.0_f64;
        let (x, y) = two_product(a, b);
        assert_eq!(x, 12.0);
        assert_eq!(y, 0.0, "no rounding error for exact product");
    }

    // ===========================================================================
    // grow_expansion
    // ===========================================================================

    /// Basic case: grow_expansion(&[1.0], 2.0) must sum to 3.0.
    ///
    /// 1.0 + 2.0 = 3.0 exactly in f64, so the result is a single component [3.0].
    #[test]
    fn grow_expansion_simple_sum() {
        let e = [1.0_f64];
        let mut h = [0.0_f64; 2];
        let len = grow_expansion(&e, 2.0, &mut h);
        assert!(len >= 1, "must produce at least one component");
        let sum: f64 = h[..len].iter().sum();
        assert_eq!(sum, 3.0_f64, "expansion sum must equal 1.0 + 2.0");
    }

    /// Case where the rounding error term h_i is non-zero.
    ///
    /// grow_expansion(&[1e-16], 1.0): because 1e-16 < ULP(1.0)/2 ≈ 1.11e-16,
    /// fl(1.0 + 1e-16) = 1.0; the low component is lost to rounding and must
    /// appear as h_i in the error slot. The result should be [1e-16, 1.0].
    #[test]
    fn grow_expansion_error_term_is_nonzero() {
        let e = [1e-16_f64];
        let mut h = [0.0_f64; 2];
        let len = grow_expansion(&e, 1.0, &mut h);
        assert_eq!(len, 2, "must produce 2 components (error + carry)");
        // Low component captures what rounding lost.
        assert_eq!(h[0], 1e-16_f64, "low component must recover 1e-16");
        // High component is the rounded sum.
        assert_eq!(h[1], 1.0_f64, "high component must be 1.0");
        // Their exact sum (as reals) is 1.0 + 1e-16.
        // Naive f64 addition of h[0] + h[1] rounds back to 1.0, confirming
        // h[0] is genuinely below the ULP of h[1] and encodes extra precision.
        assert_eq!(
            h[0] + h[1],
            1.0_f64,
            "naive sum rounds away the low component"
        );
    }

    // ===========================================================================
    // scale_expansion
    // ===========================================================================

    /// scale_expansion(&[1.0, 1e-16], 3.0) must encode 3.0 + 3*1e-16 exactly.
    ///
    /// Because `1e-16_f64` is slightly below 10^-16 (the nearest representable
    /// double to 10^-16 is ≈ 9.9999...e-17), `fl(3.0 * 1e-16)` exceeds
    /// `ULP(3.0) / 2 ≈ 2.22e-16`, so `fl(3.0 + 3*1e-16)` rounds up to
    /// `3.0 + 2^-51 = 3.0000000000000004`. The expansion must produce at least
    /// 2 components to encode the full exact result; when summed in f64 the
    /// components collapse back to that nearest representable double.
    #[test]
    fn scale_expansion_exact_product_sum() {
        let e = [1.0_f64, 1e-16_f64];
        let mut h = [0.0_f64; 4];
        let len = scale_expansion(&e, 3.0, &mut h);
        assert_eq!(len, 2, "must produce 2 components");
        // All components must be nonzero (zero-elimination invariant).
        for (i, &v) in h.iter().enumerate().take(len) {
            assert_ne!(
                v, 0.0_f64,
                "component {i} must be nonzero after zero elimination"
            );
        }
        // Core correctness check: the f64 sum of the live components must equal
        // the nearest representable double to sum(e) * b = (1.0 + 1e-16) * 3.0.
        // Verifying the f64 sum confirms the expansion encodes the full exact
        // product: the low component captures the sub-ULP residual that would
        // otherwise be lost, and when added to the high component it reconstructs
        // the exact mathematical result to single-f64 precision.
        let sum: f64 = h[..len].iter().sum();
        assert_eq!(
            sum,
            3.0_f64 + 3e-16_f64,
            "sum of components must equal fl(3.0 + 3*1e-16) = {}",
            3.0_f64 + 3e-16_f64
        );
    }

    /// scale_expansion of an empty slice must return 0 without writing to h.
    #[test]
    fn scale_expansion_empty_input() {
        let e: [f64; 0] = [];
        let mut h = [0.0_f64; 0];
        let len = scale_expansion(&e, 5.0, &mut h);
        assert_eq!(len, 0);
    }

    // ===========================================================================
    // expansion_sum
    // ===========================================================================

    /// expansion_sum encodes the exact sum of two multi-component expansions.
    ///
    /// e = [1e-16, 1.0], f = [2.0]
    /// Exact sum: 1e-16 + 1.0 + 2.0 = 3.0 + 1e-16
    ///
    /// The result must be [1e-16, 3.0]: two components that together represent
    /// the exact value 3.0 + 1e-16 which is not representable in a single f64.
    #[test]
    fn expansion_sum_exact_result() {
        let e = [1e-16_f64, 1.0_f64]; // valid non-overlapping expansion
        let f = [2.0_f64];
        let mut h = [0.0_f64; 3];
        let len = expansion_sum(&e, &f, &mut h);
        assert_eq!(len, 2, "must produce 2 components");
        // Low component captures what was lost when 1e-16 was added to 3.0.
        assert_eq!(h[0], 1e-16_f64, "low component must be 1e-16");
        // High component is the rounded 1.0 + 2.0 sum.
        assert_eq!(h[1], 3.0_f64, "high component must be 3.0");
    }

    /// expansion_sum with both inputs empty must return 0.
    #[test]
    fn expansion_sum_both_empty() {
        let e: [f64; 0] = [];
        let f: [f64; 0] = [];
        let mut h = [0.0_f64; 0];
        let len = expansion_sum(&e, &f, &mut h);
        assert_eq!(len, 0);
    }

    /// expansion_sum with an empty f must copy e (with zero elimination).
    #[test]
    fn expansion_sum_empty_f() {
        let e = [1.0_f64, 2.0_f64];
        let f: [f64; 0] = [];
        let mut h = [0.0_f64; 2];
        let len = expansion_sum(&e, &f, &mut h);
        assert_eq!(len, 2);
        assert_eq!(h[0], 1.0_f64);
        assert_eq!(h[1], 2.0_f64);
    }

    // ===========================================================================
    // Error-bound constants: literal values
    // ===========================================================================

    /// Every error-bound constant has the value of its formula in Shewchuk's
    /// `exactinit()`.
    ///
    /// The expected values are written as bit patterns, not recomputed from
    /// the formulas. A test that evaluates the same expression as the constant
    /// cannot fail, however wrong the expression is; a bit pattern derived
    /// independently can.
    ///
    /// Each pattern follows from its formula by integer arithmetic. With
    /// `ε = 2^-53`, a first-order constant `(a + bε)ε` is `a·2^-53 + b·2^-106`
    /// and a second-order constant `(a + bε)ε²` is `a·2^-106 + b·2^-159`. In
    /// all thirteen the second term is a whole number of units in the last
    /// place of the first, so the sum is exactly representable and no rounding
    /// is involved.
    ///
    /// Worked example, `CCW_ERR_BOUND_A = 3·2^-53 + 16·2^-106`: the first term
    /// is `1.5·2^-52`, whose unit in the last place is `2^-104`; the second is
    /// `2^-102`, four of those units. The exponent field is `0x3cb`
    /// (`1023 - 52`), the significand of 1.5 is `0x8_0000_0000_0000`, and
    /// adding four units gives `0x3cb8_0000_0000_0004`.
    #[test]
    fn err_bound_constants_have_their_table_values() {
        // Taking the constant by name keeps the name in the failure message
        // and the constant under test from ever disagreeing.
        macro_rules! check {
            ($constant:ident, $bits:literal) => {
                assert_eq!(
                    $constant.to_bits(),
                    $bits,
                    "{} is {:#018x}, expected {:#018x}",
                    stringify!($constant),
                    $constant.to_bits(),
                    $bits
                );
            };
        }

        check!(RESULT_ERR_BOUND, 0x3cb8_0000_0000_0002_u64); // (3 + 8ε)ε
        check!(CCW_ERR_BOUND_A, 0x3cb8_0000_0000_0004_u64); // (3 + 16ε)ε
        check!(CCW_ERR_BOUND_B, 0x3cb0_0000_0000_0003_u64); // (2 + 12ε)ε
        check!(CCW_ERR_BOUND_C, 0x3982_0000_0000_0004_u64); // (9 + 64ε)ε²
        check!(O3D_ERR_BOUND_A, 0x3ccc_0000_0000_0007_u64); // (7 + 56ε)ε
        check!(O3D_ERR_BOUND_B, 0x3cb8_0000_0000_0007_u64); // (3 + 28ε)ε
        check!(O3D_ERR_BOUND_C, 0x399a_0000_0000_0009_u64); // (26 + 288ε)ε²
        check!(ICC_ERR_BOUND_A, 0x3cd4_0000_0000_0006_u64); // (10 + 96ε)ε
        check!(ICC_ERR_BOUND_B, 0x3cc0_0000_0000_0006_u64); // (4 + 48ε)ε
        check!(ICC_ERR_BOUND_C, 0x39a6_0000_0000_0009_u64); // (44 + 576ε)ε²
        check!(ISP_ERR_BOUND_A, 0x3ce0_0000_0000_0007_u64); // (16 + 224ε)ε
        check!(ISP_ERR_BOUND_B, 0x3cc4_0000_0000_0009_u64); // (5 + 72ε)ε
        check!(ISP_ERR_BOUND_C, 0x39b1_c000_0000_000b_u64); // (71 + 1408ε)ε²

        // The epsilon convention, pinned separately: the unit roundoff 2^-53,
        // not `f64::EPSILON` (2^-52). `0x3ca` is `1023 - 53`.
        assert_eq!(EPSILON.to_bits(), 0x3ca0_0000_0000_0000_u64);
    }
}
