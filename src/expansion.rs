// src/expansion.rs

//! Floating-point expansion arithmetic, the error-free primitives beneath
//! the adaptive predicates.
//!
//! Each two-term operation here returns its result exactly, as a pair
//! `(x, y)` of `f64` values whose sum is the true mathematical result. The
//! multi-term operations extend the same guarantee to whole expansions.
//!
//! ## Background
//!
//! A **floating-point expansion** is a sequence of non-overlapping `f64` values
//! whose exact real sum is a quantity that a single `f64` cannot represent.
//! The two-term case `(x, y)`, with `|x| >= |y|` and no overlap between the
//! significands, covers every result of the exact addition or
//! multiplication of two `f64` values.
//!
//! The two-term operations are Algorithms 1–4 of Shewchuk (1997), §3. They
//! are the building blocks of the multi-term expansion arithmetic on which
//! [`crate::orient2d`] and [`crate::incircle`] operate.
//!
//! ## Exactness is conditional, and the condition is the caller's
//!
//! Exactness here means exactness in the model Shewchuk's proofs assume, an
//! unbounded exponent range. His §2.1 states this plainly ("This article
//! does not address issues of overflow and underflow, so I allow the
//! exponent to be an integer in the range ..."). The exponent range of
//! binary64 is bounded, and at either end these functions stop being exact.
//!
//! - **Overflow.** A value the function forms is infinite, and the returned pair
//!   does not sum to the true result. The result is non-finite rather than a
//!   plausible finite number, because an infinity can return to the finite
//!   range only through `inf - inf`, which is NaN.
//! - **Inexact underflow.** A value the function forms is too small to be
//!   represented exactly. The returned pair is finite, sums to something
//!   other than the true result, and carries **no indication that it is wrong**.
//!   This is the dangerous end.
//!
//! Each function states the condition it needs in its own documentation.
//! The addition primitives need only that nothing overflows, while the
//! multiplication primitives also need their product to stay above about
//! `2^-970`, and anything built on them inherits both conditions.
//!
//! The predicates in this crate establish the condition before they use
//! these primitives, through the exactness window of the private `range`
//! module, whose derivation is in `docs/exactness_window.md`. A caller who
//! uses these functions directly takes on that duty. A caller who cannot
//! establish the condition for their own input should use the predicates
//! instead, which resolve every finite input exactly by falling back to
//! integer arithmetic where floating-point expansions cannot reach.
//!
//! ## IEEE 754 dependence
//!
//! Every correctness proof assumes IEEE 754 round-to-nearest-even
//! arithmetic with no excess precision. On x86 with the x87 FPU in 80-bit
//! mode these functions are incorrect, so SSE2 (`-C target-feature=+sse2`)
//! or `x86-64`, which uses SSE2 by default, is required. Every modern Rust
//! target for x86-64 satisfies this requirement.
//!
//! ## References
//!
//! Shewchuk, J.R. (1997). Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates. *Discrete & Computational Geometry*,
//! 18(3), 305–363. <https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf>

// ===========================================================================
// Error bound constants
// ===========================================================================

/// Machine epsilon as Shewchuk's predicates.c defines it, the unit roundoff
/// `u = 2^-53` of IEEE 754 double precision.
///
/// Shewchuk's predicates.c computes this value at run time, halving 1.0
/// until `1.0 + epsilon == 1.0` under round-to-nearest-even, which yields
/// `2^-53`. In Rust it equals `f64::EPSILON / 2.0`. Rust's `f64::EPSILON` is
/// `2^-52`, one ULP of 1.0 and the smallest value for which `1.0 + e ≠ 1.0`,
/// whereas the predicates.c convention is the unit roundoff, half of that.
/// The error-bound constants below follow the predicates.c convention, so
/// they match `predicates.c` exactly.
const EPSILON: f64 = f64::EPSILON / 2.0; // 2^-53, the unit roundoff

/// Error bound for the first fast-path stage of [`crate::orient2d`].
///
/// When `|det| > CCW_ERR_BOUND_A * (|adx * bdy| + |bdx * ady| + ...)`, the
/// fast-path floating-point sign is provably correct and the exact fallback
/// need not run.
///
/// Formula: `(3 + 16ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// The second-order coefficient is 16 rather than the 8 of
/// [`RESULT_ERR_BOUND`], the two constants sharing their leading
/// coefficient and nothing else. The difference matters beyond fidelity to
/// the source. The underflow-protected filter in `orient2d` is valid only
/// with a constant above `(3 + 12ε) · ε`, which `(3 + 16ε) · ε` is and
/// `(3 + 8ε) · ε` is not.
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundA`; Shewchuk 1997,
/// Table 1.
pub(crate) const CCW_ERR_BOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;

/// Error bound for the second adaptive stage of [`crate::orient2d`].
///
/// The bound applies after one round of expansion arithmetic. An adaptive
/// result that satisfies it is returned, and any other result passes on to
/// the full exact computation.
///
/// Formula: `(2 + 12ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundB`; Shewchuk 1997,
/// Table 1.
pub(crate) const CCW_ERR_BOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;

/// Error bound for the third exact stage of [`crate::orient2d`].
///
/// The bound applies to the final multi-term expansion result. A nonzero
/// value larger than this bound has the correct sign, while a value within
/// it of zero requires the most expensive exact-arithmetic stage.
///
/// Formula: `(9 + 64ε) · ε²` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `ccwerrboundC`.
pub(crate) const CCW_ERR_BOUND_C: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

/// Error bound for the first fast-path stage of [`crate::incircle`].
///
/// This is the counterpart of [`CCW_ERR_BOUND_A`] for the `incircle`
/// determinant, whose 3×3 minors give it a larger coefficient.
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
/// The constant bounds the relative rounding error accumulated when an
/// expansion is summed with `estimate()`. Shewchuk derives it as
/// `(3 + 8ε) · ε` and uses it as a coefficient on `|det|` in the guard of
/// the final adaptive stage.
///
/// It is a constant of its own. `ccwerrboundA` is `(3 + 16ε) · ε`, with the
/// same leading coefficient and a different second-order one.
///
/// Formula: `(3 + 8ε) · ε` where `ε = 2^-53` (unit roundoff).
///
/// Source: Shewchuk, `predicates.c`, variable `resulterrbound`.
pub(crate) const RESULT_ERR_BOUND: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;

// ===========================================================================
// Error-bound constants for the degree-2 sign predicates
// ===========================================================================

/// Fast-filter error bound for [`crate::dot_sign`].
///
/// The value is that of [`CCW_ERR_BOUND_A`], because the filter evaluates
/// the same expression tree. `dot_sign` computes `ux·vx + uy·vy` with
/// `u = a − p` and `v = b − p`, evaluated as `ux·vx − (−uy)·vy`. The
/// difference `−uy` is computed as `fl(p.y − a.y)`, which equals
/// `−fl(a.y − p.y)` bit for bit because round-to-nearest is symmetric. The
/// values the filter forms are therefore exactly those of the `orient2d`
/// filter `fl(fl(acx·bcy) − fl(acy·bcx))` on the four rounded differences
/// `acx = ux`, `bcy = vx`, `acy = −uy`, `bcx = vy`, and its magnitude is
/// that of `orient2d`, term for term. The error analysis of that filter
/// (Shewchuk 1997, §4.3; Bartels, Fisikopoulos & Weiser 2023, Theorem 1(2))
/// depends only on the shape of the tree (two rounded differences, their
/// rounded products, one rounded combination) rather than on the input
/// values the differences are taken of, so it applies unchanged. Each
/// contribution matches its `orient2d` counterpart.
///
/// - The differences, one rounding each, `(1 + ε)`, are identical.
/// - The products of two differences, `(1 + ε)³`, are identical.
/// - The final combination, one rounding, is identical, since `fl(x − y)`
///   and `fl(x + y)` obey the same model and `|x ± y| ≤ |x| + |y|`.
/// - The magnitude `(|detleft| + U_N) + (|detright| + U_N)` and the final
///   `+ U_S` are identical.
///
/// Formula: `(3 + 16ε) · ε` where `ε = 2^-53` (unit roundoff).
pub(crate) const DOT_ERR_BOUND_A: f64 = CCW_ERR_BOUND_A;

/// Fast-filter relative error bound for [`crate::compare_distance`].
///
/// `compare_distance` evaluates `Q = (ux² + uy²) − (vx² + vy²)`, with
/// `u = q1 − p` and `v = q2 − p`, as `det = fl(s1 − s2)`, where
/// `s1 = fl(fl(ux²) + fl(uy²))` and `s2` likewise. It compares `|det|` with
/// `fl(fl(A · mag) + K)`, where `mag = fl(s1 + s2)`, `A` is this constant
/// and `K` is [`CMP_DIST_ERR_BOUND_ABS`]. This expression tree has four
/// products and three combinations rather than the two and one of
/// `orient2d`, so the bound is derived here rather than borrowed.
///
/// # Derivation
///
/// Every operation on finite values is modeled by gradual-underflow
/// IEEE 754, `fl(x ± y) = (x ± y)(1 + δ)` and
/// `fl(x·y) = x·y·(1 + δ) + η`, with `|δ| ≤ ε` and `|η| ≤ 2^-1075`.
/// Addition and subtraction carry no `η`, because a subnormal sum of two
/// floats is exact. Write `e3 = (1 + ε)³ − 1`, `e4 = (1 + ε)⁴ − 1`,
/// `u = 2^-1074`, and `U`, `V`, `S1 = Ux² + Uy²`, `S2`, `Q = S1 − S2` for
/// the exact values.
///
/// 1. A square, `fl(ux²) = Ux²(1 + δ₁)²(1 + δ) + η`, lies within
///    `e3·Ux² + u/2` of `Ux²`.
/// 2. A sum of two squares, both non-negative, has
///    `|s1 − S1| ≤ e4·S1 + (1 + ε)u`.
/// 3. `det = (s1 − s2)(1 + δ₀)`, so
///    `|det − Q| ≤ e4(S1 + S2) + 2(1 + ε)u + ε|det|/(1 − ε)`.
/// 4. `mag = fl(s1 + s2) ≥ (s1 + s2)(1 − ε)`, and by (2)
///    `S1 + S2 ≤ (mag/(1 − ε) + 2(1 + ε)u)/(1 − e4)`.
/// 5. The sign of `det` is the sign of `Q` when `|det − Q| < |det|`, which
///    by (3) and (4) holds when
///    `|det| > e4/((1 − e4)(1 − 2ε)) · mag + c·u`, with
///    `c = 2(1 + ε)(1 − ε)/((1 − e4)(1 − 2ε))`, just over 2.
/// 6. The computed bound satisfies
///    `fl(fl(A·mag) + K) ≥ (A·mag·(1 − ε) − u/2 + K)(1 − ε)`, which
///    dominates the right side of (5) when
///    `A(1 − ε)² ≥ e4/((1 − e4)(1 − 2ε))` and `(K − u/2)(1 − ε) ≥ c·u`.
///
/// `A = (4 + 48ε)ε` satisfies the first condition with about `10ε²` to
/// spare (the least `(4 + kε)ε` that does is `k = 39`), and `K = 4u`
/// satisfies the second, which needs just over `2.5u`. Both inequalities
/// are checked in exact rational arithmetic by `tests/degree2_bounds.rs`.
///
/// **Overflow.** If any value overflows, `s1`, `s2` or `mag` is infinite, the
/// bound is infinite or `det` is NaN, and the strict comparison fails. Such
/// input is passed on and never certified.
///
/// Formula: `(4 + 48ε) · ε` where `ε = 2^-53` (unit roundoff).
pub(crate) const CMP_DIST_ERR_BOUND_A: f64 = (4.0 + 48.0 * EPSILON) * EPSILON;

/// Fast-filter absolute error bound for [`crate::compare_distance`],
/// `4 · 2^-1074`.
///
/// It covers the absolute errors of the four squares that can land among
/// the subnormals, and it keeps the bound non-zero however small the input,
/// so an *exact* zero is never certified. [`CMP_DIST_ERR_BOUND_A`] carries the
/// derivation.
pub(crate) const CMP_DIST_ERR_BOUND_ABS: f64 = 4.0 * 5e-324;

// ===========================================================================
// Error-bound constants for compare_point_distance and compare_segment_distance
// ===========================================================================

/// Fast-filter error bound for [`crate::compare_point_distance`].
///
/// The sign of `det = fl((ux² + uy²) − d²)` is certified when
/// `|det| > PT_DIST_ERR_BOUND_A · mag + u_S`, where `mag` is the same
/// expression tree on magnitudes with `u_N` (the smallest positive normal,
/// `2^-1022`) added beside each product,
/// `mag = ((ux² + u_N) + (uy² + u_N)) + (d² + u_N)`, every term computed.
/// The placement of `u_N` is the underflow protection of Bartels,
/// Fisikopoulos & Weiser (2023). A product's absolute rounding error is at
/// most `2^-1075 = ε·u_N`, so beside each product `mag` carries a term that
/// dominates that error relatively, and the certificate holds for every
/// finite input with no underflow assumption.
///
/// # Derivation
///
/// Every operation on finite values is modeled by gradual-underflow
/// IEEE 754, `fl(x ± y) = (x ± y)(1 + δ)` and `fl(x·y) = x·y(1 + δ) + η`,
/// with `|δ| ≤ ε = 2^-53` and `|η| ≤ 2^-1075 = ε·u_N`, addition carrying
/// no `η`. Write `γk = kε/(1 − kε)` in Higham's notation, in which
/// `(1 + θj)(1 + θk)` is a `(1 + θ_{j+k})` and
/// `γj + γk + γj·γk ≤ γ_{j+k}`, and write `Ux`, `Uy`, `L = Ux² + Uy²`,
/// `D2 = d²`, `Q = L − D2` for the exact values.
///
/// 1. A computed difference is `Ux(1 + θ1)`, and its square
///    `x2 = Ux²(1 + θ3) + η` lies within `γ3·Ux² + ε·u_N` of `Ux²`.
/// 2. `s = fl(x2 + y2)` satisfies `|s − L| ≤ γ4·L + 2(1 + ε)ε·u_N`.
/// 3. `dd = fl(d·d)` satisfies `|dd − D2| ≤ γ1·D2 + ε·u_N`.
/// 4. `det = (s − dd)(1 + θ1)`, so
///    `|det − Q| ≤ γ1|det| + γ5·L + γ2·D2 + 4ε·u_N`, using
///    `(1 + γ1)γ4 ≤ γ5`, `(1 + γ1)γ1 ≤ γ2` and `(1 + γ1)(3 + 2ε)ε ≤ 4ε`.
/// 5. Each `mag` term is at least `(value·(1 − γ3) + u_N(1 − ε))`, and the
///    two outer additions cost `(1 − ε)²`, so
///    `mag ≥ (1 − γ5)(L + D2 + 2u_N)`, three `u_N(1 − ε)` dominating
///    `2u_N`.
/// 6. The right side of (4) is at most `γ5·(L + D2 + 2u_N)` beyond the
///    `γ1|det|` term, since `γ2 ≤ γ5` and `4ε ≤ 2γ5`. With (5), the sign of
///    `det` is the sign of `Q` when `|det| > γ5/((1 − γ5)(1 − γ1)) · mag`.
/// 7. The computed bound satisfies
///    `fl(fl(A·mag) + u_S) ≥ (A·mag(1 − ε) − ε·u_N + u_S)(1 − ε) ≥
///    A·mag(1 − ε)²`, because `u_S = 2^-1074 ≥ ε·u_N`. This is the purpose
///    of the `u_S` term, besides keeping the bound positive so that an exact
///    zero is never certified. The certificate therefore holds when
///    `A(1 − ε)² ≥ γ5/((1 − γ5)(1 − γ1))`.
///
/// `A = (5 + 72ε)ε` satisfies the condition of (7), and the least
/// `(5 + kε)ε` that does has `k = 66`. Every inequality above that is a
/// statement about rationals is checked in exact rational arithmetic by
/// `tests/distance_bounds.rs`, namely the three absorptions in (4), the two
/// in (6), the factor in (5), and the closing condition of (7).
///
/// **Overflow.** If any value overflows, `mag` is infinite or `det` is NaN, and
/// the strict comparison fails. Such input is passed on and never
/// certified.
///
/// Formula: `(5 + 72ε)ε` where `ε = 2^-53` (unit roundoff).
pub(crate) const PT_DIST_ERR_BOUND_A: f64 = (5.0 + 72.0 * EPSILON) * EPSILON;

/// Fast-filter error bound for the interior case of
/// [`crate::compare_segment_distance`].
///
/// The sign of `det = fl(cross² − d²·len2)` is certified when
/// `|det| > SEG_DIST_ERR_BOUND_A · mag + u_S`, with
///
/// ```text
/// cross_mag = (|ux·vy| + u_N) + (|uy·vx| + u_N)
/// len_mag   = (ux² + u_N) + (uy² + u_N)
/// mag       = (cross_mag² + u_N) + ((d² + u_N)·len_mag + u_N)
/// ```
///
/// every term computed, in the association order of the determinant. The
/// placement of `u_N` follows [`PT_DIST_ERR_BOUND_A`], and the squares need
/// no `abs`.
///
/// # Derivation
///
/// The notation is that of [`PT_DIST_ERR_BOUND_A`], with exact values
/// `U = B − A`, `V = P − A`, `X = Ux·Vy − Uy·Vx`,
/// `P₀ = |Ux·Vy| + |Uy·Vx|`, `L = Ux² + Uy²`, `D2 = d²`, `Q = X² − D2·L`,
/// and `W = P₀ + 2u_N`.
///
/// 1. A computed product of two computed differences lies within
///    `γ3·|exact| + ε·u_N` of its exact monomial, so
///    `|cross − X| ≤ γ4·P₀ + 3ε·u_N ≤ γ5·W`, and
///    `|cross| ≤ W(1 + γ5)` (`|X| ≤ P₀ ≤ W`).
/// 2. `cross_mag ≥ (1 − γ5)·W`, each term being at least
///    `|monomial|(1 − γ3) + u_N(1 − ε)` before the outer roundings.
/// 3. `|fl(cross²) − X²| ≤ (|cross| + |X|)·|cross − X| + γ1·cross² + ε·u_N
///    ≤ [(2 + γ5)γ5 + γ1(1 + γ5)²]·W² + ε·u_N ≤ γ12·W² + ε·u_N`.
/// 4. `ma = fl(fl(cross_mag²) + u_N) ≥ (1 − γ2)·cross_mag²` and
///    `ma ≥ u_N(1 − γ2)`, so with (2),
///    `|fl(cross²) − X²| ≤ C_a·ma` with
///    `C_a = γ12/((1 − γ5)²(1 − γ2)) + ε/(1 − γ2)`.
/// 5. On the other side, `|fl(x2 + y2) − L| ≤ γ4·L + 3ε·u_N` and
///    `|dd − D2| ≤ γ1·D2 + ε·u_N` give, through the product and its
///    rounding, `|t − D2·L| ≤ γ7·D2·L + 3ε·u_N·L + 5ε·u_N·D2 + 2ε·u_N`
///    for `t = fl(dd·len2)`.
/// 6. `mb = fl(fl((dd + u_N)·len_mag) + u_N)
///    ≥ (1 − γ9)(D2·L + 2u_N·D2 + u_N·L) + u_N(1 − γ2)`, from
///    `fl(dd + u_N) ≥ (1 − γ2)D2 + u_N(1 − ε)²` and
///    `len_mag ≥ (1 − γ5)L + 2u_N(1 − γ3)`. Each term of (5) is then
///    relative to a term of `mb`, giving
///    `|t − D2·L| ≤ C_b·mb` with
///    `C_b = (γ7 + 7.5ε)/(1 − γ9) + 2ε/(1 − γ2)`.
/// 7. `det = (fl(cross²) − t)(1 + θ1)` and `mag ≥ (ma + mb)(1 − ε)`, so the
///    sign of `det` is the sign of `Q` when
///    `|det| > (1 + γ1)·max(C_a, C_b) / ((1 − ε)(1 − γ1)) · mag`. The
///    computed bound dominates as in (7) of [`PT_DIST_ERR_BOUND_A`], so it
///    is enough that `A(1 − ε)²` is at least that coefficient.
///
/// `A = (17 + 32ε)ε` satisfies the condition of (7). A constant with base
/// 16 cannot, the requirement being just over `16.5ε`, and `k = 0` already
/// suffices at base 17. Every inequality above that is a statement about
/// rationals is checked in exact rational arithmetic by
/// `tests/distance_bounds.rs`.
///
/// **Overflow.** Every `mag` term is non-negative, so an overflow anywhere makes
/// `mag` infinite or `det` NaN. The strict comparison then fails, and the
/// input is passed on and never certified.
///
/// Formula: `(17 + 32ε)ε` where `ε = 2^-53` (unit roundoff).
pub(crate) const SEG_DIST_ERR_BOUND_A: f64 = (17.0 + 32.0 * EPSILON) * EPSILON;

// ===========================================================================
// Error-bound constants for orient3d: Shewchuk predicates.c `o3derrboundA/B/C`
// ===========================================================================

/// Stage 1 fast-path error bound for `orient3d`.
///
/// When `|det| ≥ O3D_ERR_BOUND_A · permanent`, the sign of `det` is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundA`.
/// Formula: (7 + 56ε)ε where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_A: f64 = (7.0 + 56.0 * EPSILON) * EPSILON;

/// Stage 2 adaptive-path error bound for `orient3d`.
///
/// When `|estimate(fin)| ≥ O3D_ERR_BOUND_B · permanent` after Stage 2, the
/// sign of the estimate is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundB`.
/// Formula: (3 + 28ε)ε where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_B: f64 = (3.0 + 28.0 * EPSILON) * EPSILON;

/// Stage 2 intermediate re-test error bound for `orient3d`.
///
/// Together with `RESULT_ERR_BOUND · |det|`, it governs the re-test after
/// the first-order tail correction, before the computation commits to
/// Stage 3.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `o3derrboundC`.
/// Formula: (26 + 288ε)ε² where ε = 2^-53.
pub(crate) const O3D_ERR_BOUND_C: f64 = (26.0 + 288.0 * EPSILON) * EPSILON * EPSILON;

// ===========================================================================
// Error-bound constants for insphere: Shewchuk predicates.c `isperrboundA/B/C`
// ===========================================================================

/// Stage 1 fast-path error bound for `insphere`.
///
/// When `|det| ≥ ISP_ERR_BOUND_A · permanent`, the sign of `det` is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `isperrboundA`.
/// Formula: (16 + 224ε)ε where ε = 2^-53.
pub(crate) const ISP_ERR_BOUND_A: f64 = (16.0 + 224.0 * EPSILON) * EPSILON;

/// Stage 2 adaptive-path error bound for `insphere`.
///
/// When `|estimate(fin)| ≥ ISP_ERR_BOUND_B · permanent` after Stage 2, the
/// sign of the estimate is correct.
///
/// Source: Shewchuk (1997) `predicates.c`, variable `isperrboundB`.
/// Formula: (5 + 72ε)ε where ε = 2^-53.
pub(crate) const ISP_ERR_BOUND_B: f64 = (5.0 + 72.0 * EPSILON) * EPSILON;

/// Stage 2 intermediate re-test error bound for `insphere`.
///
/// In Shewchuk's code, together with `RESULT_ERR_BOUND · |det|`, it governs
/// the re-test after the first-order tail correction, before the
/// computation commits to Stage 3.
///
/// This crate does not use it. `insphere` has no Stage 3, and input that its
/// Stage 2 cannot certify goes straight to the exact integer stage, because
/// a sound first-order correction would have to include the lift terms. The
/// constant remains so that the table of Shewchuk's error bounds is
/// complete, with every entry pinned by
/// `err_bound_constants_have_their_table_values`.
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
/// Returns `(x, y)`, where `x = fl(a + b)` is the ordinary rounded sum and
/// `y` is its rounding error, so that `x + y == a + b` exactly as real
/// numbers.
///
/// ## How it works
///
/// Once `x = fl(a + b)` is computed, the subtraction `bv = x - a` recovers
/// the value `b` would have had if the addition had run in the opposite
/// order, the part of `b` that survived into `x`. The difference `b - bv` is
/// then the part of `b` that rounding kept out of `x`. In the same way,
/// `a - (x - bv)` recovers the part of `a` that was lost. The sum of these
/// two residuals is `y`, the exact rounding error.
///
/// This six-operation sequence works for **any** relative magnitudes of `a` and
/// `b`. When `|a| >= |b|` is guaranteed, [`fast_two_sum`] reaches the same
/// result in four operations.
///
/// ## Exactness
///
/// The result is exact whenever `a + b` does not overflow. Underflow is
/// harmless here. If the rounded sum is subnormal, then `a + b` is a
/// multiple of `2^-1074` below `2^-1022`, which is representable, so the
/// addition was exact and `y` is zero. The module documentation states the
/// module-wide condition and who owes it.
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
/// Returns `(x, y)`, where `x = fl(a + b)` and `y` is the rounding error, so
/// that `x + y == a + b` exactly as real numbers.
///
/// ## Precondition
///
/// `|a| >= |b|` must hold before the call. Debug builds check it with
/// [`debug_assert!`] and panic when it is violated. Release builds omit the
/// check, and a violation silently produces an incorrect `y` that does **not**
/// satisfy the exact-sum property, corrupting any adaptive computation that
/// follows. The calling context always establishes the precondition.
///
/// ## Why the precondition is necessary
///
/// When `|a| < |b|`, the bits of `a` may fall below the rounding point of
/// `b` and be absorbed completely. `x - a` then no longer recovers the part
/// of `b` that survived. It recovers a quantity that mixes contributions
/// from both operands rather than decomposing cleanly into the separate
/// rounding errors of `a` and `b`. The exact-sum identity `x + y == a + b`
/// fails in this case.
///
/// ## Exactness
///
/// The condition is that of [`two_sum`]. The result is exact whenever
/// `a + b` does **not** overflow, and underflow is harmless for the same reason.
/// This condition is in addition to the `|a| >= |b|` precondition above,
/// which concerns the algorithm rather than the exponent range.
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
/// The split uses Veltkamp's constant `2^27 + 1 = 134_217_729`. The rounded
/// product `c = a·(2^27 + 1)` and the difference `c - a` isolate the high
/// part of the significand of `a` in `hi`, and `lo = a - hi` recovers the
/// rest. By Shewchuk's Theorem 17, each half has at most 26 significant bits,
/// the sign of `lo` carrying the remaining bit of the 53. A product of two
/// halves therefore has at most 52 significant bits, so it is computed
/// without rounding error.
///
/// Precondition. `a` must be finite. Non-finite input produces unspecified
/// output, so the caller establishes finiteness before the call.
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
/// Returns `(x, y)`, where `x = fl(a * b)` is the ordinary rounded product
/// and `y` is its rounding error, so that `x + y == a * b` exactly as real
/// numbers.
///
/// ## Why Veltkamp-Dekker splitting works
///
/// Veltkamp splitting (the private `split` helper) divides `a` into
/// `(a_hi, a_lo)` and `b` into `(b_hi, b_lo)`, and each half occupies at
/// most 27 significant bits. A product of two 27-bit values has at most 54
/// significant bits, at the edge of the 53-bit significand of an `f64`, but
/// the leading bit of `a_hi * b_hi` cancels exactly against `x`, the rounded
/// full product, leaving a residual that an `f64` can express. The cross
/// products `a_hi * b_lo` and `a_lo * b_hi` each fit in 54 bits, so no
/// precision is lost in computing them. The sum of all four partial products
/// minus `x` gives `y` exactly.
///
/// ## Splitting rather than FMA
///
/// An FMA-based `TwoProduct` (`x = a * b; y = fma(a, b, -x)`) is simpler and
/// faster but depends on hardware FMA support. Veltkamp-Dekker splitting is
/// correct on every IEEE 754-compliant platform, with no dependence on a
/// hardware feature and no run-time capability check.
///
/// ## Exactness
///
/// The result is exact when two conditions hold.
///
/// 1. **No overflow.** The split forms `(2^27 + 1) * a`, so `|a|` and `|b|` must
///    stay below about `2^995`, and the product `a * b` itself must be
///    finite.
/// 2. A product that is not too small. Write `a = m_a * 2^(e_a - 52)` and
///    `b = m_b * 2^(e_b - 52)` with integer significands. The exact product
///    `a * b` is then an integer multiple of `2^(e_a + e_b - 104)`, as is
///    every partial product the algorithm forms. These are representable in
///    binary64 exactly when that granule is at least `2^-1074`, that is,
///    when `e_a + e_b >= -970`, so the magnitude of the product must be
///    roughly `2^-970` (about `1e-292`) or more. Below it, `y` loses bits and
///    `x + y` is no longer `a * b`, with nothing in the return value to say
///    so.
///
/// Under the same condition each partial product also fits in 53 bits,
/// being a multiple of that granule and bounded by `|a * b|`, so it needs at
/// most 52 bits.
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

/// Computes `two_product(a, b)` for a `b` already split into
/// `(b_hi, b_lo)`.
///
/// Returns `(x, y)`, where `x = fl(a * b)` and `x + y == a * b` exactly as
/// real numbers. The result equals `two_product(a, b)`, but `b` is not split
/// again on every call. `scale_expansion` relies on this to spread the cost
/// of splitting `b` across every component of the expansion.
///
/// `b_hi` and `b_lo` must satisfy `b_hi + b_lo == b` exactly, with the
/// properties that `split(b)` guarantees.
///
/// This is the `Two_Product_Presplit` macro of Shewchuk's predicates.c.
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
/// `h` must have length at least `e.len() + 1`, which a `debug_assert!`
/// verifies in debug builds. The first `len` elements of `h` hold the
/// result, and the elements beyond `len` are unspecified.
///
/// ## Exactness
///
/// The function is built from [`two_sum`], so it is exact whenever no
/// addition along the way overflows. The running sum passes through every
/// component of `e`, so the total governs overflow rather than the
/// individual terms. The module documentation states the module-wide
/// condition.
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
/// `h` must have length at least `2 * e.len()`, which a `debug_assert!`
/// verifies in debug builds.
///
/// ## Precondition
///
/// `e` must be a valid non-overlapping expansion. Every operation in this
/// module maintains the non-overlapping property, and a caller who builds an
/// expansion by hand is responsible for the invariant.
///
/// ## Exactness
///
/// The function is built from [`two_product`] and [`two_sum`], so it needs
/// the conditions of both, no overflow and every product `e[i] * b` above
/// about `2^-970`. The smallest component of a non-overlapping expansion can
/// lie many orders of magnitude below its largest, which makes this the
/// operation where an unwary caller is most likely to lose exactness
/// silently, the leading term looking comfortably scaled while the trailing
/// ones are not. The module documentation states the module-wide
/// condition.
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
    // b is split once here, and every per-component product reuses
    // (b_hi, b_lo).
    let (b_hi, b_lo) = split(b);
    let mut h_len = 0usize;

    // The first component produces a single (q, hh) pair, which starts the
    // carry.
    let (mut q, hh) = two_product_presplit(e[0], b, b_hi, b_lo);
    if hh != 0.0 {
        h[h_len] = hh;
        h_len += 1;
    }

    for &e_i in &e[1..] {
        let (product1, product0) = two_product_presplit(e_i, b, b_hi, b_lo);

        // The low term of the product joins the running carry.
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h[h_len] = hh;
            h_len += 1;
        }

        // The high term of the product merges with the updated carry.
        // Shewchuk uses Fast_Two_Sum here, which is exact only when
        // |product1| >= |sum|. That ordering does hold for valid nonoverlapping
        // expansions, but the argument is not short, whereas two_sum is exact
        // with no ordering condition at all, for two more operations.
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
/// The algorithm takes each component of `f` in turn and grows the
/// intermediate result held in `h` by that component, through repeated
/// application of the `two_sum` kernel. The passes maintain the
/// non-overlapping invariant throughout, so the intermediate result is
/// always a valid expansion. A final zero-elimination pass compacts the
/// output.
///
/// ## Caller-owned buffer contract
///
/// `h` must have length at least `e.len() + f.len()`, which a
/// `debug_assert!` verifies in debug builds.
///
/// ## Exactness
///
/// The function is built from [`two_sum`], so it is exact whenever no
/// addition along the way overflows. Both inputs must already be valid
/// non-overlapping expansions, a property this function preserves but
/// cannot establish. The module documentation states the module-wide
/// condition.
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

    // When one operand is empty, the other is copied with zeros eliminated.
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

    // First pass: f[0] grows through all of e, writing every intermediate
    // value, zeros included, into h[0..e.len()] and the final carry into
    // h[e.len()]. The zeros must be written because the second pass reads
    // back h[findex..=hlast].
    let mut q = f[0];
    for (hindex, &e_i) in e.iter().enumerate() {
        let (q_new, h_i) = two_sum(q, e_i);
        h[hindex] = h_i; // Written even when zero, for the second pass.
        q = q_new;
    }
    let mut hlast = e.len();
    h[hlast] = q; // The final carry of growing f[0].

    // Second pass: each remaining f[i] grows through h[findex..=hlast] in
    // place. Slots h[0..findex] are already final, their components being too
    // small to interact with f[i] without violating the non-overlapping
    // invariant.
    for (fi, &f_i) in f[1..].iter().enumerate() {
        let findex = fi + 1;
        q = f_i;
        for h_slot in h.iter_mut().take(hlast + 1).skip(findex) {
            let h_now = *h_slot; // Read before the write to the same slot.
            let (q_new, h_h) = two_sum(q, h_now);
            *h_slot = h_h;
            q = q_new;
        }
        hlast += 1;
        h[hlast] = q;
    }

    // The zero-elimination pass compacts h[0..=hlast], removing exact zeros.
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
// test-lint-posture: see the lint table in Cargo.toml
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

    /// A case in which naive f64 addition loses the low bits of b.
    ///
    /// a = 2^53 (the largest f64 integer, i.e. 9007199254740992)
    /// b = 1.0
    ///
    /// fl(a + b) = a, because 1.0 is below the ULP of a, which is 2.0.
    /// two_sum must recover b = 1.0 in the error term y.
    #[test]
    fn two_sum_recovers_exact_sum() {
        let a = (1_u64 << 53) as f64; // 2^53
        let b = 1.0_f64;

        // Naive addition loses the bit.
        assert_eq!(a + b, a, "precondition: naive addition absorbs b");

        let (x, y) = two_sum(a, b);
        // x is the rounded sum, equal to a after rounding.
        assert_eq!(x, a);
        // y recovers the lost bit.
        assert_eq!(y, b);
        // The defining property is x + y == a + b exactly. Here
        // x + y = 2^53 + 1.0, which no single f64 represents, so the test
        // checks the components rather than the f64 sum x + y.
        assert_eq!(x + y, a); // loses y again, showing y is below the ULP of x
        // The exact sum a + b = 2^53 + 1 is what x and y must encode.
        let exact: i64 = (a as i64) + (b as i64);
        assert_eq!((x as i64) + (y as i64), exact);
    }

    /// A case in which both a and b lie in range for exact recovery.
    ///
    /// a = 1.0, b = 2^-53 (half of machine epsilon, below the ULP of 1.0)
    /// fl(a + b) = 1.0 (b is rounded away)
    /// two_sum recovers b in y.
    #[test]
    fn two_sum_small_b_is_recovered() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0; // 2^-53, half ULP of 1.0

        assert_eq!(a + b, a, "precondition: naive addition loses b");

        let (x, y) = two_sum(a, b);
        assert_eq!(x, 1.0);
        assert_eq!(y, b, "error term must recover b exactly");
    }

    /// two_sum is correct when b > a, having no magnitude precondition.
    #[test]
    fn two_sum_no_precondition_on_magnitude() {
        // b >> a, and the result is still exact.
        let a = 1.0_f64;
        let b = (1_u64 << 53) as f64;
        let (x, y) = two_sum(a, b);
        // x = fl(a + b) = b, since a is below the ULP of b.
        assert_eq!(x, b);
        assert_eq!(y, a);
        assert_eq!((x as i64) + (y as i64), (a as i64) + (b as i64));
    }

    // ===========================================================================
    // fast_two_sum
    // ===========================================================================

    /// The scenario of two_sum_recovers_exact_sum, with |a| >= |b|.
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

    /// fast_two_sum with a = 1.0 and b = 2^-53, as in the two_sum companion.
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

    /// A precondition violation panics in debug builds.
    ///
    /// fast_two_sum(1.0, 2.0) violates |a| >= |b|, since 1.0 < 2.0, so the
    /// debug_assert! must fire.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "fast_two_sum precondition violated")]
    fn fast_two_sum_panics_on_precondition_violation() {
        let _ = fast_two_sum(1.0, 2.0);
    }

    // ===========================================================================
    // split (private, tested directly through module visibility)
    // ===========================================================================

    /// Split satisfies hi + lo == a exactly for a representative value, one
    /// with all 53 mantissa bits set.
    #[test]
    fn split_exact_reconstruction() {
        // 2^53 - 1 has all 53 significand bits set, the largest odd integer
        // in f64.
        let a = ((1_u64 << 53) - 1) as f64;
        let (hi, lo) = split(a);
        // hi + lo equals a exactly, the central guarantee of Split.
        assert_eq!(hi + lo, a);
        // hi and lo do not overlap, so |hi| >= |lo|.
        assert!(hi.abs() >= lo.abs());
    }

    // ===========================================================================
    // two_product
    // ===========================================================================

    /// A case in which fl(a * b) loses the low-order bits.
    ///
    /// a = 2^27 + 1  (fits exactly in the high split-half of a 53-bit f64)
    /// b = 2^27 + 1
    ///
    /// a * b = 2^54 + 2^28 + 1
    ///
    /// 2^54 has more than 53 significant bits in this product (the leading bit
    /// is at position 54, with further bits at positions 28 and 0), so
    /// fl(a * b) rounds away the low bits, which two_product must recover.
    #[test]
    fn two_product_recovers_exact_product() {
        let a = (1_u64 << 27) as f64 + 1.0; // 2^27 + 1 = 134_217_729
        let b = a; // the same value

        // The true product is (2^27 + 1)^2 = 2^54 + 2*2^27 + 1, which is
        // 2^54 + 2^28 + 1. It requires 55 bits, so fl(a*b) loses at least the
        // low-order bit.
        let naive = a * b;

        let (x, y) = two_product(a, b);
        assert_eq!(x, naive, "x must equal the rounded product");

        // x + y equals the exact product, checked in i64 arithmetic.
        // (2^27 + 1)^2 = 2^54 + 2^28 + 1 = 18_014_398_777_917_441
        let exact: i64 = 18_014_398_777_917_441;
        let reconstructed: i64 = (x as i64) + (y as i64);
        assert_eq!(reconstructed, exact, "x + y must equal exact product");

        // The naive product differs from the exact value, which confirms that
        // the case demonstrates a loss of precision.
        assert_ne!(naive as i64, exact, "naive f64 product must lose bits");
    }

    /// two_product is exact for input whose product fits in 53 bits.
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

    /// The basic case, in which grow_expansion(&[1.0], 2.0) sums to 3.0.
    ///
    /// 1.0 + 2.0 = 3.0 exactly in f64, so the result is the single component
    /// [3.0].
    #[test]
    fn grow_expansion_simple_sum() {
        let e = [1.0_f64];
        let mut h = [0.0_f64; 2];
        let len = grow_expansion(&e, 2.0, &mut h);
        assert!(len >= 1, "must produce at least one component");
        let sum: f64 = h[..len].iter().sum();
        assert_eq!(sum, 3.0_f64, "expansion sum must equal 1.0 + 2.0");
    }

    /// A case in which the rounding error term h_i is non-zero.
    ///
    /// In grow_expansion(&[1e-16], 1.0), 1e-16 < ULP(1.0)/2 ≈ 1.11e-16, so
    /// fl(1.0 + 1e-16) = 1.0. The low component is lost to rounding and must
    /// appear as h_i in the error slot, giving the result [1e-16, 1.0].
    #[test]
    fn grow_expansion_error_term_is_nonzero() {
        let e = [1e-16_f64];
        let mut h = [0.0_f64; 2];
        let len = grow_expansion(&e, 1.0, &mut h);
        assert_eq!(len, 2, "must produce 2 components (error + carry)");
        // The low component captures what rounding lost.
        assert_eq!(h[0], 1e-16_f64, "low component must recover 1e-16");
        // The high component is the rounded sum.
        assert_eq!(h[1], 1.0_f64, "high component must be 1.0");
        // Their exact sum as reals is 1.0 + 1e-16. Naive f64 addition of
        // h[0] + h[1] rounds back to 1.0, which confirms that h[0] lies below
        // the ULP of h[1] and carries precision a single f64 cannot.
        assert_eq!(
            h[0] + h[1],
            1.0_f64,
            "naive sum rounds away the low component"
        );
    }

    // ===========================================================================
    // scale_expansion
    // ===========================================================================

    /// scale_expansion(&[1.0, 1e-16], 3.0) encodes 3.0 + 3*1e-16 exactly.
    ///
    /// Because `1e-16_f64` is slightly below 10^-16 (the nearest representable
    /// double to 10^-16 is ≈ 9.9999...e-17), `fl(3.0 * 1e-16)` exceeds
    /// `ULP(3.0) / 2 ≈ 2.22e-16`, so `fl(3.0 + 3*1e-16)` rounds up to
    /// `3.0 + 2^-51 = 3.0000000000000004`. The expansion needs at least two
    /// components to encode the full exact result, and summed in f64 those
    /// components collapse back to that nearest representable double.
    #[test]
    fn scale_expansion_exact_product_sum() {
        let e = [1.0_f64, 1e-16_f64];
        let mut h = [0.0_f64; 4];
        let len = scale_expansion(&e, 3.0, &mut h);
        assert_eq!(len, 2, "must produce 2 components");
        // Every component is nonzero, by the zero-elimination invariant.
        for (i, &v) in h.iter().enumerate().take(len) {
            assert_ne!(
                v, 0.0_f64,
                "component {i} must be nonzero after zero elimination"
            );
        }
        // The f64 sum of the live components must equal the nearest
        // representable double to sum(e) * b = (1.0 + 1e-16) * 3.0. This
        // confirms that the expansion encodes the full exact product. The low
        // component captures the sub-ULP residual that would otherwise be lost,
        // and added to the high component it recovers the exact result to
        // single-f64 precision.
        let sum: f64 = h[..len].iter().sum();
        assert_eq!(
            sum,
            3.0_f64 + 3e-16_f64,
            "sum of components must equal fl(3.0 + 3*1e-16) = {}",
            3.0_f64 + 3e-16_f64
        );
    }

    /// scale_expansion of an empty slice returns 0 without writing to h.
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

    /// expansion_sum encodes the *exact* sum of two multi-component expansions.
    ///
    /// e = [1e-16, 1.0], f = [2.0]
    /// Exact sum: 1e-16 + 1.0 + 2.0 = 3.0 + 1e-16
    ///
    /// The result is [1e-16, 3.0], two components that together represent the
    /// exact value 3.0 + 1e-16, which no single f64 represents.
    #[test]
    fn expansion_sum_exact_result() {
        let e = [1e-16_f64, 1.0_f64]; // valid non-overlapping expansion
        let f = [2.0_f64];
        let mut h = [0.0_f64; 3];
        let len = expansion_sum(&e, &f, &mut h);
        assert_eq!(len, 2, "must produce 2 components");
        // The low component captures what was lost when 1e-16 was added to 3.0.
        assert_eq!(h[0], 1e-16_f64, "low component must be 1e-16");
        // The high component is the rounded sum 1.0 + 2.0.
        assert_eq!(h[1], 3.0_f64, "high component must be 3.0");
    }

    /// expansion_sum with both inputs empty returns 0.
    #[test]
    fn expansion_sum_both_empty() {
        let e: [f64; 0] = [];
        let f: [f64; 0] = [];
        let mut h = [0.0_f64; 0];
        let len = expansion_sum(&e, &f, &mut h);
        assert_eq!(len, 0);
    }

    /// expansion_sum with an empty f copies e, with zeros eliminated.
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
    /// The expected values are written as bit patterns rather than recomputed
    /// from the formulas. A test that evaluates the same expression as the
    /// constant cannot fail, however wrong the expression is, whereas a bit
    /// pattern derived independently can.
    ///
    /// Each pattern follows from its formula by integer arithmetic. With
    /// `ε = 2^-53`, a first-order constant `(a + bε)ε` is `a·2^-53 + b·2^-106`
    /// and a second-order constant `(a + bε)ε²` is `a·2^-106 + b·2^-159`. In
    /// all seventeen of these, the second term is a whole number of units in the
    /// last place of the first, so the sum is exactly representable and no
    /// rounding is involved. The absolute bound is four subnormal units.
    ///
    /// For example, `CCW_ERR_BOUND_A = 3·2^-53 + 16·2^-106`. The first term is
    /// `1.5·2^-52`, whose unit in the last place is `2^-104`, and the second is
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
        check!(DOT_ERR_BOUND_A, 0x3cb8_0000_0000_0004_u64); // (3 + 16ε)ε
        check!(CMP_DIST_ERR_BOUND_A, 0x3cc0_0000_0000_0006_u64); // (4 + 48ε)ε
        check!(CMP_DIST_ERR_BOUND_ABS, 0x0000_0000_0000_0004_u64); // 4 · 2^-1074
        check!(PT_DIST_ERR_BOUND_A, 0x3cc4_0000_0000_0009_u64); // (5 + 72ε)ε
        check!(SEG_DIST_ERR_BOUND_A, 0x3ce1_0000_0000_0001_u64); // (17 + 32ε)ε

        // The epsilon convention is pinned separately, the unit roundoff 2^-53
        // rather than `f64::EPSILON` (2^-52). `0x3ca` is `1023 - 53`.
        assert_eq!(EPSILON.to_bits(), 0x3ca0_0000_0000_0000_u64);
    }
}
