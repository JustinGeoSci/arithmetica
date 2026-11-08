// src/exact_int.rs

//! Fixed-width exact integer arithmetic for the Stage 4 fallback of the
//! predicates.
//!
//! Every finite `f64` is represented exactly as an integer scaled by
//! `2^1074`. The representation is exact for every finite `f64`, because the
//! minimum positive `f64`, the minimum subnormal, is `2^-1074`, so
//! multiplying by `2^1074` always produces an integer.
//!
//! Each width follows from a worst-case bound on the bits its computation
//! needs.
//!
//! - `orient2d`: 4224 bits = 66 × 64 limbs ([`Orient2dInt`])
//! - `incircle`:  8448 bits = 132 × 64 limbs ([`Incircle2dInt`])
//! - `orient3d`: 6336 bits = 99 × 64 limbs ([`Orient3dInt`])
//! - `insphere`: 10560 bits = 165 × 64 limbs ([`InsphereInt`])
//! - constructed-point storage:  6336 bits = 99 × 64 limbs ([`RationalCoordInt`])
//! - `rational_coords_equal`:   10560 bits = 165 × 64 limbs ([`RationalEqInt`])
//! - `rational_compare_along`:  12672 bits = 198 × 64 limbs ([`RationalOrderInt`])
//! - `orient2d_rational`:        8448 bits = 132 × 64 limbs ([`RationalOrientInt`])
//!
//! # Lemma S: the width of one scaled coordinate
//!
//! Every width proof in this module and in [`crate::rational`] rests on the
//! same bound, so it is stated once here.
//!
//! The largest finite `f64` is `(2^53 − 1) × 2^971`. Scaling by `2^1074`
//! gives `(2^53 − 1) × 2^2045 < 2^2098`, so every finite `f64` `x` satisfies
//!
//! ```text
//! |x × 2^1074| < 2^2098          (2099 signed bits)
//! ```
//!
//! Two corollaries recur throughout.
//!
//! - **S1.** A difference of two scaled coordinates is `< 2^2099`
//!   (2100 signed bits).
//! - **S2.** A 2 × 2 determinant of scaled coordinate differences is
//!   `< 2^(2099+2099) + 2^(2099+2099) = 2^4199` (4200 signed bits).
//!
//! These are fixed-width types rather than arbitrary-precision integers.
//! The width proof of each predicate establishes that its arithmetic never
//! overflows the allocated bits, so the "wrapping" operations produce
//! mathematically exact results for every in-domain input. The word names
//! how Rust expresses carry-propagating limb arithmetic rather than any
//! acceptance of truncation.

use core::ops::{Add, Mul, Neg, Shl, Sub};

// ===========================================================================
// Core type
// ===========================================================================

/// Fixed-width two's-complement signed integer stored in `N` 64-bit limbs,
/// little-endian (limb 0 is the least-significant word).
///
/// # Width invariant
///
/// The caller must choose `N` large enough that no intermediate or final
/// arithmetic result overflows the allocated `N × 64` signed bits. When that
/// invariant holds, [`wrapping_add`](Self::wrapping_add),
/// [`wrapping_neg`](Self::wrapping_neg), [`wrapping_mul`](Self::wrapping_mul),
/// and [`wrapping_shl`](Self::wrapping_shl) all produce mathematically exact
/// results.
///
/// [`fits_in_signed_bits`](Self::fits_in_signed_bits), placed in a
/// `debug_assert!` after a Stage 4 determinant computation, verifies the
/// invariant structurally. A debug build then panics on a width violation
/// rather than wrapping silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WideInt<const N: usize> {
    limbs: [u64; N],
}

// ===========================================================================
// Type aliases
// ===========================================================================

/// 4224-bit signed integer for `orient2d` Stage 4 (`WideInt<66>`).
///
/// Width proof. By Lemma S each scaled input coordinate is `< 2^2098`
/// (2099 signed bits), so by S1 each coordinate difference is `< 2^2099`.
/// Each cross-product of two differences is `< 2^4198`, and the two-term
/// determinant difference is `< 2^4199` (4200 signed bits), below the
/// allocated 4224.
pub(crate) type Orient2dInt = WideInt<66>;

/// 8448-bit signed integer for `incircle` Stage 4 (`WideInt<132>`).
///
/// Width proof. Scaled coordinate differences require at most 2099 bits, and
/// lift terms (sums of two squared differences) at most 4199 bits. Each
/// product of a lift and a 2×2 sub-determinant is below `2^8398`, so the
/// three-term determinant sum is below `2^8400` (8401 signed bits), below the
/// allocated 8448.
pub(crate) type Incircle2dInt = WideInt<132>;

/// 6336-bit signed integer for `orient3d` Stage 4 (`WideInt<99>`).
///
/// Width proof. Each scaled coordinate difference requires at most 2099
/// bits. Each 2×2 cross-product, such as `bdx·cdy − bdy·cdx`, is at most
/// 4199 bits, the product of two 2099-bit values being 4198 bits and the
/// subtraction adding one bit. Each product of a z-difference and a 2×2
/// cross-product is at most 2099 + 4199 = 6298 bits. The three-term
/// determinant sum adds `⌈log₂ 3⌉ = 2` bits, for a total of 6300 bits,
/// below the allocated 6336.
pub(crate) type Orient3dInt = WideInt<99>;

/// 10560-bit signed integer for `insphere` Stage 4 (`WideInt<165>`).
///
/// Width proof. Scaled coordinate differences require at most 2099 bits.
/// Each squared difference is at most 4198 bits, and a lift term (the sum of
/// three squared differences) at most 4200 bits. Each 3×3 minor of
/// coordinate differences is at most 6300 bits, by the analysis of
/// [`Orient3dInt`]. Each product of a lift and a 3×3 minor is at most
/// 4200 + 6300 = 10500 bits. The four-term determinant sum adds
/// `⌈log₂ 4⌉ = 2` bits, for a total of 10502 bits, below the allocated
/// 10560.
pub(crate) type InsphereInt = WideInt<165>;

// ===========================================================================
// Constructed-point (rational) widths
// ===========================================================================

/// 6336-bit signed integer for the three components of a
/// [`RationalCoord2d`](crate::RationalCoord2d) (`WideInt<99>`).
///
/// # Width proof: storage
///
/// A constructed point is the intersection of segment `P1 P2` with segment
/// `Q1 Q2`, all four endpoints being **input** `f64` values, the depth-1
/// property of [`crate::rational`]. In scaled integers,
///
/// ```text
/// den    = (P2x−P1x)(Q2y−Q1y) − (P2y−P1y)(Q2x−Q1x)
/// num_t  = (Q1x−P1x)(Q2y−Q1y) − (Q1y−P1y)(Q2x−Q1x)
/// num_x  = P1x·den + num_t·(P2x−P1x)
/// num_y  = P1y·den + num_t·(P2y−P1y)
/// ```
///
/// - `den` and `num_t` are 2 × 2 determinants of scaled coordinate
///   differences, so by corollary S2 both are `< 2^4199` (4200 signed bits).
/// - `P1x·den < 2^2098 · 2^4199 = 2^6297` and `num_t·(P2x−P1x) < 2^4199 ·
///   2^2099 = 2^6298`. Their sum is `< 2^6299`.
///
/// The two stored invariants are therefore
///
/// ```text
/// |num_x|, |num_y| < 2^6299        (6300 signed bits)
/// 0 < den          < 2^4199        (4200 signed bits)
/// ```
///
/// both below the allocated 6336 bits.
///
/// # Why the numerator bound is not tightened
///
/// A tempting argument holds that a constructed point lies inside both
/// segments' bounding boxes, hence `|X| < 2^2098` by Lemma S and
/// `|num_x| = |X|·den < 2^6297`, two bits narrower. **That argument is false
/// here.** The constructor computes the intersection of the two *lines* through
/// the segments, which for a nearly parallel pair lies arbitrarily far
/// outside either bounding box. Whether the point lies on the segments is
/// the caller's question rather than a precondition of this type, so every
/// downstream proof uses the `2^6299` bound above. The chosen widths all
/// accommodate it. The tighter claim gains nothing, and it would cost a
/// silent wrong sign if a caller ever constructed a line-line crossing.
///
/// Construction establishes both invariants. The fields of
/// [`RationalCoord2d`](crate::RationalCoord2d) are private, and every
/// constructor derives them from finite `f64` input by exactly these
/// formulas, so no caller can present a value that violates them.
pub(crate) type RationalCoordInt = WideInt<99>;

/// 10560-bit signed integer for `rational_coords_equal` (`WideInt<165>`).
///
/// # Width proof
///
/// Equality of `a = (ax/da, ay/da)` and `b = (bx/db, by/db)`, with
/// `da, db > 0`, is tested by cross-multiplication rather than division,
///
/// ```text
/// ax·db − bx·da  ==  0     and     ay·db − by·da  ==  0
/// ```
///
/// By the stored bounds of [`RationalCoordInt`],
/// `|ax·db| < 2^6299 · 2^4199 = 2^10498`, and likewise `|bx·da|`. The
/// difference is `< 2^10499`, that is, **10500 signed bits**, below the
/// allocated 10560 and leaving 60 bits of margin.
pub(crate) type RationalEqInt = WideInt<165>;

/// 12672-bit signed integer for `rational_compare_along` (`WideInt<198>`).
///
/// # Width proof
///
/// The order of `a` and `b` along the direction `d = S2 − S1` is the sign of
/// `(a − b) · d`. Writing `a − b` over the common positive denominator
/// `da · db` and dropping that positive factor gives
///
/// ```text
/// sign( proj(a) − proj(b) )
///   = sign( (ax·db − bx·da)·dx + (ay·db − by·da)·dy )
/// ```
///
/// The segment **origin cancels algebraically**, so `S1` never enters the
/// determinant and only the direction does. The cancellation removes one
/// level of multiplication and the width it would have cost.
///
/// - `ax·db − bx·da` is the `RationalEqInt` quantity above, `< 2^10499`.
/// - `dx` is a scaled coordinate difference, `< 2^2099` by S1.
/// - Each product is `< 2^12598`, and the two-term sum is `< 2^12599`, that
///   is, **12600 signed bits**, below the allocated 12672 and leaving 72 bits of
///   margin.
pub(crate) type RationalOrderInt = WideInt<198>;

/// 8448-bit signed integer for `orient2d_rational` (`WideInt<132>`).
///
/// # Width proof
///
/// This is the mixed-arity case, two `f64` endpoints `Pa`, `Pb` and one
/// rational point `c = (cx/dc, cy/dc)` with `dc > 0`. The determinant
/// `(Pb−Pa) × (c−Pa)` is multiplied through by `dc`, which is positive and
/// so cannot change the sign,
///
/// ```text
/// det·dc = (Pbx−Pax)·(cy − Pay·dc) − (Pby−Pay)·(cx − Pax·dc)
/// ```
///
/// Here the `f64` endpoints are **lifted to the rational's denominator**, as
/// `Pay·dc` and `Pax·dc`, and this is where a transcription error would be
/// silent, since dropping the `dc` factor still compiles and still returns a
/// sign.
///
/// - `Pay·dc < 2^2098 · 2^4199 = 2^6297` and `|cy| < 2^6299`, so each lifted
///   difference is `< 2^6300` (6301 signed bits).
/// - Multiplied by a scaled coordinate difference (`< 2^2099` by S1), each
///   is `< 2^8399`.
/// - The two-term difference is `< 2^8400`, that is, **8401 signed bits**, below
///   the allocated 8448 and leaving 47 bits of margin.
pub(crate) type RationalOrientInt = WideInt<132>;

/// 6400-bit signed integer for rational-to-`f64` rounding (`WideInt<100>`).
///
/// # Width proof
///
/// Rounding asks one question repeatedly, whether the true coordinate
/// `(num / den) · 2^−1074` is less than, equal to, or greater than a
/// candidate midpoint `M · 2^E`. Cross-multiplying by the positive `den` and
/// by `2^1074` gives
///
/// ```text
/// num  ⋛  M · den · 2^(E + 1074)
/// ```
///
/// Write `E′ = E + 1074`. Each side is scaled by `2^max(0, −E′)` so that both
/// are integers, giving `num · 2^max(0,−E′)` against `M · den · 2^max(0,E′)`.
///
/// The `2^1074` scaling keeps this narrow. `num` already carries that
/// factor, and `E′` restores it on the other side, so the two cancel rather
/// than stack. Without it the left side would need a further 1074 bits.
///
/// - `M` is the sum of two `f64` mantissas aligned to a common exponent.
///   Each is `< 2^53`, and at a binade boundary one is shifted up by one
///   place, so `M < 2^53 + 2^54 < 2^55`.
/// - `E` is one less than the smaller of the two exponents. The smallest
///   `f64` exponent is `−1074`, so `E ≥ −1075` and `E′ ≥ −1`. The left side
///   is therefore at worst `num · 2^1`, which is `< 2^6300`.
/// - The largest `E` arises at the overflow midpoint, `E = 970`, so
///   `E′ ≤ 2044` and the right side is `< 2^55 · 2^4199 · 2^2044 = 2^6298`.
///
/// The wider side needs **6301 signed bits**. `WideInt<99>` would already
/// suffice at 6336, but its margin would be 35 bits, the narrowest in the
/// crate. One extra limb costs nothing measurable here, since rounding runs
/// per output vertex rather than per predicate call, and it leaves 99 bits,
/// so an error of a few bits in the bound cannot become a wrong answer.
pub(crate) type RoundingInt = WideInt<100>;

// ===========================================================================
// Constructors and sign predicates
// ===========================================================================

impl<const N: usize> WideInt<N> {
    /// Returns the zero value (all limbs cleared).
    #[inline]
    pub(crate) const fn zero() -> Self {
        Self { limbs: [0; N] }
    }

    /// Constructs a `WideInt<N>` from a single 64-bit unsigned value,
    /// zero-extending into the remaining limbs.
    ///
    /// The result is always non-negative, and [`Neg`] negates it when the
    /// source value represents a negative quantity.
    #[inline]
    pub(crate) fn from_u64(value: u64) -> Self {
        let mut result = Self::zero();
        result.limbs[0] = value;
        result
    }

    /// Returns the value `1`.
    #[inline]
    pub(crate) fn one() -> Self {
        Self::from_u64(1)
    }

    /// Returns `true` if all limbs are zero.
    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.limbs.iter().all(|&x| x == 0)
    }

    /// Sign-extends this value into a wider `WideInt<M>`.
    ///
    /// The three constructed-point predicates each compute in a width derived
    /// from their own proof, while a [`RationalCoord2d`](crate::RationalCoord2d)
    /// stores its components in the narrower [`RationalCoordInt`]. Widening is
    /// how a stored component enters a predicate's arithmetic, and
    /// sign-extension rather than zero-extension makes it exact for negative
    /// values.
    ///
    /// # Precondition
    ///
    /// `M >= N`. Narrowing would discard limbs, and debug builds reject it.
    /// No caller in this crate narrows.
    #[inline]
    pub(crate) fn widen<const M: usize>(&self) -> WideInt<M> {
        debug_assert!(
            M >= N,
            "WideInt::widen called with a narrower target width; limbs would be discarded"
        );
        let fill = if self.is_negative() { u64::MAX } else { 0u64 };
        let mut out = WideInt::<M> { limbs: [fill; M] };
        let n = N.min(M);
        out.limbs[..n].copy_from_slice(&self.limbs[..n]);
        out
    }

    /// Returns the magnitude of `self` (`|self|`).
    ///
    /// # Precondition
    ///
    /// `self` is not the two's-complement minimum `−2^(N×64−1)`, whose
    /// magnitude is not representable. Every width proof in this module keeps
    /// values at least 24 bits clear of that boundary, so the case is
    /// unreachable. It goes unchecked because a check would cost an `O(N)`
    /// scan on the hot path of every multiplication.
    #[inline]
    pub(crate) fn wrapping_abs(self) -> Self {
        if self.is_negative() {
            self.wrapping_neg()
        } else {
            self
        }
    }

    /// Returns one past the index of the highest nonzero limb, or `0` if all
    /// limbs are zero.
    ///
    /// It serves only to trim the multiplication loops. For a non-negative
    /// value it is the number of limbs that can contribute to a product.
    #[inline]
    fn significant_limbs(&self) -> usize {
        let mut i = N;
        while i > 0 && self.limbs[i - 1] == 0 {
            i -= 1;
        }
        i
    }

    /// Returns one past the index of the highest set bit, or `0` for zero.
    ///
    /// The value must be non-negative. For a negative two's-complement value
    /// the high limbs are all ones and the result would be `N × 64`, which is
    /// true of the representation but says nothing about the magnitude.
    /// Callers take [`wrapping_abs`](Self::wrapping_abs) first.
    ///
    /// This is the magnitude measure that rational-to-`f64` rounding needs.
    /// The difference of the numerator's and denominator's bit lengths bounds
    /// the binary exponent of the quotient, which makes a candidate exponent
    /// computable without division.
    #[inline]
    pub(crate) fn bit_length(&self) -> usize {
        debug_assert!(
            !self.is_negative(),
            "bit_length measures a magnitude; take wrapping_abs first"
        );
        let mut i = N;
        while i > 0 {
            i -= 1;
            if self.limbs[i] != 0 {
                return i * 64 + (64 - self.limbs[i].leading_zeros() as usize);
            }
        }
        0
    }

    /// Returns the 64-bit window of this value beginning at bit `shift`.
    ///
    /// Bits at or beyond `N × 64` read as zero, so a window straddling the
    /// top of the representation is zero-extended rather than wrapping. With
    /// `shift = bit_length() − 64` it yields the 64 most significant bits of
    /// a magnitude, the estimate input for rounding.
    ///
    /// The value must be non-negative, for the reason given on
    /// [`bit_length`](Self::bit_length).
    #[inline]
    pub(crate) fn bit_window(&self, shift: usize) -> u64 {
        debug_assert!(
            !self.is_negative(),
            "bit_window reads a magnitude; take wrapping_abs first"
        );
        let w = shift / 64;
        let b = shift % 64;
        let lo = if w < N { self.limbs[w] } else { 0 };
        if b == 0 {
            return lo;
        }
        let hi = if w + 1 < N { self.limbs[w + 1] } else { 0 };
        (lo >> b) | (hi << (64 - b))
    }

    /// Returns `true` if the most-significant bit (the two's-complement sign
    /// bit) of limb `N − 1` is set.
    #[inline]
    pub(crate) fn is_negative(&self) -> bool {
        (self.limbs[N - 1] >> 63) != 0
    }

    /// Returns `true` if the value is strictly positive (nonzero and not
    /// negative).
    #[inline]
    pub(crate) fn is_positive(&self) -> bool {
        !self.is_zero() && !self.is_negative()
    }

    // ===========================================================================
    // Debug-mode overflow guard
    // ===========================================================================

    /// Returns `true` if this value fits within `n` signed two's-complement
    /// bits.
    ///
    /// A `WideInt<N>` fits in `n` signed bits when every bit at positions
    /// `n − 1` through `N × 64 − 1` equals the sign bit (bit `N × 64 − 1`).
    /// This is the standard sign-extension condition. An `n`-bit signed value
    /// carries its sign at bit `n − 1`, so that bit and everything above it
    /// must repeat the sign.
    ///
    /// Equivalently, the method returns `true` exactly when
    /// `−2^(n−1) ≤ self < 2^(n−1)`.
    ///
    /// # Use as an overflow guard
    ///
    /// The width proofs for [`Orient2dInt`] and [`Incircle2dInt`] establish
    /// that the Stage 4 determinant always fits within a derived bit bound
    /// strictly less than the allocated width. The method belongs inside a
    /// `debug_assert!` immediately after the determinant is computed.
    ///
    /// ```text
    /// // orient2d: the proof gives |det| < 2^4199, i.e. 4200 signed bits.
    /// debug_assert!(
    ///     det.fits_in_signed_bits(4200),
    ///     "orient2d Stage 4 overflow: width analysis violated; result sign may be incorrect"
    /// );
    /// ```
    ///
    /// The fence is `text` rather than a doctest because `WideInt` is
    /// `pub(crate)`. A doctest compiles as an external crate and cannot name
    /// it, so the snippet could only ever be `ignore`d, which reports as a
    /// skipped test and invites the question of what is blocking it. Unit
    /// tests at the bottom of this module cover the behavior it illustrates
    /// instead.
    ///
    /// An assertion that fires means the width analysis is wrong by at least
    /// one bit and the returned sign may be incorrect, a silent wrong answer
    /// and the one failure this crate exists to rule out.
    ///
    /// # Behavior at the boundaries
    ///
    /// - With `n = 1`, only the all-zeros and all-ones values pass (`0` and
    ///   `-1`).
    /// - With `n = N * 64`, the method always returns `true`, since any value
    ///   fits in the full width.
    ///
    /// # Parameters
    ///
    /// `n` is the candidate bit width and must satisfy `1 ≤ n ≤ N × 64`.
    #[inline]
    pub(crate) fn fits_in_signed_bits(&self, n: usize) -> bool {
        debug_assert!(
            n >= 1 && n <= N * 64,
            "fits_in_signed_bits requires 1 <= n <= N*64"
        );

        let sign = self.is_negative();
        let expected_limb = if sign { u64::MAX } else { 0u64 };

        // An `n`-bit signed value carries its sign at bit `n − 1`, so bit
        // `n − 1`, rather than bit `n`, is the first bit that must already
        // equal the sign. A scan starting at `n` would accept values needing
        // `n + 1` signed bits, making every guard in the crate one bit looser
        // than the proof it checks.
        let first = n - 1;
        let upper_limb = first / 64;
        let upper_bit = first % 64;

        // In the boundary limb, bits [63 .. upper_bit] must equal the sign.
        if upper_bit != 0 && upper_limb < N {
            let mask = u64::MAX << upper_bit;
            if self.limbs[upper_limb] & mask != expected_limb & mask {
                return false;
            }
        }

        // Every limb wholly above the boundary must be pure sign extension.
        let full_start = if upper_bit != 0 {
            upper_limb + 1
        } else {
            upper_limb
        };
        for idx in full_start..N {
            if self.limbs[idx] != expected_limb {
                return false;
            }
        }

        true
    }

    // ===========================================================================
    // Arithmetic
    // ===========================================================================

    /// Adds `rhs` to `self`, propagating carry across all `N` limbs.
    ///
    /// For in-domain input, whose mathematical result fits within `N × 64`
    /// signed bits, carry never propagates past the last limb and the result
    /// is exact.
    #[inline]
    fn wrapping_add(self, rhs: Self) -> Self {
        let mut result = Self::zero();
        let mut carry = false;

        for i in 0..N {
            let (a, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (b, c2) = a.overflowing_add(u64::from(carry));
            result.limbs[i] = b;
            carry = c1 || c2;
        }

        result
    }

    /// Negates `self` using the standard two's-complement identity
    /// `−x = ¬x + 1`.
    ///
    /// In the determinant computations of `orient2d` and `incircle`, negation
    /// applies only to values well within the allocated width, so the
    /// minimum-value wrapping case never occurs in practice.
    #[inline]
    fn wrapping_neg(self) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            result.limbs[i] = !self.limbs[i];
        }
        result.wrapping_add(Self::from_u64(1))
    }

    /// Multiplies `self` by `rhs` using schoolbook multi-limb multiplication,
    /// retaining only the lower `N × 64` bits of the full product.
    ///
    /// # Correctness for signed inputs
    ///
    /// The operands are split into sign and magnitude, the magnitudes are
    /// multiplied, and the sign is reapplied. For in-domain input, whose
    /// mathematical product fits within `N × 64` signed bits, the result is
    /// exact.
    ///
    /// Multiplying magnitudes rather than the two's-complement
    /// representations makes the loop trimming below both sound **and**
    /// effective. A negative two's-complement value has its high limbs set to
    /// all ones, so nothing could be skipped, whereas a magnitude has its
    /// high limbs clear. The arithmetic result is identical either way, so
    /// this is a property of performance rather than of meaning.
    ///
    /// # Why trimming matters at these widths
    ///
    /// The constructed-point predicates compute at up to `N = 198`
    /// ([`RationalOrderInt`]). An untrimmed schoolbook product there is
    /// `N²/2 ≈ 19,600` 128-bit multiplications regardless of operand
    /// magnitude. A scaled coordinate, however, is a 53-bit significand
    /// shifted into position by the `2^1074` factor, so only a handful of
    /// limbs are ever nonzero, three or four for coordinates of ordinary
    /// magnitude. Skipping the zero limbs therefore separates a predicate
    /// that can be swept over 100,000 oracle cases from one that cannot.
    ///
    /// # Precondition
    ///
    /// Neither operand is the two's-complement minimum (see
    /// [`wrapping_abs`](Self::wrapping_abs)).
    #[inline]
    fn wrapping_mul(self, rhs: Self) -> Self {
        let negative = self.is_negative() != rhs.is_negative();
        let a = self.wrapping_abs();
        let b = rhs.wrapping_abs();

        let a_len = a.significant_limbs();
        let b_len = b.significant_limbs();

        let mut result = Self::zero();

        for i in 0..a_len {
            if a.limbs[i] == 0 {
                continue;
            }

            let mut carry = 0u128;
            let j_end = b_len.min(N - i);

            for j in 0..j_end {
                let k = i + j;
                let acc = u128::from(a.limbs[i]) * u128::from(b.limbs[j])
                    + u128::from(result.limbs[k])
                    + carry;
                // The truncation is the purpose of a schoolbook limb split.
                // The low 64 bits are this limb's digit and the high bits
                // become the carry on the next line, so nothing is discarded.
                #[allow(clippy::cast_possible_truncation)]
                let digit = acc as u64;
                result.limbs[k] = digit;
                carry = acc >> 64;
            }

            // The inner loop stops at `b_len` rather than at `N − i`, so a
            // carry leaving the last digit it wrote still has to travel. An
            // untrimmed loop absorbs this implicitly by running to the end of
            // the word array, whereas here it propagates explicitly. Dropping
            // it would lose a limb of the product.
            let mut k = i + j_end;
            while carry != 0 && k < N {
                let acc = u128::from(result.limbs[k]) + carry;
                #[allow(clippy::cast_possible_truncation)]
                let digit = acc as u64;
                result.limbs[k] = digit;
                carry = acc >> 64;
                k += 1;
            }
        }

        if negative {
            result.wrapping_neg()
        } else {
            result
        }
    }

    /// Left-shifts `self` by `shift` bits, discarding any bits that overflow
    /// the allocated width.
    ///
    /// In the `f64`-to-`WideInt` conversion the maximum shift is
    /// `biased_exp − 1 ≤ 1022`, and the mantissa occupies at most 53 bits
    /// (`1022 + 53 = 1075 < N × 64`), so no bits are discarded in practice.
    #[inline]
    fn wrapping_shl(self, shift: usize) -> Self {
        if shift >= N * 64 {
            return Self::zero();
        }

        let word_shift = shift / 64;
        let bit_shift = shift % 64;

        let mut result = Self::zero();

        for i in word_shift..N {
            let src = i - word_shift;
            result.limbs[i] |= self.limbs[src] << bit_shift;
            if bit_shift != 0 && src > 0 {
                result.limbs[i] |= self.limbs[src - 1] >> (64 - bit_shift);
            }
        }

        result
    }
}

// ===========================================================================
// Operator impls
// ===========================================================================

impl<const N: usize> Add for WideInt<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl<const N: usize> Sub for WideInt<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_add(rhs.wrapping_neg())
    }
}

impl<const N: usize> Neg for WideInt<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

impl<const N: usize> Mul for WideInt<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl<const N: usize> Shl<usize> for WideInt<N> {
    type Output = Self;
    #[inline]
    fn shl(self, rhs: usize) -> Self {
        self.wrapping_shl(rhs)
    }
}

// ===========================================================================
// f64 conversion
// ===========================================================================

/// Converts a finite `f64` to a `WideInt<N>` scaled by `2^1074`.
///
/// The result is the unique integer equal to `x × 2^1074`. The conversion is
/// exact for every finite `f64` because the minimum positive `f64` is
/// `2^-1074`, so the scaled magnitude is always a non-negative integer before
/// [`wrapping_neg`](WideInt::wrapping_neg) applies the sign.
///
/// # Derivation from IEEE 754 encoding
///
/// Normal `f64` (`biased_exp > 0`):
/// ```text
/// value   = (2^52 + mantissa) × 2^(biased_exp − 1075)
/// × 2^1074  →  (2^52 + mantissa) << (biased_exp − 1)
/// ```
///
/// Subnormal `f64` (`biased_exp == 0`):
/// ```text
/// value   = mantissa × 2^-1074
/// × 2^1074  →  mantissa   (no shift)
/// ```
///
/// Zero (`0.0` and `-0.0`) maps to integer zero.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified.
fn f64_to_wide_int<const N: usize>(x: f64) -> WideInt<N> {
    if x == 0.0 {
        return WideInt::zero();
    }

    let bits = x.to_bits();
    let negative = (bits >> 63) != 0;
    let biased_exp = ((bits >> 52) & 0x7FF) as u32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    let (integer_mantissa, shift) = if biased_exp > 0 {
        ((1u64 << 52) | mantissa, (biased_exp - 1) as usize)
    } else {
        (mantissa, 0)
    };

    let magnitude = WideInt::from_u64(integer_mantissa) << shift;
    if negative { -magnitude } else { magnitude }
}

/// Converts a finite `f64` to an [`Orient2dInt`] (4224-bit signed) scaled by
/// `2^1074`. The result is exact for all finite `f64`.
///
/// [`f64_to_wide_int`] carries the derivation. This is the conversion for
/// the coordinates of the `orient2d` Stage 4 determinant.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified.
#[inline]
pub(crate) fn f64_to_orient_int(x: f64) -> Orient2dInt {
    f64_to_wide_int(x)
}

/// Converts a finite `f64` to an [`Incircle2dInt`] (8448-bit signed) scaled by
/// `2^1074`. The result is exact for all finite `f64`.
///
/// [`f64_to_wide_int`] carries the derivation. This is the conversion for
/// the coordinates of the `incircle` Stage 4 determinant.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified.
#[inline]
pub(crate) fn f64_to_incircle_int(x: f64) -> Incircle2dInt {
    f64_to_wide_int(x)
}

/// Converts a finite `f64` to an [`Orient3dInt`] (6336-bit signed) scaled by
/// `2^1074`. The result is exact for all finite `f64`.
///
/// [`f64_to_wide_int`] carries the derivation. This is the conversion for
/// the coordinates of the `orient3d` Stage 4 determinant.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified.
#[inline]
pub(crate) fn f64_to_orient3d_int(x: f64) -> Orient3dInt {
    f64_to_wide_int(x)
}

/// Converts a finite `f64` to an [`InsphereInt`] (10560-bit signed) scaled by
/// `2^1074`. The result is exact for all finite `f64`.
///
/// [`f64_to_wide_int`] carries the derivation. This is the conversion for
/// the coordinates of the `insphere` Stage 4 determinant.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified.
#[inline]
pub(crate) fn f64_to_insphere_int(x: f64) -> InsphereInt {
    f64_to_wide_int(x)
}

/// Converts a finite `f64` to a [`RationalCoordInt`] (6336-bit signed) scaled
/// by `2^1074`. The result is exact for all finite `f64`.
///
/// [`f64_to_wide_int`] carries the derivation. This is the conversion for
/// segment endpoints entering constructed-point arithmetic.
///
/// # Precondition
///
/// `x` must be finite. Behavior on `NaN` or infinity is unspecified, and
/// every caller in [`crate::rational`] checks finiteness before converting.
#[inline]
pub(crate) fn f64_to_rational_int(x: f64) -> RationalCoordInt {
    f64_to_wide_int(x)
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

    type W = WideInt<66>;

    fn one() -> W {
        W::one()
    }

    /// `2^k` as a `WideInt<66>`.
    fn pow2(k: usize) -> W {
        one() << k
    }

    // ===========================================================================
    // fits_in_signed_bits
    // ===========================================================================

    /// The boundary claims of the method's own documentation.
    ///
    /// `n = 1` admits only `0` and `−1`, because a one-bit signed value
    /// represents exactly that pair. An off-by-one in the sign-extension scan
    /// would get this case wrong, since a scan starting at bit `n` rather
    /// than bit `n − 1` would also admit `1`.
    #[test]
    fn fits_in_one_signed_bit_admits_only_zero_and_minus_one() {
        assert!(W::zero().fits_in_signed_bits(1), "0 fits in 1 signed bit");
        assert!((-one()).fits_in_signed_bits(1), "-1 fits in 1 signed bit");
        assert!(
            !one().fits_in_signed_bits(1),
            "1 does NOT fit in 1 signed bit"
        );
        assert!(
            !(-one() - one()).fits_in_signed_bits(1),
            "-2 does NOT fit in 1 signed bit"
        );
    }

    #[test]
    fn fits_in_full_width_is_always_true() {
        for v in [W::zero(), one(), -one(), pow2(4000), -pow2(4000)] {
            assert!(
                v.fits_in_signed_bits(66 * 64),
                "every value fits in the allocated width"
            );
        }
    }

    /// `2^k` needs exactly `k + 2` signed bits, `k + 1` magnitude bits plus
    /// the sign. The check spans limb boundaries, where an off-by-one in the
    /// limb/bit split would show up and a single-limb test would not.
    #[test]
    fn fits_tracks_the_exact_signed_width_across_limb_boundaries() {
        for k in [0_usize, 1, 62, 63, 64, 65, 127, 128, 1073, 2097] {
            let v = pow2(k);
            assert!(
                v.fits_in_signed_bits(k + 2),
                "2^{k} must fit in {} signed bits",
                k + 2
            );
            assert!(
                !v.fits_in_signed_bits(k + 1),
                "2^{k} must NOT fit in {} signed bits",
                k + 1
            );
        }
    }

    /// `−2^k` fits in `k + 1` signed bits, one fewer than `+2^k`, because
    /// two's complement is asymmetric at the boundary.
    #[test]
    fn negative_power_of_two_fits_one_bit_tighter_than_positive() {
        for k in [1_usize, 63, 64, 1073] {
            let v = -pow2(k);
            assert!(
                v.fits_in_signed_bits(k + 1),
                "-2^{k} must fit in {} signed bits",
                k + 1
            );
            assert!(
                !v.fits_in_signed_bits(k),
                "-2^{k} must NOT fit in {k} signed bits"
            );
        }
    }

    // ===========================================================================
    // Lemma S, checked against the implementation rather than argued
    // ===========================================================================

    /// Lemma S. Every finite `f64` scaled by `2^1074` is `< 2^2098`, so it
    /// fits in **2099** signed bits, and `f64::MAX` witnesses that 2098 are not
    /// enough.
    ///
    /// Five width proofs rest on this bound, so it is checked here directly
    /// rather than trusted from the derivation.
    #[test]
    fn lemma_s_scaled_coordinate_needs_exactly_2099_signed_bits() {
        let m = f64_to_orient_int(f64::MAX);
        assert!(
            m.fits_in_signed_bits(2099),
            "Lemma S: a scaled finite f64 fits in 2099 signed bits"
        );
        assert!(
            !m.fits_in_signed_bits(2098),
            "Lemma S is tight: f64::MAX scaled needs all 2099 bits"
        );
        assert!(f64_to_orient_int(f64::MIN).fits_in_signed_bits(2099));
    }

    /// Corollary S1. A difference of two scaled coordinates is `< 2^2099`,
    /// so it fits in **2100** signed bits, and the widest difference witnesses
    /// that 2099 are not enough.
    #[test]
    fn corollary_s1_coordinate_difference_needs_exactly_2100_signed_bits() {
        let d = f64_to_orient_int(f64::MAX) - f64_to_orient_int(f64::MIN);
        assert!(
            d.fits_in_signed_bits(2100),
            "S1: a scaled coordinate difference fits in 2100 signed bits"
        );
        assert!(
            !d.fits_in_signed_bits(2099),
            "S1 is tight: MAX − MIN needs all 2100 bits"
        );
    }

    // ===========================================================================
    // f64 conversion
    // ===========================================================================

    #[test]
    fn f64_conversion_is_exact_at_the_representable_extremes() {
        assert!(f64_to_orient_int(0.0).is_zero());
        assert!(
            f64_to_orient_int(-0.0).is_zero(),
            "negative zero scales to integer zero, not to -0"
        );
        assert_eq!(
            f64_to_orient_int(f64::from_bits(1)),
            one(),
            "the minimum positive subnormal 2^-1074 scales to exactly 1; \
             this is why the scaling factor is 2^1074"
        );
        assert_eq!(f64_to_orient_int(1.0), pow2(1074));
        assert_eq!(f64_to_orient_int(-1.0), -pow2(1074));
        assert_eq!(f64_to_orient_int(2.0), pow2(1075));
        assert_eq!(f64_to_orient_int(0.5), pow2(1073));
    }

    // ===========================================================================
    // widen
    // ===========================================================================

    #[test]
    fn widen_sign_extends_rather_than_zero_extends() {
        let narrow: WideInt<66> = -pow2(2000);
        let wide: WideInt<198> = narrow.widen();
        assert!(
            wide.is_negative(),
            "a negative value stays negative when widened"
        );

        // Widening preserves the value, so adding the widened negation gives
        // zero. Zero-extension would leave an enormous positive residue.
        let negated: WideInt<198> = (-narrow).widen();
        assert!(
            (wide + negated).is_zero(),
            "widening must preserve the value"
        );

        let positive: WideInt<198> = pow2(2000).widen();
        assert!(positive.is_positive());
        assert!(positive.fits_in_signed_bits(2002));
        assert!(!positive.fits_in_signed_bits(2001));
    }

    // ===========================================================================
    // wrapping_mul
    // ===========================================================================

    /// The carry that leaves the trimmed inner loop.
    ///
    /// `(2^64 − 1)^2 = 2^128 − 2^65 + 1` occupies two limbs, but both operands
    /// have a significant length of one, so the trimmed inner loop writes only
    /// limb 0 and the high half reaches limb 1 solely through the trailing
    /// carry-propagation loop. Deleting that loop loses a limb of the product,
    /// a wrong answer rather than a crash, and this test catches it.
    #[test]
    fn mul_propagates_the_carry_past_the_trimmed_inner_loop() {
        let a = W::from_u64(u64::MAX);
        let expected = pow2(128) - pow2(65) + one();
        assert_eq!(a * a, expected, "(2^64 - 1)^2 = 2^128 - 2^65 + 1");
    }

    /// The zero-limb skip changes no product.
    ///
    /// A power of two has exactly one nonzero limb, at an arbitrarily high
    /// index, so the test exercises the skip in the outer loop and the
    /// trimming in both, including cases where the operands' nonzero limbs do
    /// not overlap.
    #[test]
    fn mul_of_powers_of_two_is_a_shift() {
        for (a, b) in [
            (0_usize, 0_usize),
            (1, 1),
            (63, 1),
            (64, 64),
            (1000, 1074),
            (2097, 2097),
        ] {
            assert_eq!(
                pow2(a) * pow2(b),
                pow2(a + b),
                "2^{a} * 2^{b} must equal 2^{}",
                a + b
            );
        }
    }

    #[test]
    fn mul_applies_the_sign_of_the_product() {
        let a = pow2(1000);
        let b = pow2(74);
        let p = pow2(1074);
        assert_eq!(a * b, p);
        assert_eq!((-a) * b, -p);
        assert_eq!(a * (-b), -p);
        assert_eq!((-a) * (-b), p, "two negatives give a positive product");
    }

    #[test]
    fn mul_by_zero_and_one() {
        let a = pow2(2000) - pow2(37) + one();
        assert!((a * W::zero()).is_zero());
        assert_eq!(a * one(), a);
        assert_eq!(a * (-one()), -a);
    }

    /// Multiplication is commutative, including when the two operands have
    /// very different significant lengths, the asymmetric case that trimming
    /// introduces, since the outer loop runs over the left operand only.
    #[test]
    fn mul_is_commutative_for_asymmetric_operand_widths() {
        let wide = pow2(2000) + pow2(1000) + one();
        let narrow = W::from_u64(3);
        assert_eq!(wide * narrow, narrow * wide);

        let other = pow2(130) - one();
        assert_eq!(wide * other, other * wide);
    }
}
