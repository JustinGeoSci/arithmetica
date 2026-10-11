// src/sign.rs

//! The sign of a geometric predicate result.

use core::ops::Neg;

/// All predicates in this crate return `Sign` rather than `f64` or `bool`.
/// The three variants correspond to the three mathematically distinct outcomes
/// of a predicate determinant.
///
/// # Why three variants matter
///
/// The `Zero` variant represents exact degeneracy: three collinear points, a
/// point exactly on a circle boundary. Naive floating-point predicates often
/// return a spurious non-zero sign for degenerate input. This crate's exact
/// fallback detects `Zero` correctly when the fast path cannot distinguish it
/// from a near-zero non-zero result. Callers must handle all three variants.
///
/// ```
/// use arithmetica::Sign;
///
/// fn label(s: Sign) -> &'static str {
///     match s {
///         Sign::Positive => "counterclockwise",
///         Sign::Negative => "clockwise",
///         Sign::Zero     => "collinear",
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    /// The predicate determinant is strictly positive.
    Positive,
    /// The predicate determinant is strictly negative.
    Negative,
    /// The predicate determinant is exactly zero, a degenerate configuration.
    Zero,
}

impl Neg for Sign {
    type Output = Sign;

    /// Negates the sign, corresponding to reversing vertex order.
    ///
    /// Swapping any two input points negates the `orient2d` determinant:
    /// `orient2d(pa, pb, pc) == -orient2d(pb, pa, pc)`. This expresses that
    /// relationship on the return value without a `match` at every reversal
    /// site. Negating `Zero` yields `Zero`.
    fn neg(self) -> Sign {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
        }
    }
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

    #[test]
    fn neg_is_correct() {
        assert_eq!(-Sign::Positive, Sign::Negative);
        assert_eq!(-Sign::Negative, Sign::Positive);
        assert_eq!(-Sign::Zero, Sign::Zero);
    }

    #[test]
    fn sign_is_copy() {
        let s = Sign::Positive;
        let _a = s;
        let _b = s; // Compiles only if Sign is Copy
    }
}
