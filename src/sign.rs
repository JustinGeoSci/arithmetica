// src/sign.rs

//! The sign of a geometric predicate result.

use core::ops::Neg;

/// Every predicate in this crate returns `Sign` rather than `f64` or `bool`.
/// The three variants correspond to the three mathematically distinct
/// outcomes of a predicate determinant.
///
/// # Why three variants matter
///
/// The `Zero` variant represents exact degeneracy, such as three collinear
/// points or a point exactly on a circle's boundary. Naive floating-point
/// predicates often return a spurious non-zero sign for degenerate input.
/// The exact fallback of this crate recovers `Zero` correctly when the fast
/// path cannot distinguish it from a near-zero non-zero result. Callers must
/// handle all three variants.
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

    /// Negates the sign, which corresponds to reversing the vertex order.
    ///
    /// Swapping any two input points negates the `orient2d` determinant, so
    /// `orient2d(pa, pb, pc) == -orient2d(pb, pa, pc)`. Negation expresses
    /// that relationship on the return value without a `match` at every site
    /// that reverses an order. Negating `Zero` yields `Zero`.
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
        let _b = s; // compiles only if Sign is Copy
    }
}
