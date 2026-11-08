// src/lib.rs

//! Exact geometric predicates and constructions for `f64` input.
//!
//! [`orient2d`], [`incircle`], [`orient3d`] and [`insphere`] return the sign
//! of their determinant **exactly** for every finite input, any magnitudes,
//! subnormals and exactly degenerate configurations included. A NaN or
//! infinite coordinate returns [`NonFinite`] rather than a plausible-looking
//! [`Sign`].
//!
//! ```
//! use arithmetica::{Sign, orient2d};
//!
//! assert_eq!(orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]), Ok(Sign::Positive));
//! assert_eq!(orient2d([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]), Ok(Sign::Zero));
//! assert!(orient2d([f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]).is_err());
//! ```
//!
//! # Contents
//!
//! - **Predicates**. [`orient2d`], [`incircle`], [`orient3d`], [`insphere`].
//! - **Sign predicates**. [`compare_distance`] (which of two points is nearer
//!   to a third), [`dot_sign`] (whether an angle is acute, right or obtuse),
//!   and the distance comparisons [`compare_point_distance`] and
//!   [`compare_segment_distance`] (a point's distance to a point or to a
//!   closed segment against a given `d`), on the same sequence of stages.
//! - **Exact 2D constructions**. [`RationalCoord2d`], built from `f64` input or
//!   as the exact intersection of two lines, with the exact predicates
//!   [`rational_coords_equal`], [`rational_compare_along`] and
//!   [`orient2d_rational`]. [`RationalCoord2d::f64_bracket`] reports the two
//!   `f64` neighbors of an exact coordinate and its exact position between
//!   them. No function in the crate returns an approximation of a value.
//! - **Expansion arithmetic**. The error-free primitives in [`expansion`], exact
//!   under the conditions their documentation states.
//! - **Stage counters**. [`counters`], behind the `stage-counters` feature.
//!
//! # How the predicates stay exact
//!
//! Each predicate is a sequence of stages, and each stage either certifies a
//! sign or passes the input on. The stages are numbered the same way in
//! every predicate's documentation.
//!
//! - **Stage 1**, a filter in native `f64`, applies the underflow-protected
//!   error bound of Bartels, Fisikopoulos and Weiser (2023), valid for every
//!   finite input.
//! - **Stages 2 and 3**, Shewchuk's adaptive expansion stages, run only
//!   inside an exactness window of magnitudes where binary64 provably
//!   behaves as his unbounded-exponent model, with input moved there by an
//!   exact power-of-two rescale when possible. A predicate with a single
//!   expansion stage has a Stage 2 and no Stage 3.
//! - **Stage 4**, exact fixed-width integer arithmetic, resolves everything
//!   else.
//!
//! Some predicates also have a zero filter ahead of Stage 1, which certifies
//! an exactly degenerate input by comparisons alone. Shewchuk names his error
//! bounds A, B and C. Stage 1 applies a bound of the A form, and Stages 2
//! and 3, where they test a bound, apply B and C.
//!
//! The README and `docs/exactness_window.md` in the repository give the full
//! argument and the references.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![warn(missing_docs)]

mod compare_distance;
pub mod counters;
mod dot_sign;
mod exact_int;
pub mod expansion;
mod incircle;
mod insphere;
mod orient2d;
mod orient3d;
mod range;
pub mod rational;
mod round;
mod segment_distance;
mod sign;
mod util;

pub use compare_distance::compare_distance;
pub use dot_sign::dot_sign;
pub use incircle::incircle;
pub use insphere::insphere;
pub use orient2d::orient2d;
pub use orient3d::orient3d;
pub use rational::{
    RationalCoord2d, orient2d_rational, rational_compare_along, rational_coords_equal,
};
pub use round::{BracketPosition, F64Bracket};
pub use segment_distance::{compare_point_distance, compare_segment_distance};
pub use sign::Sign;

// The README's code blocks compile and run as doctests, so the examples on
// the repository page cannot drift from the crate.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

/// Error returned by a geometric predicate when an input coordinate is not
/// finite (NaN, `+∞`, or `−∞`).
///
/// A non-finite coordinate has no geometric meaning, so the predicate cannot
/// return a `Sign`. It is reported as an error rather than a
/// plausible-looking `Sign::Zero`, which would be indistinguishable from a
/// genuine degenerate (collinear, coplanar or cospherical) result.
///
/// The type is deliberately opaque, with no public fields, so that detail
/// such as which argument was non-finite can be added later without a
/// breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonFinite {
    _private: (),
}

impl NonFinite {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self { _private: () }
    }
}

impl core::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("geometric predicate received a non-finite coordinate (NaN or infinity)")
    }
}

impl core::error::Error for NonFinite {}
