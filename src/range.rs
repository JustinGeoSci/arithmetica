// src/range.rs

//! The exactness window: the range of input magnitudes within which the
//! adaptive expansion stages are provably exact, and a lossless way to move
//! input into it.
//!
//! # The problem this solves
//!
//! Shewchuk's 1997 analysis assumes an unbounded exponent range: his §2 takes
//! it as given that no operation overflows or underflows. Under that
//! assumption `two_sum` and `two_product` are exact, expansions are
//! nonoverlapping and sum to the exact value, and the Stage A, B and C error
//! bounds hold.
//!
//! Binary64 has a bounded exponent range, and the assumption fails at both
//! ends. Near the top, a product overflows to infinity. Near the bottom, a
//! product can be smaller than the smallest subnormal and round to zero, or
//! land among the subnormals where it has fewer than 53 significant bits. In
//! either case the operation is no longer error-free, the expansion no longer
//! sums to the determinant, and a stage can certify a sign that is wrong.
//! Unguarded, this is not rare: inputs placed next to a sign boundary at the
//! extremes of the exponent range produce wrong non-zero signs at a rate of
//! about one call in eighty.
//!
//! # Lemma G
//!
//! Write `q` for the smallest exponent of a lowest set bit over the non-zero
//! input coordinates, so every coordinate is an integer multiple of `2^q`,
//! and `E` for the largest exponent of a leading bit, so every coordinate is
//! below `2^(E+1)`. Let `d` be the degree of the predicate's determinant.
//!
//! **Inside the window**, `q >= Q_MIN` and `E <= E_MAX` for the predicate,
//! every operation of the adaptive stages returns the same value in binary64
//! as it would with an unbounded exponent range, so Shewchuk's theorems apply
//! to the binary64 execution unchanged.
//!
//! The argument, in brief. Every value the stages compute is a sum of
//! products of coordinates, of degree at most `d`, so it is an integer
//! multiple of `2^(d·min(q, 0))`. The window's lower limit makes that at
//! least `2^-1074`, and a multiple of `2^-1074` below `2^-1022` is a
//! subnormal, hence exactly representable: rounding a value on that grid
//! gives the same result in both models, and the result stays on the grid.
//! The window's upper limit keeps every value, including the intermediates
//! inside `split` and `two_product`, below the overflow threshold. The error
//! bound constants are the one kind of value not on the grid; the lower limit
//! is set so that their products stay in the normal range, where relative
//! error applies.
//!
//! The limits, with the margin by which each clears what the argument needs:
//!
//! | predicate | `d` | `Q_MIN` | `E_MAX` | `d·Q_MIN` | margin below | margin above |
//! |---|---:|---:|---:|---:|---:|---:|
//! | `orient2d` | 2 | −450 | 478 | −900 | 19 bits | 26 bits |
//! | `orient3d` | 3 | −300 | 318 | −900 | 19 bits | 26 bits |
//! | `incircle` | 4 | −225 | 238 | −900 | 19 bits | 26 bits |
//! | `insphere` | 5 | −180 | 190 | −900 | 19 bits | 26 bits |
//!
//! **The binding constraint below** is the error-bound products, not the
//! grid. The smallest constant is `CCW_ERR_BOUND_C`, about `2^-102.84`. A
//! non-zero degree-`d` value is at least `2^(d·q)`, so the smallest product
//! is at least `2^(d·q - 102.84)`, and keeping that at or above `2^-1022`
//! needs `d·q >= -919`. The chosen `-900` clears it by 19 bits. The grid
//! itself only needs `d·q >= -1074`, which `-900` clears by 174 bits.
//!
//! **The binding constraint above** comes from the sum of the magnitudes of
//! an expansion's components, which is at most the formal absolute value of
//! the expression that built it, times `(1 + 2^-51)` per operation. With
//! differences below `2^(E+2)`, at most `2^10` monomials of degree `d`, and
//! the `2^27 + 1` splitting factor inside `two_product`, every value stays
//! below `2^(d·(E+2) + 29.01)`; keeping that under `2^1024` needs
//! `d·(E+2) <= 986`. The chosen `960` clears it by 26 bits.
//!
//! # Outside the window
//!
//! If the input's magnitudes all fit within one window's width but sit too
//! low or too high, multiplying every coordinate by a single power of two is
//! exact and moves the whole configuration inside, so the stages run as
//! usual. A positive factor multiplies the determinant by a positive number,
//! so the sign is unchanged. If the span from the lowest set bit to the
//! highest leading bit is wider than the window itself, no factor works and
//! the call goes straight to the exact integer stage.
//!
//! # Why the scaling is never approximate
//!
//! A tempting alternative is to scale toward the geometric mean of the input
//! magnitudes and clamp the exponent. That scaling is lossy: it can round a
//! coordinate into the subnormals, or to zero, and the stages then work on a
//! different configuration than the one the caller passed. Scaling here is
//! exact or it does not happen.
//!
//! # References
//!
//! Shewchuk, J.R. (1997). *Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates*. Discrete & Computational Geometry
//! 18(3), 305–363. §2 states the unbounded-exponent assumption.
//!
//! Bartels, T., Fisikopoulos, V., Weiser, M. (2023). *Fast floating-point
//! filters for robust predicates*. BIT Numerical Mathematics 63:31. Example 1
//! shows that Shewchuk's Stage A filter is invalid under underflow; its
//! Theorem 1(2) is the filter the predicates apply before this window is
//! consulted.
//!
//! The full argument, with the proofs of Lemma G and of the two margins, is
//! in `docs/exactness_window.md`.

/// The smallest positive normal `f64`, `2^-1022`.
///
/// Added after each product when building a magnitude for the
/// underflow-protected Stage A filter: it covers the absolute error of a
/// product that lands among the subnormals, where the relative error bound
/// does not hold (Bartels et al. 2023, rule `R7,UFP`).
pub(crate) const U_N: f64 = f64::MIN_POSITIVE;

/// The smallest positive subnormal `f64`, `2^-1074`.
///
/// Added to the final Stage A bound, which is what makes the filter valid for
/// every finite input rather than only for inputs free of underflow (Bartels
/// et al. 2023, Theorem 1(2)).
pub(crate) const U_S: f64 = 5e-324;

/// The exactness window for one predicate.
///
/// Built only by the four constants below; a window is a property of the
/// predicate's degree, not something a caller chooses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    /// The smallest admissible exponent of a coordinate's lowest set bit.
    q_min: i32,
    /// The largest admissible exponent of a coordinate's leading bit.
    e_max: i32,
}

impl Window {
    /// `d·q_min` and `d·(e_max + 2)` are the same for all four windows, `-900`
    /// and `960`, which keeps one pair of margins to verify rather than four.
    const fn for_degree(degree: i32) -> Self {
        Self {
            q_min: -900 / degree,
            e_max: 960 / degree - 2,
        }
    }

    /// The window for `orient2d`: degree 2, so `-450 ..= 478`.
    pub(crate) const ORIENT2D: Self = Self::for_degree(2);
    /// The window for `orient3d`: degree 3, so `-300 ..= 318`.
    pub(crate) const ORIENT3D: Self = Self::for_degree(3);
    /// The window for `incircle`: degree 4, so `-225 ..= 238`.
    pub(crate) const INCIRCLE: Self = Self::for_degree(4);
    /// The window for `insphere`: degree 5, so `-180 ..= 190`.
    pub(crate) const INSPHERE: Self = Self::for_degree(5);
}

/// What [`prepare`] found, and what the caller should do with the input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Prepared {
    /// The input is already inside the window. Run the stages on it as given.
    AsIs,
    /// Multiplying every coordinate by this power of two is exact and lands
    /// the input inside the window. Run the stages on the scaled coordinates;
    /// the sign is unchanged.
    Scaled(f64),
    /// The input spans more bits than the window is wide, so no single factor
    /// brings it inside. The exact integer stage is the only answer.
    OutOfWindow,
}

/// The exponents of the leading bit and of the lowest set bit of a non-zero
/// finite `f64`.
///
/// For `x = m · 2^e` with the significand `m` written as an integer, the two
/// values are the positions of the highest and lowest set bits of `x` as a
/// real number: `bit_span(0.75) == (-1, -2)`, since `0.75 = 2^-1 + 2^-2`.
///
/// Subnormals are read from the significand field directly. Their biased
/// exponent is zero, so the usual `biased - 1023` is wrong for them, and
/// their leading bit can be anywhere in the field. Getting this wrong for
/// subnormals would silently narrow the window exactly where the underflow
/// defects live.
///
/// # Panics
///
/// In debug builds, if `x` is zero, infinite or NaN. Callers filter zeros,
/// and every predicate rejects non-finite input before reaching here.
fn bit_span(x: f64) -> (i32, i32) {
    debug_assert!(
        x != 0.0 && x.is_finite(),
        "bit_span needs a non-zero finite value"
    );
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7FF) as i32;
    let field = bits & 0x000F_FFFF_FFFF_FFFF;
    if biased > 0 {
        // Normal: the implicit bit sits at 2^(biased - 1023), and the field's
        // bit k at 2^(biased - 1075 + k). `trailing_zeros` of the significand
        // with the implicit bit set is in 0..=52, so neither cast can wrap.
        let significand = field | (1_u64 << 52);
        #[allow(clippy::cast_possible_wrap)]
        let lowest = significand.trailing_zeros() as i32;
        (biased - 1023, biased - 1075 + lowest)
    } else {
        // Subnormal: the value is `field · 2^-1074`, and `field` is non-zero
        // here because `x != 0.0`, so `ilog2` is defined. Both bit positions
        // are in 0..=51, so neither cast can wrap.
        #[allow(clippy::cast_possible_wrap)]
        let leading = field.ilog2() as i32;
        #[allow(clippy::cast_possible_wrap)]
        let lowest = field.trailing_zeros() as i32;
        (leading - 1074, lowest - 1074)
    }
}

/// `2^exponent` as an exact `f64`.
///
/// # Panics
///
/// In debug builds, if the exponent is outside the normal range. Only
/// [`prepare`] calls this, and it never asks for more than `2^894` in
/// magnitude.
fn power_of_two(exponent: i32) -> f64 {
    debug_assert!(
        (-1022..=1023).contains(&exponent),
        "the scale factor must be a normal power of two"
    );
    // `exponent + 1023` is in 1..=2046 by the assertion, so the cast is safe
    // and the result has a zero significand field: an exact power of two.
    #[allow(clippy::cast_sign_loss)]
    let biased = (exponent + 1023) as u64;
    f64::from_bits(biased << 52)
}

/// Decides how to run the adaptive stages on `coords` for the given window.
///
/// One pass over the coordinates, no allocation, no division. It runs only
/// after the fast filter has declined to certify a sign, so it is off the
/// fast path.
///
/// All-zero input is reported as [`Prepared::AsIs`]: there is nothing to
/// scale, and a determinant of zeros is zero in any model.
pub(crate) fn prepare(coords: &[f64], window: Window) -> Prepared {
    let mut highest_leading = i32::MIN;
    let mut lowest_set = i32::MAX;
    for &coordinate in coords {
        if coordinate != 0.0 {
            let (leading, lowest) = bit_span(coordinate);
            if leading > highest_leading {
                highest_leading = leading;
            }
            if lowest < lowest_set {
                lowest_set = lowest;
            }
        }
    }
    if highest_leading == i32::MIN {
        return Prepared::AsIs;
    }
    if lowest_set >= window.q_min && highest_leading <= window.e_max {
        return Prepared::AsIs;
    }

    // The admissible shifts form the interval `[up, down]`: shifting by `k`
    // moves the lowest set bit to `lowest_set + k`, which must reach `q_min`,
    // and the leading bit to `highest_leading + k`, which must not pass
    // `e_max`.
    let up = window.q_min - lowest_set;
    let down = window.e_max - highest_leading;
    if up > down {
        // The input spans more bits than the window is wide.
        return Prepared::OutOfWindow;
    }

    // One of the two bounds is the violated one; take it, which is also the
    // smaller shift. Both ends of the interval are valid, so the choice is
    // only a matter of staying near the input's own magnitudes.
    let shift = if up > 0 { up } else { down };
    debug_assert!(shift != 0, "an in-window input returns AsIs above");

    // The shift is bounded by the width of the exponent range: `lowest_set` is
    // at least `-1074` and `q_min` at most `-180`, so `up <= 894`; likewise
    // `highest_leading` is at most `1023` and `e_max` at least `190`, so
    // `down >= -833`. Either way `2^shift` is a normal power of two, and
    // multiplying a coordinate by it only changes the exponent field, so the
    // product is exact.
    Prepared::Scaled(power_of_two(shift))
}

#[cfg(test)]
mod tests {
    // An exact-value assertion IS a float comparison: these tests check that
    // a scale factor is exactly a given power of two, and that scaling and
    // unscaling returns the identical bit pattern. A tolerance would defeat
    // the purpose.
    #![allow(clippy::float_cmp)]

    use super::*;

    /// `bit_span` on values whose bit layout is known by construction.
    #[test]
    fn bit_span_reads_both_ends_of_a_normal_value() {
        // 1.0 = 2^0: one set bit, so both ends are 0.
        assert_eq!(bit_span(1.0), (0, 0));
        // 0.75 = 2^-1 + 2^-2.
        assert_eq!(bit_span(0.75), (-1, -2));
        // 3.0 = 2^1 + 2^0, and the sign does not matter.
        assert_eq!(bit_span(3.0), (1, 0));
        assert_eq!(bit_span(-3.0), (1, 0));
        // 1 + 2^-52 is the float above 1.0: its lowest bit is the last one of
        // the significand field.
        assert_eq!(bit_span(1.0 + f64::EPSILON), (0, -52));
        // The largest finite value, (2 - 2^-52)·2^1023.
        assert_eq!(bit_span(f64::MAX), (1023, 971));
        // The smallest positive normal.
        assert_eq!(bit_span(f64::MIN_POSITIVE), (-1022, -1022));
    }

    /// Subnormals have a zero biased exponent, so they need the other branch.
    #[test]
    fn bit_span_reads_both_ends_of_a_subnormal() {
        // The smallest positive subnormal, 2^-1074.
        assert_eq!(bit_span(f64::from_bits(1)), (-1074, -1074));
        // 2^-1073 + 2^-1074.
        assert_eq!(bit_span(f64::from_bits(3)), (-1073, -1074));
        // The largest subnormal, (2^52 - 1)·2^-1074: every bit of the field.
        let largest = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        assert_eq!(bit_span(largest), (-1023, -1074));
        // One bit in the middle of the field, 2^(20 - 1074).
        assert_eq!(bit_span(f64::from_bits(1 << 20)), (-1054, -1054));
    }

    /// The four windows have the limits the module doc tabulates, and the two
    /// products that the margins were derived for.
    #[test]
    fn the_four_windows_have_their_tabulated_limits() {
        for (degree, window, q_min, e_max) in [
            (2, Window::ORIENT2D, -450, 478),
            (3, Window::ORIENT3D, -300, 318),
            (4, Window::INCIRCLE, -225, 238),
            (5, Window::INSPHERE, -180, 190),
        ] {
            assert_eq!(window.q_min, q_min, "degree {degree}");
            assert_eq!(window.e_max, e_max, "degree {degree}");
            assert_eq!(degree * window.q_min, -900, "degree {degree}");
            assert_eq!(degree * (window.e_max + 2), 960, "degree {degree}");
        }
    }

    /// Input already inside the window is left alone, and so is all-zero
    /// input.
    #[test]
    fn in_window_input_is_used_as_is() {
        let window = Window::ORIENT2D;
        // Parcel-scale coordinates on a 1/1024 grid.
        assert_eq!(
            prepare(
                &[1_400_000.25, 900_000.5, 1_400_017.75, 899_990.125],
                window
            ),
            Prepared::AsIs
        );
        assert_eq!(prepare(&[0.0, 0.0, 0.0, 0.0], window), Prepared::AsIs);
        assert_eq!(prepare(&[], window), Prepared::AsIs);
        // Exactly on both limits: the lowest set bit at 2^-450 and the
        // leading bit at 2^478.
        let low = power_of_two(-450);
        let high = power_of_two(478);
        assert_eq!(prepare(&[low, high, 0.0], window), Prepared::AsIs);
    }

    /// Uniformly tiny or uniformly huge input is moved by one exact factor,
    /// and the move is reversible bit for bit.
    #[test]
    fn out_of_range_input_is_rescaled_without_loss() {
        let window = Window::INCIRCLE;
        for exponent in [-1074_i32, -1000, -700, -300, 300, 700, 1000, 1023] {
            // A value with its leading bit at 2^exponent and one more set bit
            // two places below, built from bits so that the subnormal end of
            // the range is reachable.
            let coordinate = if exponent < -1022 {
                f64::from_bits(1_u64 << (exponent + 1074).unsigned_abs())
            } else {
                power_of_two(exponent)
            };
            let coords = [coordinate, coordinate * 0.5, 0.0];
            match prepare(&coords, window) {
                Prepared::Scaled(factor) => {
                    assert_eq!(
                        factor.to_bits() & 0x000F_FFFF_FFFF_FFFF,
                        0,
                        "the factor must be a power of two: {factor:e}"
                    );
                    for &c in &coords {
                        let scaled = c * factor;
                        if c != 0.0 {
                            let (leading, lowest) = bit_span(scaled);
                            assert!(lowest >= window.q_min, "exponent {exponent}");
                            assert!(leading <= window.e_max, "exponent {exponent}");
                            // Exactness: dividing back gives the same bits.
                            assert_eq!((scaled / factor).to_bits(), c.to_bits());
                        }
                    }
                    // And the scaled input is now in the window.
                    let moved: [f64; 3] =
                        [coords[0] * factor, coords[1] * factor, coords[2] * factor];
                    assert_eq!(prepare(&moved, window), Prepared::AsIs);
                }
                other => panic!("exponent {exponent} gave {other:?}"),
            }
        }
    }

    /// Input spanning more bits than the window is wide cannot be rescued by
    /// any single factor.
    #[test]
    fn input_wider_than_the_window_is_out_of_window() {
        // `insphere`'s window is 370 bits wide; 2^-1074 to 2^900 is far wider.
        let wide = [f64::from_bits(1), power_of_two(900)];
        assert_eq!(prepare(&wide, Window::INSPHERE), Prepared::OutOfWindow);
        assert_eq!(prepare(&wide, Window::ORIENT2D), Prepared::OutOfWindow);

        // One bit past the width of `orient2d`'s window, 928 bits: the lowest
        // set bit at 2^-929 and the leading bit at 2^0.
        let just_too_wide = [f64::from_bits(1) * power_of_two(145), 1.0];
        assert_eq!(bit_span(just_too_wide[0]).1, -929);
        assert_eq!(
            prepare(&just_too_wide, Window::ORIENT2D),
            Prepared::OutOfWindow
        );

        // Exactly the width, 928 bits: the lowest set bit at 2^-928. One
        // factor fits, so this one is rescaled rather than refused.
        let exactly_wide = [f64::from_bits(1) * power_of_two(146), 1.0];
        assert_eq!(bit_span(exactly_wide[0]).1, -928);
        assert!(matches!(
            prepare(&exactly_wide, Window::ORIENT2D),
            Prepared::Scaled(_)
        ));
    }

    /// A value with many significant bits is placed by its lowest set bit,
    /// not by its magnitude: a large coordinate with a long significand has
    /// its low bits near underflow.
    #[test]
    fn the_scan_places_input_by_its_lowest_set_bit() {
        let window = Window::ORIENT2D;
        // Leading bit at 2^-400, lowest at 2^-452: in range by magnitude but
        // two bits below the window's floor.
        let value = (1.0 + f64::EPSILON) * power_of_two(-400);
        assert_eq!(bit_span(value), (-400, -452));
        match prepare(&[value], window) {
            Prepared::Scaled(factor) => {
                assert_eq!(factor, 4.0, "two bits up");
                assert_eq!(bit_span(value * factor), (-398, -450));
            }
            other => panic!("expected a rescale, got {other:?}"),
        }
    }
}
