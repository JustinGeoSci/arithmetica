// tests/support/mod.rs

//! Shared machinery for the boundary-seeking tests.
//!
//! # Why these tests search for the boundary
//!
//! A floating-point evaluation of a determinant can only get the sign wrong
//! where the true determinant is small compared with the rounding error of
//! evaluating it, that is, next to the surface on which the determinant
//! changes sign. A uniformly random input is almost never there, so a test
//! that only samples can pass many thousands of cases without once testing
//! the part of the predicate that decides hard signs. Placing every call next
//! to a boundary puts each one where an error in a stage's certificate would
//! show.
//!
//! So each case here is constructed, not sampled:
//!
//! 1. draw a random configuration in a chosen magnitude profile;
//! 2. pick one coordinate and find two values of it for which the exact
//!    determinant has opposite non-zero signs;
//! 3. bisect over the ordered bit patterns of that coordinate until the two
//!    values are adjacent floats, one on each side of the sign change;
//! 4. call the predicate on those two floats and on the three beyond each,
//!    and compare every result with the exact sign.
//!
//! # The oracle
//!
//! Every finite `f64` is an integer multiple of `2^-1074`. Multiplying each
//! coordinate by `2^1074` gives an integer, and the determinant of those
//! integers is the true determinant times a positive power of two, so its
//! sign is the true sign. The integers are evaluated in `num-bigint`. No
//! floating-point operation takes part, so no rounding argument is needed
//! for the expected values.
//!
//! The module is compiled into each test binary that declares it; a binary
//! that uses only part of it would otherwise warn about the rest.
// Test-code lint posture, stated rather than left implicit:
//
//   dead_code          — each test binary that declares this module uses only
//                        the part of it that it needs.
//   cast_*             — fixture construction and bit arithmetic, where the
//                        ranges are known by construction and asserted where
//                        they are not obvious.
//   float_cmp          — an exact-value assertion IS a float comparison.
#![allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::incompatible_msrv,
    clippy::float_cmp
)]

use std::cmp::Ordering;

use arithmetica::{Sign, incircle, insphere, orient2d, orient3d};
use num_bigint::BigInt;

// ===========================================================================
// The four predicates, addressed uniformly
// ===========================================================================

/// One of the four floating-point predicates.
///
/// A configuration is a flat slice of coordinates, point after point, with
/// the reference point last, in the argument order of the predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pred {
    /// `orient2d(pa, pb, pc)`: 3 points, 2 coordinates each.
    Orient2d,
    /// `incircle(pa, pb, pc, pd)`: 4 points, 2 coordinates each.
    Incircle,
    /// `orient3d(pa, pb, pc, pd)`: 4 points, 3 coordinates each.
    Orient3d,
    /// `insphere(pa, pb, pc, pd, pe)`: 5 points, 3 coordinates each.
    Insphere,
}

impl Pred {
    /// Number of points the predicate takes.
    pub const fn points(self) -> usize {
        match self {
            Self::Orient2d => 3,
            Self::Incircle | Self::Orient3d => 4,
            Self::Insphere => 5,
        }
    }

    /// Coordinates per point.
    pub const fn dim(self) -> usize {
        match self {
            Self::Orient2d | Self::Incircle => 2,
            Self::Orient3d | Self::Insphere => 3,
        }
    }

    /// Degree of the determinant as a polynomial in the coordinates.
    ///
    /// The product of `degree` coordinate differences is what underflows or
    /// overflows, so the `Small` profile is placed by this number.
    pub const fn degree(self) -> i64 {
        match self {
            Self::Orient2d => 2,
            Self::Orient3d => 3,
            Self::Incircle => 4,
            Self::Insphere => 5,
        }
    }

    /// The exactness window's limits for this predicate, as
    /// `(q_min, e_max)`: the smallest admissible exponent of a coordinate's
    /// lowest set bit, and the largest admissible exponent of a leading bit.
    ///
    /// Written out here rather than read from the crate, which keeps them
    /// private. That is deliberate: these are the published limits from
    /// `docs/exactness_window.md`, and a test that took them from the
    /// implementation could not notice the implementation changing them.
    pub const fn window(self) -> (i32, i32) {
        match self {
            Self::Orient2d => (-450, 478),
            Self::Orient3d => (-300, 318),
            Self::Incircle => (-225, 238),
            Self::Insphere => (-180, 190),
        }
    }

    /// The predicate's public name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Orient2d => "orient2d",
            Self::Incircle => "incircle",
            Self::Orient3d => "orient3d",
            Self::Insphere => "insphere",
        }
    }
}

/// Calls the predicate under test on a flat configuration.
///
/// Every configuration these tests build is finite, so `NonFinite` here is a
/// defect in the test itself and is reported as a panic.
pub fn call(pred: Pred, p: &[f64]) -> Sign {
    let result = match pred {
        Pred::Orient2d => orient2d([p[0], p[1]], [p[2], p[3]], [p[4], p[5]]),
        Pred::Incircle => incircle([p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]),
        Pred::Orient3d => orient3d(
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], p[8]],
            [p[9], p[10], p[11]],
        ),
        Pred::Insphere => insphere(
            [p[0], p[1], p[2]],
            [p[3], p[4], p[5]],
            [p[6], p[7], p[8]],
            [p[9], p[10], p[11]],
            [p[12], p[13], p[14]],
        ),
    };
    result.expect("the configuration is finite by construction")
}

// ===========================================================================
// Exact oracle
// ===========================================================================

/// `x · 2^1074` as an exact integer.
///
/// A normal `f64` is `sig · 2^(be - 1075)` with `sig` the 53-bit significand
/// (implicit bit included) and `be` the biased exponent, so the scaled value
/// is `sig · 2^(be - 1)`. A subnormal is `m · 2^-1074`, so the scaled value is
/// the 52-bit field `m` itself. Zero is the subnormal with `m = 0`.
pub fn exact_int(x: f64) -> BigInt {
    assert!(x.is_finite(), "the oracle is defined for finite input only");
    let bits = x.to_bits();
    let negative = (bits >> 63) != 0;
    let biased = (bits >> 52) & 0x7FF;
    let field = bits & 0x000F_FFFF_FFFF_FFFF;
    let magnitude = if biased == 0 {
        BigInt::from(field)
    } else {
        BigInt::from(field | (1_u64 << 52)) << (biased - 1)
    };
    if negative { -magnitude } else { magnitude }
}

fn sign_of_int(d: &BigInt) -> Sign {
    match d.cmp(&BigInt::ZERO) {
        Ordering::Greater => Sign::Positive,
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
    }
}

/// The exact sign of the predicate's determinant on a flat configuration.
///
/// Each arm is the determinant in the crate's documentation for that
/// predicate, written over differences relative to the last point.
pub fn oracle(pred: Pred, p: &[f64]) -> Sign {
    let v: Vec<BigInt> = p.iter().map(|&x| exact_int(x)).collect();
    let det = match pred {
        Pred::Orient2d => orient2d_det(&v),
        Pred::Incircle => incircle_det(&v),
        Pred::Orient3d => orient3d_det(&v),
        Pred::Insphere => insphere_det(&v),
    };
    sign_of_int(&det)
}

/// `(ax - cx)(by - cy) - (ay - cy)(bx - cx)`.
fn orient2d_det(v: &[BigInt]) -> BigInt {
    (&v[0] - &v[4]) * (&v[3] - &v[5]) - (&v[1] - &v[5]) * (&v[2] - &v[4])
}

/// The 3×3 lifted determinant of the differences relative to `pd`.
fn incircle_det(v: &[BigInt]) -> BigInt {
    let (adx, ady) = (&v[0] - &v[6], &v[1] - &v[7]);
    let (bdx, bdy) = (&v[2] - &v[6], &v[3] - &v[7]);
    let (cdx, cdy) = (&v[4] - &v[6], &v[5] - &v[7]);
    let alift = &adx * &adx + &ady * &ady;
    let blift = &bdx * &bdx + &bdy * &bdy;
    let clift = &cdx * &cdx + &cdy * &cdy;
    alift * (&bdx * &cdy - &bdy * &cdx)
        + blift * (&cdx * &ady - &cdy * &adx)
        + clift * (&adx * &bdy - &ady * &bdx)
}

/// The 3×3 determinant of the differences relative to `pd`.
fn orient3d_det(v: &[BigInt]) -> BigInt {
    let (adx, ady, adz) = (&v[0] - &v[9], &v[1] - &v[10], &v[2] - &v[11]);
    let (bdx, bdy, bdz) = (&v[3] - &v[9], &v[4] - &v[10], &v[5] - &v[11]);
    let (cdx, cdy, cdz) = (&v[6] - &v[9], &v[7] - &v[10], &v[8] - &v[11]);
    adz * (&bdx * &cdy - &bdy * &cdx)
        + bdz * (&cdx * &ady - &cdy * &adx)
        + cdz * (&adx * &bdy - &ady * &bdx)
}

/// The 4×4 lifted determinant of the differences relative to `pe`.
fn insphere_det(v: &[BigInt]) -> BigInt {
    let diff = |point: usize, axis: usize| &v[3 * point + axis] - &v[12 + axis];
    let (aex, aey, aez) = (diff(0, 0), diff(0, 1), diff(0, 2));
    let (bex, bey, bez) = (diff(1, 0), diff(1, 1), diff(1, 2));
    let (cex, cey, cez) = (diff(2, 0), diff(2, 1), diff(2, 2));
    let (dex, dey, dez) = (diff(3, 0), diff(3, 1), diff(3, 2));

    let ab = &aex * &bey - &bex * &aey;
    let bc = &bex * &cey - &cex * &bey;
    let cd = &cex * &dey - &dex * &cey;
    let da = &dex * &aey - &aex * &dey;
    let ac = &aex * &cey - &cex * &aey;
    let bd = &bex * &dey - &dex * &bey;

    let abc = &aez * &bc - &bez * &ac + &cez * &ab;
    let bcd = &bez * &cd - &cez * &bd + &dez * &bc;
    let cda = &cez * &da + &dez * &ac + &aez * &cd;
    let dab = &dez * &ab + &aez * &bd + &bez * &da;

    let alift = &aex * &aex + &aey * &aey + &aez * &aez;
    let blift = &bex * &bex + &bey * &bey + &bez * &bez;
    let clift = &cex * &cex + &cey * &cey + &cez * &cez;
    let dlift = &dex * &dex + &dey * &dey + &dez * &dez;

    dlift * abc - clift * dab + blift * cda - alift * bcd
}

// ===========================================================================
// Deterministic generator
// ===========================================================================

/// SplitMix64 (Steele, Lea & Flood 2014), written out so that the tests need
/// no dependency and a failure reproduces from its seed on any platform.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// A generator with the given seed.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits.
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `lo..=hi`. The slight modulo bias is irrelevant here: the
    /// draws choose magnitudes to visit, not a distribution to measure.
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let span = (hi - lo + 1).cast_unsigned();
        lo + (self.next() % span).cast_signed()
    }

    /// An index in `0..len`.
    pub fn index(&mut self, len: usize) -> usize {
        let len = u64::try_from(len).expect("a slice length fits in u64");
        usize::try_from(self.next() % len).expect("an index below a usize length fits in usize")
    }

    /// True with probability one half.
    pub fn coin(&mut self) -> bool {
        self.next() & 1 == 0
    }

    /// A float with a random sign and random lower bits whose leading bit is
    /// at `2^exponent`, clamped to the representable range `-1074..=1023`.
    ///
    /// Below `-1022` the result is subnormal: the leading bit is placed in
    /// the 52-bit field and only the bits beneath it are random.
    pub fn float_with_exponent(&mut self, exponent: i64) -> f64 {
        let exponent = exponent.clamp(-1074, 1023);
        let sign = (self.next() & 1) << 63;
        let field = self.next() & 0x000F_FFFF_FFFF_FFFF;
        if exponent < -1022 {
            let top = 1_u64 << (exponent + 1074).cast_unsigned();
            return f64::from_bits(sign | top | (field & (top - 1)));
        }
        let biased = (exponent + 1023).cast_unsigned();
        f64::from_bits(sign | (biased << 52) | field)
    }
}

// ===========================================================================
// Magnitude profiles
// ===========================================================================

/// Where in the `f64` range a configuration's coordinates lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Leading bits between `2^-5` and `2^3`: ordinary magnitudes.
    Unit,
    /// All points small enough that a product of `degree` differences lands
    /// near `2^-1022`, the edge of the subnormal range; in half the cases one
    /// point is pushed up to 80 bits lower still.
    Small,
    /// Each point either huge (`2^100` to `2^1000`) or tiny (`2^-100` to
    /// `2^-1040`), chosen by coin.
    Mixed,
    /// Each point anywhere from `2^-1070` to `2^1020`, independently.
    Any,
    /// On both edges of the predicate's exactness window at once: some points
    /// with a leading bit at `e_max`, the rest with a lowest set bit at
    /// `q_min`.
    ///
    /// No single `f64` can span a whole window, so this is as close as a
    /// configuration gets to using all of one: its span from lowest set bit
    /// to highest leading bit is the window's full width, which is where the
    /// window's two margins are thinnest and where an off-by-one in the scan
    /// would show.
    Edge,
}

impl Profile {
    /// The five profiles, in the order the test names use.
    pub const ALL: [Self; 5] = [Self::Unit, Self::Small, Self::Mixed, Self::Any, Self::Edge];

    /// Lower-case name used in reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::Small => "small",
            Self::Mixed => "mixed",
            Self::Any => "any",
            Self::Edge => "edge",
        }
    }

    /// One leading-bit exponent per point.
    fn point_exponents(self, pred: Pred, rng: &mut SplitMix64) -> Vec<i64> {
        let mut exponents = vec![0_i64; pred.points()];
        match self {
            Self::Unit => {
                for e in &mut exponents {
                    *e = rng.range(-3, 3);
                }
            }
            Self::Small => {
                let base = -1022 / pred.degree() + rng.range(-12, 6);
                for e in &mut exponents {
                    *e = base + rng.range(-4, 4);
                }
                let pushed = rng.index(pred.points());
                if rng.coin() {
                    exponents[pushed] -= rng.range(0, 80);
                }
            }
            Self::Mixed => {
                let high = rng.range(100, 1000);
                let low = -rng.range(100, 1040);
                for e in &mut exponents {
                    *e = if rng.coin() {
                        high + rng.range(-3, 0)
                    } else {
                        low + rng.range(-3, 3)
                    };
                }
            }
            Self::Any => {
                for e in &mut exponents {
                    *e = rng.range(-1070, 1020);
                }
            }
            Self::Edge => {
                // `generate` handles this profile itself; the exponents it
                // reports are the leading bits it produced.
                let (q_min, e_max) = pred.window();
                for e in &mut exponents {
                    *e = if rng.coin() {
                        i64::from(e_max)
                    } else {
                        i64::from(q_min)
                    };
                }
            }
        }
        exponents
    }
}

/// A random configuration in the profile, with the exponent chosen for each
/// point.
///
/// Each coordinate has its leading bit at most two places below its point's
/// exponent, and one coordinate in sixteen is exactly zero: zero coordinates
/// are common in real input and switch off individual products.
pub fn generate(pred: Pred, profile: Profile, rng: &mut SplitMix64) -> (Vec<f64>, Vec<i64>) {
    if profile == Profile::Edge {
        return generate_on_the_window_edges(pred, rng);
    }
    let exponents = profile.point_exponents(pred, rng);
    let mut config = vec![0.0_f64; pred.points() * pred.dim()];
    for (i, coordinate) in config.iter_mut().enumerate() {
        let below = rng.range(-2, 0);
        *coordinate = if rng.next().is_multiple_of(16) {
            0.0
        } else {
            rng.float_with_exponent(exponents[i / pred.dim()] + below)
        };
    }
    (config, exponents)
}

/// A configuration sitting on both edges of the predicate's window.
///
/// A point is placed in one of two ways, by coin:
///
/// - **At the top**: a leading bit at `2^e_max`, with random bits below. The
///   value is `2^e_max · m` for a significand `m` in `[1, 2)`.
/// - **At the bottom**: a lowest set bit at `2^q_min`. The value is
///   `2^q_min · n` for an odd integer `n` below `2^52`, which is exact
///   because `n` has at most 52 significant bits and `2^q_min` is a normal
///   power of two. Its leading bit is at most `q_min + 51`, well inside the
///   window.
///
/// With at least one point of each kind the configuration's span is the
/// window's full width and it needs no rescaling, which is the case the two
/// margins were derived for.
fn generate_on_the_window_edges(pred: Pred, rng: &mut SplitMix64) -> (Vec<f64>, Vec<i64>) {
    let (q_min, e_max) = pred.window();
    let bottom = f64::from_bits(((i64::from(q_min) + 1023) as u64) << 52);
    let mut config = vec![0.0_f64; pred.points() * pred.dim()];
    let mut exponents = vec![0_i64; pred.points()];
    // Alternate so that both kinds are always present, then let the coin
    // decide the rest.
    for point in 0..pred.points() {
        let at_the_top = if point < 2 { point == 0 } else { rng.coin() };
        exponents[point] = if at_the_top {
            i64::from(e_max)
        } else {
            i64::from(q_min)
        };
        for axis in 0..pred.dim() {
            let value = if at_the_top {
                rng.float_with_exponent(i64::from(e_max))
            } else {
                // An odd integer below 2^52, so the lowest set bit is the
                // factor's own and the product is exact.
                let odd = (rng.next() & 0x000F_FFFF_FFFF_FFFF) | 1;
                let magnitude = bottom * odd as f64;
                if rng.coin() { -magnitude } else { magnitude }
            };
            config[point * pred.dim() + axis] = value;
        }
    }
    (config, exponents)
}

/// The exponents of the leading bit and of the lowest set bit of a non-zero
/// finite `f64`, as `(leading, lowest)`.
///
/// Written independently of the crate's own `range::bit_span`, which is
/// private. Two implementations of the same reading is the point: the window
/// tests use this one to work out which rescalings the crate should accept.
pub fn bit_span(x: f64) -> (i32, i32) {
    assert!(x != 0.0 && x.is_finite());
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7FF) as i32;
    let field = bits & 0x000F_FFFF_FFFF_FFFF;
    if biased > 0 {
        let significand = field | (1_u64 << 52);
        (
            biased - 1023,
            biased - 1075 + significand.trailing_zeros() as i32,
        )
    } else {
        (
            field.ilog2() as i32 - 1074,
            field.trailing_zeros() as i32 - 1074,
        )
    }
}

/// The span of a whole configuration: the largest leading-bit exponent and
/// the smallest lowest-set-bit exponent over its non-zero coordinates, or
/// `None` if every coordinate is zero.
pub fn config_span(config: &[f64]) -> Option<(i32, i32)> {
    let mut highest = i32::MIN;
    let mut lowest = i32::MAX;
    for &c in config {
        if c != 0.0 {
            let (leading, low) = bit_span(c);
            highest = highest.max(leading);
            lowest = lowest.min(low);
        }
    }
    (highest != i32::MIN).then_some((highest, lowest))
}

// ===========================================================================
// Boundary search
// ===========================================================================

/// Maps a finite `f64` to an integer so that the integer order is the
/// numeric order and adjacent floats map to adjacent integers.
///
/// Non-negative floats map to their bit pattern. Negative floats, whose bit
/// patterns grow as the value falls, are reflected. Both zeros map to 0.
pub fn ordinal(x: f64) -> i64 {
    let bits = x.to_bits().cast_signed();
    if bits < 0 { i64::MIN - bits } else { bits }
}

/// The inverse of [`ordinal`]. An ordinal past the largest finite value
/// yields an infinity or a NaN, which callers skip.
pub fn from_ordinal(k: i64) -> f64 {
    let bits = if k < 0 { i64::MIN - k } else { k };
    f64::from_bits(bits.cast_unsigned())
}

/// Finds two adjacent floats for coordinate `index` between which the exact
/// sign changes, returned as ordinals `(below, above)` with `above = below + 1`.
///
/// The bracket is `-m` and `+m` for a magnitude `m` at a growing distance
/// above the point's own exponent. A bracket is accepted only when the exact
/// signs at its ends are non-zero and opposite. Not every configuration has
/// such a coordinate (the determinant may not depend on it, or may keep its
/// sign); those return `None` and are not counted.
///
/// Bisection keeps the invariant "the sign at `below` is the sign at the
/// bracket's lower end", so a zero of the determinant that happens to be
/// exactly representable ends up at `above`. The neighbors on both sides are
/// tested either way.
fn find_boundary(
    pred: Pred,
    config: &[f64],
    index: usize,
    point_exponent: i64,
    rng: &mut SplitMix64,
) -> Option<(i128, i128)> {
    let mut probe = config.to_vec();
    for widen in [2_i64, 8, 40, 400, 3000] {
        let magnitude = rng
            .float_with_exponent((point_exponent + widen).min(1023))
            .abs();
        probe[index] = -magnitude;
        let sign_below = oracle(pred, &probe);
        probe[index] = magnitude;
        let sign_above = oracle(pred, &probe);
        if sign_below == Sign::Zero || sign_above == Sign::Zero || sign_below == sign_above {
            continue;
        }
        let mut below = i128::from(ordinal(-magnitude));
        let mut above = i128::from(ordinal(magnitude));
        while above - below > 1 {
            let middle = below + (above - below) / 2;
            probe[index] = from_ordinal(i64::try_from(middle).expect("between two i64 ordinals"));
            if oracle(pred, &probe) == sign_below {
                below = middle;
            } else {
                above = middle;
            }
        }
        return Some((below, above));
    }
    None
}

/// What a boundary run found.
#[derive(Debug, Default)]
pub struct Report {
    /// Label for messages, such as `orient2d/small`.
    pub label: String,
    /// Configurations for which a sign boundary was found.
    pub boundaries: u64,
    /// Predicate calls compared with the oracle.
    pub calls: u64,
    /// Calls that returned the opposite non-zero sign or a non-zero sign for
    /// an exactly zero determinant.
    pub wrong_nonzero: u64,
    /// Calls that returned `Zero` for a non-zero determinant.
    pub wrong_zero: u64,
    /// The first mismatch, formatted so it can be pasted into a test.
    pub first: Option<String>,
}

impl Report {
    /// Total mismatches.
    pub const fn wrong(&self) -> u64 {
        self.wrong_nonzero + self.wrong_zero
    }

    /// Adds another run's counts, keeping the earliest mismatch.
    pub fn absorb(&mut self, other: Self) {
        self.boundaries += other.boundaries;
        self.calls += other.calls;
        self.wrong_nonzero += other.wrong_nonzero;
        self.wrong_zero += other.wrong_zero;
        if self.first.is_none() {
            self.first = other.first;
        }
    }

    /// One summary line, printed by the deep tier whatever the outcome.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} boundaries, {} calls, {} wrong non-zero, {} wrong zero",
            self.label, self.boundaries, self.calls, self.wrong_nonzero, self.wrong_zero
        )
    }

    /// Panics unless every call agreed with the oracle.
    ///
    /// A run that found no boundary at all also fails: it compared nothing,
    /// and a comparison of nothing must not pass.
    pub fn assert_clean(&self) {
        assert!(
            self.calls > 0,
            "{}: no sign boundary was found, so nothing was compared",
            self.label
        );
        assert!(
            self.wrong() == 0,
            "{}\nfirst mismatch:\n{}",
            self.summary(),
            self.first.as_deref().unwrap_or("(not recorded)")
        );
    }

    /// Compares one call with the oracle and records the outcome.
    pub fn check(&mut self, pred: Pred, config: &[f64]) {
        let want = oracle(pred, config);
        let got = call(pred, config);
        self.calls += 1;
        if got == want {
            return;
        }
        if got == Sign::Zero {
            self.wrong_zero += 1;
        } else {
            self.wrong_nonzero += 1;
        }
        if self.first.is_none() {
            self.first = Some(describe(pred, config, got, want));
        }
    }
}

/// A mismatch as bit patterns, one point per line, with decimal values as
/// comments. Bit patterns are the only form that reproduces the input.
fn describe(pred: Pred, config: &[f64], got: Sign, want: Sign) -> String {
    let mut lines = vec![format!(
        "{}: got {got:?}, exact sign is {want:?}",
        pred.name()
    )];
    for (i, point) in config.chunks(pred.dim()).enumerate() {
        let bits: Vec<String> = point
            .iter()
            .map(|c| format!("b({:#018x})", c.to_bits()))
            .collect();
        let values: Vec<String> = point.iter().map(|c| format!("{c:e}")).collect();
        lines.push(format!(
            "  p{i} = [{}]  // [{}]",
            bits.join(", "),
            values.join(", ")
        ));
    }
    lines.join("\n")
}

/// Runs `cases` boundary searches for one predicate in one profile.
///
/// For each boundary found, the predicate is called on the two adjacent
/// floats that straddle the sign change and on the three floats beyond each:
/// eight calls, each compared with the oracle.
pub fn run_boundary(pred: Pred, profile: Profile, cases: u64, seed: u64) -> Report {
    let mut rng = SplitMix64::new(seed);
    let mut report = Report {
        label: format!("{}/{}", pred.name(), profile.name()),
        ..Report::default()
    };
    for _ in 0..cases {
        let (config, exponents) = generate(pred, profile, &mut rng);
        let index = rng.index(config.len());
        let point_exponent = exponents[index / pred.dim()];
        let Some((below, above)) = find_boundary(pred, &config, index, point_exponent, &mut rng)
        else {
            continue;
        };
        report.boundaries += 1;
        let mut neighbor = config;
        for k in (below - 3)..=(above + 3) {
            let value = from_ordinal(i64::try_from(k).expect("three steps from an i64 ordinal"));
            if !value.is_finite() {
                continue;
            }
            neighbor[index] = value;
            report.check(pred, &neighbor);
        }
    }
    report
}

// ===========================================================================
// Fixtures for the `orient2d` zero filter
// ===========================================================================

// Shared by `tests/zero_filter.rs`, which checks the signs, and
// `tests/zero_filter_counters.rs`, which checks the stage that produced
// them, so that the two files can never drift onto different inputs.
//
// Every coordinate below is a dyadic rational with few bits, so it is
// exactly the `f64` it is written as.

/// Three 2D points in the argument order of `orient2d`.
pub type Triple = [[f64; 2]; 3];

/// The six argument orders of a triple: three rotations, which keep the
/// orientation, and three swaps, which negate it. Zero is its own negative,
/// so an exactly degenerate triple must be `Zero` in all six.
pub fn all_orders(t: Triple) -> [Triple; 6] {
    let [a, b, c] = t;
    [
        [a, b, c],
        [b, c, a],
        [c, a, b],
        [b, a, c],
        [a, c, b],
        [c, b, a],
    ]
}

const P: [f64; 2] = [1_400_000.25, 900_000.5];
const Q: [f64; 2] = [1_400_017.75, 899_990.125];

/// A repeated point in each of the three positions, and all three equal.
///
/// Two equal points make two rows of the orientation determinant equal, so
/// the determinant is zero whatever the third point is.
pub fn duplicate_triples() -> [(&'static str, Triple); 4] {
    [
        ("pa == pb", [P, P, Q]),
        ("pa == pc", [P, Q, P]),
        ("pb == pc", [Q, P, P]),
        ("pa == pb == pc", [P, P, P]),
    ]
}

/// Three distinct points on a horizontal line and three on a vertical line.
///
/// On a horizontal line every y-difference is exactly zero, so both products
/// of the determinant have a zero factor; a vertical line does the same with
/// the x-differences.
pub fn axis_aligned_triples() -> [(&'static str, Triple); 2] {
    [
        (
            "horizontal",
            [
                [1_400_000.25, 900_000.5],
                [1_400_017.75, 900_000.5],
                [1_399_990.0, 900_000.5],
            ],
        ),
        (
            "vertical",
            [
                [1_400_000.25, 900_000.5],
                [1_400_000.25, 899_990.125],
                [1_400_000.25, 900_123.0],
            ],
        ),
    ]
}

/// Three distinct points on a line that is neither horizontal nor vertical:
/// `p(k) = [1_400_000 + k/1024, 900_000 + 3k/1024]` for `k = 0, 1, 5`.
///
/// `p(5) - p(0) = 5·(p(1) - p(0))` exactly, so the points are collinear as a
/// statement about rationals. No coordinate difference is zero and no two
/// points are equal, so the zero filter must not answer this one; the exact
/// integer stage still has to.
pub fn general_collinear_triple() -> Triple {
    let p = |k: f64| [1_400_000.0 + k / 1024.0, 900_000.0 + 3.0 * k / 1024.0];
    [p(0.0), p(1.0), p(5.0)]
}

// ===========================================================================
// Exactly degenerate constructions
// ===========================================================================

// Each builds a configuration whose determinant is zero as a statement about
// integers, not as an observation about floating-point arithmetic, and then
// multiplies the whole configuration by one power of two, which is exact and
// keeps it degenerate. These are the inputs a fast path is most tempted to
// answer `Zero` on, and their one-float neighbors are the inputs on which
// `Zero` is wrong by the smallest possible margin.
//
// Shared by `tests/boundary.rs` (the degenerate tier) and
// `tests/window.rs` (scale invariance), which needs input that reaches the
// deeper stages.

/// An exact power-of-two scale, `2^k` with `k` in `-30..=30`.
pub fn power_of_two_scale(rng: &mut SplitMix64) -> f64 {
    2.0_f64.powi(i32::try_from(rng.range(-30, 30)).expect("a small exponent"))
}

/// Small integers as exact floats.
pub fn int(rng: &mut SplitMix64, lo: i64, hi: i64) -> f64 {
    rng.range(lo, hi) as f64
}

/// Three points on one line: `a`, `a + s·d` and `a + t·d` for an integer
/// point `a`, an integer direction `d ≠ 0`, and distinct non-zero integers
/// `s` and `t`. Two differences that are multiples of one vector have a zero
/// cross product.
pub fn collinear(rng: &mut SplitMix64) -> Vec<f64> {
    let (x, y) = (int(rng, -1000, 1000), int(rng, -1000, 1000));
    let (dx, dy) = loop {
        let d = (int(rng, -50, 50), int(rng, -50, 50));
        if d != (0.0, 0.0) {
            break d;
        }
    };
    let s = int(rng, 1, 20);
    let t = -int(rng, 1, 20);
    vec![x, y, x + s * dx, y + s * dy, x + t * dx, y + t * dy]
}

/// Four points on one circle: four of the eight integer points
/// `(±a, ±b)`, `(±b, ±a)` about an integer center, with `0 < a < b`. All
/// eight are at squared distance `a² + b²` from the center.
pub fn cocircular(rng: &mut SplitMix64) -> Vec<f64> {
    let (cx, cy) = (int(rng, -1000, 1000), int(rng, -1000, 1000));
    let a = int(rng, 1, 30);
    let b = a + int(rng, 1, 30);
    let mut ring = vec![
        (a, b),
        (-a, b),
        (a, -b),
        (-a, -b),
        (b, a),
        (-b, a),
        (b, -a),
        (-b, -a),
    ];
    let mut config = Vec::with_capacity(8);
    for _ in 0..4 {
        let (x, y) = ring.swap_remove(rng.index(ring.len()));
        config.extend([cx + x, cy + y]);
    }
    config
}

/// Four points in one plane: `p + s·u + t·v` for an integer point `p`,
/// integer vectors `u` and `v`, and integer `s` and `t`. Three differences
/// in the span of two vectors have a zero triple product.
pub fn coplanar(rng: &mut SplitMix64) -> Vec<f64> {
    let p = [
        int(rng, -1000, 1000),
        int(rng, -1000, 1000),
        int(rng, -1000, 1000),
    ];
    let u = [int(rng, -30, 30), int(rng, -30, 30), int(rng, -30, 30)];
    let v = [int(rng, -30, 30), int(rng, -30, 30), int(rng, -30, 30)];
    let mut config = Vec::with_capacity(12);
    for _ in 0..4 {
        let (s, t) = (int(rng, -10, 10), int(rng, -10, 10));
        for axis in 0..3 {
            config.push(p[axis] + s * u[axis] + t * v[axis]);
        }
    }
    config
}

/// Five points on one sphere: five of the forty-eight integer points given
/// by the permutations and sign changes of `(a, b, c)` about an integer
/// center, with `0 < a < b < c`. All are at squared distance `a² + b² + c²`.
pub fn cospherical(rng: &mut SplitMix64) -> Vec<f64> {
    let center = [
        int(rng, -1000, 1000),
        int(rng, -1000, 1000),
        int(rng, -1000, 1000),
    ];
    let a = int(rng, 1, 12);
    let b = a + int(rng, 1, 12);
    let c = b + int(rng, 1, 12);
    let orders = [
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ];
    let mut shell = Vec::with_capacity(48);
    for order in orders {
        for signs in 0..8_u32 {
            let flip = |axis: u32| if signs >> axis & 1 == 0 { 1.0 } else { -1.0 };
            shell.push([order[0] * flip(0), order[1] * flip(1), order[2] * flip(2)]);
        }
    }
    let mut config = Vec::with_capacity(15);
    for _ in 0..5 {
        let point = shell.swap_remove(rng.index(shell.len()));
        for axis in 0..3 {
            config.push(center[axis] + point[axis]);
        }
    }
    config
}
