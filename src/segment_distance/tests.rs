// src/segment_distance/tests.rs

//! Unit tests for the stage each fixed case is answered at, the case choice,
//! and the fixed cases on the fast filters' edge that hold their bounds in
//! place. The integration tests check every sign against the exact oracles,
//! while these tests can see the stages.

use super::*;

fn b(bits: u64) -> f64 {
    f64::from_bits(bits)
}

// ===========================================================================
// compare_point_distance: the stage table
// ===========================================================================

/// A fixed case: the points, `d`, the exact sign, and the answering stage.
type PointCase = ([f64; 2], [f64; 2], f64, Sign, Stage);

pub(crate) fn point_fixed_cases() -> [PointCase; 7] {
    let h = 2.0_f64.powi(500);
    let t = 2.0_f64.powi(-600);
    // An odd scale whose squares exceed 2^53 bits, so `3m`, `4m`, `5m`
    // square inexactly and the exact tie is invisible to the filter.
    let m = 268_435_457.0; // 2^28 + 1
    [
        // The zero filter: p == q, so the sign is −d²'s.
        (
            [0.5, -0.25],
            [0.5, -0.25],
            3.0,
            Sign::Negative,
            Stage::ZeroFilter,
        ),
        (
            [0.5, -0.25],
            [0.5, -0.25],
            0.0,
            Sign::Zero,
            Stage::ZeroFilter,
        ),
        // The fast filter: well separated.
        ([0.0, 0.0], [3.0, 4.0], 1.0, Sign::Positive, Stage::Filter),
        // The expansion stage: a tie moved by one ulp of d. The exact value
        // is −2d·2^-52·(1 + 2^-53)-ish, far inside the filter's band.
        (
            [0.0, 0.0],
            [3.0 * m, 4.0 * m],
            5.0 * m * (1.0 + f64::EPSILON),
            Sign::Negative,
            Stage::Expansion,
        ),
        // An exact tie with inexact squares, confirmed by the integer stage.
        (
            [0.0, 0.0],
            [3.0 * m, 4.0 * m],
            5.0 * m,
            Sign::Zero,
            Stage::Exact,
        ),
        // Wider than the window: 2^500 and 2^-600 in one call.
        ([0.0, 0.0], [h, t], h, Sign::Positive, Stage::Exact),
        // Subnormal coordinates, one ulp apart in distance, rescaled into
        // the window and answered by the expansion stage.
        (
            [0.0, 0.0],
            [b(3), 0.0],
            b(2),
            Sign::Positive,
            Stage::Expansion,
        ),
    ]
}

#[test]
fn point_fixed_cases_are_answered_at_their_stage() {
    for (i, (p, q, d, sign, stage)) in point_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            compare_point_distance_staged(p, q, d),
            Ok((sign, stage)),
            "case {i}"
        );
        // Only d² enters, so the sign of d is irrelevant.
        assert_eq!(compare_point_distance(p, q, -d), Ok(sign), "case {i}, -d");
    }
}

// ===========================================================================
// compare_segment_distance: the stage table and the case choice
// ===========================================================================

/// A fixed case: the point, the segment, `d`, the exact sign, the case, and
/// the stage.
type SegmentCase = ([f64; 2], [f64; 2], [f64; 2], f64, Sign, Case, Stage);

pub(crate) fn segment_fixed_cases() -> [SegmentCase; 10] {
    let h = 2.0_f64.powi(500);
    let t = 2.0_f64.powi(-600);
    let a = [0.0, 0.0];
    let bx = [10.0, 0.0];
    [
        // The nearest point is `a`, with p beyond it and the distance
        // |p − a|. An exact tie is never certified by a filter, so the
        // integer stage confirms the zero.
        (
            [-3.0, 4.0],
            a,
            bx,
            5.0,
            Sign::Zero,
            Case::NearA,
            Stage::Exact,
        ),
        (
            [-3.0, 4.0],
            a,
            bx,
            4.0,
            Sign::Positive,
            Case::NearA,
            Stage::Filter,
        ),
        // The foot exactly at `a` (`dot = 0`). The endpoint case answers,
        // and the interior formula would agree (see the documentation of
        // `compare_segment_distance`).
        (
            [0.0, 5.0],
            a,
            bx,
            6.0,
            Sign::Negative,
            Case::NearA,
            Stage::Filter,
        ),
        // The nearest point is `b`.
        (
            [13.0, -4.0],
            a,
            bx,
            5.0,
            Sign::Zero,
            Case::NearB,
            Stage::Exact,
        ),
        // A degenerate segment is the point `a`. Both dots are zero, so the
        // first case applies. The interior formula would wrongly give the
        // sign of cross² = 0 here, which is why the choice is `≤` rather
        // than `<`.
        (
            [3.0, 4.0],
            [1.0, 1.0],
            [1.0, 1.0],
            1.0,
            Sign::Positive,
            Case::NearA,
            Stage::Filter,
        ),
        (
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            0.0,
            Sign::Zero,
            Case::NearA,
            Stage::ZeroFilter,
        ),
        // Interior, decided by the fast filter.
        (
            [5.0, 3.0],
            a,
            bx,
            1.0,
            Sign::Positive,
            Case::Interior,
            Stage::Filter,
        ),
        // Interior, one part in 2^51 from the tie. The exact value is
        // −9·(2^-50 + 2^-102)·100, far inside the filter's band, and every
        // expansion operation is exact in the window.
        (
            [5.0, 3.0],
            a,
            bx,
            3.0 * (1.0 + 2.0_f64.powi(-51)),
            Sign::Negative,
            Case::Interior,
            Stage::Expansion,
        ),
        // Interior, an exact tie, with dist(p, ab) = 3 and d = 3.
        (
            [5.0, 3.0],
            a,
            bx,
            3.0,
            Sign::Zero,
            Case::Interior,
            Stage::Exact,
        ),
        // Interior, wider than the window.
        (
            [t, t],
            [h, -h],
            [h, h],
            0.0,
            Sign::Positive,
            Case::Interior,
            Stage::Exact,
        ),
    ]
}

#[test]
fn segment_fixed_cases_are_answered_at_their_case_and_stage() {
    for (i, (p, a, bq, d, sign, case, stage)) in segment_fixed_cases().into_iter().enumerate() {
        assert_eq!(
            compare_segment_distance_staged(p, a, bq, d),
            Ok((sign, case, stage)),
            "case {i}"
        );
        // The segment is undirected, and the sign of d is irrelevant. The
        // mirrored call swaps NearA and NearB, and the sign never moves.
        assert_eq!(
            compare_segment_distance(p, bq, a, -d),
            Ok(sign),
            "case {i}, mirrored"
        );
    }
}

/// Every case is reached, and the degenerate segment goes to `NearA`.
#[test]
fn every_case_is_reached_by_the_fixed_cases() {
    let mut seen = [false; 3];
    for (p, a, bq, d, ..) in segment_fixed_cases() {
        match compare_segment_distance_staged(p, a, bq, d) {
            Ok((_, Case::NearA, _)) => seen[0] = true,
            Ok((_, Case::NearB, _)) => seen[1] = true,
            Ok((_, Case::Interior, _)) => seen[2] = true,
            Err(e) => panic!("{e:?}"),
        }
    }
    assert_eq!(seen, [true; 3], "NearA, NearB, Interior");
}

// ===========================================================================
// The edge of the fast filters' bounds
// ===========================================================================

/// Fixed cases whose filter determinant lies **between half the bound and the
/// bound**. The filter must decline them (the expansion stage answers), and a
/// halved filter constant would certify them at the filter instead. A
/// search against the exact oracle found them, and each one's `det` is
/// between 0.55 and 0.95 of `A·mag`. They hold the constant's edge in place,
/// since loosening the bound moves these cases to the filter and fails this
/// test.
#[test]
fn point_filter_declines_the_cases_on_its_bounds_edge() {
    // (p, q, d, exact sign), near-Pythagorean ties a few ulps off, at
    // magnitudes where the squares round.
    let cases: [([f64; 2], [f64; 2], f64, Sign); 6] = [
        (
            [0.0, -2.0],
            [1_316_160_597.000_001, 1_754_880_794.000_001],
            2_193_600_995.000_000_5,
            Sign::Positive,
        ),
        (
            [-3.0, 3.0],
            [755_893_986.000_001, 1_007_858_655.0],
            1_259_823_315.0,
            Sign::Positive,
        ),
        (
            [-3.0, -3.0],
            [579_255_474.000_001, 772_340_633.0],
            965_425_795.000_001,
            Sign::Negative,
        ),
        (
            [-5.0, -2.0],
            [1_542_710_818.0, 2_056_947_762.0],
            2_571_184_705.000_001,
            Sign::Negative,
        ),
        (
            [2.0, 2.0],
            [1_057_228_847.0, 1_409_638_462.000_001],
            1_762_048_075.0,
            Sign::Positive,
        ),
        (
            [1.0, -5.0],
            [667_819_828.000_001, 890_426_431.000_001],
            1_113_033_045.000_001,
            Sign::Positive,
        ),
    ];
    for (i, (p, q, d, sign)) in cases.into_iter().enumerate() {
        assert_eq!(
            compare_point_distance_staged(p, q, d),
            Ok((sign, Stage::Expansion)),
            "edge case {i}: must be declined by the filter and answered exactly"
        );
    }
}

/// As above, for the segment form's interior filter.
#[test]
fn segment_filter_declines_the_cases_on_its_bounds_edge() {
    // (p, b, d, exact sign), with a = (0, 0), for axis-aligned segments
    // whose determinant is exact and lands between 0.55·A·mag and
    // 0.95·A·mag.
    let cases: [([f64; 2], [f64; 2], f64, Sign); 6] = [
        (
            [21_330.0, 1_024.0],
            [32_768.0, 0.0],
            1_024.000_000_000_001_4,
            Sign::Negative,
        ),
        (
            [107_982.5, 256.0],
            [524_288.0, 0.0],
            -256.000_000_000_000_04,
            Sign::Negative,
        ),
        (
            [1_277_080.0, 32_768.0],
            [2_097_152.0, 0.0],
            32_768.000_000_000_04,
            Sign::Negative,
        ),
        (
            [1_617.5, 65_536.0],
            [2_048.0, 0.0],
            65_536.000_000_000_1,
            Sign::Negative,
        ),
        (
            [1_141.25, 16_384.0],
            [4_096.0, 0.0],
            16_384.000_000_000_025,
            Sign::Negative,
        ),
        (
            [1_201.25, 65_536.0],
            [8_192.0, 0.0],
            -65_536.000_000_000_07,
            Sign::Negative,
        ),
    ];
    for (i, (p, bq, d, sign)) in cases.into_iter().enumerate() {
        assert_eq!(
            compare_segment_distance_staged(p, [0.0, 0.0], bq, d),
            Ok((sign, Case::Interior, Stage::Expansion)),
            "edge case {i}: must be declined by the filter and answered exactly"
        );
    }
}

// ===========================================================================
// Non-finite d
// ===========================================================================

#[test]
fn a_non_finite_d_is_rejected_before_any_stage() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(compare_point_distance([0.0, 0.0], [1.0, 1.0], bad).is_err());
        assert!(compare_segment_distance([0.0, 1.0], [0.0, 0.0], [2.0, 0.0], bad).is_err());
    }
}

// ===========================================================================
// Case and stage coverage on random input
// ===========================================================================

/// SplitMix64, written out so the test needs no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A small grid value in `{0, 0.5, …, 7.5}`, duplicate-heavy. One in
    /// eight is nudged by `2^-30`, so near-ties with a small non-zero value
    /// (the expansion stage's regime) occur alongside exact ones.
    #[allow(clippy::cast_precision_loss)]
    fn grid(&mut self) -> f64 {
        let v = (self.next() % 16) as f64 * 0.5;
        if self.next() % 8 == 0 {
            v + 2.0_f64.powi(-30)
        } else {
            v
        }
    }
}

/// Random grid-heavy input reaches every case, and every stage of each form,
/// so the fixed tables cannot silently stop covering a path.
#[test]
fn random_grid_input_reaches_every_case_and_stage() {
    let mut rng = Rng(0x2026_1030);
    let mut cases = [0_u32; 3];
    let mut stages = [0_u32; 4];
    for _ in 0..40_000 {
        let mut v = || rng.grid();
        let (p, a, b, d) = ([v(), v()], [v(), v()], [v(), v()], v());
        let (_, case, stage) = compare_segment_distance_staged(p, a, b, d).expect("finite");
        cases[match case {
            Case::NearA => 0,
            Case::NearB => 1,
            Case::Interior => 2,
        }] += 1;
        stages[match stage {
            Stage::ZeroFilter => 0,
            Stage::Filter => 1,
            Stage::Expansion => 2,
            Stage::Exact => 3,
        }] += 1;
        let (_, pt_stage) = compare_point_distance_staged(p, a, d).expect("finite");
        stages[match pt_stage {
            Stage::ZeroFilter => 0,
            Stage::Filter => 1,
            Stage::Expansion => 2,
            Stage::Exact => 3,
        }] += 1;
    }
    assert!(
        cases.iter().all(|&n| n > 50),
        "every case is reached: {cases:?}"
    );
    assert!(
        stages.iter().all(|&n| n > 50),
        "every stage is reached: {stages:?}"
    );
}
