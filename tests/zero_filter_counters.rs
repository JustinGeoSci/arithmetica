// tests/zero_filter_counters.rs

//! Which stage answers an exactly degenerate `orient2d` call. Compiled only
//! with `--features stage-counters`.
//!
//! `tests/zero_filter.rs` checks that the answers are right. This file checks
//! what they cost: a repeated point or an axis-aligned triple should be
//! answered by the zero filter, without the exact integer stage, which is
//! about 125 times slower than the fast path. Without the filter every exact
//! zero would reach Stage 4, because no later stage certifies a zero.
//!
//! The first two tests check that the filter answers the triples it is meant
//! for. The third is a control: a collinear triple in a general direction is
//! not the filter's to answer, and must still reach Stage 4.
//!
//! # Why the tests are serialized
//!
//! The counters are process-wide and the test harness runs the tests of one
//! file on several threads. Each test resets the counters, makes one call,
//! and reads them back, so two tests interleaving would read each other's
//! calls. A mutex makes each reset-call-read sequence atomic with respect to
//! the others. Test files are separate processes, so nothing outside this
//! file can interfere.
// Test-code lint posture, stated rather than left implicit:
//
//   float_cmp          — an exact-value assertion IS a float comparison.
//   unreadable_literal — coordinates are written as the values they are.
//   unwrap_used        — in a test, a panic is the failure signal.
#![allow(clippy::float_cmp, clippy::unreadable_literal, clippy::unwrap_used)]
#![cfg(feature = "stage-counters")]

mod support;

use std::sync::{Mutex, MutexGuard, PoisonError};

use arithmetica::counters::{Counts, Predicate, reset, snapshot};
use arithmetica::{Sign, orient2d};
use support::{
    Triple, all_orders, axis_aligned_triples, duplicate_triples, general_collinear_triple,
};

static SERIAL: Mutex<()> = Mutex::new(());

/// Takes the lock even if an earlier test panicked while holding it. A
/// failed assertion in one test must not turn into a poisoned-lock failure
/// in the control.
fn serialize() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One `orient2d` call with the counters reset before it and read after it.
/// The caller holds the lock.
fn counted(t: Triple) -> (Sign, Counts) {
    reset();
    let sign = orient2d(t[0], t[1], t[2]).expect("the fixtures are finite");
    (sign, snapshot(Predicate::Orient2d))
}

/// Runs every order of every fixture and collects the calls that reached
/// Stage 4, so that a failure reports all of them and not only the first.
fn assert_answered_without_stage_four(fixtures: &[(&'static str, Triple)]) {
    let _guard = serialize();
    let mut calls = 0_u32;
    let mut reached_stage_four = Vec::new();
    for &(name, triple) in fixtures {
        for (i, order) in all_orders(triple).into_iter().enumerate() {
            let (sign, counts) = counted(order);
            assert_eq!(sign, Sign::Zero, "{name}, order {i}");
            calls += 1;
            if counts.exact != 0 || counts.exact_zero != 0 || counts.fast != 1 {
                reached_stage_four.push(format!("{name}, order {i}: {counts:?}"));
            }
        }
    }
    assert!(
        reached_stage_four.is_empty(),
        "{} of {calls} exactly degenerate calls were not answered on the fast path:\n{}",
        reached_stage_four.len(),
        reached_stage_four.join("\n")
    );
}

/// A repeated point, in each position and each argument order, is answered
/// as one fast-path call and no Stage 4 call.
#[test]
fn duplicate_point_is_answered_without_stage_four() {
    assert_answered_without_stage_four(&duplicate_triples());
}

/// So is a triple on a horizontal or a vertical line.
#[test]
fn axis_aligned_triple_is_answered_without_stage_four() {
    assert_answered_without_stage_four(&axis_aligned_triples());
}

/// Control: a collinear triple in a general direction has no zero difference
/// and no repeated point, so the filter must leave it alone. Its floating-
/// point determinant is not certifiably non-zero either, so the call ends in
/// Stage 4, which returns `Zero`: one `Exact`, one `ExactZero`, no `Fast`.
#[test]
fn general_collinear_triple_still_reaches_stage_four() {
    let _guard = serialize();
    for (i, order) in all_orders(general_collinear_triple())
        .into_iter()
        .enumerate()
    {
        let (sign, counts) = counted(order);
        assert_eq!(sign, Sign::Zero, "order {i}");
        assert_eq!(counts.exact, 1, "order {i}: {counts:?}");
        assert_eq!(counts.exact_zero, 1, "order {i}: {counts:?}");
        assert_eq!(counts.fast, 0, "order {i}: {counts:?}");
    }
}
