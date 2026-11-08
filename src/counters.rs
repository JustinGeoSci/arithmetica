// src/counters.rs

//! Optional counters recording how often each predicate reaches its exact
//! fallback.
//!
//! Enabled by the `stage-counters` feature. With the feature off, [`record`]
//! is an empty inline function and the counters do not exist, so the
//! instrumented predicates compile to the same code as uninstrumented ones.
//!
//! # Why this exists
//!
//! Every test in this crate checks the *sign* a predicate returns, and that
//! sign is correct whichever stage produces it, which is the adaptive evaluation
//! working as designed. As a consequence nothing else detects a change that
//! moves work from the fast path to the exact path. An error-bound
//! adjustment, or a guard added for an input that fuzzing found, can
//! multiply exact-path entries by a large factor while every test still
//! passes. These counters make that visible.
//!
//! # Two rules
//!
//! 1. **Observability only.** No behavior in this crate may read a counter. A
//!    predicate that branches on its own statistics is not a predicate.
//! 2. **Not a correctness signal.** A high exact-path rate means the inputs are
//!    degenerate rather than that anything is wrong. Exactness is the
//!    product, not a failure mode.
//!
//! # What is and is not distinguished
//!
//! [`Outcome::Exact`] marks entry to the Stage 4 exact-integer path, and
//! [`Outcome::Fast`] covers Stages 1 through 3 together. [`Outcome::ExactZero`]
//! additionally marks the Stage 4 calls whose result was `Zero`, recorded
//! where Stage 4 returns, outside the `*_adapt` bodies.
//!
//! Separating Stage 2 from Stage 3 would mean instrumenting the bodies of
//! `orient2d_adapt`, `incircle_adapt`, `orient3d_adapt` and `insphere_adapt`,
//! which are direct ports of Shewchuk's code and are kept line-for-line
//! comparable with it. The fast-versus-exact split is the number that
//! predicts cost, and the finer split is not.
//!
//! # Storage: process-global
//!
//! The counters are **process-global atomics**, one set for the whole process,
//! shared by every thread. They are 64-bit wherever the target has 64-bit
//! atomics. On a target without them (for example `thumbv7em-none-eabi`)
//! they are 32-bit and **saturate**, so a count that reaches `u32::MAX`
//! (4,294,967,295) stays there rather than wrapping, and a reading of
//! exactly `u32::MAX` on such a target means "at least that many". The
//! feature needs atomic read-modify-write, and on a target with none (for
//! example `thumbv6m-none-eabi`) it is refused at compile time.
//!
//! A reader that wants a single call sequence's counts must serialize the
//! calls it measures (as this crate's own counter tests do, under a lock),
//! or run them in a process of their own. Per-thread storage would need
//! `std`, which this `no_std` crate does not take. A per-thread variant, if
//! ever offered, would be a new opt-in feature rather than a change to this
//! one.
//!
//! # Aggregates, not attribution
//!
//! These are global counters, so under concurrency they report totals, never
//! which call used which stage. Per-call attribution would be a different
//! feature, a returning variant such as
//! `orient2d_with_stage(..) -> (Result<Sign, NonFinite>, Stage)`, and must
//! not be grown out of this one.

/// Which predicate a record belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicate {
    /// 2D orientation.
    Orient2d,
    /// 3D orientation.
    Orient3d,
    /// 2D in-circle.
    Incircle,
    /// 3D in-sphere.
    Insphere,
}

impl Predicate {
    /// Index into the counter arrays. Only reachable with `stage-counters`.
    #[cfg(feature = "stage-counters")]
    const fn index(self) -> usize {
        match self {
            Self::Orient2d => 0,
            Self::Orient3d => 1,
            Self::Incircle => 2,
            Self::Insphere => 3,
        }
    }

    /// All four, in index order, for iterating a snapshot.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::Orient2d,
            Self::Orient3d,
            Self::Incircle,
            Self::Insphere,
        ]
    }

    /// Lowercase name, for reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Orient2d => "orient2d",
            Self::Orient3d => "orient3d",
            Self::Incircle => "incircle",
            Self::Insphere => "insphere",
        }
    }
}

/// How a predicate call ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Resolved by Stages 1–3, native `f64` or expansion arithmetic.
    Fast,
    /// Fell through to the Stage 4 exact-integer path.
    Exact,
    /// A Stage 4 call (already counted as [`Outcome::Exact`]) whose exact
    /// result was `Zero`, an exactly degenerate input. It is recorded in
    /// addition to `Exact`, never instead of it, so `exact_zero <= exact` and
    /// the totals are unchanged. It measures how much of the exact path is
    /// spent confirming exact zeros.
    ExactZero,
    /// Rejected before any stage ran: an input coordinate was not finite.
    NonFinite,
}

impl Outcome {
    /// Index into the counter arrays. Only reachable with `stage-counters`.
    #[cfg(feature = "stage-counters")]
    const fn index(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Exact => 1,
            Self::NonFinite => 2,
            Self::ExactZero => 3,
        }
    }
}

// ===========================================================================
// Feature off: everything compiles away
// ===========================================================================

/// Record one predicate outcome.
///
/// Without the `stage-counters` feature this is an empty inline function and
/// the call vanishes at compile time.
#[cfg(not(feature = "stage-counters"))]
#[inline(always)]
pub fn record(_predicate: Predicate, _outcome: Outcome) {}

// ===========================================================================
// Feature on
// ===========================================================================

// The counters need atomic read-modify-write. A target with neither 64-bit
// nor 32-bit read-modify-write atomics (for example `thumbv6m-none-eabi`)
// cannot record, so the feature is refused there with a reason rather than
// with an unresolved-import error.
#[cfg(all(
    feature = "stage-counters",
    not(target_has_atomic = "64"),
    not(target_has_atomic = "32")
))]
compile_error!(
    "the `stage-counters` feature needs atomic read-modify-write instructions, \
     which this target does not have; build without the feature"
);

#[cfg(feature = "stage-counters")]
mod enabled {
    use super::{Outcome, Predicate};
    use core::sync::atomic::Ordering;

    /// The counter cell, an `AtomicU64` wherever the target has 64-bit
    /// atomics.
    #[cfg(target_has_atomic = "64")]
    type Cell = core::sync::atomic::AtomicU64;

    /// The counter cell on a target without 64-bit atomics (for example
    /// `thumbv7em-none-eabi`), an `AtomicU32` that saturates at `u32::MAX`
    /// (see [`record`]).
    #[cfg(not(target_has_atomic = "64"))]
    type Cell = core::sync::atomic::AtomicU32;

    /// `[predicate][outcome]`, 4 predicates × 4 outcomes.
    ///
    /// `Relaxed` is the right ordering, because these counts order nothing
    /// and synchronize nothing, and anything stronger would buy only cost.
    /// The atomic cell is not `Copy`, so the array-repeat form cannot build
    /// this. Sixteen explicit cells (four per row) keep the static
    /// `const`-initialized with no lazy-init machinery and no dependency.
    #[allow(clippy::declare_interior_mutable_const)]
    const ROW: [Cell; 4] = [Cell::new(0), Cell::new(0), Cell::new(0), Cell::new(0)];
    static COUNTS: [[Cell; 4]; 4] = [ROW, ROW, ROW, ROW];

    /// Increment one counter.
    ///
    /// With 64-bit cells the increment is a plain `fetch_add`, since a `u64`
    /// count of predicate calls cannot wrap in any realistic run. With 32-bit
    /// cells it is `narrow::add_saturating`, because 2^32 calls is a
    /// realistic run and a count that wrapped to a small number would be a
    /// silent wrong number. The count stops at `u32::MAX` instead, where it
    /// is recognizably capped.
    #[inline]
    pub fn record(predicate: Predicate, outcome: Outcome) {
        let cell = &COUNTS[predicate.index()][outcome.index()];
        #[cfg(target_has_atomic = "64")]
        cell.fetch_add(1, Ordering::Relaxed);
        #[cfg(not(target_has_atomic = "64"))]
        narrow::add_saturating(cell);
    }

    /// Read one counter, widened to `u64` where the cell is 32-bit (the
    /// conversion is the identity where it is 64-bit, hence the allow).
    #[inline]
    #[allow(clippy::useless_conversion)]
    pub fn read(predicate: Predicate, outcome: Outcome) -> u64 {
        u64::from(COUNTS[predicate.index()][outcome.index()].load(Ordering::Relaxed))
    }

    /// Zero every counter.
    pub fn reset() {
        for row in &COUNTS {
            for cell in row {
                cell.store(0, Ordering::Relaxed);
            }
        }
    }

    /// The 32-bit cell's increment. Compiled on targets without 64-bit
    /// atomics, where it is the one used, and in this crate's tests, so the
    /// saturation is tested on every host.
    #[cfg(any(test, not(target_has_atomic = "64")))]
    pub(super) mod narrow {
        use core::sync::atomic::{AtomicU32, Ordering};

        /// Adds one, unless the cell is at `u32::MAX`, where it stays.
        ///
        /// `fetch_update` stores the closure's result by compare-and-swap,
        /// retrying under contention. `checked_add` returns `None` at the
        /// maximum, and `fetch_update` then stores nothing. Concurrent
        /// increments are therefore never lost below the cap, and the cap is
        /// never passed.
        #[inline]
        pub(crate) fn add_saturating(cell: &AtomicU32) {
            let _ = cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1));
        }
    }
}

#[cfg(feature = "stage-counters")]
#[cfg_attr(docsrs, doc(cfg(feature = "stage-counters")))]
pub use enabled::record;

/// Counts for one predicate at one instant.
#[cfg(feature = "stage-counters")]
#[cfg_attr(docsrs, doc(cfg(feature = "stage-counters")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Calls resolved by Stages 1–3.
    pub fast: u64,
    /// Calls that reached the Stage 4 exact-integer path.
    pub exact: u64,
    /// Calls rejected for a non-finite input coordinate.
    pub non_finite: u64,
    /// Of the `exact` calls, those whose result was `Zero`, a subset of
    /// `exact`, not a fourth category.
    pub exact_zero: u64,
}

#[cfg(feature = "stage-counters")]
impl Counts {
    /// Total recorded calls.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.fast + self.exact + self.non_finite
    }

    /// Share of *geometrically answered* calls that needed the exact path, in
    /// `[0, 1]`.
    ///
    /// Non-finite rejections are excluded from the denominator, because they
    /// are out-of-domain inputs rather than degenerate ones, and counting
    /// them would dilute the rate this number exists to report. Returns
    /// `None` when no call has been answered.
    #[must_use]
    pub fn exact_rate(self) -> Option<f64> {
        let answered = self.fast + self.exact;
        if answered == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        Some(self.exact as f64 / answered as f64)
    }
}

/// Current counts for one predicate.
///
/// A snapshot taken while other threads are calling predicates may catch the
/// four fields at slightly different instants, since the counters are
/// independent and not read atomically as a group.
#[cfg(feature = "stage-counters")]
#[cfg_attr(docsrs, doc(cfg(feature = "stage-counters")))]
#[must_use]
pub fn snapshot(predicate: Predicate) -> Counts {
    Counts {
        fast: enabled::read(predicate, Outcome::Fast),
        exact: enabled::read(predicate, Outcome::Exact),
        non_finite: enabled::read(predicate, Outcome::NonFinite),
        exact_zero: enabled::read(predicate, Outcome::ExactZero),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

/// Set every counter back to zero.
///
/// Intended for a measurement run that wants a clean baseline. It is global,
/// so a concurrent caller's counts are discarded too.
#[cfg(feature = "stage-counters")]
#[cfg_attr(docsrs, doc(cfg(feature = "stage-counters")))]
pub fn reset() {
    enabled::reset();
}

#[cfg(all(test, feature = "stage-counters"))]
mod tests {
    use core::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// The counters are process-global and the test harness runs tests in
    /// parallel, so every test that records or resets holds this lock. It is
    /// a spin lock, because the crate is `no_std` and `std::sync::Mutex` is
    /// not available to its unit tests.
    static SERIAL: AtomicBool = AtomicBool::new(false);

    struct Serial;

    impl Serial {
        fn lock() -> Self {
            while SERIAL
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                core::hint::spin_loop();
            }
            Serial
        }
    }

    impl Drop for Serial {
        fn drop(&mut self) {
            SERIAL.store(false, Ordering::Release);
        }
    }

    #[test]
    fn counters_record_and_reset() {
        let _serial = Serial::lock();
        reset();
        record(Predicate::Orient2d, Outcome::Fast);
        record(Predicate::Orient2d, Outcome::Fast);
        record(Predicate::Orient2d, Outcome::Exact);
        record(Predicate::Incircle, Outcome::NonFinite);

        let o = snapshot(Predicate::Orient2d);
        assert_eq!(o.fast, 2);
        assert_eq!(o.exact, 1);
        assert_eq!(o.total(), 3);
        assert_eq!(
            o.exact_zero, 0,
            "ExactZero is recorded only where Stage 4 returns Zero"
        );

        let i = snapshot(Predicate::Incircle);
        assert_eq!(i.non_finite, 1);
        // Non-finite rejections are out of domain, so they do not dilute the rate.
        assert_eq!(i.exact_rate(), None);

        reset();
        assert_eq!(snapshot(Predicate::Orient2d).total(), 0);
    }

    #[test]
    fn an_exact_zero_is_counted_as_exact_and_as_exact_zero() {
        // Three collinear points on a 1/1024 grid near 1.4e6: c - a = 5(b - a)
        // exactly, so the determinant is exactly zero; Stages 1-3 find no
        // non-zero sign and Stage 4 returns Zero.
        let _serial = Serial::lock();
        reset();
        let before = snapshot(Predicate::Orient2d);
        let p = |k: f64| [1_400_000.0 + k / 1024.0, 900_000.0 + 3.0 * k / 1024.0];
        assert_eq!(
            crate::orient2d(p(0.0), p(1.0), p(5.0)),
            Ok(crate::Sign::Zero)
        );
        let after = snapshot(Predicate::Orient2d);
        // Other unit tests may call `orient2d` concurrently (only the counter
        // tests are serialized), so the counts are bounded from below only.
        assert!(after.exact > before.exact);
        assert!(after.exact_zero > before.exact_zero);
        assert!(after.exact_zero <= after.exact);
    }

    #[test]
    fn exact_rate_excludes_non_finite_from_the_denominator() {
        let c = Counts {
            fast: 3,
            exact: 1,
            non_finite: 96,
            exact_zero: 0,
        };
        // 1 of 4 answered calls, not 1 of 100 recorded ones.
        assert!((c.exact_rate().expect("answered calls exist") - 0.25).abs() < 1e-12);
    }

    /// The 32-bit cell saturates: one below the cap, an increment reaches
    /// it; at the cap, increments leave it there instead of wrapping to 0.
    /// Run on every host, so the fallback's one piece of logic is tested even
    /// where it is not the cell in use.
    #[test]
    fn the_32_bit_cell_saturates_at_its_maximum() {
        use core::sync::atomic::AtomicU32;
        let cell = AtomicU32::new(u32::MAX - 2);
        enabled::narrow::add_saturating(&cell);
        assert_eq!(cell.load(Ordering::Relaxed), u32::MAX - 1);
        enabled::narrow::add_saturating(&cell);
        assert_eq!(cell.load(Ordering::Relaxed), u32::MAX);
        for _ in 0..3 {
            enabled::narrow::add_saturating(&cell);
            assert_eq!(cell.load(Ordering::Relaxed), u32::MAX, "the count wrapped");
        }
        let fresh = AtomicU32::new(0);
        enabled::narrow::add_saturating(&fresh);
        assert_eq!(fresh.load(Ordering::Relaxed), 1);
    }
}
