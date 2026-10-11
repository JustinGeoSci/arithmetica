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
//! Every test in this crate checks the *sign* a predicate returns, and the
//! sign is correct no matter which stage produces it; that is the adaptive
//! ladder working as designed. The consequence is that nothing otherwise
//! detects a change that moves work from the fast path to the exact path. An
//! error-bound adjustment, or a guard added to close a fuzz finding, can
//! multiply exact-path entries by a large factor while every test still
//! passes. These counters make that visible.
//!
//! # Two rules
//!
//! 1. **Observability only.** No behavior in this crate may read a counter. A
//!    predicate that branches on its own statistics is not a predicate.
//! 2. **Not a correctness signal.** A high exact-path rate means the inputs
//!    are degenerate, not that anything is wrong. Exactness is the product,
//!    not a failure mode.
//!
//! # What is and is not distinguished
//!
//! [`Outcome::Exact`] marks entry to the Stage 4 exact-integer path, and
//! [`Outcome::Fast`] covers Stages 1 through 3 together. [`Outcome::ExactZero`]
//! additionally marks the Stage 4 calls whose result was `Zero`; it is
//! recorded where Stage 4 returns, outside the `*_adapt` bodies.
//!
//! Separating Stage 2 from Stage 3 would mean instrumenting the bodies of
//! `orient2d_adapt`, `incircle_adapt`, `orient3d_adapt` and `insphere_adapt`,
//! which are direct ports of Shewchuk's code and are kept line-for-line
//! comparable with it. The fast-versus-exact split is the number that
//! predicts cost; the finer split is not.
//!
//! # Aggregates, not attribution
//!
//! These are global counters, so under concurrency they report totals, never
//! which call used which stage. Per-call attribution would be a different
//! feature — a returning variant such as
//! `orient2d_with_stage(..) -> (Result<Sign, NonFinite>, Stage)` — and must
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
    /// result was `Zero`: an exactly degenerate input. Recorded *in addition
    /// to* `Exact`, never instead of it, so `exact_zero <= exact` and the
    /// totals are unchanged. It measures how much of the exact path is spent
    /// confirming exact zeros.
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

#[cfg(feature = "stage-counters")]
mod enabled {
    use super::{Outcome, Predicate};
    use core::sync::atomic::{AtomicU64, Ordering};

    /// `[predicate][outcome]`, 4 predicates × 4 outcomes.
    ///
    /// `Relaxed` is the right ordering: these counts order nothing and
    /// synchronize nothing, so anything stronger would buy only cost.
    /// `AtomicU64` is not `Copy`, so the array-repeat form cannot build this.
    /// Sixteen explicit cells (four per row) keep the static
    /// `const`-initialized with no lazy-init machinery and no dependency.
    #[allow(clippy::declare_interior_mutable_const)]
    const ROW: [AtomicU64; 4] = [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ];
    static COUNTS: [[AtomicU64; 4]; 4] = [ROW, ROW, ROW, ROW];

    /// Increment one counter.
    #[inline]
    pub fn record(predicate: Predicate, outcome: Outcome) {
        COUNTS[predicate.index()][outcome.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Read one counter.
    #[inline]
    pub fn read(predicate: Predicate, outcome: Outcome) -> u64 {
        COUNTS[predicate.index()][outcome.index()].load(Ordering::Relaxed)
    }

    /// Zero every counter.
    pub fn reset() {
        for row in &COUNTS {
            for cell in row {
                cell.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(feature = "stage-counters")]
pub use enabled::record;

/// Counts for one predicate at one instant.
#[cfg(feature = "stage-counters")]
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
    /// Non-finite rejections are excluded from the denominator: they are
    /// out-of-domain inputs, not degenerate ones, and counting them would
    /// dilute the rate this number exists to report. Returns `None` when no
    /// call has been answered.
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
/// three fields at slightly different instants; the counters are independent
/// and not read atomically as a group.
#[cfg(feature = "stage-counters")]
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
pub fn reset() {
    enabled::reset();
}

#[cfg(all(test, feature = "stage-counters"))]
mod tests {
    use core::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    /// The counters are process-global and the test harness runs tests in
    /// parallel, so every test that records or resets holds this lock. A spin
    /// lock, because the crate is `no_std` and `std::sync::Mutex` is not
    /// available to its unit tests.
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
}
