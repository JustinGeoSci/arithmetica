// examples/stage_counters.rs

//! Measures how often `orient2d` needs its exact integer stage.
//!
//! Run with `cargo run --release --example stage_counters --features
//! stage-counters`.
//!
//! Random triples almost never need it. Exactly collinear triples in a
//! general direction always do, because no floating-point stage can certify
//! a zero; the integer stage confirms it.

use arithmetica::counters::{self, Predicate};
use arithmetica::orient2d;

/// A small deterministic generator, so the example needs no dependencies.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // The top 53 bits as a uniform value in [0, 1).
        #[allow(clippy::cast_precision_loss)]
        let unit = (z >> 11) as f64 / (1_u64 << 53) as f64;
        unit
    }
}

fn report(label: &str) {
    let counts = counters::snapshot(Predicate::Orient2d);
    let rate = counts.exact_rate().unwrap_or(0.0);
    println!(
        "{label:<22} calls {:>7}  exact stage {:>7}  ({:.2}%)",
        counts.total(),
        counts.exact,
        100.0 * rate
    );
}

fn main() {
    let mut rng = SplitMix64(1);

    counters::reset();
    for _ in 0..100_000 {
        let p = |rng: &mut SplitMix64| [rng.next_f64(), rng.next_f64()];
        let (a, b, c) = (p(&mut rng), p(&mut rng), p(&mut rng));
        let _ = orient2d(a, b, c);
    }
    report("random triples");

    counters::reset();
    for i in 1..=1_000 {
        // Three points on the line y = 3x, which is neither repeated nor
        // axis-aligned, so only exact arithmetic can certify the zero.
        let t = f64::from(i);
        let _ = orient2d([t, 3.0 * t], [2.0 * t, 6.0 * t], [5.0 * t, 15.0 * t]);
    }
    report("collinear triples");
}
