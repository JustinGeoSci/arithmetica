// fuzz/fuzz_targets/incircle.rs

#![no_main]

//! No-panic fuzz target for `incircle`.
//!
//! Parses 64 bytes as four 2D points (`pa`, `pb`, `pc`, `pd`) and calls
//! `incircle` on them. Every bit pattern is passed through, NaN and infinity
//! included: the contract is that the predicate never panics, returning
//! `Err(NonFinite)` for a non-finite coordinate and the exact sign otherwise.
//! The `incircle_oracle` target checks the sign itself.

use arithmetica::incircle;
use libfuzzer_sys::fuzz_target;

/// Reads coordinate `index` of the input as a little-endian `f64`.
fn coordinate(data: &[u8], index: usize) -> f64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&data[8 * index..8 * index + 8]);
    f64::from_le_bytes(buf)
}

fuzz_target!(|data: &[u8]| {
    // Need 64 bytes: 8 coordinates (4 points of 2) of 8 bytes each.
    if data.len() < 64 {
        return;
    }
    let pa = [coordinate(data, 0), coordinate(data, 1)];
    let pb = [coordinate(data, 2), coordinate(data, 3)];
    let pc = [coordinate(data, 4), coordinate(data, 5)];
    let pd = [coordinate(data, 6), coordinate(data, 7)];

    let _ = incircle(pa, pb, pc, pd);
});
