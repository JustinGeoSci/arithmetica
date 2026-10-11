// fuzz/fuzz_targets/orient2d.rs

#![no_main]

//! No-panic fuzz target for `orient2d`.
//!
//! Parses 48 bytes as three 2D points (`pa`, `pb`, `pc`) and calls `orient2d`
//! on them. Every bit pattern is passed through, NaN and infinity included:
//! the contract is that the predicate never panics, returning
//! `Err(NonFinite)` for a non-finite coordinate and the exact sign otherwise.
//! The `orient2d_oracle` target checks the sign itself.

use arithmetica::orient2d;
use libfuzzer_sys::fuzz_target;

/// Reads coordinate `index` of the input as a little-endian `f64`.
fn coordinate(data: &[u8], index: usize) -> f64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&data[8 * index..8 * index + 8]);
    f64::from_le_bytes(buf)
}

fuzz_target!(|data: &[u8]| {
    // Need 48 bytes: 6 coordinates (3 points of 2) of 8 bytes each.
    if data.len() < 48 {
        return;
    }
    let pa = [coordinate(data, 0), coordinate(data, 1)];
    let pb = [coordinate(data, 2), coordinate(data, 3)];
    let pc = [coordinate(data, 4), coordinate(data, 5)];

    let _ = orient2d(pa, pb, pc);
});
