// fuzz_targets/insphere.rs

#![no_main]

//! No-panic fuzz target for `insphere`.
//!
//! Parses 120 bytes as five 3D points (`pa`, `pb`, `pc`, `pd`, `pe`) and
//! calls `insphere` on them. Every bit pattern is passed through, NaN and
//! infinity included. The contract is that the predicate never panics,
//! returning `Err(NonFinite)` for a non-finite coordinate and the exact sign
//! otherwise. The `insphere_oracle` target checks the sign itself.

use arithmetica::insphere;
use libfuzzer_sys::fuzz_target;

/// Reads coordinate `index` of the input as a little-endian `f64`.
fn coordinate(data: &[u8], index: usize) -> f64 {
    let mut buf = [0_u8; 8];
    buf.copy_from_slice(&data[8 * index..8 * index + 8]);
    f64::from_le_bytes(buf)
}

fuzz_target!(|data: &[u8]| {
    // Need 120 bytes: 15 coordinates (5 points of 3) of 8 bytes each.
    if data.len() < 120 {
        return;
    }
    let point = |i: usize| {
        [
            coordinate(data, 3 * i),
            coordinate(data, 3 * i + 1),
            coordinate(data, 3 * i + 2),
        ]
    };

    let _ = insphere(point(0), point(1), point(2), point(3), point(4));
});
