// tests/fuzz_corpus.rs

//! Replays every stored `cargo-fuzz` corpus input through the predicates and
//! compares each answer with an exact oracle, on the stable toolchain.
//!
//! # What this adds to fuzzing
//!
//! A corpus is the set of inputs libFuzzer kept because each one reached code
//! that no earlier input had reached. That makes it a compact record of the
//! interesting corners of the input space: near-degenerate configurations,
//! subnormal coordinates, mixed magnitudes, exact zeros. The fuzz targets
//! themselves need a nightly toolchain and a time budget, so they run rarely.
//! This test runs on every `cargo test --features fuzz-replay`, in a fraction
//! of a second, and checks every one of those inputs against the oracle.
//!
//! The decoding is exactly the fuzz targets' own: each input is read as
//! little-endian `f64` coordinates, point after point; an input shorter than
//! the target needs, or with a non-finite coordinate, is skipped, as the
//! target skips it.
//!
//! # Why it is behind a feature
//!
//! The corpus lives under `fuzz/corpus/`, which the published crate does not
//! ship. Without the feature this file compiles to nothing.
//!
//! # Why a missing corpus is a failure
//!
//! A sweep that finds no inputs and reports success would be a check passing
//! because its input is missing. Every corpus directory listed below must
//! exist and must yield at least one input that is checked.
//!
//! # Usage
//!
//! `cargo test --features fuzz-replay --test fuzz_corpus`
// Test-code lint posture, stated rather than left implicit:
//
//   unwrap_used        — in a test, a panic is the failure signal.
#![allow(clippy::unwrap_used)]
#![cfg(feature = "fuzz-replay")]

mod support;

use std::fs;
use std::path::Path;

use arithmetica::Sign;
use support::{Pred, call, oracle};

/// One corpus directory and the predicate whose decoding it uses.
///
/// The no-panic targets (`orient2d`, `incircle`) decode their input the same
/// way as the matching oracle targets, so their corpora are checked against
/// the oracle too: an input kept for reaching a new branch is as worth
/// checking as one kept by the oracle target.
const CORPORA: [(&str, Pred); 6] = [
    ("orient2d", Pred::Orient2d),
    ("orient2d_oracle", Pred::Orient2d),
    ("incircle", Pred::Incircle),
    ("incircle_oracle", Pred::Incircle),
    ("orient3d_oracle", Pred::Orient3d),
    ("insphere_oracle", Pred::Insphere),
];

/// What one directory produced.
#[derive(Default)]
struct Tally {
    checked: u32,
    skipped: u32,
    exact_zeros: u32,
    mismatches: Vec<String>,
}

/// Decodes `data` as the fuzz target for `pred` does, or returns `None` when
/// the target would skip it.
fn decode(pred: Pred, data: &[u8]) -> Option<Vec<f64>> {
    let count = pred.points() * pred.dim();
    if data.len() < 8 * count {
        return None;
    }
    let coords: Vec<f64> = (0..count)
        .map(|i| {
            let mut buf = [0_u8; 8];
            buf.copy_from_slice(&data[8 * i..8 * i + 8]);
            f64::from_le_bytes(buf)
        })
        .collect();
    coords.iter().all(|c| c.is_finite()).then_some(coords)
}

/// The input as bit patterns, one point per line: the only form that
/// reproduces it exactly.
fn describe(pred: Pred, coords: &[f64]) -> String {
    coords
        .chunks(pred.dim())
        .enumerate()
        .map(|(i, point)| {
            let bits: Vec<String> = point
                .iter()
                .map(|c| format!("{:#018x}", c.to_bits()))
                .collect();
            format!("    p{i} = [{}]", bits.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Checks every input in one corpus directory.
fn sweep(dir: &Path, pred: Pred) -> Tally {
    let mut tally = Tally::default();
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if !path.is_file() {
            continue;
        }
        let data = fs::read(&path).unwrap();
        let Some(coords) = decode(pred, &data) else {
            tally.skipped += 1;
            continue;
        };
        let want = oracle(pred, &coords);
        let got = call(pred, &coords);
        tally.checked += 1;
        tally.exact_zeros += u32::from(want == Sign::Zero);
        if got != want {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            tally.mismatches.push(format!(
                "  {name}: {} returned {got:?}, exact sign is {want:?}\n{}",
                pred.name(),
                describe(pred, &coords)
            ));
        }
    }
    tally
}

#[test]
fn every_stored_corpus_input_matches_the_exact_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/corpus");
    let mut problems = Vec::new();
    let mut total = 0_u32;

    for (name, pred) in CORPORA {
        let dir = root.join(name);
        if !dir.is_dir() {
            problems.push(format!(
                "{name}: corpus directory missing at {}",
                dir.display()
            ));
            continue;
        }
        let tally = sweep(&dir, pred);
        println!(
            "{name:16} checked {:5}  skipped {:4}  exact zeros {:4}  mismatches {}",
            tally.checked,
            tally.skipped,
            tally.exact_zeros,
            tally.mismatches.len()
        );
        if tally.checked == 0 {
            problems.push(format!("{name}: no input was checked"));
        }
        total += tally.checked;
        problems.extend(tally.mismatches);
    }

    assert!(
        problems.is_empty(),
        "{} problem(s) across {total} checked inputs:\n{}",
        problems.len(),
        problems.join("\n")
    );
}
