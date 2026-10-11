// tests/replay_fuzz_artifacts.rs

//! Replays the stored `cargo-fuzz` artifacts under `fuzz/artifacts/` and
//! checks the predicate's sign on each against an exact rational oracle.
//!
//! The fuzz targets require nightly. This test performs the same oracle
//! comparison on stable, reading the artifact files directly, so every
//! stored input is checked by an ordinary `cargo test` run.
//!
//! Usage: `cargo test --features fuzz-replay --test replay_fuzz_artifacts`
//! from the crate directory.
//!
//! # Why it is behind a feature
//!
//! The artifacts live under `fuzz/`, which the published crate does not ship,
//! so in a published checkout this test would fail for want of its input.
//! The `fuzz-replay` feature is off by default and is turned on in the
//! repository, where the artifacts are present.
//!
//! # Why this fails on an empty artifact directory
//!
//! The artifacts are minimized inputs saved by the oracle-comparison fuzz
//! targets, chosen by the fuzzer for their difficulty, which makes them some
//! of the hardest inputs on record. A replay that reports "0 artifacts" and
//! exits 0 would be a check passing because its input is missing, so an
//! absent or empty directory is a failure, not a skip. If the artifacts are ever deliberately
//! retired, delete this test with them.

#![cfg(feature = "fuzz-replay")]

use std::path::Path;

use arithmetica::{Sign, incircle, orient2d};
use num_bigint::BigInt;
use num_rational::Ratio;

fn r(x: f64) -> Ratio<BigInt> {
    Ratio::<BigInt>::from_float(x).expect("finite by prior check")
}

fn sign_of(det: &Ratio<BigInt>) -> Sign {
    let zero = Ratio::from_integer(BigInt::from(0));
    match det.cmp(&zero) {
        core::cmp::Ordering::Greater => Sign::Positive,
        core::cmp::Ordering::Less => Sign::Negative,
        core::cmp::Ordering::Equal => Sign::Zero,
    }
}

fn oracle_orient2d(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2]) -> Sign {
    let det = (r(pa[0]) - r(pc[0])) * (r(pb[1]) - r(pc[1]))
        - (r(pa[1]) - r(pc[1])) * (r(pb[0]) - r(pc[0]));
    sign_of(&det)
}

fn oracle_incircle(pa: [f64; 2], pb: [f64; 2], pc: [f64; 2], pd: [f64; 2]) -> Sign {
    let (adx, ady) = (r(pa[0]) - r(pd[0]), r(pa[1]) - r(pd[1]));
    let (bdx, bdy) = (r(pb[0]) - r(pd[0]), r(pb[1]) - r(pd[1]));
    let (cdx, cdy) = (r(pc[0]) - r(pd[0]), r(pc[1]) - r(pd[1]));
    let alift = adx.clone() * adx.clone() + ady.clone() * ady.clone();
    let blift = bdx.clone() * bdx.clone() + bdy.clone() * bdy.clone();
    let clift = cdx.clone() * cdx.clone() + cdy.clone() * cdy.clone();
    let det = alift * (bdx.clone() * cdy.clone() - bdy.clone() * cdx.clone())
        - blift * (adx.clone() * cdy - ady.clone() * cdx)
        + clift * (adx * bdy - ady * bdx);
    sign_of(&det)
}

fn read_f64(data: &[u8], offset: usize) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[offset..offset + 8]);
    f64::from_le_bytes(buf)
}

enum Verdict {
    Skipped(&'static str),
    Agrees,
    Diverges(String),
}

fn check_orient2d(data: &[u8]) -> Verdict {
    if data.len() < 48 {
        return Verdict::Skipped("under 48 bytes");
    }
    let pa = [read_f64(data, 0), read_f64(data, 8)];
    let pb = [read_f64(data, 16), read_f64(data, 24)];
    let pc = [read_f64(data, 32), read_f64(data, 40)];
    if pa.iter().chain(&pb).chain(&pc).any(|c| !c.is_finite()) {
        return Verdict::Skipped("non-finite input");
    }
    let got = orient2d(pa, pb, pc).expect("finite input");
    let want = oracle_orient2d(pa, pb, pc);
    if got == want {
        Verdict::Agrees
    } else {
        Verdict::Diverges(format!(
            "orient2d = {got:?}, oracle = {want:?}  \
             pa=[{:e},{:e}] pb=[{:e},{:e}] pc=[{:e},{:e}]",
            pa[0], pa[1], pb[0], pb[1], pc[0], pc[1]
        ))
    }
}

fn check_incircle(data: &[u8]) -> Verdict {
    if data.len() < 64 {
        return Verdict::Skipped("under 64 bytes");
    }
    let pa = [read_f64(data, 0), read_f64(data, 8)];
    let pb = [read_f64(data, 16), read_f64(data, 24)];
    let pc = [read_f64(data, 32), read_f64(data, 40)];
    let pd = [read_f64(data, 48), read_f64(data, 56)];
    if pa
        .iter()
        .chain(&pb)
        .chain(&pc)
        .chain(&pd)
        .any(|c| !c.is_finite())
    {
        return Verdict::Skipped("non-finite input");
    }
    let got = incircle(pa, pb, pc, pd).expect("finite input");
    let want = oracle_incircle(pa, pb, pc, pd);
    if got == want {
        Verdict::Agrees
    } else {
        Verdict::Diverges(format!(
            "incircle = {got:?}, oracle = {want:?}  \
             pa=[{:e},{:e}] pb=[{:e},{:e}] pc=[{:e},{:e}] pd=[{:e},{:e}]",
            pa[0], pa[1], pb[0], pb[1], pc[0], pc[1], pd[0], pd[1]
        ))
    }
}

#[test]
fn stored_fuzz_artifacts_agree_with_the_exact_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz/artifacts");
    let mut wrong: Vec<String> = Vec::new();
    let mut agreed = 0usize;
    let mut skipped = 0usize;
    let mut missing: Vec<&str> = Vec::new();

    for target in ["orient2d", "orient2d_oracle", "incircle", "incircle_oracle"] {
        let dir = root.join(target);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            missing.push(target);
            continue;
        };
        let mut names: Vec<_> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        names.sort();
        if names.is_empty() {
            println!("{target}: 0 artifacts");
            continue;
        }
        println!("{target}: {} artifacts", names.len());
        for path in names {
            let data = std::fs::read(&path).expect("artifact is readable");
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let short = &name[..name.len().min(18)];
            let verdict = if target.starts_with("incircle") {
                check_incircle(&data)
            } else {
                check_orient2d(&data)
            };
            match verdict {
                Verdict::Agrees => {
                    agreed += 1;
                    println!("  OK     {short}  agrees with the oracle");
                }
                Verdict::Skipped(why) => {
                    skipped += 1;
                    println!("  SKIP   {short}  {why}");
                }
                Verdict::Diverges(detail) => {
                    println!("  WRONG  {short}  {detail}");
                    wrong.push(format!("{target}/{short}: {detail}"));
                }
            }
        }
    }

    println!(
        "\n{} wrong, {agreed} agreed, {skipped} skipped",
        wrong.len()
    );

    assert!(
        wrong.is_empty(),
        "{} stored fuzz artifact(s) give a sign that disagrees with the exact \
         rational oracle:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );

    // A per-target directory may legitimately be absent: git does not track
    // empty directories, and the `orient2d` and `incircle` targets have no
    // stored artifacts, so their directories hold nothing to commit. What
    // must not be empty is the replay as a whole (see the module doc comment).
    if !missing.is_empty() {
        println!("no artifact directory for {missing:?} (empty is expected for some)");
    }
    assert!(
        agreed + skipped > 0,
        "replayed no artifacts at all, so this test verified nothing. The \
         stored inputs under fuzz/artifacts/ are what this test exists to \
         check; if they are gone, recover them rather than letting this pass"
    );
}
