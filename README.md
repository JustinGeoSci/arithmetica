# arithmetica

Exact geometric predicates for `f64` input.

`arithmetica` answers the sign questions that computational geometry is built on — which side of a line a point lies on, whether a point is inside a circle — and returns the **exact** answer for every finite `f64` input. Not an answer that is right for typical input; the sign of the true determinant, for any magnitudes, subnormals and near-degenerate configurations included.

It is a Rust implementation of Shewchuk's adaptive-precision predicates, extended so that the guarantee holds over the whole binary64 range rather than only where no operation overflows or underflows.

```rust
use arithmetica::{Sign, incircle, orient2d};

// Counterclockwise, clockwise and collinear triples.
assert_eq!(orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]), Ok(Sign::Positive));
assert_eq!(orient2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]), Ok(Sign::Negative));
assert_eq!(orient2d([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]), Ok(Sign::Zero));

// Near-degenerate input where naive f64 evaluation loses the answer: the
// third point is 2^-48 above the line y = x through the other two.
let naive = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
    (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
};
let (a, b, c) = ([0.5, 0.5], [12.0, 12.0], [24.0, 24.0 + 2f64.powi(-48)]);
assert_eq!(naive(a, b, c), 0.0); // rounding erased the answer
assert_eq!(orient2d(a, b, c), Ok(Sign::Positive)); // the exact sign

// The circle through (0, 0), (2, 0) and (0, 2) has center (1, 1) and passes
// through (2, 2).
let (p, q, r) = ([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
assert_eq!(incircle(p, q, r, [1.0, 1.0]), Ok(Sign::Positive)); // inside
assert_eq!(incircle(p, q, r, [3.0, 3.0]), Ok(Sign::Negative)); // outside
assert_eq!(incircle(p, q, r, [2.0, 2.0]), Ok(Sign::Zero)); // on the circle

// A NaN or infinite coordinate is an error, never a plausible-looking sign.
assert!(orient2d([f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]).is_err());
```

## What it provides

| Function | Question | `Positive` means |
|---|---|---|
| `orient2d(pa, pb, pc)` | Orientation of three 2D points | `pa → pb → pc` turns counterclockwise |
| `incircle(pa, pb, pc, pd)` | Is `pd` inside the circle through `pa`, `pb`, `pc`? | Inside, when `pa`, `pb`, `pc` are counterclockwise |
| `orient3d(pa, pb, pc, pd)` | Orientation of four 3D points | `pd` lies below the plane through `pa`, `pb`, `pc`, which appear counterclockwise seen from above |
| `insphere(pa, pb, pc, pd, pe)` | Is `pe` inside the sphere through `pa` … `pd`? | Inside, when `orient3d(pa, pb, pc, pd)` is `Positive` |

Each returns `Result<Sign, NonFinite>`, where `Sign` is `Positive`, `Negative` or `Zero`. `Zero` means the configuration is exactly degenerate (collinear, cocircular, coplanar, cospherical); it is a real answer, and code that treats it as either of the other two has a bug.

Beyond the four classical predicates:

- **Exact constructions in 2D.** `RationalCoord2d::from_segment_intersection` returns the exact intersection point of two lines given by `f64` endpoints, as a rational over fixed-width integers. `rational_coords_equal`, `rational_compare_along` and `orient2d_rational` answer exact questions about such points: are two of them the same point, in what order do they lie along a direction, which side of a line does one fall on. These are what a planar arrangement needs to stay topologically consistent.
- **Exact rounding information.** `RationalCoord2d::f64_bracket` returns the two `f64` neighbors of each exact coordinate and the coordinate's exact position between them, so the caller applies whatever rounding rule it needs. No function in the crate returns an approximation of a value.
- **Expansion arithmetic.** The `expansion` module exposes Shewchuk's error-free primitives (`two_sum`, `two_product`, `grow_expansion`, `scale_expansion`, `expansion_sum`). Their exactness depends on conditions their documentation states; the predicates establish those conditions themselves, and callers who use the primitives directly take that duty on.

```rust
use arithmetica::{BracketPosition, RationalCoord2d};

// The line y = x/3 meets the line y = 1 - x/2 at (6/5, 2/5), which has no
// exact f64 representation.
let p = RationalCoord2d::from_segment_intersection([0.0, 0.0], [3.0, 1.0], [0.0, 1.0], [2.0, 0.0])
    .expect("finite input")
    .expect("the lines are not parallel");
let [x, _y] = p.f64_bracket();
assert!(x.lo < x.hi);
assert_eq!(x.position, BracketPosition::BelowMidpoint);
assert_eq!(x.lo, 1.2); // round-to-nearest picks `lo`, which is the f64 written 1.2
```

## The guarantee

- **Every finite input gets the exact sign.** There is no input range, magnitude spread or degree of degeneracy outside which the answer becomes approximate.
- **Non-finite input is rejected.** A NaN or infinite coordinate returns `Err(NonFinite)`. A non-finite *intermediate* produced from finite input is not an error; that input is answered exactly like any other.
- **No panics.** No input causes a panic.

## How it works

Each predicate is a ladder of stages. A stage either **certifies** a sign, meaning it has proved that sign to be the sign of the exact determinant, or passes the input on. No stage returns an answer it has not certified.

1. **A filter in native `f64`**, using the underflow-protected error bound of Bartels, Fisikopoulos and Weiser (2023), Theorem 1(2). It is valid for every finite input, including input whose products underflow into the subnormals or overflow to infinity. `orient2d` first applies their zero filter (Definition 6), which certifies a repeated point or an axis-aligned triple by comparisons alone.
2. **Shewchuk's adaptive expansion stages.** Exact products and sums are accumulated into multi-term floating-point expansions and tested against his error bounds. Shewchuk's proofs assume an unbounded exponent range, so these stages run only on input inside an **exactness window**: the range of magnitudes within which binary64 provably behaves as that model does. Input outside its window is multiplied by a single power of two, which is exact and does not change the sign, when that brings it inside.
3. **Exact fixed-width integer arithmetic.** Every finite `f64` is an integer multiple of 2^-1074, so the determinant can be evaluated exactly in integers wide enough for the worst case. This stage handles input whose bit span is wider than the window, and confirms the result whenever the expansion stages return `Zero`.

The stages differ in cost by about two orders of magnitude. Input that is neither exactly degenerate nor spread across more bits than the window never reaches the integer stage.

The window is derived, with proofs and the margins it keeps, in [`docs/exactness_window.md`](docs/exactness_window.md).

## Design

- **`#![no_std]`, no allocation.** All buffers are fixed-size arrays on the stack, sized to proven worst-case bounds. The largest per-call stack use is in the adaptive stages of `incircle` and `insphere`, on the order of tens of KiB.
- **No runtime dependencies.** The dev-dependencies (`num-bigint`, `num-rational`, `proptest`) are used only by the tests.
- **No `unsafe`.** The crate is `#![forbid(unsafe_code)]`.
- **`f64` only.** The error-free transformations depend on the binary64 format, so there is no generic float abstraction.
- **IEEE 754 arithmetic required.** The proofs assume round-to-nearest-even with no extended precision. The x87 FPU in 80-bit mode does not satisfy this; every x86-64 target, which uses SSE2, does.

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `stage-counters` | off | Records, per predicate, how many calls each stage answered (`arithmetica::counters`). For measuring how often input reaches the exact path. With the feature off the counters compile to nothing. |
| `fuzz-replay` | off | Compiles the tests that replay the stored fuzz corpus and artifacts. Those files are in the repository, not in the published package. |

## Testing

Random inputs almost never land where a predicate can go wrong, so passing random tests says little about correctness. The test suite is built around inputs that do:

- **Boundary search.** For each predicate and five magnitude profiles (ordinary, tiny, huge, mixed, and the edges of the exactness window), the tests bisect one coordinate to the pair of adjacent floats on either side of a sign change, then compare the predicate on those floats and their neighbors with an exact big-integer evaluation. A deep tier makes about 18 million such calls.
- **Exact oracles.** Every expected sign comes from exact integer or rational arithmetic, never from floating point.
- **Fixed inputs** at the extremes of the range: subnormal products, determinants far below the smallest subnormal, huge and tiny coordinates in one call.
- **Fuzzing** with `cargo-fuzz` against the exact oracles, plus stable-toolchain replay of the stored corpus and artifacts.
- **Property tests** with `proptest`.

```sh
cargo test                                              # default suite
cargo test --features stage-counters                    # stage-routing tests
cargo test --features fuzz-replay                       # corpus and artifact replay
cargo test --release --test boundary -- --ignored       # deep boundary tier
cargo +nightly fuzz run orient2d_oracle                 # fuzzing (requires cargo-fuzz)
```

## Minimum supported Rust version

Rust 1.85.0, edition 2024.

## References

- J. R. Shewchuk. *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates.* Discrete & Computational Geometry 18(3):305–363, 1997. [doi:10.1007/PL00009321](https://doi.org/10.1007/PL00009321). The expansion arithmetic, the adaptive predicates and their error bounds.
- J. R. Shewchuk. *Routines for Arbitrary Precision Floating-point Arithmetic and Fast Robust Geometric Predicates* (`predicates.c`), 1996. Public domain. <https://www.cs.cmu.edu/~quake/robust.html>. The reference implementation the adaptive stages are ported from.
- T. Bartels, V. Fisikopoulos, M. Weiser. *Fast Floating-Point Filters for Robust Predicates.* BIT Numerical Mathematics 63:31, 2023. [doi:10.1007/s10543-023-00975-x](https://doi.org/10.1007/s10543-023-00975-x). The underflow-protected filter (Theorem 1(2)) and the zero filter (Definition 6).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. See also [NOTICE](NOTICE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
