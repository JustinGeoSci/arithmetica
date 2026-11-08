# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-08

### Added

- `compare_distance(p, q1, q2)`, the exact sign of `|p − q1|² − |p − q2|²`,
  which determines which of two points is nearer to a third.
- `dot_sign(p, a, b)`, the exact sign of `(a − p) · (b − p)`, which
  determines whether an angle is acute, right or obtuse.
- `compare_point_distance(p, q, d)`, the exact sign of `|p − q|² − d²`,
  which determines whether a point is closer to another than `d`, at exactly
  `d`, or farther.
- `compare_segment_distance(p, a, b, d)`, the exact sign of
  `dist²(p, ab) − d²` for the closed segment `ab`. The nearest point, an
  endpoint or the interior, is chosen exactly, and the interior case is
  evaluated as `cross² − d²·|b − a|²` with no division. A degenerate segment
  (`a == b`) is the point `a`.

All four operate on the same sequence of stages as `orient2d`. Each is exact
for every finite `f64` input and returns `Err(NonFinite)` for NaN or infinite
input. Only `d²` enters the two distance comparisons, so the sign of `d` is
irrelevant. The `stage-counters` feature records none of the four, as its
entry in the README explains.

### Changed

- The `counters` documentation now states the counters' storage, one set of
  process-global atomics shared by every thread. The behavior is unchanged
  from 0.1.0.
- The `fuzz-replay` feature compiles nothing. The replay of the stored fuzz
  corpus and artifacts no longer ships with the crate, since neither the
  package nor its repository carries those inputs. The feature is kept so
  that a build naming it still resolves, and it is removed in 0.2.

### Fixed

- The `stage-counters` feature now builds on targets without 64-bit atomics
  (for example `thumbv7em-none-eabi`), where 0.1.0 failed with an unresolved
  import of `AtomicU64`. There the counters are 32-bit and saturate at
  `u32::MAX` rather than wrapping. On every target with 64-bit atomics they
  are unchanged. A target with no atomic read-modify-write (for example
  `thumbv6m-none-eabi`) now refuses the feature with a compile-time message.

## [0.1.0] - 2026-10-11

### Added

- `orient2d`, `incircle`, `orient3d` and `insphere`, which return exact signs
  for every finite `f64` input and `Err(NonFinite)` for NaN or infinite
  coordinates.
- `RationalCoord2d`, exact 2D points from `f64` input or from the
  intersection of two lines, with `rational_coords_equal`,
  `rational_compare_along`, `orient2d_rational` and `f64_bracket`.
- `expansion`, Shewchuk's error-free floating-point primitives.
- `counters`, per-stage call counts behind the `stage-counters` feature.

[0.1.1]: https://github.com/JustinGeoSci/arithmetica/releases/tag/v0.1.1
[0.1.0]: https://github.com/JustinGeoSci/arithmetica/releases/tag/v0.1.0