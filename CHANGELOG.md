# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-11

### Added

- `orient2d`, `incircle`, `orient3d` and `insphere`: exact signs for every
  finite `f64` input, with `Err(NonFinite)` for NaN or infinite coordinates.
- `RationalCoord2d`: exact 2D points from `f64` input or from the
  intersection of two lines, with `rational_coords_equal`,
  `rational_compare_along`, `orient2d_rational` and `f64_bracket`.
- `expansion`: Shewchuk's error-free floating-point primitives.
- `counters`: per-stage call counts behind the `stage-counters` feature.

[0.1.0]: https://github.com/JustinGeoSci/arithmetica/releases/tag/v0.1.0