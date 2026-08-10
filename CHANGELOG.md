# Changelog

All notable changes to `sgraph` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The crate is distributed through git only and carries no release tags, so no
version-comparison links are provided.

## [0.1.0]

Initial release of the shared sparse/Tanner-graph engine for erasure codes:
deterministic cross-peer neighbour generation, degree distributions, a residual
sparse graph, XOR-only peeling, and the exact residual solve that finishes what
peeling cannot. Complete for the binary (GF(2)) path.

### Added

- **Deterministic PRNG and sampling** (`rng`): `SplitMix64`, unbiased bounded
  draws over a `NonZeroU32`, and Floyd distinct-k offset sampling. Sampling takes
  `&mut SplitMix64` so a degree draw and an edge draw compose into one
  reproducible stream, and validates before writing so a rejected request leaves
  the buffer and generator untouched.
- **Bounded index-keyed storage** (`index`): `Ring<T>` and `IndexSet` over a
  monotone `u64` domain, where lookup is a subtraction and retirement is a front
  drain that hands back what it dropped. `Lookup` and `Membership` distinguish
  live, absent, and retired states; every growth path checks the configured live
  span and the `u64`→`usize` conversion, rejecting input rather than evicting
  state.
- **Edge-weight seams** (`weight`): `EdgeWeight` and `ResidualCoeff<F>`, plus
  `Binary`, a zero-sized GF(2) coefficient, so one generic engine serves GF(2)
  and GF(2^m) without taxing the binary path.
- **Degree distributions** (`degree`): `DegreeDistribution` with `Constant` (a
  point mass that consumes no randomness), `Cumulative` (an explicit integer
  weight table), and `RobustSoliton` (built from Q32 fixed-point parameters for
  platform-identical tables).
- **Neighbour generators** (`neighbors`): `NeighborGen` with `Uniform`,
  `WindowedUniform`, `Rfc5053Triple` (RFC 5053 Raptor), and `ExplicitMatrix`
  (CSR parity-check) generators, the reusable `NeighborBuf` scratch, and `Edges`,
  the single validated edge-shape boundary.
- **Peeling decoder** (`peel`): `Peeler` with the residual sparse graph, reverse
  adjacency, the iterative degree-one cascade, pooled buffers, explicit
  retirement, and `StalledRow` exposing what peeling could not finish.
- **Residual solve** (`residual`): `ResidualBuilder`/`RowSink` single-pass
  assembly over explicit columns and `Solver`, full reduced row echelon form over
  any `fff` field.
- **Fixpoint driver** (`driver`): `DenseRows`, the consumer seam for
  progressively reduced dense equations, and `Resolver`, the peel → solve →
  re-peel fixpoint.
- **Typed identifiers** (`id`): `VarId` and `CheckId` `#[repr(transparent)]`
  newtypes so a variable horizon cannot be passed where a check key belongs.
- **Errors** (`error`): `GraphError` and `SolveError`.
- Frozen determinism fixtures under `tests/data/`, asserted by
  `tests/vectors.rs` against values captured from an independent implementation,
  and `tests/zero_alloc.rs` counting-allocator proofs for steady-state hot paths.
- Criterion benchmarks (`benches/graph.rs`) covering neighbour generation,
  ingest, cascade, and residual solve.
- Feature flags: `std` (default), `simd` (default, implies `std`), and
  `internals` for unstable APIs exempt from compatibility guarantees.
  `no_std` + `alloc` without default features.

### Changed

- Removed the extraction regressions from the consumer hot paths so peeling
  ingest, cascade, and residual solve allocate nothing in steady state.
