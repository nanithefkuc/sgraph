# SGRAPH Engineering Invariants

These rules apply to the entire repository. They encode decisions that are
expensive to rediscover; violating one is a bug even when the tests pass.

## Scope

`sgraph` is a sparse-graph engine, not a codec. Field arithmetic and byte-buffer
vector primitives come from `fgf`, and elimination comes from `gfm` — never
re-implement them here. Wire formats, packet headers, transport and HARQ policy,
belief-propagation soft-decision decoding, protograph lifting, and codec shells
belong to consumers.

Vocabulary is graph-theoretic: variable/check, not source/repair. This crate
serves LDPC — where a check is a parity constraint and not a transmitted symbol —
as much as it serves LT and Raptor.

## Determinism

- Encoder and decoder MUST regenerate byte-identical edge sets from the same
  check id and parameters. A change to the PRNG, the sampling algorithm, the draw
  order, or a domain-separation constant is a format break for every downstream
  consumer, not a refactor.
- The fixtures under `tests/data/` pin the PRNG streams and the neighbour offsets
  against values captured from an independent implementation, with their
  provenance recorded beside them. Moving a value is a format break; changing a
  pinned stream requires a new or versioned generator. Extend them when adding a
  generator; never retune them to match new output.
- A point-mass degree distribution MUST consume zero RNG draws, so that composing
  a degree draw ahead of an edge draw leaves the edge stream unchanged. This is
  what keeps a constant-degree generator bit-compatible with the frozen offsets.
- Public sampling entry points validate before touching caller output: a rejected
  request leaves the buffer and the generator untouched, never half-written.
- Sampling functions take `&mut SplitMix64`, never a seed, so that degree and
  edge draws compose into one stream. A separately-seeded degree draw would leave
  the two streams independent.
- Domain separation is caller-supplied. Never bake a domain constant into the
  crate: seed derivation is a wire-compatibility decision.

## Index domains

- A retired index is **gone**, not absent. An index merely below `base` that was
  never retired is absent, and inserting it grows the front. `Lookup` and
  `Membership` report which of the three states an index is in, and
  `Ring::floor`/`IndexSet::floor` give the horizon that separates the last two.
  Never collapse `Retired` into `Vacant`: a container that silently refuses a
  live index drops data, and one that silently accepts a retired index
  resurrects it.
- Dense index storage is bounded by construction. Both containers take a maximum
  live span, and every growth path checks it along with the `u64`→`usize`
  conversion. Exceeding either returns `GraphError` and leaves the structure
  exactly as it was: limits reject input and never evict state.
- Index arithmetic works in offsets from `base`. `base + len` overflows for a
  container reaching `u64::MAX`, which is why the public bound is an inclusive
  `last` rather than an exclusive `end`.
- State disappears only on an explicit `retire_below`. Nothing is evicted
  implicitly.
- Retirement hands back the values it drops so their buffers can be recycled.

## Allocation

- Peeling ingest, cascade, and residual solve allocate nothing in steady state.
  Scratch is owned and reused; symbol buffers and index lists are recycled
  through pools.
- `tests/zero_alloc.rs` counts allocations with a global allocator; extend it when
  adding an execution path. The property is tested, not asserted.
- Validation and dispatch happen once at the public boundary, never per edge.

## Residual invariants

- For every live check row, `rhs` equals the field sum of the true values of the
  variables still in its support. Known neighbours are folded out at ingest and
  dropped; the resident structure is the residual graph.
- `deficiency == |unknowns| − rank`, reported by `gfm` as a pivot count. Exact,
  not a heuristic.
- `gfm` owns elimination and full-reduction semantics. The adapter preserves its
  per-column determinedness result; do not add another elimination loop here.
- Residual columns are the consumer's complete, sorted `IndexSet` snapshot. Never
  infer them from sparse support: a missing variable can occur only in a dense
  row or no received row yet.
- A zero coefficient row with non-zero packed RHS is inconsistent. Clear solver
  outcome metadata and teach the peeler nothing from that solve.
- Consumer-owned dense rows remove a term when folding its known value, so every
  known column changes the RHS exactly once.

## Field arithmetic

- Do not write `unsafe` SIMD in this crate. There is one implementation to audit
  and it is upstream. The crate root carries `#![forbid(unsafe_code)]`.
- Call `fgf::ops` directly through `EdgeWeight`. Do not add a wrapper module that
  renames `add_assign`/`mul_add`/`mul_assign` — one convention only.
- `fgf::ops` panics on geometry violations. Validate at the public boundary and
  `debug_assert` internally so a caller never reaches a panicking kernel.
- GF(2) XOR is `fgf::ops::add_assign::<Gf8>`: `fgf` has no `Gf2`, and XOR of a
  packed element array is XOR of bytes for any field. `Gf8` is an arbitrary
  witness; say so at the call site.

## Public surface

- Everything outside `internals` is the compatibility promise. Re-export it at the
  crate root and document it — `missing_docs` and `missing_debug_implementations`
  are warnings crate-wide, allowed only on `internals`.
- Anything returning a collection on a per-symbol or per-tick path provides an
  `_into(&mut Vec<_>)` form.
- Do not ship preset geometries. Degree, domain, and overhead are
  channel-specific tradeoffs; they belong in the README with their measurement
  conditions, not in the API.
- Do not add an error variant before something constructs it. An unconstructed
  variant is a placeholder, and placeholders outlive their intent.

## Testing changes

- A bug fix MUST include a regression that fails for the observed bug.
- Never use the implementation under test as its own oracle. Independent
  references: a naive reference peeler, a naive dense solve over scalar field
  elements, `BTreeSet` for the index containers, and — for anything extracted —
  vectors captured from the source implementation before the extraction.
- Recovery tests MUST use loss patterns a plausible bug would break — stopping
  sets, bursts, reordering — not clean round trips.
- Assert exact values, never predicates that admit unintended nonzero results.
- Any performance change MUST be measured through the criterion harness with a
  saved baseline. Never land one on the strength of reasoning alone. Build both
  binaries first, then run them interleaved and pinned to one core, and take the
  minimum of several runs: back-to-back `cargo bench` invocations on a laptop
  drift by tens of percent between two identical builds, which is larger than
  most changes worth measuring. Criterion baseline artifacts are machine-specific
  and are never committed.
- Run focused regressions first, then the full matrix:

  ```sh
  cargo test --all-features
  cargo test --no-default-features
  cargo fmt --all --check
  cargo clippy --all-targets --all-features -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
  ```

## Code conventions

- Rust 1.89 or newer, edition 2024, `no_std` + `alloc` without default features.
- Errors via `Result` and the crate's own error enums. No `unwrap`/`expect` on
  fallible operations in library paths.
- `mod.rs` and `lib.rs` hold declarations only — module docs, `mod`, `pub use`,
  plain type declarations. No function bodies, no `impl` blocks.
- Do not put development history in doc comments: no milestone tags, no
  references to superseded designs, no phase numbering.
- Public documentation and comments MUST NOT reference private or unpublished
  downstream projects.
- `gen` is a reserved keyword in edition 2024. Name generator bindings and
  parameters `generator`.

## Edges

- An edge set is canonical at the public boundary: parallel support and weight
  arrays of equal length, non-empty, each variable at most once, and no zero
  coefficients. `Edges::new` is the single place that is checked, so everything
  downstream may assume it. A duplicate would fold twice during reduction and
  corrupt the residual invariant; a zero weight makes a degree-one row
  unsolvable while still looking peelable.
- Generated edge order is deterministic but **not** sorted. Nothing may assume
  otherwise, and nothing may sort in place and expect peers to agree.
- `NeighborGen::neighbors` leaves its output buffer cleared on error, never
  partially filled, so the caller can reuse one `NeighborBuf` across checks
  without observing debris.
- `NeighborBuf` owns both the parallel output arrays and the `u32` sampling
  scratch. Generators MUST take their scratch from it rather than allocating a
  temporary, or steady-state generation stops being allocation-free.
- `EdgeWeight::ELEMENT_BYTES` is what lets a symbol length be validated for
  packed-element alignment once, before any `fgf` kernel can reach its panicking
  geometry check.

## Layout

```
src/
  lib.rs             crate root: docs and declarations only
  error.rs           GraphError / SolveError
  id.rs              VarId / CheckId transparent newtypes
  rng.rs             deterministic PRNG (SplitMix64) + distinct-k sampling
  index.rs           bounded dense index-keyed storage (Ring, IndexSet)
  weight.rs          EdgeWeight / ResidualCoeff seams; Binary (ZST)
  degree.rs          DegreeDistribution + Constant, Cumulative, RobustSoliton
  neighbors/mod.rs   NeighborGen, NeighborBuf scratch, validated Edges
  neighbors/uniform.rs   Uniform and WindowedUniform generators
  neighbors/triple.rs    Rfc5053Triple (RFC 5053 Raptor only)
  neighbors/explicit.rs  ExplicitMatrix, a validated CSR parity-check matrix
  peel/              residual rows, Peeler, retirement, and buffer pools
  residual/          graph-owned rows/building and a gfm-backed Solver
  driver.rs          DenseRows consumer seam and fixpoint Resolver
tests/
  vectors.rs         asserts the frozen fixtures
  zero_alloc.rs      counting-allocator proofs for steady-state hot paths
  data/              captured PRNG and edge-offset fixtures, with provenance
benches/
  graph.rs           criterion: neighbours, ingest, cascade, residual solve
```
