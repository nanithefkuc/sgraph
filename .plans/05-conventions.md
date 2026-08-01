# Conventions

Distilled from `cafft/AGENTS.md`, `mix-dpc/AGENTS.md`, and the three manifests.
Where the siblings disagree, the choice and its reason are stated.

## `Cargo.toml`

```toml
[package]
name = "sgraph"
version = "0.1.0"
edition = "2024"
rust-version = "1.89"
description = "Sparse graph: a shared sparse/Tanner-graph engine for erasure codes"
repository = "https://github.com/nanithefkuc/sgraph"
license = "MIT"
readme = "README.md"
keywords = ["ldpc", "fountain", "erasure", "tanner", "peeling"]
categories = ["algorithms", "mathematics", "no-std"]
exclude = ["/.github", "/.plans"]
publish = false          # Distributed via git only; depends on fff by git.

[dependencies]
# Keep the extraction proof on the exact fff revision used by mix-dpc
# 6d7d4ac. Upgrade only as a separate, fingerprinted change after Phase 5.
fff = { git = "https://github.com/nanithefkuc/fff", rev = "0077ef4463310653d5f18c17a9a5f12b734d04a8" }

[features]
default = ["std", "simd"]
# Runtime CPU detection and process-wide backend caching.
std = ["fff/std"]
# Runtime-dispatched SIMD kernels; without it, portable scalar.
simd = ["std", "fff/simd"]
# Unstable APIs, exempt from this crate's compatibility guarantees.
internals = []

[dev-dependencies]
criterion = { version = "0.8", default-features = false, features = ["cargo_bench_support"] }

[[bench]]
name = "graph"
harness = false

[profile.bench]
lto = "thin"
codegen-units = 1

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

Notes on the choices:

- **`fff` pinned by `rev`.** Phase 0 uses the exact revision already pinned by
  extraction source `mix-dpc`; changing `fff` during the migration would make a
  failed interop fingerprint ambiguous. `sgraph` has no wire format of its own,
  but its arithmetic outputs become consumer wire bytes. Any later `fff` update
  is a separate change gated by the same vectors, interop tests, and benchmarks.
- **`no_std` + `alloc`.** Follow `cafft`, not `mix-dpc`. `sgraph` needs only
  `Vec`, `VecDeque`, and `Box`; the one `std`-requiring thing in the family is
  `fff`'s `LazyLock` backend dispatch, which is already gated behind
  `fff/std`. `HashMap` — the one `std` collection `mix-dpc` uses — lives in
  `hdpc.rs`, which is not coming across. `simd` implying `std` is a hard rule in
  both `fff` and `cafft`.
- **`internals` declared but curated.** Both sibling forms are legitimate: the
  per-module visibility flip (`#[cfg(feature = "internals")] pub mod x;` /
  `#[cfg(not(...))] pub(crate) mod x;`, as in `cafft/src/core/mod.rs:14-17` and
  `fff/src/kernel/mod.rs:18-72`) and the facade re-export module
  (`cafft/src/internals.rs`). Use the visibility flip for engine internals and a
  small `src/internals.rs` facade for the curated unstable set, so the code
  compiles identically either way.
- **No other dependencies, ever.** The family has no `thiserror`, no `rand`, no
  `serde`. Errors are hand-rolled; the PRNG is hand-rolled. Keep it that way.
- `[profile.release]` only if `sgraph` ever ships an example binary.

## `src/lib.rs` header

```rust
//! # Sparse Graph
//!
//! <one paragraph: the shared sparse/Tanner-graph engine that LDPC, LT, and
//! Raptor-class codecs keep re-implementing>
//!
//! Field arithmetic and byte-buffer vector primitives come from [`fff`]; this
//! crate never re-implements field arithmetic. Wire formats, packet headers,
//! transport and HARQ policy, belief-propagation soft-decision decoding,
//! protograph lifting, and codec shells stay with the consumer.
//!
//! ## Layout
//! <bulleted module map with intra-doc links>
//!
//! ## Example
//! <compiling doctest ending in `# Ok::<(), sgraph::GraphError>(())`>
//!
//! ## Feature flags
//! <std / simd / internals>

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;
```

`#![forbid(unsafe_code)]` rather than `cafft`/`fff`'s `warn`/`deny`: `sgraph`
writes no intrinsics. Per `mix-dpc/AGENTS.md` — "There is one implementation to
audit and it is upstream."

Use `::alloc::…` / `::core::…` at use sites for clarity, though `sgraph` has no
module named `core` so the absolute-path *rule* and its CI grep gate
(`cafft/AGENTS.md` "Crate hygiene") do not bind.

## Errors

Hand-rolled, no `thiserror`. One `src/error.rs`; small enums per failure domain
rather than one god-enum (`cafft` has `PlanError` + `TransformLengthError`;
`mix-dpc` has one `ConfigError` because it has one domain).

`sgraph` has two domains, so two types:

- `GraphError` — construction and graph mutation: zero/misaligned symbol length,
  invalid degree/domain, duplicate variables, support/weight length mismatch,
  zero weighted coefficient, retired or non-monotone index, checked arithmetic
  overflow, and configured live-span excess.
- `SolveError` — residual assembly and solve: unsorted/non-distinct unknown
  columns, unknown terms, right-hand-side length/alignment mismatch, checked
  scratch geometry overflow, and `InconsistentSystem`.

Every error path validates before mutation. Tests compare a state snapshot before
and after malformed input so “returned an error after partial insertion” cannot
pass.

Shape, following both siblings exactly:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    /// A symbol length of zero has no valid interpretation.
    ZeroSymbolLen,
    /// The requested degree exceeds what the distribution supports.
    DegreeOutOfRange {
        /// The rejected degree.
        degree: u32,
        /// The largest supported degree.
        max: u32,
    },
    // …
}
```

- Struct variants with named, **individually documented** public fields carrying
  both the offending value *and* the limit.
- Manual `impl Display` with `match self` and inline-captured format args:
  `write!(f, "degree {degree} out of range 1..={max}")`.
- `#[cfg(feature = "std")] impl ::std::error::Error for GraphError {}` — gated,
  because `sgraph` is `no_std`-capable (`cafft`'s form, not `mix-dpc`'s).
- Every fallible public constructor carries a `/// # Errors` section.
- No `unwrap`/`expect` on fallible operations in library paths. `try_into` on
  fixed-size slices is infallible and allowed on hot paths.
- Every `fff::ops` entry point **panics** on geometry violations. `sgraph`
  validates at its public boundary and `debug_assert`s internally, so a caller
  never reaches a panicking kernel. Public functions that can still panic carry a
  `/// # Panics` section.

## Module and doc style

- `lib.rs` and every `mod.rs` hold declarations only: module docs, `mod`,
  `pub use`, plain type declarations. **No function bodies, no `impl` blocks.**
- Each module's `//!` opens with a one-line purpose, then a submodule map, then
  (for public engine modules) a compiling doctest.
- Every public item documented; struct fields and enum-variant fields
  individually documented. Private fields get explanatory `///` too — prose
  emphasises invariants and *why*.
- Anything returning a collection on a per-symbol or per-tick path provides an
  `_into(&mut Vec<_>)` form.
- Public items are re-exported at the crate root.
- **Hard ban:** no development history in doc comments. No milestone tags, no
  phase numbering, no references to superseded designs. No references to private
  or unpublished downstream projects — which means the rustdoc must not name
  `mix-dpc`. Provenance lives in `.plans/`, which is `exclude`d from the package.
- No preset geometry constants in the API. Per `mix-dpc/AGENTS.md`: geometry is a
  channel-specific tradeoff and belongs in the README with its measurement
  conditions.

## Tests

Placement by visibility, both siblings agreeing:

- Non-public layers: `#[cfg(test)] mod tests { use super::*; … }` in-module. A
  large private suite gets its own file declared from the module
  (`#[cfg(test)] mod tests;`), as `mix-dpc` does for
  `src/internals/ldpc/tests.rs`.
- `tests/` holds integration and invariant guards:
  - `data/` — captured source fixtures with exact revision and capture command.
  - `vectors.rs` — PRNG, offset, generator-table, and protocol tuple streams.
    Changing an expected stream requires a new/versioned algorithm; file location
    is not a compatibility promise.
  - `zero_alloc.rs` — an isolated integration-test binary with one measured test,
    using a counting `#[global_allocator]` around `System`. It counts `alloc`,
    `alloc_zeroed`, and `realloc`, warms every path to the same or larger
    high-water geometry, resets the counter, then drives the deterministic
    1200-symbol stream.
  - `peeling.rs` — round-trips, cascades, reordered delivery, malformed input,
    retirement, and stopping sets.
  - `residual.rs` — exact deficiency, partial recovery, inconsistency,
    determinedness, rank additivity, and multi-iteration fixpoints.

Discipline:

- A bug fix MUST ship a regression that fails for the observed bug.
- Assert exact values, not predicates that admit unintended nonzero results.
- **Never use the implementation under test as its own oracle.** Independent
  references: a from-first-principles reference peeling decoder for small graphs,
  a naive `O(n³)` dense solve over `fff::gf8::Elem` scalars for the residual, and
  `BTreeSet`/`HashMap` for `IndexSet`/`Ring` differentials
  (`mix-dpc/src/internals/ring.rs:305` is the template).
- Recovery tests use loss patterns a plausible bug would break — stopping sets,
  bursts, reordering — not clean round trips.
- Deterministic seeds; every test isolated and full-suite safe.
- Exercise SIMD body *and* ragged tail (77-byte buffers is `mix-dpc`'s choice) and
  the `c == 0` / `c == 1` identity cases at every field boundary.
- Exercise checked `u64`/`usize` boundaries and configured live-span rejection;
  malformed public input must leave observable state unchanged.
- Zero-allocation assertions begin only after an equal-or-larger warm-up and run
  alone in their integration-test process, so unrelated test threads cannot
  perturb the global count.

## Benches

Criterion, `harness = false`, `default-features = false` with
`features = ["cargo_bench_support"]` — the defaults pull `plotters` and `rayon`
and `mix-dpc` deliberately excludes them. Groups: `peel_ingest`, `peel_cascade`,
`neighbors`, `solve`. Internal-layer groups compile out without `internals`.

Any performance change MUST be measured through the harness:
`cargo bench --features internals -- --save-baseline before`, make the change,
then `-- --baseline before`. Do not land a performance change on the strength of
reasoning alone.

## CI

`mix-dpc`'s single-job shape plus `cafft`'s MSRV and cross-target coverage. The
backend sweep and sysroot-path grep gate are `cafft`-specific and do not apply.

```yaml
on: { push: { branches: [main, master] }, pull_request: {} }
env: { CARGO_TERM_COLOR: always, RUSTFLAGS: -D warnings }

jobs:
  check:            # ubuntu, Swatinem/rust-cache@v2, components rustfmt+clippy
    - cargo fmt --all --check
    - cargo clippy --all-targets -- -D warnings
    - cargo clippy --all-targets --all-features -- -D warnings
    - cargo test
    - cargo test --all-features
    - cargo test --no-default-features
    - RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
  msrv:             # dtolnay/rust-toolchain@1.89.0
    - cargo build --all-features
  cross:
    - cargo build --target aarch64-unknown-linux-gnu --no-default-features
    - cargo build --target wasm32-unknown-unknown --no-default-features
```

Add the cross job in Phase 1, when checked `u64`→`usize` handling lands; do not
wait until the API has accumulated 64-bit-only assumptions.

Phase 6 adds **executed** generator-vector jobs on a native AArch64 runner and
`wasm32-wasip1` under Wasmtime. The Phase 1 cross job is build-only and cannot
serve as evidence for cross-platform deterministic output.

## Draft `AGENTS.md`

```markdown
# SGRAPH Engineering Invariants

These rules apply to the entire repository. They encode decisions that are
expensive to rediscover; violating one is a bug even when the tests pass.

## Scope

`sgraph` is a sparse-graph engine, not a codec. Field arithmetic and
byte-buffer vector primitives come from `fff` — never re-implement them here.
Wire formats, packet headers, transport and HARQ policy, belief-propagation
soft-decision decoding, protograph lifting, and codec shells belong to
consumers.

## Determinism

- Encoder and decoder MUST regenerate byte-identical edge sets from the same
  check id and parameters. A change to the PRNG, the sampling algorithm, the
  draw order, or a domain-separation constant is a format break for every
  downstream consumer, not a refactor.
- `tests/vectors.rs` pins PRNG, neighbour, distribution-table, and RFC tuple
  streams. Changing an expected stream requires a new/versioned algorithm.
- A point-mass degree distribution MUST consume zero RNG draws, so that
  composing a degree draw ahead of an offset draw leaves the offset stream
  unchanged.
- Domain separation is caller-supplied. Never bake a domain constant into the
  crate: seed derivation is a wire-compatibility decision.

## Allocation

- Peeling ingest, cascade, and residual solve allocate nothing in steady
  state. Scratch is owned and reused; symbol buffers and index lists are
  recycled through pools. `tests/zero_alloc.rs` warms the same-or-larger
  high-water geometry before resetting its counting allocator; extend it when
  adding an execution path.
- Validation and dispatch happen once at the public boundary, never per edge.

## Residual invariants

- For every live check row, `rhs` equals the field sum of the true values of
  the variables still in its support. Known neighbours are folded out at
  ingest and dropped; the resident structure is the residual graph.
- `deficiency == |unknowns| − rank`, computed as a pivot count. Exact, not a
  heuristic.
- The solver reaches fully **reduced** row echelon form. The per-column
  determinedness test is a single-nonzero-in-row check and is valid only under
  full reduction — stopping at echelon form breaks it silently.
- State disappears only on an explicit `retire_below`. A below-horizon index is
  gone, never absent; never conflate the two at an API boundary.
- Supports and weights have equal lengths, variables are distinct, and weighted
  coefficients are non-zero. Validate before mutating any ring, pool, or row.
- Dense index storage has configured live-span limits and checked
  `u64`→`usize` arithmetic. Limits reject input; they never trigger eviction.
- A zero-coefficient row with non-zero RHS is an inconsistent system and yields
  no recovered symbols.
- Public variable/index lookup distinguishes retired, vacant/unknown, and
  present/known states; `Option` or `bool` must not collapse gone into absent.

## Field arithmetic

- Do not write `unsafe` SIMD in this crate. There is one implementation to
  audit and it is upstream. The crate root carries `#![forbid(unsafe_code)]`.
- Call `fff::ops` directly through `EdgeWeight`. Do not add a wrapper module
  that renames `add_assign`/`mul_add`/`mul_assign` — one convention only.
- `fff::ops` panics on geometry violations. Validate at the public boundary and
  `debug_assert` internally so a caller never reaches a panicking kernel.
- GF(2) XOR is `fff::ops::add_assign::<Gf8>`: `fff` has no `Gf2`, and XOR of a
  packed element array is XOR of bytes for any field. `Gf8` is an arbitrary
  witness; say so at the call site.

## Public surface

- Everything outside `internals` is the compatibility promise. Re-export it at
  the crate root and document it — `missing_docs` and
  `missing_debug_implementations` are warnings crate-wide.
- Anything returning a collection on a per-symbol path provides an
  `_into(&mut Vec<_>)` form.
- Do not ship preset geometries. Degree, domain and overhead are
  channel-specific tradeoffs; they belong in the README with their measurement
  conditions.
- Vocabulary is graph-theoretic: variable/check, not source/repair.

## Testing changes

- A bug fix MUST include a regression that fails for the observed bug.
- Never use the implementation under test as its own oracle. Independent
  references: a naive reference peeler, a naive dense solve over scalar field
  elements, `BTreeSet`/`HashMap` for the index containers.
- Recovery tests MUST use loss patterns a plausible bug would break —
  stopping sets, bursts, reordering — not clean round trips.
- Assert exact values, never predicates that admit unintended nonzero results.
- Any performance change MUST be measured through the criterion harness with a
  saved baseline. Never land one on the strength of reasoning alone.
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
```
