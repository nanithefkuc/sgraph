# Roadmap

Each phase ends on an observable, checkable result. Captured fixtures, exact
vectors, failure-path state checks, allocation counts, and cross-crate tests are
named explicitly; a compile-only scaffold is not evidence for an algorithm.

Phase 5 is the extraction proof. Earlier phases establish the contracts in
isolation; the moment `mix-dpc` runs on `sgraph` with wire fingerprints,
residual results, and zero-allocation budgets unchanged, the extraction is
demonstrated end to end rather than argued correct.

---

## Phase 0 — Freeze the source contract and scaffold

1. Verify that the extraction source is exactly
   `6d7d4ac9fafc70c9eeed67ba8cfe654888d390c8` before capturing anything.
2. Run that revision and capture immutable fixture data **before editing it**:
   - `SplitMix64::next_u64` and `below(bound)` streams for several seeds,
     including zero and bounds that exercise Lemire rejection;
   - `distinct_offsets(neighbor_seed(id), span, k)` for a spread of ids and the
     `span == k`, `k == 1`, `k == 0`, and `k = min(dc, span)` boundaries;
   - the residual deficiency cases in `mix-dpc/tests/hdpc.rs:{79,146,228}`.
3. Keep only graph/solver fixtures in `sgraph/tests/data/`. Real packet/check
   fingerprints stay in `mix-dpc/tests/interop.rs`: `sgraph` does not own packet
   construction or an encoder loop, so duplicating those assertions here would
   either test out-of-scope code or create a second oracle.
4. Add `Cargo.toml`, `LICENSE`, `src/lib.rs`, `src/error.rs`, `AGENTS.md`,
   `README.md`, and `.github/workflows/ci.yml` with the crate boundary and lints
   from [`05-conventions.md`](05-conventions.md). Pin `fff` to the **same**
   `0077ef4463310653d5f18c17a9a5f12b734d04a8` revision used by the frozen
   `mix-dpc`; upgrading `fff` during extraction would confound the proof. Remove
   the template `add` function and test.

Fixture headers record the source revision and capture command. Changing a
pinned expected stream requires a new algorithm/versioned generator; moving a
test file is not itself a format break.

**Acceptance:** the exact source revision was observed; fixture values came from
a live run rather than source inspection; `cargo fmt --all --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test`,
`cargo test --no-default-features`, and warning-free rustdoc pass on the
scaffold.

---

## Phase 1 — Checked substrate: ids, `index`, `rng`

Add `VarId`/`CheckId` transparent newtypes. Take `Ring<T>`, `Drain`, `IndexSet`,
`BitIter`, and `SplitMix64`; lift `distinct_offsets` to take
`&mut SplitMix64`, keeping a seeded convenience wrapper.

This is not a verbatim container move:

- replace unchecked `base + len`, `u64 as usize`, and matrix/span products with
  checked geometry and errors;
- put maximum live variable/check spans in configuration so a far-away id cannot
  request a gap-sized allocation;
- make insertions below a retired base return an error, and make lookups
  distinguish `Retired` from `Vacant`/not-present rather than returning the same
  `None`/`false` for both;
- expose `SplitMix64::below(NonZeroU32)` so zero bounds are unrepresentable, and
  make the public `distinct_offsets` validate `out.len() <= span` before
  modifying caller output;
- implement `IndexSet::range` from the first relevant bitmap word, with
  `O((hi-lo)/64 + output)` complexity, rather than scanning from the set base;
- make every operation compile and behave on 32-bit `usize`.

Carry across the `Ring` sliding-span and `IndexSet`/`BTreeSet` differential
tests, extending them with `u64::MAX`, empty/full retirement, sparse-gap rejection,
and 32-bit conversion boundaries.

**Acceptance:** the captured PRNG/offset fixtures pass bit-exactly through the
new `&mut SplitMix64` API; retired and never-inserted lookups are observably
different; zero-bound construction and span-too-small calls are rejected without
partial output; rejected growth leaves state unchanged; and the library builds
for `wasm32-unknown-unknown` with `--no-default-features`. If the offset fixture
diverges, stop: every later phase inherits that wire break.

---

## Phase 2 — Binary generative half: `weight`, `degree`, `neighbors`

Add the generic `EdgeWeight` and `ResidualCoeff<F>` seams, but implement only
`Binary` here. Add `DegreeDistribution` + `Constant`, and `NeighborGen` with its
associated `Weight`, `NeighborBuf<W>`, `Edges<W>`, `Uniform`, and
`WindowedUniform`.

Constructors validate degree/domain arithmetic. Edge ingest validates parallel
lengths, uniqueness, and non-zero weights before state mutation. Generated order
is deterministic but not sorted. `EdgeWeight::ELEMENT_BYTES` lets
`Peeler::new` reject packed-symbol misalignment before a kernel call; it is `1`
for `Binary` and `F::BYTES` for Phase 7's `Weighted<F>`.

**Acceptance:**

- `WindowedUniform<Constant>` reproduces the captured `mix-dpc` offsets
  bit-exactly, including `k = min(dc, span)`;
- cloning a PRNG before and after `Constant::sample` proves the distribution
  consumes zero state;
- `Uniform` covers degree zero rejection, degree equal to domain, and
  degree-over-domain rejection;
- `WindowedUniform` rejects `base + span` overflow;
- a deliberately failing `NeighborGen` leaves `NeighborBuf` empty and reusable,
  proving error paths obey the scratch contract;
- `size_of::<Binary>() == 0`, and growing `Vec<Binary>` to 10⁶ elements performs
  zero allocations in an isolated counting-allocator test.

---

## Phase 3 — Peeling: `peel`

Implement `Peeler<Binary>`: reduce at ingest, reverse adjacency, permissive LIFO
ripple, iterative cascade, buffer/key-list pooling,
and explicit retirement. A row's cached `min_var` is the minimum of its
**current residual support** and is updated whenever a variable is removed.
Retirement remains `O(live checks)` and drops only rows that still depend on a
now-retired unknown.

All public mutation methods return `Result` and validate symbol length, retired
ids, monotone horizons, edge shape, and configured live spans before mutation.
`retire_check(CheckId)` provides explicit consumer-policy removal and recycles
the same state without requiring a variable horizon.

**Acceptance:**

- reproduce the four peeling cases from
  `internals/ldpc/tests.rs:{49,62,80,100}`: passthrough, sparse losses,
  checks-before-variables, and multi-hop cascade;
- a hand-built textbook stopping set asserts the exact `stalled_rows()` support,
  weights, and RHS values, not merely failed recovery;
- retirement tests both sides of the subtle case: a row still containing a
  below-horizon unknown is dropped, while a row whose original minimum was
  already folded out is retained and can still recover a newer variable;
- malformed lengths, duplicate variables, retired ids, and oversized gaps
  return exact errors and leave counts, rows, pools, and known values unchanged;
- explicit `retire_check` recycles the row and tolerates stale
  reverse-adjacency/ripple ids;
- after warming every pool and ring to the test's high-water geometry,
  `tests/zero_alloc.rs` observes zero allocations for steady-state ingest,
  cascade, and retirement over a deterministic 1200-symbol stream.

---

## Phase 4 — Residual solve and fixpoint resolver

Add `Row<F>`, `DenseRow<F>`, `ResidualBuilder<F>` with the single-pass push API,
scratch-owning `Solver<F>`, the explicit `DenseRows<F>` consumer contract, and a
scratch-owning `Resolver`. The resolver takes the consumer's complete
`IndexSet` of unknown variables; it never infers columns from sparse supports.
Solver coefficients are `F::Elem`; symbol rows remain packed bytes and require
`symbol_len % F::BYTES == 0`. Recovered ids and pivot rows live in reusable
scratch and are exposed as borrowed views; the resolver feeds them through the
peeler without public revalidation.

**Acceptance:**

- with unknown variables but zero sparse and zero dense rows, the resolver
  reports every supplied column undetermined rather than inferring an empty
  system from the peeler;
- exact deficiency from `mix-dpc/tests/hdpc.rs:146`: with zero sparse rows,
  `deficiency == losses - independent_dense_rows`;
- the concrete rank-deficient case from `tests/hdpc.rs:79` and rank additivity
  from `tests/hdpc.rs:228`, with exact recovered and undetermined id sets;
- a fixpoint case where solve recovery enables a sparse cascade, which reduces a
  dense row and enables a second solve;
- `DenseRow` folds each known column exactly once: a repeated resolve with no new
  knowledge leaves coefficients/RHS unchanged, while one new known value changes
  the row exactly once;
- alternating admitted/rejected dense rows retain the correct term-list/RHS
  pairing, guarding the two-pass replay bug the builder replaces;
- a determinedness regression whose answer differs between echelon and fully
  reduced echelon form;
- duplicate terms combine in the field; unknown columns must be sorted/distinct;
  RHS alignment and every `rows × cols × element_width` product are checked;
- a contradictory pair of equations returns
  `SolveError::InconsistentSystem`, publishes no recovered view from that solve,
  and teaches the peeler nothing;
- after a same-or-larger warm-up solve, the counting allocator observes zero
  allocations for builder assembly, RREF, borrowed recovery, the peeler's
  internal copy path, and the complete multi-iteration resolver.

---

## Phase 5 — Extraction proof: migrate `mix-dpc`

Immediately before migration, capture the `mix-dpc` benchmark baseline with
`cargo bench --features internals -- --save-baseline before` and record the
machine, toolchain, features, and geometry.

Rewrite `mix-dpc` to depend on the pinned `sgraph` revision. Replace
`internals/ldpc/decoder.rs` with a thin wire/header adapter around `Peeler` and
remove graph generation from `internals/ldpc/encoder.rs`. Delete
`internals/ring.rs`, `internals/solver.rs`, `internals/symbol.rs`, and the
now-unused local RNG module.
Keep the downstream `internals/ldpc` shell, `SourceWindow`, and thin encoder
accumulation loop; both adapters call the same `sgraph::WindowedUniform`.
`packet.rs`, `stream.rs`, codec policy, and `internals/hdpc.rs` remain downstream;
HDPC calls `fff::ops` directly and implements `DenseRows<Gf8>`.

The adapter retains each check's original `window_base` and calls
`retire_check` under the old horizon rule, so migration does not gain equations
from the core's more precise residual-support retirement. Fixing
`SourceWindow`'s private indexing contract remains a separate bug fix.

Clean cutover: no shims, aliases, duplicate implementations, or deprecated
paths. `mix-dpc`'s `internals` feature exposes only what it still owns.

**Acceptance — this decides whether the extraction is correct:**

- `mix-dpc/tests/interop.rs` and `tests/packet.rs` pass unchanged;
- `tests/{hdpc,stream}.rs` pass with only mechanical API/import changes and exact
  expected values unchanged;
- `tests/zero_alloc.rs` passes unchanged or with lower budgets, never higher;
- both crates pass their default, all-feature, and no-default-feature matrices,
  clippy, rustdoc, and target builds;
- `cargo bench --features internals -- --baseline before` shows no regression in
  the symbol, solver, and stream groups under the recorded conditions;
- `cargo run --release --example sim_4g` reproduces the README residual figures
  (about 0.050% mixed and 1.09% sparse-only at the frozen geometry).

A changed interop fingerprint or residual rate is a graph/wire break, not a
test-update opportunity.

---

## Phase 6 — Broaden: deterministic distributions and fixed graphs

Only after the migration proof:

- `RobustSoliton::from_q32(k, c_q32, delta_q32)` using the specified
  fixed-point, integer-only cumulative table, plus an explicit cumulative-table
  constructor; no platform-float constructor;
- the RFC 5053 degree distribution and its normative random tables;
- `Rfc5053Triple`, explicitly RFC 5053 Raptor—not RaptorQ/RFC 6330;
- validated `ExplicitMatrix` CSR ingestion.

Extend CI from cross-builds to cross-execution for the generator vectors: a
native AArch64 runner and `wasm32-wasip1` under Wasmtime. Build-only jobs do not
prove cross-peer fingerprints.

**Acceptance:**

- robust-soliton cumulative weights match an independent high-precision
  reference after the documented quantization, and table/degree fingerprints
  match on x86_64, AArch64, and Wasm;
- RFC degree and `(d,a,b)` tuple vectors match the RFC algorithm independently;
- a fixed LT configuration (parameters and seed corpus committed before tuning)
  matches an independent reference edge-for-edge and reports its recovery rate;
  the rate is a quality measurement, while vector equality is the correctness
  gate;
- invalid `k`, zero `c_q32`, out-of-range `delta_q32`, non-monotone CDFs, and
  cumulative-weight overflow return exact construction errors;
- `ExplicitMatrix` rejects malformed CSR (offset, column, duplicate, and
  dimension errors), returns an error with empty reusable scratch for an
  out-of-range check row, and reproduces exact recovery/stall results for a
  named small parity-check matrix across exhaustively enumerated erasure
  patterns.

No claim of RaptorQ support or a “published erasure threshold” is made without
implementing and testing that distinct algorithm/ensemble.

---

## Phase 7 — Weighted peeling

Implement non-zero `Weighted<F>`, deterministic weight generation with its own
domain separator and pinned draw order, weighted ingest folding,
`apply_known`, inverse scaling at degree one, and `ResidualCoeff<F>` for the same
field.

**Acceptance:**

- weighted peeler construction rejects `symbol_len % F::BYTES != 0` before
  invoking any `fff` kernel;
- hand-computed GF(256) equations with non-unit coefficients cover ingest folding,
  inverse scaling, reordered arrivals, and a multi-hop cascade;
- zero coefficients and cross-field residual embedding are rejected at compile
  time or construction, never interpreted as valid edges;
- for identical topology, binary and weighted peeling stall/recover the same
  variable ids (stopping sets are topological), while recovered values match
  their respective field equations;
- packed symbol tests cover `F::BYTES`, SIMD-sized bodies, ragged tails, and
  coefficient identities;
- the Phase 5 binary benchmarks and zero-allocation tests show no regression
  under the same recorded conditions.

---

## Phase 8 — Documentation and release hygiene

Run this immediately after the chosen release scope: Phase 5 for 0.1, or after
optional Phases 6/7 if they are included. It is gated on Phase 5 and never starts
before the extraction works end to end.

- Finalize `README.md` and `AGENTS.md` against the implemented API: scope,
  quick-start doctest, module map, invariants, feature flags, determinism/version
  policy, and measured geometry only.
- Document every public item; make `cargo doc --all-features` and the crate-level
  doctest warning-free. Keep development history and downstream-private names
  out of rustdoc.
- Keep benchmark **code** for ingest, cascade, neighbours, and solve. Do not
  commit Criterion baseline artifacts: they are machine-specific; record
  comparison conditions and results instead.
- Verify `cargo package --list` excludes `.plans/` and `.github/`, includes the
  license/readme, and contains no scaffold files.
- Run the final default/all-feature/no-default feature matrix, clippy, format,
  rustdoc, cross-target builds, and both `mix-dpc` end-to-end proofs.
