# Architecture

## Module layout

`lib.rs` and every `mod.rs` hold declarations only — module docs, `mod`,
`pub use`, plain type declarations. No function bodies, no `impl` blocks.

```
src/
  lib.rs              crate docs, scope statement, lints, declarations only
  error.rs            GraphError, SolveError

  rng.rs              SplitMix64 + distinct-k sampling primitives
  index.rs            Ring<T>, Drain, IndexSet, BitIter

  degree.rs           DegreeDistribution + Constant, RobustSoliton, Rfc5053
  neighbors/
    mod.rs            NeighborGen trait, Edges, NeighborBuf
    uniform.rs        Uniform (whole domain), WindowedUniform (bit-exact mix-dpc)
    triple.rs         Rfc5053Triple  (Phase 6)
    explicit.rs       ExplicitMatrix (Phase 6)

  weight.rs           EdgeWeight, ResidualCoeff, Binary, Weighted<F>

  peel/
    mod.rs            declarations
    graph.rs          ResidualGraph: rows, reverse adjacency, counters
    peeler.rs         Peeler: ingest, ripple, cascade, retire
    pool.rs           symbol-buffer and key-list recycling

  residual/
    mod.rs            declarations
    row.rs            Row<'a, F>, DenseRow<F> (progressive reduction)
    builder.rs        ResidualBuilder: single-pass push, column mapping, scratch
    solver.rs         Solver: scratch-owning RREF, exact rank, Report

  driver.rs           scratch-owning peel <-> solve Resolver
  internals.rs        curated unstable re-exports (feature = "internals")
```

Rationale for the split: `neighbors` + `degree` + `rng` are the *generative*
half (what the graph is), `peel` + `residual` + `driver` are the *consumptive*
half (what you do with it), `index` and `weight` are shared substrate. That is
the same shape as `cafft`'s `basis` (structure) vs `core` (engine) split.

`sgraph` has no module named `core`, so the absolute-sysroot-path rule and its CI
grep gate that `cafft` needs (`cafft/AGENTS.md` "Crate hygiene") do not apply.

## Vocabulary

| `sgraph` | `mix-dpc` | Literature |
| --- | --- | --- |
| variable node, `VarId` | source symbol index | variable / left node |
| check node, `CheckId` | repair symbol index | check / constraint / right node |
| support | `unknown` | residual neighbourhood |
| `rhs` | `reduced` | right-hand side |
| domain | window | index space |

`VarId` and `CheckId` are `#[repr(transparent)]` `u64` newtypes from Phase 1,
matching the default in [`06-risks.md`](06-risks.md) D3. They prevent confusing
a variable retirement horizon with a check key at no runtime cost. The solver
reduces variable ids to local `usize` columns; a generic id parameter adds no
useful flexibility.

## Generative half

### `DegreeDistribution`

```rust
pub trait DegreeDistribution {
    /// Draw a check degree. MUST consume zero RNG draws when the
    /// distribution is a point mass — see charter invariant 2.
    fn sample(&self, rng: &mut SplitMix64) -> u32;
    /// Upper bound, sizing neighbour scratch.
    fn max_degree(&self) -> u32;
}
```

Phase 2 implements `Constant(u32)`. Phase 6 adds
`RobustSoliton::from_q32(k, c_q32, delta_q32)`, an explicit cumulative-table
constructor, and the RFC 5053 §5.4.4.2 distribution. There is no `f64`
constructor: robust-soliton construction quantizes a documented fixed-point
formula into a canonical cumulative `u64` table, so sampling never depends on
platform math. RFC 5053 uses its normative RNG tables, not `SplitMix64`.

### `NeighborGen`

RFC 5053 Raptor neighbour selection is a `(d, a, b)` triple walked over a prime
modulus, while classic LDPC reads a fixed parity-check matrix — neither is
"uniform distinct-k with a different degree". (RaptorQ/RFC 6330 has a different
six-value tuple and is not claimed by `Rfc5053Triple`.) Neighbour selection is
therefore its own trait, and `distinct_offsets` is an implementation detail.

```rust
pub trait NeighborGen {
    type Weight: EdgeWeight;

    /// Generate the edges for `id`, clearing `out` first.
    fn neighbors(
        &self,
        id: CheckId,
        out: &mut NeighborBuf<Self::Weight>,
    ) -> Result<(), GraphError>;
    /// Upper bound on degree, for scratch sizing.
    fn max_degree(&self) -> u32;
}
```

`NeighborBuf<W>` is the pooled `(support, weights)` pair; `Edges<'a, W>` is the
borrowed view handed to the peeler. Constructors validate domain arithmetic and
maximum degree once. Generation is fallible because a finite generator such as
`ExplicitMatrix` must reject an out-of-domain `CheckId`; errors leave `out`
cleared. `Edges` validates equal lengths, non-empty support, distinct variables,
and non-zero weighted coefficients at public ingest. Generated order is
deterministic but need not be sorted.

Implementations:

- `Uniform { domain: u64, degree: D, domain_sep: u64 }` — distinct-k over a
  fixed block, seeded from `id ^ domain_sep`.
- `WindowedUniform { base: u64, span: u32, degree: D, domain_sep: u64 }` —
  **must be bit-exact with `mix-dpc`**: `SplitMix64(id ^ domain_sep)` → Floyd
  k-of-n over `[0, span)` with `k = min(degree, span)`, offsets returned as
  `base + off`. The constructor rejects `base + span` overflow. The cap is this
  generator's rule, not a core concept.
- `Rfc5053Triple` (Phase 6), implementing RFC 5053 only.
- `ExplicitMatrix` (Phase 6) — a validated CSR parity-check matrix; also the
  escape hatch for callers that bring their own graph.

Domain separation is a *caller-supplied* constant, not a baked-in mask: seed
derivation is a wire-compatibility decision, and one global
`NEIGHBOR_DOMAIN` (`mix-dpc/src/rng.rs:9`) cannot serve multiple consumers.

## Shared substrate: `EdgeWeight`

The zero-cost trick that lets one peeler serve GF(2) and GF(2^m). Detailed in
[`03-generalization.md`](03-generalization.md):

```rust
pub trait EdgeWeight: Copy + Eq + Default + 'static {
    fn one() -> Self;
    /// Packed-symbol element width; `Peeler::new` validates divisibility once.
    const ELEMENT_BYTES: usize;
    fn is_zero(self) -> bool;
    /// dst += w * src
    fn mul_add(dst: &mut [u8], w: Self, src: &[u8]);
    /// value *= w⁻¹, in place; public validation guarantees w != 0.
    fn scale_inv(value: &mut [u8], w: Self);
}

/// Embed a sparse edge coefficient into the residual solver's field.
pub trait ResidualCoeff<F: FieldKernels>: EdgeWeight {
    fn coefficient(self) -> F::Elem;
}
```

`Binary` is a ZST with `ELEMENT_BYTES = 1`: `is_zero` is always false,
`mul_add` forwards to `fff::ops::add_assign::<Gf8>`, and `scale_inv` is a no-op.
`ResidualCoeff<F>` maps it to `F::Elem::ONE`. `Weighted<F>(F::Elem)` uses
`ELEMENT_BYTES = F::BYTES`, forwards to `mul_add` / `mul_assign`, maps to its
stored coefficient only for the same field, and rejects zero weights at ingest.

Because `Binary` is a ZST, `Vec<Binary>` never allocates and carries no
per-element cost, so the peeler stores parallel `Vec<VarId>` / `Vec<W>` support
vectors with **zero overhead in the binary case** — no specialization machinery,
runtime branching, or duplicated engine.

## Consumptive half

### `Peeler<W: EdgeWeight>`

Direct descendant of `LdpcDecoder` (`mix-dpc/src/internals/ldpc/decoder.rs:30`),
with the wire type and window concept removed.

```rust
pub struct Peeler<W> {
    symbol_len: usize,
    known: Ring<Option<Vec<u8>>>,          // decoder.rs:34
    known_count: usize,
    rows: Ring<Option<CheckRow<W>>>,       // decoder.rs:38
    row_count: usize,
    waiting: Ring<Vec<CheckId>>,           // reverse adjacency, decoder.rs:42
    ripple: Vec<CheckId>,                  // permissive LIFO, decoder.rs:44
    recovered: Vec<VarId>,                 // decoder.rs:46
    unresolved: usize,                     // O(1) stall predicate, decoder.rs:48
    neighbor_buf: NeighborBuf<W>,           // `push_check_with` scratch
    pool: Pool,                            // decoder.rs:50-58
}

struct CheckRow<W> {
    rhs: Vec<u8>,
    support: Vec<VarId>,
    weights: Vec<W>,        // allocation-free when W = Binary
    min_var: Option<VarId>, // minimum of current support; updated on removal
    resolved: bool,
}
```

`PoolConfig` includes symbol/key pool caps and maximum live variable/check spans.
All dense-ring growth uses checked arithmetic and returns `GraphError` when an id
would exceed those spans; it never silently evicts or attempts a gap-sized
allocation.

The three properties worth preserving verbatim from `mix-dpc`:

- **Forward adjacency is implicit and consumed.** Edges are generated once at
  ingest and immediately reduced: known neighbours are folded into `rhs` and
  dropped. The resident structure is only the residual
  (`decoder.rs:197-211`). Nothing stores the full graph.
- **Reverse adjacency is explicit** (`waiting`), making each newly-known variable
  an `O(deg)` update rather than a scan over all rows.
- **The ripple is permissive.** Duplicate, obsolete, and retired keys are all
  allowed and re-validated on pop (`decoder.rs:357-362`). That is what makes
  enqueueing free. `apply_known` deliberately does not drive the loop, so
  recursion is flattened into a `while let` (`decoder.rs:312,356`).

Public surface (the seam of [`01-extraction-map.md`](01-extraction-map.md)):

```rust
impl<W: EdgeWeight> Peeler<W> {
    pub fn new(symbol_len: usize, cfg: PoolConfig) -> Result<Self, GraphError>;

    pub fn learn(&mut self, var: VarId, value: Vec<u8>) -> Result<(), GraphError>;
    pub fn learn_copy(&mut self, var: VarId, value: &[u8]) -> Result<(), GraphError>;
    pub fn push_check(
        &mut self,
        id: CheckId,
        edges: Edges<'_, W>,
        rhs: &[u8],
    ) -> Result<(), GraphError>;
    pub fn push_check_with<G: NeighborGen<Weight = W>>(
        &mut self,
        id: CheckId,
        gen: &G,
        rhs: &[u8],
    ) -> Result<(), GraphError>;

    pub fn variable_state(&self, var: VarId) -> VariableState<'_>;
    pub fn has_stalled(&self) -> bool;
    pub fn stalled_rows(&self) -> impl Iterator<Item = StalledRow<'_, W>>;
    pub fn drain_recovered_into(&mut self, out: &mut Vec<VarId>);

    pub fn retire_below(&mut self, horizon: VarId) -> Result<(), GraphError>;
    pub fn retire_check(&mut self, id: CheckId) -> Result<(), GraphError>;
    pub fn take_recycled(&mut self) -> Option<Vec<u8>>;
}
```

`VariableState` is `Retired`, `Unknown`, or `Known(&[u8])`; there is no public
boolean/`Option` lookup that collapses the first two states. `Ring` and
`IndexSet` expose the same distinction (or a retired-index error) at their public
lookup boundaries.

Every fallible method validates symbol length, retired ids, edge shape, non-zero
weights, monotone horizons, and checked index span before mutating state.

`retire_below` removes variable state and rows that still depend on a retired
unknown. `retire_check` lets a consumer impose stricter check-lifetime policy
without leaking window/TTL concepts into the core. Both recycle row buffers and
leave stale ripple/reverse-adjacency ids safe to ignore.

Duplicate check ids and already-known variables preserve `mix-dpc`'s idempotent
first-value-wins behaviour after that validation.

Per `mix-dpc/AGENTS.md`: anything returning a collection on a per-symbol path
gets an `_into(&mut Vec<_>)` form.

### `residual::Solver<F: FieldKernels>`

`mix-dpc`'s `solve` is a free function that allocates two matrices, a pivot map,
and a `Vec` per recovered symbol on every call
(`solver.rs:54-55,71,138`). `sgraph` makes it a struct that owns that scratch:

```rust
pub struct Solver<F: FieldKernels> {
    coeffs: Vec<F::Elem>,    // n_rows × n_cols, row-major
    symbols: Vec<u8>,        // n_rows × symbol_len packed field elements
    pivot_of_col: Vec<usize>,
    recovered: Vec<(VarId, usize)>, // id + pivot row, reused
    undetermined: Vec<VarId>,
    _field: PhantomData<F>,
}

impl<F: FieldKernels> Solver<F> {
    pub fn new() -> Self;
    pub fn solve(&mut self, sys: &System<'_, F>) -> Result<Report, SolveError>;
    pub fn recovered(&self) -> impl Iterator<Item = (VarId, &[u8])>;
    pub fn undetermined(&self) -> &[VarId];
}

pub struct Report { pub rank: usize, pub deficiency: usize }
```

Preserved from the original: full **reduced** row echelon form (the
determinedness test at `solver.rs:129-140` is only valid because the form is
reduced); flat row-major matrices so the pivot row is borrowable via
`split_at_mut`; `rank = pivot_row`; duplicate terms accumulate rather than
overwrite.

Changed: coefficient scratch is `F::Elem`; coefficient-row elimination uses the
safe scalar `fff::field::Elem` operations, while packed RHS rows use `fff::ops`.
`symbol_len % F::BYTES == 0` and all matrix products are checked, scratch is
reused, and recovered symbols are borrowed directly from pivot rows until the
next solve. A zero coefficient row with non-zero RHS returns
`SolveError::InconsistentSystem`; no recovered view is published.

### `residual::ResidualBuilder`

Replaces `codec::build_rows`, whose two passes replay their admission filters
verbatim and zip by position — a fragility the code documents about itself
(`codec.rs:521-523`). Single-pass push instead:

```rust
impl<'a, F: FieldKernels> ResidualBuilder<'a, F> {
    pub fn begin(&mut self, unknowns: &[VarId]) -> RowSink<'_, F>;
}

impl<'a, F: FieldKernels> RowSink<'a, F> {
    /// A stalled sparse row embedded into `F`.
    pub fn push_sparse<W: ResidualCoeff<F>>(&mut self, row: StalledRow<'_, W>);
    /// Any dense row the consumer owns: its field, its coefficients.
    pub fn push_dense(
        &mut self,
        terms: impl Iterator<Item = (VarId, F::Elem)>,
        rhs: &'a [u8],
    );
    pub fn finish(self) -> Result<System<'a, F>, SolveError>;
}
```

Admission (window bounds, spentness, liveness) stays with the consumer — it
simply does not call `push_*` for a row it does not want. Column mapping, term
deduplication, and term-buffer reuse (`codec::RowScratch`, `codec.rs:556-569`)
come here.

The residual solve is **always over a real field**, even when the sparse layer is
binary: elimination requires division. Binary rows widen to coefficient `ONE` at
the builder, which is exactly what `mix-dpc` does today (`codec.rs:508`).

### `driver`

The global unknown set cannot be inferred from `Peeler`: a variable may be
missing without occurring in any retained sparse row. The consumer already owns
loss discovery (`mix-dpc` calls it `missing`), so the fixpoint takes that
`IndexSet` explicitly. A scratch-owning resolver prevents the column snapshot and
recovery drain from allocating per call:

```rust
pub struct Resolver {
    columns: Vec<VarId>,
    recovered: Vec<VarId>,
}

impl Resolver {
    pub fn resolve<W, F, D>(
        &mut self,
        unknowns: &mut IndexSet,
        peeler: &mut Peeler<W>,
        dense: &mut D,
        solver: &mut Solver<F>,
        builder: &mut ResidualBuilder<'_, F>,
    ) -> Result<Report, SolveError>
    where
        W: EdgeWeight + ResidualCoeff<F>,
        F: FieldKernels,
        D: DenseRows<F>;
}
```

`DenseRows<F>` is the consumer seam: report whether any dense row is live,
progressively fold `VariableState::Known` values exactly once, and push selected
borrowed rows into `RowSink`. Admission (window bounds, spentness, retirement)
remains consumer policy. The resolver admits a sparse row only when every
residual support id is in the supplied unknown set.

Loop: drain peeled ids and remove them from `unknowns` → snapshot the sorted set
into `columns` → reduce dense rows → if no equation exists, publish rank zero,
`deficiency = unknowns.len()`, and the full column list as undetermined →
assemble → solve → remove each borrowed recovery from `unknowns`, feed it through
the peeler's validated internal copy path, and re-peel → repeat until a solve
recovers nothing. Empty-unknown and no-equation exits explicitly clear/replace
solver outcome scratch so `solver.recovered()`/`undetermined()` cannot expose a
previous solve. Inconsistency clears recovery metadata and aborts without
learning output from that solve.

Direct variable arrivals remain consumer policy: the consumer calls
`Peeler::learn` and removes that id from its unknown set before resolving. The
`dirty`/`stale` optimization around resolution (`codec.rs:174-175`) also remains
downstream.

## Why there is no `symbol` module

`mix-dpc`'s `internals/symbol.rs` is four single-statement forwards to
`fff::ops::{add_assign, mul_add, mul_add_scatter, mul_assign}` with
`debug_assert`s and no independent logic. Re-hosting it in `sgraph` would create
a second name for each operation beside `fff`'s already-canonical ones
(`fff/README.md:85-89` fixes the naming: `dst ^= src` → `add_assign`,
`dst ^= c*src` → `mul_add`, `dst = c*src` → `mul_into`, `dst *= c` →
`mul_assign`). `sgraph` calls `fff::ops` directly, through `EdgeWeight`.

Its GF(256) conformance tests (`symbol.rs:51-65` reimplements textbook 0x11B as
an independent oracle) validate `fff`'s SIMD dispatch, not `mix-dpc` behaviour;
they belong upstream in `fff`.

One concession to readability: GF(2) XOR is reached as
`fff::ops::add_assign::<Gf8>(dst, src)` because `fff` has **no `Gf2` type** and
XOR of a packed element array is XOR of bytes regardless of field. That reads
oddly enough to deserve a single private `#[inline] fn xor` inside
`weight.rs` with a comment naming `Gf8` as an arbitrary witness. The alternative,
`fff::kernel::xor`, is public only under `fff/internals`; `mix-dpc` does not
enable that feature and neither should `sgraph`.
