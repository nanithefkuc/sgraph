# Charter

## What `sgraph` is

`sgraph` is a **sparse-graph engine, not a codec.** It owns the bipartite
(Tanner) graph between variable nodes and check nodes, the deterministic
generation of that graph's edges, the peeling decoder that consumes it, and the
exact residual solve that finishes what peeling cannot.

It is the layer that LDPC, LT, and Raptor-class implementations keep rewriting:

- deterministic, cross-peer-reproducible neighbour generation from a check id
- degree distributions (constant, robust soliton, RFC 5053 table)
- a residual sparse graph that shrinks as symbols become known
- XOR-only (and, generalized, GF(2^m)) peeling with a reverse-adjacency ripple
- the stalled-peeling → dense residual solve → re-peel fixpoint, with an
  **exact** rank/deficiency signal
- index-keyed dense storage and buffer recycling that make all of the above
  allocation-free in steady state

## Scope boundary

Copy this into `src/lib.rs` and `AGENTS.md` verbatim:

> `sgraph` is a sparse-graph engine, not a codec. Field arithmetic and
> byte-buffer vector primitives come from `fff` — never re-implement them here.
> Wire formats, packet headers, transport and HARQ policy, belief-propagation
> soft-decision decoding, protograph lifting, and codec shells belong to
> consumers.

This is not decoration. It is the rule that decides every borderline item in
[`01-extraction-map.md`](01-extraction-map.md).

### In scope

| Area | Why it is here |
| --- | --- |
| Tanner-graph topology and residual representation | The crate's reason to exist. |
| `NeighborGen` implementations (uniform-over-domain, windowed uniform, RFC 5053 triple, explicit matrix) | Edge generation *is* graph structure. |
| `DegreeDistribution` implementations | Inseparable from edge generation. |
| Deterministic PRNG + distinct-k sampling | Both peers must regenerate identical edges; this is graph machinery, and the family has no `rand` dependency anywhere. |
| Peeling decoder + ripple + cascade | Identical for LDPC-erasure, LT, and Raptor. |
| Residual dense solve: RREF, exact rank, exact deficiency, per-column determinedness | The unavoidable counterpart of "peeling stalls on stopping sets". Raptor inactivation decoding needs the same thing. |
| The peel↔solve fixpoint driver | Identical for every consumer; see `mix-dpc` `codec.rs:463-478`. |
| `Ring<T>` / `IndexSet` index-keyed storage, buffer and key-list pooling | The mechanism that earns zero-allocation; generic and FEC-agnostic. |

### Out of scope

| Area | Where it belongs |
| --- | --- |
| Field arithmetic, SIMD kernels, backend dispatch | `fff`. `sgraph` writes no `unsafe` and no field loops. |
| Wire formats, packet headers, byte layouts | Consumer. `RepairHeader` (`packet.rs:40`) must not be named by `sgraph`. |
| Transport policy: TTL deadlines, retirement horizons, NACKs, HARQ, emission rate | Consumer (`mix-dpc` `stream.rs`). `sgraph` exposes retirement as a mechanism; *when* to retire is policy. |
| Dense MDS parity: Cauchy/Vandermonde recipes, `X={0..k-1} / Y={k..k+m-1}` conventions, generation tiling | Consumer, or a dense-RS crate. `fff`'s own boundary statement (`fff/src/lib.rs:115-118`) assigns coding-matrix construction to a codec layer; a Cauchy block has no sparse structure. |
| Belief propagation / soft-decision decoding (min-sum, sum-product, LLR) | Consumer. `sgraph` is erasure/algebraic only. |
| Protograph lifting, quasi-cyclic expansion, matrix design/optimization | Consumer. `sgraph` consumes a graph; it does not design one. |
| Codec shells, `Config` presets, systematic-vs-nonsystematic policy | Consumer. Per `mix-dpc` `AGENTS.md`: geometry is a channel-specific tradeoff; do not ship a preset. |

### Deliberately deferred, not rejected

- **Non-binary (GF(2^m)) peeling.** The generic trait seam lands with the
  binary generator in Phase 2 and costs the binary path nothing (see
  [`03-generalization.md`](03-generalization.md)); non-zero coefficient
  generation and the weighted implementation land in Phase 7.
- **Explicit parity-check matrix ingestion** (classic LDPC from an H matrix
  rather than a generated graph). The `NeighborGen` trait admits it; the impl is
  Phase 6.

## Load-bearing invariants

Violating one of these is a bug even when the tests pass.

1. **Reproducibility.** Encoder and decoder MUST regenerate byte-identical edge
   sets from the same `(check_id, domain, parameters)`. Any change to the PRNG,
   the sampling algorithm, the draw order, or a domain-separation constant is a
   **format break for every downstream consumer**, not a refactor. Pin it with
   fingerprint tests.

2. **Bit-exact `mix-dpc` compatibility for the windowed-uniform generator.**
   `mix-dpc`'s neighbour stream is its wire format. `sgraph`'s
   `WindowedUniform` + `Constant` MUST reproduce
   `SplitMix64(check_id ^ domain)` → Floyd k-of-n over `[0, span)` with
   `k = min(dc, span)` exactly (`rng.rs:55,65`; `encoder.rs:65-70`;
   `decoder.rs:197-200`). Consequence: a constant degree MUST consume **zero**
   RNG draws, so that threading `&mut SplitMix64` through the sampler does not
   perturb the offset stream. See [`06-risks.md`](06-risks.md) R1.

3. **Steady-state zero allocation.** Peeling ingest, cascade, and residual solve
   allocate nothing once warm. `mix-dpc` achieves this for ingest but **not** for
   `solver::solve`, which allocates two matrices, a pivot map, and a `to_vec()`
   per recovered symbol every call (`solver.rs:54-55,71,138`). `sgraph` MUST fix
   that: the solver owns reusable scratch. Enforced by a counting global
   allocator, not asserted (`mix-dpc/tests/zero_alloc.rs:17-36`).

4. **The residual invariant.** For every live check row, `rhs` equals the field
   sum of the true values of the variables still listed in its support
   (`decoder.rs:18-20`). Known neighbours are folded out at ingest and dropped;
   the resident structure is the *residual* graph, shrinking monotonically.

5. **Exactness of the deficiency signal.** `deficiency == |unknowns| − rank`
   over the assembled system, computed as a pivot count, not a heuristic
   (`solver.rs:128,141`). Determinedness is a single-nonzero-in-row test, valid
   only because the form is fully **reduced** — if the solver ever stops at
   echelon rather than reduced echelon, that test silently breaks.

6. **No implicit eviction.** State disappears only on an explicit `retire_below`.
   A below-horizon index is *gone*, never *absent*; the two must not be confused
   at any API boundary (`decoder.rs:3-8`, `ring.rs:65`).

7. **Public boundary validates; kernels panic.** Every `fff::ops` entry point
   panics on geometry violations (length mismatch, partial trailing element).
   `sgraph` MUST validate at its own public boundary and never let a caller reach
   a panicking kernel. Internal call sites carry `debug_assert`s.

8. **No second convention.** `sgraph` calls `fff::ops` directly and does not
   re-host a symbol-arithmetic wrapper module. See
   [`02-architecture.md`](02-architecture.md) §"Why there is no `symbol` module".

9. **Vocabulary is graph-theoretic, not codec-flavoured.** `variable`/`check`,
   not `source`/`repair`. `mix-dpc`'s naming leaks its application; `sgraph`
   serves LDPC (where checks are parity constraints, not transmitted symbols)
   equally.

10. **Edges are canonical at the boundary.** Supports and weights have equal
    length, every variable occurs at most once, and weighted edges have non-zero
    coefficients. Generated edge order is deterministic but need not be sorted.
    Zero-weight degree-one rows and duplicate weighted terms can otherwise
    produce silently wrong peels.

11. **Dense index storage is explicitly bounded.** `Ring`/`IndexSet` are dense
    over their live span. Every public growth path uses checked `u64`→`usize`
    geometry and a configured live-span limit; a sparse or overflowed id returns
    `GraphError` rather than attempting an unbounded allocation. Limits reject
    input and never evict state.

12. **Inconsistent systems do not yield symbols.** RREF must detect a zero
    coefficient row with a non-zero right-hand side and return
    `SolveError::InconsistentSystem`. Rank and deficiency alone describe the
    coefficient matrix, not whether the augmented system has a solution.

13. **Residual columns are explicit input.** The peeler can see only variables
    that occur in retained sparse rows. A lost variable may occur only in a
    consumer dense row—or in no received row yet—so the global unknown set must
    be supplied by the consumer. Inferring solve columns from the union of
    stalled sparse supports undercounts deficiency.
