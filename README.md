# sgraph

**Sparse graph** — a shared sparse/Tanner-graph engine for erasure codes.

`sgraph` is the layer that LDPC, LT, and Raptor-class implementations keep
re-implementing: deterministic cross-peer neighbour generation, degree
distributions, a residual sparse graph that shrinks as symbols become known,
XOR-only peeling, and the exact residual solve that finishes what peeling cannot.
These codes differ in how their graph is generated, not in how it is consumed.

Field arithmetic and byte-buffer vector primitives come from
[`fff`](https://github.com/nanithefkuc/fff); this crate never re-implements field
arithmetic. Wire formats, packet headers, transport and HARQ policy,
belief-propagation soft-decision decoding, protograph lifting, and codec shells
stay with the consumer.

## Status

Complete for the binary (GF(2)) path: deterministic sampling, the bounded
index-keyed containers, degree distributions, the uniform/RFC-5053/
explicit-matrix edge generators, the peeling decoder, the exact residual solve,
and the peel → solve → re-peel fixpoint. `Binary` is the only implemented
`EdgeWeight`; `ResidualCoeff<F>` embeds it into any `fff` field, but a non-binary
*edge* coefficient is a declared seam rather than a working implementation.

Not on crates.io, and not planned: it depends on `fff` by git, so depend on it the
same way.

```toml
[dependencies]
sgraph = { git = "https://github.com/nanithefkuc/sgraph" }
```

Requires Rust 1.89 or newer (edition 2024).

## Determinism

A check symbol travels without its graph: both peers regenerate its edge set from
the check's id alone. That makes the PRNG, the sampling algorithm, and the draw
order wire properties, frozen by the fixtures under `tests/data/`.

```rust
use sgraph::rng::{SplitMix64, distinct_offsets, seed_for};

// The domain-separation constant is yours to choose; it keeps this edge stream
// distinct from any other use of the same check ids.
const DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;

let mut edges = [0u32; 3];
let mut rng = SplitMix64::new(seed_for(42, DOMAIN));
distinct_offsets(&mut rng, 64, &mut edges)?;

// The far side, holding only the check id, recomputes the same set.
let mut peer = [0u32; 3];
let mut peer_rng = SplitMix64::new(seed_for(42, DOMAIN));
distinct_offsets(&mut peer_rng, 64, &mut peer)?;
assert_eq!(edges, peer);
# Ok::<(), sgraph::GraphError>(())
```

Sampling takes `&mut SplitMix64` rather than a seed so that a degree draw and an
edge draw compose into one reproducible stream. A degree drawn from a separately
seeded generator would leave the two independent, which is how correlated-graph
bugs get in.

## What's in the box

| Module | Contents |
| --- | --- |
| `rng` | `SplitMix64`, unbiased bounded draws over a `NonZeroU32`, and Floyd distinct-k sampling. Allocation-free, and validated before it writes. |
| `index` | `Ring<T>` and `IndexSet`: dense index-keyed storage over a monotone `u64` domain, where a lookup is a subtraction and retirement is a front drain that hands back what it dropped. Bounded by a configured live span. |
| `weight` | `EdgeWeight` and `ResidualCoeff<F>`, plus `Binary` — a zero-sized GF(2) coefficient, so one generic engine serves GF(2) and GF(2^m) without taxing the binary path. |
| `degree` | `DegreeDistribution` with `Constant` (a point mass that consumes no randomness), `Cumulative` (an explicit integer weight table), and `RobustSoliton` (built from Q32 fixed-point parameters, so the table is identical on every platform). |
| `neighbors` | `NeighborGen` with the `Uniform`, `WindowedUniform`, `Rfc5053Triple` (RFC 5053 Raptor, not RaptorQ) and `ExplicitMatrix` (CSR parity-check) generators, the reusable `NeighborBuf` scratch, and `Edges` — the one place edge shape is validated. |
| `peel` | `Peeler`: the residual sparse graph, reverse adjacency, iterative degree-one cascade, pooled buffers, and explicit retirement. `StalledRow` exposes what peeling could not finish. |
| `residual` | `ResidualBuilder`/`RowSink` single-pass assembly over explicit columns, and `Solver` — full reduced row echelon form over any `fff` field. |
| `driver` | `DenseRows`, the consumer seam for progressively reduced dense equations, and `Resolver`, the peel → solve → re-peel fixpoint. |
| `id` | `VarId` and `CheckId`: `#[repr(transparent)]` newtypes so a variable horizon cannot be passed where a check key belongs. |
| `error` | `GraphError` and `SolveError`. |

## Invariants

- **Reproducibility is a wire property.** A change to the generator, the sampling
  algorithm, or the draw order is a format break for every consumer, not a
  refactor.
- **A retired index is gone, not absent.** An index merely below `base` that was
  never retired is absent, and inserting it grows the front. `Lookup` and
  `Membership` report which of the three states an index is in; nothing is
  evicted implicitly.
- **Dense storage is bounded by construction.** Both containers take a maximum
  live span and check it — along with the `u64`→`usize` conversion — on every
  growth path. A rejection leaves the structure exactly as it was: limits reject
  input, they never evict state to make room.
- **Steady state allocates nothing.** Scratch is owned and reused, and retirement
  recycles buffers rather than dropping them.
- **No `unsafe`.** The crate root carries `#![forbid(unsafe_code)]`; every SIMD
  kernel is upstream in `fff`, where there is one implementation to audit.

## Feature flags

- `std` (default) — enables `fff`'s runtime CPU detection and its process-wide
  backend cache. Without it the crate is `no_std` + `alloc`.
- `simd` (default, implies `std`) — runtime-dispatched SIMD kernels from `fff`.
- `internals` — unstable implementation APIs, exempt from compatibility
  guarantees.

## Determinism and version policy

- **The generator is the format.** The PRNG, the sampling algorithm, the draw
  order, and the domain-separation input are all observable to a peer that
  regenerates an edge set from a check id. Changing any of them is a format
  break, and a format break gets a new or versioned generator — never a
  retuned fixture.
- **`fff` is pinned by revision, not by branch.** A consumer that depends on both
  `sgraph` and `fff` should pin the same revision so cargo resolves one copy.
- **Public surface is everything outside `internals`.** Items behind the
  `internals` feature are exempt from compatibility guarantees.
- **Field arithmetic is not a wire property here.** `sgraph` carries packed bytes
  and calls `fff`; swapping `fff` kernels changes speed, not results.

## Measured throughput

Numbers are the minimum of three interleaved runs, each pinned to one core, on
an Intel Core Ultra 7 258V under Linux 7.1 with `rustc 1.93.0`, `--all-features`,
`lto = "thin"`, and 1024-byte symbols. They describe this machine and this
geometry; re-measure before quoting them anywhere else.

| Case | Geometry | Per operation |
| --- | --- | --- |
| `WindowedUniform` neighbours | span 4096, degree 3 / 8 / 32 | 11.0 ns / 29.5 ns / 155 ns per check |
| `Peeler::push_check` (no ripple) | degree 3 / 8 / 32, 64 live rows | 114 ns / 174 ns / 515 ns per check |
| Peeling cascade | chain of 16 / 256 hops | 1.33 µs / 24.8 µs per chain |
| Residual RREF over GF(256) | 8×8 / 32×32 / 64×64 | 1.22 µs / 30.5 µs / 192 µs per solve |

Criterion baseline artifacts are machine-specific and are deliberately not
committed; record the comparison conditions and results instead.

## Development

```sh
cargo test --all-features
cargo test --no-default-features
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

`tests/vectors.rs` pins the deterministic machinery against values captured from
an independent implementation. A change that moves it is a format break, not a
refactor.

Benchmarks cover neighbour generation, ingest, cascade, and residual solve:

```sh
cargo bench --bench graph -- --save-baseline before
# ... change something ...
cargo bench --bench graph -- --baseline before
```

Compare only interleaved, core-pinned runs of the two builds. On a laptop the
run-to-run drift between two *identical* builds reaches tens of percent, which is
larger than most changes worth measuring.

## License

MIT.
