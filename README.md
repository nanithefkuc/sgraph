> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# sgraph

**Sparse graph** — a shared sparse/Tanner-graph engine for erasure codes.

`sgraph` is the layer that LDPC, LT, and Raptor-class implementations keep
re-implementing: deterministic cross-peer neighbour generation, degree
distributions, a residual sparse graph that shrinks as symbols become known,
XOR-only peeling, and the exact residual solve that finishes what peeling
cannot. These codes differ in how their graph is generated, not in how it is
consumed.

Field arithmetic and byte-buffer vector primitives come from
[`fgf`](https://github.com/nanithefkuc/fgf); this crate never re-implements
field arithmetic. Wire formats, packet headers, transport and HARQ policy,
belief-propagation soft-decision decoding, protograph lifting, and codec shells
stay with the consumer.

The crate root carries `#![forbid(unsafe_code)]`, and steady state allocates
nothing: scratch is owned and reused, and retirement recycles buffers rather
than dropping them. `Binary` — a zero-sized GF(2) coefficient — is the
implemented `EdgeWeight`; `ResidualCoeff<F>` embeds it into any `fgf` field, so
one generic engine serves GF(2) and GF(2^m) without taxing the binary path.

## Usage

The MSRV is Rust 1.89 (edition 2024).

`sgraph` is distributed through git only; it is not published to
[crates.io](https://crates.io). It depends on `fgf` by git, so depend on it the
same way. Pin the same `fgf` revision across every crate you use so cargo
resolves a single copy — the neighbour generation and residual solve feed
downstream wire formats, so a floating dependency is a format-break risk.

```toml
[dependencies]
sgraph = { git = "https://github.com/nanithefkuc/sgraph" }
```

### Features

| Feature | Result |
| --- | --- |
| `std` (default) | `fgf`'s runtime CPU detection and its process-wide backend cache |
| `simd` (default, implies `std`) | runtime-dispatched SIMD kernels from `fgf` |
| `--no-default-features` | `no_std` + `alloc`, portable scalar kernels |
| `internals` | unstable implementation APIs, exempt from compatibility guarantees |

## Determinism

A check symbol travels without its graph: both peers regenerate its edge set from
the check's id alone. That makes the PRNG, the sampling algorithm, and the draw
order wire properties, frozen by the fixtures under `tests/data/`. Changing any
of them is a format break for every consumer — a format break gets a new or
versioned generator, never a retuned fixture.

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

## Modules

| Module | Contents |
| --- | --- |
| `rng` | `SplitMix64`, unbiased bounded draws over a `NonZeroU32`, and Floyd distinct-k sampling. Allocation-free, and validated before it writes. |
| `index` | `Ring<T>` and `IndexSet`: dense index-keyed storage over a monotone `u64` domain, where a lookup is a subtraction and retirement is a front drain that hands back what it dropped. Bounded by a configured live span. |
| `weight` | `EdgeWeight` and `ResidualCoeff<F>`, plus `Binary` — a zero-sized GF(2) coefficient, so one generic engine serves GF(2) and GF(2^m) without taxing the binary path. |
| `degree` | `DegreeDistribution` with `Constant` (a point mass that consumes no randomness), `Cumulative` (an explicit integer weight table), and `RobustSoliton` (built from Q32 fixed-point parameters, so the table is identical on every platform). |
| `neighbors` | `NeighborGen` with the `Uniform`, `WindowedUniform`, `Rfc5053Triple` (RFC 5053 Raptor, not RaptorQ) and `ExplicitMatrix` (CSR parity-check) generators, the reusable `NeighborBuf` scratch, and `Edges` — the one place edge shape is validated. |
| `peel` | `Peeler`: the residual sparse graph, reverse adjacency, iterative degree-one cascade, pooled buffers, and explicit retirement. `StalledRow` exposes what peeling could not finish. |
| `residual` | `ResidualBuilder`/`RowSink` single-pass assembly over explicit columns, and `Solver` — full reduced row echelon form over any `fgf` field. |
| `driver` | `DenseRows`, the consumer seam for progressively reduced dense equations, and `Resolver`, the peel → solve → re-peel fixpoint. |
| `id` | `VarId` and `CheckId`: `#[repr(transparent)]` newtypes so a variable horizon cannot be passed where a check key belongs. |
| `error` | `GraphError` and `SolveError`. |

## Building

`sgraph` builds on stable Rust (edition 2024, MSRV 1.89) with no extra tooling
or target-feature flags — the SIMD kernels it uses from `fgf` are selected at
runtime:

```sh
cargo build                        # default: std + simd
cargo build --no-default-features  # portable no_std + alloc
cargo test --all-features
```

## Benchmarks

`cargo bench --bench graph` covers neighbour generation, ingest, cascade, and
residual solve. The numbers below are the minimum of three runs pinned to one
core, on an Intel Core Ultra 7 258V under Linux 7.1 with `rustc 1.93.0`,
`--all-features`, `lto = "thin"`, and 1024-byte symbols. They describe this
machine and this geometry; re-measure before quoting them anywhere else, and
compare only interleaved, core-pinned runs of the two builds.

| Case | Geometry | Per operation |
| --- | --- | --- |
| `WindowedUniform` neighbours | span 4096, degree 3 / 8 / 32 | 9.9 ns / 20.9 ns / 120 ns per check |
| `Peeler::push_check` (no ripple) | degree 3 / 8 / 32, 64 live rows | 138 ns / 179 ns / 436 ns per check |
| Peeling cascade | chain of 16 / 256 hops | 1.18 µs / 22.5 µs per chain |
| Residual RREF over GF(256) | 8×8 / 32×32 / 64×64 | 1.22 µs / 18.9 µs / 84.7 µs per solve |

## License

MIT.
