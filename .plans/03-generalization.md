# Generalization axes

`mix-dpc`'s sparse layer is one point in a four-dimensional space. Each axis is
generalized independently, and each has a concrete "does the existing behaviour
still fall out of it" test.

| Axis | `mix-dpc` today | Generalized to |
| --- | --- | --- |
| Degree | constant `dc`, capped by window occupancy (`encoder.rs:65`) | `DegreeDistribution`: constant, robust soliton, RFC 5053 table |
| Neighbour selection | uniform distinct-k over a sliding window (`rng.rs:65`) | `NeighborGen`: uniform, windowed uniform, RFC 5053 Raptor triple, explicit CSR matrix |
| Index domain | `u64` stream index with TTL-derived window (`AGENTS.md` invariants) | bounded monotone domain; retirement is a mechanism, horizons are consumer policy |
| Field | GF(2), unit weights, XOR only | `EdgeWeight`: `Binary` (ZST) or `Weighted<F>` over any `fff` field |

## Axis 1 — degree

Today degree is one line on each side: `eff = self.dc.min(span as usize)`
(`encoder.rs:65`, `decoder.rs:198`). That single expression is what makes the
graph *regular*; a real LT code needs a distribution.

The subtlety is stream discipline. `distinct_offsets` currently constructs its own
`SplitMix64::new(seed)` internally (`rng.rs:71`), so a degree drawn from a
separate generator would not perturb the offsets — degree and offsets would be
two independent streams keyed on the same seed. That is fine for a constant
degree and wrong for a distribution: correlated degree/offset streams are how you
get subtle graph-quality bugs that only show up as a residual-rate regression.

So: hoist the generator out, thread `&mut SplitMix64`, and draw degree first.

```rust
let mut rng = SplitMix64::new(id ^ domain_sep);
let d = degree.sample(&mut rng);              // Constant: zero draws
distinct_offsets(&mut rng, span, &mut buf[..d]);
```

**Hard constraint (charter invariant 2):** `Constant::sample` must consume **zero**
RNG draws, so that this composition is byte-identical to `mix-dpc` for the
constant-degree case and `mix-dpc` can migrate onto `sgraph` without a wire
break. Pin it with a fingerprint test against known-good offset vectors captured
from `mix-dpc` before any refactor.

`MAX_DC = 64` (`ldpc/mod.rs:25`) — a stack `[u32; 64]` scratch bound — becomes
`DegreeDistribution::max_degree()`, which sizes a pooled `NeighborBuf` instead of
a stack array. RFC 5053's table reaches degree 40, robust soliton is unbounded in
principle and truncated at `k` in practice, so a fixed stack array is no longer
tenable.

### Cross-platform distribution determinism

`RobustSoliton { k, c, delta }` cannot build its sampling table with unspecified
platform `ln`/`sqrt` calls and still satisfy the reproducibility invariant.
Construction therefore produces a canonical cumulative table of `u64` weights
using a specified fixed-point algorithm; sampling is integer-only. Parameters,
rounding, saturation, and the table fingerprint are part of the generator
contract. Phase 6 first captures an independent high-precision reference table,
then pins the quantized table and sampled degree stream on x86_64, AArch64, and
Wasm. An explicit cumulative-table constructor is also provided for protocols
that already define their own distribution.

## Axis 2 — neighbour selection

`distinct_offsets` (Floyd's k-of-n: exactly `k` bounded samples, distinctness by
an `O(k)` linear scan over the small output, no allocation) is a good algorithm
and stays. A bounded sample may consume more than one `next_u64` because Lemire
rejection is unbiased; code and documentation must not claim exactly `k` raw RNG
draws. It remains one *strategy*, not the interface:

- **RFC 5053 Raptor** selects neighbours by a `(d, a, b)` triple walked over a
  prime modulus. **RaptorQ/RFC 6330 is different**: its tuple has six values and
  is not implemented or implied by `Rfc5053Triple`.
- **Classic LDPC** reads a fixed parity-check matrix, possibly quasi-cyclic; no
  sampling at all.

Hence `NeighborGen` as the boundary, with `distinct_offsets` as an implementation
detail of `Uniform`/`WindowedUniform`.

`ExplicitMatrix` (a CSR `H`) doubles as the escape hatch for consumers who
generate graphs by means `sgraph` has never heard of. It also makes the peeler
testable against textbook matrices with known stopping sets — a much sharper
oracle than random loss patterns.

Note what stays out: **designing** the matrix. Protograph lifting, quasi-cyclic
expansion, girth optimization, PEG construction — all consumer concerns. `sgraph`
consumes a graph; it does not design one.

## Axis 3 — index domain

`mix-dpc` defines the parity graph in symbol-index (count) space, with TTL
capping the window width and driving retirement (`AGENTS.md` invariants). Two
things must be separated:

- **Mechanism** (`sgraph`): a monotone `u64` domain, `Ring<T>` storage where
  below-base means *gone* rather than *absent* (`ring.rs:65`), and
  `retire_below(horizon)` as a front drain that recycles what it drops.
- **Policy** (consumer): what the horizon is, and when to advance it. TTLs,
  deadlines, and give-up decisions are transport.

One concrete blocker: retirement currently scans every live check because checks
are keyed by check id while the horizon is a *variable* index — the two orderings
only correlate (`decoder.rs:283-297`, `O(live checks)` per retirement). Keep the
scan, but cache the minimum of each row's **current residual support** and update
it whenever support changes. An ingest-time minimum would become stale after a
known variable is folded out and could discard a useful row. Sub-linear
retirement remains a separate measured optimization; see
[`06-risks.md`](06-risks.md) R4.

`retire_check(CheckId)` is the policy escape hatch. The Phase 5 `mix-dpc`
adapter uses it with its retained `window_base` metadata to reproduce the
source's stricter “drop the whole check once its window starts below the
horizon” rule. The generalized core does not bake that policy into
`retire_below`.

`SourceWindow::slot()` has a release-mode stale-data precondition guarded only by
a `debug_assert` (`window.rs:62`). It is not imported: encoder-side storage stays
downstream, and fixing that private consumer contract is intentionally separate
from the extraction. `sgraph` shares edge generation, not symbol ownership.

## Axis 4 — field: GF(2) → GF(2^m)

### The nine sites

Every place `mix-dpc` assumes edge weight ≡ 1 and addition ≡ XOR:

| # | Site | Change |
| --- | --- | --- |
| 1 | `symbol.rs:21` `xor_into` = `ops::add_assign::<Gf8>` | GF(2^m) *addition* is still XOR, so this survives; what is absent is multiplication. |
| 2 | `encoder.rs:69` check symbol is an unweighted XOR of its neighbours | Needs `mul_add(out, w, src)`, with weights regenerable from the same seed stream on both peers. |
| 3 | `decoder.rs:206` folding a known neighbour out at ingest | Needs the weight. |
| 4 | `decoder.rs:331` `apply_known` folding | Needs the weight *after* ingest, so `unknown: Vec<u64>` must carry weights; `swap_remove` moves both in lockstep. |
| 5 | `decoder.rs:365-368` `drive_peel` asserts `variable == rhs` | With a lone unknown of weight `c` the value is `c⁻¹ · rhs`. **There is no field inverse anywhere in the peeling path today.** This is the substantive change. |
| 6 | `decoder.rs:18-20` the documented residual invariant | Restate as `rhs == Σ wᵢ · valᵢ` over the field. |
| 7 | `decoder.rs:149-150` `unresolved_equations` exports "GF(2) rows" | Export weights alongside support. |
| 8 | `ldpc/mod.rs:1-9`, `symbol.rs:3-4` module docs | The documented contract of the whole layer. |
| 9 | `rng.rs` has **no** coefficient generator at all | A weighted generator needs a paired non-zero-weight draw, with a pinned draw order and domain separator. Zero weights are rejected. Switching a graph from binary to weighted is an expected wire break because it defines a different code. |

Mitigating fact: `fff` already supplies everything needed — `mul_add`,
`mul_assign`, `mul_into`, and a total `inv` with the `inv(0) == 0` convention —
with runtime SIMD dispatch. No new arithmetic has to be written, only threaded.

### The zero-cost binary specialization

The design risk is obvious: making one engine serve both fields must not tax the
binary path, which is the overwhelmingly common case and the one that is already
optimized. Three options:

1. Two separate engines (`BinaryPeeler`, `WeightedPeeler`). No tax, but the
   peeling logic — the crate's centrepiece — gets duplicated and drifts.
2. Generic with `const IS_BINARY` branching. One engine, but branches and
   `Option<Coeff>` in the hot loop.
3. **Generic over a ZST-capable weight type.** Chosen.

```rust
pub trait EdgeWeight: Copy + Eq + Default + 'static {
    fn one() -> Self;
    const ELEMENT_BYTES: usize;
    fn is_zero(self) -> bool;
    fn mul_add(dst: &mut [u8], w: Self, src: &[u8]);   // dst += w·src
    fn scale_inv(value: &mut [u8], w: Self);           // value *= w⁻¹
}

pub struct Binary;                       // ZST; never zero
pub struct Weighted<F: FieldKernels>(pub F::Elem);
```

`Binary::ELEMENT_BYTES == 1`; `Weighted<F>::ELEMENT_BYTES == F::BYTES`.
`Peeler::new` rejects a symbol length not divisible by that width before any
`fff::ops` call can reach its panicking geometry checks.

`Binary::mul_add` forwards to `add_assign::<Gf8>` and ignores the weight;
`Binary::scale_inv` is a no-op that the optimizer deletes.

The trick that makes it free: **`Vec<Binary>` never allocates and has no
per-element cost.** Rust gives a `Vec` of a zero-sized type capacity
`usize::MAX` with a dangling pointer, so parallel `support: Vec<VarId>` /
`weights: Vec<W>` support vectors cost exactly what a bare `Vec<VarId>` costs when
`W = Binary`. `push`, `swap_remove`, and indexing on the weight vector all
compile to nothing. No specialization, no branching, no duplicated engine.

`peel/` is therefore written once, generically. Phase 2 ships and tests the
`Binary` instantiation and the conversion seam into the residual field; the
non-zero `Weighted<F>` implementation and coefficient generator land in Phase
7. That sequencing keeps the extraction proof independent of new field logic.

### What does *not* generalize

The residual solver stays over a real field even when the sparse layer is
binary: elimination requires division. `ResidualCoeff<F>` widens binary edges
to `F::Elem::ONE`, exactly as `mix-dpc` widens them to GF(256) today
(`codec.rs:508`), while `Weighted<F>` maps only into the same field. Coefficient
scratch is `Vec<F::Elem>`, not `Vec<u8>`, and packed RHS lengths must be multiples
of `F::BYTES`. Wider `fff` fields therefore consume *more* coefficient memory;
`fff` does not provide GF(2^4), so no GF(16) packing claim is made.

Making the solver generic over "maybe GF(2)" is not useful: GF(2) elimination is
a bitmatrix algorithm with a different inner loop. If wanted later, it is a
separate solver rather than a parameterization of the field-element RREF.
