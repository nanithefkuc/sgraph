# `sgraph` planning set

`sgraph` — **sparse graph**. The shared sparse/Tanner-graph engine for
erasure-coding families (LDPC, LT, and Raptor-class codes) that currently
re-implement the same peeling decoder, neighbour generator, and residual solve.

Its architectural role mirrors [`cafft`](https://github.com/nanithefkuc/cafft):
own the *structure and its algorithms*, depend on
[`fff`](https://github.com/nanithefkuc/fff) for all field arithmetic and
byte-vector kernels, and leave wire formats and codec shells to consumers.

## Reading order

| Doc | Contents |
| --- | --- |
| [`00-charter.md`](00-charter.md) | Scope, boundaries, non-goals, load-bearing invariants. Read first; everything else defers to it. |
| [`01-extraction-map.md`](01-extraction-map.md) | Item-by-item provenance from `mix-dpc`, with a keep / generalize / leave-downstream verdict and `file:line` citation for each. |
| [`02-architecture.md`](02-architecture.md) | Module layout, core types, trait seams, the sparse↔dense boundary. |
| [`03-generalization.md`](03-generalization.md) | Degree distributions, neighbour generators, the GF(2)→GF(2^m) axis, and the zero-cost binary specialization. |
| [`04-roadmap.md`](04-roadmap.md) | Phases with acceptance criteria. |
| [`05-conventions.md`](05-conventions.md) | `Cargo.toml`, lints, CI, testing discipline, draft `AGENTS.md`. |
| [`06-risks.md`](06-risks.md) | Risks, and the open decisions that need a human call. |

## Provenance

The extraction source is `mix-dpc` at `6d7d4ac` ("bump fff to 0.1.1"), a
systematic sliding-window erasure code combining a windowed sparse LT layer over
GF(2) with a dense MDS Cauchy layer over GF(256). Every `file:line` citation in
these documents refers to that revision.

The plan was written against a full read of `mix-dpc`'s sparse layer
(`internals/ldpc/`, `rng.rs`, `internals/ring.rs`), its dense layer
(`internals/{solver,symbol,hdpc}.rs`, `codec.rs`), and the `cafft`/`fff`/`mix-dpc`
conventions (`AGENTS.md`, manifests, CI, lint headers).
