# Risks and open decisions

## Risks

### R1 — The RNG stream refactor is a silent wire break (highest)

Threading `&mut SplitMix64` through `distinct_offsets` is required to make degree
and offsets one reproducible stream
([`03-generalization.md`](03-generalization.md) axis 1). But `distinct_offsets`
currently re-seeds internally from `seed` (`mix-dpc/src/rng.rs:71`), so *any*
change to what is drawn before it shifts every offset — and offsets are
`mix-dpc`'s wire format.

The failure mode is nasty: nothing panics, nothing fails to compile, tests that
only check round-trip recovery still pass, and the break surfaces as two peers
built against different `sgraph` revisions silently failing to decode.

**Mitigation:** capture golden offset vectors from `mix-dpc` **before** any
refactor (Phase 0), assert them at Phase 1, and make "point-mass degree consumes
zero draws" an explicitly tested property rather than an incidental one — test the
generator *state*, not just the offsets, because only the state test catches a
future distribution accidentally drawing on the constant path.

### R2 — Generalization erodes the optimization

`mix-dpc`'s sparse layer is already tuned: implicit-and-consumed forward
adjacency, explicit reverse adjacency, a permissive ripple that never needs
deduplication, stack-allocated offset scratch, capped buffer and key-list pools.
Every abstraction inserted between the generator and the peeler is an opportunity
to lose one.

Specific pressure points:

- `MAX_DC`'s stack `[u32; 64]` (`ldpc/mod.rs:25`) becomes a pooled heap buffer
  once degree is unbounded. Measure it; a small-vector inline capacity may be
  warranted, but only with a benchmark behind it.
- `NeighborGen::neighbors` as a trait method is a virtual call per check if the
  consumer uses a `dyn` generator. Keep it generic (monomorphized) and do not
  offer a `dyn`-friendly variant unless someone asks.
- `Vec<Binary>` costing nothing is a claim about `rustc`'s ZST handling. It is
  true, but assert it (Phase 2 acceptance) rather than trust it.

**Mitigation:** Phase 5's criterion comparison against a pre-migration baseline is
the gate. A regression there blocks the phase.

### R3 — Solver allocation removal can couple solver and peeler

The source solver returns `Vec<(u64, Vec<u8>)>`, allocating per recovery. Having
the solver request output buffers from the peeler would erase the clean
sparse/dense seam and require a shared allocator trait.

**Resolved design:** RREF already owns each recovered value in its RHS matrix.
The solver stores only reusable `(VarId, pivot_row)` metadata and exposes
borrowed `(VarId, &[u8])` views until the next solve. The resolver calls a
crate-private resident/length-validated peeler copy path; public callers use
`Peeler::learn_copy`. Both draw from the peeler pool and copy once. After warm-up
this allocates nothing, and neither layer owns or
borrows the other's allocator. Phase 4 tests the borrowed-output lifetime,
zero-allocation path, and multi-iteration driver before `mix-dpc` migration.

### R4 — Retirement stays linear

Replacing `window_base` with a per-check minimum removes the *window concept*
but does not make retirement sub-linear: checks are keyed by check id, while the
horizon is a variable id. A consumer with many live rows still pays
`O(live checks)` per retirement.

The minimum must describe **current residual support**, not the generated edge
set. If the original minimum has already been folded out, using a stale
ingest-time value would discard a still-useful equation over newer variables.

**Mitigation:** maintain the cached minimum on support removal, document the
linear scan, and pin both stale-minimum cases in Phase 3. Revisit a
minimum-ordered secondary index only after a real consumer measures retirement
as a bottleneck. Phase 5 separately pins compatibility: the downstream adapter
retains `window_base` and calls explicit `retire_check` under the source rule,
avoiding a behaviour change during extraction.

### R5 — `no_std` costs more than it looks

`cafft` is `no_std` + `alloc`; `mix-dpc` is not. `sgraph` should follow `cafft`,
and the collection audit over the extraction surface is clean: every `std`
collection in `mix-dpc` is either test-only (`internals/solver.rs:188,243`,
`internals/ring.rs:306` — all inside `#[cfg(test)]` modules, and tests may use
`std` freely) or lives in code that stays downstream (`internals/hdpc.rs:323`'s
`HashMap`, `stream.rs:339`'s `BTreeMap`). Nothing coming across needs more than
`Vec`, `VecDeque`, and `Box`. But `fff`'s backend dispatch needs `std` for
`LazyLock`, so `--no-default-features` silently means "portable scalar", and
every test must be run in both configurations or the scalar path rots.

**Mitigation:** `cargo test --no-default-features` in CI from Phase 0, not added
later. It is nearly free at the start and expensive to retrofit once
`std`-dependent code has crept in.

### R6 — The plan's scope is the whole crate

Phases 0–5 are an extraction with a hard proof at the end. Phases 6–7 are new
work whose value is real but unproven, and which no current consumer needs.

**Mitigation:** treat Phase 5 followed by release-hygiene Phase 8 as the natural
scope for 0.1. A crate that serves `mix-dpc` correctly and has honest seams for
LT/Raptor is better than one with three half-tested code families.

### R7 — New distributions can violate cross-platform determinism

Robust-soliton construction uses logarithms, square roots, normalization, and
rounding. Leaving those operations to platform math can produce a one-unit CDF
difference and therefore a different graph. RFC 5053 also specifies its own
random tables; substituting SplitMix64 would produce a plausible but non-RFC
generator.

**Mitigation:** robust-soliton sampling uses a specified fixed-point `u64` CDF
with an independent high-precision fixture; RFC 5053 uses its normative tables.
Phase 6 pins table and degree/tuple fingerprints across x86_64, AArch64, and
Wasm. RaptorQ/RFC 6330 is named as a separate, unsupported algorithm.

### R8 — Dense rings amplify sparse or hostile ids

`Ring::ensure` currently fills every gap, uses unchecked `base + len`, and casts
`u64` offsets to `usize`. A far-ahead id can request enormous memory; near
`u64::MAX` the arithmetic can wrap; a 32-bit target can truncate a valid `u64`
span.

**Mitigation:** Phase 1 adds checked geometry and configured maximum live spans
for variable and check ids. Rejection is transactional and never evicts state.
Wasm builds and boundary tests are gates, not later hardening.

### R9 — Rank does not prove the augmented system is consistent

The source solver assumes authentic rows. For contradictory input, coefficient
rank and deficiency still look valid while RREF contains `0 = nonzero`; emitting
any determined pivot value would be unsound.

**Mitigation:** after elimination, detect every zero-coefficient row with a
non-zero packed RHS, return `SolveError::InconsistentSystem`, clear recovery
metadata, and teach the peeler nothing. Phase 4 includes the contradictory-row
regression.

### R10 — The peeler cannot discover the complete unknown set

The union of stalled sparse supports is not the codec's loss set. A lost variable
may be covered only by dense rows or by no received row yet. If the driver uses
peeler-local columns, exact deficiency is understated; with zero sparse rows it
would incorrectly be zero.

**Mitigation:** `Resolver::resolve` takes the consumer-maintained `IndexSet`
explicitly, snapshots it into reusable sorted scratch, and removes learned ids.
Phase 4 gates the zero-sparse-row case and Phase 5 maps `mix-dpc::missing`
directly onto this seam.

## Open decisions

These need a human call; each has a recommendation, and each is defaulted so work
is never blocked on the answer.

### D1 — Does `mix-dpc` accept a wire break?

If yes, the RNG stream discipline can be made clean (degree always draws, one
uniform code path) and R1 largely evaporates. If no, the constant-degree path
carries a permanent "must consume zero draws" constraint.

**Default taken by this plan: no break.** Bit-exactness is invariant 2 and Phase 5
turns on `tests/interop.rs` passing unchanged. The constraint is cheap to honour
and preserving a working deployed format is worth more than a slightly tidier
sampler.

### D2 — Where does the dense MDS layer live?

`hdpc.rs`'s Cauchy construction (`hdpc.rs:58,64`) is pure field-matrix algebra
with no graph content, and `fff`'s own boundary statement (`fff/src/lib.rs:115-118`)
assigns coding-matrix construction to "a codec layer" — which is a gap, not an
assignment: `fff` says not-me, `sgraph` says not-me.

Options: leave it in `mix-dpc` (duplicated by the next consumer), push it into
`fff` (contradicts `fff`'s stated boundary), or a fourth crate.

**Default: leave it in `mix-dpc`.** It is one file and one consumer. Extracting a
shared dense-MDS crate is a decision for when there are two consumers, and making
it now would be exactly the speculative abstraction this plan avoids elsewhere.

### D3 — `VarId`/`CheckId`: aliases or newtypes?

Aliases are simpler and match `Ring<T>`'s `u64` keying. Newtypes prevent
confusing a variable index with a check index — and the two *are* confusable:
`retire_below` takes a variable horizon while indexing a check-keyed ring, which
is precisely the mix-up behind R4's awkwardness.

**Default: newtypes**, with `From`/`Into` and `#[repr(transparent)]`. The bug
class is real, the cost is zero at runtime, and it documents the seam. Revisit only
if it makes the generator APIs noisy.

### D4 — Which field does the residual solver default to?

`mix-dpc` uses GF(256) (`solver.rs` throughout). Generic over
`F: FieldKernels` is the plan, but GF(256) caps a single solve at 256 columns
before Cauchy MDS-ness fails downstream — a *consumer* constraint, not the
solver's, since RREF itself has no column limit over GF(256).

**Default: generic, with GF(256)/`Gf8` as the conventional compatibility
choice.** Coefficient scratch is `Vec<F::Elem>` and RHS buffers must align to
`F::BYTES`; wider fields cost more memory. `fff` has no GF(2^4), so the plan
makes no “GF(16) halves memory” claim.

### D5 — Should `sgraph` host encoder storage or a check builder?

The reusable guarantee on the encode side is `NeighborGen`: encoder and decoder
must regenerate the same weighted edges. Symbol ownership, block/sliding
retention, and the output wire type are consumer policy; accumulating
`Σ wᵢ·symbolᵢ` is a thin `fff::ops` loop.

**Default: leave both `SourceWindow`/`SymbolStore` and `CheckBuilder` downstream.**
This keeps the charter honest and resolves the prior extraction-map/architecture
contradiction. Revisit only when a second consumer demonstrates shared storage
or accumulation policy beyond the generator itself.
