//! Steady-state allocation claims, counted rather than asserted.
//!
//! The peeler stores a coefficient per edge in a `Vec<W>` parallel to its
//! support. That is only free for the binary case if `Vec<Binary>` genuinely never
//! allocates — the whole reason one generic engine can serve GF(2) and GF(2^m)
//! without taxing the common path.
//!
//! Two things make this file's shape non-negotiable:
//!
//! * a `#[global_allocator]` is process-wide, so this must be its own integration
//!   test rather than share a binary with tests that allocate freely;
//! * the harness runs `#[test]` functions on parallel threads, so there is
//!   exactly **one** test here and the cases are plain helpers called in
//!   sequence. Splitting them would let each contaminate the others' counts.

use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use fgf::{Gf8, gf8};
use sgraph::index::IndexSet;
use sgraph::{
    Binary, CheckId, DenseRows, Edges, Peeler, PoolConfig, ResidualBuilder, Resolver, RowSink,
    SolveError, Solver, VarId, VariableState, Weighted,
};
use std::alloc::{GlobalAlloc, Layout, System};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Counts allocation, zeroed-allocation, and reallocation calls, ignoring frees.
/// `realloc` counts because a growing `Vec` reaches for it.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[test]
fn steady_state_paths_are_allocation_free() {
    growing_a_binary_weight_vector_never_allocates();
    binary_weight_operations_are_free();
    warm_edge_generation_never_allocates();
    warm_peeling_stream_never_allocates();
    warm_weighted_peeling_stream_never_allocates();
    warm_resolver_fixpoint_never_allocates();
    the_counter_observes_a_real_allocation();
}

/// Edge generation reuses the caller's [`NeighborBuf`] for both the parallel
/// output arrays and the `u32` sampling scratch, so once that buffer has been
/// sized to `max_degree` a generator must never allocate again — however many
/// checks it produces.
fn warm_edge_generation_never_allocates() {
    use sgraph::{CheckId, Constant, NeighborBuf, NeighborGen, WindowedUniform};

    let degree = Constant::new(5).expect("non-zero degree");
    let generator = WindowedUniform::new(1_000, 64, degree, 0xA5A5_5A5A_C3C3_3C3C)
        .expect("valid window geometry");

    let mut buf: NeighborBuf<Binary> = NeighborBuf::with_capacity(generator.max_degree() as usize);
    // Warm every internal buffer to its high-water mark, including the offset
    // scratch, which is only sized on first use.
    generator
        .neighbors(CheckId::new(0), &mut buf)
        .expect("generates");

    let before = ALLOCS.load(Relaxed);
    for id in 0..10_000u64 {
        generator
            .neighbors(CheckId::new(id), &mut buf)
            .expect("generates");
        assert_eq!(buf.len(), 5);
        // The peeler validates every edge set at ingest; that must not allocate
        // either, since it happens once per received check.
        let edges = buf.edges().expect("generated edges validate");
        assert_eq!(edges.len(), 5);
    }
    let counted = ALLOCS.load(Relaxed) - before;

    assert_eq!(
        counted, 0,
        "warm edge generation allocated {counted} times over 10k checks"
    );
}

fn growing_a_binary_weight_vector_never_allocates() {
    let mut weights: Vec<Binary> = Vec::new();

    let before = ALLOCS.load(Relaxed);
    for _ in 0..1_000_000 {
        weights.push(Binary);
    }
    let counted = ALLOCS.load(Relaxed) - before;

    assert_eq!(weights.len(), 1_000_000);
    assert_eq!(
        counted, 0,
        "growing Vec<Binary> to 10^6 elements performed {counted} allocations"
    );
}

/// The operations the peeler actually performs on a support/weight pair, with the
/// weight side contributing nothing. `support` is pre-sized, so any count here
/// belongs to the weights.
fn binary_weight_operations_are_free() {
    const DEGREE: usize = 64;
    let mut support: Vec<VarId> = Vec::with_capacity(DEGREE);
    let mut weights: Vec<Binary> = Vec::with_capacity(DEGREE);

    let before = ALLOCS.load(Relaxed);
    for round in 0..10_000u64 {
        support.clear();
        weights.clear();
        for i in 0..DEGREE as u64 {
            support.push(VarId::new(round * DEGREE as u64 + i));
            weights.push(Binary);
        }
        // Folding known variables out of a residual row, one at a time.
        while support.len() > 1 {
            support.swap_remove(0);
            weights.swap_remove(0);
        }
    }
    let counted = ALLOCS.load(Relaxed) - before;

    assert_eq!(
        counted, 0,
        "steady-state support churn allocated {counted} times"
    );
}

/// A warmed decoder processes 1,200 symbols through ingest, a two-step cascade,
/// recovery draining, and retirement without reaching the allocator.
fn warm_peeling_stream_never_allocates() {
    use core::num::NonZeroUsize;

    const SYMBOL_LEN: usize = 32;
    const WARM_BATCHES: u64 = 32;
    const MEASURED_BATCHES: u64 = 600;

    let span = NonZeroUsize::new(8).expect("non-zero span");
    let config = PoolConfig::new(span, span).with_pool_capacity(8);
    let mut peeler = Peeler::<Binary>::new(SYMBOL_LEN, config).expect("valid peeler");
    let mut recovered = Vec::with_capacity(2);

    for batch in 0..WARM_BATCHES {
        peel_two(&mut peeler, &mut recovered, batch);
    }

    let before = ALLOCS.load(Relaxed);
    for batch in WARM_BATCHES..WARM_BATCHES + MEASURED_BATCHES {
        peel_two(&mut peeler, &mut recovered, batch);
    }
    let counted = ALLOCS.load(Relaxed) - before;

    assert_eq!(
        counted, 0,
        "warmed 1,200-symbol ingest/cascade/retirement stream allocated {counted} times"
    );
}

fn warm_weighted_peeling_stream_never_allocates() {
    use core::num::NonZeroUsize;

    const SYMBOL_LEN: usize = 32;
    const WARM_BATCHES: u64 = 32;
    const MEASURED_BATCHES: u64 = 600;

    let span = NonZeroUsize::new(8).expect("non-zero span");
    let config = PoolConfig::new(span, span).with_pool_capacity(8);
    let mut peeler = Peeler::<Weighted<Gf8>>::new(SYMBOL_LEN, config).expect("valid peeler");
    let mut recovered = Vec::with_capacity(2);
    for batch in 0..WARM_BATCHES {
        weighted_peel_two(&mut peeler, &mut recovered, batch);
    }

    let before = ALLOCS.load(Relaxed);
    for batch in WARM_BATCHES..WARM_BATCHES + MEASURED_BATCHES {
        weighted_peel_two(&mut peeler, &mut recovered, batch);
    }
    let counted = ALLOCS.load(Relaxed) - before;
    assert_eq!(
        counted, 0,
        "warmed weighted ingest/cascade/retirement stream allocated {counted} times"
    );
}

fn weighted_peel_two(peeler: &mut Peeler<Weighted<Gf8>>, recovered: &mut Vec<VarId>, batch: u64) {
    const SYMBOL_LEN: usize = 32;

    let first = VarId::new(batch * 2);
    let second = VarId::new(batch * 2 + 1);
    let first_value = [batch as u8; SYMBOL_LEN];
    let second_value = [batch.wrapping_mul(17) as u8; SYMBOL_LEN];
    let pair_weights = [
        Weighted::new(gf8::Elem(3)).expect("non-zero"),
        Weighted::new(gf8::Elem(5)).expect("non-zero"),
    ];
    let singleton_weight = [Weighted::new(gf8::Elem(7)).expect("non-zero")];
    let mut pair_rhs = [0u8; SYMBOL_LEN];
    fgf::ops::mul_add::<Gf8>(&mut pair_rhs, gf8::Elem(3), &first_value);
    fgf::ops::mul_add::<Gf8>(&mut pair_rhs, gf8::Elem(5), &second_value);
    let mut singleton_rhs = [0u8; SYMBOL_LEN];
    fgf::ops::mul_add::<Gf8>(&mut singleton_rhs, gf8::Elem(7), &first_value);

    peeler
        .push_check(
            CheckId::new(batch * 2),
            Edges::new(&[first, second], &pair_weights).expect("valid pair"),
            &pair_rhs,
        )
        .expect("pair ingests");
    peeler
        .push_check(
            CheckId::new(batch * 2 + 1),
            Edges::new(&[first], &singleton_weight).expect("valid singleton"),
            &singleton_rhs,
        )
        .expect("singleton ingests");
    recovered.clear();
    peeler.drain_recovered_into(recovered);
    assert_eq!(recovered, &[first, second]);
    peeler
        .retire_below(VarId::new(batch * 2 + 2))
        .expect("monotone retirement");
}

fn peel_two(peeler: &mut Peeler<Binary>, recovered: &mut Vec<VarId>, batch: u64) {
    const SYMBOL_LEN: usize = 32;

    let first = VarId::new(batch * 2);
    let second = VarId::new(batch * 2 + 1);
    let first_value = [batch as u8; SYMBOL_LEN];
    let second_value = [batch.wrapping_mul(17) as u8; SYMBOL_LEN];
    let mut pair_rhs = [0u8; SYMBOL_LEN];
    for ((out, left), right) in pair_rhs.iter_mut().zip(first_value).zip(second_value) {
        *out = left ^ right;
    }

    let pair_support = [first, second];
    let pair_weights = [Binary; 2];
    peeler
        .push_check(
            CheckId::new(batch * 2),
            Edges::new(&pair_support, &pair_weights).expect("valid pair"),
            &pair_rhs,
        )
        .expect("pair ingests");
    let singleton_support = [first];
    let singleton_weights = [Binary];
    peeler
        .push_check(
            CheckId::new(batch * 2 + 1),
            Edges::new(&singleton_support, &singleton_weights).expect("valid singleton"),
            &first_value,
        )
        .expect("singleton ingests");

    assert_eq!(
        peeler.variable_state(first),
        VariableState::Known(&first_value)
    );
    assert_eq!(
        peeler.variable_state(second),
        VariableState::Known(&second_value)
    );
    recovered.clear();
    peeler.drain_recovered_into(recovered);
    assert_eq!(recovered, &[first, second]);
    peeler
        .retire_below(VarId::new(batch * 2 + 2))
        .expect("monotone retirement");
    assert_eq!(peeler.check_count(), 0);
}

#[derive(Debug)]
struct FixedDense {
    terms: [[(VarId, gf8::Elem); 2]; 2],
    lengths: [usize; 2],
    rhs: [[u8; 32]; 2],
}

impl FixedDense {
    fn new() -> Self {
        Self {
            terms: [[(VarId::ZERO, gf8::Elem::ONE); 2]; 2],
            lengths: [0; 2],
            rhs: [[0; 32]; 2],
        }
    }

    fn reset(&mut self, ids: [VarId; 3], values: &[[u8; 32]; 3]) {
        self.terms[0][0] = (ids[0], gf8::Elem::ONE);
        self.lengths[0] = 1;
        self.rhs[0] = values[0];
        self.terms[1][0] = (ids[1], gf8::Elem::ONE);
        self.terms[1][1] = (ids[2], gf8::Elem::ONE);
        self.lengths[1] = 2;
        for ((out, left), right) in self.rhs[1].iter_mut().zip(values[1]).zip(values[2]) {
            *out = left ^ right;
        }
    }
}

impl DenseRows<Gf8> for FixedDense {
    fn has_live_rows(&self) -> bool {
        self.lengths.iter().any(|length| *length <= 1)
    }

    fn reduce_known<W: sgraph::EdgeWeight>(
        &mut self,
        peeler: &Peeler<W>,
    ) -> Result<(), SolveError> {
        for row in 0..self.terms.len() {
            let mut term = 0;
            while term < self.lengths[row] {
                let (var, coefficient) = self.terms[row][term];
                let VariableState::Known(value) = peeler.variable_state(var) else {
                    term += 1;
                    continue;
                };
                fgf::ops::mul_add::<Gf8>(&mut self.rhs[row], coefficient, value);
                self.lengths[row] -= 1;
                self.terms[row].swap(term, self.lengths[row]);
            }
        }
        Ok(())
    }

    fn push_rows(&self, sink: &mut RowSink<'_, Gf8>) {
        for row in 0..self.terms.len() {
            if self.lengths[row] <= 1 {
                sink.push_dense(
                    self.terms[row][..self.lengths[row]].iter().copied(),
                    &self.rhs[row],
                );
            }
        }
    }
}

/// The complete builder → RREF → borrowed recovery → re-peel fixpoint remains
/// allocation-free after a same-geometry warm-up.
fn warm_resolver_fixpoint_never_allocates() {
    use core::num::NonZeroUsize;

    const WARM_BATCHES: u64 = 32;
    const MEASURED_BATCHES: u64 = 400;

    let span = NonZeroUsize::new(128).expect("non-zero span");
    let config = PoolConfig::new(span, span).with_pool_capacity(16);
    let mut peeler = Peeler::<Binary>::new(32, config).expect("valid peeler");
    let mut unknowns = IndexSet::new(span);
    let mut dense = FixedDense::new();
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let mut resolver = Resolver::new();

    for batch in 0..WARM_BATCHES {
        resolve_three(
            &mut resolver,
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
            batch,
        );
    }

    let before = ALLOCS.load(Relaxed);
    for batch in WARM_BATCHES..WARM_BATCHES + MEASURED_BATCHES {
        resolve_three(
            &mut resolver,
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
            batch,
        );
    }
    let counted = ALLOCS.load(Relaxed) - before;
    assert_eq!(
        counted, 0,
        "warmed 1,200-symbol builder/RREF/resolver stream allocated {counted} times"
    );
}

#[allow(clippy::too_many_arguments)]
fn resolve_three(
    resolver: &mut Resolver,
    unknowns: &mut IndexSet,
    peeler: &mut Peeler<Binary>,
    dense: &mut FixedDense,
    solver: &mut Solver<Gf8>,
    builder: &mut ResidualBuilder<Gf8>,
    batch: u64,
) {
    let base = batch * 3;
    let ids = [VarId::new(base), VarId::new(base + 1), VarId::new(base + 2)];
    let values = [
        [batch as u8; 32],
        [batch.wrapping_mul(17) as u8; 32],
        [batch.wrapping_mul(93) as u8; 32],
    ];
    for id in ids {
        unknowns.insert(id.get()).expect("bounded unknown set");
    }
    let mut sparse_rhs = [0u8; 32];
    for ((out, left), right) in sparse_rhs.iter_mut().zip(values[0]).zip(values[1]) {
        *out = left ^ right;
    }
    peeler
        .push_check(
            CheckId::new(batch),
            Edges::new(&ids[..2], &[Binary; 2]).expect("valid sparse row"),
            &sparse_rhs,
        )
        .expect("sparse row ingests");
    dense.reset(ids, &values);

    let report = resolver
        .resolve(unknowns, peeler, dense, solver, builder)
        .expect("fixpoint resolves");
    assert_eq!(report.deficiency, 0);
    assert!(unknowns.is_empty());
    // The recovery queue is drained every batch, so its capacity is reused
    // rather than grown; an undrained queue would allocate.
    let mut drained = 0usize;
    for var in resolver.drain_recovered() {
        assert!(ids.contains(&var));
        drained += 1;
    }
    assert_eq!(drained, ids.len());
    for (id, value) in ids.into_iter().zip(values) {
        assert_eq!(peeler.variable_state(id), VariableState::Known(&value));
    }
    let horizon = VarId::new(base + 3);
    peeler.retire_below(horizon).expect("monotone retirement");
    unknowns.retire_below(horizon.get());
}

/// The control. Without it, a miswired counter would make the assertions above
/// pass vacuously.
fn the_counter_observes_a_real_allocation() {
    let mut bytes: Vec<u8> = Vec::new();

    let before = ALLOCS.load(Relaxed);
    for i in 0..1_000_000u32 {
        bytes.push(i as u8);
    }
    let counted = ALLOCS.load(Relaxed) - before;

    assert_eq!(bytes.len(), 1_000_000);
    assert!(
        counted > 0,
        "a growing Vec<u8> must allocate; the counter is not working"
    );
}
