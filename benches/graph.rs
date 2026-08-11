//! Steady-state throughput of the four hot paths: neighbour generation, ingest,
//! cascade, and residual solve.
//!
//! Every case warms its scratch before measurement, so the numbers describe the
//! allocation-free steady state that `tests/zero_alloc.rs` proves, not the
//! first-call growth. Criterion baseline artifacts are machine-specific and are
//! deliberately not committed; record the comparison conditions instead.

use criterion::{Criterion, criterion_group, criterion_main};
use fgf::{Gf8, gf8};
use sgraph::{
    Binary, CheckId, Constant, Edges, NeighborBuf, NeighborGen, Peeler, PoolConfig,
    ResidualBuilder, Solver, VarId, WindowedUniform,
};
use std::hint::black_box;
use std::num::NonZeroUsize;

/// The domain-separation constant is the consumer's; any fixed value will do for
/// a throughput measurement.
const DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;
const SYMBOL_LEN: usize = 1024;

/// Warm `NeighborBuf` reuse across checks: the per-check cost of regenerating an
/// edge set from a check id, which both peers pay for every symbol.
fn neighbors(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors");
    for degree in [3u32, 8, 32] {
        let distribution = Constant::new(degree).expect("non-zero degree");
        let generator =
            WindowedUniform::new(0, 4096, distribution, DOMAIN).expect("valid window geometry");
        let mut buf: NeighborBuf<Binary> =
            NeighborBuf::with_capacity(generator.max_degree() as usize);
        generator
            .neighbors(CheckId::new(0), &mut buf)
            .expect("warms the sampling scratch");

        group.throughput(criterion::Throughput::Elements(1));
        group.bench_function(format!("windowed_uniform/d{degree}"), |b| {
            let mut id = 0u64;
            b.iter(|| {
                id += 1;
                generator
                    .neighbors(CheckId::new(id), &mut buf)
                    .expect("generates");
                black_box(buf.support());
            });
        });
    }
    group.finish();
}

/// Ingest of a check that resolves nothing: edge validation, known-value folding,
/// and reverse-adjacency insertion, with no ripple.
///
/// The live set is held at a fixed depth by retiring below a lagging horizon on
/// every iteration, so each sample pays the same `O(DEPTH)` bookkeeping on top
/// of one ingest. A periodic bulk retirement instead would land a large spike in
/// a few random samples and swamp the signal.
fn ingest(c: &mut Criterion) {
    const DEPTH: u64 = 64;

    let mut group = c.benchmark_group("ingest");
    for degree in [3u64, 8, 32] {
        group.throughput(criterion::Throughput::Bytes(SYMBOL_LEN as u64));
        group.bench_function(format!("push_check/d{degree}"), |b| {
            let span = NonZeroUsize::new(4096).expect("non-zero span");
            let config = PoolConfig::new(span, span).with_pool_capacity(256);
            let mut peeler = Peeler::<Binary>::new(SYMBOL_LEN, config).expect("valid peeler");
            let weights = vec![Binary; degree as usize];
            let rhs = [0xA5u8; SYMBOL_LEN];
            let mut support = vec![VarId::ZERO; degree as usize];
            let mut id = 0u64;

            b.iter(|| {
                // A monotonically advancing support window: no row ever reaches
                // degree one, so this is ingest with no ripple.
                for (slot, offset) in support.iter_mut().zip(0..degree) {
                    *slot = VarId::new(id + offset);
                }
                peeler
                    .push_check(
                        CheckId::new(id),
                        Edges::new(&support, &weights).expect("valid edges"),
                        &rhs,
                    )
                    .expect("ingests");
                if id >= DEPTH {
                    peeler
                        .retire_below(VarId::new(id - DEPTH))
                        .expect("monotone retirement");
                }
                id += 1;
            });
        });
    }
    group.finish();
}

/// A chain of degree-two checks closed by one degree-one check: the final push
/// cascades through every row in the chain.
fn cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("cascade");
    for hops in [16u64, 256] {
        group.throughput(criterion::Throughput::Elements(hops + 1));
        group.bench_function(format!("chain/{hops}hops"), |b| {
            let span = NonZeroUsize::new(1024).expect("non-zero span");
            let config = PoolConfig::new(span, span).with_pool_capacity(1024);
            let mut peeler = Peeler::<Binary>::new(SYMBOL_LEN, config).expect("valid peeler");
            let rhs = [0xA5u8; SYMBOL_LEN];
            let mut recovered: Vec<VarId> = Vec::with_capacity(hops as usize + 1);
            let mut round = 0u64;

            b.iter(|| {
                // Each round works in a fresh id range on the same warm peeler,
                // so pools and rings are reused rather than reallocated.
                let base = round * (hops + 1);
                for hop in 0..hops {
                    let support = [VarId::new(base + hop), VarId::new(base + hop + 1)];
                    peeler
                        .push_check(
                            CheckId::new(base + hop),
                            Edges::new(&support, &[Binary; 2]).expect("valid pair"),
                            &rhs,
                        )
                        .expect("ingests");
                }
                // Closing the chain resolves `hops + 1` variables in one ripple.
                let tail = [VarId::new(base + hops)];
                peeler
                    .push_check(
                        CheckId::new(base + hops),
                        Edges::new(&tail, &[Binary; 1]).expect("valid singleton"),
                        &rhs,
                    )
                    .expect("ingests");
                recovered.clear();
                peeler.drain_recovered_into(&mut recovered);
                black_box(recovered.len());

                // Retirement recycles rows and known values back into the pools.
                round += 1;
                peeler
                    .retire_below(VarId::new(round * (hops + 1)))
                    .expect("monotone retirement");
            });
        });
    }
    group.finish();
}

/// Builder assembly plus full RREF over a dense GF(256) residual system — the
/// exact solve that finishes what peeling cannot.
fn solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");
    for size in [8usize, 32, 64] {
        let columns: Vec<VarId> = (0..size as u64).map(VarId::new).collect();
        // A deterministic non-singular pattern: full rank without needing an RNG.
        let coefficients: Vec<Vec<gf8::Elem>> = (0..size)
            .map(|row| {
                (0..size)
                    .map(|col| {
                        let raw = ((row * 7 + col * 13 + 1) % 255 + 1) as u8;
                        gf8::Elem(if row == col { raw | 1 } else { raw })
                    })
                    .collect()
            })
            .collect();
        let rhs = vec![0x5Au8; SYMBOL_LEN];

        let mut builder: ResidualBuilder<Gf8> = ResidualBuilder::new();
        let mut solver: Solver<Gf8> = Solver::new();
        // Warm both scratch buffers to this geometry.
        {
            let mut sink = builder.begin(&columns);
            for row in &coefficients {
                sink.push_dense(columns.iter().copied().zip(row.iter().copied()), &rhs);
            }
            let system = sink.finish().expect("well-formed system");
            solver.solve(&system).expect("full-rank system solves");
        }

        group.throughput(criterion::Throughput::Bytes(
            (size * size * SYMBOL_LEN) as u64,
        ));
        group.bench_function(format!("rref/{size}x{size}"), |b| {
            b.iter(|| {
                let mut sink = builder.begin(&columns);
                for row in &coefficients {
                    sink.push_dense(columns.iter().copied().zip(row.iter().copied()), &rhs);
                }
                let system = sink.finish().expect("well-formed system");
                let report = solver.solve(&system).expect("full-rank system solves");
                black_box(report.deficiency);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, neighbors, ingest, cascade, solve);
criterion_main!(benches);
