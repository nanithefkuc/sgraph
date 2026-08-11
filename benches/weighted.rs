//! Weighted-peeling throughput, isolated so its monomorphizations cannot perturb
//! the binary benchmark executable used for regression comparisons.

use criterion::{Criterion, criterion_group, criterion_main};
use fgf::{Gf8, gf8};
use sgraph::{CheckId, Edges, Peeler, PoolConfig, VarId, Weighted};
use std::hint::black_box;
use std::num::NonZeroUsize;

const SYMBOL_LEN: usize = 1024;

fn weighted_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted");
    for hops in [16u64, 256] {
        group.throughput(criterion::Throughput::Elements(hops + 1));
        group.bench_function(format!("cascade/{hops}hops"), |b| {
            let span = NonZeroUsize::new(1024).expect("non-zero span");
            let config = PoolConfig::new(span, span).with_pool_capacity(1024);
            let mut peeler =
                Peeler::<Weighted<Gf8>>::new(SYMBOL_LEN, config).expect("valid peeler");
            let rhs = [0xA5u8; SYMBOL_LEN];
            let pair_weights = [
                Weighted::new(gf8::Elem(3)).expect("non-zero"),
                Weighted::new(gf8::Elem(5)).expect("non-zero"),
            ];
            let singleton_weight = [Weighted::new(gf8::Elem(7)).expect("non-zero")];
            let mut recovered: Vec<VarId> = Vec::with_capacity(hops as usize + 1);
            let mut round = 0u64;

            b.iter(|| {
                let base = round * (hops + 1);
                for hop in 0..hops {
                    let support = [VarId::new(base + hop), VarId::new(base + hop + 1)];
                    peeler
                        .push_check(
                            CheckId::new(base + hop),
                            Edges::new(&support, &pair_weights).expect("valid pair"),
                            &rhs,
                        )
                        .expect("ingests");
                }
                let tail = [VarId::new(base + hops)];
                peeler
                    .push_check(
                        CheckId::new(base + hops),
                        Edges::new(&tail, &singleton_weight).expect("valid singleton"),
                        &rhs,
                    )
                    .expect("ingests");
                recovered.clear();
                peeler.drain_recovered_into(&mut recovered);
                black_box(recovered.len());
                round += 1;
                peeler
                    .retire_below(VarId::new(round * (hops + 1)))
                    .expect("monotone retirement");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, weighted_cascade);
criterion_main!(benches);
