use super::*;
use crate::degree::{Constant, DegreeDistribution};
use crate::rng::{SplitMix64, distinct_offsets_seeded, seed_for};
use crate::weight::Binary;
use alloc::vec;
use alloc::vec::Vec;

/// The domain constant the frozen fixtures were captured under.
const DOMAIN_SEP: u64 = 0xA5A5_5A5A_C3C3_3C3C;

fn c(d: u32) -> Constant {
    Constant::new(d).expect("non-zero test degree")
}

/// A distribution that can only produce edgeless checks. `Constant` refuses to be
/// built this way, so a generator's own zero-degree guard needs its own witness.
#[derive(Debug)]
struct ZeroDegree;

impl DegreeDistribution for ZeroDegree {
    fn sample(&self, _rng: &mut SplitMix64) -> u32 {
        0
    }
    fn max_degree(&self) -> u32 {
        0
    }
}

/// A distribution that consumes generator state, to prove the composed stream is
/// genuinely shared rather than two independent streams on one seed.
#[derive(Debug)]
struct DrawingDegree(u32);

impl DegreeDistribution for DrawingDegree {
    fn sample(&self, rng: &mut SplitMix64) -> u32 {
        let _ = rng.next_u64();
        self.0
    }
    fn max_degree(&self) -> u32 {
        self.0
    }
}

// --- Edges validation ------------------------------------------------------

#[test]
fn edges_accept_a_well_formed_set() {
    let support = [VarId::new(4), VarId::new(1), VarId::new(9)];
    let weights = [Binary; 3];
    let e = Edges::new(&support, &weights).unwrap();
    assert_eq!(e.len(), 3);
    assert!(!e.is_empty());
    assert_eq!(e.support(), &support);
    assert_eq!(e.min_var(), VarId::new(1), "support is unsorted");
    assert_eq!(
        e.iter().collect::<Vec<_>>(),
        [
            (VarId::new(4), Binary),
            (VarId::new(1), Binary),
            (VarId::new(9), Binary)
        ]
    );
}

#[test]
fn edges_reject_length_mismatch() {
    let support = [VarId::new(1), VarId::new(2)];
    let weights = [Binary; 1];
    assert_eq!(
        Edges::new(&support, &weights),
        Err(GraphError::EdgeLengthMismatch {
            support: 2,
            weights: 1
        })
    );
}

#[test]
fn edges_reject_empty_support() {
    assert_eq!(
        Edges::<Binary>::new(&[], &[]),
        Err(GraphError::EmptySupport)
    );
}

/// A duplicate would fold twice during reduction and silently break the residual
/// invariant, so it must be caught at the boundary rather than tolerated.
#[test]
fn edges_reject_a_duplicate_variable() {
    let support = [VarId::new(3), VarId::new(7), VarId::new(3)];
    let weights = [Binary; 3];
    assert_eq!(
        Edges::new(&support, &weights),
        Err(GraphError::DuplicateVariable { var: 3 })
    );
}

// --- NeighborBuf scratch contract -----------------------------------------

#[test]
fn neighbor_buf_round_trips_edges() {
    let mut buf: NeighborBuf<Binary> = NeighborBuf::with_capacity(4);
    assert!(buf.is_empty());
    buf.push(VarId::new(2), Binary);
    buf.push(VarId::new(5), Binary);
    assert_eq!(buf.len(), 2);
    assert_eq!(buf.support(), &[VarId::new(2), VarId::new(5)]);
    assert_eq!(buf.weights(), &[Binary, Binary]);
    assert_eq!(buf.edges().unwrap().len(), 2);

    buf.clear();
    assert!(buf.is_empty());
    assert_eq!(buf.edges(), Err(GraphError::EmptySupport));
}

#[test]
fn neighbor_buf_scratch_grows_and_persists() {
    let mut buf: NeighborBuf<Binary> = NeighborBuf::new();
    buf.offset_scratch(4).copy_from_slice(&[9, 8, 7, 6]);
    buf.fill_from_offsets(3, Binary, |o| VarId::new(u64::from(o) * 10));
    assert_eq!(
        buf.support(),
        &[VarId::new(90), VarId::new(80), VarId::new(70)]
    );
    assert_eq!(buf.weights().len(), 3);

    // A smaller later request must not shrink the allocation.
    let scratch = buf.offset_scratch(2);
    assert_eq!(scratch.len(), 2);
}

// --- Generator construction ------------------------------------------------

#[test]
fn uniform_rejects_bad_geometry() {
    assert_eq!(
        Uniform::new(0, c(1), 0).map(|_| ()),
        Err(GraphError::EmptyDomain)
    );
    assert_eq!(
        Uniform::new(16, ZeroDegree, 0).map(|_| ()),
        Err(GraphError::ZeroDegree),
        "a distribution that can only produce zero edges is a config bug"
    );
    assert_eq!(
        Uniform::new(4, c(5), 0).map(|_| ()),
        Err(GraphError::DegreeExceedsDomain {
            degree: 5,
            domain: 4
        })
    );
    assert_eq!(
        Uniform::new(u64::from(u32::MAX) + 1, c(3), 0).map(|_| ()),
        Err(GraphError::DomainTooLarge {
            domain: u64::from(u32::MAX) + 1,
            max: u64::from(u32::MAX)
        }),
        "offsets are u32, so a wider block is unaddressable"
    );
}

/// Degree exactly equal to the domain is legal and selects every variable.
#[test]
fn uniform_accepts_degree_equal_to_domain() {
    let g = Uniform::new(8, c(8), DOMAIN_SEP).unwrap();
    let mut buf = NeighborBuf::with_capacity(g.max_degree() as usize);
    g.neighbors(CheckId::new(3), &mut buf).unwrap();
    let mut got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
    got.sort_unstable();
    assert_eq!(got, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(g.domain(), 8);
    // Still a valid edge set: distinct, non-empty, unit weights.
    assert_eq!(buf.edges().unwrap().len(), 8);
}

#[test]
fn windowed_uniform_rejects_bad_geometry() {
    assert_eq!(
        WindowedUniform::new(0, 0, c(1), 0).map(|_| ()),
        Err(GraphError::EmptyDomain)
    );
    assert_eq!(
        WindowedUniform::new(0, 8, ZeroDegree, 0).map(|_| ()),
        Err(GraphError::ZeroDegree)
    );
    assert_eq!(
        WindowedUniform::new(u64::MAX, 2, c(1), 0).map(|_| ()),
        Err(GraphError::DomainOverflow {
            base: u64::MAX,
            span: 2
        }),
        "the window's last index would pass u64::MAX"
    );
    // A window ending exactly at u64::MAX is addressable.
    let g = WindowedUniform::new(u64::MAX, 1, c(1), DOMAIN_SEP).unwrap();
    let mut buf = NeighborBuf::with_capacity(1);
    g.neighbors(CheckId::new(0), &mut buf).unwrap();
    assert_eq!(buf.support(), &[VarId::new(u64::MAX)]);
}

/// Unlike `Uniform`, a window may be narrower than the degree: early in a stream
/// there is less to point at, so the degree is clamped rather than rejected.
#[test]
fn windowed_uniform_clamps_degree_to_span() {
    let g = WindowedUniform::new(100, 2, c(5), DOMAIN_SEP).unwrap();
    assert_eq!(g.max_degree(), 2, "clamped at construction");
    let mut buf = NeighborBuf::with_capacity(5);
    g.neighbors(CheckId::new(0), &mut buf).unwrap();
    assert_eq!(buf.len(), 2);
    let mut got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
    got.sort_unstable();
    assert_eq!(got, [100, 101]);
}

// --- Bit-exactness with the frozen fixtures --------------------------------

/// Charter invariant 2. `WindowedUniform` with a point-mass degree must produce
/// exactly the offsets captured from the extraction source, translated by `base`.
///
/// The cases mirror `tests/data/offsets.txt`; the fixture file itself is asserted
/// by `tests/vectors.rs`, and this checks the generator built on top of it.
#[test]
fn windowed_uniform_matches_the_frozen_offsets() {
    // (id, span, k, expected offsets)
    let cases: &[(u64, u32, usize, &[u64])] = &[
        (5, 16, 3, &[12, 8, 10]),
        (6, 16, 3, &[6, 7, 12]),
        (7, 16, 3, &[3, 8, 15]),
        (42, 64, 3, &[1, 8, 7]),
        (43, 64, 4, &[25, 17, 26, 47]),
        (1000, 64, 1, &[20]),
        (u64::MAX, 64, 3, &[44, 21, 63]),
    ];

    for &(id, span, k, expected) in cases {
        for base in [0u64, 1, 1_000, 1 << 40] {
            let g = WindowedUniform::new(base, span, c(k as u32), DOMAIN_SEP).unwrap();
            let mut buf = NeighborBuf::with_capacity(k);
            g.neighbors(CheckId::new(id), &mut buf).unwrap();

            let got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
            let want: Vec<u64> = expected.iter().map(|o| base + o).collect();
            assert_eq!(got, want, "check {id}, span {span}, base {base}");
        }
    }
}

/// `k = min(dc, span)` is the early-stream shape, and an off-by-one in the clamp
/// would silently change the graph.
#[test]
fn windowed_uniform_matches_frozen_clamped_offsets() {
    let mut buf = NeighborBuf::with_capacity(8);

    // Fixture `0 2 2 -> [0, 1]`: a nominal degree of 3 clamped to a span of 2.
    let g = WindowedUniform::new(0, 2, c(3), DOMAIN_SEP).unwrap();
    g.neighbors(CheckId::new(0), &mut buf).unwrap();
    assert_eq!(buf.len(), 2, "span 2 clamps degree 3 to 2");
    let mut got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
    got.sort_unstable();
    assert_eq!(got, [0, 1]);

    // Fixture `0 3 2 -> [1, 2]`: degree 2 over span 3, unclamped, and the order
    // is Floyd's rather than sorted.
    let g = WindowedUniform::new(0, 3, c(2), DOMAIN_SEP).unwrap();
    g.neighbors(CheckId::new(0), &mut buf).unwrap();
    let got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
    assert_eq!(got, [1, 2]);

    // Fixture `0 1 1 -> [0]`: a single-variable window clamps any degree to 1.
    let g = WindowedUniform::new(0, 1, c(4), DOMAIN_SEP).unwrap();
    g.neighbors(CheckId::new(0), &mut buf).unwrap();
    assert_eq!(
        buf.support(),
        &[VarId::ZERO],
        "a one-wide window has exactly one edge"
    );

    // Whenever the clamp binds exactly at the span, Floyd selects everything, in
    // ascending order — the property the `span == k` fixture rows record.
    let g = WindowedUniform::new(0, 8, c(40), DOMAIN_SEP).unwrap();
    g.neighbors(CheckId::new(7), &mut buf).unwrap();
    let got: Vec<u64> = buf.support().iter().map(|v| v.get()).collect();
    assert_eq!(got, [0, 1, 2, 3, 4, 5, 6, 7]);
}

/// `Uniform` over `[0, span)` is the same sampling as a window based at zero, so
/// it must agree with the raw fixture offsets too.
#[test]
fn uniform_matches_the_frozen_offsets() {
    let g = Uniform::new(64, c(3), DOMAIN_SEP).unwrap();
    let mut buf = NeighborBuf::with_capacity(3);
    g.neighbors(CheckId::new(42), &mut buf).unwrap();
    assert_eq!(
        buf.support(),
        &[VarId::new(1), VarId::new(8), VarId::new(7)]
    );
}

/// Both peers must agree. Generating twice — and from a separately constructed
/// generator — must give byte-identical edges.
#[test]
fn generation_is_reproducible() {
    let a = WindowedUniform::new(10, 32, c(4), DOMAIN_SEP).unwrap();
    let b = WindowedUniform::new(10, 32, c(4), DOMAIN_SEP).unwrap();
    let mut buf_a = NeighborBuf::new();
    let mut buf_b = NeighborBuf::new();
    for id in 0..200u64 {
        a.neighbors(CheckId::new(id), &mut buf_a).unwrap();
        b.neighbors(CheckId::new(id), &mut buf_b).unwrap();
        assert_eq!(buf_a.support(), buf_b.support(), "diverged at check {id}");
    }
}

/// A different domain constant must give a different graph, or domain separation
/// would not separate anything.
#[test]
fn domain_separation_changes_the_graph() {
    let a = WindowedUniform::new(0, 64, c(3), DOMAIN_SEP).unwrap();
    let b = WindowedUniform::new(0, 64, c(3), !DOMAIN_SEP).unwrap();
    let mut buf_a = NeighborBuf::new();
    let mut buf_b = NeighborBuf::new();
    let mut differ = 0;
    for id in 0..64u64 {
        a.neighbors(CheckId::new(id), &mut buf_a).unwrap();
        b.neighbors(CheckId::new(id), &mut buf_b).unwrap();
        if buf_a.support() != buf_b.support() {
            differ += 1;
        }
    }
    assert!(differ > 50, "only {differ}/64 checks differed");
}

/// The composed stream must be genuinely shared: a distribution that consumes
/// state has to move the offsets, or degree and edges would be two independent
/// streams keyed on one seed.
#[test]
fn a_drawing_distribution_shifts_the_offsets() {
    let point = WindowedUniform::new(0, 64, c(3), DOMAIN_SEP).unwrap();
    let drawing = WindowedUniform::new(0, 64, DrawingDegree(3), DOMAIN_SEP).unwrap();
    let mut a = NeighborBuf::new();
    let mut b = NeighborBuf::new();
    point.neighbors(CheckId::new(42), &mut a).unwrap();
    drawing.neighbors(CheckId::new(42), &mut b).unwrap();
    assert_ne!(a.support(), b.support());
}

/// Every generated edge set must satisfy the ingest invariant without further
/// repair: distinct variables, in range, non-zero weights.
#[test]
fn generated_edges_are_always_valid() {
    let g = WindowedUniform::new(1_000, 32, c(5), DOMAIN_SEP).unwrap();
    let mut buf = NeighborBuf::with_capacity(g.max_degree() as usize);
    for id in 0..500u64 {
        g.neighbors(CheckId::new(id), &mut buf).unwrap();
        let e = buf.edges().expect("generated edges must validate");
        assert_eq!(e.len(), 5);
        for v in e.support() {
            assert!(
                (1_000..1_032).contains(&v.get()),
                "variable {v} outside the window"
            );
        }
    }
}

/// The generator must agree with the underlying sampler for a point-mass degree.
/// If these ever diverge, one of the two changed the stream.
#[test]
fn generator_agrees_with_raw_sampling() {
    let g = Uniform::new(1024, c(40), DOMAIN_SEP).unwrap();
    let mut buf = NeighborBuf::new();
    g.neighbors(CheckId::new(12345), &mut buf).unwrap();

    let mut raw = vec![0u32; 40];
    distinct_offsets_seeded(seed_for(12345, DOMAIN_SEP), 1024, &mut raw).unwrap();

    let got: Vec<u32> = buf.support().iter().map(|v| v.get() as u32).collect();
    assert_eq!(got, raw);
}

/// A failing generator must leave the scratch cleared and reusable, so the next
/// check does not observe debris from the last one.
#[test]
fn a_failing_generator_leaves_the_buffer_clear() {
    /// Fails after writing, to prove `neighbors` is not trusted to be atomic by
    /// accident.
    struct Failing;

    impl NeighborGen for Failing {
        type Weight = Binary;
        fn neighbors(&self, _id: CheckId, out: &mut NeighborBuf<Binary>) -> Result<(), GraphError> {
            out.clear();
            out.push(VarId::new(1), Binary);
            out.push(VarId::new(2), Binary);
            // Contract: clear before returning an error.
            out.clear();
            Err(GraphError::EmptyDomain)
        }
        fn max_degree(&self) -> u32 {
            2
        }
    }

    let mut buf = NeighborBuf::with_capacity(4);
    buf.push(VarId::new(99), Binary);

    assert_eq!(
        Failing.neighbors(CheckId::new(0), &mut buf),
        Err(GraphError::EmptyDomain)
    );
    assert!(buf.is_empty(), "error path left debris in the scratch");
    assert_eq!(buf.edges(), Err(GraphError::EmptySupport));

    // And the buffer is still usable for a subsequent successful generation.
    let g = Uniform::new(16, c(2), DOMAIN_SEP).unwrap();
    g.neighbors(CheckId::new(1), &mut buf).unwrap();
    assert_eq!(buf.len(), 2);
}
