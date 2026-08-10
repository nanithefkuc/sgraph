//! Frozen vectors for the deterministic edge-generation machinery.
//!
//! Both peers regenerate a check symbol's edge set from its id, so the PRNG, the
//! sampling algorithm, and the draw order are wire properties. **A change that
//! moves a value in `tests/data/` is a format break for every downstream
//! consumer, not a refactor.**
//!
//! The expectations live in `tests/data/` rather than inline here so that their
//! provenance — the source revision and the capture command — travels with the
//! data. See `tests/data/README.md`, which also records what is deliberately
//! *not* pinned here and why.

use sgraph::rng::{SplitMix64, distinct_offsets, distinct_offsets_seeded, seed_for};
use sgraph::{
    Binary, CheckId, Edges, NeighborBuf, NeighborGen, Peeler, PoolConfig, Rfc5053Triple,
    RobustSoliton, Uniform, VarId, VariableState,
};
use std::num::{NonZeroU32, NonZeroUsize};

/// The domain-separation constant the captured streams were generated under.
/// `sgraph` does not bake in a domain constant; consumers choose their own.
const DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Fields of every non-comment, non-blank line of a fixture file.
fn fixture(name: &str) -> Vec<Vec<String>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");
    let text = std::fs::read_to_string(format!("{path}{name}"))
        .unwrap_or_else(|e| panic!("fixture {name} unreadable: {e}"));
    let rows: Vec<Vec<String>> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().map(str::to_owned).collect())
        .collect();
    assert!(!rows.is_empty(), "fixture {name} has no data rows");
    rows
}

fn hex(s: &str) -> u64 {
    let body = s.strip_prefix("0x").unwrap_or(s);
    u64::from_str_radix(body, 16).unwrap_or_else(|e| panic!("bad hex {s:?}: {e}"))
}

fn dec<T: std::str::FromStr>(s: &str) -> T
where
    T::Err: std::fmt::Display,
{
    s.parse()
        .unwrap_or_else(|e| panic!("bad number {s:?}: {e}"))
}

#[test]
fn splitmix64_streams_are_frozen() {
    let rows = fixture("splitmix64.txt");
    for row in &rows {
        let seed = hex(&row[0]);
        let expected: Vec<u64> = row[1..].iter().map(|s| hex(s)).collect();
        let mut rng = SplitMix64::new(seed);
        let got: Vec<u64> = (0..expected.len()).map(|_| rng.next_u64()).collect();
        assert_eq!(got, expected, "next_u64 stream changed for seed {seed:#x}");
    }
}

#[test]
fn below_streams_are_frozen() {
    let rows = fixture("below.txt");
    for row in &rows {
        let seed = hex(&row[0]);
        let bound = NonZeroU32::new(dec(&row[1])).expect("fixture bound must be non-zero");
        let expected: Vec<u32> = row[2..].iter().map(|s| dec(s)).collect();
        let mut rng = SplitMix64::new(seed);
        let got: Vec<u32> = (0..expected.len()).map(|_| rng.below(bound)).collect();
        assert_eq!(
            got, expected,
            "below({bound}) stream changed for seed {seed:#x}"
        );
    }
}

/// The vectors that matter most: these are the graph edges themselves.
#[test]
fn edge_offsets_are_frozen() {
    let rows = fixture("offsets.txt");
    for row in &rows {
        let id: u64 = dec(&row[0]);
        let span: u32 = dec(&row[1]);
        let k: usize = dec(&row[2]);
        let expected: Vec<u32> = row[3..].iter().map(|s| dec(s)).collect();
        assert_eq!(
            expected.len(),
            k,
            "fixture row for id {id} disagrees with k"
        );

        let mut got = vec![0u32; k];
        distinct_offsets_seeded(seed_for(id, DOMAIN), span, &mut got)
            .expect("fixture geometry must be valid");
        assert_eq!(
            got, expected,
            "edge set changed for check {id} over span {span}"
        );
    }
}

/// Charter invariant 2, as an executable check rather than a comment.
///
/// Sampling takes `&mut SplitMix64` so a degree draw can precede an edge draw in
/// one stream. That refactor is only compatible with the frozen vectors because
/// handing in a freshly-seeded generator consumes nothing first — so the seeded
/// convenience form and the explicit form must agree exactly.
#[test]
fn seeded_and_borrowed_sampling_agree() {
    for id in 0..256u64 {
        for span in [1u32, 2, 8, 16, 64, 1024] {
            for k in [0usize, 1, 2, 3, 4] {
                if (span as usize) < k {
                    continue;
                }
                let seed = seed_for(id, DOMAIN);

                let mut via_seed = vec![0u32; k];
                distinct_offsets_seeded(seed, span, &mut via_seed).expect("valid geometry");

                let mut via_borrow = vec![0u32; k];
                let mut rng = SplitMix64::new(seed);
                distinct_offsets(&mut rng, span, &mut via_borrow).expect("valid geometry");

                assert_eq!(via_seed, via_borrow, "id {id}, span {span}, k {k}");
            }
        }
    }
}

/// A point-mass degree distribution must be able to draw nothing, or the
/// generators could not stay bit-compatible with the frozen offsets while
/// composing a degree draw into the same stream.
#[test]
fn empty_request_consumes_no_draws() {
    let mut rng = SplitMix64::new(seed_for(7, DOMAIN));
    distinct_offsets(&mut rng, 64, &mut []).expect("zero offsets is always satisfiable");

    // The stream must be exactly where it started: a subsequent sample matches
    // the frozen vector for this id.
    let mut got = [0u32; 3];
    distinct_offsets(&mut rng, 16, &mut got).expect("valid geometry");
    assert_eq!(got, [3, 8, 15], "an empty request perturbed the stream");
}

/// Rejected geometry must not half-write caller output.
#[test]
fn oversized_request_is_rejected_without_output() {
    let mut rng = SplitMix64::new(1);
    let before = rng.next_u64();
    let mut rng = SplitMix64::new(1);

    let mut out = [u32::MAX; 4];
    let err = distinct_offsets(&mut rng, 3, &mut out).expect_err("4 offsets from span 3");
    assert!(matches!(
        err,
        sgraph::GraphError::SampleSpanTooSmall {
            span: 3,
            requested: 4
        }
    ));
    assert_eq!(out, [u32::MAX; 4], "output was modified on a rejected call");
    assert_eq!(
        rng.next_u64(),
        before,
        "generator was advanced on a rejected call"
    );
}

/// The RFC 5053 LT graph, against a reference that never ran this code.
///
/// `tests/data/rfc5053_lt.txt` was produced by a separate transcription of the
/// specification, so agreement here is cross-implementation agreement on the
/// edges themselves — the property two peers actually depend on.
#[test]
fn rfc5053_lt_edges_match_an_independent_reference() {
    let rows = fixture("rfc5053_lt.txt");
    let mut buf = NeighborBuf::<Binary>::new();
    let mut generators: Vec<(u32, Rfc5053Triple)> = Vec::new();

    for row in &rows {
        let k: u32 = dec(&row[0]);
        let id: u64 = dec(&row[1]);
        let expected_triple: (u32, u32, u32) = (dec(&row[2]), dec(&row[3]), dec(&row[4]));
        let expected_l: u32 = dec(&row[5]);
        let expected: Vec<u64> = row[6..].iter().map(|s| dec(s)).collect();

        if generators.last().map(|(seen, _)| *seen) != Some(k) {
            generators.push((k, Rfc5053Triple::new(k).expect("fixture K is normative")));
        }
        let generator = &generators.last().expect("just pushed").1;

        assert_eq!(generator.intermediate_count(), expected_l, "L for K={k}");
        assert_eq!(
            generator.triple(CheckId::new(id)),
            expected_triple,
            "Trip[{k}, {id}]"
        );

        generator
            .neighbors(CheckId::new(id), &mut buf)
            .expect("the LT walk is total");
        let got: Vec<u64> = buf.support().iter().map(|var| var.get()).collect();
        assert_eq!(got, expected, "LT neighbours for K={k}, X={id}");
    }

    assert_eq!(rows.len(), 320, "fixture corpus shrank");
}

/// A fixed LT configuration's recovery rate, over a committed id corpus.
///
/// Unlike the vector tests this is a *quality* measurement rather than a
/// correctness gate: it says how much overhead this geometry needs, and it is
/// exact only because every draw in it is deterministic. A change here is a
/// graph-quality regression to explain, not automatically a bug.
#[test]
fn robust_soliton_lt_recovery_rate_is_pinned() {
    const K: u64 = 256;
    const SYMBOL_LEN: usize = 8;
    // c = 0.05, delta = 0.5 as Q32 fractions.
    const C_Q32: u32 = 214_748_365;
    const DELTA_Q32: u32 = 1 << 31;

    let degree = RobustSoliton::from_q32(K as u32, C_Q32, DELTA_Q32).expect("valid parameters");
    let generator = Uniform::new(K, degree, DOMAIN).expect("valid geometry");
    let span = NonZeroUsize::new(4096).expect("non-zero span");
    let mut buf = NeighborBuf::<Binary>::with_capacity(K as usize);

    // Overhead in percent -> how many of the 16 trials recover every symbol.
    let mut recovered_by_overhead = Vec::new();
    for overhead_percent in [0u64, 5, 10, 20, 40] {
        let checks = K + K * overhead_percent / 100;
        let mut full = 0u32;
        for trial in 0..16u64 {
            let mut peeler =
                Peeler::<Binary>::new(SYMBOL_LEN, PoolConfig::new(span, span)).expect("valid");
            let base = trial * 100_000;
            for j in 0..checks {
                let id = CheckId::new(base + j);
                // The check's value is the XOR of its neighbours' true values,
                // which for this fixture is a byte pattern derived from the id.
                generator.neighbors(id, &mut buf).expect("valid geometry");
                let mut rhs = [0u8; SYMBOL_LEN];
                for var in buf.support() {
                    for (out, byte) in rhs.iter_mut().zip(symbol(var.get())) {
                        *out ^= byte;
                    }
                }
                let edges = Edges::new(buf.support(), buf.weights()).expect("generated edges");
                peeler.push_check(id, edges, &rhs).expect("bounded ingest");
            }
            // Every recovered value must be the true one; the count is the
            // measurement.
            let mut known = 0u64;
            for v in 0..K {
                if let VariableState::Known(value) = peeler.variable_state(VarId::new(v)) {
                    assert_eq!(value, symbol(v), "peeling recovered a wrong value for {v}");
                    known += 1;
                }
            }
            full += u32::from(known == K);
        }
        recovered_by_overhead.push((overhead_percent, full));
    }

    assert_eq!(
        recovered_by_overhead,
        vec![(0, 0), (5, 0), (10, 0), (20, 10), (40, 16)],
        "LT recovery rate moved for K={K}"
    );
}

/// The true value of variable `v` in the recovery measurement above.
fn symbol(v: u64) -> [u8; 8] {
    (v.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5A5A_A5A5_3C3C_C3C3).to_le_bytes()
}
