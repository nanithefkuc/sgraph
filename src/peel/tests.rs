use super::{Peeler, PoolConfig, VariableState};
use crate::{
    Binary, CheckId, Constant, EdgeWeight, Edges, GraphError, NeighborBuf, NeighborGen, VarId,
    Weighted, WindowedUniform,
};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use fgf::{Gf8, Gf16, gf8};

const NEIGHBOR_DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;

#[derive(Debug)]
struct Repair {
    id: CheckId,
    base: u64,
    span: u32,
    rhs: Vec<u8>,
}

fn config(span: usize) -> PoolConfig {
    let bound = NonZeroUsize::new(span).unwrap();
    PoolConfig::new(bound, bound).with_pool_capacity(span)
}

fn source_symbols(count: usize, symbol_len: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| {
            (0..symbol_len)
                .map(|j| i.wrapping_mul(131).wrapping_add(j.wrapping_mul(17)) as u8)
                .collect()
        })
        .collect()
}

fn xor_assign(dst: &mut [u8], src: &[u8]) {
    for (to, from) in dst.iter_mut().zip(src) {
        *to ^= from;
    }
}

fn encode_stream(
    count: usize,
    symbol_len: usize,
    window: usize,
    degree: u32,
    repairs_per_source: usize,
) -> (Vec<Vec<u8>>, Vec<Repair>) {
    let sources = source_symbols(count, symbol_len);
    let mut repairs = Vec::with_capacity(count * repairs_per_source);
    let mut scratch = NeighborBuf::with_capacity(usize::try_from(degree).unwrap());
    let distribution = Constant::new(degree).unwrap();
    let mut next_check = 0u64;
    for i in 0..count {
        let end = i + 1;
        let base = end.saturating_sub(window);
        let span = u32::try_from(end - base).unwrap();
        for _ in 0..repairs_per_source {
            let id = CheckId::new(next_check);
            next_check += 1;
            let generator =
                WindowedUniform::new(base as u64, span, distribution, NEIGHBOR_DOMAIN).unwrap();
            generator.neighbors(id, &mut scratch).unwrap();
            let edges = scratch.edges().unwrap();
            let mut rhs = vec![0; symbol_len];
            for &var in edges.support() {
                xor_assign(&mut rhs, &sources[usize::try_from(var.get()).unwrap()]);
            }
            repairs.push(Repair {
                id,
                base: base as u64,
                span,
                rhs,
            });
        }
    }
    (sources, repairs)
}

fn push_repairs(peeler: &mut Peeler<Binary>, repairs: &[Repair], degree: u32) {
    let distribution = Constant::new(degree).unwrap();
    for repair in repairs {
        let generator =
            WindowedUniform::new(repair.base, repair.span, distribution, NEIGHBOR_DOMAIN).unwrap();
        peeler
            .push_check_with(repair.id, &generator, &repair.rhs)
            .unwrap();
    }
}

fn assert_full_recovery(peeler: &Peeler<Binary>, sources: &[Vec<u8>]) {
    for (i, source) in sources.iter().enumerate() {
        assert_eq!(
            peeler.variable_state(VarId::new(i as u64)),
            VariableState::Known(source),
            "variable {i} was not recovered exactly"
        );
    }
}

#[test]
fn systematic_no_loss_is_passthrough() {
    let sources = source_symbols(80, 64);
    let mut peeler = Peeler::<Binary>::new(64, config(512)).unwrap();
    for (i, source) in sources.iter().enumerate() {
        peeler.learn_copy(VarId::new(i as u64), source).unwrap();
    }
    assert_full_recovery(&peeler, &sources);
    assert!(!peeler.has_stalled());
}

#[test]
fn peeling_recovers_sparse_losses() {
    let (sources, repairs) = encode_stream(96, 64, 32, 3, 3);
    let mut peeler = Peeler::<Binary>::new(64, config(512)).unwrap();
    for (i, source) in sources.iter().enumerate() {
        if i != 20 && i != 60 {
            peeler.learn_copy(VarId::new(i as u64), source).unwrap();
        }
    }
    push_repairs(&mut peeler, &repairs, 3);
    assert_full_recovery(&peeler, &sources);
}

#[test]
fn peeling_recovers_with_reordered_delivery() {
    let (sources, repairs) = encode_stream(96, 64, 32, 3, 3);
    let mut peeler = Peeler::<Binary>::new(64, config(512)).unwrap();
    push_repairs(&mut peeler, &repairs, 3);
    for (i, source) in sources.iter().enumerate() {
        if i != 15 && i != 60 {
            peeler.learn_copy(VarId::new(i as u64), source).unwrap();
        }
    }
    assert_full_recovery(&peeler, &sources);
}

#[test]
fn peeling_cascades_across_checks() {
    let (sources, repairs) = encode_stream(64, 32, 16, 3, 3);
    let mut peeler = Peeler::<Binary>::new(32, config(512)).unwrap();
    for (i, source) in sources.iter().enumerate() {
        if i != 20 && i != 21 && i != 40 {
            peeler.learn_copy(VarId::new(i as u64), source).unwrap();
        }
    }
    push_repairs(&mut peeler, &repairs, 3);
    assert_full_recovery(&peeler, &sources);
}

#[test]
fn stopping_set_exposes_exact_residual_rows() {
    let mut peeler = Peeler::<Binary>::new(2, config(8)).unwrap();
    let support = [VarId::new(0), VarId::new(1)];
    let weights = [Binary; 2];
    let rhs = [0xA5, 0x5A];
    let edges = Edges::new(&support, &weights).unwrap();
    peeler.push_check(CheckId::new(0), edges, &rhs).unwrap();
    peeler.push_check(CheckId::new(1), edges, &rhs).unwrap();

    assert!(peeler.has_stalled());
    assert_eq!(peeler.unresolved_count(), 2);
    assert_eq!(peeler.variable_state(support[0]), VariableState::Unknown);
    assert_eq!(peeler.variable_state(support[1]), VariableState::Unknown);
    let rows: Vec<_> = peeler.stalled_rows().collect();
    assert_eq!(rows.len(), 2);
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.check(), CheckId::new(index as u64));
        assert_eq!(row.support(), support);
        assert_eq!(row.weights(), weights);
        assert_eq!(row.rhs(), rhs);
    }
}

#[test]
fn retirement_uses_current_residual_support() {
    let mut peeler = Peeler::<Binary>::new(2, config(16)).unwrap();
    let values = [[0x11, 0x22], [0x33, 0x44], [0x55, 0x66]];
    let support = [VarId::new(0), VarId::new(2), VarId::new(3)];
    let weights = [Binary; 3];
    let mut rhs = [0u8; 2];
    for value in &values {
        xor_assign(&mut rhs, value);
    }
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&support, &weights).unwrap(),
            &rhs,
        )
        .unwrap();
    peeler.learn_copy(support[0], &values[0]).unwrap();
    peeler.retire_below(VarId::new(1)).unwrap();
    assert_eq!(
        peeler.check_count(),
        1,
        "folded-out old minimum is not stale"
    );
    peeler.learn_copy(support[1], &values[1]).unwrap();
    assert_eq!(
        peeler.variable_state(support[2]),
        VariableState::Known(&values[2])
    );

    let mut stale = Peeler::<Binary>::new(2, config(16)).unwrap();
    let stale_support = [VarId::new(0), VarId::new(2)];
    stale
        .push_check(
            CheckId::new(0),
            Edges::new(&stale_support, &[Binary; 2]).unwrap(),
            &[0x77, 0x88],
        )
        .unwrap();
    stale.retire_below(VarId::new(1)).unwrap();
    assert_eq!(stale.check_count(), 0);
    assert!(!stale.has_stalled());
    assert_eq!(stale.variable_state(VarId::new(0)), VariableState::Retired);
    assert_eq!(stale.variable_state(VarId::new(2)), VariableState::Unknown);
}

#[test]
fn explicit_check_retirement_is_final() {
    let mut peeler = Peeler::<Binary>::new(1, config(8)).unwrap();
    let support = [VarId::new(0), VarId::new(1)];
    let weights = [Binary; 2];
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&support, &weights).unwrap(),
            &[0xAA],
        )
        .unwrap();
    peeler.retire_check(CheckId::new(0)).unwrap();
    assert_eq!(peeler.check_count(), 0);
    peeler.learn_copy(support[0], &[0x10]).unwrap();
    peeler.learn_copy(support[1], &[0x20]).unwrap();
    assert_eq!(
        peeler.variable_state(support[0]),
        VariableState::Known(&[0x10])
    );
    assert_eq!(
        peeler.variable_state(support[1]),
        VariableState::Known(&[0x20])
    );
    assert!(matches!(
        peeler.push_check(
            CheckId::new(0),
            Edges::new(&support, &weights).unwrap(),
            &[0xAA],
        ),
        Err(GraphError::IndexRetired { .. } | GraphError::CheckRetired { .. })
    ));
}

#[test]
fn retiring_one_check_preserves_a_vacant_later_id() {
    let mut peeler = Peeler::<Binary>::new(1, config(8)).unwrap();
    let support = [VarId::new(2), VarId::new(3)];
    let weights = [Binary; 2];
    let edges = Edges::new(&support, &weights).unwrap();
    peeler.push_check(CheckId::new(2), edges, &[0xAA]).unwrap();

    peeler.retire_check(CheckId::new(0)).unwrap();
    assert!(matches!(
        peeler.push_check(CheckId::new(0), edges, &[0xAA]),
        Err(GraphError::IndexRetired { .. } | GraphError::CheckRetired { .. })
    ));
    peeler.push_check(CheckId::new(1), edges, &[0xAA]).unwrap();

    assert_eq!(peeler.check_count(), 2);
}

/// A variable learned before any check names it has no reverse-adjacency slot,
/// because `waiting` is grown only where a check registers a waiter. The first
/// check that names it must still fold it out, and the ring that anchors on that
/// check's residual support must still accept an *earlier* live variable named by
/// a later check — `Ring::ensure` grows at the front as well as the back.
#[test]
fn a_check_naming_an_already_known_variable_still_cascades() {
    let mut peeler = Peeler::<Binary>::new(2, config(16)).unwrap();
    let v0 = [0x11, 0x22];
    let v1 = [0x33, 0x44];
    let v2 = [0x55, 0x66];
    let v5 = [0x77, 0x88];
    let mut recovered = Vec::new();

    // Learned first: nothing is waiting on variable 0, so it gets no slot.
    peeler.learn_copy(VarId::new(0), &v0).unwrap();

    // Names the already-known variable 0 plus the unknown variable 5. Variable 0
    // folds out at ingest and the degree-one residual peels variable 5. This is
    // also where `waiting` first anchors, at index 5.
    let mut rhs = v0;
    xor_assign(&mut rhs, &v5);
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&[VarId::new(0), VarId::new(5)], &[Binary; 2]).unwrap(),
            &rhs,
        )
        .unwrap();
    peeler.drain_recovered_into(&mut recovered);
    assert_eq!(recovered, [VarId::new(5)]);
    assert_eq!(
        peeler.variable_state(VarId::new(5)),
        VariableState::Known(&v5)
    );

    // Names the now-known variable 5 and two unknowns below the anchor, forcing
    // `waiting` to grow at the front.
    let mut rhs = v5;
    xor_assign(&mut rhs, &v1);
    xor_assign(&mut rhs, &v2);
    peeler
        .push_check(
            CheckId::new(1),
            Edges::new(&[VarId::new(5), VarId::new(1), VarId::new(2)], &[Binary; 3]).unwrap(),
            &rhs,
        )
        .unwrap();
    assert!(peeler.has_stalled());

    // Resolving variable 2 must cascade through the front-grown slot to 1.
    peeler
        .push_check(
            CheckId::new(2),
            Edges::new(&[VarId::new(2)], &[Binary; 1]).unwrap(),
            &v2,
        )
        .unwrap();

    recovered.clear();
    peeler.drain_recovered_into(&mut recovered);
    assert_eq!(recovered, [VarId::new(2), VarId::new(1)]);
    assert_eq!(
        peeler.variable_state(VarId::new(0)),
        VariableState::Known(&v0)
    );
    assert_eq!(
        peeler.variable_state(VarId::new(1)),
        VariableState::Known(&v1)
    );
    assert_eq!(
        peeler.variable_state(VarId::new(2)),
        VariableState::Known(&v2)
    );
    assert_eq!(peeler.known_count(), 4);
    assert!(!peeler.has_stalled());
    assert_eq!(peeler.unresolved_count(), 0);
}

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    known: usize,
    checks: usize,
    unresolved: usize,
    variables: Vec<(u64, Option<Vec<u8>>)>,
    rows: Vec<(u64, Vec<u64>, Vec<u8>)>,
}

fn snapshot(peeler: &Peeler<Binary>, through: u64) -> Snapshot {
    let variables = (0..=through)
        .map(|raw| {
            let value = match peeler.variable_state(VarId::new(raw)) {
                VariableState::Known(value) => Some(value.to_vec()),
                VariableState::Retired | VariableState::Unknown => None,
            };
            (raw, value)
        })
        .collect();
    let rows = peeler
        .stalled_rows()
        .map(|row| {
            (
                row.check().get(),
                row.support().iter().map(|var| var.get()).collect(),
                row.rhs().to_vec(),
            )
        })
        .collect();
    Snapshot {
        known: peeler.known_count(),
        checks: peeler.check_count(),
        unresolved: peeler.unresolved_count(),
        variables,
        rows,
    }
}

#[test]
fn malformed_mutations_leave_observable_state_unchanged() {
    let mut peeler = Peeler::<Binary>::new(4, config(8)).unwrap();
    let support = [VarId::new(3), VarId::new(4)];
    let weights = [Binary; 2];
    peeler.learn_copy(VarId::new(2), &[1, 2, 3, 4]).unwrap();
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&support, &weights).unwrap(),
            &[4, 3, 2, 1],
        )
        .unwrap();

    let before = snapshot(&peeler, 8);
    assert_eq!(
        peeler.learn_copy(VarId::new(5), &[1, 2, 3]),
        Err(GraphError::SymbolLengthMismatch {
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(snapshot(&peeler, 8), before);
    assert_eq!(
        peeler.push_check(
            CheckId::new(1),
            Edges::new(&support, &weights).unwrap(),
            &[1, 2, 3],
        ),
        Err(GraphError::SymbolLengthMismatch {
            expected: 4,
            actual: 3
        })
    );
    assert_eq!(snapshot(&peeler, 8), before);
    assert!(matches!(
        Edges::new(&[support[0], support[0]], &weights),
        Err(GraphError::DuplicateVariable { .. })
    ));
    assert_eq!(snapshot(&peeler, 8), before);
    assert!(matches!(
        peeler.learn_copy(VarId::new(20), &[0; 4]),
        Err(GraphError::LiveSpanExceeded { .. })
    ));
    assert_eq!(snapshot(&peeler, 8), before);
    assert!(matches!(
        peeler.push_check(
            CheckId::new(20),
            Edges::new(&support, &weights).unwrap(),
            &[0; 4],
        ),
        Err(GraphError::LiveSpanExceeded { .. })
    ));
    assert_eq!(snapshot(&peeler, 8), before);

    peeler.retire_below(VarId::new(3)).unwrap();
    let retired = snapshot(&peeler, 8);
    assert!(matches!(
        peeler.learn_copy(VarId::new(2), &[0; 4]),
        Err(GraphError::IndexRetired { .. })
    ));
    assert_eq!(snapshot(&peeler, 8), retired);
    assert!(matches!(
        peeler.retire_below(VarId::new(2)),
        Err(GraphError::HorizonRegressed { .. })
    ));
    assert_eq!(snapshot(&peeler, 8), retired);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Aligned4;

impl EdgeWeight for Aligned4 {
    const ELEMENT_BYTES: usize = 4;

    fn one() -> Self {
        Self
    }

    fn is_zero(self) -> bool {
        false
    }

    fn mul_add(dst: &mut [u8], _weight: Self, src: &[u8]) {
        xor_assign(dst, src);
    }

    fn scale_inv(_value: &mut [u8], _weight: Self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ZeroWidth;

impl EdgeWeight for ZeroWidth {
    const ELEMENT_BYTES: usize = 0;

    fn one() -> Self {
        Self
    }

    fn is_zero(self) -> bool {
        false
    }

    fn mul_add(_dst: &mut [u8], _weight: Self, _src: &[u8]) {}

    fn scale_inv(_value: &mut [u8], _weight: Self) {}
}

#[test]
fn construction_rejects_invalid_symbol_geometry() {
    assert!(matches!(
        Peeler::<Binary>::new(0, config(8)),
        Err(GraphError::ZeroSymbolLen)
    ));
    assert!(matches!(
        Peeler::<Aligned4>::new(6, config(8)),
        Err(GraphError::SymbolAlignment {
            length: 6,
            element_bytes: 4
        })
    ));
    assert!(matches!(
        Peeler::<ZeroWidth>::new(4, config(8)),
        Err(GraphError::ZeroElementBytes)
    ));
}

fn weighted_rhs(terms: &[(gf8::Elem, &[u8])]) -> Vec<u8> {
    let mut rhs = vec![0; terms[0].1.len()];
    for &(coefficient, value) in terms {
        for (out, source) in rhs.iter_mut().zip(value) {
            *out ^= coefficient.mul(gf8::Elem(*source)).0;
        }
    }
    rhs
}

fn gf8_weights(values: &[u8]) -> Vec<Weighted<Gf8>> {
    values
        .iter()
        .map(|&value| Weighted::new(gf8::Elem(value)).unwrap())
        .collect()
}

#[test]
fn weighted_construction_rejects_ragged_symbols() {
    assert!(matches!(
        Peeler::<Weighted<Gf16>>::new(3, config(8)),
        Err(GraphError::SymbolAlignment {
            length: 3,
            element_bytes: 2
        })
    ));
}

#[test]
fn weighted_peeling_folds_scales_and_cascades() {
    let values = [
        vec![0x12, 0x34, 0x56, 0x78],
        vec![0x9a, 0xbc, 0xde, 0xf0],
        vec![0x55, 0xaa, 0x11, 0x22],
    ];
    let ids = [VarId::new(0), VarId::new(1), VarId::new(2)];
    let first_weights = gf8_weights(&[3, 5]);
    let second_weights = gf8_weights(&[7, 11]);
    let singleton_weight = gf8_weights(&[13]);
    let first_rhs = weighted_rhs(&[(gf8::Elem(3), &values[0]), (gf8::Elem(5), &values[1])]);
    let second_rhs = weighted_rhs(&[(gf8::Elem(7), &values[1]), (gf8::Elem(11), &values[2])]);
    let singleton_rhs = weighted_rhs(&[(gf8::Elem(13), &values[0])]);

    let mut peeler = Peeler::<Weighted<Gf8>>::new(4, config(16)).unwrap();
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&ids[..2], &first_weights).unwrap(),
            &first_rhs,
        )
        .unwrap();
    peeler
        .push_check(
            CheckId::new(1),
            Edges::new(&ids[1..], &second_weights).unwrap(),
            &second_rhs,
        )
        .unwrap();
    peeler
        .push_check(
            CheckId::new(2),
            Edges::new(&ids[..1], &singleton_weight).unwrap(),
            &singleton_rhs,
        )
        .unwrap();
    for (id, value) in ids.into_iter().zip(&values) {
        assert_eq!(peeler.variable_state(id), VariableState::Known(value));
    }

    let mut reordered = Peeler::<Weighted<Gf8>>::new(4, config(16)).unwrap();
    reordered.learn_copy(ids[1], &values[1]).unwrap();
    reordered
        .push_check(
            CheckId::new(0),
            Edges::new(&ids[..2], &first_weights).unwrap(),
            &first_rhs,
        )
        .unwrap();
    assert_eq!(
        reordered.variable_state(ids[0]),
        VariableState::Known(&values[0])
    );
}

#[test]
fn binary_and_weighted_peeling_share_stopping_sets() {
    let values = [vec![0x21; 16], vec![0x43; 16], vec![0x65; 16]];
    let ids = [VarId::new(0), VarId::new(1), VarId::new(2)];
    let supports = [&ids[..2], &ids[1..]];
    let binary_rhs = [
        weighted_rhs(&[(gf8::Elem::ONE, &values[0]), (gf8::Elem::ONE, &values[1])]),
        weighted_rhs(&[(gf8::Elem::ONE, &values[1]), (gf8::Elem::ONE, &values[2])]),
    ];
    let coefficients = [gf8_weights(&[3, 5]), gf8_weights(&[7, 11])];
    let weighted_rhs = [
        weighted_rhs(&[(gf8::Elem(3), &values[0]), (gf8::Elem(5), &values[1])]),
        weighted_rhs(&[(gf8::Elem(7), &values[1]), (gf8::Elem(11), &values[2])]),
    ];

    let mut binary = Peeler::<Binary>::new(16, config(16)).unwrap();
    let mut weighted = Peeler::<Weighted<Gf8>>::new(16, config(16)).unwrap();
    for check in 0..2 {
        binary
            .push_check(
                CheckId::new(check as u64),
                Edges::new(supports[check], &[Binary; 2]).unwrap(),
                &binary_rhs[check],
            )
            .unwrap();
        weighted
            .push_check(
                CheckId::new(check as u64),
                Edges::new(supports[check], &coefficients[check]).unwrap(),
                &weighted_rhs[check],
            )
            .unwrap();
    }
    assert_eq!(binary.unresolved_count(), weighted.unresolved_count());
    assert_eq!(binary.unresolved_count(), 2);

    binary.learn_copy(ids[1], &values[1]).unwrap();
    weighted.learn_copy(ids[1], &values[1]).unwrap();
    let mut binary_recovered: Vec<_> = binary.drain_recovered().collect();
    let mut weighted_recovered: Vec<_> = weighted.drain_recovered().collect();
    binary_recovered.sort_unstable();
    weighted_recovered.sort_unstable();
    assert_eq!(binary_recovered, weighted_recovered);
    assert_eq!(binary_recovered, vec![ids[0], ids[2]]);
    for (id, value) in ids.into_iter().zip(&values) {
        assert_eq!(binary.variable_state(id), VariableState::Known(value));
        assert_eq!(weighted.variable_state(id), VariableState::Known(value));
    }
}
