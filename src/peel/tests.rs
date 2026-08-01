use super::{Peeler, PoolConfig, VariableState};
use crate::{
    Binary, CheckId, Constant, EdgeWeight, Edges, GraphError, NeighborBuf, NeighborGen, VarId,
    WindowedUniform,
};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

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
