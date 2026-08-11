use super::{DenseRow, ResidualBuilder, Row, Solver};
use crate::index::IndexSet;
use crate::{
    Binary, CheckId, DenseRows, Edges, GraphError, Peeler, PoolConfig, Resolver, RowSink,
    SolveError, VarId, VariableState,
};
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroUsize;
use fgf::{Gf8, Gf16, gf8};

fn config() -> PoolConfig {
    let span = NonZeroUsize::new(128).unwrap();
    PoolConfig::new(span, span).with_pool_capacity(32)
}

fn unknowns(ids: &[u64]) -> IndexSet {
    let mut set = IndexSet::new(NonZeroUsize::new(128).unwrap());
    for &id in ids {
        set.insert(id).unwrap();
    }
    set
}

fn rhs(terms: &[(gf8::Elem, &[u8])], len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    for &(coefficient, value) in terms {
        fgf::ops::mul_add::<Gf8>(&mut out, coefficient, value);
    }
    out
}

fn solve_rows(
    solver: &mut Solver<Gf8>,
    builder: &mut ResidualBuilder<Gf8>,
    columns: &[VarId],
    rows: &[Row<'_, Gf8>],
) -> Result<crate::Report, SolveError> {
    let system = {
        let mut sink = builder.begin(columns);
        for &row in rows {
            sink.push_row(row);
        }
        sink.finish()?
    };
    solver.solve(&system)
}

#[test]
fn resolver_keeps_explicit_columns_without_equations() {
    let mut unknowns = unknowns(&[2, 9, 17]);
    let mut peeler = Peeler::<Binary>::new(1, config()).unwrap();
    let mut dense: Vec<DenseRow<Gf8>> = Vec::new();
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let report = Resolver::new()
        .resolve(
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
        )
        .unwrap();

    assert_eq!(report.rank, 0);
    assert_eq!(report.deficiency, 3);
    assert_eq!(
        solver.undetermined(),
        &[VarId::new(2), VarId::new(9), VarId::new(17)]
    );
    assert_eq!(solver.recovered().count(), 0);
    assert_eq!(unknowns.len(), 3);
}

#[test]
fn zero_sparse_rows_track_independent_dense_rank_exactly() {
    let columns: Vec<_> = (0u64..7).map(VarId::new).collect();
    let values: Vec<_> = (0u8..7).map(|i| [0x20 + i; 2]).collect();
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();

    for independent_rows in 0..=columns.len() {
        let mut terms = Vec::with_capacity(independent_rows);
        for column in columns.iter().take(independent_rows).copied() {
            terms.push([(column, gf8::Elem::ONE)]);
        }
        let row_views: Vec<_> = terms
            .iter()
            .enumerate()
            .map(|(index, term)| Row::new(term, &values[index]))
            .collect();
        let report = solve_rows(&mut solver, &mut builder, &columns, &row_views).unwrap();
        assert_eq!(report.rank, independent_rows);
        assert_eq!(report.deficiency, columns.len() - independent_rows);
    }
}

#[test]
fn rank_deficiency_and_sparse_dense_additivity_are_exact() {
    let columns = [VarId::new(0), VarId::new(1), VarId::new(2)];
    let x0 = [0x19; 3];
    let x1 = [0x2A; 3];
    let x2 = [0xC7; 3];
    let first_terms = [(columns[0], gf8::Elem::ONE), (columns[2], gf8::Elem::ONE)];
    let second_terms = [(columns[1], gf8::Elem::ONE), (columns[2], gf8::Elem::ONE)];
    let first_rhs = rhs(&[(gf8::Elem::ONE, &x0), (gf8::Elem::ONE, &x2)], 3);
    let second_rhs = rhs(&[(gf8::Elem::ONE, &x1), (gf8::Elem::ONE, &x2)], 3);
    let base_rows = [
        Row::new(&first_terms, &first_rhs),
        Row::new(&second_terms, &second_rhs),
    ];
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let base = solve_rows(&mut solver, &mut builder, &columns, &base_rows).unwrap();
    assert_eq!(base.rank, 2);
    assert_eq!(base.deficiency, 1);
    assert!(solver.recovered().next().is_none());
    assert_eq!(solver.undetermined(), columns);

    let third_terms = [(columns[2], gf8::Elem::ONE)];
    let full_rows = [base_rows[0], base_rows[1], Row::new(&third_terms, &x2)];
    let full = solve_rows(&mut solver, &mut builder, &columns, &full_rows).unwrap();
    assert_eq!(full.rank, 3);
    assert_eq!(full.deficiency, 0);
    let recovered: Vec<_> = solver
        .recovered()
        .map(|(var, value)| (var, value.to_vec()))
        .collect();
    assert_eq!(
        recovered,
        vec![
            (columns[0], x0.to_vec()),
            (columns[1], x1.to_vec()),
            (columns[2], x2.to_vec()),
        ]
    );
}

#[derive(Debug)]
struct UnitRows {
    rows: Vec<DenseRow<Gf8>>,
}

impl DenseRows<Gf8> for UnitRows {
    fn has_live_rows(&self) -> bool {
        self.rows
            .iter()
            .any(|row| row.is_live() && row.terms().len() <= 1)
    }

    fn reduce_known<W: crate::EdgeWeight>(&mut self, peeler: &Peeler<W>) -> Result<(), SolveError> {
        for row in &mut self.rows {
            row.reduce_known(peeler)?;
        }
        Ok(())
    }

    fn push_rows(&self, sink: &mut RowSink<'_, Gf8>) {
        for row in self
            .rows
            .iter()
            .filter(|row| row.is_live() && row.terms().len() <= 1)
        {
            sink.push_dense(row.terms().iter().copied(), row.rhs());
        }
    }
}

#[test]
fn resolver_reaches_dense_sparse_dense_fixpoint() {
    let ids = [VarId::new(0), VarId::new(1), VarId::new(2)];
    let values = [[0x13; 4], [0x57; 4], [0xB9; 4]];
    let sparse_rhs = rhs(
        &[(gf8::Elem::ONE, &values[0]), (gf8::Elem::ONE, &values[1])],
        4,
    );
    let mut peeler = Peeler::<Binary>::new(4, config()).unwrap();
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&ids[..2], &[Binary; 2]).unwrap(),
            &sparse_rhs,
        )
        .unwrap();

    let first = DenseRow::new(vec![(ids[0], gf8::Elem::ONE)], values[0].to_vec()).unwrap();
    let second_rhs = rhs(
        &[(gf8::Elem::ONE, &values[1]), (gf8::Elem::ONE, &values[2])],
        4,
    );
    let second = DenseRow::new(
        vec![(ids[1], gf8::Elem::ONE), (ids[2], gf8::Elem::ONE)],
        second_rhs,
    )
    .unwrap();
    let mut dense = UnitRows {
        rows: vec![first, second],
    };
    let mut unknowns = unknowns(&[0, 1, 2]);
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let report = Resolver::new()
        .resolve(
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
        )
        .unwrap();

    assert_eq!(report.deficiency, 0);
    assert!(unknowns.is_empty());
    for (id, value) in ids.into_iter().zip(values) {
        assert_eq!(peeler.variable_state(id), VariableState::Known(&value));
    }
    assert!(dense.rows[0].terms().is_empty());
    assert_eq!(dense.rows[1].terms(), &[(ids[2], gf8::Elem::ONE)]);
}

#[test]
fn resolver_reports_every_recovery_exactly_once() {
    let ids = [VarId::new(0), VarId::new(1), VarId::new(2)];
    let values = [[0x13; 4], [0x57; 4], [0xB9; 4]];
    let sparse_rhs = rhs(
        &[(gf8::Elem::ONE, &values[0]), (gf8::Elem::ONE, &values[1])],
        4,
    );
    let mut peeler = Peeler::<Binary>::new(4, config()).unwrap();
    peeler
        .push_check(
            CheckId::new(0),
            Edges::new(&ids[..2], &[Binary; 2]).unwrap(),
            &sparse_rhs,
        )
        .unwrap();

    let second_rhs = rhs(
        &[(gf8::Elem::ONE, &values[1]), (gf8::Elem::ONE, &values[2])],
        4,
    );
    let mut dense = UnitRows {
        rows: vec![
            DenseRow::new(vec![(ids[0], gf8::Elem::ONE)], values[0].to_vec()).unwrap(),
            DenseRow::new(
                vec![(ids[1], gf8::Elem::ONE), (ids[2], gf8::Elem::ONE)],
                second_rhs,
            )
            .unwrap(),
        ],
    };
    let mut unknowns = unknowns(&[0, 1, 2]);
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let mut resolver = Resolver::new();
    resolver
        .resolve(
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
        )
        .unwrap();

    // Both the RREF recoveries and the sparse cascade they unlock are reported,
    // each exactly once, and a second drain yields nothing.
    let mut reported: Vec<VarId> = Vec::new();
    resolver.drain_recovered_into(&mut reported);
    reported.sort_unstable();
    assert_eq!(reported, ids);
    assert_eq!(resolver.drain_recovered().count(), 0);
}

#[test]
fn dense_row_folds_each_known_term_once() {
    let ids = [VarId::new(0), VarId::new(1)];
    let values = [[0x31; 3], [0xC2; 3]];
    let coefficients = [gf8::Elem(7), gf8::Elem(19)];
    let original = [0xA5; 3];
    let mut row = DenseRow::<Gf8>::new(
        vec![(ids[0], coefficients[0]), (ids[1], coefficients[1])],
        original.to_vec(),
    )
    .unwrap();
    let mut peeler = Peeler::<Binary>::new(3, config()).unwrap();
    peeler.learn_copy(ids[0], &values[0]).unwrap();

    row.reduce_known(&peeler).unwrap();
    let after_first = row.rhs().to_vec();
    let terms_after_first = row.terms().to_vec();
    row.reduce_known(&peeler).unwrap();
    assert_eq!(row.rhs(), after_first);
    assert_eq!(row.terms(), terms_after_first);

    let mut expected = after_first;
    fgf::ops::mul_add::<Gf8>(&mut expected, coefficients[1], &values[1]);
    peeler.learn_copy(ids[1], &values[1]).unwrap();
    row.reduce_known(&peeler).unwrap();
    assert_eq!(row.rhs(), expected);
    assert!(row.terms().is_empty());
    let final_rhs = row.rhs().to_vec();
    row.reduce_known(&peeler).unwrap();
    assert_eq!(row.rhs(), final_rhs);
}

#[test]
fn single_pass_admission_keeps_terms_paired_with_rhs() {
    let columns = [VarId::new(0), VarId::new(1)];
    let values = [[0x44; 2], [0x99; 2]];
    let accepted_terms = [
        [(columns[0], gf8::Elem::ONE)],
        [(columns[1], gf8::Elem::ONE)],
    ];
    let rejected_terms = [[(columns[0], gf8::Elem::ONE)]; 2];
    let rejected_rhs = [[0xEE; 2], [0xDD; 2]];
    let mut builder = ResidualBuilder::<Gf8>::new();
    let system = {
        let mut sink = builder.begin(&columns);
        for index in 0..4 {
            if index % 2 == 0 {
                let accepted = index / 2;
                sink.push_dense(accepted_terms[accepted], &values[accepted]);
            } else {
                let rejected = index / 2;
                let _ = Row::<Gf8>::new(&rejected_terms[rejected], &rejected_rhs[rejected]);
            }
        }
        sink.finish().unwrap()
    };
    let mut solver = Solver::new();
    let report = solver.solve(&system).unwrap();
    assert_eq!(report.rank, 2);
    let recovered: Vec<_> = solver
        .recovered()
        .map(|(_, value)| value.to_vec())
        .collect();
    assert_eq!(recovered, vec![values[0].to_vec(), values[1].to_vec()]);
}

#[test]
fn full_rref_recovers_a_pivot_coupled_in_echelon_form() {
    let columns = [VarId::new(0), VarId::new(1)];
    let x0 = [0xA6; 2];
    let x1 = [0x3C; 2];
    let coupled_terms = [(columns[0], gf8::Elem::ONE), (columns[1], gf8::Elem::ONE)];
    let unit_terms = [(columns[1], gf8::Elem::ONE)];
    let coupled_rhs = rhs(&[(gf8::Elem::ONE, &x0), (gf8::Elem::ONE, &x1)], 2);
    let rows = [
        Row::new(&coupled_terms, &coupled_rhs),
        Row::new(&unit_terms, &x1),
    ];
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let report = solve_rows(&mut solver, &mut builder, &columns, &rows).unwrap();
    assert_eq!(report.rank, 2);
    assert_eq!(report.deficiency, 0);
    let recovered: Vec<_> = solver
        .recovered()
        .map(|(_, value)| value.to_vec())
        .collect();
    assert_eq!(recovered, vec![x0.to_vec(), x1.to_vec()]);
}

#[test]
fn duplicate_terms_combine_and_input_geometry_is_exact() {
    let column = [VarId::new(7)];
    let value = [0x5D; 2];
    let first = gf8::Elem(3);
    let second = gf8::Elem(5);
    let combined = first.add(second);
    let terms = [(column[0], first), (column[0], second)];
    let expected_rhs = rhs(&[(combined, &value)], 2);
    let rows = [Row::new(&terms, &expected_rhs)];
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let report = solve_rows(&mut solver, &mut builder, &column, &rows).unwrap();
    assert_eq!(report.rank, 1);
    assert_eq!(solver.recovered().next().unwrap().1, value);

    let bad_columns = [VarId::new(2), VarId::new(1)];
    let error = builder.begin(&bad_columns).finish().unwrap_err();
    assert_eq!(
        error,
        SolveError::ColumnsNotStrictlyIncreasing {
            previous: 2,
            current: 1
        }
    );
    let duplicate_columns = [VarId::new(1), VarId::new(1)];
    assert!(matches!(
        builder.begin(&duplicate_columns).finish(),
        Err(SolveError::ColumnsNotStrictlyIncreasing { .. })
    ));

    let foreign_terms = [(VarId::new(8), gf8::Elem::ONE)];
    let error = {
        let mut sink = builder.begin(&column);
        sink.push_dense(foreign_terms, &value);
        sink.finish().unwrap_err()
    };
    assert_eq!(error, SolveError::UnknownTerm { var: 8 });

    let error = {
        let mut sink = builder.begin(&column);
        sink.push_dense([(column[0], gf8::Elem::ONE)], &[1, 2]);
        sink.push_dense([(column[0], gf8::Elem::ONE)], &[1]);
        sink.finish().unwrap_err()
    };
    assert_eq!(
        error,
        SolveError::RhsLengthMismatch {
            expected: 2,
            actual: 1
        }
    );

    let mut wide_builder = ResidualBuilder::<Gf16>::new();
    let error = {
        let mut sink = wide_builder.begin(&column);
        sink.push_dense([(column[0], fgf::gf16::Elem::ONE)], &[1]);
        sink.finish().unwrap_err()
    };
    assert_eq!(
        error,
        SolveError::RhsAlignment {
            length: 1,
            element_bytes: 2
        }
    );
}

#[test]
fn inconsistent_system_publishes_and_teaches_nothing() {
    let id = VarId::new(0);
    let mut unknowns = unknowns(&[0]);
    let mut peeler = Peeler::<Binary>::new(2, config()).unwrap();
    let mut dense = vec![
        DenseRow::<Gf8>::new(vec![(id, gf8::Elem::ONE)], vec![0x11; 2]).unwrap(),
        DenseRow::<Gf8>::new(vec![(id, gf8::Elem::ONE)], vec![0x22; 2]).unwrap(),
    ];
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let error = Resolver::new()
        .resolve(
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
        )
        .unwrap_err();

    assert_eq!(error, SolveError::InconsistentSystem);
    assert_eq!(solver.recovered().count(), 0);
    assert!(solver.undetermined().is_empty());
    assert_eq!(peeler.variable_state(id), VariableState::Unknown);
    assert!(unknowns.contains(id.get()));
}

#[test]
fn resolver_preflights_all_recoveries_before_teaching_any() {
    let ids = [VarId::new(0), VarId::new(10)];
    let span = NonZeroUsize::new(8).unwrap();
    let mut peeler = Peeler::<Binary>::new(1, PoolConfig::new(span, span)).unwrap();
    let mut unknowns = unknowns(&[0, 10]);
    let mut dense = vec![
        DenseRow::<Gf8>::new(vec![(ids[0], gf8::Elem::ONE)], vec![0x11]).unwrap(),
        DenseRow::<Gf8>::new(vec![(ids[1], gf8::Elem::ONE)], vec![0x22]).unwrap(),
    ];
    let mut solver = Solver::new();
    let mut builder = ResidualBuilder::new();
    let error = Resolver::new()
        .resolve(
            &mut unknowns,
            &mut peeler,
            &mut dense,
            &mut solver,
            &mut builder,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        SolveError::Graph(GraphError::LiveSpanExceeded { .. })
    ));
    assert_eq!(peeler.known_count(), 0);
    assert_eq!(peeler.variable_state(ids[0]), VariableState::Unknown);
    assert_eq!(peeler.variable_state(ids[1]), VariableState::Unknown);
    assert_eq!(unknowns.len(), 2);
    assert_eq!(solver.recovered().count(), 0);
}
