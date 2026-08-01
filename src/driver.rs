//! Scratch-owning fixpoint between sparse peeling and exact residual solving.

use crate::index::IndexSet;
use crate::{
    DenseRow, EdgeWeight, Peeler, Report, ResidualBuilder, ResidualCoeff, RowSink, SolveError,
    Solver, VarId,
};
use alloc::vec::Vec;
use fff::FieldKernels;

/// Consumer seam for progressively reduced, selectively admitted dense rows.
pub trait DenseRows<F: FieldKernels> {
    /// Whether at least one dense row is currently admitted.
    fn has_live_rows(&self) -> bool;

    /// Fold every newly-known variable out of resident rows exactly once.
    ///
    /// # Errors
    ///
    /// Returns a symbol-geometry error before changing a row whose packed RHS is
    /// incompatible with the peeler.
    fn reduce_known<W: EdgeWeight>(&mut self, peeler: &Peeler<W>) -> Result<(), SolveError>;

    /// Push every row selected by consumer admission policy.
    fn push_rows(&self, sink: &mut RowSink<'_, F>);
}

impl<F: FieldKernels> DenseRows<F> for [DenseRow<F>] {
    fn has_live_rows(&self) -> bool {
        self.iter().any(DenseRow::is_live)
    }

    fn reduce_known<W: EdgeWeight>(&mut self, peeler: &Peeler<W>) -> Result<(), SolveError> {
        for row in self.iter_mut().filter(|row| row.is_live()) {
            row.reduce_known(peeler)?;
        }
        Ok(())
    }

    fn push_rows(&self, sink: &mut RowSink<'_, F>) {
        for row in self.iter().filter(|row| row.is_live()) {
            sink.push_dense(row.terms().iter().copied(), row.rhs());
        }
    }
}

impl<F: FieldKernels> DenseRows<F> for Vec<DenseRow<F>> {
    fn has_live_rows(&self) -> bool {
        self.as_slice().has_live_rows()
    }

    fn reduce_known<W: EdgeWeight>(&mut self, peeler: &Peeler<W>) -> Result<(), SolveError> {
        self.as_mut_slice().reduce_known(peeler)
    }

    fn push_rows(&self, sink: &mut RowSink<'_, F>) {
        self.as_slice().push_rows(sink);
    }
}

/// Reusable scratch for the complete peel → solve → re-peel fixpoint.
#[derive(Debug, Default)]
pub struct Resolver {
    columns: Vec<VarId>,
    scratch: Vec<VarId>,
    recovered: Vec<VarId>,
}

impl Resolver {
    /// Create an empty resolver.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain recovered variable ids, retaining resolver allocation.
    pub fn drain_recovered(&mut self) -> impl Iterator<Item = VarId> + '_ {
        self.recovered.drain(..)
    }

    /// Move recovered variable ids into caller-owned scratch.
    pub fn drain_recovered_into(&mut self, out: &mut Vec<VarId>) {
        out.append(&mut self.recovered);
    }

    /// Resolve until neither RREF nor sparse peeling learns another variable.
    ///
    /// `unknowns` is the consumer's complete loss set. Sparse support is never
    /// used to infer columns.
    ///
    /// # Errors
    ///
    /// Returns assembly, elimination, dense-reduction, or graph-update errors.
    /// An inconsistent solve teaches no value to the peeler.
    pub fn resolve<W, F, D>(
        &mut self,
        unknowns: &mut IndexSet,
        peeler: &mut Peeler<W>,
        dense: &mut D,
        solver: &mut Solver<F>,
        builder: &mut ResidualBuilder<F>,
    ) -> Result<Report, SolveError>
    where
        W: EdgeWeight + ResidualCoeff<F>,
        F: FieldKernels,
        D: DenseRows<F> + ?Sized,
    {
        loop {
            self.scratch.clear();
            peeler.drain_recovered_into(&mut self.scratch);
            for var in self.scratch.drain(..) {
                if unknowns.remove(var.get()) {
                    self.recovered.push(var);
                }
            }

            if unknowns.is_empty() {
                solver.clear_outcome();
                return Ok(Report::default());
            }

            self.columns.clear();
            self.columns.extend(unknowns.iter().map(VarId::new));
            dense.reduce_known(peeler)?;

            let system = {
                let mut sink = builder.begin(&self.columns);
                for row in peeler.stalled_rows() {
                    if row.support().iter().all(|var| unknowns.contains(var.get())) {
                        sink.push_sparse(row);
                    }
                }
                if dense.has_live_rows() {
                    dense.push_rows(&mut sink);
                }
                sink.finish()?
            };

            if system.row_count() == 0 {
                return Ok(solver.publish_no_equations(&self.columns));
            }
            let report = solver.solve(&system)?;
            self.scratch.clear();
            self.scratch.extend(solver.recovered().map(|(var, _)| var));
            if self.scratch.is_empty() {
                return Ok(report);
            }

            let first = self.scratch[0];
            let last = self.scratch[self.scratch.len() - 1];
            let graph_error = {
                let mut recovered = solver.recovered();
                recovered.next().and_then(|(_, value)| {
                    peeler.preflight_resolved_copies(first, last, value).err()
                })
            };
            if let Some(error) = graph_error {
                solver.clear_outcome();
                return Err(error.into());
            }
            for (var, value) in solver.recovered() {
                if unknowns.remove(var.get()) {
                    self.recovered.push(var);
                }
                peeler.learn_copy_preflighted(var, value)?;
            }
        }
    }
}
