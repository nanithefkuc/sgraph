//! Scratch-owning reduced-row-echelon residual solver.

use super::builder::{System, checked_geometry, geometry_error};
use crate::{SolveError, VarId};
use alloc::vec::Vec;
use fgf::FieldKernels;
use fgf::field::Elem;
use gfm::{Echelon, Innovation, Matrix, Ple, PleScratch, SolveScratch};

/// Rank and exact residual deficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    /// Rank of the coefficient matrix.
    pub rank: usize,
    /// `columns - rank`: additional independent equations required.
    pub deficiency: usize,
}

/// Reusable residual solver backed by `gfm` elimination.
///
/// Square systems use rank-revealing PLE; non-square systems stream rows through
/// reduced echelon so absent or redundant rows never enter a padded dense
/// matrix. Both engines grow together, so a same-or-larger warm-up covers every
/// later solve without allocating.
#[derive(Debug)]
pub struct Solver<F: FieldKernels> {
    decomposition: Option<Ple<F>>,
    rhs: Option<Matrix<F>>,
    solution: Option<Matrix<F>>,
    rref: Option<Matrix<F>>,
    ple_scratch: PleScratch<F>,
    solve_scratch: SolveScratch<F>,
    echelon: Option<Echelon<F>>,
    coefficient_row: Vec<u8>,
    rhs_row: Vec<u8>,
    columns: Vec<VarId>,
    determined: Vec<bool>,
    undetermined: Vec<VarId>,
    capacity_rows: usize,
    capacity_columns: usize,
    capacity_symbol_len: usize,
    symbol_len: usize,
    active_echelon: bool,
}

impl<F: FieldKernels> Default for Solver<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldKernels> Solver<F> {
    /// Create a solver whose elimination and outcome scratch grows and is reused.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decomposition: None,
            rhs: None,
            solution: None,
            rref: None,
            ple_scratch: PleScratch::new(),
            solve_scratch: SolveScratch::new(),
            echelon: None,
            coefficient_row: Vec::new(),
            rhs_row: Vec::new(),
            columns: Vec::new(),
            determined: Vec::new(),
            undetermined: Vec::new(),
            capacity_rows: 0,
            capacity_columns: 0,
            capacity_symbol_len: 0,
            symbol_len: 0,
            active_echelon: false,
        }
    }

    /// Reduce `system` and publish every determined column.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::GeometryOverflow`] for invalid scratch products and
    /// [`SolveError::InconsistentSystem`] when the equations contradict.
    /// Either error clears published recovery metadata.
    pub fn solve(&mut self, system: &System<'_, F>) -> Result<Report, SolveError> {
        self.clear_outcome();
        let columns = system.columns.len();
        let Some((coefficient_bytes, symbol_bytes)) =
            checked_geometry::<F>(system.rows, columns, system.symbol_len)
        else {
            return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
        };
        debug_assert_eq!(system.coefficients.len(), coefficient_bytes);
        debug_assert_eq!(system.symbols.len(), symbol_bytes);
        if columns == 0 {
            return Ok(Report::default());
        }
        self.ensure_capacity(system.rows, columns, system.symbol_len)?;
        self.publish_columns(system);
        self.active_echelon = system.rows != columns;
        let rank = if self.active_echelon {
            self.solve_echelon(system)?
        } else {
            self.solve_ple(system)?
        };
        Ok(self.finish_report(system, rank))
    }

    fn solve_echelon(&mut self, system: &System<'_, F>) -> Result<usize, SolveError> {
        let columns = system.columns.len();
        let padding = self.capacity_columns - columns;
        let echelon = self
            .echelon
            .as_mut()
            .ok_or_else(|| geometry_error::<F>(system.rows, columns, system.symbol_len))?;
        echelon.advance_prefix(self.capacity_columns);
        self.rhs_row.fill(0);
        for column in columns..self.capacity_columns {
            debug_assert!(matches!(
                echelon.absorb_unit(column, &self.rhs_row),
                Innovation::Innovative { .. }
            ));
        }

        let stride = columns * F::BYTES;
        for row in 0..system.rows {
            let coefficients = &system.coefficients[row * stride..][..stride];
            let rhs = &system.symbols[row * system.symbol_len..][..system.symbol_len];
            let innovation = if columns == self.capacity_columns
                && system.symbol_len == self.capacity_symbol_len
            {
                echelon.absorb(coefficients, rhs)
            } else {
                self.coefficient_row.fill(0);
                self.coefficient_row[..stride].copy_from_slice(coefficients);
                self.rhs_row.fill(0);
                self.rhs_row[..system.symbol_len].copy_from_slice(rhs);
                echelon.absorb(&self.coefficient_row, &self.rhs_row)
            };
            if matches!(innovation, Innovation::Inconsistent) {
                self.clear_outcome();
                return Err(SolveError::InconsistentSystem);
            }
        }
        for (column, _) in echelon.recovered() {
            if column < columns {
                self.determined[column] = true;
            }
        }
        Ok(echelon.rank() - padding)
    }

    fn solve_ple(&mut self, system: &System<'_, F>) -> Result<usize, SolveError> {
        let columns = system.columns.len();
        let stride = columns * F::BYTES;
        let decomposition = self
            .decomposition
            .as_mut()
            .ok_or_else(|| geometry_error::<F>(system.rows, columns, system.symbol_len))?;
        decomposition.redecompose_with(&mut self.ple_scratch, |matrix| {
            for row in 0..system.rows {
                matrix.row_mut(row)[..stride]
                    .copy_from_slice(&system.coefficients[row * stride..][..stride]);
            }
        });
        let rhs = self
            .rhs
            .as_mut()
            .ok_or_else(|| geometry_error::<F>(system.rows, columns, system.symbol_len))?;
        for row in 0..self.capacity_rows {
            rhs.row_mut(row).fill(0);
        }
        for row in 0..system.rows {
            rhs.row_mut(row)[..system.symbol_len]
                .copy_from_slice(&system.symbols[row * system.symbol_len..][..system.symbol_len]);
        }
        let solution = self
            .solution
            .as_mut()
            .ok_or_else(|| geometry_error::<F>(system.rows, columns, system.symbol_len))?;
        if decomposition
            .solve_into(rhs, solution, &mut self.solve_scratch)
            .is_err()
        {
            self.clear_outcome();
            return Err(SolveError::InconsistentSystem);
        }

        let rank = decomposition.rank();
        if rank == columns {
            self.determined.fill(true);
            return Ok(rank);
        }
        let rref = self
            .rref
            .as_mut()
            .ok_or_else(|| geometry_error::<F>(system.rows, columns, system.symbol_len))?;
        decomposition.rref_into(rref);
        for row in 0..self.capacity_rows {
            let mut only = None;
            for column in 0..columns {
                if rref.get(row, column).is_zero() {
                    continue;
                }
                if only.is_some() {
                    only = None;
                    break;
                }
                only = Some(column);
            }
            if let Some(column) = only {
                self.determined[column] = true;
            }
        }
        Ok(rank)
    }

    fn publish_columns(&mut self, system: &System<'_, F>) {
        self.columns.extend_from_slice(system.columns);
        self.determined.resize(system.columns.len(), false);
        self.symbol_len = system.symbol_len;
    }

    fn finish_report(&mut self, system: &System<'_, F>, rank: usize) -> Report {
        for (column, &var) in system.columns.iter().enumerate() {
            if !self.determined[column] {
                self.undetermined.push(var);
            }
        }
        Report {
            rank,
            deficiency: system.columns.len() - rank,
        }
    }

    fn ensure_capacity(
        &mut self,
        rows: usize,
        columns: usize,
        symbol_len: usize,
    ) -> Result<(), SolveError> {
        if self.decomposition.is_some()
            && rows <= self.capacity_rows
            && columns <= self.capacity_columns
            && symbol_len <= self.capacity_symbol_len
        {
            return Ok(());
        }
        self.capacity_rows = self.capacity_rows.max(rows);
        self.capacity_columns = self.capacity_columns.max(columns);
        self.capacity_symbol_len = self.capacity_symbol_len.max(symbol_len);
        let rhs_columns = self.capacity_symbol_len / F::BYTES;
        let error = || geometry_error::<F>(rows, columns, symbol_len);
        let coefficients =
            Matrix::<F>::zeros(self.capacity_rows, self.capacity_columns).map_err(|_| error())?;
        self.rhs = Some(Matrix::<F>::zeros(self.capacity_rows, rhs_columns).map_err(|_| error())?);
        self.solution =
            Some(Matrix::<F>::zeros(self.capacity_columns, rhs_columns).map_err(|_| error())?);
        self.rref = Some(
            Matrix::<F>::zeros(self.capacity_rows, self.capacity_columns).map_err(|_| error())?,
        );
        self.decomposition = Some(Ple::decompose(coefficients, &mut self.ple_scratch));
        self.echelon = Some(
            Echelon::new(self.capacity_columns, self.capacity_symbol_len, true)
                .map_err(|_| error())?,
        );
        self.coefficient_row
            .resize(self.capacity_columns * F::BYTES, 0);
        self.rhs_row.resize(self.capacity_symbol_len, 0);
        Ok(())
    }

    /// Borrow uniquely determined values until the next solve.
    pub fn recovered(&self) -> impl Iterator<Item = (VarId, &[u8])> {
        let columns = &self.columns;
        let determined = &self.determined;
        let symbol_len = self.symbol_len;
        let ple = self
            .solution
            .iter()
            .filter(move |_| !self.active_echelon)
            .flat_map(move |solution| {
                columns
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(move |(column, var)| {
                        determined[column].then_some((var, &solution.row(column)[..symbol_len]))
                    })
            });
        let echelon = self
            .echelon
            .iter()
            .filter(move |_| self.active_echelon)
            .flat_map(move |echelon| {
                echelon.recovered().filter_map(move |(column, value)| {
                    columns
                        .get(column)
                        .copied()
                        .map(|var| (var, &value[..symbol_len]))
                })
            });
        ple.chain(echelon)
    }

    /// Columns that are not uniquely determined.
    #[must_use]
    pub fn undetermined(&self) -> &[VarId] {
        &self.undetermined
    }

    pub(crate) fn publish_no_equations(&mut self, columns: &[VarId]) -> Report {
        self.clear_outcome();
        self.undetermined.extend_from_slice(columns);
        Report {
            rank: 0,
            deficiency: columns.len(),
        }
    }

    pub(crate) fn clear_outcome(&mut self) {
        self.columns.clear();
        self.determined.clear();
        self.undetermined.clear();
        self.symbol_len = 0;
        self.active_echelon = false;
    }
}
