//! Scratch-owning reduced-row-echelon residual solver.

use super::builder::{System, checked_geometry, geometry_error};
use crate::{SolveError, VarId};
use alloc::vec::Vec;
use fgf::FieldKernels;
use fgf::field::Elem;
use gfm::{Matrix, Ple, PleScratch, SolveScratch};

/// Rank and exact residual deficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    /// Rank of the coefficient matrix.
    pub rank: usize,
    /// `columns - rank`: additional independent equations required.
    pub deficiency: usize,
}

/// Reusable residual solver backed by [`gfm`]'s rank-revealing decomposition.
///
/// Matrices are sized to the largest row, column, and symbol geometry seen
/// together. Smaller systems occupy a zero-padded prefix, so a same-or-larger
/// warm-up covers every later solve without allocating.
#[derive(Debug)]
pub struct Solver<F: FieldKernels> {
    decomposition: Option<Ple<F>>,
    rhs: Option<Matrix<F>>,
    solution: Option<Matrix<F>>,
    rref: Option<Matrix<F>>,
    ple_scratch: PleScratch<F>,
    solve_scratch: SolveScratch<F>,
    columns: Vec<VarId>,
    determined: Vec<bool>,
    undetermined: Vec<VarId>,
    capacity_rows: usize,
    capacity_columns: usize,
    capacity_symbol_len: usize,
    symbol_len: usize,
}

impl<F: FieldKernels> Default for Solver<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldKernels> Solver<F> {
    /// Create a solver whose matrix and outcome scratch grows and is then reused.
    #[must_use]
    pub fn new() -> Self {
        Self {
            decomposition: None,
            rhs: None,
            solution: None,
            rref: None,
            ple_scratch: PleScratch::new(),
            solve_scratch: SolveScratch::new(),
            columns: Vec::new(),
            determined: Vec::new(),
            undetermined: Vec::new(),
            capacity_rows: 0,
            capacity_columns: 0,
            capacity_symbol_len: 0,
            symbol_len: 0,
        }
    }

    /// Reduce `system` through `gfm` and publish every determined column.
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

        if self.decomposition.is_none()
            || system.rows > self.capacity_rows
            || columns > self.capacity_columns
            || system.symbol_len > self.capacity_symbol_len
        {
            self.capacity_rows = system.rows;
            self.capacity_columns = columns;
            self.capacity_symbol_len = system.symbol_len;
            let symbol_columns = self.capacity_symbol_len / F::BYTES;
            let Some(coefficients) =
                Matrix::<F>::zeros(self.capacity_rows, self.capacity_columns).ok()
            else {
                return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
            };
            let Some(rhs) = Matrix::<F>::zeros(self.capacity_rows, symbol_columns).ok() else {
                return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
            };
            let Some(solution) = Matrix::<F>::zeros(self.capacity_columns, symbol_columns).ok()
            else {
                return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
            };
            let Some(rref) =
                Matrix::<F>::zeros(self.capacity_rows, self.capacity_columns).ok()
            else {
                return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
            };
            self.decomposition = Some(Ple::decompose(coefficients, &mut self.ple_scratch));
            self.rhs = Some(rhs);
            self.solution = Some(solution);
            self.rref = Some(rref);
        }

        let coefficient_stride = columns * F::BYTES;
        let decomposition = self.decomposition.as_mut().ok_or_else(|| {
            geometry_error::<F>(system.rows, columns, system.symbol_len)
        })?;
        decomposition.redecompose_with(&mut self.ple_scratch, |matrix| {
            for row in 0..system.rows {
                matrix.row_mut(row)[..coefficient_stride].copy_from_slice(
                    &system.coefficients[row * coefficient_stride..][..coefficient_stride],
                );
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

        self.columns.extend_from_slice(system.columns);
        self.determined.resize(columns, false);
        self.symbol_len = system.symbol_len;
        let rank = decomposition.rank();
        if rank == columns {
            self.determined.fill(true);
        } else {
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
            for (column, &var) in system.columns.iter().enumerate() {
                if !self.determined[column] {
                    self.undetermined.push(var);
                }
            }
        }
        Ok(Report {
            rank,
            deficiency: columns - rank,
        })
    }

    /// Borrow uniquely determined values until the next solve.
    pub fn recovered(&self) -> impl Iterator<Item = (VarId, &[u8])> {
        let columns = &self.columns;
        let determined = &self.determined;
        let symbol_len = self.symbol_len;
        self.solution.iter().flat_map(move |solution| {
            columns
                .iter()
                .copied()
                .enumerate()
                .filter_map(move |(column, var)| {
                    determined[column]
                        .then_some((var, &solution.row(column)[..symbol_len]))
                })
        })
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
    }
}

