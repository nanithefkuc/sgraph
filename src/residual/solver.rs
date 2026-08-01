//! Scratch-owning reduced-row-echelon residual solver.

use super::builder::{System, checked_geometry, geometry_error};
use crate::{SolveError, VarId};
use alloc::vec::Vec;
use core::marker::PhantomData;
use fff::FieldKernels;
use fff::field::Elem;

/// Rank and exact residual deficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    /// Rank of the coefficient matrix.
    pub rank: usize,
    /// `columns - rank`: additional independent equations required.
    pub deficiency: usize,
}

/// Reusable RREF solver over one `fff` field.
#[derive(Debug)]
pub struct Solver<F: FieldKernels> {
    coefficients: Vec<F::Elem>,
    symbols: Vec<u8>,
    pivot_of_col: Vec<usize>,
    recovered: Vec<(VarId, usize)>,
    undetermined: Vec<VarId>,
    symbol_len: usize,
    field: PhantomData<F>,
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
            coefficients: Vec::new(),
            symbols: Vec::new(),
            pivot_of_col: Vec::new(),
            recovered: Vec::new(),
            undetermined: Vec::new(),
            symbol_len: 0,
            field: PhantomData,
        }
    }

    /// Reduce `system` fully to RREF and publish determined columns.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::GeometryOverflow`] for invalid scratch products and
    /// [`SolveError::InconsistentSystem`] for a zero coefficient row with a
    /// non-zero right-hand side. Either error clears published recovery metadata.
    pub fn solve(&mut self, system: &System<'_, F>) -> Result<Report, SolveError> {
        self.clear_outcome();
        let columns = system.columns.len();
        let Some((coefficient_count, symbol_bytes)) =
            checked_geometry::<F>(system.rows, columns, system.symbol_len)
        else {
            return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
        };
        debug_assert_eq!(system.coefficients.len(), coefficient_count);
        debug_assert_eq!(system.symbols.len(), symbol_bytes);

        self.coefficients.clear();
        self.coefficients.extend_from_slice(system.coefficients);
        self.symbols.clear();
        self.symbols.extend_from_slice(system.symbols);
        self.pivot_of_col.clear();
        self.pivot_of_col.resize(columns, usize::MAX);
        self.symbol_len = system.symbol_len;

        let rank = self.reduce_to_rref(system.rows, columns);
        for row in 0..system.rows {
            let coefficients = &self.coefficients[row * columns..][..columns];
            if coefficients.iter().all(|coefficient| coefficient.is_zero())
                && self.symbols[row * self.symbol_len..][..self.symbol_len]
                    .iter()
                    .any(|byte| *byte != 0)
            {
                self.clear_outcome();
                return Err(SolveError::InconsistentSystem);
            }
        }

        for (column, &var) in system.columns.iter().enumerate() {
            let pivot = self.pivot_of_col[column];
            let determined = pivot != usize::MAX
                && self.coefficients[pivot * columns..][..columns]
                    .iter()
                    .filter(|coefficient| !coefficient.is_zero())
                    .count()
                    == 1;
            if determined {
                self.recovered.push((var, pivot));
            } else {
                self.undetermined.push(var);
            }
        }

        Ok(Report {
            rank,
            deficiency: columns - rank,
        })
    }
    fn reduce_to_rref(&mut self, rows: usize, columns: usize) -> usize {
        let mut pivot_row = 0usize;
        for column in 0..columns {
            let selected =
                (pivot_row..rows).find(|&row| !self.coefficients[row * columns + column].is_zero());
            let Some(selected) = selected else {
                continue;
            };
            if selected != pivot_row {
                swap_rows(&mut self.coefficients, columns, selected, pivot_row);
                swap_rows(&mut self.symbols, self.symbol_len, selected, pivot_row);
            }

            let pivot = self.coefficients[pivot_row * columns + column];
            if !pivot.is_one() {
                let inverse = pivot.inv();
                for coefficient in &mut self.coefficients[pivot_row * columns..][..columns] {
                    *coefficient = coefficient.mul(inverse);
                }
                fff::ops::mul_assign::<F>(
                    &mut self.symbols[pivot_row * self.symbol_len..][..self.symbol_len],
                    inverse,
                );
            }

            let coefficient_split = pivot_row * columns;
            let (coeff_head, coeff_tail) = self.coefficients.split_at_mut(coefficient_split);
            let (pivot_coefficients, coeff_rest) = coeff_tail.split_at_mut(columns);
            let symbol_split = pivot_row * self.symbol_len;
            let (symbol_head, symbol_tail) = self.symbols.split_at_mut(symbol_split);
            let (pivot_symbol, symbol_rest) = symbol_tail.split_at_mut(self.symbol_len);
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let (coefficient_row, symbol_row) = if row < pivot_row {
                    (
                        &mut coeff_head[row * columns..][..columns],
                        &mut symbol_head[row * self.symbol_len..][..self.symbol_len],
                    )
                } else {
                    let offset = row - pivot_row - 1;
                    (
                        &mut coeff_rest[offset * columns..][..columns],
                        &mut symbol_rest[offset * self.symbol_len..][..self.symbol_len],
                    )
                };
                let factor = coefficient_row[column];
                if factor.is_zero() {
                    continue;
                }
                for (coefficient, pivot_coefficient) in coefficient_row
                    .iter_mut()
                    .zip(pivot_coefficients.iter().copied())
                {
                    *coefficient = coefficient.sub(factor.mul(pivot_coefficient));
                }
                fff::ops::mul_add::<F>(symbol_row, factor, pivot_symbol);
            }

            self.pivot_of_col[column] = pivot_row;
            pivot_row += 1;
            if pivot_row == rows {
                break;
            }
        }
        pivot_row
    }

    /// Borrow uniquely determined values until the next solve.
    pub fn recovered(&self) -> impl Iterator<Item = (VarId, &[u8])> {
        self.recovered.iter().map(|&(var, row)| {
            (
                var,
                &self.symbols[row * self.symbol_len..][..self.symbol_len],
            )
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
        self.recovered.clear();
        self.undetermined.clear();
        self.symbol_len = 0;
    }
}

fn swap_rows<T>(matrix: &mut [T], stride: usize, first: usize, second: usize) {
    debug_assert_ne!(first, second);
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    let (head, tail) = matrix.split_at_mut(high * stride);
    head[low * stride..][..stride].swap_with_slice(&mut tail[..stride]);
}
