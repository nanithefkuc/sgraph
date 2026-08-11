//! Scratch-owning reduced-row-echelon residual solver.

use super::builder::{System, checked_geometry, geometry_error};
use crate::{SolveError, VarId};
use alloc::vec::Vec;
use core::marker::PhantomData;
use fgf::FieldKernels;
use fgf::field::Elem;

/// Rank and exact residual deficiency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Report {
    /// Rank of the coefficient matrix.
    pub rank: usize,
    /// `columns - rank`: additional independent equations required.
    pub deficiency: usize,
}

/// Reusable RREF solver over one `fgf` field.
///
/// The coefficient matrix is stored **packed**, exactly like the symbol matrix:
/// `columns * F::BYTES` bytes per row. That is what lets a row operation go
/// through `fgf::ops` instead of a scalar per-element loop — at wide geometries
/// the coefficient side is the same order of work as the symbol side, and a
/// scalar loop there costs more than the entire symbol reduction.
#[derive(Debug)]
pub struct Solver<F: FieldKernels> {
    coefficients: Vec<u8>,
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
        let Some((coefficient_bytes, symbol_bytes)) =
            checked_geometry::<F>(system.rows, columns, system.symbol_len)
        else {
            return Err(geometry_error::<F>(system.rows, columns, system.symbol_len));
        };
        debug_assert_eq!(system.coefficients.len(), coefficient_bytes);
        debug_assert_eq!(system.symbols.len(), symbol_bytes);

        self.coefficients.clear();
        self.coefficients.extend_from_slice(system.coefficients);
        self.symbols.clear();
        self.symbols.extend_from_slice(system.symbols);
        self.pivot_of_col.clear();
        self.pivot_of_col.resize(columns, usize::MAX);
        self.symbol_len = system.symbol_len;

        let stride = columns * F::BYTES;
        let rank = self.reduce_to_rref(system.rows, columns);
        for row in 0..system.rows {
            if row_is_zero::<F>(&self.coefficients[row * stride..][..stride])
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
                && elems::<F>(&self.coefficients[pivot * stride..][..stride])
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
        // Every `fgf` field is a binary extension field, so subtraction is
        // addition and one `mul_add` expresses `row -= factor * pivot` exactly.
        // The whole vectorised elimination below rests on that.
        debug_assert!(F::Elem::ONE.sub(F::Elem::ONE).is_zero());
        debug_assert_eq!(
            F::Elem::ONE.sub(F::Elem::ZERO),
            F::Elem::ONE.add(F::Elem::ZERO)
        );

        let stride = columns * F::BYTES;
        // Below one vector's worth of coefficients, two kernel dispatches cost
        // more than the handful of table multiplies they replace. Measured on
        // the `solve/rref` benchmark: at 8 columns the vectorised path is 19%
        // slower, at 32 it is 31% faster and at 64 it is 55% faster.
        let wide = stride >= NARROW_ROW_BYTES;
        let mut pivot_row = 0usize;
        for column in 0..columns {
            let selected = (pivot_row..rows)
                .find(|&row| !cell::<F>(&self.coefficients, stride, row, column).is_zero());
            let Some(selected) = selected else {
                continue;
            };
            if selected != pivot_row {
                swap_rows(&mut self.coefficients, stride, selected, pivot_row);
                swap_rows(&mut self.symbols, self.symbol_len, selected, pivot_row);
            }

            let pivot = cell::<F>(&self.coefficients, stride, pivot_row, column);
            if !pivot.is_one() {
                let inverse = pivot.inv();
                let row = &mut self.coefficients[pivot_row * stride..][..stride];
                if wide {
                    fgf::ops::mul_assign::<F>(row, inverse);
                } else {
                    scale_row::<F>(row, inverse);
                }
                fgf::ops::mul_assign::<F>(
                    &mut self.symbols[pivot_row * self.symbol_len..][..self.symbol_len],
                    inverse,
                );
            }

            let coefficient_split = pivot_row * stride;
            let (coeff_head, coeff_tail) = self.coefficients.split_at_mut(coefficient_split);
            let (pivot_coefficients, coeff_rest) = coeff_tail.split_at_mut(stride);
            let symbol_split = pivot_row * self.symbol_len;
            let (symbol_head, symbol_tail) = self.symbols.split_at_mut(symbol_split);
            let (pivot_symbol, symbol_rest) = symbol_tail.split_at_mut(self.symbol_len);
            for row in 0..rows {
                if row == pivot_row {
                    continue;
                }
                let (coefficient_row, symbol_row) = if row < pivot_row {
                    (
                        &mut coeff_head[row * stride..][..stride],
                        &mut symbol_head[row * self.symbol_len..][..self.symbol_len],
                    )
                } else {
                    let offset = row - pivot_row - 1;
                    (
                        &mut coeff_rest[offset * stride..][..stride],
                        &mut symbol_rest[offset * self.symbol_len..][..self.symbol_len],
                    )
                };
                let factor = F::read(&coefficient_row[column * F::BYTES..][..F::BYTES]);
                if factor.is_zero() {
                    continue;
                }
                if wide {
                    // One prepared coefficient drives both halves of the row
                    // operation, so the backend resolves `factor` once rather
                    // than once per buffer.
                    let factor = fgf::ops::Coeff::<F>::new(factor);
                    fgf::ops::mul_add_with::<F>(coefficient_row, &factor, pivot_coefficients);
                    fgf::ops::mul_add_with::<F>(symbol_row, &factor, pivot_symbol);
                } else {
                    fused_row::<F>(coefficient_row, pivot_coefficients, factor);
                    fgf::ops::mul_add::<F>(symbol_row, factor, pivot_symbol);
                }
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

/// One coefficient of a packed row-major matrix.
fn cell<F: FieldKernels>(matrix: &[u8], stride: usize, row: usize, column: usize) -> F::Elem {
    F::read(&matrix[row * stride + column * F::BYTES..][..F::BYTES])
}

/// Elements of one packed coefficient row.
fn elems<F: FieldKernels>(row: &[u8]) -> impl Iterator<Item = F::Elem> + '_ {
    row.chunks_exact(F::BYTES).map(F::read)
}

/// Whether every coefficient in a packed row is zero.
///
/// This reads elements rather than testing the bytes directly: an all-zero byte
/// encoding of the field's zero is a property of every field `fgf` ships today,
/// but it is not part of the `fgf` field contract, and the cost of honouring
/// the contract here is nil.
fn row_is_zero<F: FieldKernels>(row: &[u8]) -> bool {
    elems::<F>(row).all(Elem::is_zero)
}

/// Coefficient-row width below which a scalar loop beats a kernel dispatch.
const NARROW_ROW_BYTES: usize = 32;

/// `row *= scale`, elementwise over a packed row.
fn scale_row<F: FieldKernels>(row: &mut [u8], scale: F::Elem) {
    for cell in row.chunks_exact_mut(F::BYTES) {
        F::write(cell, F::read(cell).mul(scale));
    }
}

/// `row -= factor * pivot`, elementwise over two packed rows of equal width.
fn fused_row<F: FieldKernels>(row: &mut [u8], pivot: &[u8], factor: F::Elem) {
    debug_assert_eq!(row.len(), pivot.len());
    let pivot = pivot.chunks_exact(F::BYTES);
    for (cell, pivot) in row.chunks_exact_mut(F::BYTES).zip(pivot) {
        F::write(cell, F::read(cell).sub(factor.mul(F::read(pivot))));
    }
}
