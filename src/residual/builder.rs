//! Single-pass residual-system assembly with reusable scratch.

use super::row::{Row, validate_rhs};
use crate::{ResidualCoeff, SolveError, StalledRow, VarId};
use alloc::vec::Vec;
use core::marker::PhantomData;
use fgf::FieldKernels;
use fgf::field::Elem;

/// Reusable single-pass residual-system builder.
///
/// `coefficients` is a **packed** row-major matrix — `columns * F::BYTES` bytes
/// per row, the same encoding as the symbol matrix — so the solver can drive a
/// coefficient row through `fgf::ops` instead of a scalar per-element loop.
/// `term_row` stays element-typed because assembling a row means adding
/// duplicate terms in the field.
#[derive(Debug)]
pub struct ResidualBuilder<F: FieldKernels> {
    columns: Vec<VarId>,
    coefficients: Vec<u8>,
    symbols: Vec<u8>,
    term_row: Vec<F::Elem>,
    rows: usize,
    symbol_len: usize,
    error: Option<SolveError>,
    field: PhantomData<F>,
}

impl<F: FieldKernels> Default for ResidualBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FieldKernels> ResidualBuilder<F> {
    /// Create an empty builder whose scratch grows on demand and is then reused.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            coefficients: Vec::new(),
            symbols: Vec::new(),
            term_row: Vec::new(),
            rows: 0,
            symbol_len: 0,
            error: None,
            field: PhantomData,
        }
    }

    /// Begin one system over sorted, distinct explicit columns.
    ///
    /// Column-order errors are reported by [`RowSink::finish`], allowing the sink
    /// API to remain single-pass.
    pub fn begin<'a>(&'a mut self, unknowns: &[VarId]) -> RowSink<'a, F> {
        self.columns.clear();
        self.columns.extend_from_slice(unknowns);
        self.coefficients.clear();
        self.symbols.clear();
        self.rows = 0;
        self.symbol_len = 0;
        self.error = unknowns.windows(2).find_map(|pair| {
            (pair[0] >= pair[1]).then_some(SolveError::ColumnsNotStrictlyIncreasing {
                previous: pair[0].get(),
                current: pair[1].get(),
            })
        });
        RowSink { builder: self }
    }
}

/// One in-progress, single-pass residual assembly.
#[derive(Debug)]
pub struct RowSink<'a, F: FieldKernels> {
    builder: &'a mut ResidualBuilder<F>,
}

impl<'a, F: FieldKernels> RowSink<'a, F> {
    /// Number of rows admitted so far.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.builder.rows
    }

    /// Push a stalled sparse row, embedding its coefficients into `F`.
    pub fn push_sparse<W: ResidualCoeff<F>>(&mut self, row: StalledRow<'_, W>) {
        let terms = row
            .support()
            .iter()
            .copied()
            .zip(row.weights().iter().copied())
            .map(|(var, weight)| (var, weight.coefficient()));
        self.push_terms(terms, row.rhs());
    }

    /// Push a consumer-owned dense equation.
    pub fn push_dense<I>(&mut self, terms: I, rhs: &[u8])
    where
        I: IntoIterator<Item = (VarId, F::Elem)>,
    {
        self.push_terms(terms, rhs);
    }

    /// Push one borrowed [`Row`].
    pub fn push_row(&mut self, row: Row<'_, F>) {
        self.push_terms(row.terms().iter().copied(), row.rhs());
    }

    /// Finish assembly and borrow the reusable system storage.
    ///
    /// # Errors
    ///
    /// Reports column ordering, unknown terms, right-hand-side geometry, and
    /// checked matrix-product overflow.
    pub fn finish(self) -> Result<System<'a, F>, SolveError> {
        if let Some(error) = self.builder.error.take() {
            return Err(error);
        }
        Ok(System {
            columns: &self.builder.columns,
            coefficients: &self.builder.coefficients,
            symbols: &self.builder.symbols,
            rows: self.builder.rows,
            symbol_len: self.builder.symbol_len,
            field: PhantomData,
        })
    }

    fn push_terms<I>(&mut self, terms: I, rhs: &[u8])
    where
        I: IntoIterator<Item = (VarId, F::Elem)>,
    {
        if self.builder.error.is_some() {
            return;
        }
        if let Err(error) = validate_rhs::<F>(rhs) {
            self.builder.error = Some(error);
            return;
        }
        if self.builder.symbol_len == 0 {
            self.builder.symbol_len = rhs.len();
        } else if rhs.len() != self.builder.symbol_len {
            self.builder.error = Some(SolveError::RhsLengthMismatch {
                expected: self.builder.symbol_len,
                actual: rhs.len(),
            });
            return;
        }

        let columns = self.builder.columns.len();
        self.builder.term_row.clear();
        self.builder.term_row.resize(columns, F::Elem::ZERO);
        for (var, coefficient) in terms {
            let Ok(column) = self.builder.columns.binary_search(&var) else {
                self.builder.error = Some(SolveError::UnknownTerm { var: var.get() });
                self.builder.term_row.clear();
                return;
            };
            self.builder.term_row[column] = self.builder.term_row[column].add(coefficient);
        }

        let Some(rows) = self.builder.rows.checked_add(1) else {
            self.builder.error = Some(geometry_error::<F>(
                usize::MAX,
                columns,
                self.builder.symbol_len,
            ));
            return;
        };
        if checked_geometry::<F>(rows, columns, self.builder.symbol_len).is_none() {
            self.builder.error = Some(geometry_error::<F>(rows, columns, self.builder.symbol_len));
            return;
        }

        let start = self.builder.coefficients.len();
        self.builder
            .coefficients
            .resize(start + columns * F::BYTES, 0);
        fgf::ops::pack::<F>(
            &mut self.builder.coefficients[start..],
            &self.builder.term_row,
        );
        self.builder.symbols.extend_from_slice(rhs);
        self.builder.rows = rows;
    }
}

/// Borrowed, assembled residual system.
#[derive(Debug, Clone, Copy)]
pub struct System<'a, F: FieldKernels> {
    pub(super) columns: &'a [VarId],
    pub(super) coefficients: &'a [u8],
    pub(super) symbols: &'a [u8],
    pub(super) rows: usize,
    pub(super) symbol_len: usize,
    pub(super) field: PhantomData<F>,
}

impl<F: FieldKernels> System<'_, F> {
    /// Explicit unknown columns, sorted and distinct.
    #[must_use]
    pub fn columns(&self) -> &[VarId] {
        self.columns
    }

    /// Number of equations.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows
    }

    /// Packed right-hand-side length, or zero for a system with no rows.
    #[must_use]
    pub fn symbol_len(&self) -> usize {
        self.symbol_len
    }
}

/// Byte lengths of the coefficient and symbol matrices, or `None` on overflow.
pub(super) fn checked_geometry<F: FieldKernels>(
    rows: usize,
    columns: usize,
    symbol_len: usize,
) -> Option<(usize, usize)> {
    let coefficient_bytes = rows.checked_mul(columns)?.checked_mul(F::BYTES)?;
    let symbol_bytes = rows.checked_mul(symbol_len)?;
    Some((coefficient_bytes, symbol_bytes))
}

pub(super) fn geometry_error<F: FieldKernels>(
    rows: usize,
    columns: usize,
    symbol_len: usize,
) -> SolveError {
    SolveError::GeometryOverflow {
        rows,
        columns,
        symbol_len,
        coefficient_bytes: F::BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::checked_geometry;
    use fgf::{Gf8, Gf16};

    #[test]
    fn every_matrix_product_is_checked() {
        assert!(checked_geometry::<Gf8>(usize::MAX, 2, 1).is_none());
        assert!(checked_geometry::<Gf8>(usize::MAX, 1, 2).is_none());
        let coefficient_count_that_overflows_bytes = usize::MAX / 2 + 1;
        assert!(checked_geometry::<Gf16>(coefficient_count_that_overflows_bytes, 1, 1).is_none());
    }
}
