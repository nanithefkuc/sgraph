//! Borrowed equations and progressively reduced consumer-owned rows.

use crate::{EdgeWeight, Peeler, SolveError, VarId, VariableState};
use alloc::vec::Vec;
use core::marker::PhantomData;
use fgf::FieldKernels;
use fgf::field::Elem;

/// One borrowed residual equation over `F`.
#[derive(Debug, Clone, Copy)]
pub struct Row<'a, F: FieldKernels> {
    terms: &'a [(VarId, F::Elem)],
    rhs: &'a [u8],
}

impl<'a, F: FieldKernels> Row<'a, F> {
    /// Borrow a term list and its packed right-hand side.
    ///
    /// Shape and column membership are validated when a row is pushed into a
    /// [`ResidualBuilder`](crate::ResidualBuilder).
    #[must_use]
    pub fn new(terms: &'a [(VarId, F::Elem)], rhs: &'a [u8]) -> Self {
        Self { terms, rhs }
    }

    /// `(variable, coefficient)` terms.
    #[must_use]
    pub fn terms(&self) -> &'a [(VarId, F::Elem)] {
        self.terms
    }

    /// Packed right-hand side.
    #[must_use]
    pub fn rhs(&self) -> &'a [u8] {
        self.rhs
    }
}

/// A consumer-owned dense equation that folds newly-known columns once.
#[derive(Debug, Clone)]
pub struct DenseRow<F: FieldKernels> {
    terms: Vec<(VarId, F::Elem)>,
    rhs: Vec<u8>,
    live: bool,
    field: PhantomData<F>,
}

impl<F: FieldKernels> DenseRow<F> {
    /// Construct a live dense row.
    ///
    /// Duplicate terms are retained here and combined by the residual builder.
    /// Explicit zero terms are discarded because they are not edges.
    ///
    /// # Errors
    ///
    /// Rejects an empty or field-misaligned packed right-hand side.
    pub fn new(mut terms: Vec<(VarId, F::Elem)>, rhs: Vec<u8>) -> Result<Self, SolveError> {
        validate_rhs::<F>(&rhs)?;
        terms.retain(|(_, coefficient)| !coefficient.is_zero());
        Ok(Self {
            terms,
            rhs,
            live: true,
            field: PhantomData,
        })
    }

    /// Whether this row is still admitted by consumer policy.
    #[must_use]
    pub fn is_live(&self) -> bool {
        self.live
    }

    /// Exclude this row from subsequent residual systems.
    pub fn retire(&mut self) {
        self.live = false;
    }

    /// Terms not yet folded out as known.
    #[must_use]
    pub fn terms(&self) -> &[(VarId, F::Elem)] {
        &self.terms
    }

    /// Packed right-hand side after progressive reduction.
    #[must_use]
    pub fn rhs(&self) -> &[u8] {
        &self.rhs
    }

    /// Fold every currently-known term exactly once.
    ///
    /// Folded terms are removed, so calling this again without new knowledge is
    /// a no-op.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::RhsLengthMismatch`] before mutation when this row
    /// and the peeler use different symbol geometry.
    pub fn reduce_known<W: EdgeWeight>(&mut self, peeler: &Peeler<W>) -> Result<(), SolveError> {
        if !self.live {
            return Ok(());
        }
        if self.rhs.len() != peeler.symbol_len() {
            return Err(SolveError::RhsLengthMismatch {
                expected: peeler.symbol_len(),
                actual: self.rhs.len(),
            });
        }
        let mut index = 0;
        while index < self.terms.len() {
            let (var, coefficient) = self.terms[index];
            let VariableState::Known(value) = peeler.variable_state(var) else {
                index += 1;
                continue;
            };
            fgf::ops::mul_add::<F>(&mut self.rhs, coefficient, value);
            self.terms.swap_remove(index);
        }
        Ok(())
    }
}

pub(super) fn validate_rhs<F: FieldKernels>(rhs: &[u8]) -> Result<(), SolveError> {
    if rhs.is_empty() {
        return Err(SolveError::ZeroSymbolLen);
    }
    if !rhs.len().is_multiple_of(F::BYTES) {
        return Err(SolveError::RhsAlignment {
            length: rhs.len(),
            element_bytes: F::BYTES,
        });
    }
    Ok(())
}
