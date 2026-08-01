//! Residual check rows and their borrowed public view.

use crate::{CheckId, EdgeWeight, VarId};
use alloc::vec::Vec;

/// One resident check after every known variable has been folded out.
#[derive(Debug)]
pub(super) struct CheckRow<W> {
    pub(super) rhs: Vec<u8>,
    pub(super) support: Vec<VarId>,
    pub(super) weights: Vec<W>,
    pub(super) min_var: Option<VarId>,
    pub(super) resolved: bool,
}

impl<W> CheckRow<W> {
    pub(super) fn is_unresolved(&self) -> bool {
        !self.resolved && !self.support.is_empty()
    }

    pub(super) fn refresh_min_after_removing(&mut self, removed: VarId) {
        if self.min_var == Some(removed) {
            self.min_var = self.support.iter().copied().min();
        }
    }
}

/// One slot in the check-id ring.
#[derive(Debug, Default)]
pub(super) enum RowSlot<W> {
    #[default]
    Vacant,
    Retired,
    Live(CheckRow<W>),
}

/// A borrowed unresolved check equation.
///
/// `rhs` is already reduced against every known variable, and `weights` is
/// parallel to `support`.
#[derive(Debug, Clone, Copy)]
pub struct StalledRow<'a, W> {
    pub(super) check: CheckId,
    pub(super) support: &'a [VarId],
    pub(super) weights: &'a [W],
    pub(super) rhs: &'a [u8],
}

impl<'a, W: EdgeWeight> StalledRow<'a, W> {
    /// Check node that owns this equation.
    #[must_use]
    pub fn check(&self) -> CheckId {
        self.check
    }

    /// Variables still unknown in this equation.
    #[must_use]
    pub fn support(&self) -> &'a [VarId] {
        self.support
    }

    /// Coefficients parallel to [`support`](Self::support).
    #[must_use]
    pub fn weights(&self) -> &'a [W] {
        self.weights
    }

    /// Right-hand side after all known variables have been folded out.
    #[must_use]
    pub fn rhs(&self) -> &'a [u8] {
        self.rhs
    }
}
