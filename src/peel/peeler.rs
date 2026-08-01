//! Peeling engine over a shrinking residual graph.

use super::graph::{CheckRow, RowSlot, StalledRow};
use super::pool::{Pool, PoolConfig};
use crate::error::GraphError;
use crate::index::{Lookup, Ring};
use crate::neighbors::{Edges, NeighborBuf, NeighborGen};
use crate::{CheckId, EdgeWeight, VarId};
use alloc::vec::Vec;
use core::mem;

/// State of one variable at the peeler boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableState<'a> {
    /// The variable is below the explicit retirement horizon and cannot return.
    Retired,
    /// The variable is live but no value is known.
    Unknown,
    /// The variable's value is known, either by arrival or peeling.
    Known(&'a [u8]),
}

/// A residual sparse graph with iterative degree-one peeling.
#[derive(Debug)]
pub struct Peeler<W: EdgeWeight> {
    symbol_len: usize,
    known: Ring<Option<Vec<u8>>>,
    known_count: usize,
    rows: Ring<RowSlot<W>>,
    row_count: usize,
    /// Conservative **lower** bound on `min_var` across every live row.
    ///
    /// The invariant is one-sided on purpose: the mark is never above the true
    /// minimum, so a stale-low mark only costs a wasted scan in
    /// [`retire_below`](Peeler::retire_below) and can never hide a row that
    /// must be dropped. `u64::MAX` means no live row has residual support.
    row_min_low_water: u64,
    waiting: Ring<Vec<CheckId>>,
    ripple: Vec<CheckId>,
    recovered: Vec<VarId>,
    unresolved: usize,
    neighbor_buf: NeighborBuf<W>,
    pool: Pool<W>,
}

impl<W: EdgeWeight> Peeler<W> {
    /// Create an empty peeler with bounded dense index spans.
    ///
    /// # Errors
    ///
    /// * [`GraphError::ZeroSymbolLen`] when `symbol_len == 0`.
    /// * [`GraphError::ZeroElementBytes`] when `W` violates the edge-weight
    ///   contract.
    /// * [`GraphError::SymbolAlignment`] when the symbol ends with a partial
    ///   packed field element.
    pub fn new(symbol_len: usize, config: PoolConfig) -> Result<Self, GraphError> {
        if symbol_len == 0 {
            return Err(GraphError::ZeroSymbolLen);
        }
        if W::ELEMENT_BYTES == 0 {
            return Err(GraphError::ZeroElementBytes);
        }
        if !symbol_len.is_multiple_of(W::ELEMENT_BYTES) {
            return Err(GraphError::SymbolAlignment {
                length: symbol_len,
                element_bytes: W::ELEMENT_BYTES,
            });
        }
        let pool_capacity = config.resolved_pool_capacity(symbol_len);
        Ok(Self {
            symbol_len,
            known: Ring::new(config.max_variable_span()),
            known_count: 0,
            rows: Ring::new(config.max_check_span()),
            row_count: 0,
            row_min_low_water: u64::MAX,
            waiting: Ring::new(config.max_variable_span()),
            ripple: Vec::new(),
            recovered: Vec::new(),
            unresolved: 0,
            neighbor_buf: NeighborBuf::new(),
            pool: Pool::new(pool_capacity),
        })
    }

    /// Configured symbol length in bytes.
    #[must_use]
    pub fn symbol_len(&self) -> usize {
        self.symbol_len
    }

    /// Number of resident known variables.
    #[must_use]
    pub fn known_count(&self) -> usize {
        self.known_count
    }

    /// Number of retained check rows, resolved or unresolved.
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.row_count
    }

    /// Number of unresolved rows with non-empty support.
    #[must_use]
    pub fn unresolved_count(&self) -> usize {
        self.unresolved
    }

    /// State of `var`, preserving the retired/unknown distinction.
    #[must_use]
    pub fn variable_state(&self, var: VarId) -> VariableState<'_> {
        match self.known.get(var.get()) {
            Lookup::Retired => VariableState::Retired,
            Lookup::Vacant | Lookup::Live(None) => VariableState::Unknown,
            Lookup::Live(Some(value)) => VariableState::Known(value),
        }
    }

    /// Whether peeling is stalled on at least one unresolved check.
    #[must_use]
    pub fn has_stalled(&self) -> bool {
        self.unresolved != 0
    }

    /// Iterate every unresolved residual row in check-id order.
    pub fn stalled_rows(&self) -> impl Iterator<Item = StalledRow<'_, W>> {
        self.rows.iter().filter_map(|(raw, slot)| {
            let RowSlot::Live(row) = slot else {
                return None;
            };
            row.is_unresolved().then_some(StalledRow {
                check: CheckId::new(raw),
                support: &row.support,
                weights: &row.weights,
                rhs: &row.rhs,
            })
        })
    }

    /// Drain variables recovered by peeling, retaining peeler allocation.
    pub fn drain_recovered(&mut self) -> impl Iterator<Item = VarId> + '_ {
        self.recovered.drain(..)
    }

    /// Move variables recovered by peeling since the last drain into `out`.
    ///
    /// The peeler retains its allocation for reuse.
    pub fn drain_recovered_into(&mut self, out: &mut Vec<VarId>) {
        out.append(&mut self.recovered);
    }

    /// Take one recycled symbol buffer, if retirement has supplied one.
    pub fn take_recycled(&mut self) -> Option<Vec<u8>> {
        self.pool.take_symbol()
    }

    /// Learn an owned variable value without copying it.
    ///
    /// Existing values win: a duplicate is ignored after validation and its
    /// owned buffer enters the recycle pool.
    ///
    /// # Errors
    ///
    /// Returns a geometry, retirement, or live-span [`GraphError`] before graph
    /// state is changed.
    pub fn learn(&mut self, var: VarId, value: Vec<u8>) -> Result<(), GraphError> {
        self.validate_symbol(&value)?;
        if self.ensure_variable_slot(var)? {
            self.pool.recycle_symbol(value);
            return Ok(());
        }
        self.apply_known(var, value);
        self.drive_peel();
        Ok(())
    }

    /// Learn a borrowed variable value, copying into recycled storage.
    ///
    /// # Errors
    ///
    /// Returns a geometry, retirement, or live-span [`GraphError`] before graph
    /// state is changed.
    pub fn learn_copy(&mut self, var: VarId, value: &[u8]) -> Result<(), GraphError> {
        self.validate_symbol(value)?;
        if self.ensure_variable_slot(var)? {
            return Ok(());
        }
        let value = self.pool.take_symbol_copy(value);
        self.apply_known(var, value);
        self.drive_peel();
        Ok(())
    }

    pub(crate) fn preflight_resolved_copies(
        &self,
        first: VarId,
        last: VarId,
        value: &[u8],
    ) -> Result<(), GraphError> {
        self.validate_symbol(value)?;
        // One ring, one check: the learn path grows only `known`. `waiting` is
        // grown by `push_check` alone, and never by the batch this preflights.
        self.known.check_range(first.get(), last.get())
    }

    pub(crate) fn learn_copy_preflighted(
        &mut self,
        var: VarId,
        value: &[u8],
    ) -> Result<(), GraphError> {
        debug_assert_eq!(value.len(), self.symbol_len);
        if self.ensure_variable_slot(var)? {
            return Ok(());
        }
        let value = self.pool.take_symbol_copy(value);
        self.apply_known(var, value);
        self.drive_peel();
        Ok(())
    }

    /// Ingest one validated check equation.
    ///
    /// Known neighbours are folded out immediately; only the residual support is
    /// retained. Duplicate check ids are ignored after full validation.
    ///
    /// # Errors
    ///
    /// Returns a symbol, retirement, or live-span [`GraphError`] before graph
    /// state is changed.
    pub fn push_check(
        &mut self,
        id: CheckId,
        edges: Edges<'_, W>,
        rhs: &[u8],
    ) -> Result<(), GraphError> {
        self.validate_symbol(rhs)?;
        let duplicate = self.preflight_check(id, edges)?;
        if duplicate {
            return Ok(());
        }

        let row_slot = self.rows.ensure(id.get())?;
        debug_assert!(matches!(row_slot, RowSlot::Vacant));
        for &var in edges.support() {
            let _ = self.known.ensure(var.get())?;
        }

        let mut reduced = self.pool.take_symbol_copy(rhs);
        let mut support = self.pool.take_support();
        let mut weights = self.pool.take_weights();
        support.reserve(edges.len());
        weights.reserve(edges.len());
        for (var, weight) in edges.iter() {
            match self.known.get(var.get()) {
                Lookup::Live(Some(value)) => W::mul_add(&mut reduced, weight, value),
                Lookup::Live(None) => {
                    support.push(var);
                    weights.push(weight);
                }
                Lookup::Vacant | Lookup::Retired => {
                    debug_assert!(false, "preflight failed to establish variable slots");
                }
            }
        }

        let ready = support.len() == 1;
        let resolved = support.is_empty();
        let min_var = support.iter().copied().min();
        for &var in &support {
            let (waiting, pool) = (&mut self.waiting, &mut self.pool);
            let list = waiting.ensure(var.get())?;
            if list.capacity() == 0 {
                let spare = pool.take_checks();
                if spare.capacity() != 0 {
                    *list = spare;
                } else {
                    pool.recycle_checks(spare);
                }
            }
            list.push(id);
        }

        // Lower the mark for the row about to become live. It is only ever
        // lowered here: when a row's own minimum rises in
        // `refresh_min_after_removing` the mark is deliberately left stale-low,
        // which is safe because the mark is a lower bound, never an estimate.
        if let Some(min) = min_var {
            self.row_min_low_water = self.row_min_low_water.min(min.get());
        }

        let row = CheckRow {
            rhs: reduced,
            support,
            weights,
            min_var,
            resolved,
        };
        *self.rows.ensure(id.get())? = RowSlot::Live(row);
        self.row_count += 1;
        if !resolved {
            self.unresolved += 1;
        }
        if ready {
            self.ripple.push(id);
        }
        self.drive_peel();
        Ok(())
    }

    /// Generate and ingest one check with reusable internal neighbour scratch.
    ///
    /// # Errors
    ///
    /// Returns generator errors or the same errors as [`push_check`](Self::push_check).
    /// Generator scratch is restored on every path.
    pub fn push_check_with<G: NeighborGen<Weight = W>>(
        &mut self,
        id: CheckId,
        generator: &G,
        rhs: &[u8],
    ) -> Result<(), GraphError> {
        let mut scratch = mem::take(&mut self.neighbor_buf);
        let result = match generator.neighbors(id, &mut scratch) {
            Ok(()) => match scratch.edges() {
                Ok(edges) => self.push_check(id, edges, rhs),
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        };
        self.neighbor_buf = scratch;
        result
    }

    /// Retire all variable state below `horizon`.
    ///
    /// Rows still depending on a retired unknown are retired too. Rows whose old
    /// minimum was already folded out remain useful and are retained.
    ///
    /// # Errors
    ///
    /// [`GraphError::HorizonRegressed`] when the horizon moves backwards.
    pub fn retire_below(&mut self, horizon: VarId) -> Result<(), GraphError> {
        let floor = self.known.floor().max(self.waiting.floor());
        if horizon.get() < floor {
            return Err(GraphError::HorizonRegressed {
                horizon: horizon.get(),
                floor,
            });
        }

        let mut dropped_known = 0usize;
        {
            let (known, pool) = (&mut self.known, &mut self.pool);
            for value in known.retire_below(horizon.get()).flatten() {
                dropped_known += 1;
                pool.recycle_symbol(value);
            }
        }
        self.known_count -= dropped_known;
        {
            let (waiting, pool) = (&mut self.waiting, &mut self.pool);
            for list in waiting.retire_below(horizon.get()) {
                pool.recycle_checks(list);
            }
        }

        // Every live row's `min_var` is at or above the low-water mark, so once
        // the mark reaches the horizon no row can qualify and the whole-row walk
        // is pure cost. On a clean stream every row folds all its neighbours out
        // at ingest, leaving `min_var` `None` everywhere and the mark at
        // `u64::MAX`, so the scan never runs.
        if self.row_min_low_water < horizon.get() {
            let mut dropped_rows = 0usize;
            let mut dropped_unresolved = 0usize;
            let mut low_water = u64::MAX;
            {
                let (rows, pool) = (&mut self.rows, &mut self.pool);
                for (_, slot) in rows.iter_mut() {
                    let min_var = match slot {
                        RowSlot::Live(row) => row.min_var,
                        RowSlot::Vacant | RowSlot::Retired => continue,
                    };
                    match min_var {
                        Some(var) if var < horizon => {}
                        Some(var) => {
                            low_water = low_water.min(var.get());
                            continue;
                        }
                        None => continue,
                    }
                    let RowSlot::Live(row) = mem::replace(slot, RowSlot::Retired) else {
                        continue;
                    };
                    dropped_rows += 1;
                    dropped_unresolved += usize::from(row.is_unresolved());
                    pool.recycle_symbol(row.rhs);
                    pool.recycle_support(row.support);
                    pool.recycle_weights(row.weights);
                }
            }
            // The scan visited every survivor, so the mark can be tightened to
            // the exact minimum instead of merely left conservative.
            self.row_min_low_water = low_water;
            self.row_count -= dropped_rows;
            self.unresolved -= dropped_unresolved;
        }
        self.trim_rows_front();
        Ok(())
    }

    /// Explicitly retire one check row.
    ///
    /// Reverse-adjacency and ripple ids are deliberately left stale and validated
    /// when observed; removing them eagerly would require extra scans.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::IndexRetired`] or [`GraphError::CheckRetired`] when
    /// the check was already retired.
    pub fn retire_check(&mut self, id: CheckId) -> Result<(), GraphError> {
        self.rows.check_range(id.get(), id.get())?;
        match self.rows.get(id.get()) {
            Lookup::Retired => {
                return Err(GraphError::IndexRetired {
                    index: id.get(),
                    floor: self.rows.floor(),
                });
            }
            Lookup::Live(RowSlot::Retired) => {
                return Err(GraphError::CheckRetired { check: id.get() });
            }
            Lookup::Vacant | Lookup::Live(RowSlot::Vacant) => {
                *self.rows.ensure(id.get())? = RowSlot::Retired;
                self.trim_rows_front();
                return Ok(());
            }
            Lookup::Live(RowSlot::Live(_)) => {}
        }
        let row = match self.rows.get_mut(id.get()) {
            Lookup::Live(slot @ RowSlot::Live(_)) => {
                let RowSlot::Live(row) = mem::replace(slot, RowSlot::Retired) else {
                    return Ok(());
                };
                row
            }
            Lookup::Live(RowSlot::Vacant | RowSlot::Retired) | Lookup::Vacant | Lookup::Retired => {
                return Ok(());
            }
        };

        self.row_count -= 1;
        if row.is_unresolved() {
            self.unresolved -= 1;
        }
        self.pool.recycle_symbol(row.rhs);
        self.pool.recycle_support(row.support);
        self.pool.recycle_weights(row.weights);
        self.trim_rows_front();
        Ok(())
    }

    fn validate_symbol(&self, value: &[u8]) -> Result<(), GraphError> {
        if value.len() != self.symbol_len {
            return Err(GraphError::SymbolLengthMismatch {
                expected: self.symbol_len,
                actual: value.len(),
            });
        }
        Ok(())
    }

    /// Make the learn path's slot, reporting whether a value is already there.
    ///
    /// One lookup does all three jobs the learn path used to spend three on:
    /// range validation, the duplicate-value probe, and slot creation.
    /// `Ring::ensure` validates the whole range before it mutates and leaves the
    /// ring exactly as it was on error, and `known` is the only ring the learn
    /// path grows, so "limits reject input and never evict state" holds with one
    /// fallible call. Probing after `ensure` rather than before is
    /// observationally identical: creating a vacant slot for an id that is
    /// in-range and live changes nothing a caller can see, and a retired or
    /// out-of-span id is rejected by `ensure` with exactly the `GraphError` the
    /// separate preflight used to raise.
    fn ensure_variable_slot(&mut self, var: VarId) -> Result<bool, GraphError> {
        Ok(self.known.ensure(var.get())?.is_some())
    }

    fn preflight_check(&self, id: CheckId, edges: Edges<'_, W>) -> Result<bool, GraphError> {
        self.rows.check_range(id.get(), id.get())?;
        let duplicate = match self.rows.get(id.get()) {
            Lookup::Retired => {
                return Err(GraphError::IndexRetired {
                    index: id.get(),
                    floor: self.rows.floor(),
                });
            }
            Lookup::Live(RowSlot::Retired) => {
                return Err(GraphError::CheckRetired { check: id.get() });
            }
            Lookup::Live(RowSlot::Live(_)) => true,
            Lookup::Live(RowSlot::Vacant) | Lookup::Vacant => false,
        };
        let min_var = edges.min_var().get();
        let max_var = edges
            .support()
            .iter()
            .copied()
            .max()
            .map_or(min_var, VarId::get);
        self.known.check_range(min_var, max_var)?;
        self.waiting.check_range(min_var, max_var)?;
        Ok(duplicate)
    }

    fn apply_known(&mut self, var: VarId, value: Vec<u8>) {
        // `waiting` holds a slot only for variables some live check registered
        // against, so a variable nothing waits on costs one failed lookup —
        // not a `mem::take` plus write-back to discover an empty list.
        let mut keys = match self.waiting.get_mut(var.get()) {
            Lookup::Live(list) if !list.is_empty() => mem::take(list),
            Lookup::Live(_) | Lookup::Vacant | Lookup::Retired => {
                self.store_known(var, value);
                return;
            }
        };
        for &check in &keys {
            let transition = match self.rows.get_mut(check.get()) {
                Lookup::Live(RowSlot::Live(row)) if row.is_unresolved() => {
                    let Some(pos) = row.support.iter().position(|candidate| *candidate == var)
                    else {
                        continue;
                    };
                    row.support.swap_remove(pos);
                    let weight = row.weights.swap_remove(pos);
                    W::mul_add(&mut row.rhs, weight, &value);
                    row.refresh_min_after_removing(var);
                    match row.support.len() {
                        0 => {
                            row.resolved = true;
                            2
                        }
                        1 => 1,
                        _ => 0,
                    }
                }
                Lookup::Live(RowSlot::Vacant | RowSlot::Retired | RowSlot::Live(_))
                | Lookup::Vacant
                | Lookup::Retired => 0,
            };
            if transition == 1 {
                self.ripple.push(check);
            } else if transition == 2 {
                self.unresolved -= 1;
            }
        }
        keys.clear();
        if let Lookup::Live(list) = self.waiting.get_mut(var.get()) {
            *list = keys;
        } else {
            self.pool.recycle_checks(keys);
        }
        self.store_known(var, value);
    }

    fn store_known(&mut self, var: VarId, value: Vec<u8>) {
        if let Lookup::Live(slot) = self.known.get_mut(var.get()) {
            *slot = Some(value);
            self.known_count += 1;
        } else {
            self.pool.recycle_symbol(value);
        }
    }

    fn drive_peel(&mut self) {
        while let Some(check) = self.ripple.pop() {
            let peeled = match self.rows.get_mut(check.get()) {
                Lookup::Live(RowSlot::Live(row)) if !row.resolved && row.support.len() == 1 => {
                    let var = row.support[0];
                    let weight = row.weights[0];
                    let value = mem::take(&mut row.rhs);
                    row.resolved = true;
                    Some((var, weight, value))
                }
                Lookup::Live(RowSlot::Vacant | RowSlot::Retired | RowSlot::Live(_))
                | Lookup::Vacant
                | Lookup::Retired => None,
            };
            let Some((var, weight, mut value)) = peeled else {
                continue;
            };
            self.unresolved -= 1;
            W::scale_inv(&mut value, weight);
            self.recovered.push(var);
            self.apply_known(var, value);
        }
    }

    fn trim_rows_front(&mut self) {
        let prefix = self
            .rows
            .iter()
            .take_while(|(_, slot)| matches!(slot, RowSlot::Retired))
            .count();
        if prefix == 0 {
            return;
        }
        let horizon = self.rows.base().saturating_add(prefix as u64);
        for _ in self.rows.retire_below(horizon) {}
    }
}

#[cfg(test)]
mod internal_tests {
    use super::*;
    use crate::{Binary, Edges};
    use core::num::NonZeroUsize;

    #[test]
    fn explicit_retirement_tolerates_stale_adjacency_and_ripple_ids() {
        let span = NonZeroUsize::new(8).unwrap();
        let mut peeler = Peeler::<Binary>::new(1, PoolConfig::new(span, span)).unwrap();
        let support = [VarId::new(0), VarId::new(1)];
        let weights = [Binary; 2];
        let id = CheckId::new(0);
        peeler
            .push_check(id, Edges::new(&support, &weights).unwrap(), &[0xA5])
            .unwrap();
        peeler.ripple.push(id);

        peeler.retire_check(id).unwrap();
        peeler.drive_peel();
        peeler.learn_copy(support[0], &[0x11]).unwrap();
        peeler.learn_copy(support[1], &[0x22]).unwrap();

        assert_eq!(peeler.check_count(), 0);
        assert!(peeler.ripple.is_empty());
        assert_eq!(
            peeler.variable_state(support[0]),
            VariableState::Known(&[0x11])
        );
        assert_eq!(
            peeler.variable_state(support[1]),
            VariableState::Known(&[0x22])
        );
    }
}
