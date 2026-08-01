//! Reusable symbol and adjacency storage for the peeling hot path.

use crate::{CheckId, EdgeWeight, VarId};
use alloc::vec::Vec;
use core::num::NonZeroUsize;

const DEFAULT_POOL_BYTES: usize = 262_144;
const MIN_DEFAULT_POOL_ENTRIES: usize = 8;
const MAX_DEFAULT_POOL_ENTRIES: usize = 256;

/// Peeler storage bounds and recycling policy.
///
/// The two live spans bound dense index storage. Pool capacity is an entry count,
/// not a graph geometry; when left automatic it targets roughly 256 KiB of symbol
/// buffers, clamped to 8–256 entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolConfig {
    max_variable_span: NonZeroUsize,
    max_check_span: NonZeroUsize,
    pool_capacity: Option<usize>,
}

impl PoolConfig {
    /// Configure the maximum simultaneously-live variable and check spans.
    #[must_use]
    pub const fn new(max_variable_span: NonZeroUsize, max_check_span: NonZeroUsize) -> Self {
        Self {
            max_variable_span,
            max_check_span,
            pool_capacity: None,
        }
    }

    /// Override the number of retired buffers retained for reuse.
    #[must_use]
    pub const fn with_pool_capacity(mut self, entries: usize) -> Self {
        self.pool_capacity = Some(entries);
        self
    }

    /// Maximum simultaneously-live variable span.
    #[must_use]
    pub const fn max_variable_span(self) -> NonZeroUsize {
        self.max_variable_span
    }

    /// Maximum simultaneously-live check span.
    #[must_use]
    pub const fn max_check_span(self) -> NonZeroUsize {
        self.max_check_span
    }

    /// Explicit pool entry limit, or `None` for the symbol-size heuristic.
    #[must_use]
    pub const fn pool_capacity(self) -> Option<usize> {
        self.pool_capacity
    }

    pub(super) fn resolved_pool_capacity(self, symbol_len: usize) -> usize {
        self.pool_capacity.unwrap_or_else(|| {
            (DEFAULT_POOL_BYTES / symbol_len)
                .clamp(MIN_DEFAULT_POOL_ENTRIES, MAX_DEFAULT_POOL_ENTRIES)
        })
    }
}

/// Buffers reclaimed by retirement and reused by later ingest.
#[derive(Debug)]
pub(super) struct Pool<W> {
    cap: usize,
    symbols: Vec<Vec<u8>>,
    supports: Vec<Vec<VarId>>,
    weights: Vec<Vec<W>>,
    checks: Vec<Vec<CheckId>>,
}

impl<W: EdgeWeight> Pool<W> {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            cap,
            symbols: Vec::new(),
            supports: Vec::new(),
            weights: Vec::new(),
            checks: Vec::new(),
        }
    }

    pub(super) fn take_symbol(&mut self) -> Option<Vec<u8>> {
        self.symbols.pop()
    }

    pub(super) fn take_symbol_copy(&mut self, value: &[u8]) -> Vec<u8> {
        let mut buffer = self.symbols.pop().unwrap_or_default();
        buffer.clear();
        buffer.extend_from_slice(value);
        buffer
    }

    pub(super) fn recycle_symbol(&mut self, mut buffer: Vec<u8>) {
        if buffer.capacity() != 0 && self.symbols.len() < self.cap {
            buffer.clear();
            self.symbols.push(buffer);
        }
    }

    pub(super) fn take_support(&mut self) -> Vec<VarId> {
        let mut support = self.supports.pop().unwrap_or_default();
        support.clear();
        support
    }

    pub(super) fn recycle_support(&mut self, mut support: Vec<VarId>) {
        if support.capacity() != 0 && self.supports.len() < self.cap {
            support.clear();
            self.supports.push(support);
        }
    }

    pub(super) fn take_weights(&mut self) -> Vec<W> {
        let mut weights = self.weights.pop().unwrap_or_default();
        weights.clear();
        weights
    }

    pub(super) fn recycle_weights(&mut self, mut weights: Vec<W>) {
        if weights.capacity() != 0 && self.weights.len() < self.cap {
            weights.clear();
            self.weights.push(weights);
        }
    }

    pub(super) fn take_checks(&mut self) -> Vec<CheckId> {
        let mut checks = self.checks.pop().unwrap_or_default();
        checks.clear();
        checks
    }

    pub(super) fn recycle_checks(&mut self, mut checks: Vec<CheckId>) {
        if checks.capacity() != 0 && self.checks.len() < self.cap {
            checks.clear();
            self.checks.push(checks);
        }
    }
}
