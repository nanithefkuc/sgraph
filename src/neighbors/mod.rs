//! Edge generation: what a check node is connected to.
//!
//! A check symbol travels without its graph. Both peers hold the same generator
//! and rebuild the identical edge set from the check's id, which is why
//! generation is a pure function of `(id, parameters)` and why changing one is a
//! wire break rather than a refactor.
//!
//! Selection strategy is a trait rather than a parameter because the families
//! genuinely differ in kind: uniform distinct-k over a domain, RFC 5053's
//! `(d, a, b)` triple walked over a prime modulus, and a fixed parity-check
//! matrix are not one algorithm with different constants. Floyd sampling is an
//! implementation detail of the uniform generators, not the interface.
//!
//! * [`Uniform`] — distinct-k over a fixed block.
//! * [`WindowedUniform`] — distinct-k over a sliding window.
//! * [`ExplicitMatrix`] — a fixed parity-check matrix, nothing sampled.
//! * [`Rfc5053Triple`] — RFC 5053 Raptor's `(d, a, b)` triple walk.

mod explicit;
mod triple;
mod uniform;
mod weighted;

pub use explicit::ExplicitMatrix;
pub use triple::Rfc5053Triple;
pub use uniform::{Uniform, WindowedUniform};
pub use weighted::{WeightedUniform, WeightedWindowedUniform};

use crate::error::GraphError;
use crate::id::{CheckId, VarId};
use crate::weight::EdgeWeight;
use alloc::vec::Vec;

/// Generates the edges of a check node.
///
/// Implementations MUST be deterministic and free of side effects: the same
/// `(id, parameters)` yields the same edges on every peer, every run, and every
/// platform.
pub trait NeighborGen {
    /// Coefficient type carried by this generator's edges.
    type Weight: EdgeWeight;

    /// Generate the edges of `id` into `out`, which is cleared first.
    ///
    /// # Errors
    ///
    /// Generator-specific. A finite generator rejects an out-of-domain `id`. On
    /// any error `out` is left **cleared**, not partially filled, so a caller can
    /// reuse the same scratch for the next check without observing debris.
    fn neighbors(&self, id: CheckId, out: &mut NeighborBuf<Self::Weight>)
    -> Result<(), GraphError>;

    /// Largest degree this generator can produce, for sizing scratch once.
    fn max_degree(&self) -> u32;
}

/// Reusable scratch for one check's edges.
///
/// Two parallel vectors rather than a vector of pairs, so that the weight side
/// vanishes entirely when the coefficient type is zero-sized. Reused across
/// checks: [`NeighborGen::neighbors`] clears it instead of reallocating, which is
/// what keeps steady-state generation allocation-free.
///
/// It also carries the `u32` offset scratch that sampling needs, so a generator
/// never has to allocate a temporary of its own. Size the whole thing once from
/// [`NeighborGen::max_degree`] and generation stops allocating entirely.
#[derive(Debug)]
pub struct NeighborBuf<W> {
    support: Vec<VarId>,
    weights: Vec<W>,
    offsets: Vec<u32>,
}

impl<W: EdgeWeight> Default for NeighborBuf<W> {
    fn default() -> Self {
        Self::new()
    }
}

impl<W: EdgeWeight> NeighborBuf<W> {
    /// An empty buffer that will allocate on first use.
    #[must_use]
    pub fn new() -> Self {
        Self {
            support: Vec::new(),
            weights: Vec::new(),
            offsets: Vec::new(),
        }
    }

    /// A buffer sized for `degree` edges up front.
    ///
    /// Size it from [`NeighborGen::max_degree`] once and generation never
    /// allocates again.
    #[must_use]
    pub fn with_capacity(degree: usize) -> Self {
        Self {
            support: Vec::with_capacity(degree),
            weights: Vec::with_capacity(degree),
            offsets: Vec::with_capacity(degree),
        }
    }

    /// Drop every edge, keeping the allocations.
    #[inline]
    pub fn clear(&mut self) {
        self.support.clear();
        self.weights.clear();
    }

    /// Append one edge.
    #[inline]
    pub fn push(&mut self, var: VarId, weight: W) {
        self.support.push(var);
        self.weights.push(weight);
    }

    /// A `len`-element `u32` scratch region for a generator's own sampling.
    ///
    /// Grows to `len` and keeps that capacity for later checks. The contents are
    /// unspecified on entry — a generator writes before it reads. This exists so
    /// samplers need no temporary of their own; pair it with
    /// [`fill_from_offsets`](NeighborBuf::fill_from_offsets).
    pub fn offset_scratch(&mut self, len: usize) -> &mut [u32] {
        if self.offsets.len() < len {
            self.offsets.resize(len, 0);
        }
        &mut self.offsets[..len]
    }

    /// Replace the edges with the first `count` scratch offsets, each mapped to a
    /// variable by `map` and carrying `weight`.
    ///
    /// The counterpart to [`offset_scratch`](NeighborBuf::offset_scratch): it
    /// reads that scratch and writes the parallel arrays without either side
    /// needing a temporary.
    ///
    /// # Panics
    ///
    /// If `count` exceeds the current scratch length.
    pub fn fill_from_offsets(&mut self, count: usize, weight: W, map: impl Fn(u32) -> VarId) {
        assert!(
            count <= self.offsets.len(),
            "fill_from_offsets: count {count} past scratch of {}",
            self.offsets.len()
        );
        // Disjoint field borrows: reading `offsets` while writing the other two.
        // `extend` over an exact-size iterator reserves once and skips the
        // per-element capacity check a `push` loop repeats.
        let Self {
            support,
            weights,
            offsets,
        } = self;
        support.clear();
        weights.clear();
        support.extend(offsets[..count].iter().map(|&off| map(off)));
        weights.extend(core::iter::repeat_n(weight, count));
    }
    pub(super) fn fill_from_offsets_with(
        &mut self,
        count: usize,
        mut map: impl FnMut(u32) -> (VarId, W),
    ) {
        assert!(
            count <= self.offsets.len(),
            "fill_from_offsets_with: count {count} past scratch of {}",
            self.offsets.len()
        );
        let Self {
            support,
            weights,
            offsets,
        } = self;
        support.clear();
        weights.clear();
        for &offset in &offsets[..count] {
            let (var, weight) = map(offset);
            support.push(var);
            weights.push(weight);
        }
    }

    /// Number of edges held.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.support.len()
    }

    /// True when no edges are held.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.support.is_empty()
    }

    /// The variables, in generated order.
    #[inline]
    #[must_use]
    pub fn support(&self) -> &[VarId] {
        &self.support
    }

    /// The coefficients, parallel to [`support`](NeighborBuf::support).
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &[W] {
        &self.weights
    }

    /// Borrow the contents as a validated edge set.
    ///
    /// # Errors
    ///
    /// As [`Edges::new`].
    pub fn edges(&self) -> Result<Edges<'_, W>, GraphError> {
        Edges::new(&self.support, &self.weights)
    }

    #[inline]
    pub(crate) fn generated_edges(&self) -> Edges<'_, W> {
        debug_assert!(Edges::new(&self.support, &self.weights).is_ok());
        Edges {
            support: &self.support,
            weights: &self.weights,
        }
    }
}

/// A validated view of one check's edges.
///
/// Constructing this is the single place edge shape is checked, so everything
/// downstream may assume the invariant: equal lengths, non-empty, each variable
/// at most once, and no zero coefficients. Generated order is deterministic but
/// not necessarily sorted, and nothing downstream may assume otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edges<'a, W> {
    support: &'a [VarId],
    weights: &'a [W],
}

impl<'a, W: EdgeWeight> Edges<'a, W> {
    /// Validate parallel support and weight slices as an edge set.
    ///
    /// # Errors
    ///
    /// * [`GraphError::EdgeLengthMismatch`] — the parallel arrays disagree.
    /// * [`GraphError::EmptySupport`] — a check with no support constrains
    ///   nothing.
    /// * [`GraphError::DuplicateVariable`] — a repeated variable would
    ///   accumulate silently during reduction and corrupt the residual
    ///   invariant.
    /// * [`GraphError::ZeroWeight`] — a zero coefficient is not an edge; left in
    ///   place it makes a degree-one row unsolvable while still looking peelable.
    pub fn new(support: &'a [VarId], weights: &'a [W]) -> Result<Self, GraphError> {
        if support.len() != weights.len() {
            return Err(GraphError::EdgeLengthMismatch {
                support: support.len(),
                weights: weights.len(),
            });
        }
        if support.is_empty() {
            return Err(GraphError::EmptySupport);
        }
        for (i, &var) in support.iter().enumerate() {
            // Quadratic in the degree, which is bounded by the generator's
            // `max_degree` and small in every real construction. A set would cost
            // an allocation on a path that must not allocate.
            if support[..i].contains(&var) {
                return Err(GraphError::DuplicateVariable { var: var.get() });
            }
            if weights[i].is_zero() {
                return Err(GraphError::ZeroWeight { var: var.get() });
            }
        }
        Ok(Self { support, weights })
    }

    /// The variables this check constrains.
    #[inline]
    #[must_use]
    pub fn support(&self) -> &'a [VarId] {
        self.support
    }

    /// The coefficients, parallel to [`support`](Edges::support).
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &'a [W] {
        self.weights
    }

    /// Number of edges.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.support.len()
    }

    /// Always false; an empty edge set cannot be constructed.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The edges as `(variable, coefficient)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (VarId, W)> + 'a {
        let weights = self.weights;
        self.support
            .iter()
            .copied()
            .enumerate()
            .map(move |(i, v)| (v, weights[i]))
    }

    /// Lowest variable index in the support.
    ///
    /// The support is not sorted, so this is a scan. Retirement needs it to know
    /// whether a row still depends on an index that could be retired.
    #[must_use]
    pub fn min_var(&self) -> VarId {
        self.support.iter().copied().min().unwrap_or(VarId::ZERO)
    }
}

#[cfg(test)]
mod tests;
