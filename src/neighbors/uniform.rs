//! Uniform distinct-k edge generators.
//!
//! Both draw a degree, then that many distinct offsets over their domain, from
//! one [`SplitMix64`] stream seeded by `id ^ domain_sep`. [`Uniform`] addresses a
//! fixed block; [`WindowedUniform`] addresses a sliding window and clamps the
//! degree to what the window currently holds.

use super::{NeighborBuf, NeighborGen};
use crate::degree::DegreeDistribution;
use crate::error::GraphError;
use crate::id::{CheckId, VarId};
use crate::rng::{SplitMix64, sample_distinct, seed_for};
use crate::weight::{Binary, EdgeWeight};

/// Largest domain a `u32` offset can address.
const MAX_DOMAIN: u64 = u32::MAX as u64;

/// Validate a degree distribution against a domain width.
fn check_degree<D: DegreeDistribution>(degree: &D, domain: u64) -> Result<u32, GraphError> {
    if domain == 0 {
        return Err(GraphError::EmptyDomain);
    }
    let max = degree.max_degree();
    if max == 0 {
        return Err(GraphError::ZeroDegree);
    }
    if u64::from(max) > domain {
        return Err(GraphError::DegreeExceedsDomain {
            degree: max,
            domain,
        });
    }
    Ok(max)
}

/// Draw `min(degree, span)` distinct offsets into `out`'s scratch, returning how
/// many landed there.
///
/// The degree is drawn first and from the *same* generator, so a distribution
/// that consumes state shifts the offsets — which is the intent. A point mass
/// consumes nothing, leaving the offsets identical to a bare
/// `SplitMix64::new(seed)` sampling, which is what preserves wire compatibility
/// with a constant-degree code.
fn draw_offsets<D: DegreeDistribution, W: EdgeWeight>(
    seed: u64,
    span: u32,
    cap: u32,
    degree: &D,
    out: &mut NeighborBuf<W>,
) -> usize {
    let mut rng = SplitMix64::new(seed);
    let drawn = degree.sample(&mut rng);
    let k = drawn.min(span).min(cap) as usize;
    let scratch = out.offset_scratch(k);
    sample_distinct(&mut rng, span, scratch);
    k
}

/// Distinct-k edges over the fixed block `[0, domain)`.
///
/// Every check draws from the whole block, which is the shape a block LT or LDPC
/// code wants.
#[derive(Debug, Clone)]
pub struct Uniform<D> {
    span: u32,
    degree: D,
    domain_sep: u64,
    max_degree: u32,
}

impl<D: DegreeDistribution> Uniform<D> {
    /// A generator over variables `0..domain`.
    ///
    /// `domain_sep` is yours to choose. It is not baked into this crate because
    /// seed derivation is a wire-compatibility decision: it keeps one code's edge
    /// stream distinct from another's over the same check ids.
    ///
    /// # Errors
    ///
    /// * [`GraphError::EmptyDomain`] — nothing to draw from.
    /// * [`GraphError::ZeroDegree`] — the distribution can only produce edgeless
    ///   checks.
    /// * [`GraphError::DegreeExceedsDomain`] — more distinct variables could be
    ///   requested than exist.
    /// * [`GraphError::DomainTooLarge`] — offsets are `u32`, so the block cannot
    ///   be wider than [`u32::MAX`].
    pub fn new(domain: u64, degree: D, domain_sep: u64) -> Result<Self, GraphError> {
        let max_degree = check_degree(&degree, domain)?;
        if domain > MAX_DOMAIN {
            return Err(GraphError::DomainTooLarge {
                domain,
                max: MAX_DOMAIN,
            });
        }
        Ok(Self {
            span: domain as u32,
            degree,
            domain_sep,
            max_degree,
        })
    }

    /// Number of variables in the block.
    #[inline]
    #[must_use]
    pub fn domain(&self) -> u64 {
        u64::from(self.span)
    }
}

impl<D: DegreeDistribution> NeighborGen for Uniform<D> {
    type Weight = Binary;

    fn neighbors(&self, id: CheckId, out: &mut NeighborBuf<Binary>) -> Result<(), GraphError> {
        out.clear();
        let k = draw_offsets(
            seed_for(id.get(), self.domain_sep),
            self.span,
            self.max_degree,
            &self.degree,
            out,
        );
        out.fill_from_offsets(k, Binary::one(), |off| VarId::new(u64::from(off)));
        Ok(())
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.max_degree
    }
}

/// Distinct-k edges over the window `[base, base + span)`.
///
/// The degree is clamped to `span`, so a window that is not yet full produces
/// lower-degree checks rather than failing. That clamp is this generator's own
/// rule — early in a stream there is simply less to point at — and not a concept
/// the rest of the crate knows about.
#[derive(Debug, Clone)]
pub struct WindowedUniform<D> {
    base: u64,
    span: u32,
    degree: D,
    domain_sep: u64,
    max_degree: u32,
}

impl<D: DegreeDistribution> WindowedUniform<D> {
    /// A generator over variables `base..base + span`.
    ///
    /// Unlike [`Uniform`], the degree may exceed `span`: it is clamped per check
    /// rather than rejected, because a window legitimately starts out narrow.
    ///
    /// # Errors
    ///
    /// * [`GraphError::EmptyDomain`] — `span` is zero.
    /// * [`GraphError::ZeroDegree`] — the distribution can only produce edgeless
    ///   checks.
    /// * [`GraphError::DomainOverflow`] — the window's last index would pass
    ///   [`u64::MAX`].
    pub fn new(base: u64, span: u32, degree: D, domain_sep: u64) -> Result<Self, GraphError> {
        if span == 0 {
            return Err(GraphError::EmptyDomain);
        }
        if degree.max_degree() == 0 {
            return Err(GraphError::ZeroDegree);
        }
        // The window's highest index is `base + span - 1`; anything past
        // `u64::MAX` is not addressable.
        if base.checked_add(u64::from(span) - 1).is_none() {
            return Err(GraphError::DomainOverflow { base, span });
        }
        let max_degree = degree.max_degree().min(span);
        Ok(Self {
            base,
            span,
            degree,
            domain_sep,
            max_degree,
        })
    }

    /// First variable index of the window.
    #[inline]
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Number of variables in the window.
    #[inline]
    #[must_use]
    pub fn span(&self) -> u32 {
        self.span
    }
}

impl<D: DegreeDistribution> NeighborGen for WindowedUniform<D> {
    type Weight = Binary;

    fn neighbors(&self, id: CheckId, out: &mut NeighborBuf<Binary>) -> Result<(), GraphError> {
        out.clear();
        let k = draw_offsets(
            seed_for(id.get(), self.domain_sep),
            self.span,
            self.max_degree,
            &self.degree,
            out,
        );
        // In range by construction: `off < span`, and `base + span - 1` was
        // checked against `u64::MAX` at construction.
        let base = self.base;
        out.fill_from_offsets(k, Binary::one(), |off| VarId::new(base + u64::from(off)));
        Ok(())
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.max_degree
    }
}
