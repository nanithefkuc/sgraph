//! Uniform topology with deterministic non-zero field coefficients.

use super::uniform::{MAX_DOMAIN, check_degree, draw_offsets};
use super::{NeighborBuf, NeighborGen};
use crate::degree::DegreeDistribution;
use crate::error::GraphError;
use crate::id::{CheckId, VarId};
use crate::rng::{SplitMix64, seed_for};
use crate::weight::Weighted;
use core::marker::PhantomData;
use fgf::FieldKernels;

/// Draw one non-zero field element.
///
/// One `next_u64` supplies each candidate in little-endian field encoding; zero
/// is rejected and consumes another complete draw. `fgf`'s sealed field set has
/// elements no wider than eight bytes, so every candidate comes from one draw.
fn draw_weight<F: FieldKernels>(rng: &mut SplitMix64) -> Weighted<F> {
    debug_assert!(F::BYTES <= size_of::<u64>());
    loop {
        let encoded = rng.next_u64().to_le_bytes();
        let value = F::read(&encoded[..F::BYTES]);
        if let Some(weight) = Weighted::new(value) {
            return weight;
        }
    }
}

fn fill_weighted<F: FieldKernels + 'static>(
    out: &mut NeighborBuf<Weighted<F>>,
    count: usize,
    base: u64,
    id: CheckId,
    weight_domain_sep: u64,
) {
    let mut rng = SplitMix64::new(seed_for(id.get(), weight_domain_sep));
    out.fill_from_offsets_with(count, |offset| {
        (
            VarId::new(base + u64::from(offset)),
            draw_weight::<F>(&mut rng),
        )
    });
}

/// Distinct uniformly sampled variables over a fixed block, with deterministic
/// non-zero coefficients over `F`.
///
/// Topology and weights use separate caller-chosen domain separators. The
/// topology stream is therefore byte-identical to [`super::Uniform`] configured
/// with the same degree and topology separator, while coefficient draws cannot
/// perturb its offsets.
#[derive(Debug, Clone)]
pub struct WeightedUniform<F, D> {
    span: u32,
    degree: D,
    topology_domain_sep: u64,
    weight_domain_sep: u64,
    max_degree: u32,
    field: PhantomData<F>,
}

impl<F: FieldKernels, D: DegreeDistribution> WeightedUniform<F, D> {
    /// A weighted generator over variables `0..domain`.
    ///
    /// # Errors
    ///
    /// Returns the same domain and degree errors as [`super::Uniform::new`].
    pub fn new(
        domain: u64,
        degree: D,
        topology_domain_sep: u64,
        weight_domain_sep: u64,
    ) -> Result<Self, GraphError> {
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
            topology_domain_sep,
            weight_domain_sep,
            max_degree,
            field: PhantomData,
        })
    }

    /// Number of variables in the block.
    #[inline]
    #[must_use]
    pub fn domain(&self) -> u64 {
        u64::from(self.span)
    }
}

impl<F: FieldKernels + 'static, D: DegreeDistribution> NeighborGen for WeightedUniform<F, D> {
    type Weight = Weighted<F>;

    fn neighbors(
        &self,
        id: CheckId,
        out: &mut NeighborBuf<Self::Weight>,
    ) -> Result<(), GraphError> {
        out.clear();
        let count = draw_offsets(
            seed_for(id.get(), self.topology_domain_sep),
            self.span,
            self.max_degree,
            &self.degree,
            out,
        );
        fill_weighted(out, count, 0, id, self.weight_domain_sep);
        Ok(())
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.max_degree
    }
}

/// Distinct uniformly sampled variables over a sliding window, with
/// deterministic non-zero coefficients over `F`.
#[derive(Debug, Clone)]
pub struct WeightedWindowedUniform<F, D> {
    base: u64,
    span: u32,
    degree: D,
    topology_domain_sep: u64,
    weight_domain_sep: u64,
    max_degree: u32,
    field: PhantomData<F>,
}

impl<F: FieldKernels, D: DegreeDistribution> WeightedWindowedUniform<F, D> {
    /// A weighted generator over variables `base..base + span`.
    ///
    /// Degree is clamped to `span`, matching [`super::WindowedUniform`].
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EmptyDomain`], [`GraphError::ZeroDegree`], or
    /// [`GraphError::DomainOverflow`] for invalid geometry.
    pub fn new(
        base: u64,
        span: u32,
        degree: D,
        topology_domain_sep: u64,
        weight_domain_sep: u64,
    ) -> Result<Self, GraphError> {
        if span == 0 {
            return Err(GraphError::EmptyDomain);
        }
        if degree.max_degree() == 0 {
            return Err(GraphError::ZeroDegree);
        }
        if base.checked_add(u64::from(span) - 1).is_none() {
            return Err(GraphError::DomainOverflow { base, span });
        }
        let max_degree = degree.max_degree().min(span);
        Ok(Self {
            base,
            span,
            degree,
            topology_domain_sep,
            weight_domain_sep,
            max_degree,
            field: PhantomData,
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

impl<F: FieldKernels + 'static, D: DegreeDistribution> NeighborGen
    for WeightedWindowedUniform<F, D>
{
    type Weight = Weighted<F>;

    fn neighbors(
        &self,
        id: CheckId,
        out: &mut NeighborBuf<Self::Weight>,
    ) -> Result<(), GraphError> {
        out.clear();
        let count = draw_offsets(
            seed_for(id.get(), self.topology_domain_sep),
            self.span,
            self.max_degree,
            &self.degree,
            out,
        );
        fill_weighted(out, count, self.base, id, self.weight_domain_sep);
        Ok(())
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.max_degree
    }
}

#[cfg(test)]
mod tests {
    use super::{WeightedUniform, WeightedWindowedUniform};
    use crate::{
        Binary, CheckId, Constant, EdgeWeight, NeighborBuf, NeighborGen, Uniform, WindowedUniform,
    };
    use alloc::vec::Vec;
    use fgf::{Gf8, gf8};

    const TOPOLOGY_DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;
    const WEIGHT_DOMAIN: u64 = 0xD1CE_C0EF_FEE1_DEAD;

    #[test]
    fn weighted_topology_matches_binary_and_draw_order_is_pinned() {
        let degree = Constant::new(4).unwrap();
        let binary = Uniform::new(64, degree, TOPOLOGY_DOMAIN).unwrap();
        let weighted =
            WeightedUniform::<Gf8, _>::new(64, degree, TOPOLOGY_DOMAIN, WEIGHT_DOMAIN).unwrap();
        let mut binary_out: NeighborBuf<Binary> = NeighborBuf::with_capacity(4);
        let mut weighted_out = NeighborBuf::with_capacity(4);
        binary.neighbors(CheckId::new(42), &mut binary_out).unwrap();
        weighted
            .neighbors(CheckId::new(42), &mut weighted_out)
            .unwrap();

        assert_eq!(weighted_out.support(), binary_out.support());
        assert_eq!(
            weighted_out
                .weights()
                .iter()
                .map(|weight| weight.get())
                .collect::<Vec<_>>(),
            [
                gf8::Elem(0x5e),
                gf8::Elem(0x8e),
                gf8::Elem(0xa7),
                gf8::Elem(0x47)
            ]
        );
    }

    #[test]
    fn zero_weight_candidate_consumes_a_complete_rejection_draw() {
        let generator = WeightedUniform::<Gf8, _>::new(
            8,
            Constant::new(1).unwrap(),
            TOPOLOGY_DOMAIN,
            WEIGHT_DOMAIN,
        )
        .unwrap();
        let mut out = NeighborBuf::with_capacity(1);
        generator.neighbors(CheckId::new(88), &mut out).unwrap();
        assert_eq!(out.weights()[0].get(), gf8::Elem(0x3e));
    }

    #[test]
    fn weighted_window_clamps_degree_without_changing_topology() {
        let degree = Constant::new(8).unwrap();
        let binary = WindowedUniform::new(100, 3, degree, TOPOLOGY_DOMAIN).unwrap();
        let weighted =
            WeightedWindowedUniform::<Gf8, _>::new(100, 3, degree, TOPOLOGY_DOMAIN, WEIGHT_DOMAIN)
                .unwrap();
        let mut binary_out = NeighborBuf::with_capacity(3);
        let mut weighted_out = NeighborBuf::with_capacity(3);
        binary.neighbors(CheckId::new(9), &mut binary_out).unwrap();
        weighted
            .neighbors(CheckId::new(9), &mut weighted_out)
            .unwrap();
        assert_eq!(weighted_out.support(), binary_out.support());
        assert_eq!(weighted_out.len(), 3);
        assert!(
            weighted_out
                .weights()
                .iter()
                .all(|weight| !weight.is_zero())
        );
    }
}
