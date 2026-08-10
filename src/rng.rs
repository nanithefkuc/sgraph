//! Deterministic PRNG and distinct-k sampling.
//!
//! A check symbol travels without its graph: both peers regenerate the identical
//! edge set from the check's id alone. That makes reproducibility a wire
//! property, so everything here is fixed by contract — a change to the
//! generator, the sampling algorithm, or the draw order is a format break for
//! every downstream consumer, not a refactor. The fixtures under `tests/data/`
//! pin it.
//!
//! Sampling functions take `&mut SplitMix64` rather than a seed so that a degree
//! draw and an edge draw compose into one reproducible stream. Drawing a degree
//! from a separately-seeded generator would leave the two streams independent,
//! which is how correlated-graph bugs get in.
//!
//! The public entry points validate before touching caller output, so a rejected
//! request leaves the output buffer untouched rather than half-written.

use crate::error::GraphError;
use core::num::NonZeroU32;

/// `SplitMix64` — a fast, high-quality PRNG with trivial state.
///
/// Suitable for low-entropy seeds such as consecutive check ids, because its
/// increment/mix construction decorrelates nearby seeds.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed the generator. Any seed is valid, including zero.
    #[inline]
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `[0, bound)`.
    ///
    /// Lemire's multiply-shift with rejection of the biased low band, so the
    /// result is unbiased rather than merely cheap. The bound is
    /// [`NonZeroU32`] because a zero bound has no uniform value — it is
    /// unrepresentable rather than checked.
    #[inline]
    pub fn below(&mut self, bound: NonZeroU32) -> u32 {
        let bound = bound.get();
        let mut m = u64::from(self.next_u64() as u32).wrapping_mul(u64::from(bound));
        let mut lo = m as u32;
        if lo < bound {
            let threshold = bound.wrapping_neg() % bound;
            while lo < threshold {
                m = u64::from(self.next_u64() as u32).wrapping_mul(u64::from(bound));
                lo = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

/// Seed the edge stream of the check symbol `id` within the domain `domain_sep`.
///
/// The domain constant is the caller's to choose and is deliberately not baked
/// into this crate: it keeps one consumer's edge stream distinct from another's
/// over the same check ids, so it is a wire-compatibility decision. Mixing is
/// [`SplitMix64`]'s job — a bare XOR is sufficient here precisely because its
/// seeding step decorrelates nearby values.
#[inline]
#[must_use]
pub fn seed_for(id: u64, domain_sep: u64) -> u64 {
    id ^ domain_sep
}

/// Fill `out` with `out.len()` **distinct** offsets in `[0, span)`.
///
/// Floyd's algorithm for sampling k distinct values from n: exactly k draws,
/// distinctness enforced by a linear scan over the (small) output, no
/// allocation. The result is a set, not a sorted sequence — the order carries no
/// meaning, but it is fixed by contract, so do not sort in place and expect
/// callers to agree.
///
/// A request for zero offsets consumes no draws, which is what lets a
/// point-mass degree distribution leave the edge stream untouched.
///
/// # Errors
///
/// [`GraphError::SampleSpanTooSmall`] when `out.len() > span`, since fewer
/// available offsets than requested cannot yield a distinct set. Validated
/// before any draw, so `out` and the generator are both untouched on error.
pub fn distinct_offsets(
    rng: &mut SplitMix64,
    span: u32,
    out: &mut [u32],
) -> Result<(), GraphError> {
    let requested = out.len();
    if u64::try_from(requested).unwrap_or(u64::MAX) > u64::from(span) {
        return Err(GraphError::SampleSpanTooSmall { span, requested });
    }
    sample_distinct(rng, span, out);
    Ok(())
}

/// [`distinct_offsets`] from a seed, for a caller with no other draws to make.
///
/// # Errors
///
/// As [`distinct_offsets`].
pub fn distinct_offsets_seeded(seed: u64, span: u32, out: &mut [u32]) -> Result<(), GraphError> {
    distinct_offsets(&mut SplitMix64::new(seed), span, out)
}

/// Floyd's k-of-n, with the span precondition already established.
///
/// Generators call this directly once they have validated their own geometry, so
/// the check is not repeated per check symbol on the hot path.
pub(crate) fn sample_distinct(rng: &mut SplitMix64, span: u32, out: &mut [u32]) {
    let k = out.len() as u32;
    debug_assert!(span >= k, "sample_distinct: span {span} < k {k}");
    if k == 0 {
        return;
    }
    for (count, j) in ((span - k)..span).enumerate() {
        // `t` uniform in `[0, j]`; `j + 1` is non-zero by construction.
        let bound = NonZeroU32::new(j + 1).unwrap_or(NonZeroU32::MIN);
        let t = rng.below(bound);
        out[count] = if out[..count].contains(&t) { j } else { t };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An arbitrary domain constant for tests. Consumers pick their own.
    const DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).expect("test bound must be non-zero")
    }

    #[test]
    fn offsets_are_distinct_and_in_range() {
        let mut out = [0u32; 5];
        for id in 0..1000u64 {
            distinct_offsets_seeded(seed_for(id, DOMAIN), 32, &mut out).unwrap();
            for (i, &v) in out.iter().enumerate() {
                assert!(v < 32, "offset out of range: {v}");
                for &w in &out[..i] {
                    assert_ne!(v, w, "duplicate offset at check {id}");
                }
            }
        }
    }

    #[test]
    fn offsets_are_deterministic() {
        let mut a = [0u32; 4];
        let mut b = [0u32; 4];
        distinct_offsets_seeded(seed_for(42, DOMAIN), 64, &mut a).unwrap();
        distinct_offsets_seeded(seed_for(42, DOMAIN), 64, &mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn span_equal_to_k_selects_all() {
        let mut out = [0u32; 8];
        distinct_offsets_seeded(seed_for(7, DOMAIN), 8, &mut out).unwrap();
        out.sort_unstable();
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn zero_k_draws_nothing() {
        let mut rng = SplitMix64::new(1);
        let before = rng.clone();
        distinct_offsets(&mut rng, 8, &mut []).unwrap();
        assert_eq!(
            rng.0, before.0,
            "an empty request must not touch the stream"
        );
        // Zero offsets are satisfiable even from an empty domain.
        distinct_offsets(&mut rng, 0, &mut []).unwrap();
        assert_eq!(rng.0, before.0);
    }

    #[test]
    fn oversized_request_is_rejected() {
        let mut rng = SplitMix64::new(1);
        let mut out = [7u32; 3];
        assert_eq!(
            distinct_offsets(&mut rng, 2, &mut out),
            Err(GraphError::SampleSpanTooSmall {
                span: 2,
                requested: 3
            })
        );
        assert_eq!(out, [7; 3], "rejected call wrote output");
    }

    #[test]
    fn below_is_in_range() {
        let mut rng = SplitMix64::new(123);
        for bound in [1u32, 2, 3, 7, 255, 256, 1000] {
            for _ in 0..1000 {
                assert!(rng.below(nz(bound)) < bound);
            }
        }
    }

    /// A bound of one has a single valid answer and must not consume the
    /// rejection loop forever.
    #[test]
    fn below_one_is_always_zero() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..100 {
            assert_eq!(rng.below(NonZeroU32::MIN), 0);
        }
    }

    /// `below` must not be merely in-range: the rejection band exists to keep it
    /// unbiased. A bound that divides 2^32 badly is where bias would show.
    #[test]
    fn below_is_roughly_uniform() {
        const BOUND: u32 = 3;
        const DRAWS: u32 = 60_000;
        let mut counts = [0u32; BOUND as usize];
        let mut rng = SplitMix64::new(0xFEED_FACE);
        for _ in 0..DRAWS {
            counts[rng.below(nz(BOUND)) as usize] += 1;
        }
        let expected = DRAWS / BOUND;
        for (v, &c) in counts.iter().enumerate() {
            let dev = c.abs_diff(expected);
            assert!(
                dev * 50 < expected,
                "value {v} appeared {c} times, expected about {expected}"
            );
        }
    }

    /// The composition the generators rely on: consuming a degree from
    /// the same stream must shift the offsets, or degree and edges would be
    /// independent streams keyed on one seed.
    #[test]
    fn draws_before_sampling_shift_the_stream() {
        let mut plain = [0u32; 3];
        distinct_offsets_seeded(seed_for(9, DOMAIN), 64, &mut plain).unwrap();

        let mut shifted = [0u32; 3];
        let mut rng = SplitMix64::new(seed_for(9, DOMAIN));
        let _ = rng.next_u64();
        distinct_offsets(&mut rng, 64, &mut shifted).unwrap();

        assert_ne!(plain, shifted);
    }
}
