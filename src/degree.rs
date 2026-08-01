//! Check degree distributions.
//!
//! Degree is the one knob that separates a regular LDPC-style graph from an LT
//! code: a constant degree gives every check the same number of edges, while a
//! soliton or table-driven distribution spreads them so that peeling has a
//! ripple to work with.
//!
//! # Stream discipline
//!
//! [`DegreeDistribution::sample`] borrows the generator rather than taking a
//! seed, so a degree draw and the edge draw that follows it come from **one**
//! reproducible stream. Seeding a separate generator for the degree would leave
//! the two streams independent while looking correct, which is how correlated
//! graph bugs get in.
//!
//! That composition carries an obligation: a point-mass distribution MUST consume
//! zero state, or threading the degree draw ahead of the edge draw would shift
//! every offset and break wire compatibility with a constant-degree code. See the
//! charter's invariant 2, and the fixtures under `tests/data/`.

use crate::error::GraphError;
use crate::rng::SplitMix64;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::num::{NonZeroU32, NonZeroU64};

/// How many edges a check gets.
pub trait DegreeDistribution {
    /// Draw a degree.
    ///
    /// A point-mass distribution MUST leave `rng` untouched; see the module
    /// documentation for why that is load-bearing rather than an optimisation.
    fn sample(&self, rng: &mut SplitMix64) -> u32;

    /// Largest degree this distribution can ever return.
    ///
    /// Generators use it to size neighbour scratch once, and to reject a
    /// distribution that could ask for more distinct variables than the domain
    /// holds.
    fn max_degree(&self) -> u32;
}

/// Every check has exactly the same degree.
///
/// A point mass, so sampling consumes no randomness at all — which is what keeps
/// a constant-degree generator bit-compatible with a code that draws its offsets
/// from a freshly seeded generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Constant(u32);

impl Constant {
    /// A distribution fixed at `degree`.
    ///
    /// # Errors
    ///
    /// [`GraphError::ZeroDegree`] when `degree` is zero: a check with no support
    /// constrains nothing, so it is a configuration bug rather than a degenerate
    /// but valid graph.
    pub fn new(degree: u32) -> Result<Self, GraphError> {
        if degree == 0 {
            return Err(GraphError::ZeroDegree);
        }
        Ok(Self(degree))
    }

    /// The fixed degree.
    #[inline]
    #[must_use]
    pub fn degree(self) -> u32 {
        self.0
    }
}

impl DegreeDistribution for Constant {
    /// Consumes no generator state, by contract.
    #[inline]
    fn sample(&self, _rng: &mut SplitMix64) -> u32 {
        self.0
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.0
    }
}

/// A degree distribution given as an explicit cumulative weight table.
///
/// Entry `i` holds the summed weight of degrees `1..=i + 1`, so the last entry
/// is the total and degree `i + 1` carries `table[i] - table[i - 1]`. The
/// weights are bare integers rather than probabilities: only their ratios
/// matter, which is what lets a protocol transcribe its own published table
/// without first turning it into a fraction — and without a float appearing
/// anywhere. Two equal neighbouring entries are legal and mean the degree
/// between them is unreachable.
///
/// # Sampling
///
/// One value is reduced into `[0, total)` and the degree is one plus the number
/// of entries at or below it, located by binary search. The reduction is
/// unbiased on both of its branches:
///
/// * A total that fits in a [`u32`] uses [`SplitMix64::below`] — the crate's
///   32-bit Lemire multiply-shift, which rejects the biased low band.
/// * A wider total uses that same method widened to 64 bits: multiply a full
///   64-bit draw by the total into a 128-bit product, keep the high half, and
///   redraw while the low half is below `2^64 mod total`.
///
/// Which branch runs follows from the total alone, so it is part of the wire
/// format: rescaling a table across the [`u32::MAX`] boundary changes the
/// degrees a given generator state produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cumulative {
    /// Non-empty and non-decreasing, both established by [`Cumulative::new`].
    table: Vec<u64>,
    /// The last table entry, kept non-zero so sampling always has a bound.
    total: NonZeroU64,
}

impl Cumulative {
    /// Adopt a cumulative weight table.
    ///
    /// # Errors
    ///
    /// [`GraphError::EmptyCumulative`] for a table with no entries: it names no
    /// degree at all.
    ///
    /// [`GraphError::CumulativeNotMonotone`] when an entry falls below its
    /// predecessor. Sampling searches the table, so a decrease would quietly
    /// make a degree unreachable rather than fail.
    ///
    /// [`GraphError::ZeroDegree`] when the total weight is zero, since every
    /// degree then has probability zero and the distribution can only produce
    /// checks with no support.
    pub fn new(weights: Vec<u64>) -> Result<Self, GraphError> {
        if weights.is_empty() {
            return Err(GraphError::EmptyCumulative);
        }
        for (predecessor, pair) in weights.windows(2).enumerate() {
            let (previous, current) = (pair[0], pair[1]);
            if current < previous {
                return Err(GraphError::CumulativeNotMonotone {
                    index: predecessor + 1,
                    previous,
                    current,
                });
            }
        }
        let Some(total) = weights.last().copied().and_then(NonZeroU64::new) else {
            return Err(GraphError::ZeroDegree);
        };
        Ok(Self {
            table: weights,
            total,
        })
    }

    /// The cumulative weights, exactly as sampling reads them.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &[u64] {
        &self.table
    }

    /// Reduce one draw into `[0, total)`, unbiased on either width.
    fn draw(&self, rng: &mut SplitMix64) -> u64 {
        if let Ok(bound) = NonZeroU32::try_from(self.total) {
            return u64::from(rng.below(bound));
        }
        let total = self.total.get();
        let wide = u128::from(total);
        let mut product = u128::from(rng.next_u64()) * wide;
        let mut low = product as u64;
        if low < total {
            // `2^64 mod total` — the width of the low band that would be
            // over-represented if it were kept.
            let threshold = total.wrapping_neg() % total;
            while low < threshold {
                product = u128::from(rng.next_u64()) * wide;
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }
}

impl DegreeDistribution for Cumulative {
    fn sample(&self, rng: &mut SplitMix64) -> u32 {
        let draw = self.draw(rng);
        degree_at(self.table.partition_point(|&weight| weight <= draw))
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        degree_at(self.table.len() - 1)
    }
}

/// Luby's robust soliton distribution over `k` source symbols.
///
/// The ideal soliton distribution releases exactly one degree-one check in
/// expectation, so peeling stalls the moment that ripple dies. The robust
/// variant adds a `tau` term that lifts the low degrees and plants a spike near
/// `k/R`, trading a little overhead for a ripple that survives.
///
/// Construction is integer throughout: the parameters arrive as fixed-point
/// fractions, the logarithm and square root are fixed-point routines, and the
/// weights land in a cumulative `u64` table. Nothing rounds differently on
/// another target, so two peers agree on the degree of every check. The table
/// holds one entry per degree, so building it allocates `k` words.
///
/// # Parameters
///
/// `c_q32` and `delta_q32` are UQ0.32 fixed-point fractions: the value is the
/// numerator over `2^32`, so `1 << 31` is `0.5` and the representable range is
/// the open interval `(0, 1)`. `delta` is the decoding failure probability the
/// distribution is tuned for and `c` scales where the spike lands.
///
/// # Construction
///
/// Write `S` for the weight scale `2^48`, `Q(x)` for the UQ32.32 encoding of
/// `x` (that is `x * 2^32`, truncated), `ln` for the fixed-point logarithm
/// `ln_q32` and `sqrt` for the fixed-point root `sqrt_q32`. Every division
/// below truncates, and `>>` is a plain shift:
///
/// ```text
/// ln_delta = ln(delta_q32)                                  // Q(ln delta) < 0
/// R        = max(1, (c_q32 * (ln(Q(k)) - ln_delta) * sqrt(k)) >> 64)   // Q(R)
/// s        = ceil((k << 32) / R)                            // ceil(k / R)
/// ln_ratio = max(0, ln(R) - ln_delta)                       // Q(ln (R/delta))
/// spike    = (((R * ln_ratio) >> 32) * S >> 32) / k         // S * tau(s)
/// ramp     = (R * S) >> 32                                  // S * R
///
/// for i in 1..=k:
///     rho = if i == 1 { S / k } else { S / (i * (i - 1)) }
///     tau = if i < s { ramp / (i * k) } else if i == s { spike } else { 0 }
///     weight[i] = rho + tau
/// ```
///
/// The table is the running sum of `weight[1..=k]`. That is `S` times Luby's
/// `rho(1) = 1/k`, `rho(i) = 1/(i * (i - 1))`, `tau(i) = R/(i * k)` below the
/// spike, `tau(s) = R * ln(R/delta)/k` at it and zero above, with
/// `R = c * ln(k/delta) * sqrt(k)` and `s = ceil(k/R)`. The weights stay
/// unnormalised because sampling needs only their ratios: the total that
/// Luby's `beta` would divide out is the last entry of the cumulative table.
///
/// Two limits are resolved rather than left to chance. `R` is clamped up to one
/// unit in the last place, so a parameter set whose `R` truncates to zero puts
/// the spike past `k` and degenerates to the ideal soliton instead of dividing
/// by zero. A non-positive `ln(R/delta)`, which the spike formula would turn
/// into negative weight, contributes no spike at all.
///
/// `tau` is truncated at `k` and `rho` never exceeds it, so [`max_degree`] is
/// `k`.
///
/// [`max_degree`]: DegreeDistribution::max_degree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustSoliton {
    /// Source symbol count, which is also the largest degree.
    symbols: u32,
    /// The constructed table, holding the one sampling path.
    table: Cumulative,
}

impl RobustSoliton {
    /// Build the distribution from UQ0.32 parameters, as documented on the
    /// type.
    ///
    /// # Errors
    ///
    /// [`GraphError::ZeroSymbolCount`] when `k` is zero: there is no degree to
    /// distribute over.
    ///
    /// [`GraphError::ZeroSolitonC`] when `c_q32` is zero, which places no
    /// spike and leaves the construction degenerate.
    ///
    /// [`GraphError::SolitonDeltaOutOfRange`] when `delta_q32` is zero. `delta`
    /// lies in the open interval `(0, 1)`, and UQ0.32 already bounds it above,
    /// so `1..=u32::MAX` is exactly the representable range.
    ///
    /// [`GraphError::CumulativeOverflow`] if the accumulated weights ever pass
    /// [`u64::MAX`]. The scale leaves a table's total near `2^48` for every
    /// parameter set in range, so this guards the arithmetic rather than
    /// describing a reachable configuration.
    pub fn from_q32(k: u32, c_q32: u32, delta_q32: u32) -> Result<Self, GraphError> {
        if k == 0 {
            return Err(GraphError::ZeroSymbolCount);
        }
        if c_q32 == 0 {
            return Err(GraphError::ZeroSolitonC);
        }
        if delta_q32 == 0 {
            return Err(GraphError::SolitonDeltaOutOfRange { delta_q32 });
        }

        // `delta < 1` and `k >= 1`, so this difference is at least one ulp and
        // `R` is well defined without a sign to track.
        let ln_delta = ln_q32(u64::from(delta_q32));
        let ln_symbols = ln_q32(u64::from(k) << 32);
        let ln_ratio = ln_symbols.saturating_sub(ln_delta).max(0).unsigned_abs();
        // `c < 1`, `ln(k/delta) < 45` and `sqrt(k) <= 2^16`, so `R` stays under
        // `2^22` and the shifted product is nowhere near `u64::MAX`. Clamping up
        // to one ulp keeps `k/R` finite when the truncation would reach zero.
        let scaled = (u128::from(c_q32) * u128::from(ln_ratio) * u128::from(sqrt_q32(k))) >> 64;
        let r = (scaled as u64).max(1);

        let spike_index = (u128::from(k) << 32).div_ceil(u128::from(r));
        let ln_spike_ratio = ln_q32(r).saturating_sub(ln_delta).max(0).unsigned_abs();
        let spike = if ln_spike_ratio == 0 {
            0
        } else {
            let product = (u128::from(r) * u128::from(ln_spike_ratio)) >> 32;
            ((product * u128::from(WEIGHT_SCALE)) >> 32) / u128::from(k)
        };
        let ramp = (u128::from(r) * u128::from(WEIGHT_SCALE)) >> 32;

        let table = cumulate(
            k as usize,
            (1..=u64::from(k)).map(|degree| {
                let rho = if degree == 1 {
                    WEIGHT_SCALE / u64::from(k)
                } else {
                    WEIGHT_SCALE / (degree * (degree - 1))
                };
                let wide = u128::from(degree);
                let tau = match wide.cmp(&spike_index) {
                    Ordering::Less => ramp / (wide * u128::from(k)),
                    Ordering::Equal => spike,
                    Ordering::Greater => 0,
                };
                u128::from(rho) + tau
            }),
        )?;

        Ok(Self {
            symbols: k,
            table: Cumulative::new(table)?,
        })
    }

    /// The constructed cumulative weight table.
    ///
    /// Entry `i` is the summed weight of degrees `1..=i + 1`. Exposed so a
    /// consumer can fingerprint the distribution it is about to put on the
    /// wire, since the table — not the parameters — is what both peers must
    /// agree on.
    #[inline]
    #[must_use]
    pub fn cumulative(&self) -> &[u64] {
        self.table.weights()
    }
}

impl DegreeDistribution for RobustSoliton {
    /// Samples through the shared table path, so there is one sampler to pin.
    #[inline]
    fn sample(&self, rng: &mut SplitMix64) -> u32 {
        self.table.sample(rng)
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.symbols
    }
}

/// Weight scale of a constructed distribution: a weight is `2^48` times the
/// probability-like term it stands for.
///
/// Large enough that the ideal soliton's `1/(i * (i - 1))` stays non-zero out
/// to `i` near `2^24`, and small enough that a whole table sums nowhere near
/// [`u64::MAX`].
const WEIGHT_SCALE: u64 = 1 << 48;

/// `ln(2)` as a UQ0.64 fraction: `floor(ln(2) * 2^64)`.
const LN2_Q64: u64 = 0xB172_17F7_D1CF_79AB;

/// Terms of the series `ln_q32` sums.
///
/// Its argument is at most `1/3`, so term `j` is under `3^-(2j + 1)`; sixteen
/// terms leave the tail below `2^-54`, far under the `2^-32` the result is
/// truncated to.
const LN_TERMS: u32 = 16;

/// Prefix-sum per-degree weights into a cumulative table.
///
/// # Errors
///
/// [`GraphError::CumulativeOverflow`] at the first index whose running total
/// passes [`u64::MAX`]. The accumulator is wider than the table it fills, so
/// the overflow is detected rather than wrapped or saturated into a plausible
/// looking weight.
fn cumulate<I>(len: usize, weights: I) -> Result<Vec<u64>, GraphError>
where
    I: IntoIterator<Item = u128>,
{
    let mut table = Vec::with_capacity(len);
    let mut running: u128 = 0;
    for (index, weight) in weights.into_iter().enumerate() {
        running = running.saturating_add(weight);
        if running > u128::from(u64::MAX) {
            return Err(GraphError::CumulativeOverflow { index });
        }
        table.push(running as u64);
    }
    Ok(table)
}

/// Largest degree a table position can name.
///
/// Degrees are `u32`, so a table with more than [`u32::MAX`] entries cannot
/// name its tail. Such a table needs 32 GiB of weights, so the clamp bounds a
/// case that cannot arise rather than papering over one that can.
#[inline]
fn degree_at(index: usize) -> u32 {
    match u32::try_from(index) {
        Ok(index) => index.saturating_add(1),
        Err(_) => u32::MAX,
    }
}

/// Square root of `k` as a UQ32.32 fixed-point value.
///
/// Exactly `floor(sqrt(k) * 2^32)`, computed as the integer square root of
/// `k << 64` so the scaling happens before the root rather than after it. The
/// widest input, [`u32::MAX`], yields just under `2^48`.
#[inline]
fn sqrt_q32(k: u32) -> u64 {
    (u128::from(k) << 64).isqrt() as u64
}

/// Natural logarithm of a UQ32.32 fixed-point value, in UQ32.32.
///
/// `x_q32` stands for `x_q32 / 2^32` and the result stands for
/// `ln(x) * 2^32`, signed because an `x` below one has a negative logarithm.
/// Integer throughout: `x` splits as `m * 2^e` with the mantissa `m` in
/// `[1, 2)`, and `ln(x) = e * ln(2) + 2 * atanh(z)` with `z = (m - 1)/(m + 1)`,
/// which is at most `1/3` and is summed from its odd-power series. The
/// intermediate arithmetic carries 64 fractional bits, so the result differs
/// from `floor(ln(x) * 2^32)` by at most one.
///
/// Zero has no logarithm and yields [`i64::MIN`]. Every caller validates its
/// argument first, so no construction reaches that.
fn ln_q32(x_q32: u64) -> i64 {
    let Some(msb) = x_q32.checked_ilog2() else {
        return i64::MIN;
    };
    let exponent = i128::from(msb) - 32;
    let one = 1u128 << 64;
    // The mantissa in UQ1.64: `x` scaled so its leading bit sits at bit 64.
    let mantissa = u128::from(x_q32) << (64 - msb);
    let z = (((mantissa - one) << 64) / (mantissa + one)) as u64;
    let z_squared = ((u128::from(z) * u128::from(z)) >> 64) as u64;

    let mut term = z;
    let mut sum: u64 = 0;
    let mut odd: u64 = 1;
    for _ in 0..LN_TERMS {
        sum += term / odd;
        term = ((u128::from(term) * u128::from(z_squared)) >> 64) as u64;
        odd += 2;
    }

    let ln_q64 = exponent * i128::from(LN2_Q64) + i128::from(2 * sum);
    (ln_q64 >> 32) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn constant_rejects_zero() {
        assert_eq!(Constant::new(0), Err(GraphError::ZeroDegree));
        assert_eq!(Constant::new(1).unwrap().degree(), 1);
    }

    #[test]
    fn constant_always_returns_its_degree() {
        let d = Constant::new(7).unwrap();
        let mut rng = SplitMix64::new(12345);
        for _ in 0..100 {
            assert_eq!(d.sample(&mut rng), 7);
        }
        assert_eq!(d.max_degree(), 7);
    }

    /// Charter invariant 2, checked against generator state rather than against
    /// the sampled value. Only the state check catches a future point-mass
    /// distribution that draws and discards.
    #[test]
    fn constant_consumes_no_generator_state() {
        let d = Constant::new(3).unwrap();
        let mut rng = SplitMix64::new(0xABCD_1234);
        let before = rng.clone();

        for _ in 0..10 {
            let _ = d.sample(&mut rng);
        }

        // Identical state means identical futures.
        let mut after = rng;
        let mut untouched = before;
        for _ in 0..8 {
            assert_eq!(
                after.next_u64(),
                untouched.next_u64(),
                "sampling a point mass advanced the stream"
            );
        }
    }

    /// `sqrt_q32` is exact, so the bracketing identity `v^2 <= x < (v + 1)^2`
    /// holds for every input; the literals are values that identity pins.
    #[test]
    fn sqrt_q32_is_the_exact_integer_root() {
        for k in [1u32, 2, 3, 4, 8, 9, 10, 1000, 65_536, 65_537, u32::MAX] {
            let root = u128::from(sqrt_q32(k));
            let square = u128::from(k) << 64;
            assert!(root * root <= square, "root too large for k={k}");
            assert!(square < (root + 1) * (root + 1), "root too small for k={k}");
        }

        // Perfect squares land exactly on a power of two.
        assert_eq!(sqrt_q32(1), 1 << 32);
        assert_eq!(sqrt_q32(4), 1 << 33);
        assert_eq!(sqrt_q32(65_536), 1 << 40);
        // sqrt(2) = 1.41421356237309504880..., sqrt(8) = 2.82842712474619009760...
        assert_eq!(sqrt_q32(2), 6_074_000_999);
        assert_eq!(sqrt_q32(8), 12_148_001_999);
        assert_eq!(sqrt_q32(u32::MAX), 281_474_976_677_887);
    }

    /// Decimal expansions of `ln(x) * 2^32`, transcribed to their floors:
    ///
    /// | `x`                | `ln(x) * 2^32`             |
    /// |--------------------|----------------------------|
    /// | `2^-32`            | `-95265423098.22630628...` |
    /// | `2^-16`            | `-47632711549.11315314...` |
    /// | `0.5`              | `-2977044471.819572071...` |
    /// | `1 - 2^-32`        | `-1.000000000116415321...` |
    /// | `1`                | `0`                        |
    /// | `1 + 2^-32`        | `0.9999999998835846781...` |
    /// | `2`                | `2977044471.819572071...`  |
    /// | `3`                | `4718503850.813242522...`  |
    /// | `floor(e * 2^32)`  | `4294967295.800358684...`  |
    /// | `100`              | `19779055341.33308987...`  |
    /// | `65536`            | `47632711549.11315314...`  |
    /// | `2^32 - 2^-32`     | `95265423098.22630628...`  |
    #[test]
    fn ln_q32_matches_decimal_constants() {
        assert_eq!(ln_q32(1), -95_265_423_099);
        assert_eq!(ln_q32(1 << 16), -47_632_711_550);
        assert_eq!(ln_q32(1 << 31), -2_977_044_472);
        assert_eq!(ln_q32((1 << 32) - 1), -2);
        assert_eq!(ln_q32(1 << 32), 0);
        assert_eq!(ln_q32((1 << 32) + 1), 0);
        assert_eq!(ln_q32(2 << 32), 2_977_044_471);
        assert_eq!(ln_q32(3 << 32), 4_718_503_850);
        assert_eq!(ln_q32(11_674_931_554), 4_294_967_295);
        assert_eq!(ln_q32(100 << 32), 19_779_055_341);
        assert_eq!(ln_q32(1 << 48), 47_632_711_549);
        assert_eq!(ln_q32(u64::MAX), 95_265_423_098);
    }

    /// `ln` of a power of two is that power times `ln(2)`, exactly: the series
    /// contributes nothing for a mantissa of one, so only the exponent term is
    /// left and it must not drift as the exponent grows.
    #[test]
    fn ln_q32_scales_with_the_exponent() {
        for power in 0..32u64 {
            let expected = i64::try_from((u128::from(power) * u128::from(LN2_Q64)) >> 32).unwrap();
            assert_eq!(ln_q32(1u64 << (32 + power)), expected, "ln(2^{power})");
        }
        assert_eq!(ln_q32(0), i64::MIN);
    }

    #[test]
    fn cumulative_rejects_a_table_it_cannot_sample() {
        assert_eq!(
            Cumulative::new(Vec::new()),
            Err(GraphError::EmptyCumulative)
        );
        assert_eq!(
            Cumulative::new(vec![3, 9, 4, 20]),
            Err(GraphError::CumulativeNotMonotone {
                index: 2,
                previous: 9,
                current: 4,
            })
        );
        assert_eq!(Cumulative::new(vec![0, 0, 0]), Err(GraphError::ZeroDegree));
        // Only the total matters for emptiness: a leading plateau is legal.
        assert_eq!(
            Cumulative::new(vec![0, 0, 1]).unwrap().weights(),
            &[0, 0, 1]
        );
    }

    #[test]
    fn cumulate_reports_the_index_that_overflowed() {
        assert_eq!(
            cumulate(2, [u128::from(u64::MAX), 1]),
            Err(GraphError::CumulativeOverflow { index: 1 })
        );
        assert_eq!(
            cumulate(3, [1u128, 2, u128::from(u64::MAX)]),
            Err(GraphError::CumulativeOverflow { index: 2 })
        );
        // The bound is inclusive: a total of exactly `u64::MAX` still fits.
        assert_eq!(
            cumulate(2, [u128::from(u64::MAX), 0]),
            Ok(vec![u64::MAX, u64::MAX])
        );
    }

    /// Degrees `1, 2, 3, 4` carry weights `1, 3, 0, 6` out of ten. The total
    /// fits in a `u32`, so this pins the 32-bit reduction branch.
    #[test]
    fn cumulative_samples_a_narrow_table() {
        let table = Cumulative::new(vec![1, 4, 4, 10]).unwrap();
        assert_eq!(table.max_degree(), 4);

        let mut rng = SplitMix64::new(7);
        let drawn: Vec<u32> = (0..24).map(|_| table.sample(&mut rng)).collect();
        assert_eq!(
            drawn,
            vec![
                2, 4, 4, 4, 4, 4, 2, 2, 4, 4, 1, 2, 2, 4, 1, 2, 4, 4, 2, 2, 1, 2, 4, 2
            ]
        );

        // A zero-weight degree is unreachable, not merely unlikely.
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            assert_ne!(table.sample(&mut rng), 3);
        }
    }

    /// The same ratios scaled past `u32::MAX` take the widened reduction, which
    /// is a different sampler and therefore a different sequence.
    #[test]
    fn cumulative_samples_a_wide_table() {
        let table = Cumulative::new(vec![1 << 40, 4 << 40, 4 << 40, 10 << 40]).unwrap();

        let mut rng = SplitMix64::new(7);
        let drawn: Vec<u32> = (0..24).map(|_| table.sample(&mut rng)).collect();
        assert_eq!(
            drawn,
            vec![
                2, 1, 4, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2, 2, 4
            ]
        );
    }

    #[test]
    fn robust_soliton_rejects_unusable_parameters() {
        assert_eq!(
            RobustSoliton::from_q32(0, 1 << 31, 1 << 31),
            Err(GraphError::ZeroSymbolCount)
        );
        assert_eq!(
            RobustSoliton::from_q32(8, 0, 1 << 31),
            Err(GraphError::ZeroSolitonC)
        );
        assert_eq!(
            RobustSoliton::from_q32(8, 1 << 31, 0),
            Err(GraphError::SolitonDeltaOutOfRange { delta_q32: 0 })
        );
    }

    /// `k = 8`, `c = 0.5`, `delta = 0.5`, worked through the documented
    /// construction with `S = 2^48`:
    ///
    /// `R = 16840706669` (`3.9210325733...`), so `s = ceil(8/R) = 3` and the
    /// spike weight is `284127007719424`. Below it `tau(i) = ramp/(8i)` with
    /// `ramp = 1103672552259584`; above it `tau` is zero.
    ///
    /// | `i` | `rho(i) * S`      | `tau(i) * S`      | sum               |
    /// |-----|-------------------|-------------------|-------------------|
    /// | 1   | `35184372088832`  | `137959069032448` | `173143441121280` |
    /// | 2   | `140737488355328` | `68979534516224`  | `209717022871552` |
    /// | 3   | `46912496118442`  | `284127007719424` | `331039503837866` |
    /// | 4   | `23456248059221`  | `0`               | `23456248059221`  |
    /// | 5   | `14073748835532`  | `0`               | `14073748835532`  |
    /// | 6   | `9382499223688`   | `0`               | `9382499223688`   |
    /// | 7   | `6701785159777`   | `0`               | `6701785159777`   |
    /// | 8   | `5026338869833`   | `0`               | `5026338869833`   |
    #[test]
    fn robust_soliton_builds_the_documented_table() {
        let soliton = RobustSoliton::from_q32(8, 1 << 31, 1 << 31).unwrap();
        assert_eq!(
            soliton.cumulative(),
            &[
                173_143_441_121_280,
                382_860_463_992_832,
                713_899_967_830_698,
                737_356_215_889_919,
                751_429_964_725_451,
                760_812_463_949_139,
                767_514_249_108_916,
                772_540_587_978_749,
            ]
        );
        assert_eq!(soliton.max_degree(), 8);

        // The ideal soliton part alone, so a `tau` regression cannot hide in the
        // running sum: `S/k` at one and `S/(i * (i - 1))` above it.
        let table = soliton.cumulative();
        let ideal = [
            WEIGHT_SCALE / 8,
            WEIGHT_SCALE / 2,
            WEIGHT_SCALE / 6,
            WEIGHT_SCALE / 12,
            WEIGHT_SCALE / 20,
            WEIGHT_SCALE / 30,
            WEIGHT_SCALE / 42,
            WEIGHT_SCALE / 56,
        ];
        for degree in 4..8 {
            assert_eq!(table[degree] - table[degree - 1], ideal[degree]);
        }
    }

    /// `R` truncates to five ulps here, so `s = 6871947674` sits far past `k`:
    /// no spike, and `tau(i) = 40960/i` on top of the ideal soliton.
    #[test]
    fn robust_soliton_without_a_spike_is_the_ideal_soliton_plus_a_ramp() {
        let soliton = RobustSoliton::from_q32(8, 1, u32::MAX).unwrap();
        assert_eq!(
            soliton.cumulative(),
            &[
                35_184_372_129_792,
                175_921_860_505_600,
                222_834_356_637_695,
                246_290_604_707_156,
                260_364_353_550_880,
                269_746_852_781_394,
                276_448_637_947_022,
                281_474_976_821_975,
            ]
        );

        // Weight of degree `i`, minus the ideal soliton, is the ramp.
        let table = soliton.cumulative();
        assert_eq!(table[0] - WEIGHT_SCALE / 8, 40_960);
        for degree in 2..=8u64 {
            let index = degree as usize - 1;
            let weight = table[index] - table[index - 1];
            let ideal = WEIGHT_SCALE / (degree * (degree - 1));
            assert_eq!(weight - ideal, 40_960 / degree);
        }
    }

    /// `R = 24.953...` exceeds `k`, so `s = 1` and the spike lands on degree
    /// one with no ramp below it at all.
    #[test]
    fn robust_soliton_spike_can_land_on_degree_one() {
        let soliton = RobustSoliton::from_q32(4, u32::MAX, 1 << 16).unwrap();
        assert_eq!(
            soliton.cumulative(),
            &[
                25_193_125_575_180_288,
                25_333_863_063_535_616,
                25_380_775_559_654_058,
                25_404_231_807_713_279,
            ]
        );
        // Degrees above the spike carry the ideal soliton and nothing else.
        let table = soliton.cumulative();
        assert_eq!(table[1] - table[0], WEIGHT_SCALE / 2);
        assert_eq!(table[2] - table[1], WEIGHT_SCALE / 6);
        assert_eq!(table[3] - table[2], WEIGHT_SCALE / 12);
    }

    /// A single symbol leaves one degree, so sampling is a point mass — but one
    /// that still consumes a draw, since the table path always reduces one.
    #[test]
    fn robust_soliton_of_one_symbol_has_one_degree() {
        let soliton = RobustSoliton::from_q32(1, 1, u32::MAX).unwrap();
        assert_eq!(soliton.cumulative(), &[281_474_976_776_192]);
        assert_eq!(soliton.max_degree(), 1);

        let mut rng = SplitMix64::new(99);
        for _ in 0..64 {
            assert_eq!(soliton.sample(&mut rng), 1);
        }
    }

    /// The cross-platform fingerprint: integer construction and integer
    /// sampling must produce this sequence on every target.
    #[test]
    fn robust_soliton_sample_sequence_is_fixed() {
        let soliton = RobustSoliton::from_q32(8, 1 << 31, 1 << 31).unwrap();
        let mut rng = SplitMix64::new(0x5EED);
        let drawn: Vec<u32> = (0..24).map(|_| soliton.sample(&mut rng)).collect();
        assert_eq!(
            drawn,
            vec![
                1, 2, 2, 2, 1, 2, 1, 1, 3, 2, 2, 2, 1, 1, 3, 3, 5, 1, 2, 3, 3, 2, 3, 3
            ]
        );
    }

    /// Sampled frequencies must track the table's own ratios. The deviation is
    /// deterministic for a fixed seed; the largest one observed is 118 draws,
    /// so 200 is a bound the sampler passes and a biased one would not.
    #[test]
    fn robust_soliton_frequencies_track_the_table() {
        const DRAWS: u32 = 100_000;
        const TOLERANCE: i64 = 200;

        let soliton = RobustSoliton::from_q32(8, 1 << 31, 1 << 31).unwrap();
        let mut counts = [0i64; 8];
        let mut rng = SplitMix64::new(0x00C0_FFEE);
        for _ in 0..DRAWS {
            counts[soliton.sample(&mut rng) as usize - 1] += 1;
        }

        let table = soliton.cumulative();
        let total = u128::from(table[7]);
        let mut previous = 0;
        for (index, &entry) in table.iter().enumerate() {
            let weight = u128::from(entry - previous);
            previous = entry;
            let expected = i64::try_from(u128::from(DRAWS) * weight / total).unwrap();
            let deviation = counts[index] - expected;
            assert!(
                deviation.abs() <= TOLERANCE,
                "degree {} drawn {} times, expected about {expected}",
                index + 1,
                counts[index]
            );
        }
        assert_eq!(counts.iter().sum::<i64>(), i64::from(DRAWS));
    }
}
