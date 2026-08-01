//! Dense index-keyed storage over a monotone `u64` domain.
//!
//! Graph state is keyed by absolute variable and check index, and those indices
//! are dense and monotone. A hash map per lookup is wasted work: [`Ring`] holds
//! one slot per live index, so a lookup is a subtraction and a bounds check, and
//! retirement is a front drain that hands back the values it dropped so their
//! buffers can be recycled. [`IndexSet`] does the same for a set of indices, as a
//! bitmap.
//!
//! # Bounded by construction
//!
//! Dense storage over a caller-supplied index is an unbounded allocation waiting
//! to happen: one far-away id would demand a gap-sized slab. Both structures
//! therefore take a maximum live span at construction, and every growth path
//! checks it — along with the `u64`→`usize` conversion, which is not free on a
//! 32-bit target. Exceeding either returns [`GraphError`] and leaves the
//! structure **exactly** as it was. A limit rejects input; it never evicts state
//! to make room.
//!
//! # Retired is not vacant
//!
//! Three states are distinct, and collapsing them loses data:
//!
//! * **Live** — has a slot, or is a member.
//! * **Vacant** — no slot, but could have one. An index below `base` that was
//!   never retired is vacant, and storing it grows the front.
//! * **Retired** — below the retirement horizon. Gone forever; storing it is an
//!   error, not a silent no-op.
//!
//! [`Lookup`] and [`Membership`] report which, and [`Ring::floor`] /
//! [`IndexSet::floor`] give the horizon that separates the last two.

use crate::error::GraphError;
use alloc::collections::VecDeque;
use alloc::collections::vec_deque;
use core::num::NonZeroUsize;

/// State of one index in a [`Ring`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup<T> {
    /// The index has a slot.
    Live(T),
    /// The index has no slot but is not retired.
    Vacant,
    /// The index is below the retirement horizon and cannot return.
    Retired,
}

impl<T> Lookup<T> {
    /// The slot, discarding why there isn't one.
    #[inline]
    pub fn live(self) -> Option<T> {
        match self {
            Self::Live(v) => Some(v),
            Self::Vacant | Self::Retired => None,
        }
    }

    /// True when the index is retired.
    #[inline]
    #[must_use]
    pub fn is_retired(&self) -> bool {
        matches!(self, Self::Retired)
    }
}

/// State of one index in an [`IndexSet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Membership {
    /// The index is in the set.
    Member,
    /// The index is not in the set but could be added.
    Vacant,
    /// The index is below the retirement horizon and cannot return.
    Retired,
}

/// Checked span arithmetic shared by both structures.
///
/// Returns the live span that covering `[lo, hi_inclusive]` requires, as a
/// `usize`, or the reason it cannot be represented or allowed.
fn checked_span(
    index: u64,
    lo: u64,
    hi_inclusive: u64,
    limit: NonZeroUsize,
) -> Result<usize, GraphError> {
    // `hi_inclusive >= lo`, so this cannot underflow, and the inclusive form
    // means `+ 1` is the only overflow risk — which `checked_add` catches at
    // `u64::MAX`.
    let required = (hi_inclusive - lo)
        .checked_add(1)
        .ok_or(GraphError::IndexNotRepresentable { index })?;
    let required_usize =
        usize::try_from(required).map_err(|_| GraphError::IndexNotRepresentable { index })?;
    if required_usize > limit.get() {
        return Err(GraphError::LiveSpanExceeded {
            index,
            required,
            limit: limit.get(),
        });
    }
    Ok(required_usize)
}

/// Slots for the dense index range `[base, base + len)`.
///
/// Grows on demand in either direction with `T::default()` and shrinks only on
/// [`retire_below`](Ring::retire_below). This is not a fixed-capacity circular
/// buffer: capacity follows the caller's retirement, bounded by the live-span
/// limit given to [`new`](Ring::new).
#[derive(Debug)]
pub struct Ring<T> {
    base: u64,
    /// Highest horizon ever retired. Retained so an emptied ring cannot
    /// re-anchor below it and silently resurrect a retired index.
    floor: u64,
    limit: NonZeroUsize,
    items: VecDeque<T>,
}

impl<T: Default> Ring<T> {
    /// An empty ring, anchored by its first insert.
    ///
    /// `max_span` bounds the live index range, so no single id can request a
    /// gap-sized allocation.
    #[must_use]
    pub fn new(max_span: NonZeroUsize) -> Self {
        Self {
            base: 0,
            floor: 0,
            limit: max_span,
            items: VecDeque::new(),
        }
    }

    /// Lowest index that still has a slot.
    #[inline]
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// Highest index that has a slot, or `None` when empty.
    ///
    /// Inclusive, so it is well defined even when the ring reaches `u64::MAX` —
    /// an exclusive end would not be.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Option<u64> {
        let len = self.items.len() as u64;
        if len == 0 {
            None
        } else {
            Some(self.base + (len - 1))
        }
    }

    /// Highest horizon ever passed to [`retire_below`](Ring::retire_below).
    ///
    /// No index below this can hold a slot again.
    #[inline]
    #[must_use]
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// Configured maximum live span.
    #[inline]
    #[must_use]
    pub fn max_span(&self) -> NonZeroUsize {
        self.limit
    }

    /// Number of live slots.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when no index has a slot.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    fn offset(&self, index: u64) -> Option<usize> {
        if index < self.base {
            return None;
        }
        let off = index - self.base;
        if off < self.items.len() as u64 {
            Some(off as usize)
        } else {
            None
        }
    }

    /// State of `index`, and its slot when it has one.
    #[inline]
    #[must_use]
    pub fn get(&self, index: u64) -> Lookup<&T> {
        match self.offset(index) {
            Some(o) => Lookup::Live(&self.items[o]),
            None if index < self.floor => Lookup::Retired,
            None => Lookup::Vacant,
        }
    }

    /// State of `index`, and its slot mutably when it has one.
    #[inline]
    pub fn get_mut(&mut self, index: u64) -> Lookup<&mut T> {
        match self.offset(index) {
            Some(o) => Lookup::Live(&mut self.items[o]),
            None if index < self.floor => Lookup::Retired,
            None => Lookup::Vacant,
        }
    }

    /// True when `index` is below the retirement horizon.
    #[inline]
    #[must_use]
    pub fn is_retired(&self, index: u64) -> bool {
        index < self.floor
    }

    /// Validate that the ring can cover an inclusive range without mutation.
    pub(crate) fn check_range(&self, first: u64, last: u64) -> Result<(), GraphError> {
        debug_assert!(first <= last, "check_range: inverted range");
        if first < self.floor {
            return Err(GraphError::IndexRetired {
                index: first,
                floor: self.floor,
            });
        }
        let (lo, hi) = match self.last() {
            Some(live_last) => (self.base.min(first), live_last.max(last)),
            None => (first, last),
        };
        checked_span(last, lo, hi, self.limit).map(|_| ())
    }

    /// Slot for `index`, growing the ring with defaults as needed.
    ///
    /// Grows in either direction: a vacant index below `base` gets a slot.
    ///
    /// # Errors
    ///
    /// [`GraphError::IndexRetired`] when `index` is below
    /// [`floor`](Ring::floor); [`GraphError::LiveSpanExceeded`] when covering it
    /// would pass the configured maximum span;
    /// [`GraphError::IndexNotRepresentable`] when the span does not fit `usize`.
    /// On any error the ring is unchanged.
    pub fn ensure(&mut self, index: u64) -> Result<&mut T, GraphError> {
        self.check_range(index, index)?;

        if self.items.is_empty() {
            // Validate before mutating: a rejected id must leave no trace.
            checked_span(index, index, index, self.limit)?;
            self.base = index;
            self.items.push_back(T::default());
            return Ok(&mut self.items[0]);
        }

        if index < self.base {
            let extra = (self.base - index) as usize;
            self.items.reserve(extra);
            for _ in 0..extra {
                self.items.push_front(T::default());
            }
            self.base = index;
        }
        // Offset form: `base + len` would overflow for a ring reaching u64::MAX.
        let target = index - self.base;
        while self.items.len() as u64 <= target {
            self.items.push_back(T::default());
        }
        let o = (index - self.base) as usize;
        Ok(&mut self.items[o])
    }

    /// Drop every slot below `horizon`, yielding the dropped values so the caller
    /// can recycle their buffers.
    ///
    /// Advances [`floor`](Ring::floor) to `horizon`. Dropping the returned
    /// iterator without consuming it still retires the slots.
    pub fn retire_below(&mut self, horizon: u64) -> Drain<'_, T> {
        self.floor = self.floor.max(horizon);
        let n = horizon
            .saturating_sub(self.base)
            .min(self.items.len() as u64) as usize;
        self.base += n as u64;
        if self.items.len() == n {
            // Fully drained: let the next insert re-anchor rather than growing a
            // run of empty slots up to it.
            self.base = self.base.max(horizon);
        }
        Drain {
            inner: self.items.drain(..n),
        }
    }

    /// Live slots, oldest first, paired with their absolute index.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &T)> {
        let base = self.base;
        self.items
            .iter()
            .enumerate()
            .map(move |(o, v)| (base + o as u64, v))
    }

    /// Live slots, oldest first, mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut T)> {
        let base = self.base;
        self.items
            .iter_mut()
            .enumerate()
            .map(move |(o, v)| (base + o as u64, v))
    }
}

/// Values dropped by [`Ring::retire_below`], oldest first.
#[derive(Debug)]
pub struct Drain<'a, T> {
    inner: vec_deque::Drain<'a, T>,
}

impl<T> Iterator for Drain<'_, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> ExactSizeIterator for Drain<'_, T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Bits per word of an [`IndexSet`].
const BITS: u64 = 64;

/// A set of dense `u64` indices, as a bitmap over `[base, base + bits)`.
///
/// Membership, insertion, and removal are bit operations, `len` is maintained
/// rather than counted, and iteration is a word scan that skips empty words.
#[derive(Debug)]
pub struct IndexSet {
    base: u64,
    /// Highest horizon ever retired. Load-bearing here: retirement leaves `base`
    /// word-aligned *below* the horizon, so without this an insert into the
    /// masked-off remainder of the front word would resurrect a retired index.
    floor: u64,
    limit: NonZeroUsize,
    words: VecDeque<u64>,
    len: usize,
}

impl IndexSet {
    /// An empty set, anchored by its first insert.
    ///
    /// `max_span` bounds the live index range, as for [`Ring::new`]. It counts
    /// indices, not words.
    #[must_use]
    pub fn new(max_span: NonZeroUsize) -> Self {
        Self {
            base: 0,
            floor: 0,
            limit: max_span,
            words: VecDeque::new(),
            len: 0,
        }
    }

    /// Number of members.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the set has no members.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Highest horizon ever passed to
    /// [`retire_below`](IndexSet::retire_below). No index below this can become a
    /// member again.
    #[inline]
    #[must_use]
    pub fn floor(&self) -> u64 {
        self.floor
    }

    /// Configured maximum live span.
    #[inline]
    #[must_use]
    pub fn max_span(&self) -> NonZeroUsize {
        self.limit
    }

    /// Number of indices the bitmap currently spans.
    #[inline]
    fn spanned(&self) -> u64 {
        self.words.len() as u64 * BITS
    }

    /// Offset of `index` from `base`, when the bitmap covers it.
    #[inline]
    fn offset(&self, index: u64) -> Option<u64> {
        if index < self.base {
            return None;
        }
        let off = index - self.base;
        if off < self.spanned() {
            Some(off)
        } else {
            None
        }
    }

    /// Insert `index`, reporting whether it became a new member.
    ///
    /// Grows in either direction: a vacant index below `base` is accepted.
    ///
    /// # Errors
    ///
    /// As [`Ring::ensure`]. On any error the set is unchanged.
    pub fn insert(&mut self, index: u64) -> Result<bool, GraphError> {
        if index < self.floor {
            return Err(GraphError::IndexRetired {
                index,
                floor: self.floor,
            });
        }
        // Anchor on a word boundary so retirement can drop whole words.
        let aligned = index - index % BITS;

        if self.words.is_empty() {
            checked_span(index, aligned, index, self.limit)?;
            self.base = aligned;
        } else {
            let live_last = self.base + (self.spanned() - 1);
            let lo = self.base.min(aligned);
            let hi = live_last.max(index);
            checked_span(index, lo, hi, self.limit)?;

            if aligned < self.base {
                let extra = ((self.base - aligned) / BITS) as usize;
                self.words.reserve(extra);
                for _ in 0..extra {
                    self.words.push_front(0);
                }
                self.base = aligned;
            }
        }

        // Offset form: `base + spanned` would overflow for a set reaching u64::MAX.
        let off = index - self.base;
        while self.spanned() <= off {
            self.words.push_back(0);
        }
        let word = &mut self.words[(off / BITS) as usize];
        let bit = 1u64 << (off % BITS);
        if *word & bit != 0 {
            return Ok(false);
        }
        *word |= bit;
        self.len += 1;
        Ok(true)
    }

    /// Remove `index`, reporting whether it was present.
    pub fn remove(&mut self, index: u64) -> bool {
        let Some(off) = self.offset(index) else {
            return false;
        };
        let word = &mut self.words[(off / BITS) as usize];
        let bit = 1u64 << (off % BITS);
        if *word & bit == 0 {
            return false;
        }
        *word &= !bit;
        self.len -= 1;
        true
    }

    /// True when `index` is a member.
    #[inline]
    #[must_use]
    pub fn contains(&self, index: u64) -> bool {
        match self.offset(index) {
            Some(off) => self.words[(off / BITS) as usize] & (1u64 << (off % BITS)) != 0,
            None => false,
        }
    }

    /// State of `index`: a member, vacant, or retired.
    #[inline]
    #[must_use]
    pub fn status(&self, index: u64) -> Membership {
        if self.contains(index) {
            Membership::Member
        } else if index < self.floor {
            Membership::Retired
        } else {
            Membership::Vacant
        }
    }

    /// True when `index` is below the retirement horizon.
    #[inline]
    #[must_use]
    pub fn is_retired(&self, index: u64) -> bool {
        index < self.floor
    }

    /// Members in `[lo, hi)`, ascending.
    ///
    /// Seeks to the first word overlapping `lo`, so this costs one pass over the
    /// words spanning the request rather than over the whole set.
    pub fn range(&self, lo: u64, hi: u64) -> impl Iterator<Item = u64> + '_ {
        // Work in offsets from `base` throughout: `base + word * BITS` would
        // overflow for a set reaching `u64::MAX`.
        let spanned = self.spanned();
        let lo_off = lo.saturating_sub(self.base).min(spanned);
        let hi_off = hi.saturating_sub(self.base).min(spanned);
        let base = self.base;

        let words = if lo <= self.base && hi <= self.base || lo_off >= hi_off {
            0..0
        } else {
            let first = (lo_off / BITS) as usize;
            let last = ((hi_off - 1) / BITS) as usize;
            first..last + 1
        };
        let first = words.start;

        self.words
            .range(words)
            .enumerate()
            .flat_map(move |(i, &word)| {
                let word_off = (first + i) as u64 * BITS;
                let mut masked = word;
                // Only the first and last words of the span can straddle the
                // bounds, so each shift below is in `1..BITS`.
                if word_off < lo_off {
                    masked &= u64::MAX << (lo_off - word_off);
                }
                if word_off + BITS > hi_off {
                    masked &= (1u64 << (hi_off - word_off)) - 1;
                }
                BitIter { word: masked }.map(move |b| base + word_off + b)
            })
    }

    /// Every member, ascending.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        let base = self.base;
        self.words.iter().enumerate().flat_map(move |(w, &word)| {
            let word_off = w as u64 * BITS;
            BitIter { word }.map(move |b| base + word_off + b)
        })
    }

    /// Drop every member below `horizon`, advancing
    /// [`floor`](IndexSet::floor).
    pub fn retire_below(&mut self, horizon: u64) {
        self.floor = self.floor.max(horizon);
        if horizon <= self.base {
            return;
        }
        while self.base + BITS <= horizon && !self.words.is_empty() {
            let word = self.words.pop_front().unwrap_or(0);
            self.len -= word.count_ones() as usize;
            self.base += BITS;
        }
        if self.words.is_empty() {
            self.base = horizon - horizon % BITS;
            return;
        }
        // A partial word at the front. The loop above leaves
        // `base + BITS > horizon`, so `cut` is in `1..BITS`.
        let cut = horizon - self.base;
        if cut > 0 {
            let mask = (1u64 << cut) - 1;
            let word = &mut self.words[0];
            self.len -= (*word & mask).count_ones() as usize;
            *word &= !mask;
        }
    }
}

/// Set bit positions of one word, ascending.
struct BitIter {
    word: u64,
}

impl Iterator for BitIter {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<u64> {
        if self.word == 0 {
            return None;
        }
        let b = u64::from(self.word.trailing_zeros());
        self.word &= self.word - 1;
        Some(b)
    }
}

#[cfg(test)]
mod tests;
