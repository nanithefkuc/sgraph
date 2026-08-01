use super::*;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

/// Generous enough that the limit is not what these tests exercise.
fn span(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("test span must be non-zero")
}

fn ring<T: Default>(max: usize) -> Ring<T> {
    Ring::new(span(max))
}

fn set(max: usize) -> IndexSet {
    IndexSet::new(span(max))
}

/// A tiny xorshift, independent of this crate's own PRNG: the oracle for a
/// differential test must not share machinery with what it checks.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn ring_tracks_a_sliding_span() {
    let mut r: Ring<Option<u32>> = ring(1024);
    *r.ensure(100).unwrap() = Some(7);
    *r.ensure(103).unwrap() = Some(9);
    assert_eq!(r.get(100).live(), Some(&Some(7)));
    assert_eq!(r.get(101).live(), Some(&None));
    assert_eq!(r.get(103).live(), Some(&Some(9)));
    assert_eq!(r.get(104), Lookup::Vacant);
    assert_eq!(r.last(), Some(103));
    assert_eq!(r.len(), 4);

    let dropped: Vec<_> = r.retire_below(102).flatten().collect();
    assert_eq!(dropped, [7]);
    assert_eq!(r.get(100), Lookup::Retired, "retired must not resurface");
    assert_eq!(r.get(103).live(), Some(&Some(9)));
    assert_eq!(r.floor(), 102);

    // Draining everything re-anchors instead of growing a run of holes.
    let _ = r.retire_below(1_000).count();
    *r.ensure(5_000).unwrap() = Some(1);
    assert_eq!(r.iter().count(), 1);
    assert_eq!(r.base(), 5_000);
}

/// The three states must be observably different. Collapsing `Retired` into
/// `Vacant` is how a caller silently loses data.
#[test]
fn ring_distinguishes_retired_from_vacant() {
    let mut r: Ring<u32> = ring(1024);
    *r.ensure(100).unwrap() = 7;
    let _ = r.retire_below(50).count();

    assert_eq!(r.get(100), Lookup::Live(&7));
    assert_eq!(r.get(200), Lookup::Vacant, "never populated, but reachable");
    assert_eq!(r.get(10), Lookup::Retired, "below the horizon");
    assert!(r.is_retired(10) && !r.is_retired(200));
    assert!(r.get(10).is_retired());
    assert!(!r.get(200).is_retired());

    // And the same distinction on the mutating path.
    assert!(matches!(
        r.ensure(10),
        Err(GraphError::IndexRetired {
            index: 10,
            floor: 50
        })
    ));
    assert!(r.ensure(200).is_ok(), "vacant is insertable");
}

/// The hazard the `floor` field exists for. An emptied ring re-anchors on its
/// next insert, so without a retained horizon it would hand out a slot for an
/// index it had already retired.
#[test]
fn ring_never_resurrects_a_retired_index() {
    let mut r: Ring<Option<u32>> = ring(1024);
    *r.ensure(10).unwrap() = Some(1);
    let _ = r.retire_below(100).count();
    assert!(r.is_empty());

    assert!(r.ensure(10).is_err(), "retired index must stay gone");
    assert!(r.ensure(99).is_err(), "below the horizon is gone");
    assert!(r.ensure(100).is_ok(), "the horizon itself is live");
    assert_eq!(r.floor(), 100);

    // The floor is monotone: a lower horizon must not lower it.
    let _ = r.retire_below(50).count();
    assert_eq!(r.floor(), 100);
}

/// An index above the floor but below `base` is absent, not gone. Refusing it —
/// as a purely forward-growing ring would — silently drops live data.
#[test]
fn ring_grows_at_the_front() {
    let mut r: Ring<Option<u32>> = ring(1024);
    *r.ensure(500).unwrap() = Some(5);
    assert_eq!(r.base(), 500);

    *r.ensure(100).unwrap() = Some(1);
    assert_eq!(r.base(), 100);
    assert_eq!(r.get(100).live(), Some(&Some(1)));
    assert_eq!(
        r.get(500).live(),
        Some(&Some(5)),
        "front growth must not shift data"
    );
    assert_eq!(r.get(300).live(), Some(&None), "the gap takes defaults");
    assert_eq!(r.len(), 401);
}

/// Invariant 11: a far-away id must not be able to request a gap-sized
/// allocation, and a rejection must leave the structure untouched.
#[test]
fn ring_rejects_a_sparse_gap_without_mutating() {
    let mut r: Ring<u32> = ring(64);
    *r.ensure(10).unwrap() = 1;
    let before_len = r.len();
    let before_base = r.base();

    let err = r.ensure(10_000).expect_err("gap far beyond the span limit");
    assert!(matches!(
        err,
        GraphError::LiveSpanExceeded {
            index: 10_000,
            required: 9_991,
            limit: 64
        }
    ));
    assert_eq!(r.len(), before_len, "rejected growth changed the length");
    assert_eq!(r.base(), before_base, "rejected growth moved the base");
    assert_eq!(r.get(10), Lookup::Live(&1), "existing state was disturbed");

    // Rejection is symmetric: growing far down is bounded too.
    let mut r2: Ring<u32> = ring(64);
    *r2.ensure(10_000).unwrap() = 1;
    assert!(r2.ensure(10).is_err());
    assert_eq!(r2.len(), 1);

    // Exactly at the limit is allowed; one past it is not.
    let mut r3: Ring<u32> = ring(64);
    r3.ensure(0).unwrap();
    assert!(r3.ensure(63).is_ok(), "span of exactly 64 fits");
    assert!(r3.ensure(64).is_err(), "span of 65 does not");
}

/// Dense storage at the very top of the domain: an exclusive end would overflow,
/// which is why the API reports an inclusive `last`.
#[test]
fn ring_handles_the_top_of_the_domain() {
    let mut r: Ring<u32> = ring(4);
    *r.ensure(u64::MAX).unwrap() = 9;
    assert_eq!(r.base(), u64::MAX);
    assert_eq!(r.last(), Some(u64::MAX));
    assert_eq!(r.get(u64::MAX), Lookup::Live(&9));
    assert_eq!(r.len(), 1);

    *r.ensure(u64::MAX - 3).unwrap() = 1;
    assert_eq!(r.len(), 4);
    assert_eq!(r.get(u64::MAX - 3), Lookup::Live(&1));
    assert_eq!(r.get(u64::MAX), Lookup::Live(&9));

    // Retiring past the top empties it without wrapping.
    let _ = r.retire_below(u64::MAX).count();
    assert_eq!(r.len(), 1);
    assert_eq!(r.get(u64::MAX), Lookup::Live(&9));
}

#[test]
fn ring_retire_without_consuming_still_retires() {
    let mut r: Ring<Option<u32>> = ring(16);
    *r.ensure(0).unwrap() = Some(1);
    *r.ensure(1).unwrap() = Some(2);
    drop(r.retire_below(1));
    assert_eq!(r.get(0), Lookup::Retired);
    assert_eq!(r.get(1).live(), Some(&Some(2)));
    assert_eq!(r.len(), 1);
}

#[test]
fn ring_empty_and_full_retirement() {
    let mut r: Ring<u32> = ring(16);
    // Retiring an empty ring is a no-op beyond advancing the floor.
    assert_eq!(r.retire_below(10).count(), 0);
    assert_eq!(r.floor(), 10);
    assert!(r.is_empty());
    assert_eq!(r.last(), None);

    r.ensure(10).unwrap();
    r.ensure(11).unwrap();
    assert_eq!(
        r.retire_below(12).count(),
        2,
        "full drain yields everything"
    );
    assert!(r.is_empty());
    assert_eq!(r.last(), None);
}

#[test]
fn index_set_insert_reports_novelty() {
    let mut s = set(64);
    assert!(s.insert(5).unwrap(), "first insert is new");
    assert!(!s.insert(5).unwrap(), "second insert is not");
    assert_eq!(s.len(), 1);
    assert!(s.remove(5));
    assert!(!s.remove(5));
    assert!(s.is_empty());
}

/// Retirement leaves `base` word-aligned *below* the horizon, so the masked-off
/// remainder of the front word is the subtle resurrection path.
#[test]
fn index_set_never_resurrects_a_retired_index() {
    let mut s = set(1024);
    assert!(s.insert(70).unwrap());
    s.retire_below(72);
    assert_eq!(s.len(), 0);
    assert!(!s.contains(70));

    // 70 sits inside the front word, whose base is 64 — the masked region.
    assert!(s.insert(70).is_err(), "a retired index must not return");
    assert!(s.insert(64).is_err());
    assert_eq!(s.len(), 0, "a refused insert must not change len");

    assert!(s.insert(72).unwrap(), "the horizon itself is live");
    assert_eq!(s.len(), 1);

    s.retire_below(1_000);
    assert!(s.insert(500).is_err());
    assert!(s.insert(1_000).unwrap());
    assert_eq!(s.floor(), 1_000);
}

#[test]
fn index_set_distinguishes_retired_from_vacant() {
    let mut s = set(1024);
    s.insert(100).unwrap();
    s.retire_below(50);

    assert_eq!(s.status(100), Membership::Member);
    assert_eq!(s.status(200), Membership::Vacant);
    assert_eq!(s.status(10), Membership::Retired);
    assert!(s.is_retired(10) && !s.is_retired(200));
}

#[test]
fn index_set_grows_at_the_front() {
    let mut s = set(1024);
    assert!(s.insert(700).unwrap());
    assert!(
        s.insert(70).unwrap(),
        "below base but never retired: absent, not gone"
    );
    assert!(s.insert(0).unwrap());
    assert_eq!(s.len(), 3);
    assert!(s.contains(0) && s.contains(70) && s.contains(700));
    assert_eq!(s.iter().collect::<Vec<_>>(), [0, 70, 700]);
}

#[test]
fn index_set_rejects_a_sparse_gap_without_mutating() {
    let mut s = set(128);
    s.insert(10).unwrap();
    let err = s.insert(1_000_000).expect_err("far beyond the span limit");
    assert!(matches!(
        err,
        GraphError::LiveSpanExceeded { limit: 128, .. }
    ));
    assert_eq!(s.len(), 1, "rejected growth changed membership");
    assert!(s.contains(10));
    assert!(!s.contains(1_000_000));
}

#[test]
fn index_set_handles_the_top_of_the_domain() {
    let mut s = set(128);
    assert!(s.insert(u64::MAX).unwrap());
    assert!(s.contains(u64::MAX));
    assert_eq!(s.len(), 1);
    assert_eq!(s.iter().collect::<Vec<_>>(), [u64::MAX]);
    assert_eq!(s.range(u64::MAX - 1, u64::MAX).count(), 0);
    assert_eq!(s.range(u64::MAX, u64::MAX).count(), 0, "hi is exclusive");
    assert!(s.remove(u64::MAX));
}

/// Differential against `BTreeSet` over insert / remove / retire. The oracle is a
/// `BTreeSet` plus an explicit floor, which is precisely the documented
/// semantics: only below-floor indices are refused.
///
/// The index range is deliberately wide relative to the retirement horizons, so
/// inserts land below `base` constantly and the front-growth path is exercised
/// rather than incidental.
#[test]
fn index_set_matches_a_btreeset() {
    let mut a = set(4096);
    let mut b: BTreeSet<u64> = BTreeSet::new();
    let mut state = 12345u64;
    let mut floor = 0u64;

    for step in 0..2000 {
        let i = xorshift(&mut state) % 2000;
        match step % 5 {
            0..=2 => match a.insert(i) {
                Ok(new) => assert_eq!(new, b.insert(i), "insert({i}) disagreed"),
                Err(GraphError::IndexRetired { .. }) => {
                    assert!(i < floor, "insert({i}) refused a live index");
                }
                Err(e) => panic!("unexpected error at step {step}: {e}"),
            },
            3 => assert_eq!(a.remove(i), b.remove(&i), "remove({i}) disagreed"),
            _ => {
                let h = xorshift(&mut state) % 2000;
                a.retire_below(h);
                b = b.split_off(&h);
                floor = floor.max(h);
            }
        }
        assert_eq!(a.len(), b.len(), "len diverged at step {step}");
        assert_eq!(a.contains(i), b.contains(&i), "contains({i}) diverged");
        assert_eq!(a.floor(), floor, "floor diverged at step {step}");
    }

    assert_eq!(
        a.iter().collect::<Vec<_>>(),
        b.iter().copied().collect::<Vec<_>>()
    );
}

/// `range` seeks to the first relevant word rather than scanning the set, so its
/// word masking is the part that can go wrong. Sweep bounds across word
/// boundaries and past both ends.
#[test]
fn index_set_range_matches_btreeset() {
    let mut a = set(4096);
    let mut b: BTreeSet<u64> = BTreeSet::new();
    let mut state = 999u64;
    for _ in 0..400 {
        let i = 300 + xorshift(&mut state) % 500;
        a.insert(i).unwrap();
        b.insert(i);
    }
    a.retire_below(320);
    b = b.split_off(&320);

    let interesting = [
        0u64, 1, 63, 64, 65, 127, 128, 300, 319, 320, 321, 383, 384, 500, 511, 512, 700, 799, 800,
        801, 1000,
    ];
    for &lo in &interesting {
        for &hi in &interesting {
            if lo > hi {
                // `BTreeSet::range` panics on an inverted range; ours is empty.
                assert_eq!(a.range(lo, hi).count(), 0, "range({lo}, {hi}) not empty");
                continue;
            }
            assert_eq!(
                a.range(lo, hi).collect::<Vec<_>>(),
                b.range(lo..hi).copied().collect::<Vec<_>>(),
                "range({lo}, {hi}) diverged"
            );
        }
    }
}

#[test]
fn index_set_range_on_empty_set_is_empty() {
    let s = set(64);
    assert_eq!(s.range(0, u64::MAX).count(), 0);
    assert_eq!(s.range(5, 5).count(), 0);
}

/// The whole-domain span is `2^64`, which is not a `u64` and not a `usize` on
/// any target. It must be reported, not wrapped.
#[test]
fn whole_domain_span_is_not_representable() {
    let mut r: Ring<u32> = ring(usize::MAX);
    r.ensure(0).unwrap();
    assert_eq!(
        r.ensure(u64::MAX),
        Err(GraphError::IndexNotRepresentable { index: u64::MAX }),
        "a span of 2^64 must be rejected rather than wrapping to 0"
    );
    assert_eq!(r.len(), 1, "rejection mutated the ring");

    let mut s = set(usize::MAX);
    s.insert(0).unwrap();
    assert!(matches!(
        s.insert(u64::MAX),
        Err(GraphError::IndexNotRepresentable { index: u64::MAX })
    ));
    assert_eq!(s.len(), 1);
}

/// On a 32-bit target a span can exceed `usize` long before it exceeds `u64`.
/// The conversion is checked, so this is an error rather than a truncation.
#[cfg(target_pointer_width = "32")]
#[test]
fn spans_beyond_usize_are_rejected_on_32_bit() {
    let mut r: Ring<u32> = ring(usize::MAX);
    r.ensure(0).unwrap();
    let beyond = u64::from(u32::MAX) + 1;
    assert_eq!(
        r.ensure(beyond),
        Err(GraphError::IndexNotRepresentable { index: beyond })
    );
    assert_eq!(r.len(), 1);
}
