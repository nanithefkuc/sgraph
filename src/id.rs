//! Variable and check identifiers.
//!
//! A Tanner graph has two kinds of node and they are easy to confuse: a
//! retirement horizon is a variable index, while the ring it retires from is
//! keyed by check id. These newtypes make that mix-up a compile error at no
//! runtime cost.
//!
//! Storage stays keyed by raw `u64` — one [`Ring`](crate::index::Ring) holds
//! check rows while another holds per-variable adjacency, so the container is
//! agnostic and the identifiers live at the public boundary.

/// A variable node: one source symbol, known or unknown.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct VarId(pub u64);

/// A check node: one constraint over a set of variables.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct CheckId(pub u64);

macro_rules! id_impls {
    ($name:ident, $what:literal) => {
        impl $name {
            #[doc = concat!("The lowest ", $what, " identifier.")]
            pub const ZERO: Self = Self(0);

            #[doc = concat!("Wrap a raw index as a ", $what, " identifier.")]
            #[inline]
            #[must_use]
            pub const fn new(raw: u64) -> Self {
                Self(raw)
            }

            /// The underlying index.
            #[inline]
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl From<u64> for $name {
            #[inline]
            fn from(raw: u64) -> Self {
                Self(raw)
            }
        }

        impl From<$name> for u64 {
            #[inline]
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

id_impls!(VarId, "variable");
id_impls!(CheckId, "check");

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn ids_are_transparent_over_u64() {
        assert_eq!(size_of::<VarId>(), size_of::<u64>());
        assert_eq!(size_of::<CheckId>(), size_of::<u64>());
    }

    #[test]
    fn conversions_round_trip() {
        for raw in [0u64, 1, 42, u64::MAX] {
            assert_eq!(u64::from(VarId::from(raw)), raw);
            assert_eq!(CheckId::new(raw).get(), raw);
        }
        assert_eq!(VarId::ZERO.get(), 0);
    }

    #[test]
    fn ids_order_by_index() {
        assert!(VarId(1) < VarId(2));
        assert!(CheckId(u64::MAX) > CheckId(0));
    }
}
