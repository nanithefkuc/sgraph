//! Edge coefficients, and the seam that keeps the binary case free.
//!
//! A Tanner graph over GF(2) has no per-edge coefficient: a check is the XOR of
//! its neighbours. Over GF(2^m) every edge carries a field element, reduction
//! becomes a multiply-accumulate, and peeling a degree-one row needs a field
//! inverse. Those are different algorithms in the small but the same algorithm in
//! the large, so the peeler is written once against [`EdgeWeight`].
//!
//! The trick that makes the abstraction free is that [`Binary`] is a
//! zero-sized type. Rust gives `Vec<ZST>` a dangling pointer and a capacity of
//! `usize::MAX`, so a parallel `weights: Vec<Binary>` beside a
//! `support: Vec<VarId>` costs exactly what the bare support vector costs —
//! `push`, `swap_remove`, and indexing all compile away. No specialization, no
//! `const IS_BINARY` branching in the hot loop, no second engine to drift.
//!
//! Field arithmetic is always `fgf`'s. Nothing here implements a field loop.

use fgf::field::Elem;
use fgf::{FieldKernels, Gf8};

/// One edge's coefficient.
///
/// Implementations are expected to be trivially copyable and, for the binary
/// case, zero-sized. `Default` exists so support scratch can be grown without a
/// meaningful value.
pub trait EdgeWeight: Copy + Eq + Default + core::fmt::Debug + 'static {
    /// Width in bytes of one packed field element.
    ///
    /// A symbol is a packed array of elements, and `fgf`'s kernels panic on a
    /// partial trailing element. The peeler validates symbol length against this
    /// once at construction so no caller can reach that panic.
    const ELEMENT_BYTES: usize;

    /// The multiplicative identity — the only coefficient a binary edge has.
    fn one() -> Self;

    /// True when this coefficient is zero, and so not really an edge at all.
    fn is_zero(self) -> bool;

    /// `dst += w * src`, over the field.
    ///
    /// # Panics
    ///
    /// If `dst` and `src` differ in length, or either holds a partial element.
    /// Callers validate geometry once at their own boundary.
    fn mul_add(dst: &mut [u8], w: Self, src: &[u8]);

    /// `value *= w⁻¹`, in place.
    ///
    /// This is the peel step: a row whose support has collapsed to one variable
    /// of coefficient `w` determines that variable as `w⁻¹ · rhs`. Ingest
    /// validation guarantees `w` is non-zero, so the inverse always exists.
    fn scale_inv(value: &mut [u8], w: Self);
}

/// Embed a sparse edge coefficient into the residual solver's field.
///
/// Peeling can be binary while the residual solve cannot: Gaussian elimination
/// needs division, so the dense system is always over a real field. A binary
/// edge widens to `ONE`; a weighted edge maps only into its own field.
pub trait ResidualCoeff<F: FieldKernels>: EdgeWeight {
    /// This coefficient as an element of `F`.
    fn coefficient(self) -> F::Elem;
}

/// A GF(2) edge: present, with an implicit coefficient of one.
///
/// Zero-sized, so carrying a weight per edge costs nothing.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, Hash, PartialOrd, Ord)]
pub struct Binary;

impl EdgeWeight for Binary {
    /// GF(2) symbols are plain bytes: every length is aligned.
    const ELEMENT_BYTES: usize = 1;

    #[inline]
    fn one() -> Self {
        Self
    }

    /// A binary edge either exists or is absent from the support. There is no
    /// zero coefficient to represent.
    #[inline]
    fn is_zero(self) -> bool {
        false
    }

    #[inline]
    fn mul_add(dst: &mut [u8], _w: Self, src: &[u8]) {
        debug_assert_eq!(dst.len(), src.len(), "mul_add: length mismatch");
        // Addition in any GF(2^m) is XOR of the packed bytes, so this kernel is
        // field-independent; `fgf` has no `Gf2`, and `Gf8` is an arbitrary
        // witness type that selects the same byte-wise routine.
        fgf::ops::add_assign::<Gf8>(dst, src);
    }

    /// The inverse of one is one.
    #[inline]
    fn scale_inv(_value: &mut [u8], _w: Self) {}
}

impl<F: FieldKernels> ResidualCoeff<F> for Binary {
    #[inline]
    fn coefficient(self) -> F::Elem {
        F::Elem::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::mem::size_of;
    use fgf::{Gf16, gf8, gf16};

    #[test]
    fn binary_is_zero_sized() {
        assert_eq!(size_of::<Binary>(), 0);
        assert_eq!(size_of::<Option<Binary>>(), 1);
    }

    /// The claim the whole seam rests on: a parallel weight vector for the binary
    /// case costs nothing. `Vec<ZST>` never allocates, so its buffer pointer is
    /// dangling and constant no matter how much is pushed.
    #[test]
    fn binary_weight_vectors_never_allocate() {
        let mut v: Vec<Binary> = Vec::new();
        let ptr = v.as_ptr();
        for _ in 0..100_000 {
            v.push(Binary);
        }
        assert_eq!(v.len(), 100_000);
        assert_eq!(v.capacity(), usize::MAX, "a ZST Vec has unbounded capacity");
        assert_eq!(v.as_ptr(), ptr, "a ZST Vec never moves its buffer");

        // The operations the peeler performs on the weight vector are no-ops.
        v.swap_remove(0);
        assert_eq!(v.len(), 99_999);
        assert_eq!(v.as_ptr(), ptr);
    }

    #[test]
    fn binary_add_is_xor() {
        let mut dst = [0xF0u8, 0x0F, 0xAA, 0x55, 0x01];
        let src = [0x0Fu8, 0xF0, 0xFF, 0x55, 0x80];
        Binary::mul_add(&mut dst, Binary::one(), &src);
        assert_eq!(dst, [0xFF, 0xFF, 0x55, 0x00, 0x81]);
    }

    /// Self-inverse under addition is the property peeling relies on when it
    /// folds a known variable out of a row.
    #[test]
    fn binary_add_is_self_inverse() {
        let original: Vec<u8> = (0..77u8).collect();
        let mut dst = original.clone();
        let src: Vec<u8> = (0..77u8).map(|b| b.wrapping_mul(7)).collect();
        Binary::mul_add(&mut dst, Binary, &src);
        assert_ne!(dst, original);
        Binary::mul_add(&mut dst, Binary, &src);
        assert_eq!(dst, original, "adding twice must restore the original");
    }

    /// A ragged length past the SIMD width, to exercise the kernel tail rather
    /// than only its vector body.
    #[test]
    fn binary_add_handles_a_ragged_tail() {
        for len in [0usize, 1, 7, 15, 16, 17, 31, 63, 64, 65, 77, 255] {
            let src: Vec<u8> = (0..len).map(|i| (i as u8) ^ 0x5A).collect();
            let mut dst = vec![0u8; len];
            Binary::mul_add(&mut dst, Binary, &src);
            assert_eq!(dst, src, "adding to zero must copy, at len {len}");
        }
    }

    #[test]
    fn binary_scale_inv_is_identity() {
        let mut value = [1u8, 2, 3, 4];
        Binary::scale_inv(&mut value, Binary);
        assert_eq!(value, [1, 2, 3, 4]);
    }

    #[test]
    fn binary_is_never_zero() {
        assert!(!Binary.is_zero());
        assert!(!Binary::one().is_zero());
    }

    /// The residual solve is over a real field even when the graph is binary, so
    /// a binary edge must widen to that field's identity — in any field.
    #[test]
    fn binary_widens_to_field_identity() {
        assert_eq!(
            <Binary as ResidualCoeff<Gf8>>::coefficient(Binary),
            gf8::Elem(1)
        );
        assert_eq!(
            <Binary as ResidualCoeff<Gf16>>::coefficient(Binary),
            gf16::Elem(1)
        );
        // And that identity behaves as one under the field's own multiplication.
        let x = gf8::Elem(0xB7);
        assert_eq!(
            x.mul(<Binary as ResidualCoeff<Gf8>>::coefficient(Binary)),
            x
        );
    }
}
