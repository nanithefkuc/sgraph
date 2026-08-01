//! A fixed parity-check matrix, stored as compressed sparse rows.
//!
//! Nothing is sampled here: the graph is exactly what the caller supplied. That
//! makes this both the classic-LDPC generator — where `H` is a design artefact
//! rather than a draw — and the escape hatch for a consumer whose graph is built
//! by means this crate has never heard of.
//!
//! Shape is validated once, at construction, so generating a check is a lookup
//! and a copy with nothing left to reject per edge.

use super::{NeighborBuf, NeighborGen};
use crate::error::GraphError;
use crate::id::{CheckId, VarId};
use crate::weight::{Binary, EdgeWeight};
use alloc::vec::Vec;

/// Reject a column repeated within one row.
///
/// Sorts a scratch copy instead of comparing every pair, so a wide row costs
/// `d log d` rather than `d²`. Rows are the caller's order, which is why the
/// duplicate is found in a copy and never by sorting the stored row.
fn check_distinct(row: &[u64], scratch: &mut Vec<u64>) -> Result<(), GraphError> {
    scratch.clear();
    scratch.extend_from_slice(row);
    scratch.sort_unstable();
    for pair in scratch.windows(2) {
        if pair[0] == pair[1] {
            return Err(GraphError::DuplicateVariable { var: pair[0] });
        }
    }
    Ok(())
}

/// A validated parity-check matrix in compressed sparse row form.
///
/// Row `r` is `columns[offsets[r]..offsets[r + 1]]`, so `offsets` holds
/// `rows + 1` entries. Each row's stored order is the **caller's** and is
/// preserved verbatim by generation: like every generator here the order is
/// deterministic but not necessarily sorted, and nothing may sort it in place and
/// expect peers to agree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplicitMatrix {
    offsets: Vec<usize>,
    columns: Vec<u64>,
    column_count: u64,
    max_degree: u32,
}

impl ExplicitMatrix {
    /// A generator over the CSR matrix `(offsets, columns)` addressing
    /// `column_count` variables.
    ///
    /// Every shape rule is checked here, once, so generation is a pure lookup.
    ///
    /// # Errors
    ///
    /// * [`GraphError::CsrOffsetsEmpty`] — `offsets` holds `rows + 1` entries, so
    ///   it is never empty.
    /// * [`GraphError::CsrOffsetsNotMonotone`] — an offset dropped below its
    ///   predecessor, inverting a row range.
    /// * [`GraphError::CsrRowRangePastEnd`] — a row range extends past the column
    ///   entries supplied.
    /// * [`GraphError::CsrColumnOutOfRange`] — an entry names a column the matrix
    ///   does not have.
    /// * [`GraphError::EmptySupport`] — a row with no entries constrains nothing.
    /// * [`GraphError::DuplicateVariable`] — a column repeated within one row
    ///   would fold twice during reduction.
    pub fn new(
        offsets: Vec<usize>,
        columns: Vec<u64>,
        column_count: u64,
    ) -> Result<Self, GraphError> {
        if offsets.is_empty() {
            return Err(GraphError::CsrOffsetsEmpty);
        }
        for (index, pair) in offsets.windows(2).enumerate() {
            if pair[1] < pair[0] {
                return Err(GraphError::CsrOffsetsNotMonotone {
                    row: index + 1,
                    previous: pair[0],
                    current: pair[1],
                });
            }
        }

        let mut scratch = Vec::new();
        let mut max_degree = 0u32;
        for row in 0..offsets.len() - 1 {
            // Monotone offsets, so `start <= end` and only `end` can overrun.
            let (start, end) = (offsets[row], offsets[row + 1]);
            if end > columns.len() {
                return Err(GraphError::CsrRowRangePastEnd {
                    row,
                    end,
                    len: columns.len(),
                });
            }
            let entries = &columns[start..end];
            if entries.is_empty() {
                return Err(GraphError::EmptySupport);
            }
            for &column in entries {
                if column >= column_count {
                    return Err(GraphError::CsrColumnOutOfRange {
                        row,
                        column,
                        columns: column_count,
                    });
                }
            }
            check_distinct(entries, &mut scratch)?;
            // A row wider than `u32::MAX` cannot be built in this address space;
            // saturating keeps the scratch hint honest without a panicking cast.
            max_degree = max_degree.max(u32::try_from(entries.len()).unwrap_or(u32::MAX));
        }

        Ok(Self {
            offsets,
            columns,
            column_count,
            max_degree,
        })
    }

    /// Number of check rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Number of variables the matrix addresses.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> u64 {
        self.column_count
    }

    /// The columns of one row, in the order they were supplied, or `None` when
    /// `row` is past [`rows`](Self::rows).
    #[inline]
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[u64]> {
        let end = *self.offsets.get(row.checked_add(1)?)?;
        Some(&self.columns[self.offsets[row]..end])
    }
}

impl NeighborGen for ExplicitMatrix {
    type Weight = Binary;

    fn neighbors(&self, id: CheckId, out: &mut NeighborBuf<Binary>) -> Result<(), GraphError> {
        out.clear();
        let Some(row) = usize::try_from(id.get()).ok().and_then(|row| self.row(row)) else {
            return Err(GraphError::CheckOutOfRange {
                check: id.get(),
                rows: self.rows() as u64,
            });
        };
        for &column in row {
            out.push(VarId::new(column), Binary::one());
        }
        Ok(())
    }

    #[inline]
    fn max_degree(&self) -> u32 {
        self.max_degree
    }
}

#[cfg(test)]
mod tests {
    use super::ExplicitMatrix;
    use crate::error::GraphError;
    use crate::id::{CheckId, VarId};
    use crate::neighbors::{NeighborBuf, NeighborGen};
    use crate::peel::{Peeler, PoolConfig, VariableState};
    use crate::weight::Binary;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::num::NonZeroUsize;

    // The (7,4) Hamming code's parity-check matrix, transcribed from its
    // textbook form — column `c` is the binary expansion of `c + 1`:
    //
    //     1 0 1 0 1 0 1
    //     0 1 1 0 0 1 1
    //     0 0 0 1 1 1 1
    const HAMMING_OFFSETS: [usize; 4] = [0, 4, 8, 12];
    const HAMMING_COLUMNS: [u64; 12] = [0, 2, 4, 6, 1, 2, 5, 6, 3, 4, 5, 6];
    const HAMMING_ROWS: [&[u64]; 3] = [&[0, 2, 4, 6], &[1, 2, 5, 6], &[3, 4, 5, 6]];
    const HAMMING_VARS: u64 = 7;

    const SYMBOL_LEN: usize = 4;

    fn hamming() -> ExplicitMatrix {
        ExplicitMatrix::new(
            HAMMING_OFFSETS.to_vec(),
            HAMMING_COLUMNS.to_vec(),
            HAMMING_VARS,
        )
        .expect("the Hamming matrix is well formed")
    }

    fn support(columns: &[u64]) -> Vec<VarId> {
        columns.iter().copied().map(VarId::new).collect()
    }

    fn symbol(var: u64) -> [u8; SYMBOL_LEN] {
        let b = var as u8;
        [
            b.wrapping_mul(37).wrapping_add(1),
            b.wrapping_mul(53).wrapping_add(2),
            b.wrapping_mul(97).wrapping_add(3),
            b.wrapping_mul(131).wrapping_add(5),
        ]
    }

    #[test]
    fn hamming_7_4_generates_each_stored_row_verbatim() {
        let matrix = hamming();
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.columns(), 7);
        let mut out = NeighborBuf::<Binary>::with_capacity(4);
        for (row, &expected) in HAMMING_ROWS.iter().enumerate() {
            assert_eq!(matrix.row(row), Some(expected));
            matrix
                .neighbors(CheckId::new(row as u64), &mut out)
                .expect("row is in range");
            assert_eq!(out.support(), support(expected).as_slice());
            assert_eq!(out.weights(), [Binary; 4].as_slice());
        }
        assert_eq!(matrix.row(3), None);
    }

    #[test]
    fn stored_column_order_survives_unsorted() {
        let matrix = ExplicitMatrix::new(vec![0, 4], vec![6, 2, 5, 1], HAMMING_VARS)
            .expect("one unsorted row is well formed");
        let mut out = NeighborBuf::<Binary>::new();
        matrix
            .neighbors(CheckId::new(0), &mut out)
            .expect("row is in range");
        assert_eq!(out.support(), support(&[6, 2, 5, 1]).as_slice());
    }

    #[test]
    fn max_degree_is_the_widest_row() {
        assert_eq!(hamming().max_degree(), 4);
        let irregular = ExplicitMatrix::new(vec![0, 1, 4, 6], vec![0, 1, 2, 3, 4, 5], 6)
            .expect("rows of width 1, 3 and 2 are well formed");
        assert_eq!(irregular.rows(), 3);
        assert_eq!(irregular.max_degree(), 3);
    }

    /// Naive peeling over an adjacency list, written from the definition: while
    /// some row holds exactly one unknown, that row determines it. Independent of
    /// the crate entirely, so it can arbitrate the production peeler.
    fn reference_recovers(rows: &[Vec<usize>], erased: u8) -> bool {
        let mut unknown = erased;
        loop {
            let mut progressed = false;
            for row in rows {
                let mut count = 0usize;
                let mut only = 0usize;
                for &column in row {
                    if unknown >> column & 1 == 1 {
                        count += 1;
                        only = column;
                    }
                }
                if count == 1 {
                    unknown &= !(1u8 << only);
                    progressed = true;
                }
            }
            if unknown == 0 {
                return true;
            }
            if !progressed {
                return false;
            }
        }
    }

    /// Ingest every row of the matrix with the variables outside `erased` already
    /// known, and report whether all seven variables came back.
    fn production_recovers(matrix: &ExplicitMatrix, erased: u8) -> bool {
        let span = NonZeroUsize::new(8).expect("8 is non-zero");
        let mut peeler = Peeler::<Binary>::new(SYMBOL_LEN, PoolConfig::new(span, span))
            .expect("a 4-byte symbol is aligned for Binary");
        for var in 0..HAMMING_VARS {
            if erased >> var & 1 == 0 {
                peeler
                    .learn_copy(VarId::new(var), &symbol(var))
                    .expect("variable is in span");
            }
        }
        for (row, &columns) in HAMMING_ROWS.iter().enumerate() {
            let mut rhs = [0u8; SYMBOL_LEN];
            for &column in columns {
                for (to, from) in rhs.iter_mut().zip(symbol(column)) {
                    *to ^= from;
                }
            }
            peeler
                .push_check_with(CheckId::new(row as u64), matrix, &rhs)
                .expect("row is in range and its support is canonical");
        }
        let recovered = (0..HAMMING_VARS).all(|var| {
            peeler.variable_state(VarId::new(var)) == VariableState::Known(&symbol(var))
        });
        // Every variable of this matrix sits in some row, so an unrecovered one
        // must leave a row unresolved.
        assert_eq!(
            peeler.has_stalled(),
            !recovered,
            "erasure pattern {erased:#09b} disagreed on stalling"
        );
        recovered
    }

    #[test]
    fn every_erasure_pattern_matches_a_naive_reference_peeler() {
        let matrix = hamming();
        let reference: Vec<Vec<usize>> = HAMMING_ROWS
            .iter()
            .map(|row| row.iter().map(|&column| column as usize).collect())
            .collect();

        let mut recovered = Vec::new();
        let mut stalled = Vec::new();
        for erased in 0..128u8 {
            let expected = reference_recovers(&reference, erased);
            assert_eq!(
                production_recovers(&matrix, erased),
                expected,
                "erasure pattern {erased:#09b} disagreed with the reference"
            );
            if expected {
                recovered.push(erased);
            } else {
                stalled.push(erased);
            }
        }

        // Pinned from the enumeration so the comparison above cannot pass
        // vacuously: 54 of the 128 subsets of seven variables peel to completion.
        assert_eq!(recovered.len(), 54);
        assert_eq!(stalled.len(), 74);
        for var in 0..HAMMING_VARS {
            assert!(
                recovered.contains(&(1u8 << var)),
                "a single erasure of variable {var} must peel"
            );
        }
        // {4, 5, 6} is a genuine stopping set: every row meeting it meets it at
        // least twice, so no row ever reaches degree one.
        assert!(stalled.contains(&0b111_0000));
    }

    #[test]
    fn empty_offsets_are_rejected() {
        assert_eq!(
            ExplicitMatrix::new(Vec::new(), vec![0, 1], 7).unwrap_err(),
            GraphError::CsrOffsetsEmpty
        );
    }

    #[test]
    fn a_decreasing_offset_is_rejected() {
        assert_eq!(
            ExplicitMatrix::new(vec![0, 3, 1], vec![0, 1, 2], 7).unwrap_err(),
            GraphError::CsrOffsetsNotMonotone {
                row: 2,
                previous: 3,
                current: 1,
            }
        );
    }

    #[test]
    fn a_row_range_past_the_columns_is_rejected() {
        assert_eq!(
            ExplicitMatrix::new(vec![0, 2, 5], vec![0, 1, 2], 7).unwrap_err(),
            GraphError::CsrRowRangePastEnd {
                row: 1,
                end: 5,
                len: 3,
            }
        );
    }

    #[test]
    fn a_column_outside_the_matrix_is_rejected() {
        assert_eq!(
            ExplicitMatrix::new(vec![0, 2], vec![0, 7], 7).unwrap_err(),
            GraphError::CsrColumnOutOfRange {
                row: 0,
                column: 7,
                columns: 7,
            }
        );
    }

    #[test]
    fn an_empty_row_is_rejected() {
        assert_eq!(
            ExplicitMatrix::new(vec![0, 2, 2], vec![0, 1], 7).unwrap_err(),
            GraphError::EmptySupport
        );
    }

    #[test]
    fn a_repeated_column_within_one_row_is_rejected() {
        assert_eq!(
            ExplicitMatrix::new(vec![0, 3], vec![1, 4, 4], 7).unwrap_err(),
            GraphError::DuplicateVariable { var: 4 }
        );
    }

    #[test]
    fn a_rejected_check_leaves_the_buffer_cleared_and_reusable() {
        let matrix = hamming();
        let mut out = NeighborBuf::<Binary>::with_capacity(4);
        matrix
            .neighbors(CheckId::new(1), &mut out)
            .expect("row is in range");
        assert_eq!(out.len(), 4);

        assert_eq!(
            matrix.neighbors(CheckId::new(3), &mut out).unwrap_err(),
            GraphError::CheckOutOfRange { check: 3, rows: 3 }
        );
        assert!(out.is_empty());
        assert_eq!(out.len(), 0);

        assert_eq!(
            matrix
                .neighbors(CheckId::new(u64::MAX), &mut out)
                .unwrap_err(),
            GraphError::CheckOutOfRange {
                check: u64::MAX,
                rows: 3,
            }
        );
        assert!(out.is_empty());

        matrix
            .neighbors(CheckId::new(0), &mut out)
            .expect("row is in range");
        assert_eq!(out.support(), support(HAMMING_ROWS[0]).as_slice());
    }
}
