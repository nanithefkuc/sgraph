//! Crate error types.
//!
//! Fallibility lives at the public boundary. `fff`'s kernels panic on geometry
//! violations, so every value this crate hands them is validated first; internal
//! call sites carry `debug_assert`s rather than repeated checks.
//!
//! Variants are added when something constructs them. An unconstructed variant
//! is a placeholder, and placeholders outlive their intent.

/// Graph construction and mutation failed.
///
/// Every variant carries the offending value and, where there is one, the limit
/// it violated. Rejection never mutates the structure that rejected it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraphError {
    /// The index has been retired and can never hold state again.
    ///
    /// Distinct from a vacant index, which was simply never populated.
    IndexRetired {
        /// The rejected index.
        index: u64,
        /// Retirement horizon; every index below it is gone.
        floor: u64,
    },
    /// Storing the index would stretch the live span past its configured limit.
    ///
    /// Dense index storage holds one slot per index across `[base, base + len)`,
    /// so a far-away index would otherwise request a gap-sized allocation. The
    /// limit rejects the input; it never evicts existing state to make room.
    LiveSpanExceeded {
        /// The rejected index.
        index: u64,
        /// Live span that storing it would require.
        required: u64,
        /// Configured maximum live span.
        limit: usize,
    },
    /// The index cannot be addressed on this platform's `usize`.
    ///
    /// Reachable only where `usize` is narrower than 64 bits.
    IndexNotRepresentable {
        /// The rejected index.
        index: u64,
    },
    /// An edge-weight implementation reported a zero packed-element width.
    ZeroElementBytes,
    /// A symbol length of zero has no valid interpretation.
    ZeroSymbolLen,
    /// A symbol does not have the decoder's configured byte length.
    SymbolLengthMismatch {
        /// Configured symbol length.
        expected: usize,
        /// Rejected symbol length.
        actual: usize,
    },
    /// A symbol ends with a partial packed field element.
    SymbolAlignment {
        /// Rejected symbol length.
        length: usize,
        /// Width of one packed field element.
        element_bytes: usize,
    },
    /// A variable retirement horizon moved backwards.
    HorizonRegressed {
        /// Rejected horizon.
        horizon: u64,
        /// Current retirement floor.
        floor: u64,
    },
    /// A check was explicitly retired and cannot be inserted again.
    CheckRetired {
        /// Retired check identifier.
        check: u64,
    },
    /// Fewer distinct offsets are available than were requested.
    SampleSpanTooSmall {
        /// Width of the domain sampled from.
        span: u32,
        /// Number of distinct offsets requested.
        requested: usize,
    },
    /// A degree distribution that can only ever produce zero edges.
    ///
    /// A check with no support constrains nothing, so this is a configuration
    /// bug rather than a degenerate but valid graph.
    ZeroDegree,
    /// The distribution can ask for more distinct variables than the domain has.
    DegreeExceedsDomain {
        /// Largest degree the distribution can produce.
        degree: u32,
        /// Number of variables available to draw from.
        domain: u64,
    },
    /// A generator was given a domain with no variables in it.
    EmptyDomain,
    /// The domain is wider than this generator's offsets can address.
    DomainTooLarge {
        /// The rejected domain width.
        domain: u64,
        /// Largest addressable width.
        max: u64,
    },
    /// The window's last variable index would pass `u64::MAX`.
    DomainOverflow {
        /// First variable index of the window.
        base: u64,
        /// Number of variables in the window.
        span: u32,
    },
    /// An edge set's support and weight lists disagree in length.
    ///
    /// They are parallel arrays; one weight belongs to each variable.
    EdgeLengthMismatch {
        /// Number of variables supplied.
        support: usize,
        /// Number of weights supplied.
        weights: usize,
    },
    /// An edge set with no support at all.
    EmptySupport,
    /// A variable occurs more than once in one edge set.
    ///
    /// Duplicates would accumulate silently during reduction and corrupt the
    /// residual invariant.
    DuplicateVariable {
        /// The repeated variable index.
        var: u64,
    },
    /// An edge carries a zero coefficient.
    ///
    /// A zero-weight edge is not an edge; left in place it would make a
    /// degree-one row unsolvable while still looking peelable.
    ZeroWeight {
        /// Variable index of the offending edge.
        var: u64,
    },
    /// A distribution was asked to cover zero symbols.
    ZeroSymbolCount,
    /// The robust-soliton `c` parameter was zero.
    ///
    /// `c` scales the spike position; at zero there is no spike and the
    /// construction degenerates.
    ZeroSolitonC,
    /// The robust-soliton failure probability was outside `(0, 1)`.
    ///
    /// The parameter is a Q32 fraction, so the representable open interval is
    /// `1..=u32::MAX`.
    SolitonDeltaOutOfRange {
        /// The rejected Q32 failure probability.
        delta_q32: u32,
    },
    /// A cumulative distribution table had no entries.
    EmptyCumulative,
    /// A cumulative distribution table decreased.
    ///
    /// Sampling searches the table, so a non-monotone entry would silently make
    /// a degree unreachable rather than fail.
    CumulativeNotMonotone {
        /// Index of the offending entry.
        index: usize,
        /// Preceding cumulative weight.
        previous: u64,
        /// Rejected cumulative weight.
        current: u64,
    },
    /// Accumulating a distribution's weights passed [`u64::MAX`].
    CumulativeOverflow {
        /// Index at which the accumulation overflowed.
        index: usize,
    },
    /// An RFC 5053 symbol count fell outside the normative range.
    Rfc5053SymbolCountOutOfRange {
        /// The rejected source-block symbol count.
        k: u32,
        /// Smallest count the specification allows.
        min: u32,
        /// Largest count the specification allows.
        max: u32,
    },
    /// A CSR parity-check matrix had no row offsets at all.
    ///
    /// The offsets array holds `rows + 1` entries, so it is never empty.
    CsrOffsetsEmpty,
    /// CSR row offsets decreased.
    CsrOffsetsNotMonotone {
        /// Row whose offset decreased.
        row: usize,
        /// Preceding offset.
        previous: usize,
        /// Rejected offset.
        current: usize,
    },
    /// A CSR row range extends past the column array.
    CsrRowRangePastEnd {
        /// Row whose range is out of bounds.
        row: usize,
        /// End of the rejected range.
        end: usize,
        /// Number of column entries supplied.
        len: usize,
    },
    /// A CSR entry names a column outside the matrix.
    CsrColumnOutOfRange {
        /// Row holding the entry.
        row: usize,
        /// The rejected column.
        column: u64,
        /// Number of columns in the matrix.
        columns: u64,
    },
    /// A finite generator was asked for a check outside its matrix.
    CheckOutOfRange {
        /// The rejected check identifier.
        check: u64,
        /// Number of check rows the generator holds.
        rows: u64,
    },
}

impl core::fmt::Display for GraphError {
    // One flat arm per variant. Splitting it into per-concern helpers would cost
    // the compiler's exhaustiveness check, which is the only thing guaranteeing a
    // new variant gets a message.
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IndexRetired { index, floor } => {
                write!(f, "index {index} was retired below horizon {floor}")
            }
            Self::LiveSpanExceeded {
                index,
                required,
                limit,
            } => write!(
                f,
                "index {index} needs a live span of {required}, over the limit of {limit}"
            ),
            Self::IndexNotRepresentable { index } => {
                write!(f, "index {index} does not fit this platform's usize")
            }
            Self::ZeroElementBytes => {
                write!(f, "edge weight has a zero packed-element width")
            }
            Self::ZeroSymbolLen => write!(f, "symbol length must be non-zero"),
            Self::SymbolLengthMismatch { expected, actual } => {
                write!(f, "symbol has length {actual}, expected {expected}")
            }
            Self::SymbolAlignment {
                length,
                element_bytes,
            } => write!(
                f,
                "symbol length {length} is not divisible by element width {element_bytes}"
            ),
            Self::HorizonRegressed { horizon, floor } => {
                write!(
                    f,
                    "retirement horizon {horizon} is below current floor {floor}"
                )
            }
            Self::CheckRetired { check } => {
                write!(f, "check {check} was explicitly retired")
            }
            Self::SampleSpanTooSmall { span, requested } => {
                write!(
                    f,
                    "cannot draw {requested} distinct offsets from span {span}"
                )
            }
            Self::ZeroDegree => {
                write!(f, "degree distribution can only produce zero edges")
            }
            Self::DegreeExceedsDomain { degree, domain } => {
                write!(f, "degree {degree} exceeds domain of {domain} variables")
            }
            Self::EmptyDomain => write!(f, "domain contains no variables"),
            Self::DomainTooLarge { domain, max } => {
                write!(f, "domain {domain} exceeds the addressable maximum {max}")
            }
            Self::DomainOverflow { base, span } => {
                write!(f, "window base {base} plus span {span} passes u64::MAX")
            }
            Self::EdgeLengthMismatch { support, weights } => {
                write!(f, "edge set has {support} variables but {weights} weights")
            }
            Self::EmptySupport => write!(f, "edge set has no support"),
            Self::DuplicateVariable { var } => {
                write!(f, "variable {var} occurs more than once in one edge set")
            }
            Self::ZeroWeight { var } => {
                write!(f, "edge on variable {var} has a zero coefficient")
            }
            Self::ZeroSymbolCount => write!(f, "symbol count must be non-zero"),
            Self::ZeroSolitonC => write!(f, "robust-soliton c must be non-zero"),
            Self::SolitonDeltaOutOfRange { delta_q32 } => write!(
                f,
                "robust-soliton delta {delta_q32} is outside the Q32 open interval 1..=u32::MAX"
            ),
            Self::EmptyCumulative => write!(f, "cumulative distribution table is empty"),
            Self::CumulativeNotMonotone {
                index,
                previous,
                current,
            } => write!(
                f,
                "cumulative weight {current} at index {index} is below the preceding {previous}"
            ),
            Self::CumulativeOverflow { index } => {
                write!(f, "cumulative weights overflow u64 at index {index}")
            }
            Self::Rfc5053SymbolCountOutOfRange { k, min, max } => {
                write!(f, "RFC 5053 symbol count {k} is outside {min}..={max}")
            }
            Self::CsrOffsetsEmpty => write!(f, "CSR row offsets are empty"),
            Self::CsrOffsetsNotMonotone {
                row,
                previous,
                current,
            } => write!(
                f,
                "CSR offset {current} for row {row} is below the preceding {previous}"
            ),
            Self::CsrRowRangePastEnd { row, end, len } => write!(
                f,
                "CSR row {row} ends at {end}, past the {len} column entries supplied"
            ),
            Self::CsrColumnOutOfRange {
                row,
                column,
                columns,
            } => write!(
                f,
                "CSR row {row} names column {column} outside a matrix of {columns} columns"
            ),
            Self::CheckOutOfRange { check, rows } => {
                write!(f, "check {check} is outside a matrix of {rows} rows")
            }
        }
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for GraphError {}

/// Residual-system assembly or elimination failed.
///
/// Validation completes before solver outcome metadata or graph state is
/// published.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveError {
    /// Residual columns were not strictly increasing.
    ColumnsNotStrictlyIncreasing {
        /// Column immediately before the rejected one.
        previous: u64,
        /// Rejected column.
        current: u64,
    },
    /// A row mentioned a variable absent from the explicit column set.
    UnknownTerm {
        /// Variable absent from the column set.
        var: u64,
    },
    /// A row's right-hand side differs from the first row's symbol length.
    RhsLengthMismatch {
        /// Established symbol length.
        expected: usize,
        /// Rejected row length.
        actual: usize,
    },
    /// A packed right-hand side ends with a partial field element.
    RhsAlignment {
        /// Rejected right-hand-side length.
        length: usize,
        /// Stable width of one field element.
        element_bytes: usize,
    },
    /// A residual symbol has zero bytes.
    ZeroSymbolLen,
    /// Matrix scratch geometry overflowed `usize`.
    GeometryOverflow {
        /// Number of equations.
        rows: usize,
        /// Number of unknown columns.
        columns: usize,
        /// Packed right-hand-side length.
        symbol_len: usize,
        /// In-memory width of one coefficient.
        coefficient_bytes: usize,
    },
    /// Elimination produced `0 = nonzero`.
    InconsistentSystem,
    /// Teaching a solved value to the sparse graph failed.
    Graph(GraphError),
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ColumnsNotStrictlyIncreasing { previous, current } => write!(
                f,
                "residual columns are not strictly increasing at {previous}, {current}"
            ),
            Self::UnknownTerm { var } => {
                write!(
                    f,
                    "residual row mentions variable {var} outside its columns"
                )
            }
            Self::RhsLengthMismatch { expected, actual } => {
                write!(f, "residual RHS has length {actual}, expected {expected}")
            }
            Self::RhsAlignment {
                length,
                element_bytes,
            } => write!(
                f,
                "residual RHS length {length} is not divisible by field width {element_bytes}"
            ),
            Self::ZeroSymbolLen => write!(f, "residual symbol length must be non-zero"),
            Self::GeometryOverflow {
                rows,
                columns,
                symbol_len,
                coefficient_bytes,
            } => write!(
                f,
                "residual geometry {rows}x{columns}, symbol length {symbol_len}, coefficient width {coefficient_bytes} overflows usize"
            ),
            Self::InconsistentSystem => write!(f, "residual system is inconsistent"),
            Self::Graph(error) => write!(f, "residual recovery could not update graph: {error}"),
        }
    }
}

impl From<GraphError> for SolveError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

#[cfg(feature = "std")]
impl ::std::error::Error for SolveError {
    fn source(&self) -> Option<&(dyn ::std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::ColumnsNotStrictlyIncreasing { .. }
            | Self::UnknownTerm { .. }
            | Self::RhsLengthMismatch { .. }
            | Self::RhsAlignment { .. }
            | Self::ZeroSymbolLen
            | Self::GeometryOverflow { .. }
            | Self::InconsistentSystem => None,
        }
    }
}
