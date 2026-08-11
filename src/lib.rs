//! # Sparse Graph
//!
//! The sparse/Tanner-graph engine that erasure-coding families keep
//! re-implementing: deterministic neighbour generation, degree distributions,
//! a residual sparse graph, XOR-only peeling, and the exact residual solve that
//! finishes what peeling cannot. LDPC, LT, and Raptor-class codes differ in how
//! their graph is generated, not in how it is consumed.
//!
//! Field arithmetic and byte-buffer vector primitives come from [`fgf`]; this
//! crate never re-implements field arithmetic. Wire formats, packet headers,
//! transport and HARQ policy, belief-propagation soft-decision decoding,
//! protograph lifting, and codec shells stay with the consumer.
//!
//! [`fgf`]: https://github.com/nanithefkuc/fgf
//!
//! ## Layout
//! * [`rng`] and [`neighbors`] regenerate deterministic edge sets from check ids.
//! * [`degree`] supplies check-degree distributions.
//! * [`index`] provides bounded monotone [`Ring`](index::Ring) and
//!   [`IndexSet`](index::IndexSet) storage.
//! * [`weight`] separates sparse edge coefficients from their residual-field
//!   embedding.
//! * [`peel`] owns the shrinking sparse graph, reverse adjacency, and cascade.
//! * [`residual`] assembles explicit-column systems and solves them to full RREF.
//! * [`driver`] runs the peel → solve → re-peel fixpoint.
//! * [`error`] contains [`GraphError`] and [`SolveError`].
//!
//! ## Determinism
//!
//! Edge generation is reproducible across peers and across runs, which is what
//! lets a check symbol travel without its graph:
//!
//! ```
//! use sgraph::rng::{SplitMix64, distinct_offsets, seed_for};
//!
//! // A domain-separation constant is the consumer's to choose; it keeps this
//! // edge stream distinct from any other use of the same check ids.
//! const DOMAIN: u64 = 0xA5A5_5A5A_C3C3_3C3C;
//!
//! let mut edges = [0u32; 3];
//! let mut rng = SplitMix64::new(seed_for(42, DOMAIN));
//! distinct_offsets(&mut rng, 64, &mut edges)?;
//!
//! // The far side, holding only the check id, recomputes the same set.
//! let mut peer = [0u32; 3];
//! let mut peer_rng = SplitMix64::new(seed_for(42, DOMAIN));
//! distinct_offsets(&mut peer_rng, 64, &mut peer)?;
//! assert_eq!(edges, peer);
//! # Ok::<(), sgraph::GraphError>(())
//! ```
//!
//! ## Feature flags
//!
//! * `std` (default) — enables `fgf`'s runtime CPU detection and its
//!   process-wide backend cache. Without it the crate is `no_std` + `alloc`.
//! * `simd` (default, implies `std`) — runtime-dispatched SIMD kernels from
//!   `fgf`. Without it, portable scalar.
//! * `internals` — exposes unstable implementation APIs. Nothing behind it is
//!   covered by this crate's compatibility guarantees.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
#![warn(clippy::pedantic)]
// Index arithmetic is `u64` in the domain and `usize` in storage. Every such
// cast is bounded by the live span, which is a `usize` count by construction.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::module_name_repetitions)]

extern crate alloc;

pub mod degree;
pub mod driver;
pub mod error;
pub mod id;
pub mod index;
pub mod neighbors;
pub mod peel;
pub mod residual;
pub mod rng;
pub mod weight;

pub use crate::degree::{Constant, Cumulative, DegreeDistribution, RobustSoliton};
pub use crate::driver::{DenseRows, Resolver};
pub use crate::error::{GraphError, SolveError};
pub use crate::id::{CheckId, VarId};
pub use crate::neighbors::{
    Edges, ExplicitMatrix, NeighborBuf, NeighborGen, Rfc5053Triple, Uniform, WindowedUniform,
};
pub use crate::peel::{Peeler, PoolConfig, StalledRow, VariableState};
pub use crate::residual::{DenseRow, Report, ResidualBuilder, Row, RowSink, Solver, System};
pub use crate::weight::{Binary, EdgeWeight, ResidualCoeff};
