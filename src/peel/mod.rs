//! Residual sparse graph and allocation-free peeling.
//!
//! Checks are reduced against known variables at ingest. The resident graph
//! stores only unknown support, with reverse adjacency from variables to checks;
//! one newly-known variable therefore updates only the rows that mention it.

mod graph;
mod peeler;
mod pool;

pub use graph::StalledRow;
pub use peeler::{Peeler, VariableState};
pub use pool::PoolConfig;

#[cfg(test)]
mod tests;
