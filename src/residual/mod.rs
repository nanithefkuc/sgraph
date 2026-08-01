//! Exact residual-system assembly and reduced-row-echelon solving.

mod builder;
mod row;
mod solver;

pub use builder::{ResidualBuilder, RowSink, System};
pub use row::{DenseRow, Row};
pub use solver::{Report, Solver};

#[cfg(test)]
mod tests;
