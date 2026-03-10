// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

pub mod functor;
pub mod geometry;
pub mod hyper_graph_tensor;
pub mod kleisli;
pub mod mix;
// pub mod sparse; // Disabled (Might be safe but not needed)

pub use geometry::GeometryData;
pub use hyper_graph_tensor::{HyperGraphTensor, TensorConstraint, TensorNode};
pub use kleisli::{Admissible, KleisliArrow, KleisliPipeline};
pub use mix::{MaterialType, MixTensor, MIX_TENSOR_STRIDE};

// pub use sparse::SparseTensor;
