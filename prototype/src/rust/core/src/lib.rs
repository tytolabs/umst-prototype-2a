// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
#![allow(clippy::new_without_default)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::let_and_return)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::get_first)]
#![allow(clippy::useless_vec)]

//
// UMST — Material Agnostic Operating System
// Minimal Core Profile — Prototype 2a
//

pub mod formulas;
pub mod io;
pub mod optimization;
pub mod physics_kernel;
pub mod rl;
pub mod safety;
pub mod science;
pub mod sensing;
pub mod tensors;
#[cfg(test)]
pub mod tests_physics;
// Modules removed from this release (not required for paper claims):
// geometry, ibe, ml, neural, oracle, physics, profiler, robotics, search, trust, validation

pub mod epistemic_proxy_selector;

// NEW: Data provider for real experiments (UCI dataset)
pub mod data_provider;
pub use data_provider::{ConcreteSample, ProxyDataSource, SyntheticDataProvider, UCIDataProvider};

// Re-export core types
pub use science::rheology::CartridgeRegistry;

pub use physics_kernel::PhysicsKernel;
pub use rl::{PPOAgent, RLAction, RLState, RewardFunction, RewardType};
pub use tensors::MixTensor;
pub mod constitution;
pub mod ecs;
pub mod gpu;
pub mod hardware;
pub mod math;
