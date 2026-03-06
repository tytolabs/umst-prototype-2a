// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! RL Optimizer Module
//!
//! Reinforcement Learning for Autonomous Mix Optimization (Blueprint Section 6.4)
//! Implements multiple reward functions and PPO-style policy optimization.
//!
//! Modular Design:
//! - traits.rs: IRewardProvider, IDataProvider interfaces
//! - concrete_provider.rs: Concrete cartridge implementation
//! - ppo.rs: Material-agnostic PPO agent
//! - epistemic.rs: Mutual information estimation and epistemic state tracking
//! - epistemic_ppo.rs: Epistemic PPO with MI-guided exploration
//! - quantum_bounds.rs: Thermodynamic bounds for learning

pub mod concrete_provider;
pub mod ewc;
pub mod explainability;
pub mod federated;
pub mod guardrails;
pub mod liquid_ppo;
pub mod ppo;
mod reward;
pub mod state;
pub mod traits;

// Epistemic modules for paper claims
pub mod epistemic;
pub mod epistemic_ppo;
pub mod quantum_bounds;

pub use concrete_provider::ConcreteRewardProvider;
pub use ewc::EwcPenalty;
pub use explainability::{SemanticRegistry, XaiExplanation, XaiTranslator};
pub use federated::{FederatedAggregator, FederatedNode};
pub use guardrails::{GuardrailEngine, GuardrailValidation, PhysicsGuardrails, ViolationSeverity};
pub use ppo::{PPOAgent, PPOConfig, GradientVelocityTracker, MutualInformationTracker, MetaStats};
pub use reward::{RewardComponents, RewardConfig, RewardFunction, RewardType};
pub use state::{RLAction, RLState};
pub use traits::{CartridgeInfo, IDataProvider, IRewardProvider, IScienceCartridge, MaterialData};

// Re-export epistemic types
pub use epistemic::{EpistemicStateTracker, IntrinsicCuriosity, MutualInfoEstimator};
pub use epistemic_ppo::{EpistemicPPOConfig, EpistemicPPOModule, EpistemicRewardCalculator};
pub use quantum_bounds::{FluctuationTheorem, QuantumThermoBounds, ThermodynamicLRBound};
pub mod environment;
