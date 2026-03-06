// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! ECS World Data Container
//! Holds the Collections of Entities and their corresponding Contexts.

use crate::ecs::components::{
    HydrationKinetics, MixtureComposition, RheologyProfile, ThermodynamicState, ToxicityProfile,
};
use crate::physics_kernel::IndustrialResult;

// A lightweight identifier for an Entity in the simulation (e.g., a Concrete Mix)
pub type Entity = u32;

/// A lightweight representation of the physical "World"
/// Currently scaled down to evaluate single mixtures
/// at a time for the RL agent, but architecturally allows N-Entities.
pub struct World {
    // Entities
    pub next_entity_id: Entity,

    // Stores
    pub compositions: std::collections::HashMap<Entity, MixtureComposition>,
    pub thermodynamics: std::collections::HashMap<Entity, ThermodynamicState>,
    pub rheologies: std::collections::HashMap<Entity, RheologyProfile>,
    pub hydration: std::collections::HashMap<Entity, HydrationKinetics>,
    pub toxicities: std::collections::HashMap<Entity, ToxicityProfile>,
    pub epistemic_states:
        std::collections::HashMap<Entity, crate::ecs::components::EpistemicStateComponent>,

    // Final Output Container (For bridging back to UI/RL)
    pub industrial_results: std::collections::HashMap<Entity, IndustrialResult>,
}

impl World {
    pub fn new() -> Self {
        Self {
            next_entity_id: 0,
            compositions: std::collections::HashMap::new(),
            thermodynamics: std::collections::HashMap::new(),
            rheologies: std::collections::HashMap::new(),
            hydration: std::collections::HashMap::new(),
            toxicities: std::collections::HashMap::new(),
            epistemic_states: std::collections::HashMap::new(),
            industrial_results: std::collections::HashMap::new(),
        }
    }

    pub fn spawn_entity(&mut self) -> Entity {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        id
    }
}
