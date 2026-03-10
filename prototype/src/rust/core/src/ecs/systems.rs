// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! ECS System Traits
//! Defines isolated physics computations that operate purely on World State.

use crate::ecs::world::{Entity, World};
use crate::physics_kernel::PhysicsConfig;

/// An isolated scientific equation or logic engine that queries
/// and updates Entity components within the World.
pub trait System {
    fn run(&self, world: &mut World, entity: Entity, config: &PhysicsConfig);
}

// Example System (To be expanded in Phase 3.5)
pub struct PackingSystem;

impl System for PackingSystem {
    fn run(&self, world: &mut World, entity: Entity, _config: &PhysicsConfig) {
        if let Some(comp) = world.compositions.get(&entity) {
            let _packing =
                crate::physics_kernel::PhysicsKernel::compute_packing_density_ecs(&comp.materials);
            // Ideally store back into a Resource or newly spawned Component
        }
    }
}

pub struct EpistemicSelectionSystem {
    pub selector: std::sync::Mutex<crate::epistemic_proxy_selector::EpistemicProxySelector>,
}

impl System for EpistemicSelectionSystem {
    fn run(&self, world: &mut World, entity: Entity, _config: &PhysicsConfig) {
        if let Ok(mut selector) = self.selector.lock() {
            if let Some(proxy_name) = selector.select_next_proxy() {
                if let Ok(measurement) = selector.measure_proxy(&proxy_name) {
                    let epistemic_comp =
                        world.epistemic_states.entry(entity).or_insert_with(|| {
                            crate::ecs::components::EpistemicStateComponent {
                                measured_proxies: vec![],
                                proxy_values: std::collections::HashMap::new(),
                                convergence_score: 0.0,
                            }
                        });
                    epistemic_comp
                        .measured_proxies
                        .push(measurement.proxy_id.clone());
                    epistemic_comp
                        .proxy_values
                        .insert(measurement.proxy_id, measurement.value);
                    epistemic_comp.convergence_score =
                        selector.get_epistemic_state().convergence_score;
                }
            }
        }
    }
}
