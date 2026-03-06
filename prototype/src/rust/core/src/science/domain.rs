// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! Physical Domain Definitions
//! This module defines the strict, strongly-typed physical constituents
//! of the universe, decoupled from flat neural representations.

use crate::tensors::{MaterialType, MIX_TENSOR_STRIDE};
use serde::{Deserialize, Serialize};

impl From<u8> for MaterialType {
    fn from(val: u8) -> Self {
        match val {
            0 => MaterialType::Cement,
            1 => MaterialType::Aggregate,
            2 => MaterialType::Water,
            3 => MaterialType::Admixture,
            4 => MaterialType::Air,
            5 => MaterialType::SCM,
            6 => MaterialType::Fiber,
            7 => MaterialType::Nanomaterial,
            8 => MaterialType::Activator,
            9 => MaterialType::Lightweight,
            10 => MaterialType::Heavyweight,
            11 => MaterialType::Accelerator,
            12 => MaterialType::Retarder,
            13 => MaterialType::AirEntrainer,
            14 => MaterialType::Polymer,
            15 => MaterialType::Pigment,
            16 => MaterialType::Filler,
            _ => MaterialType::Air,
        }
    }
}

impl From<&crate::tensors::MixTensor> for Vec<MaterialComponent> {
    fn from(tensor: &crate::tensors::MixTensor) -> Self {
        let mut components = Vec::new();
        let num_materials = tensor.data().len() / MIX_TENSOR_STRIDE;
        for i in 0..num_materials {
            let offset = i * MIX_TENSOR_STRIDE;
            components.push(MaterialComponent {
                mass_kg: tensor.data()[offset],
                specific_gravity: tensor.data()[offset + 1],
                material_type: MaterialType::from(tensor.data()[offset + 2] as u8),
                co2_footprint: tensor.data()[offset + 3],
                cost: tensor.data()[offset + 4],
                blaine_fineness: tensor.data()[offset + 5],
                fineness_modulus: tensor.data()[offset + 6],
                shape_factor: tensor.data()[offset + 7],
            });
        }
        components
    }
}

/// A strongly-typed physical component within a mixture.
/// This replaces the brittle flat-array slicing (`data[i+7]`) used in the legacy kernel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialComponent {
    pub mass_kg: f32,
    pub specific_gravity: f32,
    pub material_type: MaterialType,
    pub co2_footprint: f32,
    pub cost: f32,
    pub blaine_fineness: f32,
    pub fineness_modulus: f32,
    pub shape_factor: f32, // 0.0 - 1.0 (1.0 = perfect sphere)
}

impl MaterialComponent {
    /// Calculate the absolute volume of this component in cubic meters.
    pub fn volume_m3(&self) -> f32 {
        if self.specific_gravity <= 0.0 {
            0.0
        } else {
            self.mass_kg / (self.specific_gravity * 1000.0)
        }
    }

    /// Helper to reliably check if this component is a solid particle.
    pub fn is_solid(&self) -> bool {
        !matches!(
            self.material_type,
            MaterialType::Water | MaterialType::Air | MaterialType::Admixture
        )
    }
}
