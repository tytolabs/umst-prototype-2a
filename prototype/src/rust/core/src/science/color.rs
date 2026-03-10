// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! ColorMath implementation (Kubelka-Munk theory) for UMST Core.
//! Calculates theoretical reflectance and sRGB values from material mixtures.

use super::domain::MaterialComponent;
use crate::tensors::MaterialType;

/// Spectral properties (Absorption K and Scattering S)
struct SpectralProps {
    k: [f32; 3], // RGB Absorption
    s: [f32; 3], // RGB Scattering
}

impl SpectralProps {
    fn new(k: [f32; 3], s: [f32; 3]) -> Self {
        SpectralProps { k, s }
    }
}

/// Helper to get generic spectral properties based on MaterialType.
/// In a production scenario, this would index via a specific ID or database trait.
fn get_spectral_props(mat_type: &MaterialType) -> SpectralProps {
    match mat_type {
        MaterialType::Cement => SpectralProps::new([0.15, 0.12, 0.10], [0.5, 0.5, 0.5]),
        MaterialType::SCM => SpectralProps::new([0.05, 0.04, 0.03], [0.7, 0.7, 0.7]),
        MaterialType::Aggregate => SpectralProps::new([0.08, 0.07, 0.05], [0.65, 0.60, 0.50]),
        // Default to a generic Bayferrox 130 Red representation for Pigment for now
        MaterialType::Pigment => SpectralProps::new([0.5, 3.25, 3.75], [0.6, 0.2, 0.15]),
        MaterialType::Water | MaterialType::Air | MaterialType::Admixture => {
            // Invisible/Transparent for base reflectance model
            SpectralProps::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        }
        _ => SpectralProps::new([0.1, 0.1, 0.1], [0.5, 0.5, 0.5]), // Generic Gray
    }
}

pub struct ColorMath;

impl ColorMath {
    /// Calculate Mix Color using Duncan-Lax Mixing Law (Kubelka-Munk)
    /// Returns (R, G, B) in 0-255 range
    pub fn calculate_mix_color(components: &[MaterialComponent]) -> (u8, u8, u8) {
        let mut total_mass = 0.0;
        let mut k_mix = [0.0, 0.0, 0.0];
        let mut s_mix = [0.0, 0.0, 0.0];

        for c in components {
            if c.mass_kg <= 0.0 || !c.is_solid() {
                continue; // Skip water/air/liquid admixtures for base solid color
            }

            let props = get_spectral_props(&c.material_type);

            for i in 0..3 {
                k_mix[i] += props.k[i] * c.mass_kg;
                s_mix[i] += props.s[i] * c.mass_kg;
            }

            total_mass += c.mass_kg;
        }

        if total_mass == 0.0 {
            return (128, 128, 128); // Default Gray
        }

        let mut rgb_out = [0.0; 3];

        for i in 0..3 {
            // Avoid div by zero
            if s_mix[i] == 0.0 {
                rgb_out[i] = 0.0;
                continue;
            }

            let ks = k_mix[i] / s_mix[i];

            // Validate math constraints for KM theory: K/S = (1-R)^2 / 2R => solve for R
            // R^2 - 2R(1 + K/S) + 1 = 0
            // R = 1 + K/S - sqrt((K/S)^2 + 2(K/S))
            let r_base = 1.0 + ks - (ks * ks + 2.0 * ks).sqrt();
            let r_clamped = r_base.clamp(0.0, 1.0);

            // Gamma Correction (Approx 2.2) to sRGB
            let srgb = r_clamped.powf(1.0 / 2.2) * 255.0;
            rgb_out[i] = srgb.clamp(0.0, 255.0);
        }

        (
            rgb_out[0].round() as u8,
            rgb_out[1].round() as u8,
            rgb_out[2].round() as u8,
        )
    }
}
