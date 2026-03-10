// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

/// ═══════════════════════════════════════════════════════════════════════════
/// MaterialType: Category-Theoretic Material Classification
/// ═══════════════════════════════════════════════════════════════════════════
///
/// This enum forms a finite category where:
/// - Objects: Each variant is a distinct material class
/// - Morphisms: from_str is a functor String → MaterialType
/// - Natural transformations: Physics engines map MaterialType → PhysicsResult
///
/// The discriminant values are stable for backward compatibility:
/// - 0-5: Original V8.3 types (Cement, Aggregate, Water, Admixture, Air, SCM)
/// - 6-15: V8.5 extension types for expanded material warehouse
///
/// Type Safety Invariant:
/// ∀ m: MaterialType, ∃! engine_set ⊆ PhysicsEngines that activates for m
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialType {
    // ═══════════════════════════════════════════════════════════════════════
    // ORIGINAL TYPES (V8.3) - Stable discriminants for backward compatibility
    // ═══════════════════════════════════════════════════════════════════════
    Cement = 0,    // Portland cement, blended cements (EN 197-1)
    Aggregate = 1, // Normal-weight aggregates (granite, limestone, basalt)
    Water = 2,     // Mixing water (potable, recycled, seawater)
    Admixture = 3, // Chemical admixtures (superplasticizers, general)
    Air = 4,       // Entrained or entrapped air (default/fallback)
    SCM = 5,       // Supplementary cementitious materials (fly ash, slag, silica fume)

    // ═══════════════════════════════════════════════════════════════════════
    // EXTENDED TYPES (V8.5) - New material categories
    // ═══════════════════════════════════════════════════════════════════════
    /// Fibers: Steel, synthetic (PP, PVA), basalt, carbon, natural
    /// Physics: FiberEngine → Tensile boost, toughness, crack bridging
    Fiber = 6,

    /// Nanomaterials: Nano-SiO₂, CNT, graphene oxide, nano-TiO₂
    /// Physics: NanoEngine → Strength gain, pore refinement, durability
    Nanomaterial = 7,

    /// Geopolymer activators: NaOH, Na-silicate, KOH solutions
    /// Physics: GeopolymerEngine → NASH gel formation (alternative binder)
    Activator = 8,

    /// Lightweight aggregates: LECA, expanded shale, pumice, perlite
    /// Physics: LightweightEngine → Density reduction, thermal insulation
    Lightweight = 9,

    /// Heavyweight aggregates: Magnetite, hematite, barite (radiation shielding)
    /// Physics: ShieldingEngine → Attenuation coefficient
    Heavyweight = 10,

    /// Accelerators: CaCl₂, Ca-nitrate, Ca-formate
    /// Physics: SetTimeEngine → Reduced set time, early strength
    Accelerator = 11,

    /// Retarders: Citric acid, tartaric acid, gluconate
    /// Physics: SetTimeEngine → Extended set time, workability
    Retarder = 12,

    /// Air entrainers: Vinsol resin, synthetic AEA
    /// Physics: FreezeThawEngine → Air void spacing, durability factor
    AirEntrainer = 13,

    /// Polymer modifiers: SBR latex, acrylic, epoxy
    /// Physics: PolymerEngine → Bond strength, impermeability
    Polymer = 14,

    /// Pigments: Iron oxides (Bayferrox), chromium oxide, titanium dioxide
    /// Physics: Minimal (aesthetics), but affects rheology slightly
    Pigment = 15,

    /// Fillers: Quartz flour, limestone filler, calcium carbonate
    /// Physics: Packing density improvement, minimal hydraulic activity
    Filler = 16,
}

impl MaterialType {
    /// Parse material type string from TypeScript/JSON to Rust enum.
    ///
    /// ═══════════════════════════════════════════════════════════════════════
    /// FUNCTOR: String → MaterialType
    /// ═══════════════════════════════════════════════════════════════════════
    ///
    /// This function is a functor from the free category of strings to the
    /// finite category of MaterialType. The mapping preserves:
    ///
    /// 1. Identity: from_str(canonical_name(T)) = T for all T
    /// 2. Composition: Multiple aliases map to the same canonical type
    ///
    /// [V8.5] Extended to handle all 16 material categories with explicit
    /// type discrimination (no more fallback to Admixture for specialized types)
    ///
    /// Graph-Theoretic View:
    /// ```text
    /// String aliases ──(from_str)──→ MaterialType ──(physics_engines)──→ Results
    ///     "fiber"    ─────────────→   Fiber(6)   ──→ FiberEngine
    ///     "nano_sio2"─────────────→ Nanomaterial(7)──→ NanoEngine
    /// ```
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            // ═══════════════════════════════════════════════════════════════
            // BINDERS (Type 0, 5)
            // ═══════════════════════════════════════════════════════════════
            "cement" | "portland" | "opc" | "cem_i" | "cem_ii" | "cem_iii" | "cem_iv" | "cem_v" => {
                MaterialType::Cement
            }

            "scm" | "flyash" | "fly_ash" | "slag" | "ggbs" | "silica_fume" | "metakaolin"
            | "calcined_clay" | "pozzolan" | "rice_husk_ash" | "volcanic_ash" | "pumice_ground" => {
                MaterialType::SCM
            }

            // ═══════════════════════════════════════════════════════════════
            // AGGREGATES (Types 1, 9, 10)
            // ═══════════════════════════════════════════════════════════════
            // Normal weight aggregates
            "aggregate" | "sand" | "gravel" | "granite" | "limestone" | "basalt" | "quartzite"
            | "dolomite" | "recycled" | "rca" | "crushed_sand" | "river_sand" | "sea_sand" => {
                MaterialType::Aggregate
            }

            // Lightweight aggregates (special physics: density reduction)
            "lightweight" | "leca" | "expanded_clay" | "expanded_shale" | "expanded_slate"
            | "pumice" | "scoria" | "perlite" | "vermiculite" | "foamed_glass" => {
                MaterialType::Lightweight
            }

            // Heavyweight aggregates (special physics: radiation shielding)
            "heavyweight" | "magnetite" | "hematite" | "barite" | "ilmenite" | "steel_shot"
            | "limonite" => MaterialType::Heavyweight,

            // ═══════════════════════════════════════════════════════════════
            // WATER (Type 2)
            // ═══════════════════════════════════════════════════════════════
            "water" | "potable_water" | "deionized" | "recycled_water" | "seawater" => {
                MaterialType::Water
            }

            // ═══════════════════════════════════════════════════════════════
            // ADMIXTURES - CHEMICAL (Types 3, 11, 12, 13)
            // ═══════════════════════════════════════════════════════════════
            // General admixtures (superplasticizers, VMA, SRA)
            "admixture"
            | "superplasticizer"
            | "pce"
            | "snf"
            | "smf"
            | "lignosulfonate"
            | "vma"
            | "sra"
            | "corrosion_inhibitor"
            | "anti_washout"
            | "pra" => MaterialType::Admixture,

            // Accelerators (special physics: set time reduction)
            "accelerator" | "calcium_chloride" | "calcium_nitrate" | "calcium_formate"
            | "sodium_thiocyanate" | "triethanolamine" | "shotcrete_accel" => {
                MaterialType::Accelerator
            }

            // Retarders (special physics: set time extension)
            "retarder" | "citric_acid" | "tartaric_acid" | "gluconate" | "phosphate_retarder"
            | "sugar" | "molasses" => MaterialType::Retarder,

            // Air entrainers (special physics: freeze-thaw durability)
            "air_entrainer" | "aer" | "vinsol" | "tall_oil" | "synthetic_aea" => {
                MaterialType::AirEntrainer
            }

            // ═══════════════════════════════════════════════════════════════
            // FIBERS (Type 6)
            // ═══════════════════════════════════════════════════════════════
            "fiber" | "steel_fiber" | "pp_fiber" | "pva_fiber" | "basalt_fiber"
            | "carbon_fiber" | "glass_fiber" | "ar_glass" | "cellulose_fiber" | "hemp_fiber"
            | "nylon_fiber" | "polypropylene" | "macro_fiber" | "micro_fiber" => {
                MaterialType::Fiber
            }

            // ═══════════════════════════════════════════════════════════════
            // NANOMATERIALS (Type 7)
            // ═══════════════════════════════════════════════════════════════
            "nanomaterial" | "nano" | "nano_silica" | "nano_sio2" | "cnt" | "mwcnt" | "swcnt"
            | "graphene" | "graphene_oxide" | "nano_tio2" | "nano_caco3" | "nano_al2o3"
            | "nano_fe2o3" | "nano_zno" | "nanoplatelet" => MaterialType::Nanomaterial,

            // ═══════════════════════════════════════════════════════════════
            // GEOPOLYMER ACTIVATORS (Type 8)
            // ═══════════════════════════════════════════════════════════════
            "activator"
            | "sodium_hydroxide"
            | "naoh"
            | "sodium_silicate"
            | "waterglass"
            | "potassium_hydroxide"
            | "koh"
            | "potassium_silicate"
            | "calcium_hydroxide" => MaterialType::Activator,

            // ═══════════════════════════════════════════════════════════════
            // POLYMER MODIFIERS (Type 14)
            // ═══════════════════════════════════════════════════════════════
            "polymer"
            | "latex"
            | "sbr"
            | "styrene_butadiene"
            | "acrylic"
            | "epoxy"
            | "polyurethane"
            | "redispersible_powder"
            | "eva" => MaterialType::Polymer,

            // ═══════════════════════════════════════════════════════════════
            // PIGMENTS (Type 15)
            // ═══════════════════════════════════════════════════════════════
            "pigment" | "iron_oxide" | "bayferrox" | "colortherm" | "titanium_dioxide"
            | "chromium_oxide" | "carbon_black" => MaterialType::Pigment,

            // ═══════════════════════════════════════════════════════════════
            // FILLERS (Type 16)
            // ═══════════════════════════════════════════════════════════════
            "filler"
            | "quartz_flour"
            | "limestone_filler"
            | "calcium_carbonate"
            | "pcc"
            | "glass_powder"
            | "wollastonite"
            | "calcium_aluminate"
            | "calcium_sulfoaluminate" => MaterialType::Filler,

            // ═══════════════════════════════════════════════════════════════
            // FALLBACK (Type 4)
            // ═══════════════════════════════════════════════════════════════
            // Unknown types default to Air (minimal physics impact)
            // This ensures type safety: every string maps to exactly one type
            _ => MaterialType::Air,
        }
    }

    /// Get the canonical name for this material type
    /// This is the inverse of from_str for the canonical form
    pub fn canonical_name(&self) -> &'static str {
        match self {
            MaterialType::Cement => "cement",
            MaterialType::Aggregate => "aggregate",
            MaterialType::Water => "water",
            MaterialType::Admixture => "admixture",
            MaterialType::Air => "air",
            MaterialType::SCM => "scm",
            MaterialType::Fiber => "fiber",
            MaterialType::Nanomaterial => "nanomaterial",
            MaterialType::Activator => "activator",
            MaterialType::Lightweight => "lightweight",
            MaterialType::Heavyweight => "heavyweight",
            MaterialType::Accelerator => "accelerator",
            MaterialType::Retarder => "retarder",
            MaterialType::AirEntrainer => "air_entrainer",
            MaterialType::Polymer => "polymer",
            MaterialType::Pigment => "pigment",
            MaterialType::Filler => "filler",
        }
    }

    /// Check if this material type affects rheology calculations
    pub fn affects_rheology(&self) -> bool {
        matches!(
            self,
            MaterialType::Cement
                | MaterialType::SCM
                | MaterialType::Admixture
                | MaterialType::Fiber
                | MaterialType::Nanomaterial
                | MaterialType::Polymer
                | MaterialType::Pigment
                | MaterialType::Lightweight
        )
    }

    /// Check if this material type contributes to binder content
    pub fn is_binder(&self) -> bool {
        matches!(
            self,
            MaterialType::Cement | MaterialType::SCM | MaterialType::Activator
        )
    }

    /// Check if this material type is a solid (for packing calculations)
    pub fn is_solid(&self) -> bool {
        !matches!(self, MaterialType::Water | MaterialType::Air)
    }
}

/// Incoming material from TypeScript (JSON)
/// This struct is designed to be FLEXIBLE and handle the complex TS Material schema
/// using `#[serde(default)]` for all optional fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialInput {
    pub id: String,
    #[serde(rename = "type", default)]
    pub material_type: String,
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default)]
    pub ecology: Option<EcologyInput>,
    #[serde(default)]
    pub economy: Option<EconomyInput>,
    #[serde(default)]
    pub properties: Option<PhysicsPropertiesInput>,
    #[serde(default)]
    pub rheology: Option<RheologyInput>,
    // Ignore all other fields from TS schema
    #[serde(flatten)]
    #[allow(dead_code)]
    _extra: std::collections::HashMap<String, serde_json::Value>,
}

fn default_density() -> f32 {
    2400.0
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EcologyInput {
    #[serde(rename = "embodiedCarbon", default)]
    pub embodied_carbon: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EconomyInput {
    #[serde(rename = "costPerKg", default)]
    pub cost_per_kg: f32,
}

/// Rheology properties from TypeScript Material.properties
/// Material-specific rheology parameters for accurate physics modeling
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RheologyInput {
    #[serde(default)]
    pub viscosity: f32, // Pa.s - plastic viscosity
    #[serde(rename = "yieldStress", default)]
    pub yield_stress: f32, // Pa - yield stress
    #[serde(default)]
    pub thixotropy: f32, // Pa/s - thixotropy index
}

/// ═══════════════════════════════════════════════════════════════════════════
/// PhysicsPropertiesInput: Material Properties for Physics Calculations
/// ═══════════════════════════════════════════════════════════════════════════
///
/// This struct captures all physics-relevant material properties from TypeScript.
/// Properties are organized by material category for clarity.
///
/// [V8.5] Extended to support:
/// - Fiber properties (length, diameter, aspect_ratio, tensile_strength)
/// - SCM properties (k_factor, reactivity)
/// - Nanomaterial properties (ssa, strength_gain)
/// - Set modifier properties (set_time_change)
/// - Air entrainer properties (air_content)
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PhysicsPropertiesInput {
    // ═══════════════════════════════════════════════════════════════════════
    // CEMENT PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Blaine fineness (m²/kg) - affects hydration rate and rheology
    #[serde(default)]
    pub blaine: f32,

    /// C3S content (%) - tricalcium silicate for early strength
    #[serde(rename = "c3s_content", default)]
    pub c3s_content: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // SCM PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// CO2 per kg material - for sustainability calculation
    #[serde(rename = "co2_kg", default)]
    pub co2_kg: f32,

    /// K-factor (cement efficiency factor) - for W/C equivalent calculation
    /// Typical values: Fly Ash 0.3-0.5, Slag 0.6-0.9, Silica Fume 2.0
    #[serde(rename = "k_factor", default)]
    pub k_factor: f32,

    /// Reactivity coefficient (0-1+) - pozzolanic/hydraulic activity index
    #[serde(default)]
    pub reactivity: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // AGGREGATE PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Fineness modulus - aggregate grading (0-8 scale)
    #[serde(default)]
    pub fm: f32,

    /// Water absorption (%) - affects effective W/C
    #[serde(default)]
    pub absorption: f32,

    /// Shape factor (0-1, 1=spherical) - affects packing
    #[serde(default)]
    pub shape: f32,

    /// Moisture content (%) - current moisture in aggregates
    #[serde(default)]
    pub moisture: f32,

    /// Maximum aggregate size (mm) - for ITZ calculations
    #[serde(rename = "max_size", default)]
    pub max_size: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIXTURE PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Dosage (% by cement weight) - for admixtures
    #[serde(default)]
    pub dosage: f32,

    /// Set time change (%) - positive = retard, negative = accelerate
    #[serde(rename = "set_time_change", default)]
    pub set_time_change: f32,

    /// Target air content (%) - for air entrainers
    #[serde(rename = "air_content", default)]
    pub air_content: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // FIBER PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Fiber length (mm)
    #[serde(default)]
    pub length: f32,

    /// Fiber diameter (mm)
    #[serde(default)]
    pub diameter: f32,

    /// Fiber aspect ratio (L/d) - calculated or explicit
    #[serde(rename = "aspect_ratio", default)]
    pub aspect_ratio: f32,

    /// Fiber tensile strength (MPa)
    #[serde(rename = "tensile_strength", default)]
    pub tensile_strength: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // NANOMATERIAL PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Specific surface area (m²/g) - for nanomaterials
    /// Also used as "ssa" in TypeScript materials
    #[serde(alias = "ssa", rename = "ssa_m2g", default)]
    pub ssa_m2g: f32,

    /// Strength gain factor (%) - expected strength increase
    #[serde(rename = "strength_gain", default)]
    pub strength_gain: f32,

    // ═══════════════════════════════════════════════════════════════════════
    // POLYMER PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════
    /// Minimum Film Temperature (°C) - for polymer film formation
    #[serde(default)]
    pub mft: f32,

    /// Flexibility factor (0-1) - polymer flexibility contribution
    #[serde(default)]
    pub flexibility: f32,
}

/// Incoming mix component from TypeScript (JSON)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixComponentInput {
    #[serde(rename = "materialId")]
    pub material_id: String,
    #[serde(default)]
    pub mass: f32,
}

/// ═══════════════════════════════════════════════════════════════════════════
/// MixTensor: Unified Material Tensor for Physics Calculations
/// ═══════════════════════════════════════════════════════════════════════════
///
/// The MixTensor is a dense vector storing all material properties in a flat
/// layout optimized for WASM performance. Each material occupies STRIDE floats.
///
/// [V9.0] Extended stride from 15 to 17 to support RAC characterization:
/// - absorption: Water absorption (%) - affects effective W/C
/// - moisture: Current moisture content (%) - affects mix water
///
/// Memory Layout (STRIDE = 17):
/// ```text
/// Index | Property        | Unit    | Used By
/// ──────┼─────────────────┼─────────┼──────────────────────────
///   0   | mass            | kg      | All engines
///   1   | sg              | -       | Packing, Volume
///   2   | type_id         | enum    | Engine routing
///   3   | co2             | kg/kg   | Sustainability
///   4   | cost            | $/kg    | Economics
///   5   | blaine          | m²/kg   | Hydration
///   6   | fm              | -       | Rheology (friction)
///   7   | shape           | 0-1     | Rheology (packing)
///   8   | viscosity       | Pa.s    | Rheology
///   9   | yield_stress    | Pa      | Rheology
///  10   | thixotropy      | Pa/s    | Rheology
///  11   | k_factor        | -       | W/C equivalent (SCM)
///  12   | reactivity      | 0-1+    | Hydration rate (SCM)
///  13   | aspect_ratio    | L/d     | Fiber efficiency
///  14   | tensile_strength| MPa     | Fiber tensile
///  15   | absorption      | %       | Effective W/C (RAC)
///  16   | moisture        | %       | Mix water correction
/// ```
#[wasm_bindgen]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixTensor {
    data: Vec<f32>,
}

/// Stride constant for MixTensor - number of f32 values per material
pub const MIX_TENSOR_STRIDE: usize = 17;

#[wasm_bindgen]
impl MixTensor {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MixTensor {
        MixTensor { data: Vec::new() }
    }

    /// Hydrate tensor directly from JSON (TypeScript sends raw JSON strings)
    /// This moves ALL marshalling logic into Rust.
    ///
    /// [V8.5] Extended to populate new fields:
    /// - k_factor (SCM efficiency)
    /// - reactivity (pozzolanic activity)
    /// - aspect_ratio (fiber L/d)
    /// - tensile_strength (fiber tensile)
    #[wasm_bindgen]
    pub fn from_json(components_json: &str, materials_json: &str) -> Result<MixTensor, JsValue> {
        let components: Vec<MixComponentInput> = serde_json::from_str(components_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse components: {}", e)))?;

        let materials: Vec<MaterialInput> = serde_json::from_str(materials_json)
            .map_err(|e| JsValue::from_str(&format!("Failed to parse materials: {}", e)))?;

        let mut tensor = MixTensor::new();
        let mut missing_materials: Vec<String> = Vec::new();

        for comp in &components {
            // Find material by ID
            if let Some(mat) = materials.iter().find(|m| m.id == comp.material_id) {
                let sg = mat.density / 1000.0; // Convert density to specific gravity
                let mat_type = MaterialType::from_str(&mat.material_type);
                let type_id = mat_type as u8;
                let co2 = mat
                    .ecology
                    .as_ref()
                    .map(|e| e.embodied_carbon)
                    .unwrap_or(0.0);
                let cost = mat.economy.as_ref().map(|e| e.cost_per_kg).unwrap_or(0.0);

                // Extract Physics Properties (with category-aware defaults)
                let props = mat.properties.as_ref();

                // ═══════════════════════════════════════════════════════════════
                // [V8.6] TYPE-AWARE PROPERTY MAPPING
                // The MixTensor has 15 slots per material. Different material types
                // use these slots differently:
                //   Slot 5 (blaine): Cement->blaine, Nano->ssa_m2g
                //   Slot 7 (shape): Agg->shape, Polymer->mft, Cement->c3s_content
                //   Slot 12 (reactivity): SCM->reactivity, Admixture->set_time_change,
                //                         AirEntrainer->air_content, Polymer->flexibility
                // ═══════════════════════════════════════════════════════════════

                // Slot 5: blaine (Cement) OR ssa (Nanomaterial)
                let blaine = match mat_type {
                    MaterialType::Nanomaterial => props
                        .map(|p| if p.ssa_m2g > 0.0 { p.ssa_m2g } else { p.blaine })
                        .unwrap_or(200.0),
                    _ => props.map(|p| p.blaine).unwrap_or(0.0),
                };

                // Slot 6: fineness modulus (Aggregate)
                let fm = props.map(|p| p.fm).unwrap_or(0.0);

                // Slot 7: shape (Aggregate) OR mft (Polymer) OR c3s (Cement)
                let shape = match mat_type {
                    MaterialType::Polymer => {
                        // For polymers, slot 7 holds MFT (minimum film temperature)
                        props
                            .map(|p| if p.mft > 0.0 { p.mft } else { p.shape })
                            .unwrap_or(15.0)
                    }
                    MaterialType::Cement => {
                        // For cement, slot 7 can hold C3S content
                        props
                            .map(|p| {
                                if p.c3s_content > 0.0 {
                                    p.c3s_content / 100.0
                                } else {
                                    p.shape
                                }
                            })
                            .unwrap_or(0.55)
                    }
                    MaterialType::Aggregate
                    | MaterialType::Lightweight
                    | MaterialType::Heavyweight => props.map(|p| p.shape).unwrap_or(0.6),
                    _ => props.map(|p| p.shape).unwrap_or(0.5),
                };

                // [V8.5] K-factor with intelligent defaults
                let k_factor = props.map(|p| p.k_factor).unwrap_or_else(|| {
                    if mat_type == MaterialType::SCM {
                        if mat.id.contains("silica") {
                            2.0
                        } else if mat.id.contains("slag") || mat.id.contains("ggbs") {
                            0.9
                        } else if mat.id.contains("fly") || mat.id.contains("ash") {
                            0.5
                        } else if mat.id.contains("metakaolin") {
                            1.5
                        } else {
                            0.5
                        }
                    } else {
                        0.0
                    }
                });

                // Slot 12: reactivity (SCM) OR set_time_change (Admixture) OR air_content (AEA) OR flexibility (Polymer)
                let reactivity = match mat_type {
                    MaterialType::Accelerator | MaterialType::Retarder => {
                        // For set-time admixtures, store set_time_change as % (-100 to +100)
                        props.map(|p| p.set_time_change).unwrap_or_else(|| {
                            if mat_type == MaterialType::Accelerator {
                                -30.0
                            } else {
                                50.0
                            } // Retarder
                        })
                    }
                    MaterialType::AirEntrainer => {
                        // For air entrainers, store target air content %
                        props.map(|p| p.air_content).unwrap_or(5.0)
                    }
                    MaterialType::Polymer => {
                        // For polymers, store flexibility factor
                        props.map(|p| p.flexibility).unwrap_or(0.5)
                    }
                    MaterialType::SCM => props.map(|p| p.reactivity).unwrap_or(0.5),
                    MaterialType::Nanomaterial => props.map(|p| p.reactivity).unwrap_or(1.0),
                    _ => props.map(|p| p.reactivity).unwrap_or(0.0),
                };

                // Slot 13: aspect_ratio (Fiber) - calculated from length/diameter if not explicit
                let aspect_ratio = props
                    .map(|p| {
                        if p.aspect_ratio > 0.0 {
                            p.aspect_ratio
                        } else if p.length > 0.0 && p.diameter > 0.0 {
                            p.length / p.diameter
                        } else {
                            0.0
                        }
                    })
                    .unwrap_or(0.0);

                // Slot 14: tensile_strength (Fiber) with intelligent defaults
                let tensile_strength = props.map(|p| p.tensile_strength).unwrap_or_else(|| {
                    if mat_type == MaterialType::Fiber {
                        if mat.id.contains("steel") {
                            1300.0
                        } else if mat.id.contains("carbon") {
                            4000.0
                        } else if mat.id.contains("basalt") {
                            3000.0
                        } else if mat.id.contains("pp") || mat.id.contains("polypropylene") {
                            500.0
                        } else if mat.id.contains("pva") {
                            1600.0
                        } else {
                            500.0
                        }
                    } else {
                        0.0
                    }
                });

                // [V9.0] Extract RAC properties: absorption and moisture
                let absorption = props.map(|p| p.absorption).unwrap_or_else(|| {
                    if mat_type == MaterialType::Aggregate {
                        // Default absorption for aggregates (0.5-2.0% for natural sand)
                        if mat.id.contains("sand") {
                            1.5
                        } else if mat.id.contains("gravel") || mat.id.contains("chips") {
                            0.8
                        } else if mat.id.contains("rac") || mat.id.contains("recycled") {
                            5.0
                        }
                        // Higher for RAC
                        else {
                            1.0
                        }
                    } else {
                        0.0
                    }
                });

                let moisture = props.map(|p| p.moisture).unwrap_or_else(|| {
                    if mat_type == MaterialType::Aggregate {
                        // Typical moisture content for aggregates (0.5-3.0%)
                        if mat.id.contains("rac") || mat.id.contains("recycled") {
                            2.0
                        }
                        // RAC often wetter
                        else {
                            1.0
                        }
                    } else {
                        0.0
                    }
                });

                // Extract Rheology Properties
                let (viscosity, yield_stress, thixotropy) = if let Some(rh) = &mat.rheology {
                    (rh.viscosity, rh.yield_stress, rh.thixotropy)
                } else {
                    (0.0, 0.0, 0.0) // Defaults
                };

                tensor.add_material(
                    comp.mass,
                    sg,
                    type_id,
                    co2,
                    cost,
                    blaine,
                    fm,
                    shape,
                    viscosity,
                    yield_stress,
                    thixotropy,
                    k_factor,
                    reactivity,
                    aspect_ratio,
                    tensile_strength,
                    absorption,
                    moisture,
                );
            } else {
                // [DIAGNOSTIC] Log missing material
                #[cfg(target_arch = "wasm32")]
                console::warn_1(
                    &format!(
                        "[MixTensor] Material '{}' ({}kg) NOT FOUND in registry!",
                        comp.material_id, comp.mass
                    )
                    .into(),
                );
                missing_materials.push(comp.material_id.clone());
            }
        }

        // [DIAGNOSTIC] Summary logging
        if !missing_materials.is_empty() {
            #[cfg(target_arch = "wasm32")]
            console::error_1(
                &format!(
                    "[MixTensor] {} materials missing: {:?}",
                    missing_materials.len(),
                    missing_materials
                )
                .into(),
            );
        }

        // Log tensor summary
        let material_count = tensor.data().len() / MIX_TENSOR_STRIDE;
        #[cfg(target_arch = "wasm32")]
        console::log_1(
            &format!(
                "[MixTensor V8.5] Created tensor with {} materials from {} components (stride={})",
                material_count,
                components.len(),
                MIX_TENSOR_STRIDE
            )
            .into(),
        );

        // [SAFETY] Validate tensor is not completely empty
        if material_count == 0 && !components.is_empty() {
            #[cfg(target_arch = "wasm32")]
            console::error_1(&JsValue::from_str(
                "[MixTensor] CRITICAL: Tensor is empty! All materials failed to resolve. Physics will return zeros."
            ));
            // Return error so callers know physics is invalid
            return Err(JsValue::from_str(&format!(
                "MixTensor empty: {} components provided but no materials resolved. Missing: {:?}",
                components.len(),
                missing_materials
            )));
        }

        Ok(tensor)
    }

    /// Add a material to the tensor with all 17 properties
    ///
    /// [V9.0] Extended with absorption, moisture for RAC characterization
    #[allow(clippy::too_many_arguments)]
    pub fn add_material(
        &mut self,
        mass: f32,
        sg: f32,
        type_id: u8,
        co2: f32,
        cost: f32,
        blaine: f32,
        fm: f32,
        shape: f32,
        viscosity: f32,
        yield_stress: f32,
        thixotropy: f32,
        k_factor: f32,
        reactivity: f32,
        aspect_ratio: f32,
        tensile_strength: f32,
        absorption: f32,
        moisture: f32,
    ) {
        // Original V8.3 properties (indices 0-10)
        self.data.push(mass); // 0
        self.data.push(sg); // 1
        self.data.push(type_id as f32); // 2
        self.data.push(co2); // 3
        self.data.push(cost); // 4
        self.data.push(blaine); // 5
        self.data.push(fm); // 6
        self.data.push(shape); // 7
        self.data.push(viscosity); // 8
        self.data.push(yield_stress); // 9
        self.data.push(thixotropy); // 10

        // [V8.5] New properties (indices 11-14)
        self.data.push(k_factor); // 11 - SCM efficiency
        self.data.push(reactivity); // 12 - Pozzolanic activity
        self.data.push(aspect_ratio); // 13 - Fiber L/d
        self.data.push(tensile_strength); // 14 - Fiber tensile (MPa)

        // [V9.0] RAC properties (indices 15-16)
        self.data.push(absorption); // 15 - Water absorption (%)
        self.data.push(moisture); // 16 - Current moisture (%)
    }

    pub fn total_mass(&self) -> f32 {
        let mut total = 0.0;
        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            total += self.data[i];
        }
        total
    }

    /// [V8.5] Enhanced water-cement ratio calculation using per-material k-factors
    ///
    /// Uses the formula: w/c_eq = water / (cement + Σ(k_i × SCM_i))
    /// Where k_i is the cement efficiency factor for each SCM
    pub fn water_cement_ratio(&self) -> f32 {
        self.water_cement_ratio_with_absorption(false)
    }

    pub fn water_cement_ratio_calibrated(&self, k_scm_aggregate: f32) -> f32 {
        let water_mass = self.mass_of(MaterialType::Water);
        let cement_mass = self.mass_of(MaterialType::Cement);
        // Compute total SCM mass across all SCM types
        let mut scm_mass = 0.0;
        let num_mats = self.data.len() / MIX_TENSOR_STRIDE;
        for i in 0..num_mats {
            let type_id = self.data[i * MIX_TENSOR_STRIDE + 2] as u8;
            if type_id == MaterialType::SCM as u8 {
                scm_mass += self.data[i * MIX_TENSOR_STRIDE];
            }
        }
        let equivalent_binder = cement_mass + (scm_mass * k_scm_aggregate);
        if equivalent_binder > 0.0 {
            water_mass / equivalent_binder
        } else {
            0.0
        }
    }

    /// [V9.0] Enhanced water-cement ratio with aggregate absorption correction
    ///
    /// For RAC materials, aggregates absorb water that reduces the effective
    /// water available for cement hydration. This function corrects for absorption.
    ///
    /// Formula: w/c_effective = (water_added - water_absorbed) / cement
    /// Where: water_absorbed = Σ(aggregate_mass × absorption% / 100)
    ///
    /// # Arguments
    /// * `apply_absorption` - Whether to apply aggregate absorption correction
    pub fn water_cement_ratio_with_absorption(&self, apply_absorption: bool) -> f32 {
        let mut water = 0.0;
        let mut cement = 0.0;
        let mut absorbed_water = 0.0;

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;
            let k_factor = self.data[i + 11]; // [V8.5] Per-material k-factor
            let absorption = self.data[i + 15]; // [V9.0] Absorption (%)
            let moisture = self.data[i + 16]; // [V9.0] Moisture (%)

            if type_id == MaterialType::Water as u8 {
                water += mass;
            } else if type_id == MaterialType::Cement as u8 {
                cement += mass;
            } else if type_id == MaterialType::SCM as u8 {
                // [V8.5] Use the material's k-factor, fallback to 0.5 if not set
                let effective_k = if k_factor > 0.0 { k_factor } else { 0.5 };
                cement += mass * effective_k;
            } else if type_id == MaterialType::Activator as u8 {
                // Geopolymer activators contribute to binder
                cement += mass * 0.3; // Lower efficiency for activators
            } else if apply_absorption && type_id == MaterialType::Aggregate as u8 {
                // [V9.0] Calculate water absorbed by aggregates
                // absorption is in % (e.g., 2.0 = 2%), convert to fraction
                let absorption_fraction = absorption / 100.0;
                absorbed_water += mass * absorption_fraction;

                // Also account for existing moisture (reduces effective absorption)
                let moisture_fraction = moisture / 100.0;
                absorbed_water -= mass * moisture_fraction;
            }
        }

        let effective_water = water - absorbed_water;

        if cement == 0.0 {
            f32::INFINITY
        } else {
            effective_water / cement
        }
    }

    pub fn total_co2(&self) -> f32 {
        let mut total = 0.0;
        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let co2_factor = self.data[i + 3];
            total += mass * co2_factor;
        }
        total
    }

    pub fn scm_ratio(&self) -> f32 {
        let mut total_cement = 0.0;
        let mut total_scm = 0.0;

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;

            if type_id == MaterialType::Cement as u8 {
                total_cement += mass;
            } else if type_id == MaterialType::SCM as u8 {
                total_scm += mass;
            }
        }

        let total_binder = total_cement + total_scm;
        if total_binder > 0.0 {
            total_scm / total_binder
        } else {
            0.0
        }
    }
}

impl MixTensor {
    /// [V8.5] Get total fiber volume fraction
    pub fn fiber_volume_fraction(&self) -> f32 {
        let mut fiber_vol = 0.0;
        let mut total_vol = 0.0;

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let sg = self.data[i + 1];
            let type_id = self.data[i + 2] as u8;

            if sg > 0.0 {
                let vol = mass / (sg * 1000.0);
                total_vol += vol;

                if type_id == MaterialType::Fiber as u8 {
                    fiber_vol += vol;
                }
            }
        }

        if total_vol > 0.0 {
            fiber_vol / total_vol
        } else {
            0.0
        }
    }

    /// [V8.5] Get weighted average fiber properties (aspect_ratio, tensile_strength)
    pub fn average_fiber_properties(&self) -> (f32, f32) {
        let mut weighted_ar = 0.0;
        let mut weighted_ts = 0.0;
        let mut total_fiber_mass = 0.0;

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;

            if type_id == MaterialType::Fiber as u8 && mass > 0.0 {
                let aspect_ratio = self.data[i + 13];
                let tensile_strength = self.data[i + 14];

                weighted_ar += aspect_ratio * mass;
                weighted_ts += tensile_strength * mass;
                total_fiber_mass += mass;
            }
        }

        if total_fiber_mass > 0.0 {
            (
                weighted_ar / total_fiber_mass,
                weighted_ts / total_fiber_mass,
            )
        } else {
            (0.0, 0.0)
        }
    }

    /// [V8.5] Check if mix contains nanomaterials
    pub fn has_nanomaterial(&self) -> bool {
        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            if self.data[i + 2] as u8 == MaterialType::Nanomaterial as u8 {
                return true;
            }
        }
        false
    }

    /// [V8.5] Get nanomaterial dosage as % of binder
    pub fn nanomaterial_dosage(&self) -> f32 {
        let mut nano_mass = 0.0;
        let mut binder_mass = 0.0;

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let mass = self.data[i];
            let type_id = self.data[i + 2] as u8;

            if type_id == MaterialType::Nanomaterial as u8 {
                nano_mass += mass;
            } else if type_id == MaterialType::Cement as u8 || type_id == MaterialType::SCM as u8 {
                binder_mass += mass;
            }
        }

        if binder_mass > 0.0 {
            (nano_mass / binder_mass) * 100.0
        } else {
            0.0
        }
    }

    // Accessor for Rust internal use (not wasm-bindgen)
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    pub fn buffer_mut(&mut self) -> &mut Vec<f32> {
        &mut self.data
    }

    /// Get the stride for this tensor
    pub fn stride(&self) -> usize {
        MIX_TENSOR_STRIDE
    }

    /// Retrieve the total mass of a specific material type
    pub fn mass_of(&self, mat_type: MaterialType) -> f32 {
        let mut total = 0.0;
        let target_id = mat_type as u8;
        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            if self.data[i + 2] as u8 == target_id {
                total += self.data[i];
            }
        }
        total
    }

    /// Check if the mix contains a specific material type
    pub fn has_material_type(&self, mat_type: MaterialType) -> bool {
        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            if self.data[i + 2] as u8 == mat_type as u8 {
                return true;
            }
        }
        false
    }

    /// [GOD-GRADE] Apply RL Action Deltas directly to the mix tensor
    /// This allows the PPO agent to "mutate" reality in the optimization loop.
    pub fn apply_action(&mut self, delta_wc: f32, delta_scms: f32, delta_sp: f32) {
        // 1. Identification Pass
        let mut cement_indices = Vec::new();
        let mut water_indices = Vec::new();
        let mut scm_indices: Vec<usize> = Vec::new();
        let mut sp_indices = Vec::new(); // Superplasticizer

        for i in (0..self.data.len()).step_by(MIX_TENSOR_STRIDE) {
            let type_id = self.data[i + 2] as u8;
            if type_id == MaterialType::Cement as u8 {
                cement_indices.push(i);
            } else if type_id == MaterialType::Water as u8 {
                water_indices.push(i);
            } else if type_id == MaterialType::Admixture as u8 {
                sp_indices.push(i);
            } else if type_id == MaterialType::SCM as u8 {
                scm_indices.push(i);
            }
        }

        // Safety: If no cement or water, we can't adjust w/c effectively
        if cement_indices.is_empty() {
            return;
        }

        // 2. Apply W/C Delta (Adjust Water Mass)
        let total_cement: f32 = cement_indices.iter().map(|&i| self.data[i]).sum();
        let total_scm: f32 = scm_indices.iter().map(|&i| self.data[i]).sum();
        let total_binder = total_cement + total_scm;

        if !water_indices.is_empty() && total_binder > 0.0 {
            // Adjust water based on TOTAL binder (Cement + SCM) for W/B ratio
            let water_change = delta_wc * total_binder;
            let num_water = water_indices.len() as f32;
            for &i in &water_indices {
                let new_mass = (self.data[i] + water_change / num_water).max(0.0);
                self.data[i] = new_mass;
            }
        }

        // 3. Apply SCM Delta (Replace Cement with SCM)
        // delta_scms is percentage shift (e.g. +0.10 means move 10% of total binder from cement to SCM)
        if delta_scms.abs() > 0.001 {
            if !scm_indices.is_empty() {
                // We have SCMs, so we perform actual replacement
                // Target SCM Ratio += delta
                // But simpler: just move mass.
                // Mass to move = delta_scms * total_binder

                // Ensure we don't try to move more mass than available cement
                let mass_to_move = (delta_scms * total_binder).min(total_cement);

                // Reduce Cement
                let num_cement = cement_indices.len() as f32;
                if num_cement > 0.0 {
                    for &i in &cement_indices {
                        self.data[i] = (self.data[i] - (mass_to_move / num_cement)).max(1.0);
                        // Safety floor
                    }
                }

                // Increase SCM
                let num_scm = scm_indices.len() as f32;
                if num_scm > 0.0 {
                    for &i in &scm_indices {
                        self.data[i] = (self.data[i] + (mass_to_move / num_scm)).max(0.0);
                    }
                }
            } else {
                // No SCM existing. Just reduce cement to simulate "leaner" mix?
                // Or ignore? Let's reduce cement to penalize "Low Strength" if goal is checking robustness.
                // If delta_scms is positive, it means we want to increase SCM, so reduce cement.
                // If delta_scms is negative, it means we want to decrease SCM, but there are none, so do nothing.
                if delta_scms > 0.0 {
                    let mass_reduce = delta_scms * total_cement; // Reduce cement by a fraction of its current mass
                    for &i in &cement_indices {
                        self.data[i] = (self.data[i] - mass_reduce).max(10.0); // Safety floor
                    }
                }
            }
        }

        // 4. Apply SP Delta (Adjust Admixture Mass)
        if !sp_indices.is_empty() {
            for &i in &sp_indices {
                let change = delta_sp * total_binder * 0.01; // 1% of binder
                self.data[i] = (self.data[i] + change).max(0.0);
            }
        }
    }
}

// ============================================================================
// [V8.5] Unit Tests for MaterialType parsing - All 16 categories
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Test that all original V8.3 types still parse correctly (backward compat)
    #[test]
    fn test_material_type_backward_compat() {
        // Cement variants
        assert_eq!(MaterialType::from_str("cement"), MaterialType::Cement);
        assert_eq!(MaterialType::from_str("Cement"), MaterialType::Cement);
        assert_eq!(MaterialType::from_str("CEMENT"), MaterialType::Cement);

        // SCM variants
        assert_eq!(MaterialType::from_str("scm"), MaterialType::SCM);
        assert_eq!(MaterialType::from_str("flyash"), MaterialType::SCM);
        assert_eq!(MaterialType::from_str("slag"), MaterialType::SCM);
        assert_eq!(MaterialType::from_str("silica_fume"), MaterialType::SCM);
        assert_eq!(MaterialType::from_str("metakaolin"), MaterialType::SCM);
        assert_eq!(MaterialType::from_str("calcined_clay"), MaterialType::SCM);

        // Aggregate variants
        assert_eq!(MaterialType::from_str("aggregate"), MaterialType::Aggregate);
        assert_eq!(MaterialType::from_str("sand"), MaterialType::Aggregate);
        assert_eq!(MaterialType::from_str("gravel"), MaterialType::Aggregate);
        assert_eq!(MaterialType::from_str("recycled"), MaterialType::Aggregate);

        // Water
        assert_eq!(MaterialType::from_str("water"), MaterialType::Water);

        // Admixture
        assert_eq!(MaterialType::from_str("admixture"), MaterialType::Admixture);
        assert_eq!(
            MaterialType::from_str("superplasticizer"),
            MaterialType::Admixture
        );

        // Unknown falls through to Air
        assert_eq!(MaterialType::from_str("unknown"), MaterialType::Air);
        assert_eq!(MaterialType::from_str(""), MaterialType::Air);

        println!("✅ Backward compatibility tests passed");
    }

    /// Test all new V8.5 material types
    #[test]
    fn test_material_type_v85_extended() {
        // ═══════════════════════════════════════════════════════════════
        // FIBERS (Type 6)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("steel_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("pp_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("pva_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("basalt_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("carbon_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("macro_fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("micro_fiber"), MaterialType::Fiber);

        // ═══════════════════════════════════════════════════════════════
        // NANOMATERIALS (Type 7)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(
            MaterialType::from_str("nanomaterial"),
            MaterialType::Nanomaterial
        );
        assert_eq!(
            MaterialType::from_str("nano_silica"),
            MaterialType::Nanomaterial
        );
        assert_eq!(
            MaterialType::from_str("nano_sio2"),
            MaterialType::Nanomaterial
        );
        assert_eq!(MaterialType::from_str("cnt"), MaterialType::Nanomaterial);
        assert_eq!(
            MaterialType::from_str("graphene_oxide"),
            MaterialType::Nanomaterial
        );
        assert_eq!(
            MaterialType::from_str("nano_tio2"),
            MaterialType::Nanomaterial
        );

        // ═══════════════════════════════════════════════════════════════
        // ACTIVATORS (Type 8) - Geopolymer
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("activator"), MaterialType::Activator);
        assert_eq!(
            MaterialType::from_str("sodium_hydroxide"),
            MaterialType::Activator
        );
        assert_eq!(MaterialType::from_str("naoh"), MaterialType::Activator);
        assert_eq!(
            MaterialType::from_str("sodium_silicate"),
            MaterialType::Activator
        );

        // ═══════════════════════════════════════════════════════════════
        // LIGHTWEIGHT AGGREGATES (Type 9)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(
            MaterialType::from_str("lightweight"),
            MaterialType::Lightweight
        );
        assert_eq!(MaterialType::from_str("leca"), MaterialType::Lightweight);
        assert_eq!(
            MaterialType::from_str("expanded_clay"),
            MaterialType::Lightweight
        );
        assert_eq!(MaterialType::from_str("pumice"), MaterialType::Lightweight);
        assert_eq!(MaterialType::from_str("perlite"), MaterialType::Lightweight);

        // ═══════════════════════════════════════════════════════════════
        // HEAVYWEIGHT AGGREGATES (Type 10)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(
            MaterialType::from_str("heavyweight"),
            MaterialType::Heavyweight
        );
        assert_eq!(
            MaterialType::from_str("magnetite"),
            MaterialType::Heavyweight
        );
        assert_eq!(
            MaterialType::from_str("hematite"),
            MaterialType::Heavyweight
        );
        assert_eq!(MaterialType::from_str("barite"), MaterialType::Heavyweight);

        // ═══════════════════════════════════════════════════════════════
        // ACCELERATORS (Type 11)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(
            MaterialType::from_str("accelerator"),
            MaterialType::Accelerator
        );
        assert_eq!(
            MaterialType::from_str("calcium_chloride"),
            MaterialType::Accelerator
        );
        assert_eq!(
            MaterialType::from_str("calcium_nitrate"),
            MaterialType::Accelerator
        );

        // ═══════════════════════════════════════════════════════════════
        // RETARDERS (Type 12)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("retarder"), MaterialType::Retarder);
        assert_eq!(
            MaterialType::from_str("citric_acid"),
            MaterialType::Retarder
        );
        assert_eq!(MaterialType::from_str("gluconate"), MaterialType::Retarder);

        // ═══════════════════════════════════════════════════════════════
        // AIR ENTRAINERS (Type 13)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(
            MaterialType::from_str("air_entrainer"),
            MaterialType::AirEntrainer
        );
        assert_eq!(MaterialType::from_str("vinsol"), MaterialType::AirEntrainer);

        // ═══════════════════════════════════════════════════════════════
        // POLYMER MODIFIERS (Type 14)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("polymer"), MaterialType::Polymer);
        assert_eq!(MaterialType::from_str("latex"), MaterialType::Polymer);
        assert_eq!(MaterialType::from_str("sbr"), MaterialType::Polymer);
        assert_eq!(MaterialType::from_str("epoxy"), MaterialType::Polymer);

        // ═══════════════════════════════════════════════════════════════
        // PIGMENTS (Type 15)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("pigment"), MaterialType::Pigment);
        assert_eq!(MaterialType::from_str("iron_oxide"), MaterialType::Pigment);
        assert_eq!(MaterialType::from_str("bayferrox"), MaterialType::Pigment);
        assert_eq!(
            MaterialType::from_str("titanium_dioxide"),
            MaterialType::Pigment
        );

        // ═══════════════════════════════════════════════════════════════
        // FILLERS (Type 16)
        // ═══════════════════════════════════════════════════════════════
        assert_eq!(MaterialType::from_str("filler"), MaterialType::Filler);
        assert_eq!(MaterialType::from_str("quartz_flour"), MaterialType::Filler);
        assert_eq!(
            MaterialType::from_str("limestone_filler"),
            MaterialType::Filler
        );
        assert_eq!(
            MaterialType::from_str("calcium_carbonate"),
            MaterialType::Filler
        );
        assert_eq!(MaterialType::from_str("wollastonite"), MaterialType::Filler);

        println!("✅ V8.5 extended material type tests passed (17 types)");
    }

    /// Test case insensitivity for new types
    #[test]
    fn test_material_type_case_insensitive() {
        // Original types
        assert_eq!(MaterialType::from_str("RECYCLED"), MaterialType::Aggregate);
        assert_eq!(MaterialType::from_str("Recycled"), MaterialType::Aggregate);
        assert_eq!(MaterialType::from_str("recycled"), MaterialType::Aggregate);

        // New types must also be case insensitive
        assert_eq!(MaterialType::from_str("FIBER"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("Fiber"), MaterialType::Fiber);
        assert_eq!(MaterialType::from_str("fiber"), MaterialType::Fiber);

        assert_eq!(
            MaterialType::from_str("NANOMATERIAL"),
            MaterialType::Nanomaterial
        );
        assert_eq!(
            MaterialType::from_str("Nanomaterial"),
            MaterialType::Nanomaterial
        );

        assert_eq!(
            MaterialType::from_str("LIGHTWEIGHT"),
            MaterialType::Lightweight
        );
        assert_eq!(
            MaterialType::from_str("Lightweight"),
            MaterialType::Lightweight
        );

        println!("✅ Case insensitivity tests passed");
    }

    /// Test helper methods on MaterialType
    #[test]
    fn test_material_type_helpers() {
        // Binder check
        assert!(MaterialType::Cement.is_binder());
        assert!(MaterialType::SCM.is_binder());
        assert!(MaterialType::Activator.is_binder());
        assert!(!MaterialType::Aggregate.is_binder());
        assert!(!MaterialType::Fiber.is_binder());

        // Solid check
        assert!(MaterialType::Cement.is_solid());
        assert!(MaterialType::Aggregate.is_solid());
        assert!(MaterialType::Fiber.is_solid());
        assert!(!MaterialType::Water.is_solid());
        assert!(!MaterialType::Air.is_solid());

        // Rheology affector check
        assert!(MaterialType::Cement.affects_rheology());
        assert!(MaterialType::SCM.affects_rheology());
        assert!(MaterialType::Fiber.affects_rheology());
        assert!(MaterialType::Nanomaterial.affects_rheology());
        assert!(!MaterialType::Water.affects_rheology());
        assert!(!MaterialType::Heavyweight.affects_rheology());

        // Canonical name check
        assert_eq!(MaterialType::Fiber.canonical_name(), "fiber");
        assert_eq!(MaterialType::Nanomaterial.canonical_name(), "nanomaterial");
        assert_eq!(MaterialType::Lightweight.canonical_name(), "lightweight");

        println!("✅ MaterialType helper method tests passed");
    }

    /// Test discriminant values are stable for serialization
    #[test]
    fn test_material_type_discriminants() {
        // Original types (V8.3) - MUST NOT CHANGE
        assert_eq!(MaterialType::Cement as u8, 0);
        assert_eq!(MaterialType::Aggregate as u8, 1);
        assert_eq!(MaterialType::Water as u8, 2);
        assert_eq!(MaterialType::Admixture as u8, 3);
        assert_eq!(MaterialType::Air as u8, 4);
        assert_eq!(MaterialType::SCM as u8, 5);

        // New types (V8.5) - Sequential from 6
        assert_eq!(MaterialType::Fiber as u8, 6);
        assert_eq!(MaterialType::Nanomaterial as u8, 7);
        assert_eq!(MaterialType::Activator as u8, 8);
        assert_eq!(MaterialType::Lightweight as u8, 9);
        assert_eq!(MaterialType::Heavyweight as u8, 10);
        assert_eq!(MaterialType::Accelerator as u8, 11);
        assert_eq!(MaterialType::Retarder as u8, 12);
        assert_eq!(MaterialType::AirEntrainer as u8, 13);
        assert_eq!(MaterialType::Polymer as u8, 14);
        assert_eq!(MaterialType::Pigment as u8, 15);

        println!("✅ Discriminant stability tests passed");
    }
}
