// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

use serde::{Deserialize, Serialize};

/// 40-dimensional God-Grade Proxy Manifold mapped to UMST SensingProxies
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyId {
    // 1. Fresh Properties (Rheology & Workability)
    SlumpFlow = 0,
    T50Time = 1,
    AirContent = 2,
    DensityFresh = 3,
    TempFresh = 4,
    SettingInitial = 5,
    SettingFinal = 6,

    // 2. Hardened Properties (Mechanics)
    F1Compressive = 7,
    F7Compressive = 8,
    F28Compressive = 9,
    FlexuralStrength = 10,
    ElasticModulus = 11,
    DensityHardened = 12,

    // 3. Durability (Service Life)
    RCPT = 13,
    WaterAbsorption = 14,
    SurfaceResistivity = 15,
    ShrinkageDrying = 16,

    // 4. Economy
    CostActual = 17,
    YieldActual = 18,
    KWhPerM3 = 19,

    // 5. Process & Visual
    VideoFlowVelocity = 20,
    VideoSettlingRate = 21,
    AcousticResonance = 22,
    VisualSegregationIndex = 23,
    EnvironmentalTemp = 24,
    EnvironmentalHumidity = 25,
    BleedingRate = 26,
    Finishability = 27,
    CrackingPlastic = 28,
    Pumpability = 29,

    // 6. Hardened Visual & Advanced (V9.5 mapping)
    VoidFractionVisual = 30,
    SurfaceTextureEntropy = 31,
    ColorWcProxy = 32,
    FaceBrightnessTop = 33,
    FaceBrightnessNorth = 34,
    FaceBrightnessSouth = 35,
    FaceBrightnessEast = 36,
    FaceBrightnessWest = 37,
    TapResonanceHz = 38,
    CrackingVisual = 39,
}

impl ProxyId {
    pub const MAX_INDEX: usize = 40;

    pub fn to_usize(self) -> usize {
        self as usize
    }
}
