// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! Entity-Component-System (ECS) Architecture
//!
//! This module decouples the physical calculation logic from flat array representations
//! into a modular ECS paradigm. Models are evaluated by Systems running Queries
//! against strictly typed Components.

pub mod components;
pub mod systems;
pub mod world;
