// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

// GPU computation module — backend-agnostic graph neural network layers.
// CPU (burn-ndarray) is always available.
// GPU (burn-wgpu / Metal) is activated with `--features gpu`.

pub mod gnn_layer;
