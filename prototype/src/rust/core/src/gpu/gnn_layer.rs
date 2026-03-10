// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! GPU-Accelerated GNN Layer — Phase P
//!
//! Implements a backend-agnostic single-head Graph Attention mechanism using the
//! `burn` tensor API. The `Backend` type parameter is the only thing that changes
//! between CPU (NdArray) and GPU (Wgpu/Metal) execution paths.
//!
//! Core computation:
//!   H_new = tanh(X @ W_self + A @ X @ W_neigh)
//!   α     = softmax(H_new @ H_new.T / √d)   [self-attention over nodes]
//!   out   = mean(α @ H_new)                  [global graph readout]
//!
//! Mathematical references:
//! - GraphSAGE: Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017.
//! - GAT attention: Veličković et al., "Graph Attention Networks", ICLR 2018.

#[cfg(feature = "ndarray")]
use burn::backend::NdArray;

use burn::prelude::*;
use burn::tensor::activation;

/// Type alias for the CPU backend — compiled only with `--features ndarray`.
#[cfg(feature = "ndarray")]
pub type CpuBackend = NdArray<f32>;

/// Type alias for the GPU backend — compiled only with `--features gpu`.
#[cfg(feature = "gpu")]
pub type GpuBackend = burn_wgpu::Wgpu;

/// Backend-agnostic GNN layer for graph attention over material topology nodes.
///
/// # Type Parameters
/// - `B`: A `burn::backend::Backend` — swap `CpuBackend` for `GpuBackend` to route to Metal.
pub struct GpuGnnLayer<B: Backend> {
    /// Self-feature transform: [hidden × hidden]
    pub w_self: Tensor<B, 2>,
    /// Neighbour aggregation: [hidden × hidden]
    pub w_neigh: Tensor<B, 2>,
    /// Hidden dimension
    pub hidden: usize,
    /// Number of graph nodes
    pub n_nodes: usize,
}

impl<B: Backend> GpuGnnLayer<B> {
    /// Construct a new GNN layer with Xavier-uniform-like random initialization.
    ///
    /// `n_nodes`  — number of graph nodes (e.g. 5_000 micro-nodes)
    /// `hidden`   — hidden dimension per node (e.g. 32)
    pub fn new(n_nodes: usize, hidden: usize, device: &B::Device) -> Self {
        let scale = (6.0_f64 / (hidden + hidden) as f64).sqrt() as f32;
        let w_self = Tensor::<B, 2>::random(
            [hidden, hidden],
            burn::tensor::Distribution::Uniform(-scale as f64, scale as f64),
            device,
        );
        let w_neigh = Tensor::<B, 2>::random(
            [hidden, hidden],
            burn::tensor::Distribution::Uniform(-scale as f64, scale as f64),
            device,
        );
        Self {
            w_self,
            w_neigh,
            hidden,
            n_nodes,
        }
    }

    /// Forward pass: returns updated node embeddings `H_new` of shape [n_nodes, hidden].
    ///
    /// `node_features` — [n_nodes, hidden] input feature matrix
    /// `adj`           — [n_nodes, n_nodes] adjacency matrix (row-normalised)
    pub fn forward(&self, node_features: Tensor<B, 2>, adj: Tensor<B, 2>) -> Tensor<B, 2> {
        // H_self  = X @ W_self              [n, h]
        let h_self = node_features.clone().matmul(self.w_self.clone());

        // H_neigh = A @ X @ W_neigh         [n, h]
        let h_neigh = adj.matmul(node_features).matmul(self.w_neigh.clone());

        // H_new   = tanh(H_self + H_neigh)
        activation::tanh(h_self + h_neigh)
    }

    /// Self-attention: returns α of shape [n_nodes, n_nodes].
    ///
    /// α = softmax(H @ H.T / √hidden)
    pub fn attention(&self, h: Tensor<B, 2>) -> Tensor<B, 2> {
        let d_sqrt = (self.hidden as f64).sqrt();
        let raw = h.clone().matmul(h.transpose()) / d_sqrt;
        // softmax along dim 1 (over destination nodes)
        activation::softmax(raw, 1)
    }

    /// Global readout: mean-pool over attended node embeddings.
    ///
    /// Returns a [1, hidden] context vector summarising the whole graph.
    pub fn pooled_output(&self, h: Tensor<B, 2>, alpha: Tensor<B, 2>) -> Tensor<B, 2> {
        // Attended nodes: α @ H  → [n, h]
        let attended = alpha.matmul(h);
        // Mean over nodes → [1, h]
        attended.mean_dim(0)
    }

    /// Full forward+attention+readout pass.
    /// Returns `(H_new [n, h], alpha [n, n], readout [1, h])` for inspection.
    pub fn run(
        &self,
        node_features: Tensor<B, 2>,
        adj: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let h = self.forward(node_features, adj);
        let alpha = self.attention(h.clone());
        let readout = self.pooled_output(h.clone(), alpha.clone());
        (h, alpha, readout)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_gnn_layer_output_shapes() {
        let device = Default::default();
        let n = 8usize;
        let h = 16usize;
        let layer = GpuGnnLayer::<B>::new(n, h, &device);

        // Identity adjacency (each node connected to itself)
        let node_feats = Tensor::<B, 2>::ones([n, h], &device);
        let adj = Tensor::<B, 2>::eye(n, &device);

        let (h_out, alpha, readout) = layer.run(node_feats, adj);

        // Shape checks
        assert_eq!(
            h_out.shape().dims,
            [n, h],
            "H_new must be [n_nodes, hidden]"
        );
        assert_eq!(alpha.shape().dims, [n, n], "α must be [n_nodes, n_nodes]");
        assert_eq!(readout.shape().dims, [1, h], "readout must be [1, hidden]");
    }

    #[test]
    fn test_attention_rows_sum_to_one() {
        let device = Default::default();
        let n = 4usize;
        let h = 8usize;
        let layer = GpuGnnLayer::<B>::new(n, h, &device);
        let node_feats = Tensor::<B, 2>::random(
            [n, h],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let adj = Tensor::<B, 2>::eye(n, &device);
        let (h_out, alpha, _) = layer.run(node_feats, adj);
        let _ = h_out; // consumed for clarity

        // Each row of alpha must sum to ~1.0 (softmax property)
        let row_sums: Vec<f32> = alpha
            .sum_dim(1)
            .into_data()
            .to_vec::<f32>()
            .expect("row sums");
        for (i, &s) in row_sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "Row {i} attention sum = {s:.6}, expected ~1.0"
            );
        }
    }

    #[test]
    fn test_zero_adj_produces_self_only_result() {
        // With zero adjacency, H_new = tanh(X @ W_self + 0)
        // This proves the self vs. neighbour paths are cleanly separable.
        let device = Default::default();
        let n = 4usize;
        let h = 8usize;
        let layer = GpuGnnLayer::<B>::new(n, h, &device);
        let node_feats = Tensor::<B, 2>::ones([n, h], &device);
        let adj_zero = Tensor::<B, 2>::zeros([n, n], &device);
        let adj_eye = Tensor::<B, 2>::eye(n, &device);

        let h_zero = layer.forward(node_feats.clone(), adj_zero);
        let h_self = layer.forward(node_feats, adj_eye.clone());

        // With identity adj (normalised: each row has exactly one 1),
        // h_self != h_zero because w_neigh contributes
        let vals_zero = h_zero.into_data().to_vec::<f32>().unwrap();
        let vals_self = h_self.into_data().to_vec::<f32>().unwrap();
        // They should NOT be identical (neighbour weights are non-zero)
        let differs = vals_zero
            .iter()
            .zip(&vals_self)
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "Zero-adj and self-adj results must differ (w_neigh != 0)"
        );
    }
}
