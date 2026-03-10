// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! Explainability (XAI) Module — Attention-to-English Translator
//!
//! Converts internal GNN Graph Attention weights (a_ij) into human-readable
//! causal statements. The GAT attention matrix captures which chemical interactions
//! the network considered most critical when proposing a formulation decision.
//!
//! This bridges the gap between "black-box" neural optimisation and the transparent,
//! engineer-reviewable reasoning required for safety-critical physical AI.
//!
//! Architecture:
//!   1. Intercept `get_attention_weights()` output from PPOAgent
//!   2. Find the (i,j) pair with maximum attention weight
//!   3. Map (i, j) node indices to chemical entities via a semantic registry
//!   4. Return a natural-language causal statement

use std::collections::HashMap;

/// Semantic registry mapping node index to chemical/physical identity.
pub struct SemanticRegistry {
    node_names: HashMap<usize, &'static str>,
    interaction_verbs: HashMap<(&'static str, &'static str), &'static str>,
}

impl SemanticRegistry {
    /// Build the default cement chemistry semantic registry.
    pub fn cement_chemistry() -> Self {
        let mut node_names = HashMap::new();
        node_names.insert(0, "Cement (C3S/C2S)");
        node_names.insert(1, "Water");
        node_names.insert(2, "Slag (SCM)");
        node_names.insert(3, "Fly Ash (SCM)");
        node_names.insert(4, "Superplasticiser");
        node_names.insert(5, "Coarse Aggregate");
        node_names.insert(6, "Fine Aggregate");
        node_names.insert(7, "C-S-H Gel Network");
        node_names.insert(8, "ITZ (Interfacial Transition Zone)");
        node_names.insert(9, "Pore Solution");

        let mut interaction_verbs: HashMap<(&str, &str), &str> = HashMap::new();
        interaction_verbs.insert(
            ("Cement (C3S/C2S)", "Water"),
            "Hydration reaction (C3S + H₂O → C-S-H + portlandite)",
        );
        interaction_verbs.insert(
            ("Water", "Cement (C3S/C2S)"),
            "Water availability constrains C3S hydration rate",
        );
        interaction_verbs.insert(
            ("Slag (SCM)", "Water"),
            "Secondary pozzolanic reaction activated by alkali",
        );
        interaction_verbs.insert(
            ("Superplasticiser", "Cement (C3S/C2S)"),
            "Steric repulsion dispersing cement particles",
        );
        interaction_verbs.insert(
            ("Water", "Pore Solution"),
            "Free water builds capillary pore pressure gradient",
        );
        interaction_verbs.insert(
            ("C-S-H Gel Network", "ITZ (Interfacial Transition Zone)"),
            "Gel densification reducing ITZ porosity",
        );

        Self {
            node_names,
            interaction_verbs,
        }
    }

    pub fn node_name(&self, idx: usize) -> &str {
        self.node_names.get(&idx).copied().unwrap_or("Unknown Node")
    }

    pub fn interaction_description(&self, src: &str, dst: &str) -> String {
        self.interaction_verbs
            .get(&(src, dst))
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{src} influences {dst}"))
    }
}

/// Translates a raw attention weight matrix into a human-readable causal statement.
pub struct XaiTranslator {
    registry: SemanticRegistry,
    /// Threshold above which we flag an attention edge as "critical"
    pub critical_threshold: f64,
}

impl XaiTranslator {
    pub fn new(registry: SemanticRegistry) -> Self {
        Self {
            registry,
            critical_threshold: 0.7,
        }
    }

    /// Generate a causal explanation from a flat attention matrix (row-major).
    /// `n_nodes` is the side length (matrix is n_nodes × n_nodes).
    pub fn explain(&self, attention_matrix: &[f64], n_nodes: usize) -> XaiExplanation {
        assert_eq!(
            attention_matrix.len(),
            n_nodes * n_nodes,
            "Attention matrix must be n_nodes² in length"
        );

        // Find the (i, j) with highest attention weight
        let (max_idx, max_weight) = attention_matrix
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &w)| (i, w))
            .unwrap_or((0, 0.0));

        let src_node = max_idx / n_nodes;
        let dst_node = max_idx % n_nodes;

        let src_name = self.registry.node_name(src_node);
        let dst_name = self.registry.node_name(dst_node);
        let mechanism = self.registry.interaction_description(src_name, dst_name);

        let is_critical = max_weight >= self.critical_threshold;

        let statement = format!(
            "Agent action driven by {} attention ({:.2}) on {} → {}. Mechanism: {}.",
            if is_critical { "CRITICAL" } else { "notable" },
            max_weight,
            src_name,
            dst_name,
            mechanism
        );

        XaiExplanation {
            statement,
            src_node,
            dst_node,
            attention_weight: max_weight,
            is_critical,
            src_name: src_name.to_string(),
            dst_name: dst_name.to_string(),
            rank: 0,
        }
    }

    /// Return the Top-K most attended edges as ranked explanations.
    /// Enables contrastive reasoning: "Agent considered A→B (0.98) over C→D (0.34)."
    pub fn explain_topk(
        &self,
        attention_matrix: &[f64],
        n_nodes: usize,
        k: usize,
    ) -> Vec<XaiExplanation> {
        assert_eq!(
            attention_matrix.len(),
            n_nodes * n_nodes,
            "Attention matrix must be n_nodes² in length"
        );
        assert!(k > 0, "k must be > 0");

        // Collect all non-zero edges with their flat indices
        let mut indexed: Vec<(usize, f64)> = attention_matrix
            .iter()
            .enumerate()
            .filter(|(_, &w)| w > 0.0)
            .map(|(i, &w)| (i, w))
            .collect();

        // Sort descending by weight
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        indexed
            .into_iter()
            .enumerate()
            .map(|(rank, (idx, weight))| {
                let src_node = idx / n_nodes;
                let dst_node = idx % n_nodes;
                let src_name = self.registry.node_name(src_node);
                let dst_name = self.registry.node_name(dst_node);
                let mechanism = self.registry.interaction_description(src_name, dst_name);
                let is_critical = weight >= self.critical_threshold;

                let statement = format!(
                    "[#{rank}] {} attention ({:.2}) on {} → {}. Mechanism: {}.",
                    if is_critical { "CRITICAL" } else { "notable" },
                    weight,
                    src_name,
                    dst_name,
                    mechanism
                );

                XaiExplanation {
                    statement,
                    src_node,
                    dst_node,
                    attention_weight: weight,
                    is_critical,
                    src_name: src_name.to_string(),
                    dst_name: dst_name.to_string(),
                    rank,
                }
            })
            .collect()
    }
}

/// Human-readable explanation from attention weight analysis.
#[derive(Debug, Clone)]
pub struct XaiExplanation {
    pub statement: String,
    pub src_node: usize,
    pub dst_node: usize,
    pub attention_weight: f64,
    pub is_critical: bool,
    pub src_name: String,
    pub dst_name: String,
    /// Rank of this explanation (0 = highest attention)
    pub rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xai_translation() {
        // Simulate: agent sees zero water (node 0 = Cement, node 1 = Water)
        // Highest attention should be on (Cement → Water) edge with weight 0.98
        let mut attention = vec![0.0_f64; 10 * 10];
        // Cement (node 0) → Water (node 1): critical deficit attention
        // Row-major index: src=0, dst=1 → index = 0 * n_nodes + 1 = 1
        attention[1] = 0.98;
        // Background: Superplasticiser (node 4) → Cement (node 0) = 4*10 = 40
        attention[40] = 0.3;

        let registry = SemanticRegistry::cement_chemistry();
        let translator = XaiTranslator::new(registry);
        let explanation = translator.explain(&attention, 10);

        println!("\nXAI Explanation:\n  {}", explanation.statement);

        // Must flag as critical
        assert!(
            explanation.is_critical,
            "Should flag as critical at weight 0.98"
        );

        // Must identify Cement and Water
        assert_eq!(explanation.src_node, 0, "Source must be Cement node (0)");
        assert_eq!(explanation.dst_node, 1, "Dest must be Water node (1)");

        // Statement must mention the hydration mechanism
        assert!(
            explanation.statement.contains("Hydration"),
            "Statement must describe hydration mechanism, got: {}",
            explanation.statement
        );
    }

    #[test]
    fn test_xai_topk_ordering() {
        // Set 3 distinct attention weights and verify topk returns them in descending order
        let mut attention = vec![0.0_f64; 10 * 10];
        attention[1] = 0.98; // Cement → Water (highest)
        attention[40] = 0.30; // Superplasticiser → Cement (medium)
        attention[72] = 0.55; // C-S-H Gel → Pore Solution (mid-high)

        let registry = SemanticRegistry::cement_chemistry();
        let translator = XaiTranslator::new(registry);
        let explanations = translator.explain_topk(&attention, 10, 3);

        assert_eq!(
            explanations.len(),
            3,
            "Should return exactly 3 explanations"
        );
        assert_eq!(explanations[0].rank, 0, "First explanation must be rank 0");
        assert_eq!(explanations[1].rank, 1, "Second explanation must be rank 1");
        assert_eq!(explanations[2].rank, 2, "Third explanation must be rank 2");

        // Must be strictly descending by weight
        assert!(
            explanations[0].attention_weight > explanations[1].attention_weight,
            "Rank 0 must have highest weight"
        );
        assert!(
            explanations[1].attention_weight > explanations[2].attention_weight,
            "Rank 1 weight must exceed rank 2"
        );

        // Top explanation must match highest weight edge (Cement→Water, index=1)
        assert_eq!(explanations[0].src_node, 0, "Top edge source: Cement");
        assert_eq!(explanations[0].dst_node, 1, "Top edge dest: Water");
        assert!(
            explanations[0].statement.starts_with("[#0]"),
            "Statement must start with rank prefix"
        );

        println!("\nTop-K XAI Explanations:");
        for expl in &explanations {
            println!("  {}", expl.statement);
        }
    }
}
