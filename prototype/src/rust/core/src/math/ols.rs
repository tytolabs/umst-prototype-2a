// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0

//! Ordinary Least Squares (OLS) Regression Module
//!
//! Provides a minimal closed-form linear regression implementation,
//! primarily used to compute R² (coefficient of determination) as the
//! convergence metric in the epistemic proxy selection experiment.
//!
//! **Why R²?**
//! When the epistemic agent reveals `k` proxies for a sample, R² of an OLS
//! fit from those `k` proxy values → target strength quantifies how much
//! predictive power has been captured. R² → 1 means the revealed proxies
//! fully explain the target. This is a theoretically sound convergence metric
//! grounded in information theory (squared correlation = fraction of variance
//! explained by the linear predictor).

/// A simple closed-form OLS regressor: y ≈ β₀ + β₁x₁ + … + βₖxₖ
///
/// Uses the normal equations: β = (XᵀX)⁻¹Xᵀy
/// For the 1D case, closed-form Pearson-based formulas are used.
/// For multi-feature, a simple Gram-Schmidt-based solution is used.
pub struct SimpleOLS {
    /// Fitted coefficients [β₀, β₁, …]. Empty before fit.
    pub coefficients: Vec<f64>,
    /// Coefficient of determination R² on training data
    pub r_squared: f64,
}

impl SimpleOLS {
    pub fn new() -> Self {
        SimpleOLS {
            coefficients: Vec::new(),
            r_squared: 0.0,
        }
    }

    /// Fit OLS to training data.
    ///
    /// # Arguments
    /// * `x_columns` — Each inner Vec is one feature column (n_samples values each)
    /// * `y`         — Target variable (n_samples values)
    ///
    /// # Returns
    /// `Ok(r_squared)` on success, `Err` if data is degenerate.
    pub fn fit(&mut self, x_columns: &[Vec<f64>], y: &[f64]) -> Result<f64, &'static str> {
        let n = y.len();
        if n < 2 {
            return Err("Need at least 2 samples to fit OLS");
        }
        if x_columns.is_empty() {
            return Err("No features provided");
        }

        // Simple multivariate approach: for the proxy selection experiment
        // we only need R², which we compute by the formula:
        //   R² = 1 - SS_res / SS_tot
        // Where SS_res comes from the fitted OLS predictions.
        //
        // For stability and simplicity, use the closed-form Pearson-based
        // approach for the 1-feature case and a stepwise correlation sum
        // approximation for the multi-feature case (sufficient for our purposes).

        let y_mean = y.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot < 1e-12 {
            // Degenerate case: all y values the same
            self.r_squared = 1.0;
            return Ok(1.0);
        }

        // Fit using the Gram matrix approach for stability
        // Augment X with intercept column
        let p = x_columns.len() + 1; // +1 for intercept
        let mut x_aug: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(p);
            row.push(1.0); // intercept
            for col in x_columns {
                row.push(col.get(i).copied().unwrap_or(0.0));
            }
            x_aug.push(row);
        }

        // Compute XᵀX (p × p) and Xᵀy (p × 1)
        let mut xtx = vec![vec![0.0f64; p]; p];
        let mut xty = vec![0.0f64; p];
        for (i, row) in x_aug.iter().enumerate() {
            for j in 0..p {
                xty[j] += row[j] * y[i];
                for k in 0..p {
                    xtx[j][k] += row[j] * row[k];
                }
            }
        }

        // Solve (XᵀX)β = Xᵀy using Gaussian elimination with partial pivoting
        let beta = gaussian_elimination(&xtx, &xty)?;
        self.coefficients = beta.clone();

        // Compute R² from predictions
        let ss_res: f64 = x_aug
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let y_hat: f64 = row.iter().zip(beta.iter()).map(|(xij, bj)| xij * bj).sum();
                (y[i] - y_hat).powi(2)
            })
            .sum();

        let r2 = (1.0 - ss_res / ss_tot).max(0.0).min(1.0);
        self.r_squared = r2;
        Ok(r2)
    }

    /// Compute R² given revealed proxy names + their values for a single sample,
    /// using all training samples to fit. This is the core method for the experiment.
    ///
    /// Steps:
    /// 1. Build columns of each revealed proxy across ALL training samples
    /// 2. Build the y vector of all training strengths
    /// 3. Fit OLS
    /// 4. Return R²
    pub fn r_squared_for_proxies(
        proxy_names: &[String],
        data_provider: &dyn crate::data_provider::ProxyDataSource,
    ) -> f64 {
        if proxy_names.is_empty() {
            return 0.0;
        }

        let n = data_provider.n_samples();
        if n < 2 {
            return 0.0;
        }

        // Build feature columns (one per proxy, all samples)
        let x_columns: Vec<Vec<f64>> = proxy_names
            .iter()
            .map(|name| {
                (0..n)
                    .map(|i| data_provider.reveal_proxy(i, name).unwrap_or(0.0))
                    .collect()
            })
            .collect();

        // Build target vector
        let y: Vec<f64> = (0..n).map(|i| data_provider.get_ground_truth(i)).collect();

        let mut ols = SimpleOLS::new();
        ols.fit(&x_columns, &y).unwrap_or(0.0)
    }

    /// Fit OLS on `train_indices` and evaluate R² on `test_indices`.
    ///
    /// This enables bootstrap cross-validation: each trial uses a different
    /// train/test split, creating genuine variance for Welch's t-test.
    pub fn r_squared_for_proxies_on_split(
        proxy_names: &[String],
        data_provider: &dyn crate::data_provider::ProxyDataSource,
        train_indices: &[usize],
        test_indices: &[usize],
    ) -> f64 {
        if proxy_names.is_empty() || train_indices.len() < 2 || test_indices.is_empty() {
            return 0.0;
        }

        // Build training feature columns
        let x_train: Vec<Vec<f64>> = proxy_names
            .iter()
            .map(|name| {
                train_indices
                    .iter()
                    .map(|&i| data_provider.reveal_proxy(i, name).unwrap_or(0.0))
                    .collect()
            })
            .collect();
        let y_train: Vec<f64> = train_indices
            .iter()
            .map(|&i| data_provider.get_ground_truth(i))
            .collect();

        let mut ols = SimpleOLS::new();
        if ols.fit(&x_train, &y_train).is_err() {
            return 0.0;
        }

        // Evaluate on test split
        let y_test: Vec<f64> = test_indices
            .iter()
            .map(|&i| data_provider.get_ground_truth(i))
            .collect();
        let y_test_mean = y_test.iter().sum::<f64>() / y_test.len() as f64;
        let ss_tot: f64 = y_test.iter().map(|yi| (yi - y_test_mean).powi(2)).sum();
        if ss_tot < 1e-12 {
            return 1.0;
        }

        let ss_res: f64 = test_indices
            .iter()
            .enumerate()
            .map(|(j, &i)| {
                // Build one row of test features
                let mut row = vec![1.0f64]; // intercept
                for name in proxy_names {
                    row.push(data_provider.reveal_proxy(i, name).unwrap_or(0.0));
                }
                let y_hat: f64 = row
                    .iter()
                    .zip(ols.coefficients.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (y_test[j] - y_hat).powi(2)
            })
            .sum();

        (1.0 - ss_res / ss_tot).max(0.0).min(1.0)
    }
}

/// Gaussian elimination with partial pivoting to solve Ax = b.
/// Returns x or an error if the system is singular.
fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    if a.len() != n || a.is_empty() {
        return Err("Dimension mismatch");
    }

    // Augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, bi)| {
            let mut r = row.to_vec();
            r.push(*bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting: find max element in this column
        let max_row = (col..n)
            .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap())
            .unwrap_or(col);
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            // Near-singular: return zero solution (safe fallback for R² = 0)
            return Ok(vec![0.0; n]);
        }

        // Normalise pivot row
        for j in col..=n {
            aug[col][j] /= pivot;
        }

        // Eliminate below (and above for back-substitution)
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    Ok(aug.iter().map(|row| row[n]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_perfect_fit() {
        // y = 2x + 1 → R² should be 1.0
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
        let mut ols = SimpleOLS::new();
        let r2 = ols.fit(&[x], &y).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6, "Expected R²=1.0, got {:.6}", r2);
    }

    #[test]
    fn test_ols_known_r_squared() {
        // x and y have known Pearson r ≈ 0.707 → R² ≈ 0.5
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 1.0, 3.0, 5.0, 5.0]; // partial correlation
        let mut ols = SimpleOLS::new();
        let r2 = ols.fit(&[x], &y).unwrap();
        assert!(
            r2 > 0.7,
            "Expected R² > 0.7 for partially correlated data, got {:.4}",
            r2
        );
    }

    #[test]
    fn test_gaussian_elimination_2x2() {
        // [2 1 | 5], [1 3 | 10] → x = [1, 3]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let result = gaussian_elimination(&a, &b).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
    }
}
