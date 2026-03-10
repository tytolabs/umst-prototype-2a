// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//!
//! Data Provider Module for Epistemic Proxy Selection
//!
//! Provides real data sources for validation experiments:
//! - UCIDataProvider: Loads UCI Concrete Compressive Strength dataset
//! - ProxyDataSource trait: Interface for data providers
//!
//! Dataset: UCI Concrete Compressive Strength dataset (1030 samples)
//! Features: cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age
//! Target: f28_compressive (28-day compressive strength in MPa)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Trait for proxy data sources - enables switching between synthetic and real data
pub trait ProxyDataSource: Send + Sync {
    /// Reveal the value of one proxy for a sample (simulates taking a measurement)
    fn reveal_proxy(&self, sample_idx: usize, proxy_name: &str) -> Option<f64>;

    /// Get ground truth (28-day strength) for a sample
    fn get_ground_truth(&self, sample_idx: usize) -> f64;

    /// List of available proxy names
    fn proxy_names(&self) -> &[String];

    /// Number of samples
    fn n_samples(&self) -> usize;

    /// Get all proxies for a sample as a hashmap
    fn get_all_proxies(&self, sample_idx: usize) -> HashMap<String, f64>;
}

/// UCI Concrete Dataset sample
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConcreteSample {
    pub cement: f64,
    pub slag: f64,
    pub fly_ash: f64,
    pub water: f64,
    pub superplasticizer: f64,
    pub coarse_agg: f64,
    pub fine_agg: f64,
    pub age: f64,
    pub strength: f64,
    pub temperature: f64,
    pub humidity: f64,
}

/// UCI Concrete Dataset Provider
/// Loads and provides access to the UCI Concrete Compressive Strength dataset
pub struct UCIDataProvider {
    samples: Vec<ConcreteSample>,
    proxy_names: Vec<String>,
}

impl UCIDataProvider {
    /// Load UCI dataset from CSV file
    pub fn from_csv<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file));

        let mut samples = Vec::new();

        for result in reader.records() {
            let record = result?;
            let sample = ConcreteSample {
                cement: record.get(0).unwrap_or("0").parse().unwrap_or(0.0),
                slag: record.get(1).unwrap_or("0").parse().unwrap_or(0.0),
                fly_ash: record.get(2).unwrap_or("0").parse().unwrap_or(0.0),
                water: record.get(3).unwrap_or("0").parse().unwrap_or(0.0),
                superplasticizer: record.get(4).unwrap_or("0").parse().unwrap_or(0.0),
                coarse_agg: record.get(5).unwrap_or("0").parse().unwrap_or(0.0),
                fine_agg: record.get(6).unwrap_or("0").parse().unwrap_or(0.0),
                age: record.get(7).unwrap_or("0").parse().unwrap_or(0.0),
                strength: record.get(8).unwrap_or("0").parse().unwrap_or(0.0),
                temperature: record.get(10).unwrap_or("20.0").parse().unwrap_or(20.0),
                humidity: record.get(11).unwrap_or("0.5").parse().unwrap_or(0.5),
            };
            samples.push(sample);
        }

        let proxy_names = vec![
            "cement".to_string(),
            "slag".to_string(),
            "fly_ash".to_string(),
            "water".to_string(),
            "superplasticizer".to_string(),
            "coarse_agg".to_string(),
            "fine_agg".to_string(),
            "age".to_string(),
            "temperature".to_string(),
            "humidity".to_string(),
        ];

        Ok(UCIDataProvider {
            samples,
            proxy_names,
        })
    }

    /// Create with pre-loaded samples (for testing)
    pub fn with_samples(samples: Vec<ConcreteSample>) -> Self {
        let proxy_names = vec![
            "cement".to_string(),
            "slag".to_string(),
            "fly_ash".to_string(),
            "water".to_string(),
            "superplasticizer".to_string(),
            "coarse_agg".to_string(),
            "fine_agg".to_string(),
            "age".to_string(),
            "temperature".to_string(),
            "humidity".to_string(),
        ];
        UCIDataProvider {
            samples,
            proxy_names,
        }
    }

    /// Calculate mutual information (using Pearson correlation as proxy)
    pub fn calculate_correlation(&self, proxy_name: &str) -> f64 {
        let mut values = Vec::new();
        let mut strengths = Vec::new();

        for sample in &self.samples {
            let proxy_value = match proxy_name {
                "cement" => sample.cement,
                "slag" => sample.slag,
                "fly_ash" => sample.fly_ash,
                "water" => sample.water,
                "superplasticizer" => sample.superplasticizer,
                "coarse_agg" => sample.coarse_agg,
                "fine_agg" => sample.fine_agg,
                "age" => sample.age,
                "temperature" => sample.temperature,
                "humidity" => sample.humidity,
                _ => continue,
            };
            values.push(proxy_value);
            strengths.push(sample.strength);
        }

        if values.len() < 2 {
            return 0.0;
        }

        Self::pearson_correlation(&values, &strengths).abs()
    }

    /// Pearson correlation coefficient
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Get all correlation values for ranking proxies by information content
    pub fn get_all_correlations(&self) -> HashMap<String, f64> {
        let mut correlations = HashMap::new();
        for name in &self.proxy_names {
            correlations.insert(name.clone(), self.calculate_correlation(name));
        }
        correlations
    }
}

impl ProxyDataSource for UCIDataProvider {
    fn reveal_proxy(&self, sample_idx: usize, proxy_name: &str) -> Option<f64> {
        let sample = self.samples.get(sample_idx)?;
        let value = match proxy_name {
            "cement" => sample.cement,
            "slag" => sample.slag,
            "fly_ash" => sample.fly_ash,
            "water" => sample.water,
            "superplasticizer" => sample.superplasticizer,
            "coarse_agg" => sample.coarse_agg,
            "fine_agg" => sample.fine_agg,
            "age" => sample.age,
            _ => return None,
        };
        Some(value)
    }

    fn get_ground_truth(&self, sample_idx: usize) -> f64 {
        self.samples
            .get(sample_idx)
            .map(|s| s.strength)
            .unwrap_or(0.0)
    }

    fn proxy_names(&self) -> &[String] {
        &self.proxy_names
    }

    fn n_samples(&self) -> usize {
        self.samples.len()
    }

    fn get_all_proxies(&self, sample_idx: usize) -> HashMap<String, f64> {
        let sample = match self.samples.get(sample_idx) {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let mut proxies = HashMap::new();
        proxies.insert("cement".to_string(), sample.cement);
        proxies.insert("slag".to_string(), sample.slag);
        proxies.insert("fly_ash".to_string(), sample.fly_ash);
        proxies.insert("water".to_string(), sample.water);
        proxies.insert("superplasticizer".to_string(), sample.superplasticizer);
        proxies.insert("coarse_agg".to_string(), sample.coarse_agg);
        proxies.insert("fine_agg".to_string(), sample.fine_agg);
        proxies.insert("age".to_string(), sample.age);

        proxies
    }
}

/// Synthetic data provider for testing (fallback when real data unavailable)
pub struct SyntheticDataProvider {
    n_samples: usize,
    proxy_names: Vec<String>,
}

impl SyntheticDataProvider {
    pub fn new(n_samples: usize) -> Self {
        SyntheticDataProvider {
            n_samples,
            proxy_names: vec![
                "cement".to_string(),
                "water".to_string(),
                "age".to_string(),
                "slag".to_string(),
                "superplasticizer".to_string(),
            ],
        }
    }
}

impl ProxyDataSource for SyntheticDataProvider {
    fn reveal_proxy(&self, _sample_idx: usize, proxy_name: &str) -> Option<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate realistic values based on proxy type
        let value = match proxy_name {
            "cement" => 300.0 + rng.gen::<f64>() * 200.0, // 300-500 kg/m³
            "water" => 150.0 + rng.gen::<f64>() * 100.0,  // 150-250 kg/m³
            "age" => 1.0 + rng.gen::<f64>() * 364.0,      // 1-365 days
            "slag" => rng.gen::<f64>() * 200.0,           // 0-200 kg/m³
            "superplasticizer" => rng.gen::<f64>() * 10.0, // 0-10 kg/m³
            _ => rng.gen::<f64>() * 100.0,
        };

        Some(value)
    }

    fn get_ground_truth(&self, sample_idx: usize) -> f64 {
        // Simplified: strength = f(cement, water, age) with some noise
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let base = 40.0 + (sample_idx as f64 % 30.0);
        base + rng.gen::<f64>() * 10.0 - 5.0
    }

    fn proxy_names(&self) -> &[String] {
        &self.proxy_names
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn get_all_proxies(&self, sample_idx: usize) -> HashMap<String, f64> {
        let mut proxies = HashMap::new();
        for name in &self.proxy_names {
            if let Some(v) = self.reveal_proxy(sample_idx, name) {
                proxies.insert(name.clone(), v);
            }
        }
        proxies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_provider() {
        let provider = SyntheticDataProvider::new(100);

        assert_eq!(provider.n_samples(), 100);
        assert!(!provider.proxy_names().is_empty());

        let value = provider.reveal_proxy(0, "cement");
        assert!(value.is_some());

        let gt = provider.get_ground_truth(0);
        assert!(gt > 0.0);
    }

    #[test]
    fn test_correlation_calculation() {
        // Create simple test data with known correlation
        let samples = vec![
            ConcreteSample {
                cement: 100.0,
                slag: 0.0,
                fly_ash: 0.0,
                water: 50.0,
                superplasticizer: 0.0,
                coarse_agg: 500.0,
                fine_agg: 400.0,
                age: 7.0,
                strength: 20.0,
                temperature: 20.0,
                humidity: 0.5,
            },
            ConcreteSample {
                cement: 200.0,
                slag: 0.0,
                fly_ash: 0.0,
                water: 100.0,
                superplasticizer: 0.0,
                coarse_agg: 500.0,
                fine_agg: 400.0,
                age: 14.0,
                strength: 35.0,
                temperature: 20.0,
                humidity: 0.5,
            },
            ConcreteSample {
                cement: 300.0,
                slag: 0.0,
                fly_ash: 0.0,
                water: 150.0,
                superplasticizer: 0.0,
                coarse_agg: 500.0,
                fine_agg: 400.0,
                age: 28.0,
                strength: 45.0,
                temperature: 20.0,
                humidity: 0.5,
            },
        ];

        let provider = UCIDataProvider::with_samples(samples);

        let corr = provider.calculate_correlation("cement");
        println!("Cement-strength correlation: {}", corr);
        assert!(corr > 0.5); // Should be positively correlated
    }
}
