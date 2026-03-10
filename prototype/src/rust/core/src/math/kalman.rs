// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! 1D Kalman Filter for Thermal Power Estimation
//!
//! Smooths the raw RAPL energy readings to isolate inference-driven heat from
//! baseline thermal inertia / noise. Key application: separating Joule heating
//! spikes caused by computing physically impossible policies from steady-state
//! background dissipation.

/// A minimal 1D Kalman filter over energy/power readings.
///
/// State: estimated true power (µW)
/// Observation: raw RAPL energy delta (µJ) converted to µW
pub struct KalmanFilter1D {
    /// Current state estimate (µW)
    pub x: f64,
    /// Estimate uncertainty (error covariance)
    pub p: f64,
    /// Process noise covariance (how much we expect state to change each step)
    pub q: f64,
    /// Measurement noise covariance (how noisy the RAPL readings are)
    pub r: f64,
}

impl KalmanFilter1D {
    /// Create a new filter with initial state and standard RAPL noise params.
    pub fn new(initial_power_uw: f64) -> Self {
        Self {
            x: initial_power_uw,
            p: 1000.0, // high initial uncertainty
            q: 10.0,   // small process noise (power changes slowly)
            r: 500.0,  // moderate RAPL measurement noise (~500 µW std dev)
        }
    }

    /// Predict step: advance time, increase uncertainty.
    /// Uncertainty grows by process noise scaled by elapsed time.
    fn predict(&mut self, dt_ms: f64) {
        self.p += self.q * dt_ms;
    }

    /// Update step: incorporate a new RAPL observation.
    ///
    /// `z` is the measured value for this timestep.
    /// `dt_ms` is the elapsed time/interval since last update.
    pub fn update(&mut self, z: f64, dt_ms: f64) -> f64 {
        self.predict(dt_ms);

        // Kalman gain
        let k = self.p / (self.p + self.r);

        // State update
        self.x += k * (z - self.x);

        // Covariance update
        self.p *= 1.0 - k;

        self.x
    }

    /// Current filtered estimate.
    pub fn estimate(&self) -> f64 {
        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_kalman_converges_to_true_value() {
        let mut kf = KalmanFilter1D::new(0.0);
        let mut rng = SmallRng::seed_from_u64(42);

        // Feed 100 noisy observations of true power = 1000 µW
        // Assuming dt = 1.0 ms per step
        for _ in 0..100 {
            let noise: f64 = rng.gen_range(-1.0..1.0);
            kf.update(1000.0 + (noise * 50.0), 1.0);
        }
        // Should converge within 5% of true value
        let err = (kf.estimate() - 1000.0).abs() / 1000.0;
        assert!(
            err < 0.05,
            "Kalman failed to converge: estimate={:.2}",
            kf.estimate()
        );
    }
}
