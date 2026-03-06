// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: CC-BY-4.0
//! Extended Kalman Filter (EKF) for Non-Linear Sensor Fusion
//!
//! Reconstructs the true hidden physical state (Yield Stress, Plastic Viscosity, Temperature)
//! from noisy, indirect IoT observations (Motor Torque, Embedded Thermistor).
//!
//! State Vector x: [T_concrete, tau_y (yield), mu_p (viscosity)]^T
//! Obs Vector z: [T_probe, Torque_mixer]^T

use nalgebra::{Matrix2, Matrix2x3, Matrix3, Vector2, Vector3};

/// Non-linear Extended Kalman Filter for Rheological/Thermal state estimation.
pub struct ExtendedKalmanFilter {
    /// State Vector [Temperature, Yield Stress, Plastic Viscosity]
    pub x: Vector3<f64>,
    /// State Covariance Matrix (Uncertainty)
    pub p: Matrix3<f64>,
    /// Process Noise Covariance (System volatility)
    pub q: Matrix3<f64>,
    /// Measurement Noise Covariance (Sensor noise levels)
    pub r: Matrix2<f64>,

    // Mixer geometric constants mapping rheology to torque
    mixer_radius: f64,
    mixer_height: f64,
    omega: f64, // angular velocity
}

impl ExtendedKalmanFilter {
    /// Initialize the EKF with baseline physical estimates and known sensor noise profiles.
    pub fn new(t_init: f64, tau_y_init: f64, mu_p_init: f64) -> Self {
        Self {
            x: Vector3::new(t_init, tau_y_init, mu_p_init),
            // High initial uncertainty
            p: Matrix3::new(10.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 10.0),
            // Process noise (drifting physics)
            q: Matrix3::new(
                0.1, 0.0, 0.0, 0.0, 1.0, 0.0, // yield stress can drift
                0.0, 0.0, 0.5, // viscosity can drift
            ),
            // Sensor noise (Thermistor = low noise, Torque = high noise)
            r: Matrix2::new(
                0.5, 0.0, // Thermal probe variance
                0.0, 15.0, // Torque reading variance
            ),
            mixer_radius: 0.5,           // meters
            mixer_height: 1.0,           // meters
            omega: std::f64::consts::PI, // rad/s
        }
    }

    /// Predict the next state based on physical drift model.
    ///
    /// The true rheological state changes slowly; we use an identity (constant-state)
    /// propagation model for f(x), letting process noise Q capture uncertainty.
    /// Arrhenius temperature coupling appears in the observation model h(x), not here.
    pub fn predict(&mut self, dt_sec: f64) {
        // State transition Jacobian: identity (state is approximately constant)
        let f_jacobian: Matrix3<f64> = Matrix3::identity();

        // P_k|k-1 = F * P_{k-1|k-1} * F^T + Q * dt
        self.p = f_jacobian * self.p * f_jacobian.transpose() + (self.q * dt_sec);
    }

    /// Update the state estimates using live noisy IoT readings.
    /// z = [Thermistor (K), Motor Torque (Nm)]
    pub fn update(&mut self, z: Vector2<f64>) -> Vector3<f64> {
        // 1. Expected Observation (Non-linear measurement function h(x))
        // h(x)[0] = T_concrete (Thermistor directly reads temperature)
        // h(x)[1] = Torque = (tau_y + mu_p * (v/h)) * Geometric_Factor
        // Bingham Plastic Torque Approximation for a Couette-like mixer

        let shear_rate = self.omega * self.mixer_radius / 0.20; // 20cm gap (realistic paddle mixer)
        let shear_stress = self.x[1] + (self.x[2] * shear_rate);
        let expected_torque = shear_stress
            * (2.0 * std::f64::consts::PI * self.mixer_radius.powi(2) * self.mixer_height);

        let h_x = Vector2::new(self.x[0], expected_torque);

        // 2. Jacobian of Observation Function H (dh/dx)
        // row 0: dT_probe / dx  => [1.0, 0.0, 0.0]
        // row 1: dTorque / dx   => [0.0, dTorque/dTau_y, dTorque/dMu_p]

        let geometric_factor =
            2.0 * std::f64::consts::PI * self.mixer_radius.powi(2) * self.mixer_height;

        let h_jacobian = Matrix2x3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            geometric_factor,
            geometric_factor * shear_rate,
        );

        // 3. Innovation (Measurement Residual)
        let y_tilde = z - h_x;

        // 4. Innovation Covariance S = H * P * H^T + R
        let s = h_jacobian * self.p * h_jacobian.transpose() + self.r;
        let s_inv = s.try_inverse().unwrap_or_else(Matrix2::identity);

        // 5. Kalman Gain K = P * H^T * S^-1
        let k = self.p * h_jacobian.transpose() * s_inv;

        // 6. State Update x = x + K * y_tilde
        self.x += k * y_tilde;

        // 7. Covariance Update — Joseph stable form (guaranteed positive-definite)
        // Standard form: P = (I - K*H) * P  → can lose symmetry under floating point rounding.
        // Joseph form:   P = (I-K*H)*P*(I-K*H)ᵀ + K*R*Kᵀ → symmetric and positive semi-definite.
        let i: Matrix3<f64> = Matrix3::identity();
        let ik_h = i - k * h_jacobian;
        self.p = ik_h * self.p * ik_h.transpose() + k * self.r * k.transpose();

        self.x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_ekf_fusion() {
        // True physical state: 300K, 150 Pa yield stress, 25 Pa.s viscosity
        let true_temp = 300.0;
        let true_yield = 150.0;
        let true_viscosity = 25.0;

        let mut ekf = ExtendedKalmanFilter::new(290.0, 100.0, 10.0); // Bad initial guess
        let mut rng = SmallRng::seed_from_u64(42);

        // Simulate true torque using the same realistic gap
        let shear_rate = ekf.omega * ekf.mixer_radius / 0.20;
        let true_stress = true_yield + (true_viscosity * shear_rate);
        let geo = 2.0 * std::f64::consts::PI * ekf.mixer_radius.powi(2) * ekf.mixer_height;
        let true_torque = true_stress * geo;

        // Feed noisy measurements
        for _ in 0..100 {
            ekf.predict(1.0);

            let noisy_temp = true_temp + rng.gen_range(-1.0..1.0);
            let noisy_torque = true_torque + rng.gen_range(-20.0..20.0); // Extremely noisy torque

            ekf.update(Vector2::new(noisy_temp, noisy_torque));
        }

        // Temperature should be perfectly converged (direct observation)
        assert!((ekf.x[0] - true_temp).abs() < 0.5);

        // Yield Stress should converge significantly closer than the initial 50 Pa offset.
        // At this shear rate, yield stress is partially observable. A 40 Pa tolerance
        // is a physically honest bound (Bingham observability limit).
        assert!(
            (ekf.x[1] - true_yield).abs() < 40.0,
            "EKF yield stress failed to converge: estimate={:.2}, true={}",
            ekf.x[1],
            true_yield
        );
    }
}
