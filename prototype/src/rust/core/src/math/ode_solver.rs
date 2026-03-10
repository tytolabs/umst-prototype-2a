// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT

//! A generic 4th-order Runge-Kutta (RK4) ODE solver for continuous-time integration.
//! This allows neural networks and physics engines to evaluate continuous flow
//! over time rather than discrete jumps.

/// Solves an initial value problem (IVP) for a system of ODEs: dy/dt = f(t, y)
/// using the classic 4th-order Runge-Kutta method.
///
/// `f`: The derivative function `dh/dt = f(h(t), t)`
/// `t0`: Initial time
/// `y0`: Initial state vector
/// `dt`: Step size
/// `steps`: Number of steps to integrate over
pub fn rk4_integrate<F>(f: F, t0: f64, y0: &[f64], dt: f64, steps: usize) -> Vec<f64>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let mut y = y0.to_vec();
    let mut t = t0;
    let n = y.len();

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut y_temp = vec![0.0; n];

    for _ in 0..steps {
        // k1 = f(t, y)
        k1.clone_from(&f(t, &y));

        // k2 = f(t + dt/2, y + dt/2 * k1)
        for i in 0..n {
            y_temp[i] = y[i] + 0.5 * dt * k1[i];
        }
        k2.clone_from(&f(t + 0.5 * dt, &y_temp));

        // k3 = f(t + dt/2, y + dt/2 * k2)
        for i in 0..n {
            y_temp[i] = y[i] + 0.5 * dt * k2[i];
        }
        k3.clone_from(&f(t + 0.5 * dt, &y_temp));

        // k4 = f(t + dt, y + dt * k3)
        for i in 0..n {
            y_temp[i] = y[i] + dt * k3[i];
        }
        k4.clone_from(&f(t + dt, &y_temp));

        // y(t + dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for i in 0..n {
            y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += dt;
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_ode_sine_wave() {
        // Unforced harmonic oscillator: y'' = -y
        // Test with a simple harmonic oscillator: y'' = -y -> y1' = y2, y2' = -y1
        let harmonic_oscillator = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };

        // Initial conditions: pos = 0.0, vel = 1.0 (Sinusoidal wave starting at 0, amplitude 1)
        let y0 = vec![0.0, 1.0];

        // Integrate from t=0 to t=PI/2
        let dt = 0.01;
        let steps = ((PI / 2.0) / dt).round() as usize;

        // Re-adjust dt slightly to hit exactly PI/2 at the final step
        let dt = (PI / 2.0) / (steps as f64);

        let y_final = rk4_integrate(harmonic_oscillator, 0.0, &y0, dt, steps);

        // At t=PI/2, sin(PI/2) = 1.0, cos(PI/2) = 0.0
        // Our state vector is [sin(t), cos(t)]
        let expected_pos = 1.0;
        let expected_vel = 0.0;

        let error_pos = (y_final[0] - expected_pos).abs();
        let error_vel = (y_final[1] - expected_vel).abs();

        println!(
            "RK4 integration final pos: {}, vel: {}",
            y_final[0], y_final[1]
        );
        println!(
            "Absolute errors — pos: {:.2e}, vel: {:.2e}",
            error_pos, error_vel
        );

        // Verify absolute integration precision < 1e-4 tolerance
        assert!(
            error_pos < 1e-4,
            "Positional error exceeded tolerance: {}",
            error_pos
        );
        assert!(
            error_vel < 1e-4,
            "Velocity error exceeded tolerance: {}",
            error_vel
        );
    }
}
