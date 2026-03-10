// SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Santosh Prabhu Shenbagamoorthy, and Studio Tyto
// SPDX-License-Identifier: MIT
//! Liquid Time-Constant (LTC) / Neural ODE integration for PPO.
//!
//! Replaces the standard discrete MLP path with a continuous-time differential equation solver.
//! Instead of mapping Action = f(State), it solves:
//! d(Action)/dt = f_theta(Action, State)
//! This matches the continuous rheological evolution of printing concrete.

use crate::math::ode_solver::rk4_integrate;
use crate::rl::state::RLState;
use rand::Rng;

/// A simple Continuous-Time Neural Actor
/// Acts as the f_theta derivative function for the ODE solver.
#[derive(Clone)]
pub struct LiquidActor {
    /// W_state dimensions: [action_dim x state_dim]
    pub w_state: Vec<Vec<f64>>,
    /// W_act dimensions: [action_dim x action_dim] (Recurrent liquid dynamics)
    pub w_act: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
}

impl LiquidActor {
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        // Initialize with small random weights (Xavier-ish)
        let w_state = (0..action_dim)
            .map(|_| (0..state_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        let w_act = (0..action_dim)
            .map(|_| (0..action_dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        let bias = vec![0.0; action_dim];

        Self {
            w_state,
            w_act,
            bias,
        }
    }

    /// Evaluates the instantaneous derivative df/dt = W_s * S + W_a * A + B
    pub fn compute_derivative(&self, action: &[f64], state: &[f64]) -> Vec<f64> {
        let action_dim = self.bias.len();
        let mut d_act = vec![0.0; action_dim];

        for i in 0..action_dim {
            let mut sum = self.bias[i];

            // Influence from external environment state
            for (j, &s_val) in state.iter().enumerate() {
                sum += self.w_state[i][j] * s_val;
            }
            // Continuous recurrent influence from the current liquid action state
            for (j, &a_val) in action.iter().enumerate() {
                sum += self.w_act[i][j] * a_val;
            }

            // Leaky-ReLU or Tanh activation derivative equivalent
            // For stability in continuous time, we often use a standard bounded nonlinearity
            d_act[i] = sum.tanh();
        }

        d_act
    }

    /// Returns the integrated action trajectory over `dt` from a raw state vector.
    /// This is the vector-input variant used by components that don't have `RLState`.
    pub fn forward_continuous_vec(&self, state: &[f64], prev_action: &[f64], dt: f64) -> Vec<f64> {
        let f = |_t: f64, a: &[f64]| -> Vec<f64> { self.compute_derivative(a, state) };
        let steps = 10;
        let micro_dt = dt / (steps as f64);
        rk4_integrate(f, 0.0, prev_action, micro_dt, steps)
    }

    /// Integrates the ODE from t=0 to t=`total_t` and returns all intermediate
    /// snapshots (one per RK4 step).  Used by MutualInfo estimator.
    pub fn trajectory(&self, state: &[f64], dt_per_step: f64, steps: usize) -> Vec<Vec<f64>> {
        let action_dim = self.bias.len();
        let mut snapshots = Vec::with_capacity(steps + 1);
        let mut current = vec![0.0_f64; action_dim];
        snapshots.push(current.clone());
        for _ in 0..steps {
            current = {
                let s = state.to_owned();
                let f = |_t: f64, a: &[f64]| -> Vec<f64> { self.compute_derivative(a, &s) };
                rk4_integrate(f, 0.0, &current, dt_per_step, 1)
            };
            snapshots.push(current.clone());
        }
        snapshots
    }

    /// Returns the integrated action trajectory over dt
    pub fn forward_continuous(&self, state: &RLState, prev_action: &[f64], dt: f64) -> Vec<f64> {
        let s_vec = state.to_vector();

        // Define the closure for the RK4 solver
        // We capture state immutably as it is the "forcing function" over this dt block
        let f = |_t: f64, a: &[f64]| -> Vec<f64> { self.compute_derivative(a, &s_vec) };

        // Solve IVP from t=0 to t=dt in 10 micro-steps
        let steps = 10;
        let micro_dt = dt / (steps as f64);

        rk4_integrate(f, 0.0, prev_action, micro_dt, steps)
    }
}

/// A wrapper agent demonstrating the Liquid continuous step
pub struct LiquidPPOAgent {
    pub actor: LiquidActor,
    pub action_dim: usize,
}

impl LiquidPPOAgent {
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            actor: LiquidActor::new(state_dim, action_dim),
            action_dim,
        }
    }

    /// Primary inference method
    pub fn get_action(&self, state: &RLState, prev_action: Option<&[f64]>, dt: f64) -> Vec<f64> {
        let fallback = vec![0.0; self.action_dim];
        let initial_action = prev_action.unwrap_or(&fallback);
        self.actor.forward_continuous(state, initial_action, dt)
    }
}
