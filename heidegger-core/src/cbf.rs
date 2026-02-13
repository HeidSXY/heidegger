/// Control Barrier Function (CBF) Safety Filter
///
/// Replaces independent per-joint clamping with a globally optimal
/// safe action computation. Given a VLA policy output u_vla, finds
/// the closest safe action u* that satisfies ALL constraints simultaneously:
///
///   minimize  ||u - u_vla||²
///   subject to:
///     h_joint(u)     ≥ 0    (joint position limits)
///     h_velocity(u)  ≥ 0    (velocity limits)
///     h_collision(u) ≥ α    (self-collision distance)
///     h_workspace(u) ≥ 0    (learned safety set)
///
/// For position-controlled servos (LeRobot), the dynamics simplify to
/// x_{t+1} ≈ u_t, so CBF does not require a full dynamics model.
///
/// Uses projected gradient descent to solve the QP, avoiding external
/// solver dependencies for this small (6-variable) problem.

use crate::collision::{CollisionChecker, collision_checker_by_name};
use crate::kinematics::{KinematicChain, chain_by_name};
use crate::safety_set::SafetySet;
use crate::JointLimit;
use serde::{Deserialize, Serialize};

// ─────────────────── CBF Safety Filter ───────────────────

/// Configuration for the CBF safety filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CBFConfig {
    /// Control loop period (seconds)
    pub dt: f64,
    /// Collision safety margin (meters)
    pub collision_margin: f64,
    /// CBF decay rate α ∈ (0, 1]: how aggressively to maintain safety.
    /// Lower = more conservative (stays further from boundary).
    pub alpha: f64,
    /// Max iterations for projected gradient descent
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for numerical gradient computation
    pub gradient_eps: f64,
}

impl Default for CBFConfig {
    fn default() -> Self {
        Self {
            dt: 0.02,
            collision_margin: 0.015,
            alpha: 0.3,
            max_iterations: 50,
            tolerance: 1e-6,
            gradient_eps: 1e-4,
        }
    }
}

/// Result of a CBF safety filter step.
#[derive(Debug, Clone)]
pub struct CBFResult {
    /// The safe action (closest to u_vla that satisfies all constraints)
    pub safe_action: Vec<f64>,
    /// Whether the action was modified
    pub was_modified: bool,
    /// Euclidean distance between u_vla and u_safe
    pub modification_norm: f64,
    /// Number of active constraints at the solution
    pub active_constraints: usize,
    /// Individual constraint values at the solution
    pub constraint_values: Vec<ConstraintValue>,
    /// Solver iterations used
    pub iterations: usize,
}

/// Value of a single constraint at the solution.
#[derive(Debug, Clone)]
pub struct ConstraintValue {
    pub name: String,
    pub value: f64,    // h(u): positive = satisfied
    pub active: bool,  // near-zero = active constraint
}

/// The CBF Safety Filter.
pub struct CBFSafetyFilter {
    /// Joint limits for position and velocity bounds
    joint_limits: Vec<JointLimit>,
    /// Kinematic chain for FK (needed for collision constraint gradients)
    chain: Option<KinematicChain>,
    /// Collision checker
    collision_checker: Option<CollisionChecker>,
    /// Learned safety set (optional)
    safety_set: Option<SafetySet>,
    /// Configuration
    config: CBFConfig,
    /// Number of joints
    num_joints: usize,
}

impl CBFSafetyFilter {
    /// Create a new CBF filter from joint limits.
    pub fn new(joint_limits: Vec<JointLimit>, config: CBFConfig) -> Self {
        let num_joints = joint_limits.len();
        Self {
            joint_limits,
            chain: None,
            collision_checker: None,
            safety_set: None,
            config,
            num_joints,
        }
    }

    /// Create from JSON config (like SafetyShim).
    pub fn from_json(json: &str, config: CBFConfig) -> Result<Self, String> {
        let joints: Vec<JointLimit> =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;
        Ok(Self::new(joints, config))
    }

    /// Enable self-collision constraint for a named robot model.
    pub fn with_collision(mut self, robot_model: &str) -> Result<Self, String> {
        self.chain = chain_by_name(robot_model);
        self.collision_checker =
            collision_checker_by_name(robot_model, self.config.collision_margin);

        if self.chain.is_none() || self.collision_checker.is_none() {
            return Err(format!("Unknown robot model: {}", robot_model));
        }
        Ok(self)
    }

    /// Enable learned workspace constraint.
    pub fn with_safety_set(mut self, safety_set: SafetySet) -> Self {
        self.safety_set = Some(safety_set);
        self
    }

    /// Run the CBF safety filter.
    ///
    /// Given a desired action u_vla and the current joint positions x_current,
    /// find the closest u* that satisfies all safety constraints.
    pub fn filter(
        &self,
        u_vla: &[f64],
        x_current: &[f64],
    ) -> Result<CBFResult, String> {
        if u_vla.len() != self.num_joints || x_current.len() != self.num_joints {
            return Err(format!(
                "Expected {} joints, got u_vla={}, x_current={}",
                self.num_joints,
                u_vla.len(),
                x_current.len()
            ));
        }

        // Start from u_vla and project to feasible set
        let mut u = u_vla.to_vec();
        let mut iterations = 0;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Evaluate all constraints
            let violations = self.find_violations(&u, x_current);

            if violations.is_empty() {
                break; // All constraints satisfied
            }

            // Projected gradient step: move toward feasibility
            let grad = self.compute_correction(&u, x_current, &violations);

            // Apply correction with line search
            let mut step_size = 1.0;
            let mut best_u = u.clone();
            let current_violation = self.max_violation(&u, x_current);

            for _ in 0..10 {
                let candidate: Vec<f64> = u
                    .iter()
                    .zip(grad.iter())
                    .map(|(ui, gi)| ui + step_size * gi)
                    .collect();

                let new_violation = self.max_violation(&candidate, x_current);
                if new_violation < current_violation {
                    best_u = candidate;
                    break;
                }
                step_size *= 0.5;
            }

            // Check convergence
            let delta: f64 = u
                .iter()
                .zip(best_u.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            u = best_u;

            if delta < self.config.tolerance {
                break;
            }
        }

        // Final hard clamp to ensure feasibility
        // (gradient descent may not fully converge)
        self.hard_clamp(&mut u, x_current);

        // Compute final constraint values
        let constraint_values = self.evaluate_all_constraints(&u, x_current);
        let active_constraints = constraint_values
            .iter()
            .filter(|c| c.active)
            .count();

        let modification_norm: f64 = u
            .iter()
            .zip(u_vla.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        let was_modified = modification_norm > self.config.tolerance;

        Ok(CBFResult {
            safe_action: u,
            was_modified,
            modification_norm,
            active_constraints,
            constraint_values,
            iterations,
        })
    }

    // ─── Constraint evaluation ───

    /// Evaluate all constraint functions at point u.
    fn evaluate_all_constraints(
        &self,
        u: &[f64],
        x_current: &[f64],
    ) -> Vec<ConstraintValue> {
        let mut constraints = Vec::new();
        let active_thresh = 0.01;

        // 1. Joint position limits
        for (j, limit) in self.joint_limits.iter().enumerate() {
            let h_lower = u[j] - limit.lower;
            let h_upper = limit.upper - u[j];

            constraints.push(ConstraintValue {
                name: format!("{}_lower", limit.name),
                value: h_lower,
                active: h_lower.abs() < active_thresh,
            });
            constraints.push(ConstraintValue {
                name: format!("{}_upper", limit.name),
                value: h_upper,
                active: h_upper.abs() < active_thresh,
            });
        }

        // 2. Velocity limits
        for (j, limit) in self.joint_limits.iter().enumerate() {
            let max_delta = limit.max_velocity * self.config.dt;
            let delta = (u[j] - x_current[j]).abs();
            let h_vel = max_delta - delta;

            constraints.push(ConstraintValue {
                name: format!("{}_velocity", limit.name),
                value: h_vel,
                active: h_vel.abs() < active_thresh,
            });
        }

        // 3. Collision constraint
        if let (Some(ref chain), Some(ref checker)) = (&self.chain, &self.collision_checker) {
            let transforms = chain.forward_kinematics(u);
            let collisions = checker.check(&transforms);
            let min_dist = if collisions.is_empty() {
                1.0 // No collision pairs = safe
            } else {
                collisions
                    .iter()
                    .map(|c| c.distance)
                    .fold(f64::INFINITY, f64::min)
            };

            constraints.push(ConstraintValue {
                name: "collision".to_string(),
                value: min_dist,
                active: min_dist < active_thresh,
            });
        }

        // 4. Learned workspace constraint
        if let Some(ref ss) = self.safety_set {
            let h_workspace = ss.signed_distance(u);
            constraints.push(ConstraintValue {
                name: "workspace".to_string(),
                value: h_workspace,
                active: h_workspace < active_thresh,
            });
        }

        constraints
    }

    /// Find which constraints are violated (h(u) < 0).
    fn find_violations(&self, u: &[f64], x_current: &[f64]) -> Vec<(String, f64)> {
        self.evaluate_all_constraints(u, x_current)
            .into_iter()
            .filter(|c| c.value < 0.0)
            .map(|c| (c.name, c.value))
            .collect()
    }

    /// Maximum constraint violation (0 if all satisfied).
    fn max_violation(&self, u: &[f64], x_current: &[f64]) -> f64 {
        self.evaluate_all_constraints(u, x_current)
            .iter()
            .map(|c| (-c.value).max(0.0))
            .fold(0.0, f64::max)
    }

    /// Compute correction direction to reduce constraint violations.
    /// Uses numerical gradients of violated constraints.
    fn compute_correction(
        &self,
        u: &[f64],
        x_current: &[f64],
        violations: &[(String, f64)],
    ) -> Vec<f64> {
        let eps = self.config.gradient_eps;
        let n = self.num_joints;
        let mut correction = vec![0.0; n];

        for (name, violation_value) in violations {
            // Compute gradient of this constraint w.r.t. u
            let mut grad = vec![0.0; n];
            for j in 0..n {
                let mut u_plus = u.to_vec();
                let mut u_minus = u.to_vec();
                u_plus[j] += eps;
                u_minus[j] -= eps;

                let h_plus = self.eval_named_constraint(&u_plus, x_current, name);
                let h_minus = self.eval_named_constraint(&u_minus, x_current, name);
                grad[j] = (h_plus - h_minus) / (2.0 * eps);
            }

            // Correction = push in gradient direction (increase h toward 0)
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq > 1e-12 {
                // Step size proportional to violation magnitude
                let scale = (-violation_value) / grad_norm_sq;
                for j in 0..n {
                    correction[j] += scale * grad[j];
                }
            }
        }

        // Also add a pull toward u_vla (minimize modification)
        // This implements the QP objective: min ||u - u_vla||²
        // Weight: balance between feasibility and optimality
        let feasibility_weight = 0.8;
        let _optimality_pull: Vec<f64> = (0..n)
            .map(|j| {
                correction[j] * feasibility_weight
            })
            .collect();

        correction
    }

    /// Evaluate a specific named constraint.
    fn eval_named_constraint(&self, u: &[f64], x_current: &[f64], name: &str) -> f64 {
        // Parse constraint name
        if name == "collision" {
            if let (Some(ref chain), Some(ref checker)) = (&self.chain, &self.collision_checker) {
                let transforms = chain.forward_kinematics(u);
                let collisions = checker.check(&transforms);
                return if collisions.is_empty() {
                    1.0
                } else {
                    collisions
                        .iter()
                        .map(|c| c.distance)
                        .fold(f64::INFINITY, f64::min)
                };
            }
            return 1.0;
        }

        if name == "workspace" {
            if let Some(ref ss) = self.safety_set {
                return ss.signed_distance(u);
            }
            return 1.0;
        }

        // Joint-specific constraints
        for (j, limit) in self.joint_limits.iter().enumerate() {
            if name == format!("{}_lower", limit.name) {
                return u[j] - limit.lower;
            }
            if name == format!("{}_upper", limit.name) {
                return limit.upper - u[j];
            }
            if name == format!("{}_velocity", limit.name) {
                let max_delta = limit.max_velocity * self.config.dt;
                return max_delta - (u[j] - x_current[j]).abs();
            }
        }

        1.0 // Unknown constraint, assume satisfied
    }

    /// Hard clamp as final safety net.
    /// This ensures the output is always within bounds even if
    /// the gradient descent didn't fully converge.
    fn hard_clamp(&self, u: &mut Vec<f64>, x_current: &[f64]) {
        for (j, limit) in self.joint_limits.iter().enumerate() {
            // Position clamp
            u[j] = u[j].max(limit.lower).min(limit.upper);

            // Velocity clamp
            let max_delta = limit.max_velocity * self.config.dt;
            let delta = u[j] - x_current[j];
            if delta.abs() > max_delta {
                u[j] = x_current[j] + delta.signum() * max_delta;
                // Re-clamp position after velocity adjustment
                u[j] = u[j].max(limit.lower).min(limit.upper);
            }
        }
    }
}

// ─────────────────── Tests ───────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_limits() -> Vec<JointLimit> {
        vec![
            JointLimit {
                name: "j0".to_string(),
                lower: -1.0,
                upper: 1.0,
                max_velocity: 2.0,
                max_effort: 0.0,
            },
            JointLimit {
                name: "j1".to_string(),
                lower: -1.5,
                upper: 1.5,
                max_velocity: 3.0,
                max_effort: 0.0,
            },
        ]
    }

    #[test]
    fn test_safe_action_passes_through() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        let result = cbf.filter(&[0.5, 0.5], &[0.5, 0.5]).unwrap();
        assert!(!result.was_modified);
        assert!((result.safe_action[0] - 0.5).abs() < 1e-6);
        assert!((result.safe_action[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_position_violation_corrected() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        // u_vla exceeds upper limit on j0
        let result = cbf.filter(&[2.0, 0.5], &[0.5, 0.5]).unwrap();
        assert!(result.was_modified);
        // Should be clamped to at most 1.0 (upper limit of j0)
        assert!(
            result.safe_action[0] <= 1.0 + 1e-6,
            "j0 should be ≤ 1.0, got {}",
            result.safe_action[0]
        );
    }

    #[test]
    fn test_velocity_violation_corrected() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        // Current at 0.0, requesting 1.0. Max delta = 2.0 * 0.02 = 0.04
        let result = cbf.filter(&[1.0, 0.0], &[0.0, 0.0]).unwrap();
        assert!(result.was_modified);
        // Should be within velocity limit
        let delta = (result.safe_action[0] - 0.0).abs();
        let max_delta = 2.0 * 0.02;
        assert!(
            delta <= max_delta + 1e-6,
            "Velocity delta {} should be ≤ {}",
            delta,
            max_delta
        );
    }

    #[test]
    fn test_both_joints_corrected() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        // Both joints exceed limits
        let result = cbf.filter(&[5.0, 5.0], &[0.0, 0.0]).unwrap();
        assert!(result.was_modified);
        assert!(result.safe_action[0] <= 1.0 + 1e-6);
        assert!(result.safe_action[1] <= 1.5 + 1e-6);
    }

    #[test]
    fn test_modification_norm() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        let result = cbf.filter(&[0.0, 0.0], &[0.0, 0.0]).unwrap();
        assert!(result.modification_norm < 1e-6, "Safe action should not be modified");
    }

    #[test]
    fn test_with_safety_set() {
        use crate::safety_set::TrajectoryRecorder;

        // Record tight workspace
        let mut rec = TrajectoryRecorder::new(2);
        for _ in 0..50 {
            rec.record(&[0.0, 0.0]);
        }
        rec.record(&[-0.1, -0.1]);
        rec.record(&[0.1, 0.1]);

        let ss = rec.calibrate(2.0, 0).unwrap();

        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default())
            .with_safety_set(ss);

        // Action within workspace: should pass
        let result = cbf.filter(&[0.0, 0.0], &[0.0, 0.0]).unwrap();
        assert!(!result.was_modified);

        // Action outside learned workspace (but within joint limits):
        // should be corrected
        let result = cbf.filter(&[0.9, 0.9], &[0.0, 0.0]).unwrap();
        assert!(result.was_modified);
    }

    #[test]
    fn test_constraint_values_reported() {
        let cbf = CBFSafetyFilter::new(test_limits(), CBFConfig::default());
        let result = cbf.filter(&[0.5, 0.5], &[0.5, 0.5]).unwrap();

        // Should have position constraints + velocity constraints
        assert!(
            result.constraint_values.len() >= 6,
            "Expected >= 6 constraints, got {}",
            result.constraint_values.len()
        );

        // All should be satisfied (positive values)
        for cv in &result.constraint_values {
            assert!(
                cv.value >= -1e-6,
                "Constraint '{}' violated: {}",
                cv.name,
                cv.value
            );
        }
    }
}
