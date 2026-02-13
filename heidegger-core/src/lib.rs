pub mod kinematics;
pub mod collision;
pub mod safety_set;
pub mod cbf;

use serde::{Deserialize, Serialize};

/// A single joint's safety specification, parsed from URDF or manual config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimit {
    pub name: String,
    /// Lower position limit in radians
    pub lower: f64,
    /// Upper position limit in radians
    pub upper: f64,
    /// Maximum absolute velocity in rad/s (0 = unlimited)
    pub max_velocity: f64,
    /// Maximum absolute effort/torque in Nm (0 = unlimited)
    pub max_effort: f64,
}

/// The result of a safety check on a single action step.
#[derive(Debug, Clone)]
pub struct SafetyCheckResult {
    /// The action after safety clamping (may differ from input)
    pub safe_action: Vec<f64>,
    /// Whether any clamping was applied
    pub was_clamped: bool,
    /// Per-joint details of what happened
    pub details: Vec<JointCheckDetail>,
}

/// What happened to a single joint during safety check.
#[derive(Debug, Clone)]
pub struct JointCheckDetail {
    pub joint_name: String,
    pub original_value: f64,
    pub clamped_value: f64,
    pub reason: Option<ClampReason>,
}

#[derive(Debug, Clone)]
pub enum ClampReason {
    /// Position would exceed lower limit
    BelowLowerLimit { limit: f64 },
    /// Position would exceed upper limit
    AboveUpperLimit { limit: f64 },
    /// Velocity would exceed max
    VelocityExceeded { max: f64, actual: f64 },
}

impl std::fmt::Display for ClampReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClampReason::BelowLowerLimit { limit } => {
                write!(f, "Below lower limit ({:.3} rad)", limit)
            }
            ClampReason::AboveUpperLimit { limit } => {
                write!(f, "Above upper limit ({:.3} rad)", limit)
            }
            ClampReason::VelocityExceeded { max, actual } => {
                write!(f, "Velocity {:.3} rad/s exceeds max {:.3}", actual, max)
            }
        }
    }
}

/// The core safety shim. Stateless per-call, configuration loaded once.
pub struct SafetyShim {
    pub joints: Vec<JointLimit>,
    /// Control loop dt in seconds, used for velocity estimation
    pub dt: f64,
}

impl SafetyShim {
    /// Create a SafetyShim from a list of joint limits.
    pub fn new(joints: Vec<JointLimit>, dt: f64) -> Self {
        Self { joints, dt }
    }

    /// Create from a JSON string (for easy Python interop before URDF parser).
    pub fn from_json(json: &str, dt: f64) -> Result<Self, String> {
        let joints: Vec<JointLimit> =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;
        Ok(Self::new(joints, dt))
    }

    /// Core safety check: clamp raw action to safe ranges.
    ///
    /// `raw_action`: target joint positions from VLA model (radians)
    /// `current_positions`: current joint positions (radians)
    ///
    /// Returns the clamped action and details of any interventions.
    pub fn check(
        &self,
        raw_action: &[f64],
        current_positions: &[f64],
    ) -> Result<SafetyCheckResult, String> {
        if raw_action.len() != self.joints.len() {
            return Err(format!(
                "Action dimension {} != joint count {}",
                raw_action.len(),
                self.joints.len()
            ));
        }
        if current_positions.len() != self.joints.len() {
            return Err(format!(
                "Current positions dimension {} != joint count {}",
                current_positions.len(),
                self.joints.len()
            ));
        }

        let mut safe_action = Vec::with_capacity(raw_action.len());
        let mut details = Vec::with_capacity(raw_action.len());
        let mut was_clamped = false;

        for (i, joint) in self.joints.iter().enumerate() {
            let raw = raw_action[i];
            let current = current_positions[i];
            let mut clamped = raw;
            let mut reason = None;

            // 1. Position limit clamping
            if clamped < joint.lower {
                reason = Some(ClampReason::BelowLowerLimit { limit: joint.lower });
                clamped = joint.lower;
            } else if clamped > joint.upper {
                reason = Some(ClampReason::AboveUpperLimit { limit: joint.upper });
                clamped = joint.upper;
            }

            // 2. Velocity clamping (if max_velocity > 0 and dt > 0)
            if joint.max_velocity > 0.0 && self.dt > 0.0 {
                let velocity = (clamped - current) / self.dt;
                if velocity.abs() > joint.max_velocity {
                    let max_delta = joint.max_velocity * self.dt;
                    let actual_vel = velocity;
                    clamped = current + max_delta * velocity.signum();
                    // Re-clamp to position limits after velocity adjustment
                    clamped = clamped.clamp(joint.lower, joint.upper);
                    reason = Some(ClampReason::VelocityExceeded {
                        max: joint.max_velocity,
                        actual: actual_vel,
                    });
                }
            }

            if (clamped - raw).abs() > 1e-9 {
                was_clamped = true;
            }

            details.push(JointCheckDetail {
                joint_name: joint.name.clone(),
                original_value: raw,
                clamped_value: clamped,
                reason,
            });

            safe_action.push(clamped);
        }

        Ok(SafetyCheckResult {
            safe_action,
            was_clamped,
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_joints() -> Vec<JointLimit> {
        vec![
            JointLimit {
                name: "joint1".to_string(),
                lower: -1.57,
                upper: 1.57,
                max_velocity: 3.14,
                max_effort: 10.0,
            },
            JointLimit {
                name: "joint2".to_string(),
                lower: -0.78,
                upper: 2.35,
                max_velocity: 2.0,
                max_effort: 10.0,
            },
        ]
    }

    #[test]
    fn test_safe_action_passes_through() {
        let shim = SafetyShim::new(test_joints(), 0.02);
        // Deltas: joint1=0.01 (max_delta=0.0628), joint2=0.02 (max_delta=0.04) — both within limits
        let result = shim.check(&[0.51, 1.02], &[0.5, 1.0]).unwrap();
        assert!(!result.was_clamped);
        assert_eq!(result.safe_action, vec![0.51, 1.02]);
    }

    #[test]
    fn test_position_clamp_upper() {
        let shim = SafetyShim::new(test_joints(), 0.02);
        let result = shim.check(&[3.0, 1.0], &[0.0, 0.0]).unwrap();
        assert!(result.was_clamped);
        assert!(result.safe_action[0] <= 1.57);
    }

    #[test]
    fn test_position_clamp_lower() {
        let shim = SafetyShim::new(test_joints(), 0.02);
        let result = shim.check(&[-3.0, 1.0], &[0.0, 0.0]).unwrap();
        assert!(result.was_clamped);
        assert!(result.safe_action[0] >= -1.57);
    }

    #[test]
    fn test_velocity_clamp() {
        let shim = SafetyShim::new(test_joints(), 0.02); // 50Hz
        // Joint2 max_velocity=2.0, dt=0.02 → max_delta=0.04 rad per step
        // Trying to jump from 0.0 to 1.0 in one step → velocity=50 rad/s → clamped
        let result = shim.check(&[0.0, 1.0], &[0.0, 0.0]).unwrap();
        assert!(result.was_clamped);
        // Should only move by max_delta = 0.04
        assert!((result.safe_action[1] - 0.04).abs() < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch() {
        let shim = SafetyShim::new(test_joints(), 0.02);
        let result = shim.check(&[0.5], &[0.0, 0.0]);
        assert!(result.is_err());
    }
}
