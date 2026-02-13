/// Forward Kinematics for serial robot chains.
///
/// Given a chain of joints (each with DH-like parameters),
/// computes the 3D pose (position + orientation) of every link frame.

/// A 4x4 homogeneous transformation matrix stored as [f64; 16], column-major.
/// This avoids pulling in nalgebra for the core (keeping it `no_std`-friendly).
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    /// Column-major 4x4 matrix
    pub m: [f64; 16],
}

impl Transform {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            m: [
                1.0, 0.0, 0.0, 0.0, // col 0
                0.0, 1.0, 0.0, 0.0, // col 1
                0.0, 0.0, 1.0, 0.0, // col 2
                0.0, 0.0, 0.0, 1.0, // col 3
            ],
        }
    }

    /// Get element at (row, col), 0-indexed.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.m[col * 4 + row]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.m[col * 4 + row] = val;
    }

    /// Extract translation (x, y, z) from the transform.
    pub fn translation(&self) -> [f64; 3] {
        [self.at(0, 3), self.at(1, 3), self.at(2, 3)]
    }

    /// Multiply two transforms: self * other
    pub fn mul(&self, other: &Transform) -> Transform {
        let mut result = Transform { m: [0.0; 16] };
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.at(row, k) * other.at(k, col);
                }
                result.set(row, col, sum);
            }
        }
        result
    }

    /// Create a rotation around Z axis by angle (radians).
    pub fn rot_z(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut t = Self::identity();
        t.set(0, 0, c);
        t.set(0, 1, -s);
        t.set(1, 0, s);
        t.set(1, 1, c);
        t
    }

    /// Create a rotation around Y axis by angle (radians).
    pub fn rot_y(angle: f64) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        let mut t = Self::identity();
        t.set(0, 0, c);
        t.set(0, 2, s);
        t.set(2, 0, -s);
        t.set(2, 2, c);
        t
    }

    /// Create a translation.
    pub fn translate(x: f64, y: f64, z: f64) -> Self {
        let mut t = Self::identity();
        t.set(0, 3, x);
        t.set(1, 3, y);
        t.set(2, 3, z);
        t
    }

    /// Transform a 3D point.
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        [
            self.at(0, 0) * p[0] + self.at(0, 1) * p[1] + self.at(0, 2) * p[2] + self.at(0, 3),
            self.at(1, 0) * p[0] + self.at(1, 1) * p[1] + self.at(1, 2) * p[2] + self.at(1, 3),
            self.at(2, 0) * p[0] + self.at(2, 1) * p[1] + self.at(2, 2) * p[2] + self.at(2, 3),
        ]
    }
}

/// A joint in the kinematic chain.
/// Uses a simplified parameterization (axis + offset) instead of full DH,
/// matching how most URDF/MuJoCo models define joints.
#[derive(Debug, Clone)]
pub struct KinematicJoint {
    /// Joint name.
    pub name: String,
    /// Rotation axis: "z" or "y" (covers all common serial arm joints).
    pub axis: JointAxis,
    /// Fixed translation from parent frame to this joint's origin (meters).
    pub origin_xyz: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
pub enum JointAxis {
    Z,
    Y,
}

/// A kinematic chain: sequence of joints that forms a serial robot.
#[derive(Debug, Clone)]
pub struct KinematicChain {
    pub joints: Vec<KinematicJoint>,
}

impl KinematicChain {
    pub fn new(joints: Vec<KinematicJoint>) -> Self {
        Self { joints }
    }

    /// Compute forward kinematics: given joint angles, return the transform
    /// of each joint frame relative to the base frame.
    ///
    /// Returns `joints.len() + 1` transforms:
    ///   [0] = base frame (identity)
    ///   [1] = after joint 0 rotation
    ///   ...
    ///   [n] = after joint n-1 rotation (end effector)
    pub fn forward_kinematics(&self, joint_angles: &[f64]) -> Vec<Transform> {
        assert_eq!(
            joint_angles.len(),
            self.joints.len(),
            "Joint angle count must match joint count"
        );

        let mut transforms = Vec::with_capacity(self.joints.len() + 1);
        let mut current = Transform::identity();
        transforms.push(current);

        for (i, joint) in self.joints.iter().enumerate() {
            // 1. Translate to joint origin
            let translation =
                Transform::translate(joint.origin_xyz[0], joint.origin_xyz[1], joint.origin_xyz[2]);

            // 2. Rotate by joint angle around the specified axis
            let rotation = match joint.axis {
                JointAxis::Z => Transform::rot_z(joint_angles[i]),
                JointAxis::Y => Transform::rot_y(joint_angles[i]),
            };

            current = current.mul(&translation).mul(&rotation);
            transforms.push(current);
        }

        transforms
    }

    /// Get the positions of each joint frame origin in world coordinates.
    /// Convenience function for collision checking and visualization.
    pub fn joint_positions(&self, joint_angles: &[f64]) -> Vec<[f64; 3]> {
        self.forward_kinematics(joint_angles)
            .iter()
            .map(|t| t.translation())
            .collect()
    }
}

/// Create a KinematicChain for the SO-ARM101.
/// Dimensions based on the MuJoCo model and known mechanical specs.
pub fn so_arm101_chain() -> KinematicChain {
    KinematicChain::new(vec![
        KinematicJoint {
            name: "base_rotation".to_string(),
            axis: JointAxis::Z,
            origin_xyz: [0.0, 0.0, 0.02], // base plate height
        },
        KinematicJoint {
            name: "shoulder".to_string(),
            axis: JointAxis::Y,
            origin_xyz: [0.0, 0.0, 0.02], // shoulder mount
        },
        KinematicJoint {
            name: "elbow".to_string(),
            axis: JointAxis::Y,
            origin_xyz: [0.0, 0.0, 0.095], // upper arm length
        },
        KinematicJoint {
            name: "wrist_flex".to_string(),
            axis: JointAxis::Y,
            origin_xyz: [0.0, 0.0, 0.095], // forearm length
        },
        KinematicJoint {
            name: "wrist_roll".to_string(),
            axis: JointAxis::Z,
            origin_xyz: [0.0, 0.0, 0.05], // wrist length
        },
        KinematicJoint {
            name: "gripper".to_string(),
            axis: JointAxis::Y,
            origin_xyz: [0.0, 0.0, 0.008], // gripper mount
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let t = Transform::identity();
        let p = t.transform_point([1.0, 2.0, 3.0]);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 2.0).abs() < 1e-10);
        assert!((p[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_translation() {
        let t = Transform::translate(1.0, 2.0, 3.0);
        let p = t.transform_point([0.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 2.0).abs() < 1e-10);
        assert!((p[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_z_90deg() {
        let t = Transform::rot_z(std::f64::consts::FRAC_PI_2);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        // Rotating (1,0,0) by 90° around Z → (0,1,0)
        assert!(p[0].abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
        assert!(p[2].abs() < 1e-10);
    }

    #[test]
    fn test_rot_y_90deg() {
        let t = Transform::rot_y(std::f64::consts::FRAC_PI_2);
        let p = t.transform_point([1.0, 0.0, 0.0]);
        // Rotating (1,0,0) by 90° around Y → (0,0,-1)
        assert!(p[0].abs() < 1e-10);
        assert!(p[1].abs() < 1e-10);
        assert!((p[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_so_arm101_zero_pose() {
        let chain = so_arm101_chain();
        let angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let positions = chain.joint_positions(&angles);

        // At zero pose, the arm should be pointing straight up.
        // Base at z=0.02, shoulder at z=0.04, elbow at z=0.135...
        let base = positions[0]; // identity origin
        assert!(base[2].abs() < 1e-10);

        // End effector should be on the Z axis (x≈0, y≈0) at zero pose
        let ee = positions[positions.len() - 1];
        assert!(ee[0].abs() < 0.01, "EE x should be ~0 at zero pose, got {}", ee[0]);
        assert!(ee[1].abs() < 0.01, "EE y should be ~0 at zero pose, got {}", ee[1]);
        // Total height ≈ 0.02 + 0.02 + 0.095 + 0.095 + 0.05 + 0.008 = 0.288
        assert!(
            (ee[2] - 0.288).abs() < 0.01,
            "EE z should be ~0.288 at zero pose, got {}",
            ee[2]
        );
    }

    #[test]
    fn test_so_arm101_shoulder_90deg() {
        let chain = so_arm101_chain();
        // Shoulder at 90° → arm extends horizontally forward
        let angles = [0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0];
        let positions = chain.joint_positions(&angles);

        let ee = positions[positions.len() - 1];
        // After shoulder 90°, the arm should extend along X axis
        // X should be roughly the total length after shoulder
        assert!(
            ee[0] > 0.1,
            "EE should extend forward (x>0.1) when shoulder=90°, got x={}",
            ee[0]
        );
    }
}
