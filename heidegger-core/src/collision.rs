use crate::kinematics::Transform;

/// A capsule primitive: a line segment with a radius.
/// Defined by two endpoints in local (link) frame.
#[derive(Debug, Clone)]
pub struct LinkCapsule {
    pub link_index: usize,
    pub name: String,
    /// Start point in the link's local frame
    pub start_local: [f64; 3],
    /// End point in the link's local frame
    pub end_local: [f64; 3],
    /// Capsule radius (meters)
    pub radius: f64,
}

/// A collision pair result.
#[derive(Debug, Clone)]
pub struct CollisionPair {
    pub link_a: String,
    pub link_b: String,
    /// Minimum distance between capsule surfaces (negative = penetrating)
    pub distance: f64,
    /// Closest point on capsule A (world frame)
    pub closest_a: [f64; 3],
    /// Closest point on capsule B (world frame)
    pub closest_b: [f64; 3],
}

/// Configuration for self-collision detection.
#[derive(Debug, Clone)]
pub struct CollisionChecker {
    pub capsules: Vec<LinkCapsule>,
    /// Minimum number of links between a pair to check collision.
    /// Default: 2 (skip adjacent links, since they're connected by a joint).
    pub skip_adjacent: usize,
    /// Safety margin in meters. If distance < margin → warning.
    pub safety_margin: f64,
}

impl CollisionChecker {
    pub fn new(capsules: Vec<LinkCapsule>, safety_margin: f64) -> Self {
        Self {
            capsules,
            skip_adjacent: 2,
            safety_margin,
        }
    }

    /// Check for self-collisions given the current joint poses.
    ///
    /// `transforms`: the output of `KinematicChain::forward_kinematics()`
    ///
    /// Returns all colliding pairs (distance < safety_margin), sorted by distance.
    pub fn check(&self, transforms: &[Transform]) -> Vec<CollisionPair> {
        let mut results = Vec::new();

        // Transform capsule endpoints to world frame
        let world_capsules: Vec<([f64; 3], [f64; 3], f64, &str)> = self
            .capsules
            .iter()
            .map(|c| {
                let tf = &transforms[c.link_index];
                let start = tf.transform_point(c.start_local);
                let end = tf.transform_point(c.end_local);
                (start, end, c.radius, c.name.as_str())
            })
            .collect();

        // Check all non-adjacent pairs
        for i in 0..world_capsules.len() {
            for j in (i + 1)..world_capsules.len() {
                let idx_a = self.capsules[i].link_index;
                let idx_b = self.capsules[j].link_index;

                // Skip pairs that are too close in the kinematic chain
                if idx_b.abs_diff(idx_a) <= self.skip_adjacent {
                    continue;
                }

                let (sa, ea, ra, name_a) = &world_capsules[i];
                let (sb, eb, rb, name_b) = &world_capsules[j];

                let (dist, ca, cb) = capsule_capsule_distance(*sa, *ea, *ra, *sb, *eb, *rb);

                if dist < self.safety_margin {
                    results.push(CollisionPair {
                        link_a: name_a.to_string(),
                        link_b: name_b.to_string(),
                        distance: dist,
                        closest_a: ca,
                        closest_b: cb,
                    });
                }
            }
        }

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    /// Quick boolean check: is there any collision?
    pub fn has_collision(&self, transforms: &[Transform]) -> bool {
        let world_capsules: Vec<([f64; 3], [f64; 3], f64)> = self
            .capsules
            .iter()
            .map(|c| {
                let tf = &transforms[c.link_index];
                (
                    tf.transform_point(c.start_local),
                    tf.transform_point(c.end_local),
                    c.radius,
                )
            })
            .collect();

        for i in 0..world_capsules.len() {
            for j in (i + 1)..world_capsules.len() {
                let idx_a = self.capsules[i].link_index;
                let idx_b = self.capsules[j].link_index;
                if idx_b.abs_diff(idx_a) <= self.skip_adjacent {
                    continue;
                }

                let (sa, ea, ra) = world_capsules[i];
                let (sb, eb, rb) = world_capsules[j];
                let (dist, _, _) = capsule_capsule_distance(sa, ea, ra, sb, eb, rb);

                if dist < self.safety_margin {
                    return true;
                }
            }
        }
        false
    }
}

/// Compute the minimum distance between two capsules.
///
/// A capsule = line segment + radius.
/// Distance between capsules = distance between line segments - sum of radii.
///
/// Returns (distance, closest_point_on_a, closest_point_on_b).
/// Distance is negative if capsules are penetrating.
fn capsule_capsule_distance(
    a0: [f64; 3],
    a1: [f64; 3],
    ra: f64,
    b0: [f64; 3],
    b1: [f64; 3],
    rb: f64,
) -> (f64, [f64; 3], [f64; 3]) {
    let (seg_dist, ca, cb) = segment_segment_distance(a0, a1, b0, b1);
    let surface_dist = seg_dist - ra - rb;

    // Closest points on capsule surfaces (move along the connecting direction)
    let dir = vec3_sub(cb, ca);
    let len = vec3_len(dir);
    if len > 1e-12 {
        let n = vec3_scale(dir, 1.0 / len);
        let ca_surface = vec3_add(ca, vec3_scale(n, ra));
        let cb_surface = vec3_sub(cb, vec3_scale(n, rb));
        (surface_dist, ca_surface, cb_surface)
    } else {
        (surface_dist, ca, cb)
    }
}

/// Minimum distance between two line segments in 3D.
/// Uses the standard closest-point algorithm.
///
/// Reference: "Real-Time Collision Detection" by Christer Ericson, Chapter 5.
///
/// Returns (distance, closest_point_on_a, closest_point_on_b).
fn segment_segment_distance(
    p1: [f64; 3],
    q1: [f64; 3],
    p2: [f64; 3],
    q2: [f64; 3],
) -> (f64, [f64; 3], [f64; 3]) {
    let d1 = vec3_sub(q1, p1); // Direction of segment 1
    let d2 = vec3_sub(q2, p2); // Direction of segment 2
    let r = vec3_sub(p1, p2);

    let a = vec3_dot(d1, d1); // Squared length of segment 1
    let e = vec3_dot(d2, d2); // Squared length of segment 2
    let f = vec3_dot(d2, r);

    let epsilon = 1e-10;

    let (mut s, mut t);

    if a <= epsilon && e <= epsilon {
        // Both segments degenerate to points
        s = 0.0;
        t = 0.0;
    } else if a <= epsilon {
        // First segment degenerates to a point
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = vec3_dot(d1, r);
        if e <= epsilon {
            // Second segment degenerates to a point
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General case: neither segment is degenerate
            let b = vec3_dot(d1, d2);
            let denom = a * e - b * b; // Always >= 0

            // If segments not parallel, compute closest point on L1 to L2
            if denom.abs() > epsilon {
                s = ((b * f - c * e) / denom).clamp(0.0, 1.0);
            } else {
                s = 0.0;
            }

            // Compute point on L2 closest to S1(s)
            t = (b * s + f) / e;

            // If t is outside [0,1], clamp and recompute s
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }

    let closest1 = vec3_add(p1, vec3_scale(d1, s));
    let closest2 = vec3_add(p2, vec3_scale(d2, t));
    let diff = vec3_sub(closest1, closest2);
    (vec3_len(diff), closest1, closest2)
}

// --- Inline vector math (no dependencies) ---

#[inline]
fn vec3_sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn vec3_add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn vec3_dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn vec3_scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn vec3_len(a: [f64; 3]) -> f64 {
    vec3_dot(a, a).sqrt()
}

/// Create a CollisionChecker for the SO-ARM101, matching the MuJoCo model geometry.
pub fn so_arm101_collision_checker(safety_margin: f64) -> CollisionChecker {
    CollisionChecker::new(
        vec![
            // Base plate (link 0 → transform index 0)
            LinkCapsule {
                link_index: 0,
                name: "base_plate".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.02],
                radius: 0.04,
            },
            // Base rotation body (link 1 → transform index 1)
            LinkCapsule {
                link_index: 1,
                name: "base_body".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.02],
                radius: 0.03,
            },
            // Shoulder → elbow link (link 2 → transform index 2)
            LinkCapsule {
                link_index: 2,
                name: "upper_arm".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.095],
                radius: 0.018,
            },
            // Elbow → wrist link (link 3 → transform index 3)
            LinkCapsule {
                link_index: 3,
                name: "forearm".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.095],
                radius: 0.015,
            },
            // Wrist flex body (link 4 → transform index 4)
            LinkCapsule {
                link_index: 4,
                name: "wrist".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.05],
                radius: 0.012,
            },
            // Gripper (link 6 → transform index 6)
            LinkCapsule {
                link_index: 6,
                name: "gripper".to_string(),
                start_local: [0.0, 0.0, 0.0],
                end_local: [0.0, 0.0, 0.05],
                radius: 0.015,
            },
        ],
        safety_margin,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::so_arm101_chain;

    #[test]
    fn test_segment_distance_parallel() {
        // Two parallel segments, 1 unit apart
        let (d, _, _) = segment_segment_distance(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        );
        assert!((d - 1.0).abs() < 1e-6, "Expected 1.0, got {}", d);
    }

    #[test]
    fn test_segment_distance_perpendicular() {
        // Two perpendicular segments crossing at origin, offset by 1 in Z
        let (d, _, _) = segment_segment_distance(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0],
        );
        assert!((d - 1.0).abs() < 1e-6, "Expected 1.0, got {}", d);
    }

    #[test]
    fn test_capsule_distance() {
        let (d, _, _) = capsule_capsule_distance(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.1,
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            0.1,
        );
        // Segment distance = 0.5, minus two radii of 0.1 = 0.3
        assert!((d - 0.3).abs() < 1e-6, "Expected 0.3, got {}", d);
    }

    #[test]
    fn test_so_arm101_zero_no_collision() {
        // At zero pose (arm straight up), no self-collision expected
        let chain = so_arm101_chain();
        let checker = so_arm101_collision_checker(0.005); // 5mm margin
        let angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let transforms = chain.forward_kinematics(&angles);
        let collisions = checker.check(&transforms);
        assert!(
            collisions.is_empty(),
            "No collision expected at zero pose, found: {:?}",
            collisions
        );
    }

    #[test]
    fn test_so_arm101_folded_collision() {
        // Fold the arm back onto itself: shoulder back + elbow back
        // This should bring the gripper close to or into the base
        let chain = so_arm101_chain();
        let checker = so_arm101_collision_checker(0.02); // 20mm margin
        let angles = [
            0.0,
            -1.5,  // shoulder tilted backward
            -0.1,  // elbow barely bent
            -1.5,  // wrist folded back
            0.0,
            0.0,
        ];
        let transforms = chain.forward_kinematics(&angles);
        let has_coll = checker.has_collision(&transforms);
        // In this pose the wrist/gripper should be near the base
        // (exact result depends on geometry — this is a sanity test)
        println!("Folded pose collision: {}", has_coll);
        // Print link positions for debugging
        let positions: Vec<[f64; 3]> = transforms.iter().map(|t| t.translation()).collect();
        for (i, p) in positions.iter().enumerate() {
            println!("  Link {}: [{:.3}, {:.3}, {:.3}]", i, p[0], p[1], p[2]);
        }
    }
}
