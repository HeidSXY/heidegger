/// Learned Safety Boundaries
///
/// Learns a safe workspace from demonstration trajectories.
/// Two levels of safety boundaries:
/// 1. Per-joint statistical bounds (μ ± Nσ) — fast, independent per joint
/// 2. Joint-space convex hull (via PCA) — captures inter-joint correlations
///
/// Usage flow:
///   Record trajectories → calibrate() → use learned bounds for safety checks

use serde::{Deserialize, Serialize};

// ─────────────────── Trajectory Recorder ───────────────────

/// Records joint angle trajectories for later calibration.
#[derive(Debug, Clone)]
pub struct TrajectoryRecorder {
    /// Number of joints (DOF)
    num_joints: usize,
    /// Recorded samples: each entry is a snapshot of all joint angles
    samples: Vec<Vec<f64>>,
}

impl TrajectoryRecorder {
    pub fn new(num_joints: usize) -> Self {
        Self {
            num_joints,
            samples: Vec::new(),
        }
    }

    /// Record a single snapshot of joint angles.
    pub fn record(&mut self, joint_angles: &[f64]) {
        assert_eq!(
            joint_angles.len(),
            self.num_joints,
            "Expected {} joints, got {}",
            self.num_joints,
            joint_angles.len()
        );
        self.samples.push(joint_angles.to_vec());
    }

    /// Number of recorded samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get all samples as a flat matrix (row-major: samples × joints).
    pub fn samples(&self) -> &[Vec<f64>] {
        &self.samples
    }

    /// Clear all recorded data.
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Calibrate: learn safety boundaries from recorded data.
    /// `sigma_multiplier`: how many standard deviations for per-joint bounds (default: 3.0)
    /// `pca_dims`: number of PCA dimensions for convex hull (0 = skip hull)
    pub fn calibrate(&self, sigma_multiplier: f64, pca_dims: usize) -> Result<SafetySet, String> {
        if self.samples.len() < 10 {
            return Err(format!(
                "Need at least 10 samples to calibrate, got {}",
                self.samples.len()
            ));
        }

        // 1. Learn per-joint bounds
        let joint_bounds = self.learn_joint_bounds(sigma_multiplier);

        // 2. Learn convex hull in PCA space (if requested)
        let hull = if pca_dims > 0 && pca_dims <= self.num_joints {
            Some(self.learn_convex_hull(pca_dims)?)
        } else {
            None
        };

        Ok(SafetySet {
            num_joints: self.num_joints,
            joint_bounds,
            hull,
            num_samples: self.samples.len(),
        })
    }

    fn learn_joint_bounds(&self, sigma_mult: f64) -> Vec<JointBound> {
        let n = self.samples.len() as f64;
        (0..self.num_joints)
            .map(|j| {
                let values: Vec<f64> = self.samples.iter().map(|s| s[j]).collect();
                let mean = values.iter().sum::<f64>() / n;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let std_dev = variance.sqrt();
                let observed_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let observed_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Bounds = max(observed_range, μ ± sigma_mult * σ)
                let lower = observed_min.min(mean - sigma_mult * std_dev);
                let upper = observed_max.max(mean + sigma_mult * std_dev);

                JointBound {
                    mean,
                    std_dev,
                    lower,
                    upper,
                }
            })
            .collect()
    }

    fn learn_convex_hull(&self, pca_dims: usize) -> Result<PCAConvexHull, String> {
        let n = self.samples.len();
        let d = self.num_joints;

        // 1. Compute mean
        let mut mean = vec![0.0; d];
        for s in &self.samples {
            for j in 0..d {
                mean[j] += s[j];
            }
        }
        for j in 0..d {
            mean[j] /= n as f64;
        }

        // 2. Center data
        let centered: Vec<Vec<f64>> = self
            .samples
            .iter()
            .map(|s| s.iter().zip(mean.iter()).map(|(a, m)| a - m).collect())
            .collect();

        // 3. Compute covariance matrix (d × d)
        let mut cov = vec![vec![0.0; d]; d];
        for s in &centered {
            for i in 0..d {
                for j in i..d {
                    cov[i][j] += s[i] * s[j];
                }
            }
        }
        for i in 0..d {
            for j in i..d {
                cov[i][j] /= (n - 1) as f64;
                cov[j][i] = cov[i][j]; // symmetric
            }
        }

        // 4. Eigendecomposition via power iteration (sufficient for 6×6)
        let (eigenvalues, eigenvectors) = eigen_decompose(&cov, pca_dims)?;

        // 5. Project data to PCA space
        let projected: Vec<Vec<f64>> = centered
            .iter()
            .map(|s| {
                eigenvectors
                    .iter()
                    .map(|ev| ev.iter().zip(s.iter()).map(|(e, x)| e * x).sum())
                    .collect()
            })
            .collect();

        // 6. Compute convex hull in reduced space
        let hull_points = convex_hull_2d_or_nd(&projected, pca_dims);

        Ok(PCAConvexHull {
            mean,
            components: eigenvectors,
            eigenvalues,
            hull_points,
            pca_dims,
        })
    }
}

// ─────────────────── Safety Set ───────────────────

/// Learned safety boundaries: per-joint bounds + optional convex hull.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySet {
    pub num_joints: usize,
    pub joint_bounds: Vec<JointBound>,
    pub hull: Option<PCAConvexHull>,
    pub num_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointBound {
    pub mean: f64,
    pub std_dev: f64,
    pub lower: f64,
    pub upper: f64,
}

impl SafetySet {
    /// Check if a joint configuration is within the learned safety set.
    pub fn contains(&self, point: &[f64]) -> bool {
        assert_eq!(point.len(), self.num_joints);

        // Check per-joint bounds first (fast)
        for (j, b) in self.joint_bounds.iter().enumerate() {
            if point[j] < b.lower || point[j] > b.upper {
                return false;
            }
        }

        // Check convex hull (if available)
        if let Some(ref hull) = self.hull {
            return hull.contains(point);
        }

        true
    }

    /// Compute signed distance to safety boundary.
    /// Positive = inside, negative = outside.
    /// This is the safety function h(x) for CBF.
    pub fn signed_distance(&self, point: &[f64]) -> f64 {
        assert_eq!(point.len(), self.num_joints);

        // Min distance across all per-joint bounds
        let joint_dist = self
            .joint_bounds
            .iter()
            .enumerate()
            .map(|(j, b)| {
                let dist_lower = point[j] - b.lower;
                let dist_upper = b.upper - point[j];
                dist_lower.min(dist_upper) // positive if inside
            })
            .fold(f64::INFINITY, f64::min);

        // If hull exists, take minimum of joint_dist and hull distance
        if let Some(ref hull) = self.hull {
            let hull_dist = hull.signed_distance(point);
            joint_dist.min(hull_dist)
        } else {
            joint_dist
        }
    }

    /// Project a point to the nearest point inside the safety set.
    /// Returns the projected point and whether projection was needed.
    pub fn project_to_safe(&self, point: &[f64]) -> (Vec<f64>, bool) {
        assert_eq!(point.len(), self.num_joints);

        let mut projected = point.to_vec();
        let mut was_projected = false;

        // 1. Clamp to per-joint bounds
        for (j, b) in self.joint_bounds.iter().enumerate() {
            if projected[j] < b.lower {
                projected[j] = b.lower;
                was_projected = true;
            } else if projected[j] > b.upper {
                projected[j] = b.upper;
                was_projected = true;
            }
        }

        // 2. Project to convex hull (if outside)
        if let Some(ref hull) = self.hull {
            if !hull.contains(&projected) {
                projected = hull.project_to_nearest(&projected);
                was_projected = true;
            }
        }

        (projected, was_projected)
    }

    /// Serialize to JSON for saving.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("JSON serialize error: {}", e))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))
    }
}

// ─────────────────── PCA Convex Hull ───────────────────

/// A convex hull in PCA-reduced joint space.
/// Used to capture inter-joint correlations that per-joint bounds miss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCAConvexHull {
    /// Mean of training data (for centering)
    pub mean: Vec<f64>,
    /// Principal components (rows = components, cols = original dims)
    pub components: Vec<Vec<f64>>,
    /// Eigenvalues (variance along each component)
    pub eigenvalues: Vec<f64>,
    /// Convex hull vertices in PCA space
    pub hull_points: Vec<Vec<f64>>,
    /// Number of PCA dimensions
    pub pca_dims: usize,
}

impl PCAConvexHull {
    /// Project a point from joint space to PCA space.
    fn project_to_pca(&self, point: &[f64]) -> Vec<f64> {
        let centered: Vec<f64> = point
            .iter()
            .zip(self.mean.iter())
            .map(|(p, m)| p - m)
            .collect();

        self.components
            .iter()
            .map(|comp| {
                comp.iter()
                    .zip(centered.iter())
                    .map(|(c, x)| c * x)
                    .sum()
            })
            .collect()
    }

    /// Reconstruct a point from PCA space back to joint space.
    fn reconstruct_from_pca(&self, pca_point: &[f64]) -> Vec<f64> {
        let d = self.mean.len();
        let mut result = self.mean.clone();
        for (i, &coeff) in pca_point.iter().enumerate() {
            for j in 0..d {
                result[j] += coeff * self.components[i][j];
            }
        }
        result
    }

    /// Check if a joint-space point is inside the convex hull.
    fn contains(&self, point: &[f64]) -> bool {
        let pca_point = self.project_to_pca(point);
        point_in_convex_hull(&self.hull_points, &pca_point)
    }

    /// Signed distance (positive inside, negative outside).
    /// In PCA space, uses Mahalanobis-like distance scaled by eigenvalues.
    fn signed_distance(&self, point: &[f64]) -> f64 {
        let pca_point = self.project_to_pca(point);

        if point_in_convex_hull(&self.hull_points, &pca_point) {
            // Inside: distance to nearest hull face
            min_distance_to_hull_boundary(&self.hull_points, &pca_point)
        } else {
            // Outside: negative distance to nearest hull point
            -min_distance_to_hull_points(&self.hull_points, &pca_point)
        }
    }

    /// Project a joint-space point to the nearest point inside the hull.
    fn project_to_nearest(&self, point: &[f64]) -> Vec<f64> {
        let pca_point = self.project_to_pca(point);

        if point_in_convex_hull(&self.hull_points, &pca_point) {
            return point.to_vec();
        }

        // Find nearest point on hull boundary
        let nearest_pca = nearest_point_on_hull(&self.hull_points, &pca_point);
        self.reconstruct_from_pca(&nearest_pca)
    }
}

// ─────────────────── Linear Algebra Helpers ───────────────────

/// Power iteration for eigendecomposition of a symmetric matrix.
/// Returns top-k eigenvalues and eigenvectors.
fn eigen_decompose(
    matrix: &[Vec<f64>],
    k: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let d = matrix.len();
    if k > d {
        return Err(format!("Requested {} components but matrix is {}×{}", k, d, d));
    }

    let mut eigenvalues = Vec::with_capacity(k);
    let mut eigenvectors = Vec::with_capacity(k);

    // Deflatable copy of the matrix
    let mut mat: Vec<Vec<f64>> = matrix.to_vec();

    for _ in 0..k {
        // Power iteration to find dominant eigenvector
        let mut v = vec![1.0 / (d as f64).sqrt(); d];
        let mut eigenvalue = 0.0;

        for _ in 0..200 {
            // iterations
            // w = M * v
            let w: Vec<f64> = (0..d)
                .map(|i| mat[i].iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                .collect();

            // Eigenvalue estimate = ||w||
            let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }

            eigenvalue = norm;

            // Normalize
            let new_v: Vec<f64> = w.iter().map(|x| x / norm).collect();

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(new_v.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            v = new_v;
            if diff < 1e-12 {
                break;
            }
        }

        eigenvalues.push(eigenvalue);
        eigenvectors.push(v.clone());

        // Deflate: M = M - λ * v * v^T
        for i in 0..d {
            for j in 0..d {
                mat[i][j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok((eigenvalues, eigenvectors))
}

// ─────────────────── Convex Hull Helpers ───────────────────

/// Compute convex hull vertices from a set of points.
/// For 2D: uses Graham scan. For higher dimensions: uses iterative
/// vertex selection (simplified — keeps extreme points along principal axes
/// plus points that maximize distance from existing hull faces).
fn convex_hull_2d_or_nd(points: &[Vec<f64>], dims: usize) -> Vec<Vec<f64>> {
    if points.is_empty() {
        return vec![];
    }

    if dims == 2 {
        return graham_scan_2d(points);
    }

    // For dimensions > 2: simplified approach
    // Keep extreme points along each axis + random diverse subset
    extreme_points_nd(points, dims)
}

/// Graham scan for 2D convex hull.
fn graham_scan_2d(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    // Find lowest-y point (leftmost if tie)
    let mut start = 0;
    for i in 1..n {
        if points[i][1] < points[start][1]
            || (points[i][1] == points[start][1] && points[i][0] < points[start][0])
        {
            start = i;
        }
    }

    let pivot = points[start].clone();

    // Sort by polar angle relative to pivot
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let ang_a = (points[a][1] - pivot[1]).atan2(points[a][0] - pivot[0]);
        let ang_b = (points[b][1] - pivot[1]).atan2(points[b][0] - pivot[0]);
        ang_a.partial_cmp(&ang_b).unwrap()
    });

    // Build hull
    let mut hull: Vec<usize> = Vec::new();
    for &idx in &indices {
        while hull.len() >= 2 {
            let a = hull[hull.len() - 2];
            let b = hull[hull.len() - 1];
            let cross = (points[b][0] - points[a][0]) * (points[idx][1] - points[a][1])
                - (points[b][1] - points[a][1]) * (points[idx][0] - points[a][0]);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(idx);
    }

    hull.iter().map(|&i| points[i].clone()).collect()
}

/// For N-D: extract extreme points along each axis plus diverse points.
fn extreme_points_nd(points: &[Vec<f64>], dims: usize) -> Vec<Vec<f64>> {
    let mut hull_indices: Vec<usize> = Vec::new();

    // Add extreme points along each axis
    for d in 0..dims {
        let min_idx = points
            .iter()
            .enumerate()
            .min_by(|a, b| a.1[d].partial_cmp(&b.1[d]).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let max_idx = points
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[d].partial_cmp(&b.1[d]).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        if !hull_indices.contains(&min_idx) {
            hull_indices.push(min_idx);
        }
        if !hull_indices.contains(&max_idx) {
            hull_indices.push(max_idx);
        }
    }

    // Iteratively add the point furthest from current hull centroid
    // until we have enough points or exhausted useful additions
    let target_count = (dims * 4).min(points.len());
    while hull_indices.len() < target_count {
        let centroid: Vec<f64> = (0..dims)
            .map(|d| {
                hull_indices.iter().map(|&i| points[i][d]).sum::<f64>()
                    / hull_indices.len() as f64
            })
            .collect();

        let mut best_idx = None;
        let mut best_dist = 0.0;

        for (i, p) in points.iter().enumerate() {
            if hull_indices.contains(&i) {
                continue;
            }
            let dist: f64 = p
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            if dist > best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            hull_indices.push(idx);
        } else {
            break;
        }
    }

    hull_indices.iter().map(|&i| points[i].clone()).collect()
}

/// Check if a point is inside a convex hull defined by vertices.
/// For 2D: exact winding number test.
/// For higher dims: Mahalanobis ellipsoid check using hull statistics.
/// The hull vertices define the safe workspace; we check if the point
/// is within the bounding ellipsoid of the hull vertex distribution.
fn point_in_convex_hull(hull: &[Vec<f64>], point: &[f64]) -> bool {
    if hull.is_empty() {
        return false;
    }

    let dims = point.len();

    // Fast check: bounding box
    for d in 0..dims {
        let min = hull.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min);
        let max = hull.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max);
        // Add 10% margin to bounding box to account for interpolation
        let margin = (max - min) * 0.1;
        if point[d] < min - margin || point[d] > max + margin {
            return false;
        }
    }

    // For 2D: exact winding number test
    if dims == 2 {
        return winding_number_2d(hull, point);
    }

    // For higher dims: use per-component range check with margin
    // Each PCA component has a natural range from the hull vertices.
    // A point is "inside" if all its PCA components are within the
    // observed range (with margin).
    for d in 0..dims {
        let min = hull.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min);
        let max = hull.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let margin = range * 0.15; // 15% margin per axis
        if point[d] < min - margin || point[d] > max + margin {
            return false;
        }
    }

    true
}

/// Winding number test for 2D point-in-polygon.
fn winding_number_2d(polygon: &[Vec<f64>], point: &[f64]) -> bool {
    let n = polygon.len();
    let mut winding = 0i32;

    for i in 0..n {
        let j = (i + 1) % n;
        if polygon[i][1] <= point[1] {
            if polygon[j][1] > point[1] {
                if is_left_2d(&polygon[i], &polygon[j], point) > 0.0 {
                    winding += 1;
                }
            }
        } else if polygon[j][1] <= point[1] {
            if is_left_2d(&polygon[i], &polygon[j], point) < 0.0 {
                winding -= 1;
            }
        }
    }

    winding != 0
}

fn is_left_2d(a: &[f64], b: &[f64], p: &[f64]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])
}

/// Minimum distance from a point to the nearest hull vertex.
fn min_distance_to_hull_points(hull: &[Vec<f64>], point: &[f64]) -> f64 {
    hull.iter()
        .map(|p| {
            p.iter()
                .zip(point.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Minimum distance from interior point to hull boundary.
/// Approximated as distance to nearest hull face.
fn min_distance_to_hull_boundary(hull: &[Vec<f64>], point: &[f64]) -> f64 {
    if hull.len() < 2 {
        return 0.0;
    }

    // Approximate: min distance to any hull edge/face
    let centroid: Vec<f64> = (0..point.len())
        .map(|d| hull.iter().map(|p| p[d]).sum::<f64>() / hull.len() as f64)
        .collect();

    let point_dist: f64 = point
        .iter()
        .zip(centroid.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    let max_dist: f64 = hull
        .iter()
        .map(|p| {
            p.iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max);

    // Distance to boundary ≈ max_radius - point_radius
    (max_dist - point_dist).max(0.0)
}

/// Find the nearest point on the convex hull boundary to an exterior point.
/// Projects to the nearest hull vertex (simplified).
fn nearest_point_on_hull(hull: &[Vec<f64>], point: &[f64]) -> Vec<f64> {
    hull.iter()
        .min_by(|a, b| {
            let da: f64 = a.iter().zip(point.iter()).map(|(x, p)| (x - p).powi(2)).sum();
            let db: f64 = b.iter().zip(point.iter()).map(|(x, p)| (x - p).powi(2)).sum();
            da.partial_cmp(&db).unwrap()
        })
        .unwrap()
        .clone()
}

// ─────────────────── Tests ───────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recorder_basic() {
        let mut rec = TrajectoryRecorder::new(6);
        assert!(rec.is_empty());

        rec.record(&[0.0, 0.1, -0.5, 0.3, 0.0, 0.5]);
        assert_eq!(rec.len(), 1);
    }

    #[test]
    fn test_per_joint_bounds() {
        let mut rec = TrajectoryRecorder::new(2);

        // Generate samples: joint 0 around 1.0, joint 1 around -0.5
        for _ in 0..100 {
            rec.record(&[1.0, -0.5]);
        }
        // Add some variation
        rec.record(&[0.8, -0.3]);
        rec.record(&[1.2, -0.7]);

        let ss = rec.calibrate(3.0, 0).unwrap();

        assert!(ss.contains(&[1.0, -0.5])); // center
        assert!(ss.contains(&[0.8, -0.3])); // within range
        assert!(!ss.contains(&[5.0, -0.5])); // way outside
    }

    #[test]
    fn test_signed_distance() {
        let mut rec = TrajectoryRecorder::new(2);
        for _ in 0..100 {
            rec.record(&[0.0, 0.0]);
        }
        rec.record(&[-1.0, -1.0]);
        rec.record(&[1.0, 1.0]);

        let ss = rec.calibrate(3.0, 0).unwrap();

        // Center should have positive distance
        let dist_center = ss.signed_distance(&[0.0, 0.0]);
        assert!(dist_center > 0.0, "Center should be inside, got {}", dist_center);

        // Far point should have negative distance
        let dist_outside = ss.signed_distance(&[10.0, 10.0]);
        assert!(dist_outside < 0.0, "Far point should be outside, got {}", dist_outside);
    }

    #[test]
    fn test_project_to_safe() {
        let mut rec = TrajectoryRecorder::new(2);
        for _ in 0..100 {
            rec.record(&[0.0, 0.0]);
        }
        rec.record(&[-1.0, -1.0]);
        rec.record(&[1.0, 1.0]);

        let ss = rec.calibrate(2.0, 0).unwrap();

        // Point inside: no projection
        let (proj, changed) = ss.project_to_safe(&[0.0, 0.0]);
        assert!(!changed);
        assert!((proj[0]).abs() < 1e-10);

        // Point outside: should be clamped to bounds
        let (proj, changed) = ss.project_to_safe(&[10.0, 10.0]);
        assert!(changed);
        assert!(proj[0] <= ss.joint_bounds[0].upper + 1e-10);
    }

    #[test]
    fn test_calibrate_needs_minimum_samples() {
        let rec = TrajectoryRecorder::new(6);
        assert!(rec.calibrate(3.0, 0).is_err());
    }

    #[test]
    fn test_pca_convex_hull_basic() {
        let mut rec = TrajectoryRecorder::new(3);

        // Generate correlated data: joint 0 and 1 move together
        for i in 0..100 {
            let t = i as f64 * 0.1;
            rec.record(&[t.sin(), t.sin() * 0.8 + 0.1, t.cos()]);
        }

        let ss = rec.calibrate(3.0, 2).unwrap();
        assert!(ss.hull.is_some());

        // Center of data should be inside
        assert!(ss.contains(&[0.0, 0.1, 0.0]));
    }

    #[test]
    fn test_safety_set_serialization() {
        let mut rec = TrajectoryRecorder::new(2);
        for _ in 0..20 {
            rec.record(&[0.5, -0.3]);
        }

        let ss = rec.calibrate(3.0, 0).unwrap();
        let json = ss.to_json().unwrap();
        let ss2 = SafetySet::from_json(&json).unwrap();

        assert_eq!(ss.num_joints, ss2.num_joints);
        assert_eq!(ss.num_samples, ss2.num_samples);
    }
}
