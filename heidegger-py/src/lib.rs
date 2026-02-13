use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};
use heidegger_core::{JointLimit, SafetyShim as RustSafetyShim};
use heidegger_core::kinematics::{self, KinematicChain};
use heidegger_core::collision::{self, CollisionChecker};
use heidegger_core::safety_set::{
    TrajectoryRecorder as RustRecorder,
    SafetySet as RustSafetySet,
};
use heidegger_core::cbf::{
    CBFSafetyFilter as RustCBFFilter,
    CBFConfig,
};

/// Python-facing SafetyShim class — position/velocity clamping.
#[pyclass]
struct SafetyShim {
    inner: RustSafetyShim,
    /// Operation mode: "guardian" (clamp + reject) or "shadow" (observe only)
    mode: String,
}

#[pymethods]
impl SafetyShim {
    /// Create a SafetyShim from a JSON config string.
    ///
    /// Args:
    ///     config_json: JSON string with joint limits array.
    ///     dt: Control loop period in seconds.
    ///     mode: "guardian" (default, clamps actions) or "shadow" (observe only).
    #[new]
    #[pyo3(signature = (config_json, dt, mode = "guardian"))]
    fn new(config_json: &str, dt: f64, mode: &str) -> PyResult<Self> {
        let inner = RustSafetyShim::from_json(config_json, dt)
            .map_err(|e| PyValueError::new_err(e))?;
        let mode = match mode {
            "guardian" | "shadow" => mode.to_string(),
            _ => return Err(PyValueError::new_err(
                format!("Invalid mode '{}'. Use 'guardian' or 'shadow'.", mode)
            )),
        };
        Ok(SafetyShim { inner, mode })
    }

    /// Create a SafetyShim with simple joint limits (no JSON needed).
    #[staticmethod]
    #[pyo3(signature = (names, lowers, uppers, max_velocities, dt, mode = "guardian"))]
    fn from_limits(
        names: Vec<String>,
        lowers: Vec<f64>,
        uppers: Vec<f64>,
        max_velocities: Vec<f64>,
        dt: f64,
        mode: &str,
    ) -> PyResult<Self> {
        if names.len() != lowers.len()
            || names.len() != uppers.len()
            || names.len() != max_velocities.len()
        {
            return Err(PyValueError::new_err("All lists must have same length"));
        }

        let joints: Vec<JointLimit> = names
            .into_iter()
            .zip(lowers.into_iter())
            .zip(uppers.into_iter())
            .zip(max_velocities.into_iter())
            .map(|(((name, lower), upper), max_velocity)| JointLimit {
                name,
                lower,
                upper,
                max_velocity,
                max_effort: 0.0,
            })
            .collect();

        let mode = match mode {
            "guardian" | "shadow" => mode.to_string(),
            _ => return Err(PyValueError::new_err(
                format!("Invalid mode '{}'. Use 'guardian' or 'shadow'.", mode)
            )),
        };

        Ok(SafetyShim {
            inner: RustSafetyShim::new(joints, dt),
            mode,
        })
    }

    /// Check a raw action for safety and return the clamped version.
    ///
    /// In "guardian" mode: returns the clamped safe action.
    /// In "shadow" mode: returns the original action unchanged, but reports
    /// what *would* have been clamped via the `would_clamp` field.
    fn check<'py>(
        &self,
        py: Python<'py>,
        raw_action: Vec<f64>,
        current_positions: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .check(&raw_action, &current_positions)
            .map_err(|e| PyValueError::new_err(e))?;

        let dict = PyDict::new(py);

        let is_shadow = self.mode == "shadow";

        if is_shadow {
            // Shadow mode: return original action, note what would have changed
            dict.set_item("safe_action", &raw_action)?;
            dict.set_item("was_clamped", false)?;
            dict.set_item("would_clamp", result.was_clamped)?;
        } else {
            // Guardian mode: return clamped action
            dict.set_item("safe_action", &result.safe_action)?;
            dict.set_item("was_clamped", result.was_clamped)?;
            dict.set_item("would_clamp", result.was_clamped)?;
        }

        dict.set_item("mode", &self.mode)?;

        let violations: Vec<Bound<'py, PyDict>> = result
            .details
            .iter()
            .filter(|d| d.reason.is_some())
            .map(|d| {
                let v = PyDict::new(py);
                v.set_item("joint", &d.joint_name).unwrap();
                v.set_item("original", d.original_value).unwrap();
                v.set_item("clamped", d.clamped_value).unwrap();
                v.set_item("reason", d.reason.as_ref().unwrap().to_string())
                    .unwrap();
                v
            })
            .collect();

        dict.set_item("violations", violations)?;
        Ok(dict)
    }

    /// Get the number of joints.
    #[getter]
    fn num_joints(&self) -> usize {
        self.inner.joints.len()
    }

    /// Get joint names.
    #[getter]
    fn joint_names(&self) -> Vec<String> {
        self.inner.joints.iter().map(|j| j.name.clone()).collect()
    }

    /// Get or set the operation mode.
    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }

    #[setter]
    fn set_mode(&mut self, mode: &str) -> PyResult<()> {
        match mode {
            "guardian" | "shadow" => {
                self.mode = mode.to_string();
                Ok(())
            }
            _ => Err(PyValueError::new_err(
                format!("Invalid mode '{}'. Use 'guardian' or 'shadow'.", mode)
            )),
        }
    }
}

/// Python-facing CollisionGuard — forward kinematics + self-collision detection.
#[pyclass]
struct CollisionGuard {
    chain: KinematicChain,
    checker: CollisionChecker,
    model_name: String,
}

#[pymethods]
impl CollisionGuard {
    /// Create a CollisionGuard for a given robot model.
    ///
    /// Args:
    ///     safety_margin: Minimum distance between links (meters) before
    ///                    a collision is reported.
    ///     robot_model: Robot model name ("so_arm101", "so_arm100", "koch_v1_1").
    ///                  Defaults to "so_arm101" for backwards compatibility.
    #[new]
    #[pyo3(signature = (safety_margin, robot_model = "so_arm101"))]
    fn new(safety_margin: f64, robot_model: &str) -> PyResult<Self> {
        let chain = kinematics::chain_by_name(robot_model)
            .ok_or_else(|| PyValueError::new_err(
                format!("Unknown robot model '{}'. Available: so_arm101, so_arm100, koch_v1_1", robot_model)
            ))?;

        let checker = collision::collision_checker_by_name(robot_model, safety_margin)
            .ok_or_else(|| PyValueError::new_err(
                format!("No collision checker for '{}'. Available: so_arm101, so_arm100, koch_v1_1", robot_model)
            ))?;

        Ok(CollisionGuard {
            chain,
            checker,
            model_name: robot_model.to_string(),
        })
    }

    /// Compute forward kinematics: returns list of [x, y, z] positions
    /// for each joint frame (including base and end-effector).
    fn forward_kinematics<'py>(
        &self,
        py: Python<'py>,
        joint_angles: Vec<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let positions = self.chain.joint_positions(&joint_angles);
        let py_positions: Vec<Vec<f64>> = positions
            .iter()
            .map(|p| vec![p[0], p[1], p[2]])
            .collect();
        Ok(PyList::new(py, py_positions).unwrap())
    }

    /// Check for self-collisions. Returns True if any collision is detected.
    fn has_collision(&self, joint_angles: Vec<f64>) -> bool {
        let transforms = self.chain.forward_kinematics(&joint_angles);
        self.checker.has_collision(&transforms)
    }

    /// Detailed collision check. Returns a list of collision pair dicts.
    fn check_collisions<'py>(
        &self,
        py: Python<'py>,
        joint_angles: Vec<f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let transforms = self.chain.forward_kinematics(&joint_angles);
        let collisions = self.checker.check(&transforms);

        let results: Vec<Bound<'py, PyDict>> = collisions
            .iter()
            .map(|c| {
                let d = PyDict::new(py);
                d.set_item("link_a", &c.link_a).unwrap();
                d.set_item("link_b", &c.link_b).unwrap();
                d.set_item("distance", c.distance).unwrap();
                d.set_item("closest_a", c.closest_a.to_vec()).unwrap();
                d.set_item("closest_b", c.closest_b.to_vec()).unwrap();
                d
            })
            .collect();

        Ok(PyList::new(py, results).unwrap())
    }

    /// Get joint frame positions (for visualization).
    fn joint_positions(&self, joint_angles: Vec<f64>) -> Vec<Vec<f64>> {
        self.chain
            .joint_positions(&joint_angles)
            .iter()
            .map(|p| vec![p[0], p[1], p[2]])
            .collect()
    }

    /// Get the robot model name.
    #[getter]
    fn robot_model(&self) -> &str {
        &self.model_name
    }
}

// ─────────────────── Trajectory Recorder ───────────────────

/// Python-facing TrajectoryRecorder — records joint trajectories for calibration.
#[pyclass]
struct TrajectoryRecorder {
    inner: RustRecorder,
}

#[pymethods]
impl TrajectoryRecorder {
    /// Create a new TrajectoryRecorder.
    ///
    /// Args:
    ///     num_joints: Number of joints (DOF) of the robot.
    #[new]
    fn new(num_joints: usize) -> Self {
        Self {
            inner: RustRecorder::new(num_joints),
        }
    }

    /// Record a single snapshot of joint angles.
    fn record(&mut self, joint_angles: Vec<f64>) {
        self.inner.record(&joint_angles);
    }

    /// Number of recorded samples.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.len()
    }

    /// Clear all recorded data.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Calibrate: learn safety boundaries from recorded data.
    ///
    /// Args:
    ///     sigma_multiplier: Number of std devs for per-joint bounds (default: 3.0).
    ///     pca_dims: PCA dimensions for convex hull (0 = skip hull, default: 3).
    ///
    /// Returns:
    ///     A PySafetySet that can be used with CBFSafetyFilter.
    #[pyo3(signature = (sigma_multiplier = 3.0, pca_dims = 3))]
    fn calibrate(&self, sigma_multiplier: f64, pca_dims: usize) -> PyResult<PySafetySet> {
        let ss = self
            .inner
            .calibrate(sigma_multiplier, pca_dims)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PySafetySet { inner: ss })
    }
}

// ─────────────────── Safety Set ───────────────────

/// Python-facing SafetySet — learned safety boundaries.
#[pyclass]
struct PySafetySet {
    inner: RustSafetySet,
}

#[pymethods]
impl PySafetySet {
    /// Check if a joint configuration is within the learned safety set.
    fn contains(&self, point: Vec<f64>) -> bool {
        self.inner.contains(&point)
    }

    /// Compute signed distance to safety boundary.
    /// Positive = inside, negative = outside.
    fn signed_distance(&self, point: Vec<f64>) -> f64 {
        self.inner.signed_distance(&point)
    }

    /// Project a point to the nearest point inside the safety set.
    /// Returns (projected_point, was_projected).
    fn project_to_safe(&self, point: Vec<f64>) -> (Vec<f64>, bool) {
        self.inner.project_to_safe(&point)
    }

    /// Number of joints.
    #[getter]
    fn num_joints(&self) -> usize {
        self.inner.num_joints
    }

    /// Number of training samples used.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples
    }

    /// Per-joint bounds as list of dicts.
    fn joint_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let bounds: Vec<Bound<'py, PyDict>> = self
            .inner
            .joint_bounds
            .iter()
            .map(|b| {
                let d = PyDict::new(py);
                d.set_item("mean", b.mean).unwrap();
                d.set_item("std_dev", b.std_dev).unwrap();
                d.set_item("lower", b.lower).unwrap();
                d.set_item("upper", b.upper).unwrap();
                d
            })
            .collect();
        Ok(PyList::new(py, bounds).unwrap())
    }

    /// Serialize to JSON.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json().map_err(|e| PyValueError::new_err(e))
    }

    /// Load from JSON.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = RustSafetySet::from_json(json)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(PySafetySet { inner })
    }
}

// ─────────────────── CBF Safety Filter ───────────────────

/// Python-facing CBFSafetyFilter — globally optimal safe action via QP.
///
/// Replaces independent per-joint clamping with a simultaneous constraint
/// optimization that finds the closest safe action to the VLA policy output.
#[pyclass]
struct PyCBFSafetyFilter {
    inner: RustCBFFilter,
}

#[pymethods]
impl PyCBFSafetyFilter {
    /// Create a CBF safety filter.
    ///
    /// Args:
    ///     config_json: JSON joint limits (same format as SafetyShim).
    ///     dt: Control loop period (seconds).
    ///     collision_margin: Self-collision safety margin (meters, default: 0.015).
    ///     alpha: CBF decay rate (0,1], lower = more conservative (default: 0.3).
    ///     robot_model: Optional robot model for collision constraints.
    ///     safety_set_json: Optional JSON of a learned SafetySet.
    #[new]
    #[pyo3(signature = (
        config_json,
        dt = 0.02,
        collision_margin = 0.015,
        alpha = 0.3,
        robot_model = None,
        safety_set_json = None
    ))]
    fn new(
        config_json: &str,
        dt: f64,
        collision_margin: f64,
        alpha: f64,
        robot_model: Option<&str>,
        safety_set_json: Option<&str>,
    ) -> PyResult<Self> {
        let config = CBFConfig {
            dt,
            collision_margin,
            alpha,
            ..CBFConfig::default()
        };

        let mut filter = RustCBFFilter::from_json(config_json, config)
            .map_err(|e| PyValueError::new_err(e))?;

        // Enable collision constraint if robot model specified
        if let Some(model) = robot_model {
            filter = filter
                .with_collision(model)
                .map_err(|e| PyValueError::new_err(e))?;
        }

        // Enable learned workspace constraint if safety set provided
        if let Some(ss_json) = safety_set_json {
            let ss = RustSafetySet::from_json(ss_json)
                .map_err(|e| PyValueError::new_err(e))?;
            filter = filter.with_safety_set(ss);
        }

        Ok(PyCBFSafetyFilter { inner: filter })
    }

    /// Run the CBF safety filter.
    ///
    /// Args:
    ///     u_vla: Desired action from VLA policy.
    ///     x_current: Current joint positions.
    ///
    /// Returns:
    ///     Dict with: safe_action, was_modified, modification_norm,
    ///     active_constraints, constraint_values, iterations.
    fn filter<'py>(
        &self,
        py: Python<'py>,
        u_vla: Vec<f64>,
        x_current: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self
            .inner
            .filter(&u_vla, &x_current)
            .map_err(|e| PyValueError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("safe_action", &result.safe_action)?;
        dict.set_item("was_modified", result.was_modified)?;
        dict.set_item("modification_norm", result.modification_norm)?;
        dict.set_item("active_constraints", result.active_constraints)?;
        dict.set_item("iterations", result.iterations)?;

        let constraints: Vec<Bound<'py, PyDict>> = result
            .constraint_values
            .iter()
            .map(|cv| {
                let d = PyDict::new(py);
                d.set_item("name", &cv.name).unwrap();
                d.set_item("value", cv.value).unwrap();
                d.set_item("active", cv.active).unwrap();
                d
            })
            .collect();
        dict.set_item("constraint_values", constraints)?;

        Ok(dict)
    }
}

/// The Python module.
#[pymodule]
fn heidegger(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SafetyShim>()?;
    m.add_class::<CollisionGuard>()?;
    m.add_class::<TrajectoryRecorder>()?;
    m.add_class::<PySafetySet>()?;
    m.add_class::<PyCBFSafetyFilter>()?;
    Ok(())
}
