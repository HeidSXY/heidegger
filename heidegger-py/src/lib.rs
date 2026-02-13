use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};
use heidegger_core::{JointLimit, SafetyShim as RustSafetyShim};
use heidegger_core::kinematics::{self, KinematicChain};
use heidegger_core::collision::{self, CollisionChecker};

/// Python-facing SafetyShim class — position/velocity clamping.
#[pyclass]
struct SafetyShim {
    inner: RustSafetyShim,
}

#[pymethods]
impl SafetyShim {
    /// Create a SafetyShim from a JSON config string.
    #[new]
    fn new(config_json: &str, dt: f64) -> PyResult<Self> {
        let inner = RustSafetyShim::from_json(config_json, dt)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok(SafetyShim { inner })
    }

    /// Create a SafetyShim with simple joint limits (no JSON needed).
    #[staticmethod]
    fn from_limits(
        names: Vec<String>,
        lowers: Vec<f64>,
        uppers: Vec<f64>,
        max_velocities: Vec<f64>,
        dt: f64,
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

        Ok(SafetyShim {
            inner: RustSafetyShim::new(joints, dt),
        })
    }

    /// Check a raw action for safety and return the clamped version.
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
        dict.set_item("safe_action", &result.safe_action)?;
        dict.set_item("was_clamped", result.was_clamped)?;

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
}

/// Python-facing CollisionGuard — forward kinematics + self-collision detection.
#[pyclass]
struct CollisionGuard {
    chain: KinematicChain,
    checker: CollisionChecker,
}

#[pymethods]
impl CollisionGuard {
    /// Create a CollisionGuard for the SO-ARM101 with a given safety margin (meters).
    #[new]
    fn new(safety_margin: f64) -> Self {
        CollisionGuard {
            chain: kinematics::so_arm101_chain(),
            checker: collision::so_arm101_collision_checker(safety_margin),
        }
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
}

/// The Python module.
#[pymodule]
fn heidegger(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SafetyShim>()?;
    m.add_class::<CollisionGuard>()?;
    Ok(())
}
