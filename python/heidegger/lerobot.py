"""
Heidegger LeRobot Integration
==============================

Drop-in safety wrapper for HuggingFace LeRobot policies.
Intercepts action outputs and applies deterministic safety checks
before they reach the robot actuators.

Usage:
    from heidegger.lerobot import SafetyWrapper

    safe_policy = SafetyWrapper(
        policy=your_policy,
        robot_model="so_arm101",
    )
    action = safe_policy.select_action(observation)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Union

try:
    from heidegger import SafetyShim, CollisionGuard
except ImportError:
    raise ImportError(
        "Heidegger native module not found. "
        "Please install with: cd heidegger && maturin develop"
    )

# Resolve models directory relative to package root
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Built-in robot model registry
ROBOT_CONFIGS = {
    "so_arm101": "so_arm101_joints.json",
    "koch_v1_1": "koch_v1_1_joints.json",
    "so_arm100": "so_arm100_joints.json",
}


def _load_config(robot_model: str) -> str:
    """Load joint config JSON for a named robot model."""
    if robot_model in ROBOT_CONFIGS:
        config_path = _MODELS_DIR / ROBOT_CONFIGS[robot_model]
    else:
        # Treat as a direct file path
        config_path = Path(robot_model)

    if not config_path.exists():
        available = list(ROBOT_CONFIGS.keys())
        raise FileNotFoundError(
            f"Robot config not found: {config_path}\n"
            f"Available built-in models: {available}\n"
            f"Or pass a path to your own joints.json file."
        )

    return config_path.read_text()


class SafetyWrapper:
    """
    Wraps a LeRobot policy with Heidegger safety checks.

    The wrapper intercepts `select_action()`, runs position clamping,
    velocity limiting, and self-collision detection on the output,
    and returns the safe version.

    Args:
        policy: A LeRobot policy object with `select_action(obs)` method.
        robot_model: Name of a built-in robot ("so_arm101", "koch_v1_1",
                     "so_arm100") or path to a joints.json file.
        dt: Control loop period in seconds (default: 0.02 = 50Hz).
        safety_margin: Collision detection margin in meters (default: 0.015).
        collision_check: Whether to run self-collision detection (default: True).
        mode: "guardian" (default, clamps actions) or "shadow" (observe only,
              useful during training to avoid polluting demonstrations).
        verbose: Print safety interventions to console (default: False).

    Example:
        >>> safe_policy = SafetyWrapper(policy, robot_model="so_arm101")
        >>> action = safe_policy.select_action(observation)
        >>> # Shadow mode (training): observe without interfering
        >>> safe_policy = SafetyWrapper(policy, mode="shadow")
    """

    def __init__(
        self,
        policy,
        robot_model: str = "so_arm101",
        dt: float = 0.02,
        safety_margin: float = 0.015,
        collision_check: bool = True,
        mode: str = "guardian",
        verbose: bool = False,
    ):
        self.policy = policy
        self.verbose = verbose
        self.collision_check = collision_check
        self._mode = mode

        # Load safety shim with mode
        config_json = _load_config(robot_model)
        self.shim = SafetyShim(config_json, dt, mode)

        # Load collision guard — all built-in models now fully supported
        self.guard = None
        if collision_check:
            try:
                self.guard = CollisionGuard(safety_margin, robot_model)
            except Exception:
                warnings.warn(
                    f"Self-collision detection not available for '{robot_model}'. "
                    f"Only position/velocity clamping will be applied. "
                    f"Set collision_check=False to suppress this warning.",
                    stacklevel=2,
                )

        # State tracking
        self._last_action = None
        self._intervention_count = 0
        self._total_steps = 0

    def select_action(self, observation, **kwargs):
        """
        Run the wrapped policy's select_action, then apply safety checks.

        Args:
            observation: The observation dict passed to the policy.
            **kwargs: Additional arguments forwarded to the policy.

        Returns:
            Safe action tensor/array (same type as policy output).
        """
        # Get raw action from policy
        raw_action = self.policy.select_action(observation, **kwargs)

        # Convert to list for Heidegger
        action_list = self._to_list(raw_action)

        # Use last action as "current position" for velocity clamping
        # On first step, assume current == target (no velocity limit)
        if self._last_action is None:
            current = action_list
        else:
            current = self._last_action

        # Run safety check
        result = self.shim.check(action_list, current)
        safe_list = result["safe_action"]

        # Collision check (if available)
        collision_rejected = False
        if self.guard is not None:
            if self.guard.has_collision(safe_list):
                collision_rejected = True
                safe_list = current  # Stay at current position

        # Update state
        self._last_action = safe_list
        self._total_steps += 1

        if result["was_clamped"] or collision_rejected:
            self._intervention_count += 1
            if self.verbose:
                reasons = [v["reason"] for v in result.get("violations", [])]
                if collision_rejected:
                    reasons.append("COLLISION_REJECTED")
                print(
                    f"[Heidegger] Step {self._total_steps}: "
                    f"Intercepted ({', '.join(reasons)})"
                )

        # Convert back to original type
        return self._from_list(raw_action, safe_list)

    @property
    def stats(self) -> dict:
        """Return safety intervention statistics."""
        return {
            "total_steps": self._total_steps,
            "interventions": self._intervention_count,
            "intervention_rate": (
                self._intervention_count / self._total_steps
                if self._total_steps > 0
                else 0.0
            ),
        }

    def reset(self):
        """Reset internal state (call between episodes)."""
        self._last_action = None
        self._intervention_count = 0
        self._total_steps = 0

    # --- Type conversion helpers ---

    @staticmethod
    def _to_list(action) -> list:
        """Convert action tensor/ndarray to plain list."""
        if hasattr(action, "cpu"):
            # PyTorch tensor
            return action.cpu().detach().numpy().flatten().tolist()
        elif hasattr(action, "tolist"):
            # NumPy array
            return action.flatten().tolist()
        elif isinstance(action, (list, tuple)):
            return list(action)
        else:
            raise TypeError(
                f"Unsupported action type: {type(action)}. "
                f"Expected torch.Tensor, np.ndarray, or list."
            )

    @staticmethod
    def _from_list(original, safe_list: list):
        """Convert safe_list back to the original action type."""
        if hasattr(original, "cpu"):
            # PyTorch tensor — reconstruct with same device/dtype
            import torch
            return torch.tensor(
                safe_list, dtype=original.dtype, device=original.device
            ).reshape(original.shape)
        elif hasattr(original, "tolist"):
            # NumPy array
            import numpy as np
            return np.array(safe_list, dtype=original.dtype).reshape(original.shape)
        else:
            return safe_list

    # --- Forward other attributes to the wrapped policy ---

    def __getattr__(self, name):
        """Forward attribute access to the wrapped policy."""
        return getattr(self.policy, name)


class CBFSafetyWrapper:
    """
    Wraps a LeRobot policy with CBF-based safety filter.

    Instead of independent per-joint clamping, uses a Control Barrier Function
    (QP optimization) to find the globally optimal safe action. Supports
    learned workspace boundaries for data-driven safety.

    Args:
        policy: A LeRobot policy object with `select_action(obs)` method.
        robot_model: Name of a built-in robot or path to joints.json.
        dt: Control loop period in seconds (default: 0.02).
        safety_margin: Collision detection margin in meters (default: 0.015).
        alpha: CBF decay rate (0,1], lower = more conservative (default: 0.3).
        safety_set_json: Optional JSON string of a learned SafetySet.
        verbose: Print safety interventions to console (default: False).

    Example:
        >>> # Basic CBF (joint + velocity + collision constraints)
        >>> safe_policy = CBFSafetyWrapper(policy, robot_model="so_arm101")
        >>>
        >>> # With learned workspace boundaries
        >>> with open("safety_set.json") as f:
        ...     ss_json = f.read()
        >>> safe_policy = CBFSafetyWrapper(
        ...     policy, robot_model="so_arm101", safety_set_json=ss_json
        ... )
        >>> action = safe_policy.select_action(observation)
    """

    def __init__(
        self,
        policy,
        robot_model: str = "so_arm101",
        dt: float = 0.02,
        safety_margin: float = 0.015,
        alpha: float = 0.3,
        safety_set_json: Optional[str] = None,
        verbose: bool = False,
    ):
        from heidegger import PyCBFSafetyFilter

        self.policy = policy
        self.verbose = verbose

        config_json = _load_config(robot_model)
        self.cbf = PyCBFSafetyFilter(
            config_json,
            dt=dt,
            collision_margin=safety_margin,
            alpha=alpha,
            robot_model=robot_model,
            safety_set_json=safety_set_json,
        )

        # State tracking
        self._last_action = None
        self._intervention_count = 0
        self._total_steps = 0

    def select_action(self, observation, **kwargs):
        """Run policy + CBF safety filter."""
        raw_action = self.policy.select_action(observation, **kwargs)
        action_list = SafetyWrapper._to_list(raw_action)

        if self._last_action is None:
            current = action_list
        else:
            current = self._last_action

        result = self.cbf.filter(action_list, current)
        safe_list = result["safe_action"]

        self._last_action = safe_list
        self._total_steps += 1

        if result["was_modified"]:
            self._intervention_count += 1
            if self.verbose:
                print(
                    f"[Heidegger CBF] Step {self._total_steps}: "
                    f"Modified (norm={result['modification_norm']:.4f}, "
                    f"active={result['active_constraints']}, "
                    f"iters={result['iterations']})"
                )

        return SafetyWrapper._from_list(raw_action, safe_list)

    @property
    def stats(self) -> dict:
        """Return safety intervention statistics."""
        return {
            "total_steps": self._total_steps,
            "interventions": self._intervention_count,
            "intervention_rate": (
                self._intervention_count / self._total_steps
                if self._total_steps > 0
                else 0.0
            ),
        }

    def reset(self):
        """Reset internal state (call between episodes)."""
        self._last_action = None
        self._intervention_count = 0
        self._total_steps = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)


def calibrate_from_trajectories(
    trajectory_file: str,
    num_joints: int = 6,
    sigma_multiplier: float = 3.0,
    pca_dims: int = 3,
    output_file: Optional[str] = None,
) -> str:
    """
    Calibrate safety boundaries from saved trajectory data.

    Args:
        trajectory_file: Path to JSON file containing trajectory data.
            Format: {"trajectories": [[j0, j1, ...], [j0, j1, ...], ...]}
        num_joints: Number of joints.
        sigma_multiplier: Std dev multiplier for per-joint bounds.
        pca_dims: PCA dimensions for convex hull (0 = skip).
        output_file: Optional output path for safety set JSON.

    Returns:
        JSON string of the learned SafetySet.
    """
    from heidegger import TrajectoryRecorder

    with open(trajectory_file, "r") as f:
        data = json.load(f)

    trajectories = data.get("trajectories", data.get("samples", []))
    if not trajectories:
        raise ValueError(f"No trajectory data found in {trajectory_file}")

    rec = TrajectoryRecorder(num_joints)
    for sample in trajectories:
        rec.record(sample)

    ss = rec.calibrate(sigma_multiplier, pca_dims)

    json_str = ss.to_json()
    if output_file:
        with open(output_file, "w") as f:
            f.write(json_str)
        print(f"Safety set saved to {output_file} ({ss.num_samples} samples)")

    return json_str

