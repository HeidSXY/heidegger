#!/usr/bin/env python3
"""
CBF Safety Filter Demo
=======================

Demonstrates the difference between classic per-joint clamping (SafetyShim)
and the CBF safety filter (PyCBFSafetyFilter).

Shows:
1. Both approaches on the same dangerous action
2. How learned safety boundaries add workspace constraints
3. The full Record → Calibrate → Deploy pipeline

Usage:
    cd heidegger
    source .venv/bin/activate
    python examples/cbf_demo.py
"""

import json
import math
import sys
import os

# Add python/ to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from heidegger import (
    SafetyShim,
    CollisionGuard,
    TrajectoryRecorder,
    PySafetySet,
    PyCBFSafetyFilter,
)


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_result(label: str, values: list, modified: bool = None):
    vals = ", ".join(f"{v:+.4f}" for v in values)
    status = ""
    if modified is not None:
        status = " 🔴 MODIFIED" if modified else " 🟢 SAFE"
    print(f"  {label}: [{vals}]{status}")


# ─────── Robot Configuration ───────

with open(os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101_joints.json")) as f:
    config_json = f.read()

config = json.loads(config_json)
DT = 0.02  # 50Hz control loop

# ─────── Demo 1: Classic vs CBF on the same dangerous action ───────

print_header("Demo 1: Classic Clamping vs CBF Safety Filter")

current_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
dangerous_action = [3.5, -2.0, 1.5, 4.0, -1.0, 2.5]  # Way outside limits

print("  Current position: all zeros")
print_result("Dangerous VLA output", dangerous_action)
print()

# Classic: SafetyShim (per-joint clamp)
shim = SafetyShim(config_json, DT, "guardian")
result = shim.check(dangerous_action, current_pos)
print("  --- SafetyShim (per-joint clamp) ---")
print_result("Safe action", result["safe_action"], result["was_clamped"])
violations = [d for d in result.get("details", []) if d.get("reason")]
for v in violations:
    print(f"    ⚠ {v['joint_name']}: {v['reason']}")

# CBF: Globally optimal
print()
cbf = PyCBFSafetyFilter(config_json, dt=DT, robot_model="so_arm101")
result = cbf.filter(dangerous_action, current_pos)
print("  --- CBF Safety Filter (global QP) ---")
print_result("Safe action", result["safe_action"], result["was_modified"])
print(f"    Modification norm: {result['modification_norm']:.4f}")
print(f"    Active constraints: {result['active_constraints']}")
print(f"    QP iterations: {result['iterations']}")

# ─────── Demo 2: Subtle velocity violation ───────

print_header("Demo 2: Subtle Velocity Violation")

current_pos = [0.5, -0.3, 0.1, 0.8, 0.0, 0.3]
# This action is within position limits but requires too-fast movement
fast_action = [1.0, 0.5, -0.5, 1.5, 0.5, -0.5]

print_result("Current position", current_pos)
print_result("Fast VLA action  ", fast_action)
print()

# Check max velocity deltas
for i, jl in enumerate(config):
    delta = abs(fast_action[i] - current_pos[i])
    max_delta = jl["max_velocity"] * DT
    if delta > max_delta:
        print(f"    Joint {jl['name']}: Δ={delta:.3f} rad > max_Δ={max_delta:.3f} rad "
              f"(vel={delta/DT:.1f} > max={jl['max_velocity']:.1f} rad/s)")

result_shim = shim.check(fast_action, current_pos)
print(f"\n  SafetyShim: clamped={result_shim['was_clamped']}")
print_result("  SafetyShim action", result_shim["safe_action"])

result_cbf = cbf.filter(fast_action, current_pos)
print(f"\n  CBF Filter: modified={result_cbf['was_modified']}")
print_result("  CBF action       ", result_cbf["safe_action"])

# ─────── Demo 3: Learned Safety Boundaries ───────

print_header("Demo 3: Learned Safety Boundaries (Record → Calibrate → Deploy)")

# Simulate a typical workspace from teleoperation
print("  Step 1: Recording 500 teleoperation samples...")
recorder = TrajectoryRecorder(6)
for i in range(500):
    t = i * 0.02
    recorder.record([
        0.3 + 0.4 * math.sin(t * 0.5),       # shoulder: gentle sweep
        -0.5 + 0.3 * math.cos(t * 0.3),       # upper arm: slow wave
        0.1 + 0.15 * math.sin(t * 0.7),       # elbow: small range
        0.5 + 0.3 * math.cos(t * 0.4),        # wrist pitch
        0.0 + 0.2 * math.sin(t * 0.6),        # wrist roll
        0.3 + 0.1 * math.cos(t * 0.8),        # gripper
    ])
print(f"    Recorded {recorder.num_samples} samples ✓")

# Calibrate
print("\n  Step 2: Calibrating safety set (μ ± 3σ + PCA hull)...")
safety_set = recorder.calibrate(sigma_multiplier=3.0, pca_dims=3)
print(f"    Joints: {safety_set.num_joints}")

bounds = safety_set.joint_bounds()
for i, b in enumerate(bounds):
    print(f"    J{i}: [{b['lower']:+.3f}, {b['upper']:+.3f}]  "
          f"(μ={b['mean']:+.3f}, σ={b['std_dev']:.3f})")

# Test containment
center = [0.3, -0.5, 0.1, 0.5, 0.0, 0.3]
outside = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
print(f"\n    Contains workspace center: {safety_set.contains(center)} ✓")
print(f"    Contains faraway point:    {safety_set.contains(outside)} ✓")

# Deploy with CBF + learned workspace
print("\n  Step 3: Deploy CBF with learned workspace...")
ss_json = safety_set.to_json()
cbf_full = PyCBFSafetyFilter(
    config_json, dt=DT,
    robot_model="so_arm101",
    safety_set_json=ss_json,
)

# Action inside learned workspace
result = cbf_full.filter(center, center)
print(f"\n    Action in workspace:     modified={result['was_modified']} (expected: False)")

# Action outside learned workspace
result = cbf_full.filter(outside, current_pos)
print(f"    Action outside workspace: modified={result['was_modified']} (expected: True)")
print(f"    Constraints active: {result['active_constraints']}")
print_result("    CBF safe action ", result["safe_action"])

# ─────── Demo 4: Statistics ───────

print_header("Demo 4: Deployment Statistics Simulation")

print("  Simulating 100 VLA inference steps with noise...")

cbf_interventions = 0
shim_interventions = 0
pos = [0.3, -0.5, 0.1, 0.5, 0.0, 0.3]

for step in range(100):
    t = step * DT
    # Simulate VLA output: mostly safe but occasional spikes
    noise_scale = 0.02 if step % 20 != 0 else 0.5  # spike every 20 steps
    vla_output = [
        pos[j] + noise_scale * math.sin(t * (j + 1) * 3.7 + j)
        for j in range(6)
    ]

    # CBF filter
    r_cbf = cbf_full.filter(vla_output, pos)
    if r_cbf["was_modified"]:
        cbf_interventions += 1

    # Classic shim
    r_shim = shim.check(vla_output, pos)
    if r_shim["was_clamped"]:
        shim_interventions += 1

    # Update position (use CBF output)
    pos = r_cbf["safe_action"]

print(f"\n  Results over 100 steps:")
print(f"    SafetyShim interventions: {shim_interventions} ({shim_interventions}%)")
print(f"    CBF Filter interventions: {cbf_interventions} ({cbf_interventions}%)")
print(f"    CBF catches more because it includes collision + workspace constraints")

# ─────── Summary ───────

print_header("Summary")
print("  SafetyShim: Fast, simple, per-joint. Good for basic protection.")
print("  CBF Filter: Globally optimal, multi-constraint. Recommended for deployment.")
print("  Learned Boundaries: Record → Calibrate → Deploy. Data-driven safety.")
print()
print("  Full pipeline:")
print("    1. pip install maturin && maturin develop")
print("    2. Record trajectories during teleoperation")
print("    3. python -m heidegger.calibrate -t data.json -o safety.json")
print("    4. Use CBFSafetyWrapper with safety_set_json in production")
print()
