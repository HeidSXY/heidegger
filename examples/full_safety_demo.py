"""
Heidegger Full Safety Demo: SO-ARM101
======================================

Demonstrates ALL safety features:
1. Joint position clamping
2. Velocity limiting  
3. Self-collision detection (Forward Kinematics + Capsule geometry)

Runs a MuJoCo simulation comparing:
- UNSAFE: Raw noisy VLA actions → direct motor control
- SAFE:   Same actions → Heidegger filter → motor control

Usage:
    source .venv310/bin/activate
    python examples/full_safety_demo.py
"""

import mujoco
import numpy as np
import json
import os
import time

from heidegger import SafetyShim, CollisionGuard


def create_sim():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def generate_trajectory_with_collision_risk(n_steps: int) -> np.ndarray:
    """
    Generate a trajectory that intentionally passes through self-collision zones.
    Simulates a VLA model trying to reach behind/under the arm.
    """
    t = np.linspace(0, 1, n_steps)
    trajectory = np.zeros((n_steps, 6))

    # Phase 1 (0-40%): Normal reaching forward
    # Phase 2 (40-70%): Try to reach under the arm (collision risk!)
    # Phase 3 (70-100%): Try to curl back (extreme collision risk!)
    
    for i, ti in enumerate(t):
        if ti < 0.4:
            # Normal reach forward
            p = ti / 0.4
            trajectory[i] = [0.3*p, 0.6*p, -1.2*p, 0.3*p, 0.0, 0.5]
        elif ti < 0.7:
            # Dangerous: shoulder forward + elbow tight + wrist curling back
            p = (ti - 0.4) / 0.3
            trajectory[i] = [
                0.3,
                0.6 + 0.5*p,       # shoulder keeps going up
                -1.2 - 1.1*p,      # elbow curls tighter (toward -2.3)
                0.3 - 1.8*p,       # wrist curls back toward base
                0.0,
                0.5
            ]
        else:
            # Extreme: full curl-back (maximum collision risk)
            p = (ti - 0.7) / 0.3
            trajectory[i] = [
                0.3 - 0.3*p,
                1.1 - 0.5*p,       # shoulder settles
                -2.3,              # elbow fully curled
                -1.5 - 0.07*p,    # wrist keeps going
                0.0,
                0.5
            ]

    return trajectory


def inject_vla_noise(trajectory: np.ndarray, noise_level: float = 0.15,
                     spike_prob: float = 0.02, spike_scale: float = 4.0) -> np.ndarray:
    rng = np.random.RandomState(42)
    noisy = trajectory.copy()
    noisy += rng.randn(*trajectory.shape) * noise_level
    spike_mask = rng.rand(*trajectory.shape) < spike_prob
    spikes = rng.randn(*trajectory.shape) * spike_scale
    noisy[spike_mask] += spikes[spike_mask]
    return noisy


def run_simulation(model, data, actions: np.ndarray) -> dict:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    n_steps = len(actions)
    results = {
        "joint_positions": np.zeros((n_steps, 6)),
        "joint_velocities": np.zeros((n_steps, 6)),
        "actions_applied": np.zeros((n_steps, 6)),
        "ee_positions": np.zeros((n_steps, 3)),
    }

    for i in range(n_steps):
        data.ctrl[:6] = actions[i]
        for _ in range(10):
            mujoco.mj_step(model, data)

        results["joint_positions"][i] = data.qpos[:6]
        results["joint_velocities"][i] = data.qvel[:6]
        results["actions_applied"][i] = actions[i]

        ee_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_pos")
        ee_adr = model.sensor_adr[ee_sensor_id]
        results["ee_positions"][i] = data.sensordata[ee_adr:ee_adr + 3]

    return results


def full_safety_check(noisy_actions, shim, guard, n_steps):
    """
    Apply the FULL Heidegger pipeline:
    1. Clamp positions and velocities (SafetyShim)
    2. Check self-collision (CollisionGuard)
    3. If collision detected, reject and hold previous position
    """
    safe_actions = np.zeros_like(noisy_actions)
    current_pos = np.zeros(6)
    
    stats = {
        "position_clamps": 0,
        "velocity_clamps": 0,
        "collision_blocks": 0,
        "collision_details": [],
    }

    for i in range(n_steps):
        # Step 1: Position + Velocity clamping
        result = shim.check(noisy_actions[i].tolist(), current_pos.tolist())
        clamped = np.array(result["safe_action"])
        
        if result["was_clamped"]:
            for v in result["violations"]:
                if "Velocity" in v["reason"]:
                    stats["velocity_clamps"] += 1
                else:
                    stats["position_clamps"] += 1

        # Step 2: Self-collision check
        if guard.has_collision(clamped.tolist()):
            # COLLISION! Reject this action, hold previous safe position
            stats["collision_blocks"] += 1
            collisions = guard.check_collisions(clamped.tolist())
            if len(collisions) > 0:
                detail = f"Step {i}: {collisions[0]['link_a']} ↔ {collisions[0]['link_b']} ({collisions[0]['distance']*1000:.1f}mm)"
                stats["collision_details"].append(detail)
            safe_actions[i] = current_pos  # Hold position
        else:
            safe_actions[i] = clamped
            current_pos = clamped

    return safe_actions, stats


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101_joints.json")
    with open(config_path) as f:
        config_json = f.read()

    shim = SafetyShim(config_json, dt=0.02)
    guard = CollisionGuard(safety_margin=0.015)  # 15mm safety margin
    model, data = create_sim()

    print("=" * 70)
    print("  HEIDEGGER FULL SAFETY DEMO v2")
    print("  Position Clamping + Velocity Limiting + Self-Collision Detection")
    print("=" * 70)

    # Generate trajectory with collision risk
    n_steps = 500
    clean = generate_trajectory_with_collision_risk(n_steps)
    noisy = inject_vla_noise(clean, noise_level=0.15, spike_prob=0.02, spike_scale=4.0)
    print(f"\n📐 Trajectory: {n_steps} steps, 10s @ 50Hz")
    print(f"   Phase 1 (0-4s): Normal reach forward")
    print(f"   Phase 2 (4-7s): Reach under arm (collision risk 🔥)")
    print(f"   Phase 3 (7-10s): Full curl-back (extreme collision risk 💀)")
    print(f"   Noise: σ=0.15 rad + 2% chance of 4σ spikes\n")

    # --- Run WITHOUT Heidegger ---
    print("🔴 Running simulation WITHOUT any safety layer...")
    t0 = time.perf_counter()
    unsafe_results = run_simulation(model, data, noisy)
    t_unsafe = time.perf_counter() - t0

    # Check how many collision frames WITHOUT safety
    unsafe_collisions = 0
    worst_penetration = 0.0
    for i in range(n_steps):
        if guard.has_collision(noisy[i].tolist()):
            unsafe_collisions += 1
            cols = guard.check_collisions(noisy[i].tolist())
            for c in cols:
                worst_penetration = min(worst_penetration, c["distance"])

    print(f"   Done in {t_unsafe:.2f}s")
    print(f"   ⚠️  Collision frames: {unsafe_collisions}/{n_steps}")
    if worst_penetration < 0:
        print(f"   💀 Worst penetration: {worst_penetration*1000:.1f} mm INTO another link")

    # --- Run WITH Heidegger ---
    print("\n🟢 Running simulation WITH full Heidegger safety pipeline...")
    t0 = time.perf_counter()
    safe_actions, stats = full_safety_check(noisy, shim, guard, n_steps)
    t_safety = time.perf_counter() - t0
    safe_results = run_simulation(model, data, safe_actions)
    print(f"   Safety processing: {t_safety*1000:.1f}ms ({t_safety/n_steps*1e6:.1f} μs/step)")

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n🔴 WITHOUT Heidegger:")
    print(f"   Collision frames: {unsafe_collisions}")
    if worst_penetration < 0:
        print(f"   Worst penetration: {abs(worst_penetration)*1000:.1f} mm")
    print(f"   Max |velocity|:   {np.max(np.abs(unsafe_results['joint_velocities'])):.2f} rad/s")

    print(f"\n🟢 WITH Heidegger:")
    print(f"   Position clamps:  {stats['position_clamps']}")
    print(f"   Velocity clamps:  {stats['velocity_clamps']}")
    print(f"   Collision blocks:  {stats['collision_blocks']}")
    print(f"   Max |velocity|:   {np.max(np.abs(safe_results['joint_velocities'])):.2f} rad/s")

    if stats["collision_details"]:
        print(f"\n   Sample blocked collisions:")
        for d in stats["collision_details"][:5]:
            print(f"     ❌ {d}")
        if len(stats["collision_details"]) > 5:
            print(f"     ... and {len(stats['collision_details']) - 5} more")

    # Verify no collisions in safe trajectory
    safe_collision_count = 0
    for i in range(n_steps):
        if guard.has_collision(safe_actions[i].tolist()):
            safe_collision_count += 1

    print(f"\n✅ Collision-free verification: ", end="")
    if safe_collision_count == 0:
        print("PASS — zero collisions in safe trajectory")
    else:
        print(f"FAIL — {safe_collision_count} collisions found!")

    # End effector comparison
    unsafe_ee = unsafe_results["ee_positions"]
    safe_ee = safe_results["ee_positions"]
    ee_diff = np.linalg.norm(unsafe_ee - safe_ee, axis=1)
    print(f"\n🎯 End Effector Trajectory Deviation:")
    print(f"   Mean: {np.mean(ee_diff)*1000:.1f} mm")
    print(f"   Max:  {np.max(ee_diff)*1000:.1f} mm")

    # Overall summary
    print(f"\n" + "=" * 70)
    print(f"  SUMMARY")
    print(f"  ├─ Position clamping:    catches out-of-range joints")
    print(f"  ├─ Velocity limiting:    prevents sudden jerks")
    print(f"  └─ Collision detection:  prevents self-destruction")
    print(f"")
    print(f"  Total safety overhead:   {t_safety/n_steps*1e6:.1f} μs per control step")
    print(f"  At 50Hz loop:            {t_safety/n_steps*1e6/20000*100:.2f}% of control budget")
    print(f"=" * 70)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "full_demo_results.npz"),
             clean=clean, noisy=noisy, safe_actions=safe_actions,
             unsafe_joint_pos=unsafe_results["joint_positions"],
             safe_joint_pos=safe_results["joint_positions"],
             unsafe_ee=unsafe_results["ee_positions"],
             safe_ee=safe_results["ee_positions"])
    print(f"\n💾 Results saved to results/full_demo_results.npz")


if __name__ == "__main__":
    main()
