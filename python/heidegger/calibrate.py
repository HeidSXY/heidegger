#!/usr/bin/env python3
"""
Heidegger Safety Calibration CLI
=================================

Learn safety boundaries from robot trajectory data.

Usage:
    # Record trajectories during teleoperation, then calibrate:
    python -m heidegger.calibrate \
        --trajectories recordings/demo1.json \
        --output safety_set.json \
        --sigma 3.0 \
        --pca-dims 3

    # Then use with CBFSafetyWrapper:
    from heidegger import CBFSafetyWrapper
    with open("safety_set.json") as f:
        ss_json = f.read()
    safe_policy = CBFSafetyWrapper(policy, safety_set_json=ss_json)
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Heidegger: Learn safety boundaries from trajectory data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--trajectories", "-t",
        required=True,
        help="Path to trajectory JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        default="safety_set.json",
        help="Output path for safety set JSON (default: safety_set.json)",
    )
    parser.add_argument(
        "--num-joints", "-n",
        type=int,
        default=6,
        help="Number of joints (default: 6)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Std dev multiplier for per-joint bounds (default: 3.0)",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=3,
        help="PCA dimensions for convex hull, 0=skip (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed calibration info",
    )

    args = parser.parse_args()

    from heidegger import TrajectoryRecorder, PySafetySet

    # Load data
    with open(args.trajectories, "r") as f:
        data = json.load(f)

    trajectories = data.get("trajectories", data.get("samples", []))
    if not trajectories:
        print(f"Error: No trajectory data found in {args.trajectories}")
        print("Expected JSON format: {\"trajectories\": [[j0,j1,...], ...]}")
        sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectory samples")
    print(f"  Joints: {args.num_joints}")
    print(f"  Sigma: {args.sigma}")
    print(f"  PCA dims: {args.pca_dims}")

    # Record
    rec = TrajectoryRecorder(args.num_joints)
    for sample in trajectories:
        if len(sample) != args.num_joints:
            print(f"Warning: sample has {len(sample)} values, expected {args.num_joints}")
            continue
        rec.record(sample)

    print(f"\nRecorded {rec.num_samples} valid samples")

    # Calibrate
    ss = rec.calibrate(args.sigma, args.pca_dims)
    print(f"\nCalibrated safety set:")
    print(f"  Joints: {ss.num_joints}")
    print(f"  Samples: {ss.num_samples}")

    if args.verbose:
        bounds = ss.joint_bounds()
        for i, b in enumerate(bounds):
            print(f"  J{i}: [{b['lower']:.4f}, {b['upper']:.4f}] "
                  f"(μ={b['mean']:.4f}, σ={b['std_dev']:.4f})")

    # Save
    json_str = ss.to_json()
    with open(args.output, "w") as f:
        f.write(json_str)
    print(f"\nSaved to {args.output}")

    # Verify round-trip
    ss2 = PySafetySet.from_json(json_str)
    assert ss2.num_joints == ss.num_joints
    print("Verified JSON round-trip ✓")


if __name__ == "__main__":
    main()
