<p align="center">
  <h1 align="center">🛡️ Heidegger</h1>
  <p align="center">
    <strong>A Deterministic Safety Layer for VLA-Driven Robot Arms</strong>
  </p>
  <p align="center">
    <a href="./README_ZH.md">中文文档</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#cbf-safety-filter">CBF Safety Filter</a> ·
    <a href="#lerobot-integration">LeRobot Integration</a> ·
    <a href="#interactive-simulator">Interactive Simulator</a>
  </p>
</p>

---

## The Problem

VLA (Vision-Language-Action) models are revolutionizing robotic manipulation — but they hallucinate. A single bad prediction can send a robot arm crashing into itself, drilling through a table, or bending joints past their physical limits.

Current safety approaches in the open-source VLA ecosystem (like [LeRobot](https://github.com/huggingface/lerobot)) are mostly **manual**: developers rely on `torque_limit` parameters, physical leader-arm constraints, or simply standing next to the robot ready to pull the plug.

**Heidegger** is a Rust-powered deterministic safety layer that sits between your VLA model and the robot actuators. It enforces physical constraints in real-time (~26μs overhead), regardless of how wrong the AI's predictions are.

> *Named after Martin Heidegger's concept of "Zuhandenheit" (ready-to-hand) — the best tool is one you don't notice.*

## How It Works

```
VLA Model Output → [ Heidegger Safety Layer ] → Safe Actuator Commands
                           │
                 ┌─────────┤─────────┐
                 ▼         ▼         ▼
          Per-Joint     CBF QP     Learned
          Clamping    Optimizer    Workspace
                 │         │         │
                 └─────────┤─────────┘
                           ▼
                     Safe Output ✅
```

### Two Safety Modes

| Mode | Approach | Use Case |
|:-----|:---------|:---------|
| **SafetyShim** (Classic) | Independent per-joint position + velocity clamping | Simple, predictable, zero-config |
| **CBF Safety Filter** ✨ | Globally optimal QP with joint, velocity, collision & workspace constraints | Maximum safety with minimal intervention |

### Five-Layer Protection

| Layer | What It Does | Latency |
|:------|:-------------|:--------|
| **L1 — Position Clamping** | Enforces per-joint angle limits | ~1 μs |
| **L2 — Velocity Limiting** | Caps joint velocity to prevent sudden jerks | ~1 μs |
| **L3 — Self-Collision Detection** | FK + capsule collision to reject dangerous poses | ~20 μs |
| **L4 — Workspace Boundary** | Prevents arm from penetrating table/floor surfaces | ~3 μs |
| **L5 — Learned Safety Set** ✨ | Data-driven workspace boundaries from demonstrations | ~5 μs |

**Total pipeline: ~26 μs per control step** — invisible to your Python control loop.

## Quick Start

### Prerequisites

- **Rust** toolchain (1.70+)
- **Python** 3.9+
- **maturin** (`pip install maturin`)

### Install

```bash
git clone https://github.com/HeidSXY/heidegger.git
cd heidegger

python3.10 -m venv .venv
source .venv/bin/activate

pip install maturin
maturin develop

# For simulator demos
pip install mujoco numpy fastapi uvicorn websockets Pillow
```

### Basic Usage (Classic Mode)

```python
from heidegger import SafetyShim, CollisionGuard

# Load joint configuration
with open("models/so_arm101_joints.json") as f:
    config = f.read()

shim = SafetyShim(config, dt=0.02)           # 50Hz control loop
guard = CollisionGuard(safety_margin=0.015)   # 15mm safety margin

# In your control loop:
def safe_step(vla_action, current_position):
    # L1 + L2: Position clamping + velocity limiting
    result = shim.check(vla_action, current_position)
    safe_action = result["safe_action"]

    # L3: Self-collision detection
    if guard.has_collision(safe_action):
        return current_position  # Reject dangerous pose

    return safe_action
```

## CBF Safety Filter

The **Control Barrier Function** mode replaces per-joint clamping with globally optimal safe action computation. Instead of modifying each joint independently, it solves a Quadratic Program to find the **closest feasible action** to the VLA output:

```
u* = argmin ‖u - u_vla‖²
     s.t.   joint position limits
            velocity limits
            self-collision margins
            learned workspace boundaries
```

### Why CBF > Clamping?

| | SafetyShim (Clamp) | CBF Filter |
|---|---|---|
| **Approach** | Per-joint independent | Global QP optimization |
| **Optimality** | Greedy (can distort trajectories) | Min-norm modification |
| **Constraints** | Position + velocity only | + collision + learned workspace |
| **Data-driven** | ❌ | ✅ Learned safety set |
| **Formal guarantee** | ❌ | ✅ CBF invariance |

### Usage

```python
from heidegger import PyCBFSafetyFilter
import json

with open("models/so_arm101_joints.json") as f:
    config = f.read()

# Basic CBF (joint + velocity + collision)
cbf = PyCBFSafetyFilter(config, dt=0.02, robot_model="so_arm101")
result = cbf.filter(vla_action, current_position)
safe_action = result["safe_action"]
# result also contains: was_modified, modification_norm, active_constraints, iterations

# With learned workspace boundaries
with open("safety_set.json") as f:
    ss_json = f.read()
cbf = PyCBFSafetyFilter(config, dt=0.02, robot_model="so_arm101", safety_set_json=ss_json)
```

### Learned Safety Boundaries

Learn the safe workspace from human demonstrations — the robot stays where you showed it's safe to go:

```python
from heidegger import TrajectoryRecorder

# Step 1: Record during teleoperation
recorder = TrajectoryRecorder(num_joints=6)
for joint_angles in teleoperation_stream:
    recorder.record(joint_angles)

# Step 2: Calibrate (per-joint bounds + PCA convex hull)
safety_set = recorder.calibrate(sigma_multiplier=3.0, pca_dims=3)
safety_set_json = safety_set.to_json()

# Step 3: Save for deployment
with open("safety_set.json", "w") as f:
    f.write(safety_set_json)
```

Or use the CLI:

```bash
python -m heidegger.calibrate -t recordings.json -o safety_set.json --sigma 3.0 --pca-dims 3 -v
```

## LeRobot Integration

Heidegger provides drop-in wrappers for [HuggingFace LeRobot](https://github.com/huggingface/lerobot) policies:

### Classic Mode (Per-Joint Clamping)

```python
from heidegger.lerobot import SafetyWrapper

safe_policy = SafetyWrapper(
    policy=your_lerobot_policy,
    robot_model="so_arm101",     # or "koch_v1_1", "so_arm100"
    dt=0.02,
    safety_margin=0.015,
)
action = safe_policy.select_action(observation)
```

### CBF Mode (Recommended)

```python
from heidegger.lerobot import CBFSafetyWrapper

# With learned workspace boundaries
with open("safety_set.json") as f:
    ss_json = f.read()

safe_policy = CBFSafetyWrapper(
    policy=your_lerobot_policy,
    robot_model="so_arm101",
    safety_set_json=ss_json,    # optional: learned workspace
)
action = safe_policy.select_action(observation)
print(safe_policy.stats)  # {"total_steps": 100, "interventions": 3, "intervention_rate": 0.03}
```

Both wrappers intercept `select_action()`, run safety checks, and return the safe version. Your existing LeRobot code doesn't change.

## Interactive Simulator

A web-based tool for visualizing how Heidegger protects the robot in real-time.

```bash
python tools/server.py
# Open http://localhost:8000
```

**Features:**
- 🎚️ **6 joint sliders** — drag to control the robot in real-time
- 🔴🟢 **Dual viewport** — raw commands (top) vs. Heidegger-filtered (bottom)
- ⚠️ **4-layer safety indicators** — CLAMP / VEL / COL / BND status
- 🎯 **Preset bad cases** — one-click dangerous scenarios (table penetration, self-collision)
- 🎲 **VLA noise injection** — simulate VLA hallucinations with adjustable intensity

## Performance

Benchmarked on Apple M1:

| Operation | Latency |
|:----------|:--------|
| Position clamping (6 joints) | 0.8 μs |
| Velocity limiting (6 joints) | 1.2 μs |
| Forward kinematics (7 frames) | 3.5 μs |
| Self-collision check (10 pairs) | 20.5 μs |
| CBF QP solve (6-DOF, 25+ constraints) | ~50 μs |
| **Full pipeline** | **~26 μs** (classic) / **~75 μs** (CBF) |

## Architecture

```
heidegger/
├── heidegger-core/          # Pure Rust safety logic (zero external deps beyond serde)
│   └── src/
│       ├── lib.rs           # SafetyShim: position & velocity clamping
│       ├── kinematics.rs    # Forward kinematics (homogeneous transforms)
│       ├── collision.rs     # Capsule-based self-collision detection
│       ├── safety_set.rs    # Learned safety boundaries (PCA + convex hull)
│       └── cbf.rs           # CBF safety filter (projected gradient descent QP)
├── heidegger-py/            # PyO3 Python bindings
│   └── src/lib.rs           # 5 Python classes (SafetyShim, CollisionGuard,
│                            #   TrajectoryRecorder, PySafetySet, PyCBFSafetyFilter)
├── python/heidegger/        # Python package
│   ├── __init__.py          # Public API (8 exports)
│   ├── lerobot.py           # LeRobot wrappers (SafetyWrapper + CBFSafetyWrapper)
│   └── calibrate.py         # CLI: learn safety boundaries from trajectory data
├── models/                  # Robot configurations
│   ├── so_arm101_joints.json
│   ├── koch_v1_1_joints.json
│   └── so_arm100_joints.json
├── tools/                   # Interactive simulator
│   ├── server.py            # FastAPI + WebSocket + MuJoCo
│   └── static/index.html    # Dual-viewport Web UI
└── examples/                # Demo scripts
```

## Supported Robots

| Robot | Position Limits | Velocity Limits | FK | Collision | CBF |
|:------|:---:|:---:|:---:|:---:|:---:|
| **SO-ARM101** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Koch v1.1** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SO-ARM100** | ✅ | ✅ | ✅ | ✅ | ✅ |

Adding a new robot requires:
1. A joint configuration JSON file (angle limits, velocity limits)
2. A kinematic chain definition (joint axes + offsets)
3. Capsule approximations for collision geometry

## Why Rust?

| | Python | C++ | **Rust** |
|:--|:--|:--|:--|
| GC pauses | 50-200ms | None | **None** |
| Memory safety | Runtime exceptions | Segfaults | **Compile-time guaranteed** |
| Control loop latency | 1-10ms | 0.01-0.1ms | **0.01-0.05ms** |
| Python interop | Native | pybind11 | **PyO3 (zero-copy)** |

The safety layer must never freeze, crash, or introduce latency spikes. Rust is the only modern language that guarantees all three.

## Roadmap

- [x] Position clamping (L1)
- [x] Velocity limiting (L2)
- [x] Self-collision detection (L3)
- [x] Workspace boundary checking (L4)
- [x] Learned safety boundaries (L5) — PCA + convex hull
- [x] CBF safety filter — projected gradient descent QP
- [x] Full multi-robot support (SO-ARM101, Koch v1.1, SO-ARM100)
- [x] Interactive web simulator (dual viewport)
- [x] LeRobot integration (SafetyWrapper + CBFSafetyWrapper)
- [x] Calibration CLI (`python -m heidegger.calibrate`)
- [x] Shadow mode (observe violations without enforcing)
- [ ] Benchmark report with comparison data
- [ ] MuJoCo simulation: clamp vs. CBF visual comparison
- [ ] Action anomaly detection (spike/oscillation/stall)
- [ ] Black box event logger

## Related Work

- [AEGIS/VLSA](https://arxiv.org/abs/2412.12267) — CBF-based safety layer (Python, obstacle avoidance only)
- [SafeVLA](https://arxiv.org/abs/2503.14729) — Constrained RL for VLA alignment (requires retraining)
- [dora-rs](https://github.com/dora-rs/dora) — Rust robotics data flow framework (communication, not safety)

Heidegger differs by being: Rust-native (not Python), covering full safety stack (not just collision), CBF-optimized (not just clamping), data-driven (learned boundaries), and drop-in usable (not requiring model retraining).

## License

[MIT](./LICENSE)

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## Acknowledgments

- [MuJoCo](https://mujoco.org/) — Physics simulation
- [PyO3](https://pyo3.rs/) — Rust-Python bindings
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — The VLA ecosystem we're building for
- [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) — Robot arm hardware
