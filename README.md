<p align="center">
  <h1 align="center">🛡️ Heidegger</h1>
  <p align="center">
    <strong>A Deterministic Safety Layer for VLA-Driven Robot Arms</strong>
  </p>
  <p align="center">
    <a href="./README_ZH.md">中文文档</a> ·
    <a href="#quick-start">Quick Start</a> ·
    <a href="#interactive-simulator">Interactive Simulator</a> ·
    <a href="./docs/Project_Heidegger_Whitepaper_v1.md">Whitepaper</a>
  </p>
</p>

---

## The Problem

Vision-Language-Action (VLA) models are revolutionizing robotic manipulation — but they hallucinate. A single bad prediction can send a robot arm crashing into itself, drilling through a table, or bending joints past their physical limits.

**Heidegger** is a deterministic safety layer that sits between a VLA model and the robot actuators. It enforces physical constraints in real-time, regardless of how wrong the AI's predictions are.

## How It Works

```
VLA Model Output → [ Heidegger Safety Layer ] → Safe Actuator Commands
                         │
                    ┌────┴────┐
                    ▼         ▼
              Position    Velocity
              Clamping    Limiting
                    │         │
                    └────┬────┘
                         ▼
                   Self-Collision
                    Detection
                         │
                         ▼
                   Safe Output ✅
```

### Three-Layer Protection

| Layer | What It Does | Speed |
|:------|:-------------|:------|
| **Position Clamping** | Enforces per-joint angle limits (configurable via JSON) | ~1 μs |
| **Velocity Limiting** | Caps joint velocity to prevent sudden jerks | ~1 μs |
| **Self-Collision Detection** | Uses forward kinematics + capsule collision to reject dangerous poses | ~20 μs |

**Total pipeline: ~26 μs per control step** — fast enough for 1kHz real-time control loops.

## Architecture

```
heidegger/
├── heidegger-core/          # Pure Rust safety logic
│   └── src/
│       ├── lib.rs           # SafetyGuard: position & velocity clamping
│       ├── kinematics.rs    # Custom FK using homogeneous transforms
│       └── collision.rs     # Capsule-based self-collision detection
├── heidegger-py/            # PyO3 Python bindings
│   └── src/lib.rs           # SafetyShim + CollisionGuard
├── python/heidegger/        # Python package
├── models/                  # Robot definitions
│   ├── so_arm101.xml        # MuJoCo model for SO-ARM101
│   └── so_arm101_joints.json # Joint configuration
├── tools/                   # Interactive simulator
│   ├── server.py            # FastAPI + WebSocket backend
│   └── static/index.html    # Web UI
└── examples/                # Demo scripts
```

## Quick Start

### Prerequisites

- **Rust** toolchain (1.70+)
- **Python** 3.9+
- **maturin** (`pip install maturin`)

### Install

```bash
# Clone
git clone https://github.com/HeidSXY/heidegger.git
cd heidegger

# Create virtual environment
python3.10 -m venv .venv310
source .venv310/bin/activate

# Build & install (compiles Rust → Python extension)
pip install maturin
maturin develop

# Install additional dependencies for demos
pip install mujoco numpy imageio imageio-ffmpeg Pillow
```

### Basic Usage

```python
from heidegger import SafetyShim, CollisionGuard
import json

# Load joint configuration
config = json.dumps({
    "joints": [
        {"name": "base_rotation", "min": -1.57, "max": 1.57, "max_velocity": 4.0},
        {"name": "shoulder",      "min": -1.57, "max": 1.57, "max_velocity": 3.0},
        {"name": "elbow",         "min": -2.36, "max": 0.0,  "max_velocity": 4.0},
        {"name": "wrist_flex",    "min": -1.75, "max": 1.75, "max_velocity": 5.0},
        {"name": "wrist_roll",    "min": -1.57, "max": 1.57, "max_velocity": 5.0},
        {"name": "gripper",       "min": 0.0,   "max": 0.8,  "max_velocity": 8.0}
    ]
})

# Initialize
shim = SafetyShim(config, dt=0.02)  # 50Hz control loop
guard = CollisionGuard(safety_margin=0.015)  # 15mm safety margin

# In your control loop:
def safe_step(vla_action, current_position):
    # Layer 1 & 2: Position clamping + velocity limiting
    result = shim.check(vla_action, current_position)
    safe_action = result["safe_action"]
    
    # Layer 3: Self-collision detection
    if guard.has_collision(safe_action):
        return current_position  # Reject dangerous pose
    
    return safe_action
```

## Interactive Simulator

A web-based tool for visualizing the safety layer in real-time.

```bash
# Install web dependencies
pip install fastapi uvicorn websockets Pillow

# Launch
python tools/server.py
# Open http://localhost:8000
```

**Features:**
- 🎚️ **6 joint sliders** — drag to control the robot in real-time
- 🔴🟢 **Side-by-side view** — raw commands (left) vs. Heidegger-filtered (right)
- ⚠️ **Safety status panel** — real-time collision and clamping feedback
- 🎯 **Preset bad cases** — one-click dangerous scenarios (table penetration, reverse joints, self-collision)
- 🎲 **VLA noise injection** — simulate VLA hallucinations

## Performance

Benchmarked on Apple M1:

| Operation | Latency |
|:----------|:--------|
| Position clamping (6 joints) | 0.8 μs |
| Velocity limiting (6 joints) | 1.2 μs |
| Forward kinematics (7 frames) | 3.5 μs |
| Self-collision check (10 pairs) | 20.5 μs |
| **Full pipeline** | **~26 μs** |

## Supported Robots

Currently supports:
- **SO-ARM101** (6-DOF desktop arm)

The architecture is designed to be robot-agnostic. Adding a new robot requires:
1. A joint configuration JSON file
2. A kinematic chain definition (DH parameters or transform matrices)
3. Capsule approximations for collision geometry

## Roadmap

- [x] Position clamping
- [x] Velocity limiting
- [x] Self-collision detection
- [x] Interactive web simulator
- [ ] Workspace boundary checking (table/floor penetration prevention)
- [ ] Environment collision detection
- [ ] Real robot deployment integration
- [ ] ROS2 node wrapper
- [ ] Multi-robot support

## License

[MIT](./LICENSE)

## Contributing

Contributions are welcome! Please open an issue or submit a PR.

## Acknowledgments

- [MuJoCo](https://mujoco.org/) — Physics simulation
- [PyO3](https://pyo3.rs/) — Rust-Python bindings
- [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) — Robot arm hardware
