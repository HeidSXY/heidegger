<p align="center">
  <h1 align="center">🛡️ Heidegger</h1>
  <p align="center">
    <strong>VLA 驱动机械臂的确定性安全层</strong>
  </p>
  <p align="center">
    <a href="./README.md">English</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#cbf-安全过滤器">CBF 安全过滤器</a> ·
    <a href="#lerobot-集成">LeRobot 集成</a> ·
    <a href="#交互式仿真器">交互式仿真器</a>
  </p>
</p>

---

## 问题

VLA（视觉-语言-动作）模型正在变革机器人操作——但它们会「幻觉」。一个错误的预测就可能让机械臂撞向自己、穿过桌面、或把关节弯到物理极限之外。

当前开源 VLA 生态（如 [LeRobot](https://github.com/huggingface/lerobot)）的安全手段基本是**手动**的：设定 `torque_limit` 参数、靠领导臂的物理限位「顺便」约束、或者——人站在旁边准备拔插头。

**Heidegger** 是一个 Rust 驱动的确定性安全层，位于 VLA 模型和机器人执行器之间。无论 AI 的预测多离谱，它都能以 ~26μs 的开销实时强制执行物理约束。

> *命名来自海德格尔的「上手状态」(Zuhandenheit) 概念——最好的工具是你注意不到的工具。*

## 工作原理

```
VLA 模型输出 → [ Heidegger 安全层 ] → 安全的执行器指令
                       │
              ┌────────┤────────┐
              ▼        ▼        ▼
          逐关节     CBF QP    学习的
          钳制      优化器    工作空间
              │        │        │
              └────────┤────────┘
                       ▼
                  安全输出 ✅
```

### 两种安全模式

| 模式 | 方式 | 适用场景 |
|:-----|:-----|:---------|
| **SafetyShim**（经典） | 逐关节独立位置+速度钳制 | 简单、可预测、零配置 |
| **CBF 安全过滤器** ✨ | 全局最优 QP，联合约束关节/速度/碰撞/工作空间 | 最大安全性，最小干预 |

### 五层保护

| 层级 | 功能 | 延迟 |
|:-----|:-----|:-----|
| **L1 — 位置钳制** | 强制每个关节角度在合法范围内 | ~1 μs |
| **L2 — 速度限制** | 限制关节转速，防止突变 | ~1 μs |
| **L3 — 自碰撞检测** | 正运动学 + 胶囊体碰撞，拒绝危险姿态 | ~20 μs |
| **L4 — 工作空间边界** | 防止机械臂穿透桌面/地面 | ~3 μs |
| **L5 — 学习安全集** ✨ | 从示教轨迹学习的数据驱动工作空间边界 | ~5 μs |

**全流程延迟：~26 μs**（经典模式）/ **~75 μs**（CBF 模式）— 对 Python 控制循环完全透明。

## 快速开始

### 环境要求

- **Rust** 工具链 (1.70+)
- **Python** 3.9+
- **maturin** (`pip install maturin`)

### 安装

```bash
git clone https://github.com/HeidSXY/heidegger.git
cd heidegger

python3.10 -m venv .venv
source .venv/bin/activate

pip install maturin
maturin develop

# 仿真器依赖
pip install mujoco numpy fastapi uvicorn websockets Pillow
```

### 基本用法（经典模式）

```python
from heidegger import SafetyShim, CollisionGuard

with open("models/so_arm101_joints.json") as f:
    config = f.read()

shim = SafetyShim(config, dt=0.02)           # 50Hz 控制频率
guard = CollisionGuard(safety_margin=0.015)   # 15mm 安全余量

def safe_step(vla_action, current_position):
    result = shim.check(vla_action, current_position)
    safe_action = result["safe_action"]
    if guard.has_collision(safe_action):
        return current_position
    return safe_action
```

## CBF 安全过滤器

**控制屏障函数 (Control Barrier Function)** 模式用全局最优安全动作计算替代逐关节钳制。它求解二次规划（QP），找到**最接近 VLA 输出的可行动作**：

```
u* = argmin ‖u - u_vla‖²
     s.t.   关节位置限制
            速度限制
            自碰撞安全距离
            学习的工作空间边界
```

### 为什么 CBF 优于钳制？

| | SafetyShim（钳制） | CBF 过滤器 |
|---|---|---|
| **方式** | 逐关节独立 | 全局 QP 优化 |
| **最优性** | 贪心（可能扭曲轨迹） | 最小范数修正 |
| **约束** | 仅位置+速度 | + 碰撞 + 学习工作空间 |
| **数据驱动** | ❌ | ✅ 学习安全集 |
| **形式化保证** | ❌ | ✅ CBF 不变性 |

### 使用方法

```python
from heidegger import PyCBFSafetyFilter
import json

with open("models/so_arm101_joints.json") as f:
    config = f.read()

# 基本 CBF（关节 + 速度 + 碰撞约束）
cbf = PyCBFSafetyFilter(config, dt=0.02, robot_model="so_arm101")
result = cbf.filter(vla_action, current_position)
safe_action = result["safe_action"]

# 加入学习的工作空间边界
with open("safety_set.json") as f:
    ss_json = f.read()
cbf = PyCBFSafetyFilter(config, dt=0.02, robot_model="so_arm101", safety_set_json=ss_json)
```

### 学习安全边界

从人类示教中学习安全工作空间——机器人只在你示范过的安全区域内运动：

```python
from heidegger import TrajectoryRecorder

# 1. 遥操作时录制
recorder = TrajectoryRecorder(num_joints=6)
for joint_angles in teleoperation_stream:
    recorder.record(joint_angles)

# 2. 校准（逐关节统计边界 + PCA 凸包）
safety_set = recorder.calibrate(sigma_multiplier=3.0, pca_dims=3)

# 3. 保存用于部署
with open("safety_set.json", "w") as f:
    f.write(safety_set.to_json())
```

CLI 工具：

```bash
python -m heidegger.calibrate -t recordings.json -o safety_set.json --sigma 3.0 --pca-dims 3 -v
```

## LeRobot 集成

Heidegger 提供 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 策略的即插即用封装：

### 经典模式

```python
from heidegger.lerobot import SafetyWrapper

safe_policy = SafetyWrapper(
    policy=your_lerobot_policy,
    robot_model="so_arm101",
)
action = safe_policy.select_action(observation)
```

### CBF 模式（推荐）

```python
from heidegger.lerobot import CBFSafetyWrapper

with open("safety_set.json") as f:
    ss_json = f.read()

safe_policy = CBFSafetyWrapper(
    policy=your_lerobot_policy,
    robot_model="so_arm101",
    safety_set_json=ss_json,
)
action = safe_policy.select_action(observation)
print(safe_policy.stats)  # {"total_steps": 100, "interventions": 3, "intervention_rate": 0.03}
```

## 交互式仿真器

基于 Web 的实时安全层可视化工具。

```bash
python tools/server.py
# 浏览器打开 http://localhost:8000
```

**功能：**
- 🎚️ **6 个关节滑条** — 拖动实时控制机械臂
- 🔴🟢 **双视口对比** — 原始指令（上）vs Heidegger 过滤后（下）
- ⚠️ **4 层安全指示灯** — CLAMP / VEL / COL / BND 实时状态
- 🎯 **预设危险场景** — 一键演示穿桌、自碰撞
- 🎲 **VLA 噪声注入** — 模拟 VLA 幻觉，可调强度

## 性能

Apple M1 基准测试:

| 操作 | 延迟 |
|:-----|:-----|
| 位置钳制 (6 关节) | 0.8 μs |
| 速度限制 (6 关节) | 1.2 μs |
| 正运动学 (7 坐标系) | 3.5 μs |
| 自碰撞检测 (10 对碰撞体) | 20.5 μs |
| CBF QP 求解 (6-DOF, 25+ 约束) | ~50 μs |
| **全流程** | **~26 μs**（经典）/ **~75 μs**（CBF） |

## 项目结构

```
heidegger/
├── heidegger-core/          # 纯 Rust 安全逻辑（除 serde 零外部依赖）
│   └── src/
│       ├── lib.rs           # SafetyShim: 位置钳制 + 速度限制
│       ├── kinematics.rs    # 正运动学（齐次变换矩阵）
│       ├── collision.rs     # 胶囊体自碰撞检测
│       ├── safety_set.rs    # 学习安全边界（PCA + 凸包）
│       └── cbf.rs           # CBF 安全过滤器（投影梯度下降 QP）
├── heidegger-py/            # PyO3 Python 绑定
│   └── src/lib.rs           # 5 个 Python 类
├── python/heidegger/        # Python 包
│   ├── __init__.py          # 公共 API（8 个导出）
│   ├── lerobot.py           # LeRobot 封装（SafetyWrapper + CBFSafetyWrapper）
│   └── calibrate.py         # CLI: 从轨迹数据学习安全边界
├── models/                  # 机器人配置
│   ├── so_arm101_joints.json
│   ├── koch_v1_1_joints.json
│   └── so_arm100_joints.json
├── tools/                   # 交互式仿真器
│   ├── server.py            # FastAPI + WebSocket + MuJoCo
│   └── static/index.html    # 双视口 Web UI
└── examples/                # 示例脚本
```

## 支持的机器人

| 机器人 | 位置限制 | 速度限制 | FK | 碰撞 | CBF |
|:-------|:---:|:---:|:---:|:---:|:---:|
| **SO-ARM101** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Koch v1.1** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **SO-ARM100** | ✅ | ✅ | ✅ | ✅ | ✅ |

## 为什么用 Rust？

| | Python | C++ | **Rust** |
|:--|:--|:--|:--|
| GC 暂停 | 50-200ms | 无 | **无** |
| 内存安全 | 运行时异常 | 段错误 | **编译期保证** |
| 控制延迟 | 1-10ms | 0.01-0.1ms | **0.01-0.05ms** |
| Python 互操作 | 原生 | pybind11 | **PyO3 (零拷贝)** |

安全层绝不能卡顿、崩溃、或引入延迟尖峰。Rust 是唯一能同时保证这三点的现代语言。

## 路线图

- [x] 位置钳制 (L1)
- [x] 速度限制 (L2)
- [x] 自碰撞检测 (L3)
- [x] 工作空间边界检查 (L4)
- [x] 学习安全边界 (L5) — PCA + 凸包
- [x] CBF 安全过滤器 — 投影梯度下降 QP
- [x] 全机型支持（SO-ARM101, Koch v1.1, SO-ARM100）
- [x] 交互式 Web 仿真器（双视口）
- [x] LeRobot 集成（SafetyWrapper + CBFSafetyWrapper）
- [x] 校准 CLI（`python -m heidegger.calibrate`）
- [x] Shadow 模式（观察违规但不干预）
- [ ] 基准测试报告与对比数据
- [ ] MuJoCo 仿真：钳制 vs CBF 可视化对比
- [ ] 动作异常检测（跳变/振荡/停滞）
- [ ] 黑匣子事件记录器

## 相关工作

- [AEGIS/VLSA](https://arxiv.org/abs/2412.12267) — 基于 CBF 的安全层（Python，仅障碍物避碰）
- [SafeVLA](https://arxiv.org/abs/2503.14729) — VLA 约束学习对齐（需重训模型）
- [dora-rs](https://github.com/dora-rs/dora) — Rust 机器人数据流框架（通信层，非安全层）

Heidegger 的不同之处：Rust 原生（非 Python）、覆盖完整安全栈（不仅碰撞）、CBF 优化（非逐关节钳制）、数据驱动（学习边界）、即插即用（无需重训模型）。

## 开源协议

[MIT](./LICENSE)

## 参与贡献

欢迎提交 Issue 和 PR！

## 致谢

- [MuJoCo](https://mujoco.org/) — 物理仿真引擎
- [PyO3](https://pyo3.rs/) — Rust-Python 绑定
- [HuggingFace LeRobot](https://github.com/huggingface/lerobot) — 我们服务的 VLA 生态
- [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) — 机械臂硬件
