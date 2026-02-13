<p align="center">
  <h1 align="center">🛡️ Heidegger</h1>
  <p align="center">
    <strong>VLA 驱动机械臂的确定性安全层</strong>
  </p>
  <p align="center">
    <a href="./README.md">English</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#交互式仿真器">交互式仿真器</a> ·
    <a href="./docs/Project_Heidegger_Whitepaper_v1.md">白皮书</a>
  </p>
</p>

---

## 问题

VLA（视觉-语言-动作）模型正在变革机器人操作——但它们会"幻觉"。一个错误的预测就可能让机械臂撞向自己、穿过桌面、或把关节弯到物理极限之外。

**Heidegger** 是一个确定性安全层，位于 VLA 模型和机器人执行器之间。无论 AI 的预测多离谱，它都能实时强制执行物理约束。

## 工作原理

```
VLA 模型输出 → [ Heidegger 安全层 ] → 安全的执行器指令
                       │
                  ┌────┴────┐
                  ▼         ▼
              位置钳制    速度限制
                  │         │
                  └────┬────┘
                       ▼
                  自碰撞检测
                       │
                       ▼
                  安全输出 ✅
```

### 三层保护

| 层级 | 功能 | 延迟 |
|:-----|:-----|:-----|
| **位置钳制** | 强制每个关节角度在合法范围内（JSON 可配置） | ~1 μs |
| **速度限制** | 限制关节转速，防止突变 | ~1 μs |
| **自碰撞检测** | 正运动学 + 胶囊体碰撞，拒绝危险姿态 | ~20 μs |

**全流程延迟：~26 μs** — 足够 1kHz 实时控制回路。

## 项目结构

```
heidegger/
├── heidegger-core/          # 纯 Rust 安全逻辑
│   └── src/
│       ├── lib.rs           # SafetyGuard: 位置钳制 & 速度限制
│       ├── kinematics.rs    # 正运动学（齐次变换矩阵）
│       └── collision.rs     # 胶囊体自碰撞检测
├── heidegger-py/            # PyO3 Python 绑定
│   └── src/lib.rs           # SafetyShim + CollisionGuard
├── python/heidegger/        # Python 包
├── models/                  # 机器人定义
│   ├── so_arm101.xml        # SO-ARM101 MuJoCo 模型
│   └── so_arm101_joints.json # 关节配置
├── tools/                   # 交互式仿真器
│   ├── server.py            # FastAPI + WebSocket 后端
│   └── static/index.html    # Web UI
└── examples/                # 示例脚本
```

## 快速开始

### 环境要求

- **Rust** 工具链 (1.70+)
- **Python** 3.9+
- **maturin** (`pip install maturin`)

### 安装

```bash
# 克隆
git clone https://github.com/HeidSXY/heidegger.git
cd heidegger

# 创建虚拟环境
python3.10 -m venv .venv310
source .venv310/bin/activate

# 编译安装（Rust → Python 扩展）
pip install maturin
maturin develop

# 安装演示依赖
pip install mujoco numpy imageio imageio-ffmpeg Pillow
```

### 基本用法

```python
from heidegger import SafetyShim, CollisionGuard
import json

# 加载关节配置
with open("models/so_arm101_joints.json") as f:
    config = f.read()

# 初始化
shim = SafetyShim(config, dt=0.02)        # 50Hz 控制频率
guard = CollisionGuard(safety_margin=0.015) # 15mm 安全余量

# 在控制循环中使用:
def safe_step(vla_action, current_position):
    # 第1、2层: 位置钳制 + 速度限制
    result = shim.check(vla_action, current_position)
    safe_action = result["safe_action"]
    
    # 第3层: 自碰撞检测
    if guard.has_collision(safe_action):
        return current_position  # 拒绝危险姿态
    
    return safe_action
```

## 交互式仿真器

基于 Web 的实时安全层可视化工具。

```bash
# 安装 Web 依赖
pip install fastapi uvicorn websockets Pillow

# 启动
python tools/server.py
# 浏览器打开 http://localhost:8000
```

**功能：**
- 🎚️ **6 个关节滑条** — 拖动实时控制机械臂
- 🔴🟢 **左右对比** — 原始指令（左）vs Heidegger 过滤后（右）
- ⚠️ **安全状态面板** — 实时显示碰撞检测和钳制详情
- 🎯 **预设危险场景** — 一键演示穿桌、反关节、自碰撞
- 🎲 **VLA 噪声注入** — 模拟 VLA 幻觉

## 性能

Apple M1 基准测试:

| 操作 | 延迟 |
|:-----|:-----|
| 位置钳制 (6 关节) | 0.8 μs |
| 速度限制 (6 关节) | 1.2 μs |
| 正运动学 (7 坐标系) | 3.5 μs |
| 自碰撞检测 (10 对碰撞体) | 20.5 μs |
| **全流程** | **~26 μs** |

## 支持的机器人

当前支持:
- **SO-ARM101** (6 自由度桌面机械臂)

架构设计为机器人无关的。添加新机器人只需:
1. 关节配置 JSON 文件
2. 运动学链定义（DH 参数或变换矩阵）
3. 碰撞几何的胶囊体近似

## 路线图

- [x] 位置钳制
- [x] 速度限制
- [x] 自碰撞检测
- [x] 交互式 Web 仿真器
- [ ] 工作空间边界检查（桌面/地面穿透防护）
- [ ] 环境碰撞检测
- [ ] 实体机器人部署集成
- [ ] ROS2 节点封装
- [ ] 多机器人支持

## 开源协议

[MIT](./LICENSE)

## 参与贡献

欢迎提交 Issue 和 PR！

## 致谢

- [MuJoCo](https://mujoco.org/) — 物理仿真引擎
- [PyO3](https://pyo3.rs/) — Rust-Python 绑定
- [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) — 机械臂硬件
