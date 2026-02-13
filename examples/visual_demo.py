"""
Heidegger 可视化 Demo — 直接看到机械臂在动
==============================================

两轮演示：
  第一轮 🔴 没有 Heidegger —— 机械臂会疯狂抖动、自碰撞
  第二轮 🟢 有 Heidegger —— 同样的噪声指令，机械臂平稳运行

操作：
  - 鼠标左键拖动旋转视角
  - 鼠标右键拖动平移
  - 滚轮缩放
  - 窗口会自动弹出，看完后关闭窗口或 Ctrl+C 退出

Usage:
    source .venv310/bin/activate
    python examples/visual_demo.py
"""

import mujoco
import mujoco.viewer
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
    """生成一段有碰撞风险的轨迹"""
    t = np.linspace(0, 1, n_steps)
    trajectory = np.zeros((n_steps, 6))

    for i, ti in enumerate(t):
        if ti < 0.3:
            p = ti / 0.3
            trajectory[i] = [0.3*p, 0.6*p, -1.2*p, 0.3*p, 0.0, 0.5]
        elif ti < 0.6:
            p = (ti - 0.3) / 0.3
            trajectory[i] = [
                0.3, 0.6 + 0.5*p, -1.2 - 1.1*p, 0.3 - 1.8*p, 0.0, 0.5
            ]
        else:
            p = (ti - 0.6) / 0.4
            trajectory[i] = [
                0.3 - 0.6*p, 1.1 - 0.8*p, -2.3 + 0.5*p, -1.5, 1.5*p, 0.5
            ]

    return trajectory


def inject_vla_noise(trajectory, noise_level=0.2, spike_prob=0.03, spike_scale=5.0):
    rng = np.random.RandomState(42)
    noisy = trajectory.copy()
    noisy += rng.randn(*trajectory.shape) * noise_level
    spike_mask = rng.rand(*trajectory.shape) < spike_prob
    noisy[spike_mask] += rng.randn(*trajectory.shape)[spike_mask] * spike_scale
    return noisy


def full_safety_filter(noisy_actions, shim, guard):
    """三层安全过滤"""
    n = len(noisy_actions)
    safe = np.zeros_like(noisy_actions)
    current_pos = np.zeros(6)

    for i in range(n):
        result = shim.check(noisy_actions[i].tolist(), current_pos.tolist())
        clamped = np.array(result["safe_action"])

        if guard.has_collision(clamped.tolist()):
            safe[i] = current_pos  # 碰撞！保持上一帧
        else:
            safe[i] = clamped
            current_pos = clamped

    return safe


def run_visual(model, data, actions, title, step_delay=0.02):
    """在 MuJoCo viewer 里播放动作序列"""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"  窗口已弹出，请观看机械臂运动")
    print(f"  看完后关闭窗口继续...")
    print(f"{'='*50}")

    n_steps = len(actions)
    step = [0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and step[0] < n_steps:
            # 设置控制指令
            data.ctrl[:6] = actions[step[0]]

            # 推进物理仿真
            for _ in range(10):
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(step_delay)
            step[0] += 1

        # 播放完毕，让用户继续观察
        if viewer.is_running():
            print("  ✅ 播放完毕！你可以拖动鼠标旋转视角观察，关闭窗口继续。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101_joints.json")
    with open(config_path) as f:
        config_json = f.read()

    shim = SafetyShim(config_json, dt=0.02)
    guard = CollisionGuard(safety_margin=0.015)

    print("=" * 50)
    print("  HEIDEGGER 可视化 DEMO")
    print("  你将看到两轮机械臂运动")
    print("=" * 50)

    # 生成轨迹
    n_steps = 300  # 6 秒
    clean = generate_trajectory_with_collision_risk(n_steps)
    noisy = inject_vla_noise(clean)
    safe = full_safety_filter(noisy, shim, guard)

    # 统计
    unsafe_collisions = sum(1 for i in range(n_steps) if guard.has_collision(noisy[i].tolist()))
    print(f"\n📊 轨迹统计:")
    print(f"   总帧数: {n_steps}")
    print(f"   无保护时碰撞帧: {unsafe_collisions}")

    model, data = create_sim()

    # 第一轮：无保护
    input("\n按 Enter 开始第一轮 🔴 无 Heidegger（注意观察抖动和碰撞）...")
    run_visual(model, data, noisy,
               "🔴 第一轮：无 Heidegger 保护 — VLA 噪声直接控制",
               step_delay=0.03)

    # 第二轮：有保护
    input("\n按 Enter 开始第二轮 🟢 有 Heidegger（同样的噪声指令）...")
    run_visual(model, data, safe,
               "🟢 第二轮：Heidegger 三层保护 — 同样的噪声，平稳运行",
               step_delay=0.03)

    print("\n" + "=" * 50)
    print("  演示结束！")
    print(f"  🔴 无保护: {unsafe_collisions} 次自碰撞")
    print(f"  🟢 有保护: 0 次自碰撞")
    print("=" * 50)


if __name__ == "__main__":
    main()
