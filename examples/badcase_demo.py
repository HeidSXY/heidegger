"""
Heidegger Bad Case 展示
========================

展示 4 个具体的危险场景（无碰撞物理，穿透可见）：
  Scene 1: 手臂穿过桌面 — VLA 指向桌子下方的目标
  Scene 2: 反关节 — 肘部向错误方向弯折
  Scene 3: 自碰撞 — 夹爪折叠撞入底座
  Scene 4: 极端旋转 — base 疯狂旋转 + 手臂伸展

每个场景从正常姿态缓慢过渡到危险姿态，停留 3 秒。
然后并排展示：有 Heidegger 时同样的指令被安全处理。

Usage:
    source .venv310/bin/activate
    python examples/badcase_demo.py
"""

import mujoco
import numpy as np
import os
import imageio

from heidegger import SafetyShim, CollisionGuard


# 无碰撞模型 — 手臂可以穿过桌面和自己
GHOST_MODEL_XML = """
<mujoco model="so_arm101_ghost">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>
  <option gravity="0 0 -9.81" timestep="0.002">
    <flag contact="disable"/>
  </option>
  
  <default>
    <joint damping="0.3" armature="0.05"/>
    <geom type="capsule" friction="1 0.005 0.001"/>
    <position kp="150" kv="3"/>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.75 0.75 0.75" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="arm_dark" rgba="0.6 0.15 0.1 1"/>
    <material name="arm_red" rgba="0.85 0.2 0.15 1"/>
    <material name="arm_orange" rgba="0.9 0.4 0.1 1"/>
    <material name="gripper_mat" rgba="0.7 0.25 0.2 1"/>
  </asset>
  
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="grid_mat"/>
    
    <!-- 半透明桌面 让穿透更明显 -->
    <body name="table" pos="0 0 0.0">
      <geom name="table_top" type="box" size="0.3 0.3 0.005" rgba="0.5 0.4 0.3 0.7" mass="10"/>
    </body>
    
    <body name="target_cup" pos="0.18 0.08 0.04">
      <geom name="cup_body" type="cylinder" size="0.022 0.035" rgba="0.9 0.2 0.2 0.8" mass="0.05"/>
    </body>
    
    <body name="base_mount" pos="0 0 0.005">
      <geom name="base_plate" type="cylinder" size="0.04 0.015" material="arm_dark" mass="0.2"/>
      <body name="base_link" pos="0 0 0.015">
        <joint name="base_rotation" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
        <geom name="base_body" type="cylinder" size="0.03 0.02" material="arm_red" mass="0.15"/>
        <body name="shoulder_link" pos="0 0 0.02">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="shoulder_body" type="capsule" fromto="0 0 0 0 0 0.095" size="0.018" material="arm_dark" mass="0.12"/>
          <body name="elbow_link" pos="0 0 0.095">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom name="elbow_body" type="capsule" fromto="0 0 0 0 0 0.095" size="0.015" material="arm_red" mass="0.10"/>
            <body name="wrist_flex_link" pos="0 0 0.095">
              <joint name="wrist_flex" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom name="wrist_flex_body" type="capsule" fromto="0 0 0 0 0 0.05" size="0.012" material="arm_dark" mass="0.06"/>
              <body name="wrist_roll_link" pos="0 0 0.05">
                <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom name="wrist_roll_body" type="cylinder" size="0.015 0.008" material="arm_orange" mass="0.04"/>
                <body name="gripper_base" pos="0 0 0.008">
                  <joint name="gripper" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                  <geom name="finger_left" type="box" size="0.004 0.003 0.025" pos="0.008 0 0.025" material="gripper_mat" mass="0.02"/>
                  <geom name="finger_right" type="box" size="0.004 0.003 0.025" pos="-0.008 0 0.025" material="gripper_mat" mass="0.02"/>
                  <site name="end_effector" pos="0 0 0.05" size="0.005" rgba="1 0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <light pos="0.3 0.3 0.8" dir="-0.3 -0.3 -0.8" diffuse="0.9 0.9 0.9"/>
    <light pos="-0.3 -0.3 0.8" dir="0.3 0.3 -0.8" diffuse="0.5 0.5 0.5"/>
  </worldbody>
  
  <actuator>
    <position name="a0" joint="base_rotation" ctrlrange="-6.28 6.28" kp="150" forcerange="-8 8"/>
    <position name="a1" joint="shoulder"      ctrlrange="-3.14 3.14" kp="250" forcerange="-8 8"/>
    <position name="a2" joint="elbow"         ctrlrange="-3.14 3.14" kp="200" forcerange="-8 8"/>
    <position name="a3" joint="wrist_flex"    ctrlrange="-3.14 3.14" kp="120" forcerange="-5 5"/>
    <position name="a4" joint="wrist_roll"    ctrlrange="-6.28 6.28" kp="100" forcerange="-5 5"/>
    <position name="a5" joint="gripper"       ctrlrange="-3.14 3.14" kp="60"  forcerange="-3 3"/>
  </actuator>

  <sensor>
    <framepos name="ee_pos" objtype="site" objname="end_effector"/>
  </sensor>
</mujoco>
"""


def create_ghost_sim():
    m = mujoco.MjModel.from_xml_string(GHOST_MODEL_XML)
    return m, mujoco.MjData(m)


def create_safe_sim():
    path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101.xml")
    m = mujoco.MjModel.from_xml_path(path)
    return m, mujoco.MjData(m)


def lerp_pose(start, end, t):
    """线性插值两个姿态"""
    t = max(0.0, min(1.0, t))
    return start + (end - start) * t


def generate_badcase_sequence():
    """
    生成 bad case 序列：
    每个场景 = 过渡（2秒）+ 停留（3秒）+ 回零（1秒）
    """
    fps = 30
    
    zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # 定义每个 bad case 的目标姿态
    scenes = [
        {
            "name": "穿桌: 手臂直接插入桌面",
            "pose": np.array([0.3, 1.8, -0.3, 0.5, 0.0, 0.3]),
            # shoulder 猛往前+下，elbow 接近直，手臂会穿过桌面
            "transition_s": 2.5,
            "hold_s": 3.0,
            "camera": {"azimuth": 160, "elevation": -15, "distance": 0.50},
        },
        {
            "name": "反关节: 肘部反向弯折",
            "pose": np.array([0.0, 0.5, 1.8, -0.5, 0.0, 0.3]),
            # elbow 正值 = 反向弯折（正常范围是 -2.36 到 0）
            "transition_s": 2.5,
            "hold_s": 3.0,
            "camera": {"azimuth": 120, "elevation": -20, "distance": 0.50},
        },
        {
            "name": "自碰撞: 夹爪折入底座",
            "pose": np.array([0.0, 1.0, -2.3, -1.5, 0.0, 0.3]),
            # shoulder forward + elbow tight + wrist back = gripper into base
            "transition_s": 2.5,
            "hold_s": 3.0,
            "camera": {"azimuth": 140, "elevation": -25, "distance": 0.45},
        },
        {
            "name": "疯狂旋转: 伸展手臂 + base 猛转",
            "pose": np.array([5.0, 0.8, -0.5, 0.0, 3.0, 0.5]),
            # base 转 5 弧度 ≈ 快 1 圈，手臂伸展
            "transition_s": 2.0,
            "hold_s": 3.0,
            "camera": {"azimuth": 145, "elevation": -30, "distance": 0.55},
        },
    ]
    
    all_actions = []
    scene_info = []
    
    for scene in scenes:
        pose = scene["pose"]
        t_trans = int(scene["transition_s"] * fps)
        t_hold = int(scene["hold_s"] * fps)
        t_return = int(1.0 * fps)
        
        start_frame = len(all_actions)
        
        # 过渡: 零位 → 危险姿态
        for i in range(t_trans):
            t = i / t_trans
            # 缓入缓出曲线
            t_smooth = t * t * (3 - 2 * t)
            all_actions.append(lerp_pose(zero, pose, t_smooth))
        
        # 停留在危险姿态
        for _ in range(t_hold):
            all_actions.append(pose.copy())
        
        # 快速回到零位
        for i in range(t_return):
            t = i / t_return
            t_smooth = t * t * (3 - 2 * t)
            all_actions.append(lerp_pose(pose, zero, t_smooth))
        
        # 短暂停顿
        for _ in range(int(0.5 * fps)):
            all_actions.append(zero.copy())
        
        scene_info.append({
            "name": scene["name"],
            "start": start_frame,
            "end": len(all_actions),
            "camera": scene["camera"],
        })
    
    return np.array(all_actions), scene_info


def full_safety_filter(actions, shim, guard):
    n = len(actions)
    safe = np.zeros_like(actions)
    cur = np.zeros(6)
    for i in range(n):
        res = shim.check(actions[i].tolist(), cur.tolist())
        clamped = np.array(res["safe_action"])
        if guard.has_collision(clamped.tolist()):
            safe[i] = cur
        else:
            safe[i] = clamped
            cur = clamped
    return safe


def render_comparison(model_ghost, data_ghost, model_safe, data_safe,
                      actions_raw, actions_safe, path, fps=30):
    width, height = 640, 480
    
    mujoco.mj_resetData(model_ghost, data_ghost)
    mujoco.mj_resetData(model_safe, data_safe)
    mujoco.mj_forward(model_ghost, data_ghost)
    mujoco.mj_forward(model_safe, data_safe)

    rl = mujoco.Renderer(model_ghost, width=width, height=height)
    rr = mujoco.Renderer(model_safe, width=width, height=height)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.50
    cam.azimuth = 145
    cam.elevation = -20
    cam.lookat[:] = [0.0, 0.0, 0.08]
    opt = mujoco.MjvOption()

    n = len(actions_raw)
    frames = []

    for i in range(n):
        data_ghost.ctrl[:6] = actions_raw[i]
        data_safe.ctrl[:6] = actions_safe[i]
        
        for _ in range(10):
            mujoco.mj_step(model_ghost, data_ghost)
            mujoco.mj_step(model_safe, data_safe)

        rl.update_scene(data_ghost, cam, opt)
        rr.update_scene(data_safe, cam, opt)
        fl = rl.render().copy()
        fr = rr.render().copy()

        combined = np.concatenate([fl, fr], axis=1)
        # 分割线
        combined[:, width-1:width+1, :] = 255
        # 红色标记（左）
        combined[8:28, 8:28, :] = [220, 50, 50]
        # 绿色标记（右）
        combined[8:28, width+8:width+28, :] = [50, 220, 50]
        
        frames.append(combined)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n} frames...")

    rl.close()
    rr.close()

    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
    duration = len(frames) / fps
    print(f"  ✅ {os.path.basename(path)} ({duration:.1f}s, {len(frames)} frames)")


def render_unsafe_only(model, data, actions, path, fps=30):
    """只渲染 unsafe 版本的特写"""
    width, height = 640, 480
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.45
    cam.azimuth = 145
    cam.elevation = -20
    cam.lookat[:] = [0.0, 0.0, 0.08]
    opt = mujoco.MjvOption()

    frames = []
    for i in range(len(actions)):
        data.ctrl[:6] = actions[i]
        for _ in range(10):
            mujoco.mj_step(model, data)

        renderer.update_scene(data, cam, opt)
        frame = renderer.render().copy()
        frames.append(frame)

    renderer.close()

    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  ✅ {os.path.basename(path)} ({len(frames)/fps:.1f}s)")


def main():
    cfg = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101_joints.json")
    with open(cfg) as f:
        config_json = f.read()

    shim = SafetyShim(config_json, dt=0.02)
    guard = CollisionGuard(safety_margin=0.015)

    out = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  HEIDEGGER BAD CASE 展示")
    print("  4 个具体危险场景，每个场景停留 3 秒")
    print("=" * 60)

    # 生成 bad case 序列
    raw_actions, scenes = generate_badcase_sequence()
    safe_actions = full_safety_filter(raw_actions, shim, guard)

    print(f"\n📊 {len(raw_actions)} 帧 ({len(raw_actions)/30:.1f}s)\n")
    for s in scenes:
        print(f"  🔴 {s['name']}")

    # 渲染 unsafe 特写
    print(f"\n🔴 渲染 bad case 特写...")
    m_ghost, d_ghost = create_ghost_sim()
    render_unsafe_only(m_ghost, d_ghost, raw_actions,
                       os.path.join(out, "badcases.mp4"))

    # 渲染并排对比
    print(f"\n🎬 渲染并排对比...")
    m_ghost2, d_ghost2 = create_ghost_sim()
    m_safe, d_safe = create_safe_sim()
    render_comparison(m_ghost2, d_ghost2, m_safe, d_safe,
                      raw_actions, safe_actions,
                      os.path.join(out, "badcase_comparison.mp4"))

    print(f"\n{'='*60}")
    print(f"  完成！")
    print(f"  open results/badcases.mp4            ← 纯 bad case 特写")
    print(f"  open results/badcase_comparison.mp4  ← 并排对比")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
