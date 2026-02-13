"""
Heidegger 交互式安全仿真工具 — 后端
=====================================

FastAPI + WebSocket 后端：
- 接收用户关节角度
- 并行运行两个 MuJoCo 仿真（无限制 vs 安全）
- 离屏渲染双画面
- 推送渲染帧 + 安全状态

Usage:
    source .venv310/bin/activate
    python tools/server.py
    # Open http://localhost:8000
"""

import asyncio
import base64
import io
import json
import os
import time

import mujoco
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from heidegger import SafetyShim, CollisionGuard

app = FastAPI()

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ──── MuJoCo Models ──────────────────────────────────────────────

GHOST_MODEL_XML = """
<mujoco model="so_arm101_ghost">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="480" offheight="360"/>
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
    <texture type="2d" name="grid" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.78 0.78 0.78" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="arm_dark" rgba="0.55 0.12 0.08 1"/>
    <material name="arm_red" rgba="0.82 0.18 0.12 1"/>
    <material name="arm_orange" rgba="0.88 0.38 0.08 1"/>
    <material name="gripper_mat" rgba="0.65 0.22 0.18 1"/>
  </asset>
  
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" material="grid_mat"/>
    <body name="table" pos="0 0 0.0">
      <geom name="table_top" type="box" size="0.3 0.3 0.005" rgba="0.5 0.4 0.3 0.6" mass="10"/>
    </body>
    <body name="target" pos="0.18 0.08 0.04">
      <geom type="cylinder" size="0.022 0.035" rgba="0.9 0.2 0.2 0.7" mass="0.05"/>
    </body>
    
    <body name="base_mount" pos="0 0 0.005">
      <geom name="base_plate" type="cylinder" size="0.04 0.015" material="arm_dark" mass="0.2"/>
      <body name="base_link" pos="0 0 0.015">
        <joint name="j0" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
        <geom type="cylinder" size="0.03 0.02" material="arm_red" mass="0.15"/>
        <body name="shoulder_link" pos="0 0 0.02">
          <joint name="j1" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.095" size="0.018" material="arm_dark" mass="0.12"/>
          <body name="elbow_link" pos="0 0 0.095">
            <joint name="j2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.095" size="0.015" material="arm_red" mass="0.10"/>
            <body name="wrist_flex_link" pos="0 0 0.095">
              <joint name="j3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.012" material="arm_dark" mass="0.06"/>
              <body name="wrist_roll_link" pos="0 0 0.05">
                <joint name="j4" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="cylinder" size="0.015 0.008" material="arm_orange" mass="0.04"/>
                <body name="gripper_base" pos="0 0 0.008">
                  <joint name="j5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                  <geom type="box" size="0.004 0.003 0.025" pos="0.008 0 0.025" material="gripper_mat" mass="0.02"/>
                  <geom type="box" size="0.004 0.003 0.025" pos="-0.008 0 0.025" material="gripper_mat" mass="0.02"/>
                  <site name="ee" pos="0 0 0.05" size="0.005" rgba="1 0 0 1"/>
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
    <position name="a0" joint="j0" ctrlrange="-6.28 6.28" kp="150"/>
    <position name="a1" joint="j1" ctrlrange="-3.14 3.14" kp="250"/>
    <position name="a2" joint="j2" ctrlrange="-3.14 3.14" kp="200"/>
    <position name="a3" joint="j3" ctrlrange="-3.14 3.14" kp="120"/>
    <position name="a4" joint="j4" ctrlrange="-6.28 6.28" kp="100"/>
    <position name="a5" joint="j5" ctrlrange="-3.14 3.14" kp="60"/>
  </actuator>
  <sensor><framepos name="ee_pos" objtype="site" objname="ee"/></sensor>
</mujoco>
"""


class SimPair:
    """Manages two parallel MuJoCo simulations + Heidegger safety."""

    def __init__(self):
        # Ghost model (no contact, no joint limits)
        self.model_ghost = mujoco.MjModel.from_xml_string(GHOST_MODEL_XML)
        self.data_ghost = mujoco.MjData(self.model_ghost)

        # Safe model (from the project's constrained model)
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101.xml")
        self.model_safe = mujoco.MjModel.from_xml_path(model_path)
        self.data_safe = mujoco.MjData(self.model_safe)

        # Renderers
        self.renderer_ghost = mujoco.Renderer(self.model_ghost, width=480, height=360)
        self.renderer_safe = mujoco.Renderer(self.model_safe, width=480, height=360)

        # Camera
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 0.48
        self.camera.azimuth = 145
        self.camera.elevation = -22
        self.camera.lookat[:] = [0.0, 0.0, 0.08]
        self.scene_opt = mujoco.MjvOption()

        # Safety
        config_path = os.path.join(os.path.dirname(__file__), "..", "models", "so_arm101_joints.json")
        with open(config_path) as f:
            config_json = f.read()
        self.shim = SafetyShim(config_json, dt=0.02)
        self.guard = CollisionGuard(safety_margin=0.015)

        # State
        self.current_safe_pos = np.zeros(6)

        # Reset
        mujoco.mj_resetData(self.model_ghost, self.data_ghost)
        mujoco.mj_resetData(self.model_safe, self.data_safe)
        mujoco.mj_forward(self.model_ghost, self.data_ghost)
        mujoco.mj_forward(self.model_safe, self.data_safe)

    def step(self, raw_angles: list[float]) -> dict:
        """Run one step: apply raw angles to ghost, filtered to safe."""
        t0 = time.perf_counter_ns()

        TABLE_Z = 0.012  # table surface height (m)

        # 1. Apply raw to ghost
        self.data_ghost.ctrl[:6] = raw_angles
        for _ in range(5):
            mujoco.mj_step(self.model_ghost, self.data_ghost)

        # 2. Heidegger filter — Layer 1 & 2: position clamping + velocity limiting
        result = self.shim.check(raw_angles, self.current_safe_pos.tolist())
        clamped = np.array(result["safe_action"])
        was_clamped = result["was_clamped"]
        violations = result["violations"]

        # Layer 3: Self-collision check
        has_collision = self.guard.has_collision(clamped.tolist())
        collision_details = []
        if has_collision:
            cols = self.guard.check_collisions(clamped.tolist())
            collision_details = [
                {"a": c["link_a"], "b": c["link_b"], "dist": round(c["distance"] * 1000, 1)}
                for c in cols[:3]
            ]

        # Layer 4: Workspace boundary check (FK z-height)
        # Skip frames 0-1 (base origin + base_link, fixed to table)
        boundary_violation = False
        boundary_min_z = 999.0
        if not has_collision:
            fk_positions = self.guard.forward_kinematics(clamped.tolist())
            movable_z = [p[2] for p in fk_positions[2:]]  # skip base frames
            min_z = min(movable_z) if movable_z else 999.0
            boundary_min_z = round(min_z * 1000, 1)  # mm
            if min_z < TABLE_Z:
                boundary_violation = True

        # Determine safe output
        if has_collision or boundary_violation:
            safe_angles = self.current_safe_pos.copy()
        else:
            safe_angles = clamped.copy()
            self.current_safe_pos = clamped.copy()

        # 3. Apply safe to constrained model
        self.data_safe.ctrl[:6] = safe_angles
        for _ in range(5):
            mujoco.mj_step(self.model_safe, self.data_safe)

        safety_us = (time.perf_counter_ns() - t0) / 1000

        # 4. Render
        self.renderer_ghost.update_scene(self.data_ghost, self.camera, self.scene_opt)
        frame_ghost = self.renderer_ghost.render()

        self.renderer_safe.update_scene(self.data_safe, self.camera, self.scene_opt)
        frame_safe = self.renderer_safe.render()

        # 5. Encode to JPEG
        from PIL import Image
        buf_g = io.BytesIO()
        Image.fromarray(frame_ghost).save(buf_g, format="JPEG", quality=70)
        b64_ghost = base64.b64encode(buf_g.getvalue()).decode()

        buf_s = io.BytesIO()
        Image.fromarray(frame_safe).save(buf_s, format="JPEG", quality=70)
        b64_safe = base64.b64encode(buf_s.getvalue()).decode()

        return {
            "ghost": b64_ghost,
            "safe": b64_safe,
            "was_clamped": was_clamped,
            "has_collision": has_collision,
            "collisions": collision_details,
            "boundary_violation": boundary_violation,
            "boundary_min_z": boundary_min_z,
            "violations": violations,
            "safe_angles": safe_angles.tolist(),
            "safety_us": round(safety_us, 1),
        }

    def reset(self):
        mujoco.mj_resetData(self.model_ghost, self.data_ghost)
        mujoco.mj_resetData(self.model_safe, self.data_safe)
        mujoco.mj_forward(self.model_ghost, self.data_ghost)
        mujoco.mj_forward(self.model_safe, self.data_safe)
        self.current_safe_pos = np.zeros(6)


# Global sim pair
sim: SimPair | None = None


@app.on_event("startup")
async def startup():
    global sim
    sim = SimPair()
    print("✅ Dual MuJoCo simulations initialized")


@app.get("/")
async def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path) as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔗 WebSocket connected")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "angles":
                angles = msg["angles"]
                result = sim.step(angles)
                await ws.send_text(json.dumps(result))
            elif msg.get("type") == "reset":
                sim.reset()
                result = sim.step([0.0] * 6)
                await ws.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Heidegger Interactive Safety Simulator")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
