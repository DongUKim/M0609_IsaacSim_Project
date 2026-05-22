"""Scan candidate home poses and report EE position.

For each candidate joint configuration, set the robot joints, settle for a few
sim steps, then log the resulting fingertip_center world-frame pose.

Goal: pick a home pose that puts the EE roughly above the table (x ≈ +0.4,
y ≈ 0, z ≈ 0.3) so the policy doesn't have to swing 180° to reach the cube.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-M0609-Lift-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m0609_lift  # noqa: F401
from m0609_lift.joint_pos_env_cfg import M0609LiftEnvCfg_PLAY

REPORT_PATH = "/tmp/home_pose_scan.txt"

PI = math.pi
H = PI / 2.0
Q = PI / 4.0

CANDIDATES = {
    "B7  j2=0, j3=+90 (baseline)":          [0.0, 0.0, H,   0.0, 0.0, 0.0],
    "B7a B7 + j5=-90":                      [0.0, 0.0, H,   0.0, -H, 0.0],
    "B7b B7 + j5=+90":                      [0.0, 0.0, H,   0.0,  H, 0.0],
    "B7c B7 + j5=180":                      [0.0, 0.0, H,   0.0, PI, 0.0],
    "C1  ready: j2=-30, j3=+90, j5=-60":    [0.0, -PI/6, H,   0.0, -PI/3, 0.0],
    "C2  ready: j2=-45, j3=+90, j5=-45":    [0.0, -Q,    H,   0.0, -Q,    0.0],
    "C3  ready: j2=-60, j3=+90, j5=-30":    [0.0, -PI/3, H,   0.0, -PI/6, 0.0],
    "C4  j2=-30, j3=+60, j5=-30":           [0.0, -PI/6, PI/3, 0.0, -PI/6, 0.0],
}


def main() -> None:
    out = open(REPORT_PATH, "w", buffering=1)
    out.write("=== HOME POSE SCAN ===\n")
    out.write("Cube spawns at (+0.5, 0, ~0.02).  Goal: pick a pose with EE near (+0.4, 0, ~0.3).\n\n")

    try:
        cfg = M0609LiftEnvCfg_PLAY()
        cfg.scene.num_envs = 1
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()

        scene = env.unwrapped.scene
        robot = scene["robot"]
        ee_frame = scene["ee_frame"]
        cube = scene["object"]
        device = env.unwrapped.device
        n_envs = 1
        act_dim = env.action_space.shape[-1]
        zero_action = torch.zeros((n_envs, act_dim), device=device)

        # Joint name → idx
        name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
        arm_names = [f"joint_{i}" for i in range(1, 7)]
        arm_idx = [name_to_idx[n] for n in arm_names]
        body_to_idx = {n: i for i, n in enumerate(robot.data.body_names)}
        gripper_body_idx = body_to_idx.get("gripper_body", -1)

        cube_w = cube.data.root_pos_w[0].cpu().numpy()
        out.write(f"cube world xyz: ({cube_w[0]:+.3f}, {cube_w[1]:+.3f}, {cube_w[2]:+.3f})\n\n")
        header = f"{'pose':38s}  {'EE x':>7s}  {'EE y':>7s}  {'EE z':>7s}  {'dist':>5s}  {'fingerDir(x,y,z)':>22s}  pointing"
        out.write(header + "\n")
        out.write("-" * len(header) + "\n")

        for label, q in CANDIDATES.items():
            # Build full joint-pos tensor (arm part = candidate, gripper part = 0).
            full = robot.data.default_joint_pos[0].clone()
            for n, v in zip(arm_names, q):
                full[name_to_idx[n]] = float(v)
            full_b = full.unsqueeze(0)  # (1, n_joints)
            zero_v = torch.zeros_like(full_b)

            # Write joint state to sim, then step a few times so FrameTransformer updates.
            robot.write_joint_state_to_sim(full_b, zero_v)
            for _ in range(8):
                env.step(zero_action)

            ee_w = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
            cube_w = cube.data.root_pos_w[0].cpu().numpy()
            d = float(((ee_w - cube_w) ** 2).sum() ** 0.5)

            # Finger-direction unit vector: fingertip_center - gripper_body, in world frame.
            if gripper_body_idx >= 0:
                gb_w = robot.data.body_pos_w[0, gripper_body_idx].cpu().numpy()
                v = ee_w - gb_w
                norm = float((v * v).sum() ** 0.5) + 1e-9
                vn = v / norm
                # Classify: "down" means vn z < -0.7, "fwd"  means vn x > 0.7, etc.
                dir_x, dir_y, dir_z = float(vn[0]), float(vn[1]), float(vn[2])
                if dir_z < -0.7:
                    pointing = "DOWN ↓"
                elif dir_z > 0.7:
                    pointing = "UP ↑"
                elif abs(dir_x) > 0.7:
                    pointing = "+X →" if dir_x > 0 else "-X ←"
                elif abs(dir_y) > 0.7:
                    pointing = "+Y" if dir_y > 0 else "-Y"
                else:
                    pointing = "tilted"
                fdir = f"({dir_x:+.2f},{dir_y:+.2f},{dir_z:+.2f})"
            else:
                fdir = "  (no gripper_body)   "
                pointing = "?"
            out.write(f"{label:38s}  {ee_w[0]:+.3f}  {ee_w[1]:+.3f}  {ee_w[2]:+.3f}  {d:.3f}  {fdir:>22s}  {pointing}\n")

        out.write("\n[DONE]\n")
        env.close()
    except Exception as e:
        import traceback
        out.write(f"\n[ERROR] {type(e).__name__}: {e}\n")
        out.write(traceback.format_exc())
    finally:
        out.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
