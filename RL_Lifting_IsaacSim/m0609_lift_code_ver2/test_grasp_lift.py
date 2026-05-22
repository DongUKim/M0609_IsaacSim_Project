"""Diagnostic 2: actual grasp + lift test, plus geometry sanity-check.

Steps:
  1. Reset env. Print: home-pose EE (fingertip_center) world-frame position,
     cube world-frame position, distance between them.
  2. Teleport cube directly between the fingers (fingertip_center xy, ground z).
  3. Close gripper for N steps.
  4. Drive joint_2 upward (lift the arm) for M steps.
  5. Report: did the cube z follow the arm? (yes → grip holds. no → grip slipped.)

This tells us — independent of policy — whether the kinematic chain + the
imperfect mimic coupling is *strong enough to grasp + lift the dex cube*.
"""

from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-M0609-Lift-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
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

REPORT_PATH = "/tmp/grasp_lift_report.txt"


def main() -> None:
    out = open(REPORT_PATH, "w", buffering=1)
    out.write("=== M0609 GRASP + LIFT TEST ===\n")
    try:
        cfg = M0609LiftEnvCfg_PLAY()
        cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()

        scene = env.unwrapped.scene
        robot = scene["robot"]
        cube = scene["object"]
        ee_frame = scene["ee_frame"]

        device = env.unwrapped.device
        n_envs = env.unwrapped.num_envs
        act_dim = env.action_space.shape[-1]
        action = torch.zeros((n_envs, act_dim), device=device)

        # --- (1) home-pose geometry sanity ---
        ee_w = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
        cube_w = cube.data.root_pos_w[0].cpu().numpy()
        d = float(((ee_w - cube_w) ** 2).sum() ** 0.5)
        out.write(f"\n[GEOMETRY @ home pose]\n")
        out.write(f"  fingertip_center world xyz : ({ee_w[0]:+.3f}, {ee_w[1]:+.3f}, {ee_w[2]:+.3f})\n")
        out.write(f"  cube           world xyz : ({cube_w[0]:+.3f}, {cube_w[1]:+.3f}, {cube_w[2]:+.3f})\n")
        out.write(f"  distance EE -> cube      : {d:.3f} m\n")
        out.write(f"  reaching reward floor    : 1 - tanh({d:.3f}/0.3) = {1 - (torch.tanh(torch.tensor(d/0.3)).item()):.3f}\n")

        # --- (1.5) keep arm at home pose (gripper open), let it settle ---
        # BinaryJointAction convention: action >= 0 -> open, action < 0 -> close.
        hold_action = torch.zeros((n_envs, act_dim), device=device)
        hold_action[:, 6] = 1.0  # open gripper while we move the cube into place

        for _ in range(20):
            env.step(hold_action)
        ee_w = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
        out.write(f"\n[HOLD@HOME] EE after settling: ({ee_w[0]:+.3f}, {ee_w[1]:+.3f}, {ee_w[2]:+.3f})\n")

        # --- (1.6) measure where the inner-finger bodies actually live in
        #          world coordinates with the gripper FULLY CLOSED.  This tells
        #          us where the cube needs to be for the fingers to clamp it.
        body_to_idx = {n: i for i, n in enumerate(robot.data.body_names)}
        out.write("\n[BODIES] articulation body names:\n  ")
        out.write(", ".join(robot.data.body_names))
        out.write("\n")

        close_only = hold_action.clone()
        close_only[:, 6] = -1.0
        for _ in range(30):
            env.step(close_only)
        for name in ("left_inner_finger", "right_inner_finger", "gripper_body",
                     "left_phantom_pad", "right_phantom_pad"):
            if name in body_to_idx:
                p = robot.data.body_pos_w[0, body_to_idx[name]].cpu().numpy()
                out.write(f"  closed pose of {name:22s}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})\n")
        ee_closed = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
        out.write(f"  closed pose of fingertip_center      : ({ee_closed[0]:+.3f}, {ee_closed[1]:+.3f}, {ee_closed[2]:+.3f})\n")
        if "left_inner_finger" in body_to_idx and "right_inner_finger" in body_to_idx:
            lf = robot.data.body_pos_w[0, body_to_idx["left_inner_finger"]].cpu().numpy()
            rf = robot.data.body_pos_w[0, body_to_idx["right_inner_finger"]].cpu().numpy()
            gap = float(((lf - rf) ** 2).sum() ** 0.5)
            mid = (lf + rf) / 2.0
            out.write(f"  finger-to-finger gap (link origins) : {gap:.4f} m\n")
            out.write(f"  finger-pair midpoint                 : ({mid[0]:+.3f}, {mid[1]:+.3f}, {mid[2]:+.3f})\n")
        if "left_phantom_pad" in body_to_idx and "right_phantom_pad" in body_to_idx:
            lp = robot.data.body_pos_w[0, body_to_idx["left_phantom_pad"]].cpu().numpy()
            rp = robot.data.body_pos_w[0, body_to_idx["right_phantom_pad"]].cpu().numpy()
            pad_gap = float(((lp - rp) ** 2).sum() ** 0.5)
            pad_mid = (lp + rp) / 2.0
            out.write(f"  phantom-pad gap (link origins)       : {pad_gap:.4f} m\n")
            out.write(f"  phantom-pad midpoint                 : ({pad_mid[0]:+.3f}, {pad_mid[1]:+.3f}, {pad_mid[2]:+.3f})\n")

        # Reopen gripper for the actual trials.
        for _ in range(15):
            env.step(hold_action)
        ee_w = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
        out.write(f"\n[REOPEN] EE after re-opening: ({ee_w[0]:+.3f}, {ee_w[1]:+.3f}, {ee_w[2]:+.3f})\n")

        # --- (2) freeze-cube grasp test: hold cube at fingertip, close gripper,
        #         then release the freeze and lift.  If grip is mechanically
        #         possible, cube should follow the arm up.
        ee_anchor = ee_w.copy()
        import math

        for z_offset_mm in (+10, +5, 0, -5, -10, -15, -20, -25, -30, -40):
            out.write(f"\n--- TRIAL z_offset = {z_offset_mm} mm below fingertip (frozen close) ---\n")

            # Reopen gripper, settle.
            for _ in range(15):
                env.step(hold_action)

            cube_z = float(ee_anchor[2]) + z_offset_mm * 1e-3
            cube_xy = (float(ee_anchor[0]), float(ee_anchor[1]))

            # Close gripper while continually re-pinning cube at the target xyz
            # (zero velocity).  This effectively freezes gravity while the
            # fingers swing in.  The cube is still a dynamic body — only its
            # state is overwritten each step.
            close_action = hold_action.clone()
            close_action[:, 6] = -1.0
            pin_pose = torch.tensor(
                [[cube_xy[0], cube_xy[1], cube_z, 1.0, 0.0, 0.0, 0.0]],
                device=device,
            )
            zero_vel = torch.zeros((1, 6), device=device)
            for s in range(30):
                cube.write_root_pose_to_sim(pin_pose)
                cube.write_root_velocity_to_sim(zero_vel)
                env.step(close_action)
            ee_after = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
            cube_after = cube.data.root_pos_w[0].cpu().numpy()
            out.write(
                f"  after close (pinned): EE z={ee_after[2]:+.3f}  cube z={cube_after[2]:+.3f}\n"
            )

            # Now release the pin and lift.  Gripper stays closed.  Cube is
            # free.  If the fingers actually grip the cube, it follows the arm.
            lift_action = torch.zeros((n_envs, act_dim), device=device)
            lift_action[:, 1] = -math.pi / 6.0
            lift_action[:, 6] = -1.0
            for s in range(60):
                env.step(lift_action)
                if s in (0, 5, 10, 20, 40, 59):
                    ee_l = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
                    cube_l = cube.data.root_pos_w[0].cpu().numpy()
                    out.write(
                        f"  lift step {s:3d}  EE z={ee_l[2]:+.3f}  cube z={cube_l[2]:+.3f}\n"
                    )
            cube_lift = cube.data.root_pos_w[0].cpu().numpy()
            held = cube_lift[2] - cube_after[2]
            verdict = "GRASP HOLDS" if held > 0.05 else ("WEAK GRIP" if held > 0.005 else "NO GRIP")
            out.write(f"  ===> z_offset={z_offset_mm}mm: cube_lifted={held:+.3f} m  [{verdict}]\n")

        env.close()
        out.write("\n[DONE] OK\n")
    except Exception as e:
        import traceback
        out.write(f"\n[ERROR] {type(e).__name__}: {e}\n")
        out.write(traceback.format_exc())
    finally:
        out.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
