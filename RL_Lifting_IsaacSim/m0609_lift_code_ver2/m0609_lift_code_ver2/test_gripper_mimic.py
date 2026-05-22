"""Diagnostic: does the RG2 gripper close coherently?

Bypasses the RL policy. Resets the env, holds the arm at home pose, sends a
close-gripper command, and logs ALL gripper-related joint positions over time.

If mimic coupling works:  every RG2 joint moves smoothly toward its closed angle.
If mimic coupling broken: only the actively-driven joints move (finger_joint,
right_inner_knuckle_joint); passive joints stay near 0 OR drift randomly.
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
parser.add_argument("--steps_open", type=int, default=30)
parser.add_argument("--steps_close", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m0609_lift  # noqa: F401 — registers Isaac-M0609-Lift-v0 / -Play-v0
from m0609_lift.joint_pos_env_cfg import M0609LiftEnvCfg_PLAY


REPORT_PATH = "/tmp/gripper_mimic_report.txt"


def main() -> None:
    out = open(REPORT_PATH, "w", buffering=1)  # line-buffered
    out.write("=== M0609 RG2 GRIPPER MIMIC DIAGNOSTIC ===\n")

    try:
        cfg = M0609LiftEnvCfg_PLAY()
        cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=cfg)
        env.reset()
        robot = env.unwrapped.scene["robot"]

        name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
        out.write("\n[INFO] All articulation joints:\n")
        for n, i in name_to_idx.items():
            out.write(f"  [{i:2d}] {n}\n")

        rg2_joints = [
            "finger_joint",                # active (driver)
            "right_inner_knuckle_joint",   # active (driver)
            "left_inner_knuckle_joint",    # passive
            "left_outer_knuckle_joint",    # passive
            "right_outer_knuckle_joint",   # passive
            "left_inner_finger_joint",     # passive
            "right_inner_finger_joint",    # passive
        ]
        rg2_idx = [name_to_idx[n] for n in rg2_joints if n in name_to_idx]
        rg2_names = [n for n in rg2_joints if n in name_to_idx]
        out.write(f"\n[INFO] Tracking {len(rg2_names)} RG2 joints: {rg2_names}\n\n")

        n_envs = env.unwrapped.num_envs
        device = env.unwrapped.device
        act_dim = env.action_space.shape[-1]
        action = torch.zeros((n_envs, act_dim), device=device)

        def log_step(step: int, phase: str) -> None:
            q = robot.data.joint_pos[0, rg2_idx].cpu().numpy()
            cells = " ".join(f"{v:+.3f}" for v in q)
            out.write(f"  step {step:3d} [{phase:5s}]  {cells}\n")

        header = "       step       " + " ".join(f"{n[:14]:>14s}" for n in rg2_names)
        out.write(header + "\n")
        out.write("=" * len(header) + "\n")

        # Isaac Lab BinaryJointAction convention: action < 0 -> close, action >= 0 -> open.
        action[:, -1] = 1.0
        out.write("\n[PHASE 1] OPEN gripper (action[-1] = +1.0)\n")
        for s in range(args_cli.steps_open):
            env.step(action)
            if s % 5 == 0 or s == args_cli.steps_open - 1:
                log_step(s, "open")

        action[:, -1] = -1.0
        out.write("\n[PHASE 2] CLOSE gripper (action[-1] = -1.0)\n")
        for s in range(args_cli.steps_close):
            env.step(action)
            if s % 5 == 0 or s == args_cli.steps_close - 1:
                log_step(s, "close")

        final_q = robot.data.joint_pos[0, rg2_idx].cpu().numpy()
        out.write("\n[FINAL] Joint angles after close (rad):\n")
        for n, v in zip(rg2_names, final_q):
            marker = "  <-- driver" if n in {"finger_joint", "right_inner_knuckle_joint"} else "  (passive)"
            out.write(f"  {n:30s}  {v:+.4f}{marker}\n")

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
