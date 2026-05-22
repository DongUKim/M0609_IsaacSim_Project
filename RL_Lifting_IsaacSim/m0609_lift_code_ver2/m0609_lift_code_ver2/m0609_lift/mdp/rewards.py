# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the M0609 lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    std: float = 0.08,
    gripper_action_name: str = "gripper_action",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward commanding gripper-close while the EE is near the object.

    Uses the policy's *action intent* (raw command from the BinaryJointAction
    term) instead of the realised finger joint position.  Reason: when the
    fingers stall against the cube during a clumsy grasp the actual finger_pos
    barely changes, so an "actual closure" reward would give zero feedback for
    a policy that *did* try to close.  Rewarding the intent decouples the
    learning signal from the contact dynamics and lets the policy discover
    "close gripper at the cube" even if the first attempts physically fail.

    BinaryJointAction convention: raw action < 0 → close, ≥ 0 → open.

    Args:
        std: Distance scale for the tanh proximity kernel (metres).
        gripper_action_name: Name of the gripper action term in ActionsCfg.
        object_cfg: Scene entity config for the target object.
        ee_frame_cfg: Scene entity config for the end-effector frame sensor.

    Returns:
        Reward in [0, 1]: high only when EE is close *and* policy commanded close.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(cube_pos_w - ee_w, dim=1)

    proximity = 1.0 - torch.tanh(dist / std)

    gripper_term = env.action_manager.get_term(gripper_action_name)
    gripper_close_intent = (gripper_term.raw_actions[:, 0] < 0).float()

    return proximity * gripper_close_intent
