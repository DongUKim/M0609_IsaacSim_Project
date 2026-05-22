# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M0609 joint-position-control lift environment configuration (claude_rl.md §4.2, §5)."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .doosan import DOOSAN_M0609_RG2_LIFT_CFG, EE_BODY_NAME, GRIPPER_CLOSED_QPOS, GRIPPER_OPEN_QPOS
from .lift_env_cfg import LiftEnvCfg


@configclass
class M0609LiftEnvCfg(LiftEnvCfg):
    """Full-training env: 4096 envs, obs corruption on."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------ #
        # Robot
        # ------------------------------------------------------------------ #
        self.scene.robot = DOOSAN_M0609_RG2_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------------------ #
        # Actions (claude_rl.md §4.2) — 7-dim: 6 arm + 1 binary gripper
        # ------------------------------------------------------------------ #
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )
        # After URDF simplification (doosan.py gen=v4) only the two outer-knuckle
        # joints are revolute; each side is a single rigid finger pivoting around
        # its outer knuckle.  finger_joint (right) closes positive, left mirrors
        # with opposite sign per the URDF axis conventions.
        _CLOSED = GRIPPER_CLOSED_QPOS[0]
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint", "left_outer_knuckle_joint"],
            open_command_expr={
                "finger_joint":              0.0,
                "left_outer_knuckle_joint":  0.0,
            },
            close_command_expr={
                "finger_joint":              +_CLOSED,
                "left_outer_knuckle_joint":  -_CLOSED,
            },
        )

        # ------------------------------------------------------------------ #
        # Command: target pose body (used only for metric computation)
        # ------------------------------------------------------------------ #
        self.commands.object_pose.body_name = EE_BODY_NAME

        # ------------------------------------------------------------------ #
        # Object — DexCube scaled to ~0.04m (claude_rl.md §3.4)
        # ------------------------------------------------------------------ #
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.0, 0.03], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # ------------------------------------------------------------------ #
        # EE frame — fingertip_center virtual link injected into the URDF at
        # the midpoint between left/right inner finger pivots.  No offset needed.
        # ------------------------------------------------------------------ #
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",  # root link of the M0609 chain
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class M0609LiftEnvCfg_PLAY(M0609LiftEnvCfg):
    """Play variant: 50 envs, no obs noise (claude_rl.md §5.2)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
