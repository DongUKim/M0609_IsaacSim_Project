# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract LiftEnvCfg for the M0609 lift task.

Mirrors the structure of
``isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg``
but with the robot base at world-origin (table at x=0.55) and
target-pose ranges tuned for the M0609 reach envelope (claude_rl.md §3/4).

Concrete subclasses must set:
  - scene.robot
  - scene.ee_frame
  - scene.object
  - actions.arm_action
  - actions.gripper_action
  - commands.object_pose.body_name
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Scene with a robot on the floor, SeattleLabTable at x=0.55 (claude_rl.md §3.1-3.2)."""

    # filled in by concrete env cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    # Table at (0.55, 0, 0), long-axis along Y — same USD as the Franka lift reference.
    # claude_rl.md §3.2: top face z ≈ 0.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.55, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane at z=-1.05 so the table top sits at z≈0 (matches Franka lift reference).
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Pose command terms."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set in concrete cfg (e.g. "link_6")
        # Fixed target for one full episode: resampling_time > episode_length_s.
        # claude_rl.md §3.5: episode_length_s=5.0 → resampling=(6.0, 6.0).
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Positions in robot root frame (world frame since robot is at origin).
            # x=0.4-0.6 places the target above the table (table at x=0.55).
            # Note: claude_rl.md §3.5 lists pos_x=(-0.1,0.1) but that lands behind
            # the robot.  Using Franka-lift-equivalent range instead (history.md §1).
            pos_x=(0.4, 0.6),
            pos_y=(-0.3, 0.3),
            pos_z=(0.20, 0.40),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specs; filled in by concrete env cfg."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Single policy observation group (claude_rl.md §4.3, total dim=29)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)           # 6+2 gripper (all joints)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)           # same
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)  # 3
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )  # 7
        actions = ObsTerm(func=mdp.last_action)               # 7 (6 arm + 1 gripper)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Randomize cube xy around its init position (claude_rl.md §3.7).
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms — Franka-lift recipe baseline."""

    # 1) Reach: dense shaping toward the cube.
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=1.0,
    )

    # 2) Lift indicator. minimal_height를 큐브 안착 중심 높이(0.03) + α 로 낮춰서
    #    "조금만 들어도 보상" 시작하도록 만든다 — 절벽 제거.
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )

    # 3) Goal tracking (lifted 후에만 활성). minimal_height 동일하게 0.04.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # 4) Drop penalty (그대로 유지).
    dropping_penalty = RewTerm(
        func=mdp.is_terminated_term,
        params={"term_keys": "object_dropping"},
        weight=-5.0,
    )

    # 5) 정규화 페널티 (그대로 유지).
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination conditions (claude_rl.md §4.5)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms (claude_rl.md §4.6)."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50_000_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 50_000_000},
    )
    # Ramp dropping_penalty from -5 → -10 after policy has stably discovered lift
    # (~30M env steps = ~305 iter). Early ramp-up noisifies learning signal before lift emerges.
    dropping_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dropping_penalty", "weight": -10.0, "num_steps": 30_000_000},
    )


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract lift environment configuration for M0609."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz control: sim at 100 Hz, decimate ×2.
        self.decimation = 2
        self.episode_length_s = 5.0  # claude_rl.md §4.7
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
