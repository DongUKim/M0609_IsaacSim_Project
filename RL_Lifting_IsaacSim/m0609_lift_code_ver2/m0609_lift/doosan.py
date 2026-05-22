# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M0609 + OnRobot RG2 ArticulationCfg for the lift task.

Source URDF resolution order (first path whose M0609 URDF exists wins):
  1. ``M0609_SRC_DIR`` environment variable
  2. ``<m0609_lift_code>/robots/``  (bundled copy alongside this package)
  3. ``/home/deeptree/dev_ws/isaac_sim/src``  (original dev-machine default)

The combined URDF is cached in ``<this package>/cache/`` and regenerated
automatically whenever either source URDF changes (SHA-1 hash check).
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

# ---------------------------------------------------------------------------
# Source URDF path resolution
# ---------------------------------------------------------------------------
_THIS_PKG = Path(__file__).resolve().parent   # .../m0609_lift_code/m0609_lift/
_BUNDLED  = _THIS_PKG.parent / "robots"       # .../m0609_lift_code/robots/
_ORIGINAL = Path("/home/deeptree/dev_ws/isaac_sim/src")

_src_env = os.environ.get("M0609_SRC_DIR")
if _src_env:
    M0609_SOURCE_DIR = Path(_src_env)
elif (_BUNDLED / "doosan-robot2" / "urdf" / "m0609_isaac_sim.urdf").exists():
    M0609_SOURCE_DIR = _BUNDLED
else:
    M0609_SOURCE_DIR = _ORIGINAL

M0609_URDF_PATH = M0609_SOURCE_DIR / "doosan-robot2" / "urdf" / "m0609_isaac_sim.urdf"
RG2_URDF_PATH   = M0609_SOURCE_DIR / "onrobot_rg2"   / "urdf" / "onrobot_rg2.urdf"

# Combined URDF cache — lives inside this package so it travels with the code.
_CACHE_DIR = _THIS_PKG / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_URDF_PATH = _CACHE_DIR / "m0609_rg2.urdf"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOME_POSE: tuple[float, ...] = (0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0)
GRIPPER_OPEN_QPOS:   tuple[float, float] = (0.0, 0.0)
GRIPPER_CLOSED_QPOS: tuple[float, float] = (1.15, 1.15)  # near URDF limit ±1.18

GRIPPER_DRIVE_MAX_FORCE:  float = 1.0e6
GRIPPER_DRIVE_STIFFNESS:  float = 1.0e5
GRIPPER_DRIVE_DAMPING:    float = 1.0e3
# After URDF simplification (gen=v4) only the two outer-knuckle joints remain
# revolute; the inner_knuckle/inner_finger joints are converted to fixed so the
# gripper is a clean two-joint parallel-style mechanism (no closed loop).
GRIPPER_DRIVEN_JOINTS: tuple[str, str] = ("finger_joint", "left_outer_knuckle_joint")

# EE link tracked by FrameTransformerCfg — virtual fixed link injected at
# the geometric midpoint between the two RG2 inner finger pivots.
EE_BODY_NAME = "fingertip_center"


# ---------------------------------------------------------------------------
# Combined URDF generation
# ---------------------------------------------------------------------------

def _build_combined_urdf(
    m0609_path: Path = M0609_URDF_PATH,
    rg2_path: Path = RG2_URDF_PATH,
    out_path: Path = COMBINED_URDF_PATH,
) -> Path:
    """Merge the M0609 and RG2 URDFs into a single combined URDF.

    Mesh ``filename`` attributes in both URDFs are rewritten to absolute paths
    so the combined file works from any working directory.  Only regenerates
    ``out_path`` when sources change (SHA-1 hash short-circuit).

    Args:
        m0609_path: Source path to the M0609 URDF.
        rg2_path:   Source path to the RG2 URDF.
        out_path:   Destination path for the merged URDF.

    Returns:
        ``out_path`` as a :class:`Path`.

    Raises:
        FileNotFoundError: If either source URDF is missing.
    """
    if not m0609_path.exists():
        raise FileNotFoundError(
            f"M0609 URDF not found: {m0609_path}\n"
            f"Set the M0609_SRC_DIR env var or place robot sources in {_BUNDLED}"
        )
    if not rg2_path.exists():
        raise FileNotFoundError(
            f"RG2 URDF not found: {rg2_path}\n"
            f"Set the M0609_SRC_DIR env var or place robot sources in {_BUNDLED}"
        )

    m0609_text = m0609_path.read_text()
    rg2_text   = rg2_path.read_text()

    # Include the build-script gen tag in the hash so post-processing changes
    # (e.g. converting joints to fixed) invalidate the cache automatically.
    _GEN_TAG = "v10"  # fingertip_center z 0.135 → 0.11387 보정
    src_hash = hashlib.sha1((m0609_text + rg2_text + _GEN_TAG).encode()).hexdigest()[:12]
    if out_path.exists():
        if f"src_hash={src_hash}" in out_path.read_text(errors="ignore")[:256]:
            return out_path

    # Rewrite M0609 mesh paths to absolute (URDF ships with absolute paths
    # baked in — replace whatever prefix exists with the actual directory).
    m0609_dir = m0609_path.parent
    m0609_text = re.sub(
        r'filename="[^"]+/meshes/',
        f'filename="{m0609_dir.as_posix()}/meshes/',
        m0609_text,
    )

    # Rewrite RG2 relative mesh paths (../meshes/) to absolute.
    rg2_dir = rg2_path.parent
    rg2_text = re.sub(
        r'filename="\.\./meshes/',
        f'filename="{rg2_dir.parent.as_posix()}/meshes/',
        rg2_text,
    )

    # Strip RG2 outer <robot> wrapper.
    rg2_inner = re.search(r"<robot[^>]*>(.*)</robot>", rg2_text, flags=re.DOTALL)
    if rg2_inner is None:
        raise RuntimeError("Failed to parse RG2 URDF root element.")
    rg2_body = rg2_inner.group(1)

    # Remove RG2's standalone world link and quick_changer_joint (replaced below).
    rg2_body = re.sub(r"<link\s+name=\"world\"\s*/>", "", rg2_body)
    rg2_body = re.sub(
        r"<joint\s+name=\"quick_changer_joint\"\s+type=\"fixed\">.*?</joint>",
        "",
        rg2_body,
        flags=re.DOTALL,
    )
    rg2_body = re.sub(r"<gazebo[^>]*>.*?</gazebo>", "", rg2_body, flags=re.DOTALL)
    rg2_body = re.sub(r"<transmission[^>]*>.*?</transmission>", "", rg2_body, flags=re.DOTALL)

    # PhysX cannot model the RG2's four-bar closed kinematic chain from URDF
    # mimic constraints (URDF is tree-only, mimic is not preserved by Isaac Sim).
    # Driving all 6 finger-side joints independently fights itself and prevents
    # the outer knuckles from reaching their targets.  Collapse the chain by
    # converting the four inner joints to fixed — each side becomes a single
    # rigid finger that pivots around its outer-knuckle joint, leaving only
    # finger_joint (right outer) and left_outer_knuckle_joint as revolute drivers.
    for jname in (
        "left_inner_knuckle_joint",
        "right_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_inner_finger_joint",
    ):
        rg2_body = re.sub(
            rf'(<joint\s+name="{jname}"\s+)type="revolute"',
            r'\1type="fixed"',
            rg2_body,
        )

    # The RG2 inner_finger collision mesh has a complex shape that, under our
    # simplified (no closed-loop mimic) kinematics, does not form a planar
    # parallel clamp at any closure angle.  Drop its collision so it can't
    # interfere with the phantom-pad clamp added below.  Visuals are kept so
    # the gripper still LOOKS like an RG2.
    rg2_body = re.sub(
        r'(<link\s+name="(?:left|right)_inner_finger">.*?)<collision>.*?</collision>',
        r'\1',
        rg2_body,
        flags=re.DOTALL,
    )

    # Phantom finger-pad clamp: two box colliders, one fixed-child of each
    # outer_knuckle.  At outer_knuckle = ±q_close the boxes meet at the
    # gripper centreline, providing a clean planar grasp that doesn't depend
    # on the RG2 mesh geometry.  Origin rpy_y on each pad rotates the box so
    # its inner face is perpendicular to the gripper centreline at q_close.
    # The ±0.020 x-offset places the inner face ~12 mm inboard of the
    # outer_knuckle pivot at full close, giving room for a 25 mm cube.
    Q_CLOSE = 1.15  # must match GRIPPER_CLOSED_QPOS
    # Pad collision origin xyz placed in outer_knuckle local frame so that
    # after the joint rotates by ±q_close the pad ends up:
    #   - meeting at the gripper centreline (x_local ±0.014 → ~12 mm gap at close)
    #   - covering the fingertip area in z (longer box, 80 mm tall)
    phantom_pads = f"""
    <link name="right_phantom_pad">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.001"/>
            <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
        </inertial>
        <collision>
            <origin rpy="0 {-Q_CLOSE:.6f} 0" xyz="-0.020 0 0.080"/>
            <geometry>
                <box size="0.005 0.030 0.080"/>
            </geometry>
        </collision>
    </link>
    <joint name="right_phantom_pad_joint" type="fixed">
        <parent link="right_outer_knuckle"/>
        <child link="right_phantom_pad"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
    </joint>

    <link name="left_phantom_pad">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.001"/>
            <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>
        </inertial>
        <collision>
            <origin rpy="0 {Q_CLOSE:.6f} 0" xyz="0.020 0 0.080"/>
            <geometry>
                <box size="0.005 0.030 0.080"/>
            </geometry>
        </collision>
    </link>
    <joint name="left_phantom_pad_joint" type="fixed">
        <parent link="left_outer_knuckle"/>
        <child link="left_phantom_pad"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
    </joint>
"""
    rg2_body = rg2_body + phantom_pads

    # Strip M0609 outer <robot> wrapper.
    m0609_inner = re.search(r"<robot[^>]*>(.*)</robot>", m0609_text, flags=re.DOTALL)
    if m0609_inner is None:
        raise RuntimeError("Failed to parse M0609 URDF root element.")
    m0609_body = m0609_inner.group(1)

    mount_joint = """
    <joint name="m0609_rg2_mount" type="fixed">
        <parent link="link_6"/>
        <child link="quick_changer"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
    </joint>
"""

    # Virtual massless link at the midpoint between the two inner finger pivots
    # (in gripper_body local frame).
    # Derived from knuckle joint origins:
    #   left_outer_knuckle:  x=0.04405, z=0.07590
    #   right_outer_knuckle: x=0.01005, z=0.07590  → midpoint x=0.02705
    # inner_finger_joint adds z=0.03797 → total z ≈ 0.11387
    fingertip_center = """
    <link name="fingertip_center"/>
    <joint name="fingertip_center_joint" type="fixed">
        <parent link="gripper_body"/>
        <child link="fingertip_center"/>
        <origin rpy="0 0 0" xyz="0.02705 -0.00953 0.11387"/>
    </joint>
"""

    combined = (
        f'<?xml version="1.0"?>\n'
        f"<!-- Auto-generated combined M0609 + RG2 URDF. src_hash={src_hash} gen=v4 -->\n"
        f'<robot name="m0609_rg2">\n'
        f"{m0609_body}\n"
        f"{mount_joint}\n"
        f"{rg2_body}\n"
        f"{fingertip_center}\n"
        f"</robot>\n"
    )

    out_path.write_text(combined)
    return out_path


# Eagerly build / verify on import.
_combined_urdf = _build_combined_urdf()

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
DOOSAN_M0609_RG2_LIFT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(COMBINED_URDF_PATH),
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Disabled: after URDF simplification (gen=v4) the inner_knuckle and
            # inner_finger links are fixed children of gripper_body / outer_knuckle.
            # Their meshes overlap mid-closure and self-collide if enabled, which
            # ricochets the outer knuckle away from its target.  Cube (external)
            # collisions are unaffected.
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        collider_type="convex_decomposition",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": HOME_POSE[0],
            "joint_2": HOME_POSE[1],
            "joint_3": HOME_POSE[2],
            "joint_4": HOME_POSE[3],
            "joint_5": HOME_POSE[4],
            "joint_6": HOME_POSE[5],
            # Only finger_joint (right outer knuckle) and left_outer_knuckle_joint
            # are revolute after URDF simplification; the four inner joints are now
            # fixed and don't appear in the articulation.
            "finger_joint":              GRIPPER_OPEN_QPOS[0],
            "left_outer_knuckle_joint":  -GRIPPER_OPEN_QPOS[1],
        },
    ),
    actuators={
        "m0609_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit_sim=9600.0,
            velocity_limit_sim=2.618,
            stiffness=3000.0,
            damping=200.0,
        ),
        # Only the two outer-knuckle joints remain revolute after URDF
        # simplification (gen=v4).  Each side is now a single rigid finger.
        "rg2_drive": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "left_outer_knuckle_joint",
            ],
            effort_limit_sim=GRIPPER_DRIVE_MAX_FORCE,
            velocity_limit_sim=2.0,
            stiffness=GRIPPER_DRIVE_STIFFNESS,
            damping=GRIPPER_DRIVE_DAMPING,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
