# SimulationApp 은 반드시 모든 omniverse import 보다 먼저 실행되어야 함.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pathlib import Path
import sys
import os
import time

import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Usd, UsdPhysics

from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)

from isaacsim.robot_setup.assembler import RobotAssembler

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from m0609_pick_place_controller import PickPlaceController

M0609_URDF_PATH = str(BASE_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ONROBOT_URDF_PATH = str(BASE_DIR / "onrobot_rg2/urdf/onrobot_rg2.urdf")
M0609_RMPFLOW_CONFIG_PATH = str(BASE_DIR / "m0609_rmpflow_common.yaml")
M0609_DESCRIPTION_PATH = str(BASE_DIR / "m0609_rg2_description.yaml")

EE_LINK_NAME = "link_6"
GRIPPER_BASE_LINK = "angle_bracket"


# =====================================================================
# 유틸 함수
# =====================================================================
def import_urdf(urdf_path, fix_base=True):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF 파일이 존재하지 않습니다: {urdf_path}")

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = True
    import_config.import_inertia_tensor = True
    import_config.fix_base = fix_base
    import_config.distance_scale = 1.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength = 1e10
    import_config.default_position_drive_damping = 1e5

    _, artic_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )

    if artic_path is None:
        raise RuntimeError(f"URDF import 실패: {urdf_path}")

    robot_root = artic_path.rsplit("/", 1)[0] or artic_path
    print(f"  [OK] URDF import: {urdf_path}")
    print(f"       → articulation = {artic_path}")
    print(f"       → robot root   = {robot_root}")
    return robot_root, artic_path


def assemble_robot(stage, robot_base, robot_base_mount, robot_attach, robot_attach_mount, assembly_namespace, variant_name):
    assembler = RobotAssembler()
    assembler.begin_assembly(
        stage,
        robot_base,
        robot_base_mount,
        robot_attach,
        robot_attach_mount,
        assembly_namespace,
        variant_name,
    )
    assembler.assemble()
    assembler.finish_assemble()


def find_prim_path_by_name(root_path, link_name):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == link_name:
            return str(prim.GetPath())
    return None


# =====================================================================
# Task
# =====================================================================
class DoosanPickPlaceTask(BaseTask):

    def __init__(self, name,
                 cube_initial_position=None,
                 goal_position=None):
        super().__init__(name=name, offset=None)
        self._goal_position = (
            goal_position if goal_position is not None
            else np.array([0.55, -0.35, 0.0])
        )
        self._cube_initial_position = (
            cube_initial_position if cube_initial_position is not None
            else np.array([0.30, 0.4, 0.0515 / 2.0])
        )
        self._task_achieved = False

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # ── Step 1: URDF Import ──────────────────────────────
        print("\n" + "=" * 60)
        print("[Step 1] URDF Import")
        print("=" * 60)

        robot_root, robot_artic_path = import_urdf(M0609_URDF_PATH, fix_base=True)
        gripper_root, gripper_artic_path = import_urdf(ONROBOT_URDF_PATH, fix_base=False)

        # ── Step 2: RobotAssembler 결합 ──────────────────────
        print("\n" + "=" * 60)
        print("[Step 2] RobotAssembler 결합")
        print("=" * 60)

        robot_ee_path = (
            find_prim_path_by_name(robot_root, EE_LINK_NAME)
            or f"{robot_root}/{EE_LINK_NAME}"
        )
        gripper_base_path = (
            find_prim_path_by_name(gripper_root, GRIPPER_BASE_LINK)
            or f"{gripper_root}/{GRIPPER_BASE_LINK}"
        )

        print(f"  Robot EE:      {robot_ee_path}")
        print(f"  Gripper Base:  {gripper_base_path}")

        stage = omni.usd.get_context().get_stage()
        assemble_robot(
            stage,
            robot_root,
            robot_ee_path,
            gripper_root,
            gripper_base_path,
            "Gripper",
            "m0609_rg2",
        )
        print("  [OK] 결합 완료")

        # 결합 후 EE 경로 재탐색
        robot_ee_path = find_prim_path_by_name(robot_root, EE_LINK_NAME)
        print(f"  assembled ee path = {robot_ee_path}")

        # ── Gripper joint drive 강화 ─────────────────────────
        for joint_name in ["finger_joint", "right_inner_knuckle_joint"]:
            joint_path = find_prim_path_by_name(robot_root, joint_name)
            if joint_path:
                joint_prim = stage.GetPrimAtPath(joint_path)
                for drive_type in ["angular", "linear"]:
                    drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
                    if drive:
                        drive.GetMaxForceAttr().Set(1e6)
                        drive.GetStiffnessAttr().Set(1e5)
                        drive.GetDampingAttr().Set(1e3)
                        print(f"  [OK] {drive_type} drive 강화: {joint_path}")

        for _ in range(10):
            simulation_app.update()

        # ── Step 3: ParallelGripper + SingleManipulator ──────
        print("\n" + "=" * 60)
        print("[Step 3] ParallelGripper + SingleManipulator")
        print("=" * 60)

        gripper = ParallelGripper(
            end_effector_prim_path=robot_ee_path,
            joint_prim_names=["finger_joint", "right_inner_knuckle_joint"],
            joint_opened_positions=np.array([0.0, 0.0]),
            joint_closed_positions=np.array([0.5, 0.5]),
            action_deltas=np.array([-0.5, -0.5]),
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=robot_root,
                name="m0609_robot",
                end_effector_prim_path=robot_ee_path,
                gripper=gripper,
            )
        )

        # ── Cube (마찰력 포함) ───────────────────────────────
        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=self._cube_initial_position,
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.0, 0.0, 1.0]),
                mass=0.05,
                physics_material=cube_material,
            )
        )

        # ── Goal marker ─────────────────────────────────────
        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=self._goal_position,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        # ── 그리퍼 내부 마찰력 ───────────────────────────────
        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=1.8,
            dynamic_friction=1.4,
            restitution=0.0,
        )

        left_finger_path = find_prim_path_by_name(robot_root, "left_inner_finger")
        right_finger_path = find_prim_path_by_name(robot_root, "right_inner_finger")

        print(f"  Left finger:  {left_finger_path}")
        print(f"  Right finger: {right_finger_path}")

        if left_finger_path:
            left_finger_geom = SingleGeometryPrim(
                prim_path=left_finger_path,
                name="left_finger_geom",
            )
            left_finger_geom.apply_physics_material(finger_material)

        if right_finger_path:
            right_finger_geom = SingleGeometryPrim(
                prim_path=right_finger_path,
                name="right_finger_geom",
            )
            right_finger_geom.apply_physics_material(finger_material)

        print("\n  [완료] 씬 구성 성공!\n")

    def get_observations(self):
        cube_position, _ = self._cube.get_world_pose()
        current_joint_positions = self._robot.get_joint_positions()
        return {
            self._robot.name: {
                "joint_positions": current_joint_positions,
            },
            self._cube.name: {
                "position": cube_position,
                "goal_position": self._goal_position,
            },
        }

    def pre_step(self, control_index, simulation_time):
        cube_position, _ = self._cube.get_world_pose()
        if (not self._task_achieved
                and np.mean(np.abs(self._goal_position - cube_position)) < 0.02):
            self._cube.get_applied_visual_material().set_color(
                color=np.array([0.0, 1.0, 0.0])
            )
            self._task_achieved = True

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        self._cube.get_applied_visual_material().set_color(
            color=np.array([0.0, 0.0, 1.0])
        )
        self._task_achieved = False


# =====================================================================
# 메인
# =====================================================================

class DoosanPickNPlace():

    def __init__(self):
        pass

    def main(self):
        my_world = World(stage_units_in_meters=1.0)

        # ── Task 등록 및 씬 구성 ─────────────────────────────────
        task = DoosanPickPlaceTask(name="doosan_pick_place_task")
        my_world.add_task(task)
        my_world.reset()

        # ── Task에서 생성된 객체 가져오기 ────────────────────────
        robot = my_world.scene.get_object("m0609_robot")
        goal_position = task._goal_position

        robot.initialize()
        robot.gripper.initialize(
            physics_sim_view=my_world.physics_sim_view,
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )

        # ── Joint 정보 출력 ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("[Step 4] Joint 정보")
        print("=" * 60)
        print(f"  DOF: {robot.num_dof}")
        for i, name in enumerate(robot.dof_names):
            print(f"  [{i:2d}] {name}")
        print("=" * 60)

        # ── Controller 생성 ──────────────────────────────────────
        controller = PickPlaceController(
            name="m0609_pick_place_controller",
            gripper=robot.gripper,
            robot_articulation=robot,
            end_effector_initial_height=0.30,
            events_dt=[0.008, 0.005, 0.02, 0.1, 0.0025, 0.01, 0.0025, 1, 0.008, 0.08],
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=EE_LINK_NAME,
        )

        print("\n[Pick & Place 시작]\n")
        was_playing = False
        task_done = False

        while simulation_app.is_running():
            my_world.step(render=True)
            time.sleep(0.01)
            is_playing = my_world.is_playing()

            # ── 재생 시작 시 리셋 ────────────────────────────────
            if is_playing and not was_playing:
                my_world.reset()
                robot.initialize()
                robot.gripper.initialize(
                    physics_sim_view=my_world.physics_sim_view,
                    articulation_apply_action_func=robot.apply_action,
                    get_joint_positions_func=robot.get_joint_positions,
                    set_joint_positions_func=robot.set_joint_positions,
                    dof_names=robot.dof_names,
                )
                controller.reset()
                task_done = False

            # ── 매 스텝 제어 ────────────────────────────────────
            if is_playing and not task_done:
                obs = task.get_observations()
                cube_position = obs["target_cube"]["position"]
                current_joints = obs["m0609_robot"]["joint_positions"]

                actions = controller.forward(
                    picking_position=cube_position,
                    placing_position=goal_position,
                    current_joint_positions=current_joints,
                    end_effector_offset=np.array([0.0, 0.0, 0.2]),
                )
                robot.apply_action(actions)

                if controller.is_done():
                    print("[완료] Pick & Place 성공!")
                    task_done = True
                    my_world.pause()

            was_playing = is_playing

        simulation_app.close()


if __name__ == "__main__":
    picknplace = DoosanPickNPlace()
    picknplace.main()