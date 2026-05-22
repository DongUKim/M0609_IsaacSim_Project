# SimulationApp 은 반드시 모든 omniverse import 보다 먼저 실행되어야 함.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from pathlib import Path
import sys
import os

import cv2
import numpy as np
import omni.kit.app
import omni.kit.commands
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

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

from m0609_rmpflow_controller import RMPFlowController
from m0609_pick_place_controller import PickPlaceController
from realsense_mount import attach_realsense_d455
from wrist_camera import WristCamera
from vision_tracker import BlueBlockTracker
from visual_servo_controller import VisualServoController
from camera_viewer import CameraViewer

M0609_URDF_PATH = str(BASE_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ONROBOT_URDF_PATH = str(BASE_DIR / "onrobot_rg2/urdf/onrobot_rg2.urdf")
M0609_RMPFLOW_CONFIG_PATH = str(BASE_DIR / "m0609_rmpflow_common.yaml")
M0609_DESCRIPTION_PATH = str(BASE_DIR / "m0609_rg2_description.yaml")

EE_LINK_NAME = "link_6"
GRIPPER_BASE_LINK = "angle_bracket"
GRIPPER_GRASP_LINK = "gripper_body"
GRIP_JOINT_PATH = "/World/grip_fixed_joint"

# 카메라 offset (mesh + sensor 공유). 위치/자세 튜닝 시 이 값만 수정.
CAM_OFFSET_T = (0.0, 0.045, 0.05)
CAM_OFFSET_RPY = (0.0, -90.0, 90.0)
CAM_RES = (640, 480)

# OmniVision 카메라 기준으로 sensor 만 추가 회전 (mesh 무관).
# rpy_deg=(roll, pitch, yaw) [deg]  — Z=yaw 90° 는 이미지 90° 회전
CAM_SENSOR_EXTRA_RPY = (0.0, 0.0, 90.0)

# 블럭 탐지를 위해 먼저 이동하는 홈 자세
HOME_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_JOINT_POSITIONS_DEG = np.array([0.0, 0.0, 70.0, 0.0, 0.0, 0.0])
HOME_REACHED_JOINT_TOL_DEG = 1.0

RUN_MODE = "VISUAL_SERVO_CENTER_THEN_DESCEND"
PICK_CONTROLLER_INITIAL_HEIGHT = 0.25
PICK_CONTROLLER_EE_OFFSET = np.array([0.0, 0.0, 0.2])

# RMPFlow 가 튀지 않도록 EE workspace 클램프 범위
_WS_X = (0.2, 0.6)
_WS_Y = (-0.5, 0.5)

# 블럭 탐지 홈 자세 도달 후 손목 관절을 조정하는 설정
HOME_JOINT_5_NAME = "joint_5"
HOME_JOINT_5_OFFSET_DEG = 90.0
HOME_SPIN_DURATION_SEC = 4.0
CONTROL_DT = 1.0 / 60.0
SERVO_PIXEL_TO_WORLD_XY = np.array([
    [0.0, -1.0],
    [-1.0, 0.0],
])


# =====================================================================
# 유틸 함수 (원본과 동일)
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


def assemble_robot(stage, robot_base, robot_base_mount,
                   robot_attach, robot_attach_mount,
                   assembly_namespace, variant_name):
    assembler = RobotAssembler()
    assembler.begin_assembly(
        stage, robot_base, robot_base_mount,
        robot_attach, robot_attach_mount,
        assembly_namespace, variant_name,
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


def _attach_cube_to_link(stage, joint_path, link_path, cube_path):
    """Phase 4 진입 시 큐브를 그리퍼 링크에 FixedJoint 로 결속.

    PhysX 기반 마찰 그립이 가속/측면 핀치력에 약하므로 lift 동안 강제 부착으로 회피한다.
    현재의 cube↔link 상대 pose 를 캡처하여 joint local frame 으로 사용.
    """
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    link_prim = stage.GetPrimAtPath(link_path)
    cube_prim = stage.GetPrimAtPath(cube_path)
    if not link_prim.IsValid() or not cube_prim.IsValid():
        print(f"[grip_joint] invalid prim — link={link_path} cube={cube_path}")
        return False

    link_xf = UsdGeom.Xformable(link_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    cube_xf = UsdGeom.Xformable(cube_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rel = cube_xf * link_xf.GetInverse()
    rel_pos = rel.ExtractTranslation()
    rel_rot = rel.ExtractRotationQuat()
    rot_imag = rel_rot.GetImaginary()

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(cube_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_pos))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(
        rel_rot.GetReal(),
        float(rot_imag[0]), float(rot_imag[1]), float(rot_imag[2]),
    ))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    print(f"[grip_joint] attached: {cube_path} → {link_path}")
    return True


def _detach_grip_joint(stage, joint_path):
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
        print("[grip_joint] detached")


def _apply_ee_target(cspace_controller, target_pos, robot, target_orientation=None):
    """RMPFlow cspace controller 로 EE 를 target_pos 로 이동."""
    actions = cspace_controller.forward(
        target_end_effector_position=target_pos,
        target_end_effector_orientation=target_orientation,
    )
    robot.apply_action(actions)


def _find_joint_index(robot, joint_name, fallback_index=None):
    if joint_name in robot.dof_names:
        return robot.dof_names.index(joint_name)
    for index, dof_name in enumerate(robot.dof_names):
        if dof_name.endswith(joint_name):
            return index
    if fallback_index is not None and fallback_index < len(robot.dof_names):
        return fallback_index
    raise RuntimeError(f"{joint_name} DOF 를 찾을 수 없습니다: {robot.dof_names}")


def _find_joint_indices(robot, joint_names):
    return np.array([
        _find_joint_index(robot, joint_name, fallback_index=index)
        for index, joint_name in enumerate(joint_names)
    ])


def _get_home_joint_5_target(start_joint_positions, joint_5_index, elapsed_sec):
    progress = min(elapsed_sec / HOME_SPIN_DURATION_SEC, 1.0)
    joint_5_offset = np.deg2rad(HOME_JOINT_5_OFFSET_DEG) * progress
    return np.array([start_joint_positions[joint_5_index] + joint_5_offset])



# =====================================================================
# Task
# =====================================================================
class DoosanPickPlaceTask(BaseTask):

    def __init__(self, name, cube_initial_position=None, goal_position=None):
        super().__init__(name=name, offset=None)
        self._goal_position = (
            goal_position if goal_position is not None
            else np.array([0.55, -0.35, 0.0])
        )
        self._cube_initial_position = (
            cube_initial_position if cube_initial_position is not None
            else np.array([0.30, 0.0, 0.0515 / 2.0])
        )
        self._task_achieved = False
        self._wrist_camera = None

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        scene.add_default_ground_plane()

        # ── Step 1: URDF Import ──────────────────────────────
        print("\n" + "=" * 60)
        print("[Step 1] URDF Import")
        print("=" * 60)

        robot_root, _ = import_urdf(M0609_URDF_PATH, fix_base=True)
        gripper_root, _ = import_urdf(ONROBOT_URDF_PATH, fix_base=False)

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
            robot_root, robot_ee_path,
            gripper_root, gripper_base_path,
            "Gripper", "m0609_rg2",
        )
        print("  [OK] 결합 완료")

        robot_ee_path = find_prim_path_by_name(robot_root, EE_LINK_NAME)
        print(f"  assembled ee path = {robot_ee_path}")

        self._gripper_body_path = find_prim_path_by_name(robot_root, GRIPPER_GRASP_LINK)
        print(f"  gripper body path = {self._gripper_body_path}")

        # ── Gripper joint drive 설정 (URDF effort limit 50 Nm 기준) ──
        for joint_name in ["finger_joint", "right_inner_knuckle_joint"]:
            joint_path = find_prim_path_by_name(robot_root, joint_name)
            if joint_path:
                joint_prim = stage.GetPrimAtPath(joint_path)
                for drive_type in ["angular", "linear"]:
                    drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
                    if drive:
                        drive.GetMaxForceAttr().Set(100.0)
                        drive.GetStiffnessAttr().Set(1000.0)
                        drive.GetDampingAttr().Set(50.0)
                        print(f"  [OK] {drive_type} drive 설정: {joint_path}")

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
            joint_closed_positions=np.array([0.8, 0.8]),
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
                color=np.array([1.0, 0.0, 0.0]),
                mass=0.01,
                physics_material=cube_material,
            )
        )

        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker",
                name="goal_marker",
                position=self._goal_position,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=4.0,
            dynamic_friction=3.0,
            restitution=0.0,
        )

        gripper_contact_links = [
            "left_inner_finger",
            "right_inner_finger",
            "left_inner_knuckle",
            "right_inner_knuckle",
            "left_outer_knuckle",
            "right_outer_knuckle",
        ]
        for i, link_name in enumerate(gripper_contact_links):
            link_path = find_prim_path_by_name(robot_root, link_name)
            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"gripper_geom_{i}",
                ).apply_physics_material(finger_material)
                print(f"  [OK] friction 적용: {link_path}")
            else:
                print(f"  [SKIP] 경로 없음: {link_name}")

        # ── Step 4: RealSense mesh + WristCamera ─────────────
        print("\n" + "=" * 60)
        print("[Step 4] RealSense D455 + WristCamera")
        print("=" * 60)

        gripper_camera_parent = find_prim_path_by_name(robot_root, GRIPPER_BASE_LINK)
        if gripper_camera_parent is None:
            raise RuntimeError(
                f"{GRIPPER_BASE_LINK} prim 을 찾을 수 없습니다 (robot_root={robot_root}). "
                "RobotAssembler 결합이 완료된 후 호출되는지 확인하세요."
            )
        print(f"  Camera parent = {gripper_camera_parent}")

        self._realsense_prim_path = attach_realsense_d455(
            parent_prim_path=gripper_camera_parent,
            child_name="realsense_d455",
            translation=CAM_OFFSET_T,
            rpy_deg=CAM_OFFSET_RPY,
        )

        # USD reference 가 해결될 때까지 몇 프레임 대기한 뒤 물리 비활성화
        for _ in range(5):
            simulation_app.update()
        _stage = omni.usd.get_context().get_stage()
        for _prim in Usd.PrimRange(_stage.GetPrimAtPath(self._realsense_prim_path)):
            if _prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(_prim).GetRigidBodyEnabledAttr().Set(False)
                print(f"  [OK] RigidBodyAPI 비활성화: {_prim.GetPath()}")
            if _prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(_prim).GetCollisionEnabledAttr().Set(False)
                print(f"  [OK] CollisionAPI 비활성화: {_prim.GetPath()}")

        # RealSense D455 USD 내장 OmniVision 카메라를 직접 사용
        _OV_CAM_NAME = "Camera_OmniVision_OV9782_Color"
        ov_cam_path = find_prim_path_by_name(self._realsense_prim_path, _OV_CAM_NAME)
        if ov_cam_path:
            print(f"  Using built-in camera: {ov_cam_path}")
            # CAM_SENSOR_EXTRA_RPY 를 prim 에 직접 오버라이드 (mesh 무관, sensor 만 회전)
            from pxr import Vt
            _cam_prim = _stage.GetPrimAtPath(ov_cam_path)
            _xf = UsdGeom.Xformable(_cam_prim)
            _existing = [op.GetOpName() for op in _xf.GetOrderedXformOps()]
            _rot_op = _xf.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat, opSuffix="extra")
            _rot_op.Set(float(CAM_SENSOR_EXTRA_RPY[2]))
            _cam_prim.GetAttribute("xformOpOrder").Set(
                Vt.TokenArray(_existing + [_rot_op.GetOpName()])
            )
            print(f"  [OK] camera extra yaw = {CAM_SENSOR_EXTRA_RPY[2]}°")
            self._wrist_camera = WristCamera.from_existing_prim(
                prim_path=ov_cam_path,
                resolution=CAM_RES,
            )
        else:
            print(f"  {_OV_CAM_NAME} not found — creating custom sensor")
            self._wrist_camera = WristCamera(
                parent_prim_path=self._realsense_prim_path,
                name="wrist_rgb",
                resolution=CAM_RES,
                translation=(0.0, 0.0, 0.0),
                rpy_deg=CAM_SENSOR_EXTRA_RPY,
            )
        print(f"  WristCamera prim = {self._wrist_camera._prim_path}")

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
            color=np.array([1.0, 0.0, 0.0])
        )
        self._task_achieved = False
        # world.reset() 후 RealSense USD 내 RigidBodyAPI 비활성화
        # (set_up_scene 시점에는 reference 내부 prim 이 아직 미해결일 수 있음)
        if hasattr(self, "_realsense_prim_path") and self._realsense_prim_path:
            stage = omni.usd.get_context().get_stage()
            for prim in Usd.PrimRange(stage.GetPrimAtPath(self._realsense_prim_path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)


# =====================================================================
# 메인
# =====================================================================
class DoosanPickNPlace:

    def __init__(self):
        pass

    def _init_robot(self, my_world, robot):
        robot.initialize()
        robot.gripper.initialize(
            physics_sim_view=my_world.physics_sim_view,
            articulation_apply_action_func=robot.apply_action,
            get_joint_positions_func=robot.get_joint_positions,
            set_joint_positions_func=robot.set_joint_positions,
            dof_names=robot.dof_names,
        )

    def main(self):
        my_world = World(stage_units_in_meters=1.0)

        task = DoosanPickPlaceTask(name="doosan_pick_place_task")
        my_world.add_task(task)
        my_world.reset()

        robot = my_world.scene.get_object("m0609_robot")

        self._init_robot(my_world, robot)
        task._wrist_camera.initialize()

        print("\n" + "=" * 60)
        print("[Step 5] Joint 정보")
        print("=" * 60)
        print(f"  DOF: {robot.num_dof}")
        for i, name in enumerate(robot.dof_names):
            print(f"  [{i:2d}] {name}")
        home_joint_indices = _find_joint_indices(robot, HOME_JOINT_NAMES)
        home_joint_positions = np.deg2rad(HOME_JOINT_POSITIONS_DEG)
        home_reached_joint_tol = np.deg2rad(HOME_REACHED_JOINT_TOL_DEG)
        joint_5_index = _find_joint_index(robot, HOME_JOINT_5_NAME, fallback_index=4)
        print("  HOME joints:")
        for joint_name, joint_index, joint_deg in zip(
                HOME_JOINT_NAMES, home_joint_indices, HOME_JOINT_POSITIONS_DEG):
            print(f"    {joint_name}: [{joint_index}] {joint_deg:.1f} deg")
        print(f"  HOME joint_5: [{joint_5_index}] {robot.dof_names[joint_5_index]}")
        print("=" * 60)

        cspace_controller = RMPFlowController(
            name="m0609_visual_servo_rmpflow_controller",
            robot_articulation=robot,
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=EE_LINK_NAME,
        )

        pick_place_controller = PickPlaceController(
            name="m0609_pick_place_controller",
            gripper=robot.gripper,
            robot_articulation=robot,
            end_effector_initial_height=PICK_CONTROLLER_INITIAL_HEIGHT,
            events_dt=[0.008, 0.005, 0.02, 0.02, 0.001, 0.01, 0.005, 0.05, 0.008, 0.08],
            urdf_path=M0609_URDF_PATH,
            robot_description_path=M0609_DESCRIPTION_PATH,
            rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
            end_effector_frame_name=EE_LINK_NAME,
        )
        print(
            f"  Pick lift target z ~= "
            f"{PICK_CONTROLLER_INITIAL_HEIGHT + PICK_CONTROLLER_EE_OFFSET[2]:.3f} m"
        )

        tracker = BlueBlockTracker()
        servo = VisualServoController(
            image_size=CAM_RES,
            pixel_to_world_xy=SERVO_PIXEL_TO_WORLD_XY,
        )
        viewer = CameraViewer(enabled=True)

        state = "MOVE_TO_HOME"
        home_spin_start_joints = None
        home_spin_elapsed = 0.0
        home_spin_last_log_sec = -1
        servo_hold_z = None
        servo_hold_orientation = None
        was_playing = False
        prev_pick_event = -1
        cube_prim_path = "/World/target_cube"
        stage = omni.usd.get_context().get_stage()

        print(f"\n[Visual Tracking 시작] mode={RUN_MODE}\n")

        try:
            while simulation_app.is_running():
                my_world.step(render=True)
                is_playing = my_world.is_playing()

                if is_playing and not was_playing:
                    my_world.reset()
                    self._init_robot(my_world, robot)
                    task._wrist_camera.initialize()
                    cspace_controller.reset()
                    pick_place_controller.reset()
                    servo.reset()
                    _detach_grip_joint(stage, GRIP_JOINT_PATH)
                    state = "MOVE_TO_HOME"
                    home_spin_start_joints = None
                    home_spin_elapsed = 0.0
                    home_spin_last_log_sec = -1
                    servo_hold_z = None
                    servo_hold_orientation = None
                    prev_pick_event = -1
                    was_playing = True
                    continue

                if not is_playing:
                    was_playing = False
                    continue

                # ── 카메라 프레임 + 검출 ─────────────────────
                rgb = task._wrist_camera.get_rgb()
                det = None
                if rgb is not None:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    det = tracker.detect(bgr)

                # ── OpenCV 별도 윈도우 갱신 ──────────────────
                obs = task.get_observations()
                current_joints = obs["m0609_robot"]["joint_positions"]
                ee_pos, ee_orientation = robot.end_effector.get_world_pose()
                current_xy = ee_pos[:2]

                key = viewer.update(
                    rgb, det, state_str=state,
                    extra_lines=[
                        f"mode={RUN_MODE}",
                        f"ee_xy={current_xy.round(3)}",
                        f"ee_z={ee_pos[2]:.3f}",
                        f"locked={servo.is_locked()}",
                        f"frame_ok={rgb is not None}",
                    ],
                )
                if key == ord('q'):
                    break

                # ── 상태기계 ─────────────────────────────────
                if state == "MOVE_TO_HOME":
                    robot.set_joint_positions(
                        home_joint_positions,
                        joint_indices=home_joint_indices,
                    )
                    home_joint_error = np.max(np.abs(
                        current_joints[home_joint_indices] - home_joint_positions
                    ))
                    if home_joint_error < home_reached_joint_tol:
                        print("[state] MOVE_TO_HOME → Detecting")
                        home_spin_start_joints = None
                        home_spin_elapsed = 0.0
                        home_spin_last_log_sec = -1
                        state = "Detecting"

                elif state == "Detecting":
                    if home_spin_start_joints is None:
                        home_spin_start_joints = current_joints.copy()
                        home_spin_elapsed = 0.0
                        print("[state] Detecting: joint_5 +90deg start")

                    home_spin_elapsed = min(
                        home_spin_elapsed + CONTROL_DT,
                        HOME_SPIN_DURATION_SEC,
                    )
                    target_positions = _get_home_joint_5_target(
                        home_spin_start_joints,
                        joint_5_index,
                        home_spin_elapsed,
                    )
                    robot.set_joint_positions(
                        target_positions,
                        joint_indices=np.array([joint_5_index]),
                    )
                    log_sec = int(home_spin_elapsed)
                    if log_sec != home_spin_last_log_sec:
                        target_deg = np.rad2deg(target_positions)
                        current_deg = np.rad2deg(current_joints[joint_5_index])
                        print(
                            f"[HOME_SPIN] joint_5 current={current_deg:.1f}deg "
                            f"target={target_deg[0]:.1f}deg"
                        )
                        home_spin_last_log_sec = log_sec

                    if home_spin_elapsed >= HOME_SPIN_DURATION_SEC:
                        print("[state] HOME_SPIN → SEARCH")
                        state = "SEARCH"

                elif state == "SEARCH":
                    if det is not None and det.found:
                        servo_hold_z = float(ee_pos[2])
                        servo_hold_orientation = ee_orientation.copy()
                        print(
                            f"[state] SEARCH → SERVO  "
                            f"hold_z={servo_hold_z:.3f}"
                        )
                        state = "SERVO"

                elif state == "SERVO":
                    if det is not None:
                        target_xy, err_px = servo.update(current_xy, det)
                    else:
                        servo.reset()
                        target_xy = current_xy.copy()
                        err_px = float("inf")
                    target_xy[0] = np.clip(target_xy[0], *_WS_X)
                    target_xy[1] = np.clip(target_xy[1], *_WS_Y)
                    target = np.array([target_xy[0], target_xy[1], servo_hold_z])
                    _apply_ee_target(
                        cspace_controller,
                        target,
                        robot,
                        target_orientation=servo_hold_orientation,
                    )
                    if servo.is_locked():
                        cube_position = obs["target_cube"]["position"]
                        print(
                            f"[state] SERVO → PICK_AND_PLACE  "
                            f"cube_pos={cube_position.round(3)}"
                        )
                        pick_place_controller.reset()
                        state = "PICK_AND_PLACE"

                elif state == "PICK_AND_PLACE":
                    cube_position = obs["target_cube"]["position"]
                    actions = pick_place_controller.forward(
                        picking_position=cube_position,
                        placing_position=task._goal_position,
                        current_joint_positions=current_joints,
                        end_effector_offset=PICK_CONTROLLER_EE_OFFSET,
                    )
                    robot.apply_action(actions)
                    _ev = getattr(pick_place_controller, "_event", -1)

                    # event 3 (close) 종료 → event 4 (lift) 진입: cube 를 그리퍼에 결속
                    if _ev == 4 and prev_pick_event == 3:
                        if task._gripper_body_path is not None:
                            _attach_cube_to_link(
                                stage,
                                GRIP_JOINT_PATH,
                                task._gripper_body_path,
                                cube_prim_path,
                            )
                    # event 7 (open) 종료 → event 8 (lift) 진입: 결속 해제
                    # phase 7 동안 그리퍼는 열리지만 FixedJoint 가 큐브를 잡고 있으므로
                    # 큐브가 공중에서 떨어지지 않음. lift 직전에만 detach 하여 패드 위에 안착.
                    elif _ev == 8 and prev_pick_event == 7:
                        _detach_grip_joint(stage, GRIP_JOINT_PATH)
                    prev_pick_event = _ev

                    if my_world.current_time_step_index % 30 == 0:
                        print(
                            f"[P&P] event={_ev}  "
                            f"cube_z={cube_position[2]:.4f}  "
                            f"cube_xy=({cube_position[0]:.3f},{cube_position[1]:.3f})  "
                            f"ee_z={ee_pos[2]:.4f}"
                        )
                    if pick_place_controller.is_done():
                        print("[완료] Pick & Place 완료!")
                        state = "DONE"
                        my_world.pause()

                # DONE: my_world.step(render=True) 만 계속 실행

        finally:
            viewer.close()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            simulation_app.close()


if __name__ == "__main__":
    DoosanPickNPlace().main()
