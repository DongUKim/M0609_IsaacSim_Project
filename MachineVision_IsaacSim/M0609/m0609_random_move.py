# SimulationApp 은 반드시 모든 omniverse import 보다 먼저 실행이 되어야함.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.asset.importer.urdf import _urdf

import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.robot_setup.assembler", True)

from isaacsim.robot_setup.assembler import RobotAssembler

from pxr import Usd

import omni.kit.commands
import omni.usd
import numpy as np
import os


# M0609, OnRobot-RG2 URDF 경로 지정
M0609_URDF_PATH   = "/home/deeptree/dev_ws/isaac_sim/src/doosan-robot2/urdf/m0609_isaac_sim.urdf"
ONROBOT_URDF_PATH = "/home/deeptree/dev_ws/isaac_sim/src/onrobot_rg2/urdf/onrobot_rg2.urdf"

EE_LINK_NAME      = "link_6"
GRIPPER_BASE_LINK = "angle_bracket"


# =====================================================================
#  유틸 함수들 (extension 버전과 동일)
# =====================================================================
def import_urdf(urdf_path, fix_base=True):
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF 파일이 존재하지 않습니다: {urdf_path}")

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints            = False
    import_config.convex_decomp                 = True
    import_config.import_inertia_tensor         = True
    import_config.fix_base                      = fix_base
    import_config.distance_scale                = 1.0
    import_config.default_drive_type            = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.default_drive_strength        = 1e4
    import_config.default_position_drive_damping = 1e3

    result, artic_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )

    if artic_path is None:
        print(f"  [ERROR] URDF import 실패: {urdf_path}")
        return None, None

    robot_root = artic_path.rsplit("/", 1)[0]
    if not robot_root:
        robot_root = artic_path

    print(f"  [OK] URDF import: {urdf_path}")
    print(f"       → articulation = {artic_path}")
    print(f"       → robot root   = {robot_root}")
    return robot_root, artic_path


def assemble_robot(stage, robot_base, robot_base_mount,
                   robot_attach, robot_attach_mount,
                   assembly_namespace, variant_name):
    stage = omni.usd.get_context().get_stage()
    assembler = RobotAssembler()
    assembler.begin_assembly(stage, robot_base, robot_base_mount,
                             robot_attach, robot_attach_mount,
                             assembly_namespace, variant_name)
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
#  컨트롤러 (extension 버전과 동일)
# =====================================================================
class DoosanPickPlaceController:

    STATES = [
        "OPEN_GRIPPER",
        "MOVE_TO_PRE_PICK",
        "MOVE_TO_PICK",
        "CLOSE_GRIPPER",
        "LIFT_UP",
        "MOVE_TO_PRE_PLACE",
        "MOVE_TO_PLACE",
        "OPEN_GRIPPER_2",
        "RETREAT",
        "DONE",
    ]

    def __init__(self, name, gripper, robot_articulation):
        self._name    = name
        self._gripper = gripper
        self._robot   = robot_articulation
        self._state   = 0
        self._step    = 0

        self._pre_pick_joints  = np.deg2rad([  0, -30,  90, -60, -90, 0])
        self._pick_joints      = np.deg2rad([  0, -10, 110, -100, -90, 0])
        self._lift_joints      = np.deg2rad([  0, -30,  90, -60, -90, 0])
        self._pre_place_joints = np.deg2rad([ 40, -30,  90, -60, -90, 0])
        self._place_joints     = np.deg2rad([ 40, -10, 110, -100, -90, 0])
        self._retreat_joints   = np.deg2rad([ 40, -30,  90, -60, -90, 0])

        self._durations = [40, 100, 80, 40, 100, 100, 80, 40, 100, 1]

    def forward(self, current_joint_positions, **kwargs):
        state_name = self.STATES[self._state]
        num_dof = len(current_joint_positions)

        if self._step == 0:
            self._enter_state(state_name)

        self._step += 1
        if self._step >= self._durations[self._state]:
            self._step = 0
            if self._state < len(self.STATES) - 1:
                self._state += 1

        return self._make_action(state_name, num_dof, current_joint_positions)

    def is_done(self):
        return self.STATES[self._state] == "DONE"

    def reset(self):
        self._state = 0
        self._step  = 0

    def _enter_state(self, state_name):
        if state_name in ("OPEN_GRIPPER", "OPEN_GRIPPER_2"):
            self._gripper.open()
        elif state_name == "CLOSE_GRIPPER":
            self._gripper.close()

    def _make_action(self, state_name, num_dof, current_positions):
        arm_map = {
            "OPEN_GRIPPER":      self._pre_pick_joints,
            "MOVE_TO_PRE_PICK":  self._pre_pick_joints,
            "MOVE_TO_PICK":      self._pick_joints,
            "CLOSE_GRIPPER":     self._pick_joints,
            "LIFT_UP":           self._lift_joints,
            "MOVE_TO_PRE_PLACE": self._pre_place_joints,
            "MOVE_TO_PLACE":     self._place_joints,
            "OPEN_GRIPPER_2":    self._place_joints,
            "RETREAT":           self._retreat_joints,
            "DONE":              self._retreat_joints,
        }
        target_arm = arm_map.get(state_name, self._pre_pick_joints)
        positions = current_positions.copy()
        positions[:6] = target_arm
        return ArticulationAction(joint_positions=positions)


# =====================================================================
#  메인
# =====================================================================
def main():
    # ── World 생성 ───────────────────────────────────────────
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    # ── Step 1: URDF Import ──────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 1] URDF Import")
    print("=" * 60)

    robot_root, robot_artic_path = import_urdf(M0609_URDF_PATH, fix_base=True)
    if robot_root is None:
        raise RuntimeError("M0609 URDF import 실패")

    gripper_root, gripper_artic_path = import_urdf(ONROBOT_URDF_PATH, fix_base=False)
    if gripper_root is None:
        raise RuntimeError("OnRobot URDF import 실패")

    # ── Step 2: RobotAssembler 결합 ──────────────────────────
    print("\n" + "=" * 60)
    print("[Step 2] RobotAssembler 결합")
    print("=" * 60)

    robot_ee_path     = find_prim_path_by_name(robot_root, EE_LINK_NAME)
    gripper_base_path = find_prim_path_by_name(gripper_root, GRIPPER_BASE_LINK)

    if robot_ee_path is None:
        robot_ee_path = f"{robot_root}/{EE_LINK_NAME}"
    if gripper_base_path is None:
        gripper_base_path = f"{gripper_root}/{GRIPPER_BASE_LINK}"

    print(f"  Robot EE:      {robot_ee_path}")
    print(f"  Gripper Base:  {gripper_base_path}")

    stage = omni.usd.get_context().get_stage()
    assemble_robot(
        stage,
        robot_root,          # base robot
        robot_ee_path,       # base mount
        gripper_root,        # attach robot
        gripper_base_path,   # attach mount
        "Gripper",           # namespace
        "m0609_rg2",         # variant name
    )
    print("  [OK] 결합 완료")

    for _ in range(10):
        simulation_app.update()

    # ── Step 3: ParallelGripper + SingleManipulator ──────────
    print("\n" + "=" * 60)
    print("[Step 3] ParallelGripper + SingleManipulator")
    print("=" * 60)

    gripper = ParallelGripper(
        end_effector_prim_path=robot_ee_path,
        joint_prim_names=["finger_joint",
                          "right_inner_knuckle_joint"],
        joint_opened_positions=np.array([0.0,  0.0]),
        joint_closed_positions=np.array([0.625, 0.625]),
        action_deltas=np.array([-0.625, -0.625]),
    )
    robot = my_world.scene.add(
        SingleManipulator(
            prim_path=robot_root,
            name="m0609_robot",
            end_effector_prim_path=robot_ee_path,
            gripper=gripper,
        )
    )

    # ── 큐브 추가 ────────────────────────────────────────────
    cube = my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=np.array([0.4, 0.0, 0.0515 / 2.0]),
            scale=np.array([0.0515, 0.0515, 0.0515]),
            color=np.array([0, 0, 1.0]),
        )
    )

    goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])

    my_world.scene.add(
        VisualCuboid(
            prim_path="/World/goal_marker",
            name="goal_marker",
            position=goal_position + np.array([0, 0, 0.21]),
            scale=np.array([0.06, 0.06, 0.001]),
            color=np.array([0, 1.0, 0]),
        )
    )

    # ── World reset 전 스테이지 안정화 ───────────────────────
    for _ in range(10):
        simulation_app.update()

    # ── World reset (physics 초기화) ─────────────────────────
    my_world.reset()

    # ── Joint 정보 출력 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Step 4] Joint 정보")
    print("=" * 60)
    print(f"  DOF: {robot.num_dof}")
    for i, name in enumerate(robot.dof_names):
        print(f"  [{i:2d}] {name}")
    print("=" * 60)

    # ── 컨트롤러 생성 ────────────────────────────────────────
    controller = DoosanPickPlaceController(
        name="doosan_pp_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
    )

    # ── 메인 시뮬레이션 루프 ─────────────────────────────────
    print("\n[Pick & Place 시작]\n")

    was_playing = False
    reset_needed = False
    warmup_frames = 0

    while simulation_app.is_running():
        my_world.step(render=True)

        is_playing = my_world.is_playing()

        # stop → play 전환 감지
        if is_playing and not was_playing:
            my_world.reset()  # ← physics view 재생성
            controller.reset()
            reset_needed = True
            warmup_frames = 0
            print("\n[Pick & Place 시작]\n")

        if is_playing and reset_needed:
            warmup_frames += 1
            if warmup_frames >= 10:  # 몇 프레임 안정화 대기
                reset_needed = False
                robot.gripper.set_joint_positions(
                    robot.gripper.joint_opened_positions
                )

        if is_playing and not reset_needed:
            if not controller.is_done():
                current_joint_positions = robot.get_joint_positions()
                if current_joint_positions is not None:
                    actions = controller.forward(
                        current_joint_positions=current_joint_positions,
                    )
                    robot.apply_action(actions)
            else:
                print("[완료] Pick & Place 성공!")
                my_world.pause()

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()