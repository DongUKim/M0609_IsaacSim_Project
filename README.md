# M0609 Isaac Sim Project

Doosan **M0609** 6-DOF 로봇팔 + **OnRobot RG2** 그리퍼를 **NVIDIA Isaac Sim**에서 다루는 세 가지 프로젝트 모음입니다. 실물 로봇 연동(MoveIt2), 비전 기반 Pick & Place, 강화학습(Lift)을 각각 독립 디렉터리로 구성했습니다.

---

## 구성

| 디렉터리 | 주제 | 한 줄 설명 |
|---|---|---|
| [`Link_IsaacSim/`](Link_IsaacSim) | Isaac Sim ↔ MoveIt2 ↔ 실물 로봇 연동 | Isaac Sim 시뮬레이션을 MoveIt2 경유로 실제 M0609 + RG2와 동기화 |
| [`MachineVision_IsaacSim/`](MachineVision_IsaacSim) | 비전 기반 Visual Servo Pick & Place | 손목 카메라(RealSense D455)로 블록을 탐지하고 visual servoing으로 pick & place |
| [`RL_Lifting_IsaacSim/`](RL_Lifting_IsaacSim) | 강화학습 Lift | Isaac Lab + rsl_rl(PPO)로 큐브 들어올리기 정책 학습 |

---

## 1. Link_IsaacSim — 실물 로봇 연동

Isaac Sim 물리 시뮬레이션을 MoveIt2를 통해 실제 Doosan M0609 + RG2 그리퍼와 동기화하는 통합 시스템입니다.

```
Isaac Sim ──/isaac_joint_states──▶ MoveIt2 ──joint_trajectory──▶ Real Robot (M0609 + RG2)
```

- `m0609_urdf/` — M0609 URDF / USD 자산
- `onrobot_rg2/` — RG2 그리퍼 자산
- `doosan_ros2/` — Doosan ROS 2 드라이버 / 인터페이스
- `IsaacSim-ros_workspaces/` — Isaac Sim용 ROS 2 워크스페이스 빌드 스크립트
- `isaac_moveit_m0609_rg2.py` — Isaac Sim ↔ MoveIt2 통합 실행 스크립트

자세한 빌드/실행은 [`Link_IsaacSim/README.md`](Link_IsaacSim/README.md) 참조.

## 2. MachineVision_IsaacSim — 비전 Pick & Place

손목 카메라(RealSense D455)로 빨간 블록을 탐지하고, image-based visual servoing으로 정렬한 뒤 pick & place를 수행합니다.

```
MOVE_TO_HOME → Detecting → SEARCH → SERVO → PICK_AND_PLACE → DONE
```

- `M0609/m0609_pick_place_visual.py` — 메인 실행 스크립트
- `M0609/vision_tracker.py` — HSV 기반 블록 탐지
- `M0609/visual_servo_controller.py` — 픽셀 에러 → EE XY 보정(P 제어)
- `m0609_rmpflow_controller.py`, `m0609_description.yaml`, `m0609_rmpflow_common.yaml` — RMPFlow 모션 정책

자세한 실행/파라미터는 [`MachineVision_IsaacSim/README.md`](MachineVision_IsaacSim/README.md) 참조.

## 3. RL_Lifting_IsaacSim — 강화학습 Lift

Isaac Lab + rsl_rl(PPO)로 M0609이 큐브를 잡아 들어올리는 정책을 학습합니다.

- `m0609_lift_code_ver2/` — 학습/재생 코드 (`train.py`, `play.py`, 환경 설정 `m0609_lift/`)
- `example_model_8500.pt` — 학습된 예시 정책 체크포인트
- Task ID: `Isaac-M0609-Lift-v0`(학습), `Isaac-M0609-Lift-Play-v0`(재생)

```bash
# 학습
python train.py --task Isaac-M0609-Lift-v0 --num_envs 4096 --headless

# 재생
python play.py --task Isaac-M0609-Lift-Play-v0 --num_envs 50 --checkpoint <model.pt>
```

자세한 사용법은 [`RL_Lifting_IsaacSim/m0609_lift_code_ver2/usage.md`](RL_Lifting_IsaacSim/m0609_lift_code_ver2/usage.md) 참조.

---

## 공통 환경

| 항목 | 사양 |
|---|---|
| 로봇 | Doosan M0609 (6-DOF) |
| 그리퍼 | OnRobot RG2 (병렬 그리퍼) |
| 카메라 (Vision) | Intel RealSense D455 |
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 |
| Python | 3.11 (Isaac Sim 번들) |
| ROS 배포판 | Humble (Link_IsaacSim) |
| GPU | NVIDIA RTX 시리즈 (드라이버 535+) |

> Isaac Sim 스크립트는 시스템 `python3`가 아닌 **Isaac Sim 번들 Python**(`python.sh`)으로 실행해야 합니다.

---

## 빠른 시작

```bash
git clone https://github.com/DongUKim/M0609_IsaacSim_Project.git
cd M0609_IsaacSim_Project
```

각 하위 디렉터리의 README / `usage.md`에서 빌드·실행 절차를 확인하세요.

---

## 라이선스

[Apache License 2.0](LICENSE)
