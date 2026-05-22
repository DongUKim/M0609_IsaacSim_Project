# M0609 Vision — Visual Servo Pick & Place

Doosan M0609 6-DOF 로봇팔 + OnRobot RG2 그리퍼를 NVIDIA Isaac Sim에서 시뮬레이션합니다.
손목 카메라(RealSense D455)로 빨간 블록을 탐지하고, visual servoing으로 정렬한 뒤 pick & place를 수행합니다.

## 시스템 요구사항

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 시리즈 (드라이버 535+) |
| Isaac Sim | 5.1.0 |
| Python | 3.11 (Isaac Sim 번들) |
| OpenCV | 4.x (`pip install opencv-python`) |
| SciPy | (`pip install scipy`) |

> OpenCV와 SciPy는 Isaac Sim의 번들 Python 환경에 설치해야 합니다.

## 설치

```bash
# 1. 저장소 클론
git clone <repo-url> ~/dev_ws/isaac_sim/src/m0609_vision
cd ~/dev_ws/isaac_sim/src/m0609_vision

# 2. Isaac Sim 번들 Python에 의존성 설치
~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh -m pip install opencv-python scipy
```

Isaac Sim은 별도로 설치되어 있어야 합니다. 설치 경로 기본값: `~/dev_ws/isaac_sim/isaacsim/`

## 실행

```bash
cd ~/dev_ws/isaac_sim/src/m0609_vision

~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh M0609/m0609_pick_place_visual.py
```

alias를 등록한 경우:
```bash
isaac-py M0609/m0609_pick_place_visual.py
```

> **주의**: 스크립트는 반드시 `m0609_vision/` 디렉터리에서 실행해야 상대 경로가 정상 동작합니다.

## 동작 흐름

```
MOVE_TO_HOME → Detecting → SEARCH → SERVO → PICK_AND_PLACE → DONE
```

| 단계 | 설명 |
|------|------|
| `MOVE_TO_HOME` | 홈 자세(`joint_3=70°`)로 이동, 카메라 시야 확보 |
| `Detecting` | `joint_5`를 90° 회전시켜 카메라 방향 조정 |
| `SEARCH` | 손목 카메라로 빨간 블록 탐지 대기 |
| `SERVO` | 픽셀 에러를 줄이도록 EE XY 위치 보정 (image-based visual servoing) |
| `PICK_AND_PLACE` | 탐지 위치 기준으로 pick & place 실행 |
| `DONE` | 완료, 시뮬레이션 일시정지 |

## 화면 구성

- **Isaac Sim 뷰포트**: 3D 시뮬레이션 뷰
- **OpenCV 창 (Camera View)**: 손목 카메라 영상 + 블록 마스크 + crosshair + 상태 텍스트

## 주요 파라미터 (`m0609_pick_place_visual.py` 상단)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `CAM_OFFSET_T` | `(0.0, 0.045, 0.05)` | 카메라 위치 오프셋 (m) |
| `CAM_OFFSET_RPY` | `(0.0, -90.0, 90.0)` | 카메라 자세 오프셋 (deg) |
| `CAM_RES` | `(640, 480)` | 카메라 해상도 |
| `HOME_JOINT_POSITIONS_DEG` | `[0, 0, 70, 0, 0, 0]` | 홈 자세 (각 joint, deg) |
| `_WS_X`, `_WS_Y` | `(0.2, 0.6)`, `(-0.5, 0.5)` | EE 작업 공간 클램프 범위 (m) |

## 프로젝트 구조

```
m0609_vision/
├── M0609/
│   ├── m0609_pick_place_visual.py      # 메인 실행 스크립트
│   ├── m0609_pick_place_controller.py  # PickPlaceController 래퍼
│   ├── m0609_rmpflow_controller.py     # RMPFlow 모션 정책 래퍼
│   ├── m0609_rg2_description.yaml      # Lula 로봇 디스크립터
│   ├── m0609_rmpflow_common.yaml       # RMPFlow 튜닝 파라미터
│   ├── vision_tracker.py               # HSV 기반 빨간 블록 탐지
│   ├── visual_servo_controller.py      # 픽셀 에러 → EE XY 보정 (P 제어)
│   ├── wrist_camera.py                 # Isaac Sim 카메라 센서 래퍼
│   ├── realsense_mount.py              # RealSense D455 mesh 부착
│   ├── camera_viewer.py                # OpenCV 카메라 뷰어
│   ├── doosan-robot2/urdf/             # M0609 URDF + 메시
│   └── onrobot_rg2/urdf/              # RG2 그리퍼 URDF + 메시
└── README.md
```

## 블록 색상 변경

`vision_tracker.py`의 `BlueBlockTracker`는 기본적으로 **빨간색**(HSV 0°·170° 부근)을 탐지합니다.
다른 색상을 탐지하려면 `lower_hsv1`, `upper_hsv1`, `lower_hsv2`, `upper_hsv2` 범위를 수정하세요.

시뮬레이션 내 블록 색상은 `m0609_pick_place_visual.py`에서 변경할 수 있습니다:
```python
color=np.array([1.0, 0.0, 0.0])  # RGB, 현재: 빨간색
```

## 트러블슈팅

**`SimulationApp` import 오류**
Isaac Sim이 설치된 Python으로 실행했는지 확인하세요. 시스템 `python3`로는 동작하지 않습니다.

**카메라 영상이 안 보임**
DISPLAY 환경변수가 설정되어 있어야 합니다. SSH 접속 시 `-X` 옵션을 사용하거나 로컬에서 실행하세요.

**`isaacsim.robot_setup.assembler` 오류**
Isaac Sim 5.1.0 버전이 아닌 경우 extension 이름이 다를 수 있습니다.

**그리퍼가 블록을 집지 못함**
`m0609_pick_place_visual.py`의 `events_dt` 파라미터와 그리퍼 drive stiffness를 조정하세요.
기본값은 `events_dt[3]=0.02` (그리퍼 닫기 ~0.83초), stiffness=1000 Nm/rad입니다.
