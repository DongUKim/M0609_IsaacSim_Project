# M0609 + RG2 Visual Tracking 구현 명세

> Isaac Sim에서 동작하는 기존 Doosan M0609 + OnRobot RG2 pick&place 코드에
> **wrist-mounted 카메라 + OpenCV 기반 visual servoing**을 추가하기 위한
> Claude Code 작업 명세서.

---

## 0. 작업 개요

### 0.1 목표
1. **RealSense D455 mesh 를 그리퍼 `angle_bracket` 에 시각적으로 부착**한다.
   시뮬 뷰포트에서 카메라 모델이 그리퍼에 붙은 채로 함께 움직여야 한다.
2. 동일 위치에 RGB 카메라 sensor 를 띄워 eye-in-hand 영상을 얻는다.
   (mesh = 보여주기용, sensor = 이미지 획득용; 둘이 같은 transform 공유)
3. 시뮬레이션 매 스텝마다 카메라 RGB 프레임을 OpenCV 로 가져온다.
4. 색상(HSV) 기반으로 타겟 블럭의 픽셀 중심 `(cx, cy)` 를 추출한다.
5. 픽셀 에러 `(cx - W/2, cy - H/2)` 를 0으로 만드는 방향으로
   end-effector 의 XY 평면 위치를 보정한다 (image-based visual servoing).
6. 화면 중앙 ±tolerance 픽셀 안에 블럭이 들어오면 "lock" 상태로 간주하고,
   기존 pick & place 시퀀스를 그 위치에서 이어서 실행한다.
7. **Isaac Sim 뷰포트와 완전히 분리된 별도의 OpenCV 윈도우** 로 카메라 영상 +
   mask + crosshair + 상태 텍스트를 실시간 표시한다 (`camera_viewer.py` 모듈로 캡슐화).

### 0.2 비목표 (이번 스코프 외)
- 캘리브레이션된 정확한 카메라 내부 파라미터 추정 (대신 Isaac Sim의 focal length 그대로 사용).
- Z 방향(깊이) 비주얼 서보잉 — Z 는 기존처럼 정해진 approach height 로 내려간다.
- 다중 객체 추적 — 한 번에 가장 큰 파란색 컨투어 1개만 추적한다.
- ROS 연동.

### 0.3 동작 모드 (런타임 상태기계)
다음 4가지 상태를 명시적으로 가진다.

| 상태 | 설명 | 종료 조건 |
|-----|------|-----------|
| `SEARCH` | 사전에 정의한 search pose 로 이동, 시야에 블럭이 들어올 때까지 대기 | mask area > min_area |
| `SERVO` | 픽셀 에러를 줄이도록 EE의 XY 를 보정 | 픽셀 에러 < tolerance, N프레임 연속 |
| `PICK_PLACE` | 현재 EE XY 위치를 picking_position 으로 고정, 기존 PickPlaceController 실행 | controller.is_done() |
| `DONE` | 정지, 시뮬레이션은 계속 렌더 | (사용자 입력 또는 reset) |

`SEARCH → SERVO → PICK_PLACE → DONE` 순으로 전이한다.

---

## 1. 기존 코드 구조 (수정 시 참고)

```
project_root/
├── m0609_pick_place_fixed_target.py     # 메인 실행 스크립트
├── m0609_pick_place_controller.py       # PickPlaceController wrapper
├── m0609_rmpflow_controller.py          # RMPFlow 모션 정책
├── m0609_description.yaml               # Lula 로봇 디스크립터
├── m0609_rmpflow_common.yaml            # RMPflow 파라미터
├── doosan-robot2/urdf/m0609_isaac_sim.urdf
└── onrobot_rg2/urdf/onrobot_rg2.urdf
```

핵심 흐름 요약:
- `import_urdf()` 로 로봇/그리퍼 URDF를 따로 import.
- `RobotAssembler` 로 `link_6` ↔ `angle_bracket` 결합.
- `ParallelGripper` + `SingleManipulator` 등록.
- `DoosanPickPlaceTask` 가 cube/goal/마찰을 셋업.
- 메인 루프에서 `controller.forward(picking_position, placing_position, ...)` 호출.

수정할 때 **이 구조는 깨지 않는다.** 카메라 + 비전은 이 위에 *덮어씌우는* 방식.

---

## 2. 새로 만들/수정할 파일

| 파일 | 종류 | 역할 |
|------|------|------|
| `realsense_mount.py` | **신규** | Isaac Sim 내장 RealSense D455 USD 를 angle_bracket 에 reference/parent |
| `wrist_camera.py` | **신규** | 카메라 sensor prim 생성/초기화/RGB 획득 헬퍼 |
| `vision_tracker.py` | **신규** | OpenCV HSV masking → centroid 추출 |
| `visual_servo_controller.py` | **신규** | 픽셀 에러 → EE XY target 보정기 (P 제어) |
| `camera_viewer.py` | **신규** | OpenCV imshow 기반 디버그 뷰어 (Isaac Sim 과 분리된 별도 윈도우) |
| `m0609_pick_place_visual.py` | **신규** | `m0609_pick_place_fixed_target.py` 를 베이스로 한 visual tracking 메인 |
| `m0609_pick_place_controller.py` | 그대로 | 변경 없음 |
| `m0609_rmpflow_controller.py` | 그대로 | 변경 없음 |
| `m0609_description.yaml` | 그대로 | 변경 없음 |
| `m0609_rmpflow_common.yaml` | 그대로 | 변경 없음 |

URDF는 건드리지 않는다 — 카메라는 코드에서 동적으로 prim을 만들어 그리퍼 link 아래에 parent 시킨다.

---

## 3. 단계별 구현 계획

### Step 1A — `realsense_mount.py` (시각 모델 부착)

**책임**
- Isaac Sim 내장 RealSense D455 USD 를 그리퍼 `angle_bracket` 아래에 reference 로 attach.
- 시뮬레이션 뷰포트에서 카메라 mesh 가 그리퍼와 함께 움직이도록 transform 고정.
- mesh 의 위치/회전 offset 을 통일된 단일 값으로 노출 → Step 1B 의 sensor 와 동일 값 공유.

**왜 RobotAssembler 가 아니라 USD reference 인가**
- `RobotAssembler` 는 articulation ↔ articulation 결합용 (fixed joint 생성).
- RealSense USD 는 articulation 이 아니라 단순 mesh + rigid prim 묶음이라
  reference 후 부모 Xform 의 자식으로 두는 것만으로 충분히 "부착된 것처럼" 동작한다.
- 부모(`angle_bracket`) 가 이미 RobotAssembler 로 `link_6` 에 결합되어 있어,
  팔이 움직이면 자동으로 RealSense 까지 따라온다.

**구현 가이드**
```python
# realsense_mount.py
from pxr import Usd, UsdGeom, Gf
import omni.usd

# Isaac Sim 자산 루트 탐색 (버전별 호환)
def _get_assets_root():
    try:
        from isaacsim.storage.native import get_assets_root_path  # 4.5+ / 5.x
        return get_assets_root_path()
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # 4.x
        return get_assets_root_path()

# 내장 RealSense D455 USD 의 표준 경로
# (버전에 따라 'rsd455.usd' 가 'Isaac/Sensors/Intel/RealSense/' 아래 위치)
REALSENSE_D455_USD_REL = "/Isaac/Sensors/Intel/RealSense/rsd455.usd"

def attach_realsense_d455(parent_prim_path: str,
                          child_name: str = "realsense_d455",
                          translation=(0.05, 0.0, 0.02),
                          rpy_deg=(180.0, 0.0, 0.0),
                          usd_path: str | None = None) -> str:
    """
    parent_prim_path 아래에 RealSense D455 mesh 를 reference 로 부착.
    반환: 생성된 RealSense Xform 의 prim path (Step 1B 에서 sensor 위치 기준으로 사용)
    """
    stage = omni.usd.get_context().get_stage()
    if usd_path is None:
        root = _get_assets_root()
        if root is None:
            raise RuntimeError("Isaac Sim assets root 를 찾을 수 없습니다. Nucleus 연결 확인.")
        usd_path = root + REALSENSE_D455_USD_REL

    rs_prim_path = f"{parent_prim_path}/{child_name}"

    # 이미 있으면 정리 (재실행 안전성)
    existing = stage.GetPrimAtPath(rs_prim_path)
    if existing and existing.IsValid():
        stage.RemovePrim(rs_prim_path)

    rs_prim = stage.DefinePrim(rs_prim_path, "Xform")
    rs_prim.GetReferences().AddReference(usd_path)

    # transform 적용 (translate → rotateXYZ 순)
    xform = UsdGeom.Xformable(rs_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rpy_deg))

    print(f"[realsense_mount] D455 attached at {rs_prim_path}")
    print(f"                  source USD = {usd_path}")
    return rs_prim_path
```

**주의사항**
- Isaac Sim 의 RealSense 에셋은 보통 `/NVIDIA/Assets/Isaac/<ver>/Isaac/Sensors/Intel/RealSense/`
  아래에 위치하지만 버전마다 다르다. `get_assets_root_path()` 가 None 을 반환하면
  로컬 캐시 경로(`~/.local/share/ov/pkg/...`) 를 직접 `usd_path` 로 넘겨도 된다.
- D455 가 없으면 D435 (`rsd435.usd`) 로 fallback 하도록 try 문 추가 가능.
- mesh 안에 정의된 default camera prim 을 sensor 로 그대로 쓰는 것도 가능하지만,
  USD asset 의 내부 구조가 버전마다 변할 수 있어 **권장 X**.
  대신 Step 1B 에서 동일 위치/자세로 별도 Camera sensor 를 만든다.

### Step 1B — `wrist_camera.py` (RGB sensor)

**책임**
- Step 1A 에서 만든 RealSense mesh 와 **동일한 부모/transform** 을 가지는 카메라 sensor prim 생성.
- 카메라 위치/자세 offset 적용 (그리퍼 정면을 향하도록).
- `get_rgb()` 호출 시 `(H, W, 3)` uint8 ndarray 반환.

**구현 가이드**
```python
import numpy as np
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation as R

class WristCamera:
    def __init__(self, parent_prim_path: str,
                 name: str = "wrist_camera",
                 resolution=(640, 480),
                 frequency: int = 30,
                 # angle_bracket 기준 offset (튜닝 필요)
                 translation=(0.05, 0.0, 0.02),
                 # 카메라 -Z 가 그리퍼 정면을 향하도록 (Isaac Sim 카메라 convention)
                 # angle_bracket frame 에 따라 rpy(deg) 조정 필요
                 rpy_deg=(180.0, 0.0, 0.0)):
        self._prim_path = f"{parent_prim_path}/{name}"
        quat_xyzw = R.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        # Isaac Sim Camera 는 quaternion (w, x, y, z) 순서
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        self.camera = Camera(
            prim_path=self._prim_path,
            name=name,
            resolution=resolution,
            frequency=frequency,
            translation=np.array(translation),
            orientation=quat_wxyz,
        )
        self.resolution = resolution

    def initialize(self):
        # World.reset() 이후 호출
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()  # 깊이 (디버그용)

    def get_rgb(self):
        rgba = self.camera.get_rgba()  # (H, W, 4)
        if rgba is None or rgba.size == 0:
            return None
        return rgba[..., :3].copy()

    @property
    def width(self):  return self.resolution[0]
    @property
    def height(self): return self.resolution[1]
```

**주의사항**
- Isaac Sim 5.x 에서는 `isaacsim.sensors.camera.Camera`. 4.x 는 `omni.isaac.sensor.Camera`.
  사용 중인 버전에 맞게 import 변경 (코드 상단에 try/except 로 둘 다 처리하면 안전).
- `frequency` 가 너무 높으면 simulation 가 느려진다. 30Hz 권장.
- `translation`, `rpy_deg` 는 **반드시 튜닝**해야 한다. Step 4 의 디버그 시각화에서 확인.
- **중요**: Step 1A 에서 RealSense mesh 에 사용한 `translation`, `rpy_deg` 와 **동일한 값** 을
  여기서도 써야 mesh 가 보여주는 위치와 실제 카메라 시점이 일치한다.
  → 메인에서 한 번만 정의하고 두 함수에 동일하게 넘기는 패턴 권장.
  ```python
  # 메인에서
  CAM_OFFSET_T   = (0.05, 0.0, 0.02)
  CAM_OFFSET_RPY = (180.0, 0.0, 0.0)
  attach_realsense_d455(parent, translation=CAM_OFFSET_T, rpy_deg=CAM_OFFSET_RPY)
  wrist_cam = WristCamera(parent, translation=CAM_OFFSET_T, rpy_deg=CAM_OFFSET_RPY)
  ```
- 더 정확하게는 sensor 의 부모를 Step 1A 에서 만든 RealSense Xform 으로 두고
  sensor 의 local transform 은 zero 로 두는 방식도 가능 (mesh 와 sensor 가 강제 동기화):
  ```python
  rs_prim_path = attach_realsense_d455(angle_bracket_path, ...)
  wrist_cam = WristCamera(parent_prim_path=rs_prim_path,
                          translation=(0,0,0), rpy_deg=(0,0,0))
  ```
  이 방식을 **권장** 한다 (transform 한 곳만 튜닝하면 됨).

### Step 2 — `vision_tracker.py`

**책임**
- BGR 이미지에서 파란색 블럭의 중심 픽셀 좌표 추정.
- 검출 신뢰도(컨투어 면적), mask 영상도 함께 반환.

```python
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class Detection:
    found: bool
    cx: float = 0.0
    cy: float = 0.0
    area: float = 0.0
    bbox: tuple = (0, 0, 0, 0)  # x, y, w, h
    mask: np.ndarray = None

class BlueBlockTracker:
    def __init__(self,
                 lower_hsv=(100, 120, 50),
                 upper_hsv=(130, 255, 255),
                 min_area=200,
                 morph_kernel=5):
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def detect(self, bgr: np.ndarray) -> Detection:
        if bgr is None:
            return Detection(found=False)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Detection(found=False, mask=mask)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_area:
            return Detection(found=False, area=area, mask=mask)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return Detection(found=False, mask=mask)
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(largest)
        return Detection(found=True, cx=cx, cy=cy, area=area,
                         bbox=(x, y, w, h), mask=mask)
```

**주의사항**
- `m0609_pick_place_fixed_target.py` 에서 cube 색상이 `[0.0, 0.0, 1.0]` (RGB 파랑).
  Isaac Sim의 디폴트 PBR 셰이딩에 따라 실제 픽셀 색이 어둡게 나올 수 있으므로
  `lower_hsv[1]` (S 하한) 를 100~150 사이에서 튜닝.
- Camera 의 출력은 RGB. OpenCV 는 BGR. **반드시 `cv2.cvtColor(rgb, COLOR_RGB2BGR)`** 로 변환 후 사용.

### Step 3 — `visual_servo_controller.py`

**책임**
- 픽셀 에러 → 카메라/EE frame 의 XY delta target 으로 변환.
- 단순 P 제어로 시작, 안정화되면 PI 추가.
- `update(current_ee_pos, detection) -> target_ee_pos` 인터페이스.

**좌표계 가정**
- 카메라가 거의 수직 아래를 본다고 가정 (그리퍼 끝 → 테이블).
- 화면 +x (오른쪽) 는 world `-y` 또는 `+y` 일 수 있음 → **부호는 실험으로 확정**.
- 따라서 매핑은:
  ```
  delta_world_x = -kp_x * (cx - W/2)
  delta_world_y = -kp_y * (cy - H/2)
  ```
  부호는 디버그 화면 보면서 1회 결정.

```python
import numpy as np

class VisualServoController:
    def __init__(self,
                 image_size,                      # (W, H)
                 kp=0.0008,                       # px → m gain (튜닝)
                 max_step=0.02,                   # m, 1프레임당 최대 이동
                 tolerance_px=8,                  # 픽셀 에러 임계
                 lock_frames=15,                  # 연속 N 프레임 안정 시 lock
                 axis_sign=(-1.0, -1.0)):         # (sign_x, sign_y) — 실험으로 결정
        self.W, self.H = image_size
        self.kp = kp
        self.max_step = max_step
        self.tolerance_px = tolerance_px
        self.lock_frames = lock_frames
        self.axis_sign = axis_sign
        self._stable_count = 0

    def reset(self):
        self._stable_count = 0

    def is_locked(self) -> bool:
        return self._stable_count >= self.lock_frames

    def update(self, current_ee_xy: np.ndarray, det) -> tuple:
        """
        returns (target_ee_xy: np.ndarray(2), error_px: float)
        det.found 가 False 면 현재 위치 유지.
        """
        if not det.found:
            self._stable_count = 0
            return current_ee_xy.copy(), float("inf")

        ex_px = det.cx - self.W / 2.0
        ey_px = det.cy - self.H / 2.0
        err_px = float(np.hypot(ex_px, ey_px))

        if err_px < self.tolerance_px:
            self._stable_count += 1
        else:
            self._stable_count = 0

        sx, sy = self.axis_sign
        dx = sx * self.kp * ex_px
        dy = sy * self.kp * ey_px

        # 클램프
        step = float(np.hypot(dx, dy))
        if step > self.max_step:
            scale = self.max_step / step
            dx *= scale
            dy *= scale

        target = current_ee_xy + np.array([dx, dy])
        return target, err_px
```

### Step 3.5 — `camera_viewer.py` (Isaac Sim 과 분리된 OpenCV 윈도우)

**책임**
- 카메라 RGB 프레임 + 검출 결과(bbox, centroid) + 상태 텍스트를 그려서
  `cv2.imshow` 로 별도 OS 윈도우에 표시.
- mask 영상은 보조 윈도우로 함께 표시.
- Isaac Sim 의 viewport (Kit GUI) 와는 **완전히 독립** — Isaac Sim 이 죽어도 코드 충돌 없음.
- 호출 인터페이스를 1줄로 단순화: `viewer.update(rgb, detection, state_str)`.

**왜 별도 모듈로 빼는가**
- 메인 루프가 너무 길어지지 않도록.
- 헤드리스(`headless=True`) 모드에서는 `imshow` 대신 디스크 저장으로 쉽게 swap 가능.
- 추후 multiprocessing 분리(아래 옵션)나 ROS topic publish 로 교체할 때 진입점이 한 곳.

**구현 가이드**
```python
# camera_viewer.py
import cv2
import numpy as np
from typing import Optional

class CameraViewer:
    """Isaac Sim 과 별개의 OpenCV 윈도우로 wrist 카메라를 표시."""

    def __init__(self,
                 window_name: str = "wrist_camera",
                 mask_window_name: str = "mask",
                 show_mask: bool = True,
                 enabled: bool = True):
        self.window_name = window_name
        self.mask_window_name = mask_window_name
        self.show_mask = show_mask
        self.enabled = enabled
        self._initialized = False

    def _ensure_init(self):
        if self._initialized or not self.enabled:
            return
        # cv2.WINDOW_NORMAL → 사용자 리사이즈 가능
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)
        if self.show_mask:
            cv2.namedWindow(self.mask_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.mask_window_name, 320, 240)
        self._initialized = True

    def update(self,
               rgb: Optional[np.ndarray],
               detection=None,
               state_str: str = "",
               extra_lines: Optional[list] = None) -> int:
        """
        매 시뮬 step 에서 호출.
        rgb: WristCamera.get_rgb() 결과 (RGB) 또는 None.
        detection: vision_tracker.Detection (None 가능).
        state_str: 상단에 표시할 상태 (e.g. "SEARCH" / "SERVO" / ...).
        extra_lines: 디버그 출력 라인 리스트.
        반환: cv2.waitKey 가 받은 키 (-1 if no input).
        """
        if not self.enabled:
            return -1
        self._ensure_init()
        if rgb is None:
            # 빈 화면 표시 (signal 살아있는지 확인용)
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(blank, "no frame", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(self.window_name, blank)
            return cv2.waitKey(1) & 0xFF

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        # crosshair (이미지 중앙)
        cv2.line(bgr, (W // 2, 0), (W // 2, H), (0, 255, 255), 1)
        cv2.line(bgr, (0, H // 2), (W, H // 2), (0, 255, 255), 1)

        # detection overlay
        if detection is not None and getattr(detection, "found", False):
            x, y, w, h = detection.bbox
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(bgr, (int(detection.cx), int(detection.cy)),
                       4, (0, 0, 255), -1)
            err_x = detection.cx - W / 2
            err_y = detection.cy - H / 2
            cv2.putText(bgr, f"err=({err_x:+.1f},{err_y:+.1f})px",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        # 상단 상태 텍스트
        if state_str:
            cv2.putText(bgr, f"state: {state_str}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 부가 디버그 라인
        if extra_lines:
            y0 = 50
            for i, line in enumerate(extra_lines):
                cv2.putText(bgr, str(line), (10, y0 + 20 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        cv2.imshow(self.window_name, bgr)

        if self.show_mask and detection is not None and detection.mask is not None:
            cv2.imshow(self.mask_window_name, detection.mask)

        # 1 ms 대기 — Isaac Sim 메인 루프를 막지 않음
        key = cv2.waitKey(1) & 0xFF
        return key

    def close(self):
        if self._initialized:
            cv2.destroyWindow(self.window_name)
            if self.show_mask:
                cv2.destroyWindow(self.mask_window_name)
            self._initialized = False
```

**사용 예시 (메인에서)**
```python
from camera_viewer import CameraViewer
viewer = CameraViewer(enabled=True)   # headless 면 enabled=False

# 매 step:
key = viewer.update(rgb, det, state_str=state,
                    extra_lines=[f"ee={current_xy.round(3)}",
                                 f"locked={servo.is_locked()}"])
if key == ord('q'):
    break
```

**주의사항**
- `cv2.imshow` 와 `cv2.waitKey` 는 **반드시 같은 스레드** 에서 호출. 메인 스레드에서 호출하는 게 정석.
- `cv2.waitKey(0)` 을 쓰면 시뮬이 멈춘다 → 항상 `cv2.waitKey(1)`.
- Linux + Wayland 환경에서 imshow 윈도우가 안 뜨면 `XDG_SESSION_TYPE=x11` 로 강제하거나
  `enabled=False` + `cv2.imwrite("/tmp/wrist_%05d.png", bgr)` 로 디버그.
- 키보드 'q' 입력으로 루프 종료를 받고 싶으면 위 `key == ord('q')` 패턴 사용.

**옵션: 별도 프로세스 분리 (필요 시)**
imshow 가 Isaac Sim Kit 의 GL 컨텍스트와 드물게 충돌하는 경우, viewer 를 별도 프로세스로 분리:
```python
# 의사코드
from multiprocessing import Process, Queue
def viewer_proc(q):
    while True:
        item = q.get()
        if item is None: break
        rgb, det, state = item
        # 같은 update 로직 수행, cv2.imshow + waitKey
```
메인은 Queue 에 frame 만 push. 처음에는 단일 스레드(위 기본 구현) 로 시작하고,
충돌이 관측될 때만 이 옵션으로 전환.

### Step 4 — `m0609_pick_place_visual.py` (메인)

**구조** (`m0609_pick_place_fixed_target.py` 를 복제 후 수정)

1. 기존 import 에 추가:
   ```python
   import cv2
   from realsense_mount import attach_realsense_d455
   from wrist_camera import WristCamera
   from vision_tracker import BlueBlockTracker
   from visual_servo_controller import VisualServoController
   from camera_viewer import CameraViewer
   ```

2. 파일 상단 상수에 카메라 offset 단일 정의:
   ```python
   # angle_bracket 기준 카메라 offset (mesh + sensor 가 공유)
   CAM_OFFSET_T   = (0.05, 0.0, 0.02)
   CAM_OFFSET_RPY = (180.0, 0.0, 0.0)
   CAM_RES        = (640, 480)
   ```

3. `DoosanPickPlaceTask.set_up_scene` 에서, 그리퍼 결합 후
   `find_prim_path_by_name(robot_root, "angle_bracket")` 로 그리퍼 base 경로 확보 →
   **RealSense mesh attach + 카메라 sensor 생성 (둘 다 같은 부모/오프셋)**.
   ```python
   self._gripper_camera_parent = find_prim_path_by_name(robot_root, "angle_bracket")
   print(f"  Camera parent (angle_bracket) = {self._gripper_camera_parent}")

   # ── (a) 시각 모델: RealSense D455 mesh ───────────────────────
   self._realsense_prim_path = attach_realsense_d455(
       parent_prim_path=self._gripper_camera_parent,
       child_name="realsense_d455",
       translation=CAM_OFFSET_T,
       rpy_deg=CAM_OFFSET_RPY,
   )
   print(f"  RealSense mesh attached at {self._realsense_prim_path}")

   # ── (b) 실제 RGB sensor: 같은 RealSense Xform 의 자식으로 두기 ─
   #     이렇게 하면 mesh 가 어떻게 움직이든 sensor 와 자동 동기화.
   self._wrist_camera = WristCamera(
       parent_prim_path=self._realsense_prim_path,
       name="wrist_rgb",
       resolution=CAM_RES,
       translation=(0.0, 0.0, 0.0),    # 부모(=mesh) 가 이미 offset 가짐
       rpy_deg=(0.0, 0.0, 0.0),
   )
   # initialize 는 World.reset() 이후 메인에서 호출
   ```
   `task` 객체에 `self._wrist_camera`, `self._realsense_prim_path` 보관.

4. 메인 (`DoosanPickNPlace.main`) 변경:
   - `my_world.reset()` 직후 `task._wrist_camera.initialize()` 호출.
   - tracker / servo / **viewer** 인스턴스 생성:
     ```python
     tracker = BlueBlockTracker()
     servo = VisualServoController(image_size=CAM_RES)
     viewer = CameraViewer(enabled=True)   # 별도 OpenCV 창. headless 면 False.
     state = "SEARCH"
     SEARCH_POSE_XY = np.array([0.30, 0.40])      # cube 초기 위치 근처
     SEARCH_HEIGHT  = 0.45                        # 시야 확보 가능 높이
     locked_xy = None
     ```
   - 매 스텝 처리 루프를 상태기계로 재작성 (다음 절 참고).
   - 종료 직전에 `viewer.close(); cv2.destroyAllWindows()` 호출.

4. 메인 루프 상태기계 의사코드
   ```python
   while simulation_app.is_running():
       my_world.step(render=True)
       if not my_world.is_playing():
           was_playing = False
           continue
       if not was_playing:
           my_world.reset()
           robot.initialize()
           robot.gripper.initialize(...)  # 기존과 동일
           task._wrist_camera.initialize()
           controller.reset()
           servo.reset()
           state = "SEARCH"
           was_playing = True
           continue

       rgb = task._wrist_camera.get_rgb()
       det = None
       if rgb is not None:
           bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
           det = tracker.detect(bgr)

       # ── 별도 OpenCV 윈도우 갱신 (Isaac Sim 뷰포트와 무관) ──
       key = viewer.update(rgb, det, state_str=state,
                           extra_lines=[f"frame_ok={rgb is not None}"])
       if key == ord('q'):
           break

       ee_pos, _ = robot.end_effector.get_world_pose()  # 또는 별도 fk
       current_xy = ee_pos[:2]

       if state == "SEARCH":
           # 단순히 SEARCH_POSE 위로 보내고, det.found 면 SERVO 진입
           target = np.array([SEARCH_POSE_XY[0], SEARCH_POSE_XY[1], SEARCH_HEIGHT])
           apply_pose_target(controller, target, current_joints)
           if det is not None and det.found:
               state = "SERVO"

       elif state == "SERVO":
           target_xy, err = servo.update(current_xy, det)
           target = np.array([target_xy[0], target_xy[1], SEARCH_HEIGHT])
           apply_pose_target(controller, target, current_joints)
           if servo.is_locked():
               locked_xy = target_xy.copy()
               state = "PICK_PLACE"

       elif state == "PICK_PLACE":
           # 정렬된 XY 위에서 cube 의 실제 z (지면 위)는 task 내부값을 사용
           cube_z = task._cube_initial_position[2]
           picking_position = np.array([locked_xy[0], locked_xy[1], cube_z])
           actions = controller.forward(
               picking_position=picking_position,
               placing_position=task._goal_position,
               current_joint_positions=current_joints,
               end_effector_offset=np.array([0.0, 0.0, 0.2]),
           )
           robot.apply_action(actions)
           if controller.is_done():
               state = "DONE"
               my_world.pause()
   ```
   `apply_pose_target()` 는 RMPFlow 의 cspace controller 한 번 호출하는 헬퍼.
   `PickPlaceController` 내부의 cspace controller 를 직접 쓰면 충돌 회피까지 자동.

5. **디버그 시각화는 `camera_viewer.CameraViewer` 가 전담** (Step 3.5 참고).
   메인 코드에서는 `viewer.update(rgb, det, state)` 한 줄만 호출.
   별도의 OpenCV 윈도우(`wrist_camera`, `mask`)가 Isaac Sim 뷰포트와는 분리된
   OS 레벨 윈도우로 뜬다.

**주의**
- `cv2.imshow` 는 메인 스레드에서만. Step 3.5 의 `CameraViewer` 가 이미 그렇게 호출.
- 헤드리스 모드(`SimulationApp({"headless": True})`) 에서는 `viewer = CameraViewer(enabled=False)`
  로 두고, 필요 시 `cv2.imwrite` 로 디스크에 저장하는 코드를 viewer.update 내부에 한 줄 추가.

### Step 5 — 기능 검증 체크리스트

- [ ] **시뮬레이터 뷰포트에 RealSense D455 모델이 그리퍼에 부착되어 보인다.**
  팔이 움직일 때 RealSense 도 함께 움직여야 한다 (덜렁거리거나 떨어지면 attach 실패).
- [ ] **별도의 OpenCV 윈도우 두 개**(`wrist_camera`, `mask`)가 Isaac Sim 과 분리되어 뜬다.
  Isaac Sim 뷰포트를 닫아도 OpenCV 윈도우는 계속 떠 있어야 한다 (정상).
- [ ] 그리퍼가 움직이면 `wrist_camera` 윈도우 영상이 따라 움직인다.
- [ ] 큐브 초기 위치(`[0.30, 0.40, 0.025]`)가 화면에 들어오면 mask 윈도우에 흰 영역이 뜬다.
- [ ] `SEARCH → SERVO` 전이가 즉시 발생한다 (콘솔 로그 권장, viewer 의 state 텍스트로도 확인).
- [ ] SERVO 상태에서 cross-hair 가 cube 중심으로 수렴한다 (10~30 step 안에 안정화).
- [ ] 안정화 후 자동으로 `PICK_PLACE` 로 넘어가고, 기존 pick & place 가 성공한다.
- [ ] cube 초기 위치를 `[0.25, 0.35, 0.025]` 등으로 바꿔도(코드 변경 없이) 잡는다.
- [ ] 키보드 'q' 로 OpenCV 윈도우에서 정상 종료 가능하다.

---

## 4. 파라미터 튜닝 가이드

| 파라미터 | 위치 | 시작값 | 증상별 조정 |
|---------|------|--------|-------------|
| `CAM_OFFSET_T` | 메인 (mesh + sensor 공유) | `(0.05, 0, 0.02)` | 화면에 그리퍼 손가락이 너무 많이 보이면 z↑, mesh 가 그리퍼 안에 박히면 x↑ |
| `CAM_OFFSET_RPY` | 메인 (mesh + sensor 공유) | `(180, 0, 0)` | 영상이 거꾸로면 x 축 180° 또는 yaw 180° 회전 |
| RealSense USD path | `realsense_mount.py` | D455 | D455 가 없으면 `rsd435.usd` 로 fallback |
| HSV `lower/upper` | `vision_tracker.py` | `(100,120,50)`/`(130,255,255)` | 검출 안 되면 S 하한↓, 잡음 많으면 V 하한↑ |
| `min_area` | `vision_tracker.py` | 200 | 작은 노이즈 잡히면 ↑ |
| `kp` | `visual_servo_controller.py` | 0.0008 | 진동하면 ↓, 너무 느리면 ↑ |
| `max_step` | `visual_servo_controller.py` | 0.02 m | RMPflow 가 따라잡지 못하면 ↓ |
| `tolerance_px` | `visual_servo_controller.py` | 8 | lock 너무 까다로우면 ↑ |
| `axis_sign` | `visual_servo_controller.py` | `(-1, -1)` | 카메라 회전 결과에 따라 `(±1, ±1)` 조합 4가지 중 실험으로 1개 선택 |
| `enabled` | `camera_viewer.py` | `True` | headless 시뮬에서는 `False` 로 두고 imwrite 사용 |

**`axis_sign` 결정 절차** (1회만 하면 됨)
1. 큐브를 화면 우측에 두고 시뮬 실행.
2. 한 스텝 동안 `dx, dy` 계산값과 EE 가 실제로 움직인 방향을 콘솔에 찍는다.
3. 큐브가 화면 중앙으로 오는 부호 조합을 채택.

---

## 5. 알려진 이슈 / 트러블슈팅

1. **`Camera.get_rgba()` 가 `None` 반환**
   - 첫 몇 프레임에서 발생. `world.step` 을 5~10번 추가로 돌린 후 polling.
   - `frequency` 가 너무 낮으면 매 step 마다 새 프레임이 안 나올 수 있음 → 30 권장.

2. **카메라가 그리퍼와 충돌(렌더 가림)**
   - `translation` 의 x 를 0.07 정도까지 늘려 그리퍼 전방으로 빼냄.
   - 또는 부모 prim 을 `angle_bracket` 대신 `link_6` 으로 바꿈.

3. **HSV 마스크가 비어있다**
   - rgb 를 디스크에 저장(`cv2.imwrite`) 후 GIMP/색상피커로 실제 H, S, V 확인.
   - Isaac Sim 의 디폴트 lighting 에서 색이 뿌옇게 보이면 `static_friction` 매트리얼은
     색에 영향 없으므로 무관, 큐브에 emissive 를 살짝 주는 것도 옵션.

4. **SERVO 단계에서 EE 가 진동한다**
   - `kp` 를 절반으로.
   - `max_step` 도 함께 줄임.
   - 안정화되면 D 항 추가 고려:
     `dx = sx*(kp*ex + kd*(ex - ex_prev))`.

5. **RMPFlow 가 갑자기 큰 자코비안 변화로 튐**
   - SEARCH/SERVO 단계에서 target 을 `np.clip(current + delta, lo, hi)` 로 워크스페이스 제한.
   - 권장 범위: `x ∈ [0.2, 0.6], y ∈ [-0.5, 0.5], z ∈ [0.1, 0.7]`.

6. **`cv2.imshow` 윈도우가 안 뜬다 (Linux + headless GUI)**
   - `simulation_app = SimulationApp({"headless": False})` 인지 확인.
   - 그래도 안 되면 `cv2.imwrite("/tmp/wrist.png", vis)` 로 저장 후 외부 뷰어.

7. **카메라 prim path 충돌**
   - 동일 path 가 이미 있으면 `Camera()` 생성자가 실패.
   - 시작 시 `omni.usd.get_context().get_stage().RemovePrim(path)` 로 정리.

8. **RealSense USD 가 로드되지 않는다 (`get_assets_root_path()` 가 None)**
   - Isaac Sim 첫 실행 시 Nucleus 자산을 받지 못한 상태일 수 있음. Isaac Sim Launcher 에서
     "Cache" → 자산 다운로드 확인.
   - 방화벽 때문이면 로컬에 USD 를 한 번 받은 뒤 `usd_path=` 파라미터로 절대경로 직접 지정.
   - `rsd455.usd` 가 없는 버전이면 `rsd435.usd` 로 fallback (`REALSENSE_D455_USD_REL` 상수 변경).

9. **RealSense mesh 가 그리퍼에서 따로 떠있다 (joint 없이 둥둥)**
   - `attach_realsense_d455` 의 `parent_prim_path` 가 잘못된 경로일 가능성.
   - `find_prim_path_by_name(robot_root, "angle_bracket")` 가 `None` 을 반환하면
     RobotAssembler 결합이 끝나기 전에 호출된 것 — `set_up_scene` 에서 결합 *직후* 호출 확인.
   - 메쉬가 너무 큰 스케일로 들어오면 USD 안의 단위가 cm 일 수 있음.
     `xform.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))` 로 보정.

10. **OpenCV 윈도우가 Isaac Sim 종료 시 같이 죽거나 freeze**
    - 정상 동작은 "Isaac Sim 종료해도 OpenCV 윈도우는 잠깐 살아있다 → 메인 함수 끝나며 close".
    - freeze 라면 메인 루프 종료 후 `viewer.close(); cv2.destroyAllWindows()` 가 호출되는지 확인.
    - 그래도 문제면 Step 3.5 의 multiprocessing 옵션으로 분리.

11. **OpenCV 윈도우는 뜨는데 영상이 시커멓다**
    - `WristCamera.initialize()` 가 `World.reset()` 이후 호출되었는지 확인.
    - 첫 5~10 프레임은 검은색일 수 있음 (sensor warm-up). `viewer.update` 가 None 처리하므로 무시.

---

## 6. 향후 확장 아이디어 (이번 작업에서는 구현하지 않음)

- 깊이 채널(`add_distance_to_image_plane_to_frame`)로 cube z 추정 → 고정 height 제거.
- ArUco/QR 마커로 정밀 6D pose 추정.
- 2개 이상의 색상을 동시에 추적해 매니퓰레이션 sequencing.
- ROS2 토픽으로 영상 퍼블리시.
- HSV 범위를 OpenCV 트랙바로 GUI 튜닝.

---

## 7. Claude Code 실행 체크리스트 (작업 순서)

> 아래 순서대로 진행. 각 단계가 끝날 때마다 시뮬을 한 번 실행해서 회귀를 확인.

1. **백업**: `cp m0609_pick_place_fixed_target.py m0609_pick_place_fixed_target.bak.py`
2. **`realsense_mount.py` 작성**, 단독 import 테스트.
3. **`wrist_camera.py` 작성**, 단독 import 테스트 (`python -c "from wrist_camera import WristCamera"`).
4. **`vision_tracker.py` 작성**, 더미 BGR 이미지로 unit test.
   ```python
   import numpy as np, cv2
   from vision_tracker import BlueBlockTracker
   img = np.zeros((480, 640, 3), np.uint8); img[200:280, 280:360] = (255, 0, 0)  # BGR blue
   det = BlueBlockTracker().detect(img); assert det.found and 300 < det.cx < 340
   ```
5. **`visual_servo_controller.py` 작성**, 더미 detection 으로 unit test.
6. **`camera_viewer.py` 작성**, 단독 테스트 — 더미 RGB 이미지를 `update()` 에 넣고
   윈도우가 뜨는지 확인.
   ```python
   import numpy as np
   from camera_viewer import CameraViewer
   v = CameraViewer()
   img = np.full((480, 640, 3), 64, np.uint8); img[:, 320:] = 200
   v.update(img, None, "TEST")
   import time; time.sleep(2)
   v.close()
   ```
7. **`m0609_pick_place_visual.py` 작성** — 기존 메인을 복제 후 위 명세대로 수정.
   - 첫 실행 시 RealSense mesh 가 그리퍼에 붙는지, OpenCV 윈도우가 별도로 뜨는지만 먼저 확인
     (서보잉 로직은 다음 단계).
8. **`CAM_OFFSET_T`, `CAM_OFFSET_RPY` 튜닝** — RealSense 가 그리퍼 손에 자연스럽게 붙도록.
9. **`axis_sign` 1회 결정** — 큐브를 화면 우측에 배치하고 부호 4조합 실험.
10. **HSV 튜닝** — `cv2.imwrite` 또는 viewer 의 mask 윈도우로 한 프레임 확인 후 색상 범위 확정.
11. **gain/tolerance 튜닝** — Step 5 체크리스트 통과시키기.
12. **회귀 테스트** — cube 초기 위치를 3가지로 바꾸며 모두 성공하는지 확인.

---

## 8. Isaac Sim 버전 호환성 메모

- 업로드된 코드는 `isaacsim.core.api`, `isaacsim.robot.manipulators` 사용 → Isaac Sim 4.5+ / 5.x.
- 카메라 import 는 우선 다음 순서로 시도:
  ```python
  try:
      from isaacsim.sensors.camera import Camera           # 5.x
  except ImportError:
      try:
          from omni.isaac.sensor import Camera             # 4.x 호환
      except ImportError:
          raise ImportError("Camera class not found — check Isaac Sim version")
  ```
- `scipy` 미설치 환경이면 `Rotation` 대신 quaternion 직접 계산:
  ```python
  # rpy = (180, 0, 0) → quat (w, x, y, z) = (0, 1, 0, 0)
  ```

---

## 부록 A. 디렉토리 최종 모습

```
project_root/
├── m0609_pick_place_fixed_target.py     # (그대로, 비교용)
├── m0609_pick_place_visual.py           # ★ 신규 메인 (실제 실행)
├── m0609_pick_place_controller.py
├── m0609_rmpflow_controller.py
├── realsense_mount.py                   # ★ 신규 (RealSense D455 mesh attach)
├── wrist_camera.py                      # ★ 신규 (RGB sensor)
├── vision_tracker.py                    # ★ 신규 (HSV → centroid)
├── visual_servo_controller.py           # ★ 신규 (픽셀 에러 → EE XY)
├── camera_viewer.py                     # ★ 신규 (별도 OpenCV 윈도우)
├── m0609_description.yaml
├── m0609_rmpflow_common.yaml
├── doosan-robot2/...
└── onrobot_rg2/...
```

실행:
```bash
python m0609_pick_place_visual.py
```
