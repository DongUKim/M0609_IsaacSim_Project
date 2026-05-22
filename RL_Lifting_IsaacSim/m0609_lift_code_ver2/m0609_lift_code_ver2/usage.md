# M0609 Lift RL — ver2 사용 가이드

`m0609_lift_code_ver2`는 `m0609_lift_code`에서 학습 정체 원인 5가지를 수정한 버전입니다.
원본과 디렉토리가 분리되어 있으므로 두 버전을 동시에 보관하거나 비교 실험할 수 있습니다.

---

## 주요 변경 요약 (원본 대비)

| 항목 | 원본 | ver2 |
|------|------|------|
| fingertip_center z | 0.13500 m | 0.11387 m (21 mm 보정) |
| 보상: gripper_close 항목 | weight 2.0 + 20.0 | **제거** (호버링 local optimum 방지) |
| 보상: minimal_height | 0.15 m | 0.04 m (절벽 제거) |
| dropping_penalty 강화 시점 | 5 M steps | 30 M steps |
| 큐브 초기 z | 0.06 m | 0.03 m (자유낙하 제거) |
| arm_action scale | 0.3 | 0.5 |
| init_noise_std | 0.3 | 1.0 |

---

## 전제 조건

```bash
# 이 디렉토리 위치
cd /home/deeptree/dev_ws/isaac_sim/IsaacLab/m0609_lift_code_ver2

# venv 활성화 (매 새 셸마다)
isaaclab
```

로봇 URDF 소스 파일은 기본적으로 `/home/deeptree/dev_ws/isaac_sim/src`에서 읽습니다.
다른 경로에 있다면 환경 변수로 지정합니다:

```bash
export M0609_SRC_DIR=/path/to/your/robot_sources
```

---

## 첫 실행 전 확인사항

### URDF 캐시

`m0609_lift/cache/m0609_rg2.urdf`가 없으면 첫 import 시 자동으로 생성됩니다.
이미 존재하면 내용을 확인해서 `gen=v10`이 포함되어 있는지 확인합니다:

```bash
head -3 m0609_lift/cache/m0609_rg2.urdf
# 출력에 gen=v10 이 있어야 함
# 없으면 삭제 후 재생성:
rm m0609_lift/cache/m0609_rg2.urdf
```

### 지오메트리 검증 (권장)

학습 전에 fingertip 위치와 큐브 초기 높이가 올바른지 확인합니다:

```bash
python test_grasp_lift.py --num_envs 1
cat /tmp/grasp_lift_report.txt
```

확인 항목:
- `cube world xyz` z값 ≈ **0.03** (큐브가 테이블에 안착한 상태)
- `fingertip_center world xyz` z값이 `phantom-pad midpoint` z값과 **5 mm 이내** 일치
- 스윕 결과 중 하나 이상에서 `GRASP HOLDS` 출력

---

## 학습 실행

모든 명령은 `m0609_lift_code_ver2/` 디렉토리 안에서 실행합니다.

### 스모크 테스트 (권장: 학습 전 sanity check)

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 256 \
    --headless \
    --max_iterations 100
```

약 10–15분 소요. 기대 reward 진행:
- 0–20 iter: 음수 ~ 0 (탐색)
- 20–60 iter: +5 ~ +20 (reach 발현)
- 60–100 iter: +30 이상 (lift 발현 시작)

100 iter 후 mean reward가 +5 미만에 정체되면 `test_grasp_lift.py` 결과를 다시 확인하세요.

### 정규 학습 (헤드리스, 4096 환경)

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 4096 \
    --headless
```

로그는 `logs/rsl_rl/m0609_lift/<날짜-시간>/` 에 저장됩니다.

### 뷰어와 함께 학습 (소규모)

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 64
```

### 체크포인트에서 재개

`--load_run`에는 **폴더 이름만** 입력하고, 체크포인트 파일은 **`--checkpoint`에 별도로** 지정합니다.
`get_checkpoint_path`가 내부적으로 `logs/rsl_rl/m0609_lift/` + `--load_run` + `--checkpoint`를 조합해 전체 경로를 만들기 때문입니다.

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 256 \
    --headless \
    --resume \
    --load_run 2026-05-07_09-40-51 \
    --checkpoint model_1450.pt \
    --max_iterations 10000
```

> **흔한 실수** — 아래처럼 전체 경로를 `--load_run`에 넣으면 `ValueError: No runs present` 에러가 발생합니다.
>
> ```bash
> # 잘못된 예시 (에러 발생)
> --load_run logs/rsl_rl/m0609_lift/2026-05-07_09-40-51/model_1450.pt
> ```

`--checkpoint`를 생략하면 해당 run 폴더에서 알파벳 순 가장 마지막 체크포인트가 자동 선택됩니다:

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --load_run 2026-05-07_09-40-51
```

### 멀티 GPU 학습

```bash
python train.py \
    --task Isaac-M0609-Lift-v0 \
    --num_envs 4096 \
    --headless \
    --distributed
```

---

## 체크포인트 재생 (play)

```bash
python play.py \
    --task Isaac-M0609-Lift-Play-v0 \
    --num_envs 50 \
    --checkpoint logs/rsl_rl/m0609_lift/<run>/model_<iter>.pt
```

실시간 속도로 재생:

```bash
python play.py \
    --task Isaac-M0609-Lift-Play-v0 \
    --num_envs 16 \
    --real-time \
    --checkpoint logs/rsl_rl/m0609_lift/<run>/model_<iter>.pt
```

영상 녹화:

```bash
python play.py \
    --task Isaac-M0609-Lift-Play-v0 \
    --num_envs 16 \
    --video \
    --video_length 300 \
    --checkpoint logs/rsl_rl/m0609_lift/<run>/model_<iter>.pt
```

---

## Task ID 목록

| Task ID | 용도 |
|---------|------|
| `Isaac-M0609-Lift-v0` | 학습용 (4096 env, obs 노이즈 ON) |
| `Isaac-M0609-Lift-Play-v0` | 재생용 (50 env, obs 노이즈 OFF) |

---

## 로그 구조

```
logs/rsl_rl/m0609_lift/
└── <날짜_시간>/
    ├── model_<iter>.pt        # 주기적 체크포인트 (50 iter마다)
    ├── config.yaml            # 실험 설정 스냅샷
    └── ...                    # TensorBoard 이벤트 파일 등
```

TensorBoard로 학습 곡선 확인:

```bash
tensorboard --logdir logs/rsl_rl/m0609_lift
```

---

## 선택 수정 (학습 결과 보고 결정)

### 수정 6: PD 게인 완화 (큐브를 잡았다 떨어뜨리는 패턴 반복 시)

`m0609_lift/doosan.py`의 `m0609_arm` actuator:

```python
# 현재 (ver2)
stiffness=3000.0, damping=200.0

# 수정 후
stiffness=1500.0, damping=100.0
```

변경 후 반드시 `_GEN_TAG`를 `"v11"`로 올리거나 캐시 파일을 삭제합니다.

### 수정 7: 그리퍼 관측 중복 정리 (minor)

`m0609_lift/lift_env_cfg.py`의 `ObservationsCfg.PolicyCfg`:

```python
# 현재: joint_pos_rel, joint_vel_rel (arm 6 + gripper 2 = 8 dim 각각)

# 수정 후: finger_joint 하나만 포함 (arm 6 + 1 = 7 dim)
joint_pos = ObsTerm(
    func=mdp.joint_pos_rel,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "finger_joint"])},
)
joint_vel = ObsTerm(
    func=mdp.joint_vel_rel,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "finger_joint"])},
)
```

### 향후: grasp가 발현되지 않을 때 gripper_close 약하게 재추가

`m0609_lift/lift_env_cfg.py`의 `RewardsCfg`에 추가:

```python
gripper_close_at_grasp_point = RewTerm(
    func=mdp.gripper_close_near_object,
    params={"std": 0.03},
    weight=1.0,   # lifting_object weight(15)보다 반드시 작아야 함
)
```

---

## 원본(ver1)과 비교 실험할 때

두 디렉토리는 독립적이므로 터미널을 두 개 열어 각각 실행하면 됩니다.
로그 디렉토리가 같은 이름(`m0609_lift`)이므로, 구분하려면 `--experiment_name`을 다르게 지정합니다:

```bash
# ver1
cd m0609_lift_code
python train.py --task Isaac-M0609-Lift-v0 --num_envs 4096 --headless \
    --experiment_name m0609_lift_v1

# ver2
cd m0609_lift_code_ver2
python train.py --task Isaac-M0609-Lift-v0 --num_envs 4096 --headless \
    --experiment_name m0609_lift_v2
```
