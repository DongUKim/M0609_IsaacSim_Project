# RL_Lifting_IsaacSim — M0609 Lift 강화학습

Doosan **M0609** + **OnRobot RG2**가 테이블 위 큐브를 잡아 목표 위치까지 들어올리도록 **Isaac Lab + rsl_rl(PPO)**로 학습하는 프로젝트입니다.

이 문서는 단순 실행법뿐 아니라, 환경을 지금의 형태로 만들기까지 거쳐온 **설계 고찰** — RG2 그리퍼 모델링 문제, 보상 체계 구성, 학습 정체(plateau) 해소 과정 — 을 함께 정리합니다. 실행 절차만 필요하면 [`m0609_lift_code_ver2/usage.md`](m0609_lift_code_ver2/usage.md)를 참조하세요.

---

## 디렉터리 구성

```
RL_Lifting_IsaacSim/
├── example_model_8500.pt              # 학습 완료 예시 정책 (8500 iter)
└── m0609_lift_code_ver2/
    ├── train.py / play.py             # 학습 / 재생 엔트리포인트
    ├── cli_args.py                    # rsl_rl CLI 인자 파서
    ├── test_grasp_lift.py             # 지오메트리/grasp 검증 스크립트
    ├── test_gripper_mimic.py          # 그리퍼 mimic 동작 점검
    ├── scan_home_poses.py             # 홈 자세 스윕 점검
    ├── usage.md                       # 실행 가이드
    └── m0609_lift/                    # 환경 패키지
        ├── __init__.py               # Gym task 등록
        ├── doosan.py                 # M0609+RG2 ArticulationCfg / URDF 결합·가공
        ├── lift_env_cfg.py           # 추상 환경(Scene/Obs/Reward/Term/Curriculum)
        ├── joint_pos_env_cfg.py      # 관절 위치 제어 구체 환경 + Play 변형
        ├── agents/rsl_rl_ppo_cfg.py  # PPO 하이퍼파라미터
        ├── mdp/rewards.py            # 커스텀 보상 함수
        └── cache/m0609_rg2.urdf      # 자동 생성된 결합 URDF 캐시
```

> `m0609_lift_code_ver2/` 안에는 동일 코드의 중첩 사본(`m0609_lift_code_ver2/`)이 보관되어 있습니다. 비교/백업용입니다.

## Task ID

| Task ID | 용도 | 환경 수 | 관측 노이즈 |
|---|---|---|---|
| `Isaac-M0609-Lift-v0` | 학습 | 4096 | ON |
| `Isaac-M0609-Lift-Play-v0` | 재생 | 50 | OFF |

---

## 1. 환경 구성 (Environment Design)

### 1.1 씬 레이아웃 — Franka Lift 레퍼런스에 정렬

Isaac Lab의 Franka lift 예제와 좌표계를 맞춰서 검증된 보상 레시피를 그대로 재사용할 수 있게 했습니다.

- 로봇 베이스를 **월드 원점**(`pos=(0,0,0)`)에 고정 → root frame = world frame, 관측/명령 좌표 변환이 단순해짐.
- 테이블(`SeattleLabTable`)을 `x=0.55`에 배치하고, 그라운드 플레인을 `z=-1.05`로 내려 **테이블 윗면이 z≈0**이 되도록 맞춤. 이후 모든 높이 임계값(`minimal_height`, drop 기준)이 "테이블 기준"으로 직관적으로 해석됨.
- 큐브는 `DexCube`를 0.7배 스케일(약 2.5~4 cm)로 사용, 초기 위치 `(0.40, 0.0, 0.03)`.
- 50 Hz 제어: 물리 100 Hz(`sim.dt=0.01`)에 `decimation=2`. 에피소드 길이 5초.

### 1.2 RG2 그리퍼 모델링 — 가장 큰 난관

RG2는 4절 링크(four-bar) **폐쇄 운동 사슬**을 가진 평행 그리퍼입니다. 그런데:

- **URDF는 트리 구조만 표현**할 수 있고, 폐쇄 루프는 `mimic` 제약으로 흉내 내는데, Isaac Sim은 이 mimic을 보존하지 않습니다.
- 손가락 쪽 6개 관절을 모두 독립 구동하면 서로 힘이 충돌(fighting)하여 outer knuckle이 목표각에 도달하지 못합니다.

이 문제를 `doosan.py`의 URDF 후처리(`_build_combined_urdf`)에서 다음과 같이 해결했습니다:

1. **사슬 붕괴(collapse)** — 안쪽 4개 관절(`*_inner_knuckle_joint`, `*_inner_finger_joint`)을 `fixed`로 변환. 좌/우 손가락이 각각 **outer knuckle 하나를 축으로 회전하는 단일 강체**가 되어, 구동 관절은 `finger_joint`(우)와 `left_outer_knuckle_joint`(좌) **2개만** 남음.
2. **자기충돌 비활성화** — 단순화 후 inner 링크 메시들이 닫힘 중간에 겹쳐 self-collision을 일으키며 outer knuckle을 튕겨내므로 `enabled_self_collisions=False`. (외부 큐브와의 충돌은 영향 없음)
3. **팬텀 패드 클램프(phantom-pad clamp)** — RG2 inner_finger의 복잡한 충돌 메시는 단순화된 운동학에서 어떤 닫힘각에서도 평면 평행 클램프를 형성하지 못합니다. 그래서 inner_finger **충돌 메시는 제거**(시각 메시는 유지 → 여전히 RG2처럼 보임)하고, 각 outer knuckle의 fixed 자식으로 **박스 콜라이더 2개**를 추가했습니다. `q_close=1.15`에서 두 박스가 그리퍼 중심선에서 만나 깨끗한 평면 파지면을 제공합니다.
4. **`fingertip_center` 가상 링크** — 두 손가락 피벗의 기하학적 중점에 질량 없는 링크를 주입해 EE 기준점으로 사용. z값을 `0.135 → 0.11387`(약 21 mm)로 보정한 것이 grasp 정렬의 핵심이었습니다(아래 §3 참조).

### 1.3 결합 URDF 캐시

M0609 URDF와 RG2 URDF를 하나로 병합하고 메시 경로를 절대경로로 재작성한 결과를 `cache/m0609_rg2.urdf`에 저장합니다. 소스 URDF나 가공 로직(`_GEN_TAG`)이 바뀌면 **SHA-1 해시로 변경을 감지해 자동 재생성**합니다. 로봇 소스 경로는 `M0609_SRC_DIR` 환경변수 → 번들 `robots/` → 개발머신 기본경로 순으로 탐색합니다.

### 1.4 액추에이터 / 솔버

| 그룹 | 관절 | stiffness | damping | 비고 |
|---|---|---|---|---|
| `m0609_arm` | joint_1~6 | 3000 | 200 | effort 9600, vel 2.618 |
| `rg2_drive` | finger_joint, left_outer_knuckle_joint | 1e5 | 1e3 | 강한 파지력 |

암 솔버 반복 12회, 큐브 솔버 반복 16회로 접촉 안정성을 확보했습니다.

### 1.5 관측 / 행동 / 종료

- **행동(7-dim)**: 6관절 위치(`scale=0.5`, default offset) + **이진 그리퍼**(open=0, close=±1.15). 그리퍼는 `BinaryJointPositionActionCfg`로 좌/우 outer knuckle을 부호 반전시켜 함께 구동.
- **관측**: `joint_pos_rel`, `joint_vel_rel`, `object_position`(root frame), `target_object_position`(pose command), `last_action`. 학습 시 관측 노이즈(corruption) ON, Play 시 OFF.
- **종료**: 시간 초과(5초) / 큐브가 `z < -0.05`로 떨어지면 `object_dropping`.

---

## 2. 보상 체계 (Reward Design)

베이스라인은 검증된 **Franka lift 레시피**이며, M0609/RG2 특성과 학습 거동에 맞춰 조정했습니다. (`lift_env_cfg.py: RewardsCfg`)

| 보상 항목 | 함수 | std | weight | 역할 |
|---|---|---|---|---|
| `reaching_object` | object_ee_distance | 0.1 | **+1.0** | EE→큐브 dense shaping (탐색 유도) |
| `lifting_object` | object_is_lifted | — | **+15.0** | 큐브가 `minimal_height=0.04` 이상이면 보상 |
| `object_goal_tracking` | object_goal_distance | 0.3 | **+16.0** | 들어올린 뒤 목표 추종(coarse) |
| `object_goal_tracking_fine` | object_goal_distance | 0.05 | **+5.0** | 목표 근처 미세 추종 |
| `dropping_penalty` | is_terminated_term | — | **-5.0** → -10.0 | 낙하 페널티(커리큘럼 강화) |
| `action_rate` | action_rate_l2 | — | **-1e-4** → -1e-2 | 행동 변화 정규화 |
| `joint_vel` | joint_vel_l2 | — | **-1e-4** → -1e-2 | 관절 속도 정규화 |

### 설계 고찰

- **`minimal_height` 절벽 제거 (0.15 → 0.04)** — 들어올림 보상이 0.15 m에서 갑자기 켜지면, 정책 입장에서는 "큐브에 접근만 해도 손해 없는데 들어올리려다 떨어뜨리면 손해"라는 **보상 절벽**이 생겨 호버링에 머뭅니다. 임계값을 큐브 안착 중심(0.03 m)+α인 0.04 m로 낮춰 **조금만 들어도 즉시 보상**이 시작되도록 해 절벽을 없앴습니다.
- **`gripper_close` 보상 제거** — ver1에는 "그리퍼를 닫으면 보상"(weight 2.0+20.0) 항목이 있었는데, 정책이 큐브를 들지 않고 **그 자리에서 그리퍼만 여닫는 호버링 local optimum**에 빠지는 원인이었습니다. 들어올림/목표추종 보상만 남기자 이 함정이 사라졌습니다.
- **낙하 페널티 커리큘럼 (-5 → -10, 30 M steps)** — 페널티를 학습 초기부터 강하게 주면, 정책이 lift를 발견하기도 전에 "큐브 근처에 가면 떨어뜨려 벌점 받는다"는 노이즈 신호가 학습을 교란합니다. 그래서 **lift가 안정적으로 발현된 뒤(~30 M env steps ≈ 305 iter)** 페널티를 강화합니다. 정규화 항(`action_rate`, `joint_vel`)도 같은 이유로 50 M steps 후에 강화(`-1e-4 → -1e-2`)합니다.

### 사용하지 않는(옵션) 보상 — `mdp/rewards.py`

`gripper_close_near_object`는 현재 `RewardsCfg`에 **포함되어 있지 않지만**, grasp가 끝내 발현되지 않을 때 약하게(weight < lifting) 재투입할 수 있도록 남겨둔 함수입니다. 핵심 아이디어:

> **실현된 손가락 위치가 아니라 정책의 "닫기 의도"(raw action < 0)를 보상한다.** 어설픈 파지로 손가락이 큐브에 걸려 멈추면 실제 finger_pos는 거의 변하지 않아, "실제 닫힘"을 보상하면 닫으려 *시도한* 정책에 0 피드백이 갑니다. 의도를 보상해 학습 신호를 접촉 동역학과 분리하면, 첫 시도가 물리적으로 실패해도 "큐브 앞에서 그리퍼 닫기"를 발견할 수 있습니다.

재투입 시 `weight`는 반드시 `lifting_object`(15)보다 작게 두어 호버링 local optimum이 다시 생기지 않게 해야 합니다.

---

## 3. 학습 정체(Plateau) 해소 — ver1 → ver2

`m0609_lift_code_ver2`는 ver1에서 **학습이 정체되던 5가지 원인**을 수정한 버전입니다.

| 항목 | ver1 | ver2 | 이유 |
|---|---|---|---|
| `fingertip_center` z | 0.135 m | **0.11387 m** | EE 기준점이 실제 파지면보다 21 mm 높아 grasp 정렬 실패 |
| `gripper_close` 보상 | 2.0 + 20.0 | **제거** | 호버링 local optimum 유발 |
| `minimal_height` | 0.15 m | **0.04 m** | 들어올림 보상 절벽 제거 |
| 낙하 페널티 강화 시점 | 5 M steps | **30 M steps** | lift 발현 전 신호 교란 방지 |
| 큐브 초기 z | 0.06 m | **0.03 m** | 자유낙하(드롭) 제거 → 시작부터 테이블 안착 |
| `arm_action scale` | 0.3 | **0.5** | 도달 가능 작업공간 확대 |
| `init_noise_std` | 0.3 | **1.0** | 초기 탐색 폭 확대 |

### 학습 전 지오메트리 검증

`fingertip_center` z 보정처럼 기하 오차는 학습 실패의 주범이므로, 본 학습 전에 검증 스크립트로 확인하는 것을 권장합니다:

```bash
python test_grasp_lift.py --num_envs 1
cat /tmp/grasp_lift_report.txt
```

- `cube world xyz` z ≈ **0.03** (테이블 안착)
- `fingertip_center` z가 phantom-pad 중점 z와 **5 mm 이내** 일치
- 스윕 중 하나 이상에서 `GRASP HOLDS` 출력

---

## 4. PPO 하이퍼파라미터 (`agents/rsl_rl_ppo_cfg.py`)

| 항목 | 값 | 항목 | 값 |
|---|---|---|---|
| actor/critic hidden | [256, 128, 64], ELU | learning_rate | 1e-4 (adaptive) |
| init_noise_std | 1.0 | gamma / lam | 0.98 / 0.95 |
| num_steps_per_env | 24 | clip_param | 0.2 |
| num_learning_epochs | 5 | entropy_coef | 0.006 |
| num_mini_batches | 4 | desired_kl | 0.02 |
| max_iterations | 1500 | max_grad_norm | 1.0 |

`schedule="adaptive"` + `desired_kl=0.02`로 KL 발산에 따라 학습률을 자동 조정합니다.

---

## 5. 실행 (요약)

```bash
# venv 활성화 (Isaac Lab)
isaaclab

# 스모크 테스트 (sanity check, ~10–15분)
python train.py --task Isaac-M0609-Lift-v0 --num_envs 256 --headless --max_iterations 100

# 정규 학습
python train.py --task Isaac-M0609-Lift-v0 --num_envs 4096 --headless

# 체크포인트 재생
python play.py --task Isaac-M0609-Lift-Play-v0 --num_envs 50 \
    --checkpoint logs/rsl_rl/m0609_lift/<run>/model_<iter>.pt
```

기대 reward 진행: 0–20 iter 음수~0(탐색) → 20–60 iter +5~+20(reach) → 60–100 iter +30↑(lift 발현). 100 iter 후 +5 미만에 정체되면 §3의 지오메트리 검증부터 다시 확인하세요.

전체 옵션(체크포인트 재개, 멀티 GPU, 영상 녹화, 로그 구조 등)은 [`m0609_lift_code_ver2/usage.md`](m0609_lift_code_ver2/usage.md)에 정리되어 있습니다.

---

## 환경 요구사항

| 항목 | 사양 |
|---|---|
| OS | Ubuntu 22.04 |
| Isaac Sim / Isaac Lab | 5.x / 대응 버전 |
| Python | 3.11 (Isaac Sim 번들) |
| RL 프레임워크 | rsl_rl (PPO) |
| GPU | NVIDIA RTX (대규모 병렬 환경용 VRAM 권장) |
