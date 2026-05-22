# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotic manipulation simulation workspace for the **Doosan M0609** 6-DOF arm with an **OnRobot RG2** parallel gripper, running in **NVIDIA Isaac Sim 5.1.0**. The project name is `tribo_DualArm`.

## Running Simulations

Scripts must be run through Isaac Sim's Python interpreter, not system Python:

```bash
# From the workspace root
cd /home/deeptree/dev_ws/isaac_sim

# Run with Isaac Sim's bundled Python
./isaacsim/python.sh src/m0609_pick_place_fixed_target.py

# Alternatively, launch the GUI and load a script via the Script Editor
./isaacsim/isaac-sim.sh
```

`SimulationApp` **must** be instantiated before any other `omniverse`/`isaacsim` imports — this is a hard Isaac Sim requirement.

## Source Structure

```
src/
├── M0609/
│   ├── doosan-robot2/         # M0609 URDF, meshes, USD models
│   └── onrobot_rg2/           # RG2 gripper URDF, meshes, USD
├── OnRobot-RG2FT-ROS/         # ROS2 packages for OnRobot RG2-FT hardware driver
├── robotiq/                   # ROS1/catkin packages for Robotiq grippers
├── Sensor/                    # Isaac Sim camera/fisheye sensor examples
├── save_files/                # Saved USD scene files
├── m0609_pick_place_fixed_target.py   # Main simulation entry point
├── m0609_pick_place_controller.py     # PickPlaceController subclass
├── m0609_rmpflow_controller.py        # RMPFlowController wrapping Lula/RMPFlow
├── m0609_description.yaml             # Robot kinematics/joint-limits descriptor
└── m0609_rmpflow_common.yaml          # RMPFlow motion planner tuning params
```

## Architecture

### Simulation Pipeline (`m0609_pick_place_fixed_target.py`)

1. **URDF Import** — `import_urdf()` loads the arm (`M0609/doosan-robot2/urdf/m0609_isaac_sim.urdf`) and gripper (`M0609/onrobot_rg2/urdf/onrobot_rg2.urdf`) into the Isaac Sim USD stage via the `isaacsim.asset.importer.urdf` extension.
2. **Assembly** — `RobotAssembler` rigidly attaches the gripper to the arm's `link_6` end-effector.
3. **Scene setup** — `DoosanPickPlaceTask(BaseTask)` adds a `DynamicCuboid` target and a `VisualCuboid` goal marker; physics materials set friction coefficients.
4. **Control** — `PickPlaceController` (extends `isaacsim.robot.manipulators.controllers.PickPlaceController`) wraps `RMPFlowController`, which uses `mg.lula.motion_policies.RmpFlow` for collision-free trajectory generation.
5. **Simulation loop** — `World.step(render=True)` drives physics; controller `.forward()` is called every step until `controller.is_done()`.

### Controller Layer

- `RMPFlowController` (`m0609_rmpflow_controller.py`) — thin wrapper around `isaacsim.robot_motion.motion_generation` (Lula/RMPFlow). Requires three config files: URDF, `m0609_description.yaml`, `m0609_rmpflow_common.yaml`.
- `PickPlaceController` (`m0609_pick_place_controller.py`) — sequences gripper open/close and arm motions via `events_dt` timing list; delegates Cartesian motion to `RMPFlowController`.

### Key USD/URDF Paths

| Asset | Path |
|---|---|
| M0609 Isaac Sim URDF | `M0609/doosan-robot2/urdf/m0609_isaac_sim.urdf` |
| OnRobot RG2 URDF | `M0609/onrobot_rg2/urdf/onrobot_rg2.urdf` |
| M0609 USD model | `M0609/doosan-robot2/usd/m0609.usd` |
| RMPFlow config | `src/m0609_rmpflow_common.yaml` |
| Robot descriptor | `src/m0609_description.yaml` |

### ROS Packages (hardware drivers, not simulation)

- `OnRobot-RG2FT-ROS/` — ROS2 packages for the physical OnRobot RG2-FT gripper (action server, MoveIt2 config, message definitions).
- `robotiq/` — ROS1/catkin metapackage for Robotiq 2F/3F grippers and F/T sensors.

## Environment Requirements

- **Isaac Sim 5.1.0** installed at `/home/deeptree/dev_ws/isaac_sim/isaacsim/`
- **Python 3.11** (bundled with Isaac Sim)
- **Ubuntu 22.04**, NVIDIA GPU with RTX-class driver
- The `isaacsim.robot_setup.assembler` extension must be enabled before calling `RobotAssembler` — done at import time via `manager.set_extension_enabled_immediate(...)`.



# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
