# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the M0609 lift task (claude_rl.md §5.2).

Task IDs:
  Isaac-M0609-Lift-v0        — full training (4096 envs, obs noise on)
  Isaac-M0609-Lift-Play-v0   — checkpoint replay (50 envs, noise off)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-M0609-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:M0609LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M0609LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-M0609-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:M0609LiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M0609LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
