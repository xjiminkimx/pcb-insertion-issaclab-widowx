import gymnasium as gym

from .agents.rl_games_logstd_safety import apply_patch as _apply_rl_games_logstd_safety_patch
from . import widowx_pcb_env_cfg

# Prevent Normal.sample failures when exp(log_std) underflows (see agents/rl_games_logstd_safety.py).
_apply_rl_games_logstd_safety_patch()

gym.register(
    id="Isaac-WidowX-PCB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": widowx_pcb_env_cfg.WidowXPcbEnvCfg,
        "rl_games_cfg_entry_point": f"{__name__}.agents.rl_games_ppo_cfg:WidowXPcbPPOCfg",
    },
)