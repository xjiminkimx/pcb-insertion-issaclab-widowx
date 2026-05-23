# rl_games_ppo_cfg.py
from .rl_games_logstd_safety import apply_patch as _apply_rl_games_logstd_safety_patch

# See rl_games_logstd_safety.py (exp(log_std) underflow / Normal.sample on zero std).
_apply_rl_games_logstd_safety_patch()

WidowXPcbPPOCfg = {
    "params": {
        "seed": 42,
        
        "algo": {
            "name": "a2c_continuous"
        },
        "model": {
            "name": "continuous_a2c_logstd"
        },
        
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "continuous": {
                    "mu_activation": "None",
                    "sigma_activation": "None",
                    "mu_init": {"name": "default"},
                    # Raised from -1.0 → -0.5 so the initial action std ≈ exp(-0.5) ≈ 0.61
                    # (was exp(-1.0) ≈ 0.37).  More stochastic early exploration before the
                    # policy hardens its approach trajectory.
                    "sigma_init": {"name": "const_initializer", "val": -0.5},
                    "fixed_sigma": False,
                }
            },
            "mlp": {
                "units": [256, 128, 64],
                "activation": "elu",
                "initializer": {"name": "default"},
            },
        },
        
        "load_checkpoint": False,

        "env": {
            "name": "rlgpu",
            "env_name": "rlgpu",
            "clip_observations": 10.0,
            "clip_actions": 0.3,
        },
        
        "config": {
            "name": "WidowX_PCB_RL",
            "full_experiment_name": "widowx_pcb",
            # rl-games writes TensorBoard summaries under the experiment log directory.
            # This name appears in the path and helps you filter runs in TensorBoard UI.
            "env_name": "rlgpu", # env_name은 config 내부에 있는 것이 표준입니다.
            "device": "cuda:0",  # GPU 강제 할당
            "ppo": True,
            "mixed_precision": False,
            # Mild L2 on all params (including fixed log_std) — discourages extreme negative log_std.
            "weight_decay": 1e-5,
            "normalize_input": True,
            "normalize_value": True,
            "value_bootstrap": True,
            # Enable richer training diagnostics (policy/value losses, KL, etc.) in summaries.
            # These are shown in TensorBoard scalar dashboards.
            "use_diagnostics": True,

            # ✅ 추가: 에러의 원인인 어드밴티지 정규화 설정 추가
            "normalize_advantage": True,
            
            # Must match --num_envs argument passed to train.py.
            "num_actors": 4096,
            "reward_shaper": {"scale_value": 0.1},
            
            # PPO 학습 하이퍼파라미터
            "gamma": 0.99,
            "tau": 0.95,
            "learning_rate": 1e-4,
            "lr_schedule": "adaptive",
            "kl_threshold": 0.012,
            "score_to_win": 20000,
            "max_epochs": 500,
            "save_best_after": 50,
            "save_frequency": 25,
            "print_stats": True,
            
            # 미니배치 및 최적화 설정
            "grad_norm": 0.5,
            # Raised to 1e-2 to keep the policy stochastic longer.
            # At 2e-3 the policy was collapsing to a narrow grasp trajectory too early.
            # The adaptive LR will reduce the update magnitude when KL spikes, so a
            # higher entropy coef is safe — it just prevents premature convergence.
            "entropy_coef": 1e-2,
            "truncate_grads": True,
            "e_clip": 0.2,
            # Longer horizon gives the value function more context for delayed grasp/push rewards.
            "horizon_length": 128,
            # Rollout = 4096 envs × 128 steps = 524288; minibatch 8192 → 64 minibatches × 8 epochs.
            "minibatch_size": 8192,
            "mini_epochs": 8,
            "critic_coef": 2,
            "clip_value": True,
            "seq_len": 4,
            "bounds_loss_coef": 0.0001,
        }
    }
}

# Phase-specific experiment names (separate TensorBoard / checkpoint folders).
WidowXPcbGraspPPOCfg = {
    **WidowXPcbPPOCfg,
    "params": {
        **WidowXPcbPPOCfg["params"],
        "config": {
            **WidowXPcbPPOCfg["params"]["config"],
            "name": "WidowX_PCB_Grasp_RL",
            "full_experiment_name": "widowx_pcb_grasp",
        },
    },
}

WidowXPcbPushPPOCfg = {
    **WidowXPcbPPOCfg,
    "params": {
        **WidowXPcbPPOCfg["params"],
        "config": {
            **WidowXPcbPPOCfg["params"]["config"],
            "name": "WidowX_PCB_Push_RL",
            "full_experiment_name": "widowx_pcb_push",
        },
    },
}
