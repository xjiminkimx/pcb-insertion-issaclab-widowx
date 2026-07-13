# rl_games_ppo_cfg.py
from .rl_games_logstd_safety import apply_patch as _apply_rl_games_logstd_safety_patch

# See rl_games_logstd_safety.py (exp(log_std) underflow / Normal.sample on zero std).
_apply_rl_games_logstd_safety_patch()

WidowXPcbPPOBaseCfg = {
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
            # ``name`` is the log folder under ``logs/rl_games/<name>/``.
            # ``full_experiment_name`` is ``"."`` so Isaac Lab does not add a redundant nested run dir.
            "name": "widowx_pcb_base",
            "full_experiment_name": ".",
            # rl-games writes TensorBoard summaries under the experiment log directory.
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

            "normalize_advantage": True,
            
            # Must match --num_envs argument passed to train.py.
            "num_actors": 4096,
            "reward_shaper": {"scale_value": 0.1},
            
            # PPO 학습 하이퍼파라미터
            "gamma": 0.99,
            "tau": 0.95,
            "learning_rate": 5e-4,
            "lr_schedule": "adaptive",
            "kl_threshold": 0.012,
            "score_to_win": 20000,
            "max_epochs": 200,
            "save_best_after": 20,
            "save_frequency": 15,
            "print_stats": True,
            
            # 미니배치 및 최적화 설정
            "grad_norm": 0.5,
            # Raised to 1e-2 to keep the policy stochastic longer.
            # At 2e-3 the policy was collapsing to a narrow grasp trajectory too early.
            # The adaptive LR will reduce the update magnitude when KL spikes, so a
            # higher entropy coef is safe — it just prevents premature convergence.
            "entropy_coef": 1e-2,
            "truncate_grads": True,
            "e_clip": 0.3,
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

# Push — open-jaw trailing-edge approach + +Y slide (18-dim VIC: 6×[Δq, K, ζ]).
WidowXPcbPushPPOCfg = {
    **WidowXPcbPPOBaseCfg,
    "params": {
        **WidowXPcbPPOBaseCfg["params"],
        "env": {
            **WidowXPcbPPOBaseCfg["params"]["env"],
            # Task-space OSC: smaller Δxyz; orientation axes locked in env cfg.
            "clip_actions": 0.20,
        },
        "network": {
            **WidowXPcbPPOBaseCfg["params"]["network"],
            "space": {
                **WidowXPcbPPOBaseCfg["params"]["network"]["space"],
                "continuous": {
                    **WidowXPcbPPOBaseCfg["params"]["network"]["space"]["continuous"],
                    # Slightly wider init std for 18-dim impedance action space.
                    "sigma_init": {"name": "const_initializer", "val": -0.8},
                },
            },
        },
        "config": {
            **WidowXPcbPPOBaseCfg["params"]["config"],
            "name": "widowx_pcb_push",
            "full_experiment_name": ".",
            "entropy_coef": 1e-2,
            "max_epochs": 210,
            "reward_shaper": {"scale_value": 1.0},
        },
    },
}

# Deprecated alias (pre-push rename).
WidowXPcbStraddlePPOCfg = WidowXPcbPushPPOCfg

WidowXPcbSlidePPOCfg = {
    **WidowXPcbPPOBaseCfg,
    "params": {
        **WidowXPcbPPOBaseCfg["params"],
        "env": {
            **WidowXPcbPPOBaseCfg["params"]["env"],
            "clip_actions": 0.12,
        },
        "network": {
            **WidowXPcbPPOBaseCfg["params"]["network"],
            "space": {
                **WidowXPcbPPOBaseCfg["params"]["network"]["space"],
                "continuous": {
                    **WidowXPcbPPOBaseCfg["params"]["network"]["space"]["continuous"],
                    "sigma_init": {"name": "const_initializer", "val": -1.2},
                },
            },
        },
        "config": {
            **WidowXPcbPPOBaseCfg["params"]["config"],
            "name": "widowx_pcb_slide",
            "full_experiment_name": ".",
            "reward_shaper": {"scale_value": 0.25},
            "use_diagnostics": False,
            "entropy_coef": 1e-2,
            "max_epochs": 100,
            "horizon_length": 256,
            "mini_epochs": 8,
        },
    },
}

# ---------------------------------------------------------------------------
# Phase 3 — Insert (arm only, 6 DoF, force control): slot insertion from slide terminal states.
#
# This is a *separate* tuned config (per user request).  The insert task differs
# from grasp in important ways, so the hyper-parameters are tuned accordingly:
#
#   * Action space is 6 arm joints only (gripper held closed by PD, no action term).
#     Force/torque control: policy outputs torques [N·m], arm stiffness=0.
#   * Reward weights are O(1-200) (see RewardsInsertPhaseCfg) → ``reward_shaper``
#     must be strong enough for credit assignment over ~370 mm travel, but below 1.0
#     to keep value targets finite (NaN at 1.0 with dense penalties).
#   * Early training needs broader exploration (higher ``entropy_coef``, ``sigma_init``,
#     ``clip_actions``) to discover the +Y push joint combination from varied grasp poses.
#   * ``horizon_length`` is raised toward 2 s of control steps so GAE sees more of each
#     12 s episode before the delayed insertion-success signal.
#   * Insertion is a longer-horizon skill and must generalise over varied grasp states.
# ---------------------------------------------------------------------------
WidowXPcbInsertPPOCfg = {
    **WidowXPcbPPOBaseCfg,
    "params": {
        **WidowXPcbPPOBaseCfg["params"],
        "env": {
            **WidowXPcbPPOBaseCfg["params"]["env"],
            # Tighter clip + per-joint scales in ActionsCfgInsert — limit lift amplitude.
            "clip_actions": 0.15,
        },
        "network": {
            **WidowXPcbPPOBaseCfg["params"]["network"],
            "space": {
                **WidowXPcbPPOBaseCfg["params"]["network"]["space"],
                "continuous": {
                    **WidowXPcbPPOBaseCfg["params"]["network"]["space"]["continuous"],
                    # Lower action std — less jitter / detach-prone exploration.
                    "sigma_init": {"name": "const_initializer", "val": -1.2},
                },
            },
        },
        "config": {
            **WidowXPcbPPOBaseCfg["params"]["config"],
            "name": "widowx_pcb_insert",
            "full_experiment_name": ".",
            # Stronger shaper for sparse +Y state/velocity terms (weights 50–100 in env cfg).
            "reward_shaper": {"scale_value": 0.25},
            # explained_variance diagnostic divides by ~0 early on → TensorBoard NaN warnings.
            "use_diagnostics": False,
            # Reduced from 3e-2 after early +Y discovery — less noisy detach-prone exploration.
            "entropy_coef": 1e-2,
            # Max epochs for insert training (scripts/train_insert.sh uses this unless --max_iterations).
            "max_epochs": 60,
            # ~2 s of control steps (256 × 8 ms) per rollout chunk vs 1 s at 128.
            "horizon_length": 256,
            "mini_epochs": 8,
        },
    },
}
