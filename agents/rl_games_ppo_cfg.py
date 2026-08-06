# rl_games_ppo_cfg.py
from .rl_games_logstd_safety import apply_patch as _apply_rl_games_logstd_safety_patch

# See rl_games_logstd_safety.py (exp(log_std) underflow / Normal.sample on zero std).
_apply_rl_games_logstd_safety_patch()

WidowXPcbPPOBaseCfg = {
    "params": {
        # -1 → Isaac Lab train/play resample a fresh seed each run (see train.py ``--seed -1``).
        "seed": -1,
        
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
            "save_frequency": 10,
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

# Approach — open-jaw trailing-edge straddle (18-dim VIC: 6×[Δq, K, ζ]).
WidowXPcbApproachPPOCfg = {
    **WidowXPcbPPOBaseCfg,
    "params": {
        **WidowXPcbPPOBaseCfg["params"],
        "env": {
            **WidowXPcbPPOBaseCfg["params"]["env"],
            # Task-space OSC: Δxyz / Δrpy.  Raised 0.20 → 0.25 so exploration can leave the
            # reset pose (with pose_rel, clip×position_scale caps per-step EE travel).
            "clip_actions": 0.25,
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
            "name": "widowx_pcb_approach",
            "full_experiment_name": ".",
            "entropy_coef": 1e-2,
            "max_epochs": 210,
            "reward_shaper": {"scale_value": 1.0},
        },
    },
}

WidowXPcbInsertPPOCfg = {
    **WidowXPcbPPOBaseCfg,
    "params": {
        **WidowXPcbPPOBaseCfg["params"],
        "env": {
            **WidowXPcbPPOBaseCfg["params"]["env"],
            # Raised 0.25 -> 0.5: at 0.25 the policy could reach only ~mid-range Δpose targets and
            # stiffness, starving the +Y push authority.  With the stiffness floors added in the env
            # cfg the arm stays firm, so a wider clip gives real forward-push headroom.
            "clip_actions": 0.25,
        },
        "network": {
            **WidowXPcbPPOBaseCfg["params"]["network"],
            "space": {
                **WidowXPcbPPOBaseCfg["params"]["network"]["space"],
                "continuous": {
                    **WidowXPcbPPOBaseCfg["params"]["network"]["space"]["continuous"],
                    "sigma_init": {"name": "const_initializer", "val": -0.8},
                },
            },
        },
        "config": {
            **WidowXPcbPPOBaseCfg["params"]["config"],
            "name": "widowx_pcb_insert",
            "full_experiment_name": ".",
            "reward_shaper": {"scale_value": 1.0},
            "use_diagnostics": False,
            "entropy_coef": 1e-2,
            "max_epochs": 180,
            "horizon_length": 256,
            "mini_epochs": 8,
            # 0.99 -> 0.995 (2026-08-02).  Control runs at 125 Hz here (step_dt 0.008 s), half the
            # rate the stock 0.99 is usually quoted for, so the effective lookahead was
            # ``dt/(1-gamma)`` = 0.8 s -- shorter than a single push stroke.  The phase's payoff
            # structure is back-loaded (travel milestones, then ``insert_success_bonus``), so credit
            # for starting a push has to survive several seconds of discounting to reach the steps
            # that decide it.  0.995 doubles the lookahead to 1.6 s, sized against the 8 s episode
            # the Insert cfg now runs.
            "gamma": 0.995,
        },
    },
}