#!/usr/bin/env python3
"""Evaluate success rate of trained WidowX PCB policies.

Modes:
  approach — parallel Approach rollouts; fraction of episodes hitting ``approach_success``
  insert   — parallel Insert rollouts from the Approach terminal-state buffer
  chain    — sequential Approach→Insert live handovers (``play_chain`` style)

Examples (from package root)::

    python -u scripts/eval_success.py approach --num_episodes 200 --num_envs 256 --headless
    python -u scripts/eval_success.py insert   --num_episodes 200 --num_envs 256 --headless
    python -u scripts/eval_success.py chain    --num_episodes 50 --headless

    # Explicit checkpoints + JSON summary:
    python -u scripts/eval_success.py chain \\
        --approach_checkpoint logs/rl_games/widowx_pcb_approach/nn/widowx_pcb_approach.pth \\
        --insert_checkpoint logs/rl_games/widowx_pcb_insert/nn/widowx_pcb_insert.pth \\
        --num_episodes 30 --out logs/rl_games/eval/chain_summary.json --headless
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime

import numpy as np

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))

if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(
    description="Evaluate Approach / Insert / Chain success rates.",
)
parser.add_argument(
    "mode",
    choices=("approach", "insert", "chain"),
    help="Which policy / pipeline to evaluate.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint for approach/insert modes (ignored by chain).",
)
parser.add_argument(
    "--approach_checkpoint",
    type=str,
    default=None,
    help="Approach .pth for chain (also used if --checkpoint unset in approach mode).",
)
parser.add_argument(
    "--insert_checkpoint",
    type=str,
    default=None,
    help="Insert .pth for chain (also used if --checkpoint unset in insert mode).",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help="Number of completed episodes (or full chains) to score.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=64,
    help="Parallel envs for approach/insert (chain forces 1).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Safety cap on env steps (defaults: approach 50k, insert 100k, chain per-episode caps).",
)
parser.add_argument(
    "--approach_max_steps",
    type=int,
    default=2000,
    help="Chain only: max Approach control steps per episode.",
)
parser.add_argument(
    "--insert_max_steps",
    type=int,
    default=4000,
    help="Chain only: max Insert control steps after handover.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Env / rl-games seed. -1 samples a fresh seed and prints it.",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Optional JSON summary path (default: logs/rl_games/eval/<mode>_*.json).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=0,
    help="Print progress every N completed episodes (0 = auto: every 10%%).",
)
parser.add_argument("--gui", action="store_true", help="Show Isaac Sim viewport.")

# Approach success overrides (approach + chain)
parser.add_argument("--closedness", type=float, default=None, metavar="Q")
parser.add_argument("--min_tip_down_deg", type=float, default=None, metavar="DEG")
parser.add_argument("--tip_mid", type=float, default=None, metavar="Q")
parser.add_argument("--no_pitch_gate", action="store_true")
parser.add_argument("--no_tip_mid_gate", action="store_true")
parser.add_argument("--no_jaw_level_gate", action="store_true", help="Disable jaw_level success conjunct.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.gui and not getattr(args_cli, "headless", False):
    args_cli.headless = True
if args_cli.mode == "chain" and int(args_cli.num_envs) != 1:
    print(f"[INFO] chain mode forces --num_envs 1 (was {args_cli.num_envs})")
    args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.managers import EventManager  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.agents.checkpoint_compat import (  # noqa: E402
    ensure_approach_checkpoint_compatible,
)
from isaaclab_tasks.manager_based.widowx_pcb.agents.rl_games_ppo_cfg import (  # noqa: E402
    WidowXPcbApproachPPOCfg,
    WidowXPcbInsertPPOCfg,
)

_ACTION_DIM = 18
_DEFAULT_EVAL_DIR = os.path.join(_WORKSPACE, "logs", "rl_games", "eval")


# ── paths / seeds ─────────────────────────────────────────────────────────────


def _default_ckpt(task_dir: str, filename: str) -> str:
    candidates = [
        os.path.join(_WORKSPACE, "logs", "rl_games", task_dir, "weight_saved", filename),
        os.path.join(_WORKSPACE, "logs", "rl_games", task_dir, "nn", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _resolve_ckpt(user_path: str | None, default_path: str, label: str) -> str:
    path = os.path.abspath(user_path or default_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    return path


def _resolve_seed(seed: int | None) -> int:
    if seed is None or int(seed) < 0:
        return int(random.randint(0, 10000))
    return int(seed)


def _unwrap_obs(obs):
    return obs["obs"] if isinstance(obs, dict) else obs


def _checkpoint_action_dim(checkpoint_path: str) -> int | None:
    ckpt = torch.load(retrieve_file_path(checkpoint_path), map_location="cpu", weights_only=False)
    weight = ckpt.get("model", {}).get("a2c_network.mu.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def _validate_action_dim(checkpoint_path: str, expected: int, phase: str) -> None:
    dim = _checkpoint_action_dim(checkpoint_path)
    if dim is None:
        print(f"[WARN] Could not read action dim from {phase} checkpoint; skipping validation.")
        return
    if dim != expected:
        raise ValueError(
            f"{phase} checkpoint action dim {dim} != env ({expected}): {checkpoint_path}"
        )


# ── approach success overrides ────────────────────────────────────────────────


def _apply_approach_success_overrides(env_cfg: cfg.WidowXPcbApproachEnvCfg) -> dict:
    params = dict(env_cfg.terminations.approach_success.params)

    closedness = (
        float(args_cli.closedness)
        if args_cli.closedness is not None
        else float(params.get("closedness_threshold", cfg._APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD))
    )
    params["closedness_threshold"] = closedness

    if args_cli.no_tip_mid_gate:
        params["tip_mid_thickness_std"] = None
        tip_mid = None
    else:
        tip_mid = (
            float(args_cli.tip_mid)
            if args_cli.tip_mid is not None
            else float(
                params.get(
                    "tip_mid_thickness_threshold",
                    cfg._APPROACH_SUCCESS_TIP_MID_THICKNESS_THRESHOLD,
                )
            )
        )
        params["tip_mid_thickness_threshold"] = tip_mid
        params["tip_mid_thickness_std"] = params.get(
            "tip_mid_thickness_std", cfg._APPROACH_MID_THICKNESS_STD_M
        )

    if args_cli.no_pitch_gate:
        params["min_tip_down_deg"] = None
        min_tip_down = None
    else:
        min_tip_down = (
            float(args_cli.min_tip_down_deg)
            if args_cli.min_tip_down_deg is not None
            else float(params.get("min_tip_down_deg", cfg._APPROACH_SUCCESS_MIN_TIP_DOWN_DEG))
        )
        params["min_tip_down_deg"] = min_tip_down

    if args_cli.no_jaw_level_gate:
        params["jaw_level_std"] = None
        jaw_level = None
    else:
        jaw_level = float(
            params.get("jaw_level_threshold", getattr(cfg, "_APPROACH_SUCCESS_JAW_LEVEL_THRESHOLD", 0.5))
        )
        params["jaw_level_threshold"] = jaw_level
        params["jaw_level_std"] = params.get(
            "jaw_level_std", getattr(cfg, "_APPROACH_JAW_LEVEL_STD_M", 0.004)
        )

    env_cfg.terminations.approach_success.params = params
    if hasattr(env_cfg.rewards, "approach_success_bonus"):
        bonus_params = dict(env_cfg.rewards.approach_success_bonus.params)
        for k in (
            "closedness_threshold",
            "tip_mid_thickness_std",
            "tip_mid_thickness_threshold",
            "jaw_level_std",
            "jaw_level_threshold",
            "min_tip_down_deg",
        ):
            if k in params:
                bonus_params[k] = params[k]
        env_cfg.rewards.approach_success_bonus.params = bonus_params

    return {
        "closedness_threshold": closedness,
        "tip_mid_threshold": tip_mid,
        "min_tip_down_deg": min_tip_down,
        "jaw_level_threshold": jaw_level,
    }


def _print_approach_gates(applied: dict) -> None:
    tip = (
        f", tip_mid>={applied['tip_mid_threshold']:.3f}"
        if applied["tip_mid_threshold"] is not None
        else ", tip_mid=OFF"
    )
    pitch = (
        f", tip_down<=-{applied['min_tip_down_deg']:.1f}deg"
        if applied["min_tip_down_deg"] is not None
        else ", pitch_gate=OFF"
    )
    jaw = (
        f", jaw_level>={applied['jaw_level_threshold']:.3f}"
        if applied.get("jaw_level_threshold") is not None
        else ", jaw_level=OFF"
    )
    print(
        f"[INFO] Approach success gates: closedness>={applied['closedness_threshold']:.3f}"
        f"{tip}{pitch}{jaw}",
        flush=True,
    )


# ── env / player ──────────────────────────────────────────────────────────────


def _wrap_env(gym_env, agent_cfg: dict):
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    return RlGamesVecEnvWrapper(
        gym_env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups
    )


def _register_rlgames_env(env) -> None:
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )


def _make_player(agent_cfg: dict, env, checkpoint_path: str, phase: str) -> BasePlayer:
    agent_cfg = copy.deepcopy(agent_cfg)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    _validate_action_dim(checkpoint_path, _ACTION_DIM, phase)
    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    player.restore(retrieve_file_path(checkpoint_path))
    player.reset()
    return player


def _fired_terms(base_env, env_id: int) -> list[str]:
    names: list[str] = []
    for name in base_env.termination_manager.active_terms:
        term = base_env.termination_manager.get_term(name)
        if bool(term[env_id].item()):
            names.append(name)
    return names or ["unknown"]


# ── parallel evaluation (approach / insert) ───────────────────────────────────


def _eval_parallel(
    *,
    task_id: str,
    env_cfg,
    agent_cfg: dict,
    checkpoint: str,
    phase: str,
    success_term: str,
    num_episodes: int,
    max_steps: int,
    print_every: int,
) -> dict:
    gym_env = gym.make(task_id, cfg=env_cfg)
    env = _wrap_env(gym_env, agent_cfg)
    _register_rlgames_env(env)
    player = _make_player(agent_cfg, env, checkpoint, phase)
    base_env = env.unwrapped

    obs = _unwrap_obs(env.reset())
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    successes = 0
    episodes = 0
    fail_counts: Counter[str] = Counter()
    step = 0

    print(
        f"[INFO] Evaluating {phase}: episodes={num_episodes} num_envs={base_env.num_envs} "
        f"ckpt={checkpoint}",
        flush=True,
    )

    while episodes < num_episodes and step < max_steps:
        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        obs, _, dones, _ = env.step(actions)
        obs = _unwrap_obs(obs)
        step += 1

        if not dones.any():
            continue

        success_mask = base_env.termination_manager.get_term(success_term)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        for eid_t in done_ids:
            eid = int(eid_t.item())
            if episodes >= num_episodes:
                break
            episodes += 1
            if bool(success_mask[eid].item()):
                successes += 1
            else:
                reason = "+".join(_fired_terms(base_env, eid))
                fail_counts[reason] += 1

            if print_every > 0 and episodes % print_every == 0:
                rate = 100.0 * successes / episodes
                print(
                    f"  [{phase}] episodes={episodes}/{num_episodes} "
                    f"success={successes} ({rate:.1f}%) env_steps={step}",
                    flush=True,
                )

        if player.is_rnn and player.states is not None:
            for s in player.states:
                s[:, dones, :] = 0.0

    gym_env.close()

    rate = 100.0 * successes / max(episodes, 1)
    summary = {
        "mode": phase.lower(),
        "checkpoint": checkpoint,
        "num_episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(episodes, 1),
        "success_rate_pct": rate,
        "env_steps": step,
        "num_envs": int(env_cfg.scene.num_envs),
        "fail_reasons": dict(fail_counts),
        "hit_max_steps": step >= max_steps and episodes < num_episodes,
    }
    print(
        f"\n[RESULT] {phase}: {successes}/{episodes} = {rate:.2f}% "
        f"(env_steps={step}"
        + (", HIT --max_steps before finishing" if summary["hit_max_steps"] else "")
        + ")",
        flush=True,
    )
    if fail_counts:
        top = ", ".join(f"{k}={v}" for k, v in fail_counts.most_common(8))
        print(f"[RESULT] Failure terms: {top}", flush=True)
    return summary


# ── chain evaluation ──────────────────────────────────────────────────────────


def _write_handover_pose(base_env, jp, pp, pq) -> None:
    """Teleport env 0 back to the Approach-success pose (undo Approach auto-reset)."""
    robot = base_env.scene["robot"]
    pcb = base_env.scene["pcb"]
    device = base_env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    jp_t = torch.as_tensor(jp, device=device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(jp_t, torch.zeros_like(jp_t), env_ids=env_ids)
    robot.update(0.0)

    pp_t = torch.as_tensor(pp, device=device, dtype=torch.float32).unsqueeze(0)
    pq_t = torch.as_tensor(pq, device=device, dtype=torch.float32).unsqueeze(0)
    pos_w = pp_t + base_env.scene.env_origins[env_ids, :3]
    pcb.write_root_pose_to_sim(torch.cat([pos_w, pq_t], dim=-1), env_ids=env_ids)
    pcb.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=device, dtype=torch.float32), env_ids=env_ids
    )
    pcb.update(0.0)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()


def _save_handover_npz(path: str, jp, pp, pq, joint_names: list[str]) -> None:
    np.savez_compressed(
        path,
        joint_pos=np.asarray(jp, dtype=np.float32).reshape(1, -1),
        pcb_pos_env=np.asarray(pp, dtype=np.float32).reshape(1, 3),
        pcb_quat=np.asarray(pq, dtype=np.float32).reshape(1, 4),
        joint_names=np.array(joint_names),
    )


def _reconfigure_mdp(base_env, mdp_cfg) -> None:
    base_env.cfg.observations = mdp_cfg.observations
    base_env.cfg.actions = mdp_cfg.actions
    base_env.cfg.rewards = mdp_cfg.rewards
    base_env.cfg.events = mdp_cfg.events
    base_env.cfg.terminations = mdp_cfg.terminations
    base_env.cfg.curriculum = getattr(mdp_cfg, "curriculum", None)
    base_env.cfg.episode_length_s = mdp_cfg.episode_length_s
    base_env.event_manager = EventManager(base_env.cfg.events, base_env)
    base_env.load_managers()
    base_env.episode_length_buf.zero_()
    base_env.common_step_counter = 0


def _soft_start_insert(base_env, handover_npz: str):
    """Begin Insert without ``scene.reset()`` (avoids PCB/robot default-pose flash)."""
    del handover_npz
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.int64)
    if "reset" in base_env.event_manager.available_modes:
        env_step_count = base_env._sim_step_counter // base_env.cfg.decimation
        base_env.event_manager.apply(
            mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
        )
    base_env.extras["log"] = {}
    for name in (
        "observation_manager",
        "action_manager",
        "reward_manager",
        "curriculum_manager",
        "command_manager",
        "event_manager",
        "termination_manager",
        "recorder_manager",
    ):
        mgr = getattr(base_env, name, None)
        if mgr is None or not hasattr(mgr, "reset"):
            continue
        info = mgr.reset(env_ids)
        if info:
            base_env.extras["log"].update(info)
    base_env.episode_length_buf[env_ids] = 0
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()
    base_env.obs_buf = base_env.observation_manager.compute(update_history=True)
    return base_env.obs_buf


def _eval_chain(
    *,
    approach_ckpt: str,
    insert_ckpt: str,
    num_episodes: int,
    seed: int,
) -> dict:
    approach_cfg = cfg.WidowXPcbApproachEnvCfg()
    approach_cfg.scene.num_envs = 1
    approach_cfg.seed = seed
    applied = _apply_approach_success_overrides(approach_cfg)
    _print_approach_gates(applied)

    gym_env = gym.make("Isaac-WidowX-PCB-Approach-v0", cfg=approach_cfg)
    base_env = gym_env.unwrapped
    approach_agent_cfg = copy.deepcopy(WidowXPcbApproachPPOCfg)
    insert_agent_cfg = copy.deepcopy(WidowXPcbInsertPPOCfg)
    approach_agent_cfg["params"]["seed"] = seed
    insert_agent_cfg["params"]["seed"] = seed

    approach_ok = 0
    insert_ok = 0
    chain_ok = 0
    fail_stage: Counter[str] = Counter()

    print(
        f"[INFO] Evaluating chain: episodes={num_episodes} "
        f"approach={approach_ckpt} insert={insert_ckpt}",
        flush=True,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="widowx_eval_chain_") as tmp:
            for ep in range(num_episodes):
                print(f"\n========== Chain eval {ep + 1}/{num_episodes} ==========", flush=True)

                if ep > 0:
                    approach_cfg = cfg.WidowXPcbApproachEnvCfg()
                    approach_cfg.scene.num_envs = 1
                    approach_cfg.seed = seed
                    applied = _apply_approach_success_overrides(approach_cfg)
                    _reconfigure_mdp(base_env, approach_cfg)

                env = _wrap_env(gym_env, approach_agent_cfg)
                _register_rlgames_env(env)
                player = _make_player(approach_agent_cfg, env, approach_ckpt, "Approach")

                robot = base_env.scene["robot"]
                pcb = base_env.scene["pcb"]
                joint_names = list(robot.joint_names)

                obs = _unwrap_obs(env.reset())
                _ = player.get_batch_size(obs, 1)
                if player.is_rnn:
                    player.init_rnn()

                handover = None
                for step in range(int(args_cli.approach_max_steps)):
                    prev_joint = robot.data.joint_pos.clone()
                    prev_pcb_pos_w = pcb.data.root_pos_w.clone()
                    prev_pcb_quat = pcb.data.root_quat_w.clone()

                    with torch.inference_mode():
                        actions = player.get_action(
                            player.obs_to_torch(obs), is_deterministic=True
                        )
                    obs, _, dones, _ = env.step(actions)
                    obs = _unwrap_obs(obs)

                    if not dones.any():
                        continue

                    success = base_env.termination_manager.get_term("approach_success")
                    if bool(success[0].item()):
                        jp = prev_joint[0].detach().cpu().numpy()
                        pp = (
                            prev_pcb_pos_w[0] - base_env.scene.env_origins[0]
                        ).detach().cpu().numpy()
                        pq = prev_pcb_quat[0].detach().cpu().numpy()
                        _write_handover_pose(base_env, jp, pp, pq)
                        handover = (jp, pp, pq, joint_names)
                        approach_ok += 1
                        print(f"[INFO] Approach SUCCESS at step={step}", flush=True)
                        break

                    if player.is_rnn and player.states is not None:
                        for s in player.states:
                            s[:, dones, :] = 0.0

                if handover is None:
                    fail_stage["approach_fail"] += 1
                    print("[WARN] Approach failed — chain counted as failure.", flush=True)
                    continue

                jp, pp, pq, joint_names = handover
                npz_path = os.path.join(tmp, f"handover_ep{ep + 1}.npz")
                _save_handover_npz(npz_path, jp, pp, pq, joint_names)

                insert_cfg = cfg.WidowXPcbInsertEnvCfg()
                insert_cfg.scene.num_envs = 1
                insert_cfg.seed = seed
                insert_cfg.events.reset_robot_from_approach.params["straddle_states_path"] = npz_path
                insert_cfg.events.reset_pcb_from_approach.params["straddle_states_path"] = npz_path
                _reconfigure_mdp(base_env, insert_cfg)
                _write_handover_pose(base_env, jp, pp, pq)

                env = _wrap_env(gym_env, insert_agent_cfg)
                _register_rlgames_env(env)
                player = _make_player(insert_agent_cfg, env, insert_ckpt, "Insert")

                obs_dict = _soft_start_insert(base_env, npz_path)
                obs = _unwrap_obs(env._process_obs(obs_dict))
                _ = player.get_batch_size(obs, 1)
                if player.is_rnn:
                    player.init_rnn()

                for step in range(int(args_cli.insert_max_steps)):
                    with torch.inference_mode():
                        actions = player.get_action(
                            player.obs_to_torch(obs), is_deterministic=True
                        )
                    obs, _, dones, _ = env.step(actions)
                    obs = _unwrap_obs(obs)
                    if not dones.any():
                        continue
                    success = base_env.termination_manager.get_term("insert_success")
                    if bool(success[0].item()):
                        insert_ok += 1
                        chain_ok += 1
                        print(f"[INFO] Insert SUCCESS at step={step}", flush=True)
                    else:
                        terms = _fired_terms(base_env, 0)
                        fail_stage["insert:" + "+".join(terms)] += 1
                        print(
                            f"[INFO] Insert ended without success (terms={terms})",
                            flush=True,
                        )
                    break
                else:
                    fail_stage["insert_max_steps"] += 1
                    print("[WARN] Insert hit --insert_max_steps.", flush=True)

                done_so_far = ep + 1
                print(
                    f"[progress] chains={done_so_far}/{num_episodes} "
                    f"approach_ok={approach_ok} insert_ok={insert_ok} "
                    f"chain_ok={chain_ok} "
                    f"({100.0 * chain_ok / done_so_far:.1f}%)",
                    flush=True,
                )
    finally:
        gym_env.close()

    n = num_episodes
    summary = {
        "mode": "chain",
        "approach_checkpoint": approach_ckpt,
        "insert_checkpoint": insert_ckpt,
        "num_episodes": n,
        "approach_successes": approach_ok,
        "approach_success_rate": approach_ok / max(n, 1),
        "insert_successes_given_approach": insert_ok,
        "insert_success_rate_given_approach": insert_ok / max(approach_ok, 1),
        "chain_successes": chain_ok,
        "success_rate": chain_ok / max(n, 1),
        "success_rate_pct": 100.0 * chain_ok / max(n, 1),
        "fail_stages": dict(fail_stage),
        "approach_gates": applied,
    }
    print(
        f"\n[RESULT] Chain: {chain_ok}/{n} = {summary['success_rate_pct']:.2f}% full successes\n"
        f"         Approach alone: {approach_ok}/{n} "
        f"({100.0 * approach_ok / max(n, 1):.1f}%)\n"
        f"         Insert | Approach OK: {insert_ok}/{max(approach_ok, 1)} "
        f"({100.0 * insert_ok / max(approach_ok, 1):.1f}%)",
        flush=True,
    )
    if fail_stage:
        top = ", ".join(f"{k}={v}" for k, v in fail_stage.most_common(8))
        print(f"[RESULT] Fail stages: {top}", flush=True)
    return summary


# ── main ──────────────────────────────────────────────────────────────────────


def _default_out_path(mode: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(_DEFAULT_EVAL_DIR, f"{mode}_{stamp}.json")


def _write_summary(summary: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote summary → {path}", flush=True)


def main() -> None:
    seed = _resolve_seed(args_cli.seed)
    print(f"[INFO] Using seed: {seed}", flush=True)

    num_episodes = int(args_cli.num_episodes)
    print_every = int(args_cli.print_every)
    if print_every <= 0:
        print_every = max(1, num_episodes // 10)

    default_approach = _default_ckpt("widowx_pcb_approach", "widowx_pcb_approach.pth")
    default_insert = _default_ckpt("widowx_pcb_insert", "widowx_pcb_insert.pth")

    if args_cli.mode == "approach":
        ckpt = ensure_approach_checkpoint_compatible(
            _resolve_ckpt(
                args_cli.checkpoint or args_cli.approach_checkpoint,
                default_approach,
                "Approach",
            )
        )
        env_cfg = cfg.WidowXPcbApproachEnvCfg()
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.seed = seed
        applied = _apply_approach_success_overrides(env_cfg)
        _print_approach_gates(applied)
        agent_cfg = copy.deepcopy(WidowXPcbApproachPPOCfg)
        agent_cfg["params"]["seed"] = seed
        max_steps = int(args_cli.max_steps or 50_000)
        summary = _eval_parallel(
            task_id="Isaac-WidowX-PCB-Approach-v0",
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            checkpoint=ckpt,
            phase="Approach",
            success_term="approach_success",
            num_episodes=num_episodes,
            max_steps=max_steps,
            print_every=print_every,
        )
        summary["seed"] = seed
        summary["approach_gates"] = applied

    elif args_cli.mode == "insert":
        ckpt = _resolve_ckpt(
            args_cli.checkpoint or args_cli.insert_checkpoint,
            default_insert,
            "Insert",
        )
        env_cfg = cfg.WidowXPcbInsertEnvCfg()
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.seed = seed
        agent_cfg = copy.deepcopy(WidowXPcbInsertPPOCfg)
        agent_cfg["params"]["seed"] = seed
        max_steps = int(args_cli.max_steps or 100_000)
        summary = _eval_parallel(
            task_id="Isaac-WidowX-PCB-Insert-v0",
            env_cfg=env_cfg,
            agent_cfg=agent_cfg,
            checkpoint=ckpt,
            phase="Insert",
            success_term="insert_success",
            num_episodes=num_episodes,
            max_steps=max_steps,
            print_every=print_every,
        )
        summary["seed"] = seed
        # Surface which straddle buffer Insert is sampling from.
        try:
            npz = env_cfg.events.reset_robot_from_approach.params.get("straddle_states_path")
            summary["straddle_states_path"] = npz
            print(f"[INFO] Insert straddle buffer: {npz}", flush=True)
        except Exception:  # noqa: BLE001
            pass

    else:  # chain
        approach_ckpt = ensure_approach_checkpoint_compatible(
            _resolve_ckpt(args_cli.approach_checkpoint, default_approach, "Approach")
        )
        insert_ckpt = _resolve_ckpt(args_cli.insert_checkpoint, default_insert, "Insert")
        summary = _eval_chain(
            approach_ckpt=approach_ckpt,
            insert_ckpt=insert_ckpt,
            num_episodes=num_episodes,
            seed=seed,
        )
        summary["seed"] = seed

    out = args_cli.out or _default_out_path(args_cli.mode)
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")
    _write_summary(summary, out)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
