#!/usr/bin/env python3
"""Collect successful Approach-phase (straddle) terminal states for Slide training.

Usage (from workspace root)::

    python scripts/collect_approach_states.py \\
        --checkpoint logs/rl_games/widowx_pcb_approach/nn/widowx_pcb_approach.pth \\
        --num_envs 4096 \\
        --num_states 500 \\
        --out data/approach_terminal_states.npz \\
        --headless
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import numpy as np

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))

if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Collect approach (straddle) terminal states.")
parser.add_argument("--checkpoint", required=True, help="Path to trained Approach .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--num_states", type=int, default=500, help="Target number of states to collect.")
parser.add_argument(
    "--out",
    default=os.path.join(_WORKSPACE, "data", "approach_terminal_states.npz"),
    help="Output .npz path.",
)
parser.add_argument("--max_steps", type=int, default=50_000, help="Safety cap on total env steps.")
parser.add_argument("--gui", action="store_true", help="Show Isaac Sim viewport (default: headless).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.gui:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.agents.rl_games_ppo_cfg import WidowXPcbApproachPPOCfg  # noqa: E402

_APPROACH_ACTION_DIM = 18


def _checkpoint_action_dim(checkpoint_path: str) -> int | None:
    ckpt = torch.load(retrieve_file_path(checkpoint_path), map_location="cpu", weights_only=False)
    weight = ckpt.get("model", {}).get("a2c_network.mu.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


def _validate_checkpoint_for_phase(checkpoint_path: str, expected_dim: int, phase: str) -> None:
    ckpt_dim = _checkpoint_action_dim(checkpoint_path)
    if ckpt_dim is None:
        print("[WARN] Could not read action dim from checkpoint; skipping validation.")
        return
    if ckpt_dim == expected_dim:
        return
    raise ValueError(
        f"Checkpoint action dim {ckpt_dim} does not match {phase} env ({expected_dim}).\n"
        f"  checkpoint: {checkpoint_path}"
    )


def _build_env_and_player(checkpoint_path: str, num_envs: int):
    env_cfg = cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = num_envs

    agent_cfg = copy.deepcopy(WidowXPcbApproachPPOCfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make("Isaac-WidowX-PCB-Approach-v0", cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    _validate_checkpoint_for_phase(checkpoint_path, _APPROACH_ACTION_DIM, "Approach")
    player.restore(retrieve_file_path(checkpoint_path))
    player.reset()

    return env, player


def collect(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    env, player = _build_env_and_player(args.checkpoint, args.num_envs)

    robot = env.unwrapped.scene["robot"]
    pcb = env.unwrapped.scene["pcb"]
    joint_names: list[str] = robot.joint_names

    joint_pos_list: list[np.ndarray] = []
    pcb_pos_list: list[np.ndarray] = []
    pcb_quat_list: list[np.ndarray] = []

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    base_env = env.unwrapped
    total_done = 0
    total_success = 0
    step = 0

    while sum(x.shape[0] for x in joint_pos_list) < args.num_states and step < args.max_steps:
        prev_joint_pos = robot.data.joint_pos.clone()
        prev_pcb_pos_w = pcb.data.root_pos_w.clone()
        prev_pcb_quat_w = pcb.data.root_quat_w.clone()

        with torch.inference_mode():
            obs_torch = player.obs_to_torch(obs)
            actions = player.get_action(obs_torch, is_deterministic=True)
        obs, _, dones, _ = env.step(actions)
        step += 1

        if not dones.any():
            continue

        success_mask = base_env.termination_manager.get_term("approach_success")
        total_done += int(dones.sum())
        total_success += int(success_mask.sum())

        if success_mask.any():
            env_ids = success_mask.nonzero(as_tuple=True)[0]
            jp = prev_joint_pos[env_ids].cpu().numpy().astype(np.float32)
            pp = (prev_pcb_pos_w[env_ids] - base_env.scene.env_origins[env_ids]).cpu().numpy().astype(np.float32)
            pq = prev_pcb_quat_w[env_ids].cpu().numpy().astype(np.float32)

            joint_pos_list.append(jp)
            pcb_pos_list.append(pp)
            pcb_quat_list.append(pq)
            collected = sum(x.shape[0] for x in joint_pos_list)
            rate = 100.0 * total_success / max(total_done, 1)
            print(
                f"  step {step:6d} | +{env_ids.numel():3d} states | total {collected:4d}/{args.num_states}"
                f" | success rate {rate:.1f}%"
            )

        if player.is_rnn and player.states is not None:
            for s in player.states:
                s[:, dones, :] = 0.0

    print(
        f"\n[INFO] Collection done: {total_success} straddle successes out of {total_done} terminal episodes "
        f"({100.0 * total_success / max(total_done, 1):.1f}% success rate)"
    )

    env.close()

    if not joint_pos_list:
        print(
            "[WARN] No successful straddle episodes collected. "
            "Check the Approach checkpoint and that approach_success fires."
        )
        return

    joint_pos = np.concatenate(joint_pos_list, axis=0)[: args.num_states]
    pcb_pos = np.concatenate(pcb_pos_list, axis=0)[: args.num_states]
    pcb_quat = np.concatenate(pcb_quat_list, axis=0)[: args.num_states]

    lc_idx = list(joint_names).index("left_carriage_joint") if "left_carriage_joint" in joint_names else -1
    if lc_idx >= 0:
        gap = joint_pos[:, lc_idx] * 2.0
        print(
            f"[INFO] Gripper gap stats (m)  "
            f"mean={gap.mean():.4f}  min={gap.min():.4f}  max={gap.max():.4f}  "
            f"p95={np.percentile(gap, 95):.4f}"
        )

    np.savez_compressed(
        args.out,
        joint_pos=joint_pos,
        pcb_pos_env=pcb_pos,
        pcb_quat=pcb_quat,
        joint_names=np.array(joint_names),
    )
    print(f"[INFO] Saved {joint_pos.shape[0]} states → {args.out}")


if __name__ == "__main__":
    collect(args)
    simulation_app.close()
