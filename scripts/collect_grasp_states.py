#!/usr/bin/env python3
"""Collect successful terminal states from the trained Grasp-phase policy.

Usage (from workspace root)::

    python scripts/collect_grasp_states.py \\
        --checkpoint logs/rl_games/widowx_pcb_grasp/nn/widowx_pcb_grasp.pth \\
        --num_envs 256 \\
        --num_states 2000 \\
        --out data/grasp_terminal_states.npz

The script rolls out the policy, detects episodes terminated by ``grasp_success``,
and saves the robot joint positions + PCB root pose (position + quaternion) of
every successful terminal step.

Saved .npz keys
---------------
``joint_pos``   : (N, n_joints) float32 — robot joint positions (env-local, rad / m)
``pcb_pos_env`` : (N, 3) float32        — PCB root position (env-local, m)
``pcb_quat``    : (N, 4) float32        — PCB root quaternion (w, x, y, z)
``joint_names`` : list[str]             — ordered joint name list (matching joint_pos columns)

Design follows Sequential Dexterity (Chen et al. CoRL 2023) §3.2:
save the *terminal distribution* of the first sub-policy and replay it as
the initial distribution of the second sub-policy.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import numpy as np

# ── Isaac Lab bootstrap ────────────────────────────────────────────────────────
_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))

if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Collect grasp terminal states.")
parser.add_argument("--checkpoint", required=True, help="Path to trained .pth checkpoint.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel environments.")
parser.add_argument("--num_states", type=int, default=2000, help="Target number of states to collect.")
parser.add_argument("--out", default=os.path.join(_WORKSPACE, "data", "grasp_terminal_states.npz"),
                    help="Output .npz path.")
parser.add_argument("--max_steps", type=int, default=20_000,
                    help="Safety cap: stop after this many env steps regardless of num_states.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ────────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.agents.rl_games_ppo_cfg import WidowXPcbGraspPPOCfg  # noqa: E402

_GRASP_ACTION_DIM = 7   # arm (6) + gripper (1)
_SLIDE_ACTION_DIM = 7   # arm (6) + gripper (1); insert remains 6 (arm only)


def _checkpoint_action_dim(checkpoint_path: str) -> int | None:
    """Return policy output dim from rl-games ``a2c_network.mu.weight`` (rows)."""
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
    hint = ""
    if ckpt_dim == _SLIDE_ACTION_DIM and expected_dim == _GRASP_ACTION_DIM:
        hint = (
            "This checkpoint is from Slide/Insert (6 arm-only actions). "
            "Use scripts/collect_slide_states.py, or pass a Grasp checkpoint "
            "(logs/rl_games/widowx_pcb_grasp/nn/...)."
        )
    elif ckpt_dim == _GRASP_ACTION_DIM and expected_dim == _SLIDE_ACTION_DIM:
        hint = (
            "This checkpoint is from Grasp (7 actions). "
            "Use scripts/collect_grasp_states.py, or pass a Slide checkpoint."
        )
    elif ckpt_dim == 6 and expected_dim == 7:
        hint = (
            "This checkpoint uses the old 6-DoF Slide policy (arm only). "
            "Retrain Slide with gripper in the action space, or use an older env cfg."
        )
    raise ValueError(
        f"Checkpoint action dim {ckpt_dim} does not match {phase} env ({expected_dim}).\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  {hint}"
    )


def _build_env_and_player(checkpoint_path: str, num_envs: int):
    """Create wrapped env + rl-games player (same pattern as Isaac Lab play.py)."""
    env_cfg = cfg.WidowXPcbGraspEnvCfg()
    env_cfg.scene.num_envs = num_envs

    agent_cfg = copy.deepcopy(WidowXPcbGraspPPOCfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make("Isaac-WidowX-PCB-Grasp-v0", cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    _validate_checkpoint_for_phase(checkpoint_path, _GRASP_ACTION_DIM, "Grasp")
    # rl-games restore() uses torch.load(..., weights_only=False) internally.
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
    # RlGamesVecEnvWrapper returns {"obs": tensor}; rl-games player expects the inner tensor here.
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    base_env = env.unwrapped  # direct access to ManagerBasedRLEnv
    total_done = 0
    total_success = 0

    step = 0
    while sum(x.shape[0] for x in joint_pos_list) < args.num_states and step < args.max_steps:
        # ── Save state BEFORE stepping ─────────────────────────────────────────
        # env.step() resets done environments internally before returning, so
        # robot.data.joint_pos for done envs is already the RESET state after
        # env.step() returns. We must cache the terminal state here, before the
        # physics step that might cause a termination.
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

        # ── Identify true grasp_success terminations ───────────────────────────
        # base_env.reset_terminated includes ALL non-timeout terminations
        # (grasp_success, pcb_tilt, pcb_fallen, etc.), so we must query the
        # specific term to avoid collecting failure states.
        # termination_manager._term_dones is set during compute() and is NOT
        # cleared by _reset_idx(), so get_term() is valid after env.step().
        success_mask = base_env.termination_manager.get_term("grasp_success")

        total_done    += int(dones.sum())          # all episode endings
        total_success += int(success_mask.sum())   # grasp_success only

        if success_mask.any():
            env_ids = success_mask.nonzero(as_tuple=True)[0]

            # Use the PRE-STEP state — the actual terminal pose before the env reset it.
            jp = prev_joint_pos[env_ids].cpu().numpy().astype(np.float32)
            pp = (prev_pcb_pos_w[env_ids] - base_env.scene.env_origins[env_ids]).cpu().numpy().astype(np.float32)
            pq = prev_pcb_quat_w[env_ids].cpu().numpy().astype(np.float32)

            joint_pos_list.append(jp)
            pcb_pos_list.append(pp)
            pcb_quat_list.append(pq)
            collected = sum(x.shape[0] for x in joint_pos_list)
            rate = 100.0 * total_success / max(total_done, 1)
            print(f"  step {step:6d} | +{env_ids.numel():3d} states | total {collected:4d}/{args.num_states}"
                  f" | success rate {rate:.1f}%")

        if player.is_rnn and player.states is not None:
            for s in player.states:
                s[:, dones, :] = 0.0

    print(f"\n[INFO] Collection done: {total_success} successes out of {total_done} terminal episodes "
          f"({100.0 * total_success / max(total_done, 1):.1f}% success rate)")

    env.close()

    if not joint_pos_list:
        print("[WARN] No successful grasp episodes collected. "
              "Check that the grasp policy checkpoint is trained, "
              "and that the grasp_success termination fires (gripper gap < PCB_Z * 1.1).")
        return

    joint_pos = np.concatenate(joint_pos_list, axis=0)[: args.num_states]
    pcb_pos = np.concatenate(pcb_pos_list, axis=0)[: args.num_states]
    pcb_quat = np.concatenate(pcb_quat_list, axis=0)[: args.num_states]

    # ── Validate buffer quality ────────────────────────────────────────────────
    # The first joint in WidowX is the left_carriage; its value equals half the
    # gripper gap. Print stats so users can verify the buffer is sane.
    left_carriage_idx = list(joint_names).index("left_carriage_joint") if "left_carriage_joint" in joint_names else -1
    if left_carriage_idx >= 0:
        gap = joint_pos[:, left_carriage_idx] * 2.0  # full gap in metres
        print(f"[INFO] Gripper gap stats (m)  "
              f"mean={gap.mean():.4f}  min={gap.min():.4f}  max={gap.max():.4f}  "
              f"p95={np.percentile(gap, 95):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
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
