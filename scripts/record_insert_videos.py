#!/usr/bin/env python3
"""Record rollout videos for the Insert-phase policy.

Fast defaults: 854x480, capture every 4 sim steps (~30 fps playback), 1 episode.

Usage (from workspace root)::

    python scripts/record_insert_videos.py --headless
    python scripts/record_insert_videos.py --headless --num_envs 4 --num_episodes 4 --out insert-4env.mp4
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from datetime import datetime

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))

if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Record Insert-phase rollout videos.")
parser.add_argument(
    "--checkpoint",
    default=os.path.join(_WORKSPACE, "logs/rl_games/widowx_pcb_insert/nn/widowx_pcb_insert.pth"),
    help="Path to trained .pth checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (tiles in a sqrt grid when > 1).")
parser.add_argument("--num_episodes", type=int, default=1, help="Env-0 episode segments to record.")
parser.add_argument(
    "--video_fps",
    type=int,
    default=30,
    help="Output video frame rate.",
)
parser.add_argument(
    "--render_stride",
    type=int,
    default=4,
    help="Capture one frame every N env steps (4 ≈ real-time at 30 fps; fewer renders = faster).",
)
parser.add_argument("--video_width", type=int, default=854, help="Output video width (full frame when tiling).")
parser.add_argument("--video_height", type=int, default=480, help="Output video height (full frame when tiling).")
parser.add_argument(
    "--out",
    default=None,
    help="Output .mp4 path (default: logs/.../videos/play/insert-episode-<time>.mp4).",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="Use the most recent last_*.pth instead of the best checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.agents.rl_games_ppo_cfg import WidowXPcbInsertPPOCfg  # noqa: E402


def _resolve_checkpoint() -> str:
    ckpt_dir = os.path.join(_WORKSPACE, "logs/rl_games/widowx_pcb_insert/nn")
    if args.use_last_checkpoint:
        last = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("last_") and f.endswith(".pth")],
            key=lambda name: os.path.getmtime(os.path.join(ckpt_dir, name)),
            reverse=True,
        )
        if last:
            return os.path.join(ckpt_dir, last[0])
    if os.path.isfile(args.checkpoint):
        return args.checkpoint
    raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}")


def _tile_grid_shape(num_envs: int) -> tuple[int, int]:
    cols = int(math.ceil(math.sqrt(num_envs)))
    rows = int(math.ceil(num_envs / cols))
    return cols, rows


def _build_env_and_player(checkpoint_path: str):
    env_cfg = cfg.WidowXPcbInsertEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    cols, rows = _tile_grid_shape(args.num_envs)
    if args.num_envs > 1:
        tile_w = max(1, args.video_width // cols)
        tile_h = max(1, args.video_height // rows)
        env_cfg.viewer.resolution = (tile_w, tile_h)
    else:
        env_cfg.viewer.resolution = (args.video_width, args.video_height)

    base_eye = np.array(env_cfg.viewer.eye, dtype=np.float64)
    base_lookat = np.array(env_cfg.viewer.lookat, dtype=np.float64)

    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    agent_cfg = copy.deepcopy(WidowXPcbInsertPPOCfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make("Isaac-WidowX-PCB-Insert-v0", cfg=env_cfg, render_mode="rgb_array")
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    player.restore(retrieve_file_path(checkpoint_path))
    player.reset()

    return env, player, max_steps, base_eye, base_lookat, cols, rows


def _render_frame(sim_env, base_eye: np.ndarray, base_lookat: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Render a single frame; tiles parallel envs when num_envs > 1."""
    num_envs = sim_env.num_envs
    if num_envs <= 1:
        return sim_env.render()

    origins = sim_env.scene.env_origins.cpu().numpy()
    tiles: list[np.ndarray] = []
    cam_prim = sim_env.cfg.viewer.cam_prim_path

    for env_idx in range(num_envs):
        origin = origins[env_idx]
        eye = tuple(origin + base_eye)
        target = tuple(origin + base_lookat)
        sim_env.sim.set_camera_view(eye=eye, target=target, camera_prim_path=cam_prim)
        sim_env.sim.render()
        tile = sim_env.render(recompute=True)
        tiles.append(tile)

    grid_rows: list[np.ndarray] = []
    for row in range(rows):
        row_tiles = tiles[row * cols : (row + 1) * cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.zeros_like(row_tiles[0]))
        grid_rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(grid_rows, axis=0)


def _resolve_output_path() -> str:
    if args.out:
        return args.out
    video_root = os.path.join(_WORKSPACE, "logs/rl_games/widowx_pcb_insert/videos/play")
    os.makedirs(video_root, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    if args.num_envs > 1:
        return os.path.join(video_root, f"insert-{args.num_envs}env-{stamp}.mp4")
    return os.path.join(video_root, f"insert-episode-{stamp}.mp4")


def main():
    checkpoint_path = _resolve_checkpoint()
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(
        f"[INFO] Envs: {args.num_envs}, episodes (env-0): {args.num_episodes}, "
        f"output {args.video_width}x{args.video_height}, stride={args.render_stride}, fps={args.video_fps}"
    )

    env, player, max_steps, base_eye, base_lookat, cols, rows = _build_env_and_player(checkpoint_path)
    sim_env = env.unwrapped

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    completed_episodes = 0
    step_in_episode = 0
    frames: list[np.ndarray] = []

    frames.append(_render_frame(sim_env, base_eye, base_lookat, cols, rows))

    while simulation_app.is_running() and completed_episodes < args.num_episodes:
        with torch.inference_mode():
            obs_torch = player.obs_to_torch(obs)
            actions = player.get_action(obs_torch, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)

            if len(dones) > 0 and player.is_rnn and player.states is not None:
                for state in player.states:
                    state[:, dones, :] = 0.0

        step_in_episode += 1
        if step_in_episode % args.render_stride == 0:
            frames.append(_render_frame(sim_env, base_eye, base_lookat, cols, rows))

        episode_done = bool(dones[0]) or step_in_episode >= max_steps
        if episode_done:
            completed_episodes += 1
            print(f"[INFO] Finished env-0 episode {completed_episodes}/{args.num_episodes}")
            if completed_episodes >= args.num_episodes:
                break
            step_in_episode = 0
            if args.num_envs == 1:
                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                frames.append(_render_frame(sim_env, base_eye, base_lookat, cols, rows))

    out_path = _resolve_output_path()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    imageio.mimsave(out_path, frames, fps=args.video_fps)
    duration_s = len(frames) / float(args.video_fps)
    print(f"[INFO] Saved {len(frames)} frames ({duration_s:.1f}s) → {out_path}")

    env.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
