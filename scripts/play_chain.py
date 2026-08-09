#!/usr/bin/env python3
"""Play Approach then Insert as one continuous chain (live handover).

Runs the Approach policy until straddle success, freezes that terminal state into a
one-row buffer, then loads the Insert env/policy from that exact pose — no offline
``collect_approach_states`` step.

Approach success gates (closedness / tip-mid / tip-down pitch) are overridable via CLI
so you can probe handoffs that are looser or stricter than training defaults.

Usage (from workspace root)::

    python -u scripts/play_chain.py \\
        --approach_checkpoint logs/rl_games/widowx_pcb_approach/weight_saved/widowx_pcb_approach.pth \\
        --insert_checkpoint logs/rl_games/widowx_pcb_insert/weight_saved/widowx_pcb_insert_05mm_dent.pth \\
        --num_envs 1

    # Looser pitch gate for demos that never quite hit training tip-down:
    python -u scripts/play_chain.py ... --min_tip_down_deg 12 --closedness 0.35

    # Record Approach+Insert as one half-speed (0.5× realtime) mp4:
    python -u scripts/play_chain.py --video --headless

Note:
    Approach → Insert stays in one SimulationContext. Closing the Approach gym env and
    ``gym.make``-ing Insert in the same Kit process hangs (Isaac Lab clears the sim
    singleton on ``env.close()``). Handover reconfigures MDP managers in-place instead.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import tempfile
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
    description="Chain-play Approach → Insert with a live straddle handover.",
)
parser.add_argument(
    "--approach_checkpoint",
    type=str,
    default=None,
    help="Approach .pth (default: weight_saved/ then nn/ widowx_pcb_approach.pth).",
)
parser.add_argument(
    "--insert_checkpoint",
    type=str,
    default=None,
    help="Insert .pth (default: weight_saved/ then nn/ widowx_pcb_insert.pth).",
)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (1 recommended for GUI).")
parser.add_argument("--episodes", type=int, default=1, help="How many full Approach→Insert chains to run.")
parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Env / rl-games seed. Default -1 samples a fresh seed each run (print it to reproduce).",
)
parser.add_argument(
    "--approach_max_steps",
    type=int,
    default=2000,
    help="Max control steps waiting for Approach success per episode.",
)
parser.add_argument(
    "--insert_max_steps",
    type=int,
    default=4000,
    help="Max control steps in Insert after handover (covers ~8 s episode + margin).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=32,
    help="Print closedness / pitch / lead status every N control steps.",
)

# ── Approach success overrides (defaults = training termination) ─────────────
parser.add_argument(
    "--closedness",
    type=float,
    default=None,
    metavar="Q",
    help="Override approach_success closedness threshold in [0,1] "
    "(training default: _APPROACH_SUCCESS_CLOSEDNESS_THRESHOLD).",
)
parser.add_argument(
    "--min_tip_down_deg",
    type=float,
    default=None,
    metavar="DEG",
    help="Override minimum tip-down pitch in degrees (positive = tip below horizontal; "
    "training default: _APPROACH_SUCCESS_MIN_TIP_DOWN_DEG). Ignored if --no_pitch_gate.",
)
parser.add_argument(
    "--tip_mid",
    type=float,
    default=None,
    metavar="Q",
    help="Override tip mid-thickness index threshold in [0,1] "
    "(training default: _APPROACH_SUCCESS_TIP_MID_THICKNESS_THRESHOLD). Ignored if --no_tip_mid_gate.",
)
parser.add_argument(
    "--no_pitch_gate",
    action="store_true",
    help="Disable the tip-down pitch conjunct of approach_success.",
)
parser.add_argument(
    "--no_tip_mid_gate",
    action="store_true",
    help="Disable the tip mid-thickness conjunct of approach_success (closedness ± pitch only).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable Insert leading-edge vs success-box console prints.",
)
parser.add_argument(
    "--gui",
    action="store_true",
    help="Show Isaac Sim viewport (default: headless unless --gui).",
)
parser.add_argument(
    "--video",
    action="store_true",
    help="Record Approach+Insert into one mp4 at half-speed (0.5× realtime). Forces --num_envs 1.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Directory for recorded videos "
    "(default: logs/rl_games/widowx_pcb_chain/videos/play).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.gui and not getattr(args_cli, "headless", False):
    args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True
    if int(args_cli.num_envs) != 1:
        print(f"[INFO] --video forces --num_envs 1 (was {args_cli.num_envs})")
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
from isaaclab_tasks.manager_based.widowx_pcb.mdp_custom import (  # noqa: E402
    gripper_wrist_pitch_deg_signed_obs,
    straddle_finger_target_closedness,
    straddle_tip_mid_thickness_shaping,
)

_ACTION_DIM = 18
_DEFAULT_VIDEO_DIR = os.path.join(
    _WORKSPACE, "logs", "rl_games", "widowx_pcb_chain", "videos", "play"
)
# Elevated corner view (diagonal baseplate, robot left / magazine right).
# lookat Z raised so the fixture sits lower in frame; eye pulled ~12% closer for a mild zoom.
_CHAIN_VIEWER_EYE = (0.87, 0.88, 0.60)
_CHAIN_VIEWER_LOOKAT = (0.25, 0.35, 0.26)
_CHAIN_VIEWER_RESOLUTION = (1920, 1080)


def _default_ckpt(task_dir: str, filename: str) -> str:
    """Prefer ``weight_saved/`` then ``nn/`` for saved demo checkpoints."""
    candidates = [
        os.path.join(_WORKSPACE, "logs", "rl_games", task_dir, "weight_saved", filename),
        os.path.join(_WORKSPACE, "logs", "rl_games", task_dir, "nn", filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


_DEFAULT_APPROACH_CKPT = _default_ckpt("widowx_pcb_approach", "widowx_pcb_approach.pth")
_DEFAULT_INSERT_CKPT = _default_ckpt("widowx_pcb_insert", "widowx_pcb_insert.pth")


def _lock_chain_camera(base_env) -> None:
    """Re-apply the chain eye/lookat onto ``/OmniverseKit_Persp`` (video + GUI)."""
    vcc = getattr(base_env, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.update_view_location(_CHAIN_VIEWER_EYE, _CHAIN_VIEWER_LOOKAT)
    elif hasattr(base_env, "sim"):
        base_env.sim.set_camera_view(eye=_CHAIN_VIEWER_EYE, target=_CHAIN_VIEWER_LOOKAT)


def _render_frame(env) -> np.ndarray | None:
    """Grab an RGB frame from the Isaac env (env 0). Returns HxWx3 uint8 or None."""
    base = env.unwrapped
    # Lock every capture so GUI orbit / Insert cfg cannot drift the recorded angle.
    _lock_chain_camera(base)
    try:
        frame = base.render()
    except Exception as exc:  # noqa: BLE001 — keep play running if camera fails
        print(f"[WARN] render() failed: {exc}")
        return None
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _append_frame(frames: list[np.ndarray] | None, env) -> None:
    if frames is None:
        return
    frame = _render_frame(env)
    if frame is not None:
        frames.append(frame)


def _save_halfspeed_video(frames: list[np.ndarray], out_path: str, step_dt: float) -> str:
    """Write frames so playback is 0.5× realtime (each control step lasts ``2 * step_dt``)."""
    if not frames:
        raise RuntimeError("No frames captured; cannot write video.")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # Realtime would be fps = 1/step_dt; halfspeed → half that.
    fps = max(1.0 / (2.0 * float(step_dt)), 1.0)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio  # type: ignore

    imageio.mimsave(out_path, frames, fps=fps)
    dur_s = len(frames) / fps
    sim_s = len(frames) * float(step_dt)
    print(
        f"[INFO] Saved halfspeed video: {out_path}  "
        f"({len(frames)} frames, fps={fps:.2f}, play={dur_s:.1f}s for {sim_s:.1f}s sim)",
        flush=True,
    )
    return out_path


def _resolve_ckpt(path: str | None, default: str, label: str) -> str:
    ckpt = path or default
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"{label} checkpoint not found: {ckpt}")
    return os.path.abspath(ckpt)


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


def _unwrap_obs(obs):
    if isinstance(obs, dict):
        return obs["obs"]
    return obs


def _register_rlgames_env(env) -> None:
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
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


def _wrap_env(env, agent_cfg: dict):
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    return RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)


def _apply_approach_success_overrides(env_cfg: cfg.WidowXPcbApproachEnvCfg) -> dict:
    """Mutate ``approach_success`` termination params from CLI; return effective criteria."""
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

    env_cfg.terminations.approach_success.params = params
    # Keep the one-shot bonus reward in sync so TensorBoard / reward logs stay consistent.
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
        "std": float(params.get("std", cfg._APPROACH_SUCCESS_STD_M)),
    }


def _approach_term_params(base_env) -> dict:
    """Resolved ``approach_success`` kwargs from the live termination manager."""
    mgr = base_env.termination_manager
    if hasattr(mgr, "get_term_cfg"):
        return dict(mgr.get_term_cfg("approach_success").params)
    # Fallback for older Isaac Lab: private term cfg map.
    return dict(mgr._term_cfgs["approach_success"].params)  # type: ignore[attr-defined]


def _approach_live_metrics(base_env, success_params: dict) -> dict[str, float]:
    """Closedness / tip-mid / pitch for env 0 (console progress)."""
    term = _approach_term_params(base_env)
    closedness = straddle_finger_target_closedness(
        base_env,
        float(term["std"]),
        term["pcb_cfg"],
        term["left_finger_cfg"],
        term["right_finger_cfg"],
        term["gripper_joint_cfg"],
        float(term["half_length_m"]),
        finger_offset_m=float(term.get("finger_offset_m", 0.020)),
        tip_offset_m=float(term.get("tip_offset_m", 0.0)),
        wrist_body_cfg=term.get("wrist_body_cfg"),
        width_gap_target_left_m=term.get("width_gap_target_left_m"),
        width_gap_target_right_m=term.get("width_gap_target_right_m"),
    )
    pitch = gripper_wrist_pitch_deg_signed_obs(
        base_env,
        term["left_finger_cfg"],
        term["right_finger_cfg"],
        term["wrist_body_cfg"],
    )
    tip_mid_v = float("nan")
    if term.get("tip_mid_thickness_std") is not None:
        tip_mid = straddle_tip_mid_thickness_shaping(
            base_env,
            float(term["tip_mid_thickness_std"]),
            term["pcb_cfg"],
            term["left_finger_cfg"],
            term["right_finger_cfg"],
            term["gripper_joint_cfg"],
            float(term["half_length_m"]),
            finger_offset_m=float(term.get("finger_offset_m", 0.020)),
            tip_offset_m=float(term.get("tip_offset_m", 0.0)),
            wrist_body_cfg=term.get("wrist_body_cfg"),
            width_gap_target_left_m=term.get("width_gap_target_left_m"),
            width_gap_target_right_m=term.get("width_gap_target_right_m"),
        )
        tip_mid_v = float(tip_mid[0].item())
    return {
        "closedness": float(closedness[0].item()),
        "pitch_deg": float(pitch[0].item()),
        "tip_mid": tip_mid_v,
        "need_closedness": float(success_params["closedness_threshold"]),
        "need_pitch": success_params["min_tip_down_deg"],
        "need_tip_mid": success_params["tip_mid_threshold"],
    }


def _save_handover_npz(
    path: str,
    joint_pos: np.ndarray,
    pcb_pos_env: np.ndarray,
    pcb_quat: np.ndarray,
    joint_names: list[str],
) -> None:
    np.savez_compressed(
        path,
        joint_pos=joint_pos.astype(np.float32)[None, ...],
        pcb_pos_env=pcb_pos_env.astype(np.float32)[None, ...],
        pcb_quat=pcb_quat.astype(np.float32)[None, ...],
        joint_names=np.array(joint_names),
    )


def _write_handover_pose(
    base_env,
    joint_pos: np.ndarray,
    pcb_pos_env: np.ndarray,
    pcb_quat: np.ndarray,
) -> None:
    """Teleport env 0 back to the Approach-success pose (undo Approach auto-reset)."""
    robot = base_env.scene["robot"]
    pcb = base_env.scene["pcb"]
    device = base_env.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    jp = torch.as_tensor(joint_pos, device=device, dtype=torch.float32).unsqueeze(0)
    jv = torch.zeros_like(jp)
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
    robot.update(0.0)

    pp = torch.as_tensor(pcb_pos_env, device=device, dtype=torch.float32).unsqueeze(0)
    pq = torch.as_tensor(pcb_quat, device=device, dtype=torch.float32).unsqueeze(0)
    pos_w = pp + base_env.scene.env_origins[env_ids, :3]
    pcb.write_root_pose_to_sim(torch.cat([pos_w, pq], dim=-1), env_ids=env_ids)
    pcb.write_root_velocity_to_sim(torch.zeros(1, 6, device=device, dtype=torch.float32), env_ids=env_ids)
    pcb.update(0.0)

    base_env.scene.write_data_to_sim()
    base_env.sim.forward()


def _soft_start_insert(base_env, handover_npz: str):
    """Begin Insert without ``scene.reset()`` (avoids PCB/robot default-pose flash).

    Applies Insert ``reset`` events (handover npz + bookkeeping) and manager buffer resets
    only. The live Approach terminal pose is preserved across the MDP reconfigure.
    """
    del handover_npz  # path already baked into event params by ``_build_insert_cfg``
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.int64)

    # Do NOT call ``scene.reset`` — that snaps rigid bodies to defaults before events run.
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


def _apply_chain_camera(env_cfg_or_base, *, live: bool = False) -> None:
    """Force the elevated corner camera on a cfg (pre-make) or live env (viewport + cfg)."""
    viewer = env_cfg_or_base.cfg.viewer if live else env_cfg_or_base.viewer
    viewer.eye = _CHAIN_VIEWER_EYE
    viewer.lookat = _CHAIN_VIEWER_LOOKAT
    viewer.resolution = _CHAIN_VIEWER_RESOLUTION
    if live:
        _lock_chain_camera(env_cfg_or_base)
        print(
            f"[INFO] Chain camera: eye={_CHAIN_VIEWER_EYE} lookat={_CHAIN_VIEWER_LOOKAT} "
            f"res={_CHAIN_VIEWER_RESOLUTION}",
            flush=True,
        )


def _print_approach_gates(applied: dict) -> None:
    print(
        "[INFO] Approach success gates: "
        f"closedness>={applied['closedness_threshold']:.3f}"
        + (
            f", tip_mid>={applied['tip_mid_threshold']:.3f}"
            if applied["tip_mid_threshold"] is not None
            else ", tip_mid=OFF"
        )
        + (
            f", tip_down<=-{applied['min_tip_down_deg']:.1f}deg"
            if applied["min_tip_down_deg"] is not None
            else ", pitch_gate=OFF"
        ),
        flush=True,
    )


def _resolve_seed(seed: int | None) -> int:
    """Match Isaac Lab train.py: ``-1`` / ``None`` → random int in ``[0, 10000]``."""
    if seed is None or int(seed) < 0:
        return int(random.randint(0, 10000))
    return int(seed)


def _build_approach_cfg(num_envs: int, seed: int) -> tuple[cfg.WidowXPcbApproachEnvCfg, dict]:
    env_cfg = cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    applied = _apply_approach_success_overrides(env_cfg)
    return env_cfg, applied


def _build_insert_cfg(num_envs: int, handover_npz: str, seed: int) -> cfg.WidowXPcbInsertEnvCfg:
    env_cfg = cfg.WidowXPcbInsertEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.events.reset_robot_from_approach.params["straddle_states_path"] = handover_npz
    env_cfg.events.reset_pcb_from_approach.params["straddle_states_path"] = handover_npz
    if args_cli.debug and hasattr(env_cfg.events, "insert_success_debug"):
        env_cfg.events.insert_success_debug.params["enable_print"] = True
        env_cfg.events.insert_success_debug.params["print_env_id"] = 0
        env_cfg.events.insert_success_debug.params["print_every_control_steps"] = int(
            args_cli.print_every
        )
    return env_cfg


def _reconfigure_mdp(base_env, mdp_cfg) -> None:
    """Swap Approach/Insert managers on a live env without destroying SimulationContext.

    ``ManagerBasedEnv.close()`` calls ``sim.clear_instance()``. A second ``gym.make`` in
    the same Kit process then hangs while recreating the singleton — which is exactly what
    broke Approach→Insert chaining. Keep the scene/sim and only rebuild MDP managers.
    """
    base_env.cfg.observations = mdp_cfg.observations
    base_env.cfg.actions = mdp_cfg.actions
    base_env.cfg.rewards = mdp_cfg.rewards
    base_env.cfg.events = mdp_cfg.events
    base_env.cfg.terminations = mdp_cfg.terminations
    base_env.cfg.curriculum = getattr(mdp_cfg, "curriculum", None)
    base_env.cfg.episode_length_s = mdp_cfg.episode_length_s
    # EventManager is constructed in ``__init__``, not inside ``load_managers``.
    base_env.event_manager = EventManager(base_env.cfg.events, base_env)
    base_env.load_managers()
    base_env.episode_length_buf.zero_()
    base_env.common_step_counter = 0


def _bind_player(gym_env, agent_cfg: dict, checkpoint: str, phase: str):
    """Wrap + register + restore an rl-games player against the live gym env."""
    env = _wrap_env(gym_env, agent_cfg)
    _register_rlgames_env(env)
    player = _make_player(agent_cfg, env, checkpoint, phase)
    return env, player


def _run_approach_until_success(
    env,
    player: BasePlayer,
    base_env,
    applied: dict,
    frames: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], float] | None:
    """Step Approach until success. Does **not** close the env (needed for Insert handoff)."""
    robot = base_env.scene["robot"]
    pcb = base_env.scene["pcb"]
    joint_names = list(robot.joint_names)
    step_dt = float(base_env.step_dt)

    obs = _unwrap_obs(env.reset())
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()
    _append_frame(frames, env)

    for step in range(int(args_cli.approach_max_steps)):
        prev_joint = robot.data.joint_pos.clone()
        prev_pcb_pos_w = pcb.data.root_pos_w.clone()
        prev_pcb_quat = pcb.data.root_quat_w.clone()

        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        obs, _, dones, _ = env.step(actions)
        obs = _unwrap_obs(obs)

        if dones.any():
            success = base_env.termination_manager.get_term("approach_success")
            if bool(success[0].item()):
                # ``step()`` already auto-reset onto a new Approach episode — undo that
                # so the same PCB/arm pose continues into Insert.
                jp = prev_joint[0].detach().cpu().numpy()
                pp = (prev_pcb_pos_w[0] - base_env.scene.env_origins[0]).detach().cpu().numpy()
                pq = prev_pcb_quat[0].detach().cpu().numpy()
                _write_handover_pose(base_env, jp, pp, pq)
                _append_frame(frames, env)
                print(
                    f"[INFO] Approach SUCCESS at step={step} — handing over to Insert "
                    f"(gates: closedness>={applied['closedness_threshold']:.3f}"
                    f", tip_mid={applied['tip_mid_threshold']}"
                    f", tip_down={applied['min_tip_down_deg']})",
                    flush=True,
                )
                return jp, pp, pq, joint_names, step_dt

            _append_frame(frames, env)
            # Failure / timeout: keep going until approach_max_steps (auto-reset continues).
            if player.is_rnn and player.states is not None:
                for s in player.states:
                    s[:, dones, :] = 0.0
        else:
            _append_frame(frames, env)

        if step % int(args_cli.print_every) == 0:
            m = _approach_live_metrics(base_env, applied)
            tip_s = (
                f" tip_mid={m['tip_mid']:.3f}/{m['need_tip_mid']:.3f}"
                if m["need_tip_mid"] is not None
                else ""
            )
            pitch_need = (
                f"/{-m['need_pitch']:.1f}" if m["need_pitch"] is not None else "/OFF"
            )
            print(
                f"[approach] step={step} "
                f"closedness={m['closedness']:.3f}/{m['need_closedness']:.3f}"
                f"{tip_s} pitch={m['pitch_deg']:.1f}{pitch_need} deg",
                flush=True,
            )

    print("[WARN] Approach did not succeed within --approach_max_steps; aborting chain.")
    return None


def _run_insert_phase(
    env,
    player: BasePlayer,
    base_env,
    handover_npz: str,
    frames: list[np.ndarray] | None = None,
) -> bool:
    """Play Insert on the already-reconfigured live env. Does **not** close the env."""
    # Soft start: keep the live Approach PCB (no ``scene.reset`` default-pose flash).
    obs_dict = _soft_start_insert(base_env, handover_npz)
    obs = _unwrap_obs(env._process_obs(obs_dict))
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()
    _append_frame(frames, env)

    print(
        f"[INFO] Insert phase started from live Approach pose (handover: {handover_npz})",
        flush=True,
    )
    for step in range(int(args_cli.insert_max_steps)):
        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        obs, _, dones, _ = env.step(actions)
        obs = _unwrap_obs(obs)
        _append_frame(frames, env)

        if not dones.any():
            continue

        success = base_env.termination_manager.get_term("insert_success")
        if bool(success[0].item()):
            print(f"[INFO] Insert SUCCESS at step={step}", flush=True)
            return True

        term_names = []
        for name in base_env.termination_manager.active_terms:
            t = base_env.termination_manager.get_term(name)
            if bool(t[0].item()):
                term_names.append(name)
        print(
            f"[INFO] Insert episode ended at step={step} without success "
            f"(terms={term_names or ['unknown']})",
            flush=True,
        )
        return False

    print("[WARN] Insert hit --insert_max_steps without termination.", flush=True)
    return False


def main() -> None:
    approach_ckpt = ensure_approach_checkpoint_compatible(
        _resolve_ckpt(args_cli.approach_checkpoint, _DEFAULT_APPROACH_CKPT, "Approach")
    )
    insert_ckpt = _resolve_ckpt(args_cli.insert_checkpoint, _DEFAULT_INSERT_CKPT, "Insert")
    print(f"[INFO] Approach checkpoint: {approach_ckpt}")
    print(f"[INFO] Insert checkpoint:   {insert_ckpt}")
    if args_cli.video:
        video_dir = os.path.abspath(args_cli.video_dir or _DEFAULT_VIDEO_DIR)
        print(f"[INFO] Video recording ON (halfspeed) → {video_dir}")

    num_envs = int(args_cli.num_envs)
    seed = _resolve_seed(args_cli.seed)
    print(f"[INFO] Using seed: {seed}", flush=True)
    render_mode = "rgb_array" if args_cli.video else None
    approach_cfg, applied = _build_approach_cfg(num_envs, seed)
    _apply_chain_camera(approach_cfg, live=False)
    _print_approach_gates(applied)

    # One gym env / SimulationContext for the whole chain (and all --episodes).
    gym_env = gym.make("Isaac-WidowX-PCB-Approach-v0", cfg=approach_cfg, render_mode=render_mode)
    base_env = gym_env.unwrapped
    _apply_chain_camera(base_env, live=True)
    approach_agent_cfg = copy.deepcopy(WidowXPcbApproachPPOCfg)
    insert_agent_cfg = copy.deepcopy(WidowXPcbInsertPPOCfg)
    approach_agent_cfg["params"]["seed"] = seed
    insert_agent_cfg["params"]["seed"] = seed

    n_ok = 0
    step_dt = float(base_env.step_dt)
    try:
        with tempfile.TemporaryDirectory(prefix="widowx_chain_") as tmp:
            for ep in range(int(args_cli.episodes)):
                print(f"\n========== Chain episode {ep + 1}/{args_cli.episodes} ==========")
                frames: list[np.ndarray] | None = [] if args_cli.video else None

                if ep > 0:
                    approach_cfg, applied = _build_approach_cfg(num_envs, seed)
                    _print_approach_gates(applied)
                    print("[INFO] Reconfiguring live env → Approach", flush=True)
                    _reconfigure_mdp(base_env, approach_cfg)
                    _apply_chain_camera(base_env, live=True)

                env, player = _bind_player(
                    gym_env, approach_agent_cfg, approach_ckpt, "Approach"
                )
                handover = _run_approach_until_success(
                    env, player, base_env, applied, frames=frames
                )
                if handover is None:
                    continue
                jp, pp, pq, joint_names, step_dt = handover

                npz_path = os.path.join(tmp, f"handover_ep{ep + 1}.npz")
                _save_handover_npz(npz_path, jp, pp, pq, joint_names)

                insert_cfg = _build_insert_cfg(num_envs, npz_path, seed)
                print(
                    "[INFO] Reconfiguring live env Approach → Insert "
                    "(same PCB / arm pose, no scene.reset)",
                    flush=True,
                )
                _reconfigure_mdp(base_env, insert_cfg)
                # Re-assert terminal pose after manager rebuild, then soft-start Insert.
                _write_handover_pose(base_env, jp, pp, pq)
                _apply_chain_camera(base_env, live=True)
                env, player = _bind_player(
                    gym_env, insert_agent_cfg, insert_ckpt, "Insert"
                )
                if _run_insert_phase(env, player, base_env, npz_path, frames=frames):
                    n_ok += 1

                if frames:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out = os.path.join(
                        args_cli.video_dir or _DEFAULT_VIDEO_DIR,
                        f"chain_ep{ep + 1}_{stamp}_halfspeed.mp4",
                    )
                    try:
                        _save_halfspeed_video(frames, out, step_dt)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[ERROR] Failed to write video: {exc}", flush=True)
    finally:
        gym_env.close()

    print(f"\n[INFO] Chain done: {n_ok}/{args_cli.episodes} full Approach→Insert successes.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
