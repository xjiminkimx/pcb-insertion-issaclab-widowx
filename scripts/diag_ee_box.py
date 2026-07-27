"""Instrument the Approach EE translation box: is it called, and what is it anchored to?

The arm sinks ~70 mm per episode with ZERO actions.  ``task_position_box_enabled`` was switched on to
bound that, but the sag did not change even with a 10 mm box -- so either the clamp never runs, or its
anchor (``env._slide_reset_ee_pos_w``, written by the ``store_reset_ee_pose`` reset event) is not the
home pose.  This traces both: call count, the anchor, and the vertical offset before/after clamping.

Run:
    python -u scripts/diag_ee_box.py --num_envs 4 --steps 120 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--vert_box_m", type=float, default=None, help="Override the vertical box half-range.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks.manager_based.widowx_pcb.widowx_pcb_env_cfg as cfg  # noqa: E402


def main() -> None:
    env_cfg = cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.vert_box_m is not None:
        env_cfg.actions.arm_action.vertical_half_range_m = float(args.vert_box_m)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    term = env.action_manager.get_term("arm_action")
    print(f"[cfg ] task_position_box_enabled = {getattr(term.cfg, 'task_position_box_enabled', None)}")
    print(f"[cfg ] vertical_half_range_m     = {getattr(term.cfg, 'vertical_half_range_m', None)}")
    print(f"[term] _pose_rel_idx             = {term._pose_rel_idx}")
    print(f"[term] _task_vertical_axis_w     = {term._task_vertical_axis_w}")
    print(f"[env ] has _slide_reset_ee_pos_w = {hasattr(env, '_slide_reset_ee_pos_w')}")
    if hasattr(env, "_slide_reset_ee_pos_w"):
        print(f"[env ] anchor p0 (env 0)         = {env._slide_reset_ee_pos_w[0].tolist()}")
    term._compute_ee_pose()
    print(f"[term] live EE pose  (env 0)     = {term._ee_pose_w[0, :3].tolist()}")

    stats = {"calls": 0}
    original = term._clamp_pose_rel_to_reset_position_box
    trace: list[tuple[int, float, float, float]] = []

    def wrapped() -> None:
        stats["calls"] += 1
        idx = term._pose_rel_idx
        term._compute_ee_pose()
        z_cur = float(term._ee_pose_w[0, 2])
        z0 = float(env._slide_reset_ee_pos_w[0, 2]) if hasattr(env, "_slide_reset_ee_pos_w") else float("nan")
        original()
        dz_cmd = float(term._processed_actions[0, idx + 2])
        trace.append((stats["calls"], z0, z_cur, dz_cmd))

    term._clamp_pose_rel_to_reset_position_box = wrapped

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        env.step(zero)

    print(f"\n[trace] clamp called {stats['calls']} times in {args.steps} steps")
    print("  step |   anchor z0 |    live z   | offset (mm) | commanded dz_b (mm)")
    for i, z0, z_cur, dz in trace[:: max(1, len(trace) // 20)]:
        print(f"  {i:4d} | {z0:11.4f} | {z_cur:11.4f} | {(z_cur - z0) * 1000:+11.2f} | {dz * 1000:+11.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
