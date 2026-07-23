#!/usr/bin/env python3
"""Diagnostic: bypass PPO entirely and command a constant max +Y push to see if the
Slide-phase OSC/action pipeline can physically move the PCB leading edge at all.

Usage (from workspace root)::

    python scripts/diag_push_authority.py --num_envs 8 --steps 300 --headless
"""

from __future__ import annotations

import argparse
import os
import sys

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))
if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Open-loop push authority diagnostic.")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--position_scale", type=float, default=None, help="Override arm_action.position_scale")
parser.add_argument("--ty_stiffness_max", type=float, default=None, help="Override ty stiffness upper limit (N/m)")
parser.add_argument("--damping_ratio_action", type=float, default=0.3, help="Raw damping-ratio action [-1,1]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.mdp_custom import pcb_leading_short_edge_center_env  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402


def main():
    env_cfg = cfg.WidowXPcbSlideEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.position_scale is not None:
        env_cfg.actions.arm_action.position_scale = args.position_scale
        print(f"[diag] override position_scale={args.position_scale}")
    if args.ty_stiffness_max is not None:
        limits = list(env_cfg.actions.arm_action.motion_stiffness_limits_per_axis)
        lo, _hi = limits[1]
        limits[1] = (lo, args.ty_stiffness_max)
        env_cfg.actions.arm_action.motion_stiffness_limits_per_axis = tuple(limits)
        print(f"[diag] override ty stiffness limits={limits[1]}")
    env = gym.make("Isaac-WidowX-PCB-Slide-v0", cfg=env_cfg).unwrapped

    obs, _ = env.reset()

    pcb_cfg = SceneEntityCfg("pcb")
    pcb_cfg.resolve(env.scene)

    action = torch.zeros(env.num_envs, 18, device=env.device)
    # pose_rel: [tx,ty,tz,rx,ry,rz] -> push hard +Y only.
    action[:, 1] = 1.0
    # stiffness (normalized [-1,1] -> per-axis limits): max on tx/ty/rz (controlled axes).
    action[:, 6 + 0] = 1.0
    action[:, 6 + 1] = 1.0
    action[:, 6 + 5] = 1.0
    # damping ratio: mid-high (critically damped-ish).
    action[:, 12 + 0] = args.damping_ratio_action
    action[:, 12 + 1] = args.damping_ratio_action
    action[:, 12 + 5] = args.damping_ratio_action

    def lead_y():
        lead = pcb_leading_short_edge_center_env(env.unwrapped, pcb_cfg, cfg._HALF_LENGTH_M, cfg.PUSH_AXIS_WORLD)
        return lead[:, 1].detach().cpu()

    robot = env.scene["robot"]
    ee_body_id = robot.find_bodies(cfg._EE_OSC_BODY_NAME)[0][0]
    arm_joint_ids = robot.find_joints(["joint_[0-5]"])[0]
    print(f"[diag] arm joint effort limits: {robot.data.joint_effort_limits[0, arm_joint_ids].cpu().tolist()}")
    print(f"[diag] arm joint vel limits:    {robot.data.joint_vel_limits[0, arm_joint_ids].cpu().tolist()}")

    def ee_y():
        return (robot.data.body_pos_w[:, ee_body_id, 1] - env.scene.env_origins[:, 1]).detach().cpu()

    def ee_z():
        return (robot.data.body_pos_w[:, ee_body_id, 2] - env.scene.env_origins[:, 2]).detach().cpu()

    y0 = lead_y()
    ee_y0 = ee_y()
    ee_z0 = ee_z()
    print(f"[diag] step=0 lead_y mean={y0.mean():.6f} min={y0.min():.6f} max={y0.max():.6f}")
    print(f"[diag] step=0 ee_y  mean={ee_y0.mean():.6f} ee_z mean={ee_z0.mean():.6f}")

    for step in range(1, args.steps + 1):
        obs, rew, terminated, truncated, info = env.step(action)
        if step % 25 == 0 or step == 1:
            y = lead_y()
            dy = y - y0
            ey = ee_y()
            dey = ey - ee_y0
            applied = robot.data.applied_torque[:, arm_joint_ids].detach().cpu()
            limits = robot.data.joint_effort_limits[:, arm_joint_ids].detach().cpu()
            frac_sat = (applied.abs() >= 0.98 * limits).float().mean(dim=0)
            ez = ee_z() - ee_z0
            qvel = robot.data.joint_vel[:, arm_joint_ids].detach().cpu()
            print(
                f"[diag] step={step:4d} lead_y mean={y.mean():.6f} dY_mean={dy.mean():.6f} "
                f"dY_min={dy.min():.6f} dY_max={dy.max():.6f} | ee_dY_mean={dey.mean():.6f} "
                f"ee_dY_max={dey.max():.6f} | ee_dZ mean={ez.mean():.6f} min={ez.min():.6f} "
                f"max={ez.max():.6f} | max|qvel|={qvel.abs().max():.3f}"
            )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
