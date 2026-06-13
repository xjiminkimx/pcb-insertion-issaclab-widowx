"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/grasp shaping, regularization, rail reset, and drop detection.

Grasp and insert are separate registered envs (``Isaac-WidowX-PCB-Grasp-v0``,
``Isaac-WidowX-PCB-Insert-v0``); each uses its own reward config with no in-episode phase gating.
"""

import torch
import numpy as np
import isaaclab.utils.math as math_utils
from dataclasses import MISSING
from collections.abc import Sequence
from pathlib import Path

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

# World push axis (must match ``PUSH_AXIS_WORLD`` in env cfg).
_DEFAULT_PUSH_AXIS_WORLD = (0.0, 1.0, 0.0)

# ---------------------------------------------------------
# Insert-phase actions — relative joint deltas with position-target clamps
# ---------------------------------------------------------
from isaaclab.envs.mdp.actions import joint_actions  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg  # noqa: E402


class RelativeJointPositionActionWithPosLimits(joint_actions.RelativeJointPositionAction):
    """Relative joint position action that clamps final PD targets per joint."""

    cfg: "RelativeJointPositionActionWithPosLimitsCfg"

    def __init__(self, cfg: "RelativeJointPositionActionWithPosLimitsCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_pos_limits: torch.Tensor | None = None
        if cfg.joint_pos_limits is not None:
            limits = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.joint_pos_limits,
                self._joint_names,
                preserve_order=self.cfg.preserve_order,
            )
            limits[:, index_list] = torch.tensor(value_list, device=self.device)
            self._joint_pos_limits = limits

    def apply_actions(self) -> None:
        targets = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        if self._joint_pos_limits is not None:
            targets = torch.clamp(
                targets,
                min=self._joint_pos_limits[:, :, 0],
                max=self._joint_pos_limits[:, :, 1],
            )
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class RelativeJointPositionActionWithPosLimitsCfg(RelativeJointPositionActionCfg):
    """Relative joint deltas with optional clamps on the commanded joint **positions**."""

    class_type: type[ActionTerm] = RelativeJointPositionActionWithPosLimits
    joint_pos_limits: dict[str, tuple[float, float]] | None = None
    """Per-joint ``(min, max)`` on ``q_current + scaled_action`` (rad), keyed by joint name."""


# ---------------------------------------------------------
# Gripper kinematics — body origins vs pad tips (wxai carriage offset)
# ---------------------------------------------------------
def _gripper_tip_offset_direction_w(
    robot,
    left: torch.Tensor,
    right: torch.Tensor,
    wrist_body_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """Unit vector from wrist toward jaw bodies (continue distally to pad tips)."""
    mid = 0.5 * (left + right)
    if wrist_body_cfg is not None and len(wrist_body_cfg.body_ids) > 0:
        wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
        d = mid - wrist
    else:
        span = right - left
        u = span / torch.norm(span, dim=-1, keepdim=True).clamp_min(1e-6)
        z_up = torch.tensor([0.0, 0.0, 1.0], device=mid.device, dtype=mid.dtype).unsqueeze(0).expand_as(mid)
        d = torch.cross(u, z_up, dim=-1)
        x_pref = torch.tensor([1.0, 0.0, 0.0], device=mid.device, dtype=mid.dtype).unsqueeze(0).expand_as(mid)
        flip = (torch.sum(d * x_pref, dim=-1, keepdim=True) < 0.0).to(dtype=d.dtype)
        d = torch.where(flip, -d, d)
    return d / torch.norm(d, dim=-1, keepdim=True).clamp_min(1e-6)


def gripper_finger_tips_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of left/right **pad tips** (body origin + distal offset along wrist→jaw axis)."""
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    if tip_offset_m <= 0.0:
        return left, right
    fwd = _gripper_tip_offset_direction_w(robot, left, right, wrist_body_cfg)
    off = float(tip_offset_m) * fwd
    return left + off, right + off


def gripper_midpoint_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """World position at the midpoint between pad tips (or body origins when offset is 0)."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    return 0.5 * (left + right)


def gripper_midpoint_lin_vel_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World linear velocity at the jaw midpoint (average of finger body velocities)."""
    robot: Articulation = env.scene[left_finger_cfg.name]
    left_ids, _ = robot.find_bodies(left_finger_cfg.body_names)
    right_ids, _ = robot.find_bodies(right_finger_cfg.body_names)
    v_left = robot.data.body_lin_vel_w[:, left_ids[0], :3]
    v_right = robot.data.body_lin_vel_w[:, right_ids[0], :3]
    return 0.5 * (v_left + v_right)


def _pcb_off_axis_speed(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """``sqrt(v_x² + v_z²)`` of the PCB root (insertion off-axis speed)."""
    v = env.scene[pcb_cfg.name].data.root_lin_vel_w
    return torch.sqrt(torch.square(v[:, 0]) + torch.square(v[:, 2]))


def _gripper_tip_params(
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> dict:
    """Kwargs bundle for pad-tip offset (passed from env cfg)."""
    return {"tip_offset_m": tip_offset_m, "wrist_body_cfg": wrist_body_cfg}


def _gripper_closedness_to_target(
    gq: torch.Tensor,
    open_width_m: float,
    closed_target_m: float,
) -> torch.Tensor:
    """Normalized closedness in ``[0, 1]``: 0 at ``open_width_m``, 1 at ``closed_target_m``.

    ``closed_target_m`` should match the carriage joint value when the jaws pinch the PCB
    (typically ``PCB_Z * 0.5`` for symmetric carriage travel).
    """
    denom = max(float(open_width_m) - float(closed_target_m), 1e-6)
    return ((float(open_width_m) - gq) / denom).clamp(0.0, 1.0)


def _gripper_gap_below_threshold(gq: torch.Tensor, max_gripper_gap_m: float) -> torch.Tensor:
    """True when the carriage joint opening is below ``max_gripper_gap_m``."""
    return gq < float(max_gripper_gap_m)


def _world_z_up_batch(env: ManagerBasedRLEnv, dtype: torch.dtype) -> torch.Tensor:
    """Unit world +Z (vertical, normal to the env XY plane), shape ``(num_envs, 3)``."""
    return torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=dtype).unsqueeze(0).expand(env.num_envs, 3)


def _world_axis_batch(
    env: ManagerBasedRLEnv,
    axis_world: tuple[float, float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unit axis in world frame, shape ``(num_envs, 3)``."""
    ax = torch.tensor(axis_world, device=env.device, dtype=dtype)
    ax = ax / torch.norm(ax).clamp(min=1e-6)
    return ax.unsqueeze(0).expand(env.num_envs, 3)


def _gripper_rail_unit_lr(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit jaw-rail direction (left→right) and separation norm."""
    v = right - left
    n = torch.norm(v, dim=-1)
    u_lr = v / n.clamp(min=1e-6).unsqueeze(-1)
    return u_lr, n


def gripper_rail_align_world_z(
    env: ManagerBasedRLEnv,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """``|dot(u_lr, world +Z)|`` in ``[0, 1]`` — jaw rail vertical (⊥ XY plane, top/bottom close)."""
    u_lr, _ = _gripper_rail_unit_lr(left, right)
    z_up = _world_z_up_batch(env, u_lr.dtype)
    return torch.abs(torch.sum(u_lr * z_up, dim=-1))


def gripper_rail_horizontal_component(
    env: ManagerBasedRLEnv,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Horizontal fraction of jaw rail in ``[0, 1]`` — 1 when rail lies in the XY plane (wrong for pinch)."""
    u_lr, _ = _gripper_rail_unit_lr(left, right)
    return torch.sqrt(u_lr[:, 0] ** 2 + u_lr[:, 1] ** 2 + 1e-12).clamp(0.0, 1.0)


def _gripper_top_bottom_near_gate(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float,
    width_weight: float,
    tip_offset_m: float,
    wrist_body_cfg: SceneEntityCfg | None,
    min_finger_sep_m: float,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D proximity gate to trailing-edge centre and finger-separation ok mask.

    Uses ``exp(-dist / gate_dist_m)`` only — no separate along-axis gate (orientation
    rewards are already gated by 3D distance to the edge centre).
    """
    _, n = _gripper_rail_unit_lr(left, right)
    sep_ok = (n > float(min_finger_sep_m)).to(dtype=n.dtype)
    dist, _, _, _ = _trailing_edge_weighted_distance(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    near = torch.exp(-dist / (float(gate_dist_m) + 1e-9))
    return near, sep_ok


def gripper_wrist_carriage_align_axis(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    tip_offset_m: float = 0.0,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """``|dot(u_wc, axis_world)|`` in ``[0, 1]`` — wrist→carriage mid parallel to push axis (+Y)."""
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    mid = 0.5 * (left + right)
    d = mid - wrist
    u_wc = d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
    axis = _world_axis_batch(env, axis_world, u_wc.dtype)
    return torch.abs(torch.sum(u_wc * axis, dim=-1))


def _gripper_belt_corridor_x_bounds_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    origins: torch.Tensor,
    wrist_body_cfg: SceneEntityCfg | None = None,
    corridor_tip_offset_m: float = 0.0,
    jaw_lateral_half_width_m: float = 0.010,
    wrist_lateral_half_width_m: float = 0.012,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conservative env-local X span and per-body Z for belt-corridor checks.

    Samples ``gripper_left`` / ``gripper_right`` (optional distal offset) and ``link_6``,
    then expands each point by a lateral half-width so the penalty matches collision meshes
    that extend beyond body origins.

    Returns ``x_lo, x_hi, z_left, z_right, z_wrist`` (``z_wrist`` is zeros when no wrist cfg).
    """
    robot = env.scene[left_finger_cfg.name]
    left, right = gripper_finger_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        corridor_tip_offset_m,
        wrist_body_cfg,
    )
    left_env = left - origins
    right_env = right - origins
    jaw_hw = float(jaw_lateral_half_width_m)

    x_lo = torch.minimum(left_env[:, 0] - jaw_hw, right_env[:, 0] - jaw_hw)
    x_hi = torch.maximum(left_env[:, 0] + jaw_hw, right_env[:, 0] + jaw_hw)
    z_left = left_env[:, 2]
    z_right = right_env[:, 2]
    z_wrist = torch.zeros_like(z_left)

    if wrist_body_cfg is not None and len(wrist_body_cfg.body_ids) > 0:
        wrist_env = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]] - origins
        w_hw = float(wrist_lateral_half_width_m)
        x_lo = torch.minimum(x_lo, wrist_env[:, 0] - w_hw)
        x_hi = torch.maximum(x_hi, wrist_env[:, 0] + w_hw)
        z_wrist = wrist_env[:, 2]

    return x_lo, x_hi, z_left, z_right, z_wrist


# ---------------------------------------------------------
# 관측 (Observations)
# ---------------------------------------------------------
def gripper_midpoint_position_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.2,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Env-local position at the midpoint between pad tips (true grasp center).

    ``gripper_left`` / ``gripper_right`` USD body origins sit on the carriage housing; add
    ``tip_offset_m`` (typically ~0.06 m) along wrist→jaw so rewards/obs match contact pads.
    """
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)
    return mid - env.scene.env_origins


def asset_root_pos_env(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Env-local root position ``(N, 3)`` — subtracts ``env_origins`` from ``root_pos_w``."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def pcb_leading_short_edge_center_env(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Env-local position of the leading short-edge face centre ``(N, 3)``."""
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m, axis_world)
    return lead_w - env.scene.env_origins[:, :3]


def joint_pos_rel_episode_reset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reset_joint_pos_attr: str = "_insert_reset_joint_pos",
) -> torch.Tensor:
    """Joint positions relative to the robot pose stored at insert reset (grasp buffer row).

    Unlike ``joint_pos_rel`` (vs USD default / HOME), this zeros at the actual Phase-1
    terminal pose so the policy sees deltas from the grasp configuration.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos
    if hasattr(env, reset_joint_pos_attr):
        ref = getattr(env, reset_joint_pos_attr)
        if ref.shape == q.shape:
            return q - ref
    return q - robot.data.default_joint_pos


# ---------------------------------------------------------
# 보상 (Rewards - Task)
# ---------------------------------------------------------
def pcb_body_axis_x_world(env: ManagerBasedRLEnv, pcb_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unit vector of PCB body +X (long edge) in world frame."""
    pcb = env.scene[pcb_cfg.name]
    q = pcb.data.root_quat_w
    local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    x_w = math_utils.quat_apply(q, local_x)
    return x_w / torch.norm(x_w, dim=-1, keepdim=True).clamp_min(1e-6)


def pcb_body_axis_y_world(env: ManagerBasedRLEnv, pcb_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unit vector of PCB body +Y (short in-plane axis) in world frame."""
    pcb = env.scene[pcb_cfg.name]
    q = pcb.data.root_quat_w
    local_y = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    y_w = math_utils.quat_apply(q, local_y)
    return y_w / torch.norm(y_w, dim=-1, keepdim=True).clamp_min(1e-6)


def pcb_body_axis_z_world(env: ManagerBasedRLEnv, pcb_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unit vector of PCB body +Z (thickness / face normal) in world frame."""
    pcb = env.scene[pcb_cfg.name]
    q = pcb.data.root_quat_w
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    z_w = math_utils.quat_apply(q, local_z)
    return z_w / torch.norm(z_w, dim=-1, keepdim=True).clamp_min(1e-6)


def pcb_body_x_push_sign(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """``+1`` when body +X aligns with ``axis_world``; ``-1`` when opposite (shape ``N``)."""
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    a = torch.tensor(axis_world, device=x_w.device, dtype=x_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    s = torch.sum(x_w * a.unsqueeze(0).expand_as(x_w), dim=-1)
    return torch.where(torch.abs(s) < 1e-6, torch.ones_like(s), torch.sign(s))


def pcb_trailing_short_edge_center_w(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """World position of the trailing short-edge face centre (opposite push / slot direction)."""
    pcb = env.scene[pcb_cfg.name]
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    sign = pcb_body_x_push_sign(env, pcb_cfg, axis_world)
    return pcb.data.root_pos_w - sign.unsqueeze(-1) * float(half_length_m) * x_w


def pcb_leading_short_edge_center_w(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """World position of the leading short-edge face centre (toward push / slot direction)."""
    pcb = env.scene[pcb_cfg.name]
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    sign = pcb_body_x_push_sign(env, pcb_cfg, axis_world)
    return pcb.data.root_pos_w + sign.unsqueeze(-1) * float(half_length_m) * x_w


def _gripper_mid_trailing_edge_errors(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    target_offset_w: torch.Tensor | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCB-frame errors vs the trailing short-edge **face centre** target.

    Returns ``along, width, thick, in_plane, edge_dist`` where width is body +Y (short-edge width).
    """
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)
    long_axis = pcb_body_axis_x_world(env, pcb_cfg)
    y_axis = pcb_body_axis_y_world(env, pcb_cfg)
    z_axis = pcb_body_axis_z_world(env, pcb_cfg)
    target = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    if target_offset_w is not None:
        target = target + target_offset_w
    delta = mid - target
    along = torch.sum(delta * long_axis, dim=-1)
    width = torch.sum(delta * y_axis, dim=-1)
    thick = torch.sum(delta * z_axis, dim=-1)
    in_plane = torch.sqrt(width * width + thick * thick + 1e-12)
    edge_dist = torch.sqrt(along * along + width * width + thick * thick + 1e-12)
    return along, width, thick, in_plane, edge_dist


def _finger_trailing_edge_grasp_targets(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-space grasp targets on the trailing short edge: centre ± half thickness along body +Z."""
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    h = float(pcb_half_thickness_m)
    return center + z_w * h, center - z_w * h


def _finger_tip_errors_vs_trailing_target(
    tip_w: torch.Tensor,
    target_w: torch.Tensor,
    long_axis: torch.Tensor,
    y_axis: torch.Tensor,
    z_axis: torch.Tensor,
    width_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCB-frame errors for one jaw tip vs a trailing-edge target (along / width / thick / 3D dist)."""
    delta = tip_w - target_w
    along = torch.sum(delta * long_axis, dim=-1)
    width = torch.sum(delta * y_axis, dim=-1)
    thick = torch.sum(delta * z_axis, dim=-1)
    ww = float(width_weight)
    # Use clamp(min=0) + 1e-6 to avoid NaN from negative fp16 subnormals under mixed precision.
    sq_sum = (along * along + (ww * width) * (ww * width) + thick * thick).clamp(min=0.0)
    dist = torch.sqrt(sq_sum + 1e-6)
    return along, width, thick, dist


def _fingers_trailing_edge_geometry(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.00125,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> dict[str, torch.Tensor]:
    """Per-jaw geometry vs opposite-side trailing-edge targets (not the jaw midpoint)."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    left_tgt, right_tgt = _finger_trailing_edge_grasp_targets(
        env, pcb_cfg, half_length_m, pcb_half_thickness_m
    )
    long_axis = pcb_body_axis_x_world(env, pcb_cfg)
    y_axis = pcb_body_axis_y_world(env, pcb_cfg)
    z_axis = pcb_body_axis_z_world(env, pcb_cfg)
    along_l, width_l, thick_l, dist_l = _finger_tip_errors_vs_trailing_target(
        left, left_tgt, long_axis, y_axis, z_axis, width_weight
    )
    along_r, width_r, thick_r, dist_r = _finger_tip_errors_vs_trailing_target(
        right, right_tgt, long_axis, y_axis, z_axis, width_weight
    )
    return {
        "left_tgt": left_tgt,
        "right_tgt": right_tgt,
        "along_l": along_l,
        "along_r": along_r,
        "width_l": width_l,
        "width_r": width_r,
        "thick_l": thick_l,
        "thick_r": thick_r,
        "dist_l": dist_l,
        "dist_r": dist_r,
    }


def _trailing_edge_along_gate(
    along: torch.Tensor,
    along_sigma_m: float = 0.025,
) -> torch.Tensor:
    """≈1 at the **trailing** short edge (|along| small); ≈0 at the leading (slot-side) edge."""
    return torch.exp(-torch.abs(along) / (float(along_sigma_m) + 1e-9))


def _finger_thickness_offsets(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed offset of each jaw tip along PCB body +Z from the board centre (thickness axis)."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    w_left = torch.sum((left - pcb_pos) * z_w, dim=-1)
    w_right = torch.sum((right - pcb_pos) * z_w, dim=-1)
    return w_left, w_right


def gripper_mid_to_pcb_trailing_edge_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    width_weight: float = 3.0,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Bottleneck distance: max of left/right jaw tip distance to per-finger trailing-edge targets."""
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    return torch.maximum(geom["dist_l"], geom["dist_r"])


def _trailing_edge_xy_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Width-weighted horizontal distance to the trailing edge centre and PCB-frame errors.

    Returns ``dist_xy, along, width`` (ignores thickness — use for XY gates / approach shaping).
    """
    along, width, _, _, _ = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    ww = float(width_weight)
    dist_xy = torch.sqrt(along * along + (ww * width) * (ww * width) + 1e-12)
    return dist_xy, along, width


def _trailing_edge_weighted_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    width_weight: float = 3.0,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-finger trailing-edge distances; returns bottleneck dist and mean along/width/thick errors."""
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    dist = torch.maximum(geom["dist_l"], geom["dist_r"])
    along = 0.5 * (geom["along_l"] + geom["along_r"])
    width = 0.5 * (geom["width_l"] + geom["width_r"])
    thick = 0.5 * (geom["thick_l"] + geom["thick_r"])
    return dist, along, width, thick


def _trailing_edge_near_factor(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.10,
    near_along_m: float = 0.030,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trailing-edge 3D proximity factor in ``[0, 1]`` plus ``(dist_3d, |thick|)``."""
    dist, along, _, thick = _trailing_edge_weighted_distance(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    along_gate = _trailing_edge_along_gate(along, near_along_m)
    near = along_gate * torch.exp(-dist / (float(gate_dist_m) + 1e-9))
    return near, dist, torch.abs(thick)


def _trailing_edge_xy_near_factor(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.10,
    near_along_m: float = 0.030,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias for :func:`_trailing_edge_near_factor` (full 3D target, not XY-only)."""
    return _trailing_edge_near_factor(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m=gate_dist_m,
        near_along_m=near_along_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def gripper_trailing_edge_proximity_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    sigma_m: float = 0.10,
    near_along_m: float = 0.030,
    width_weight: float = 3.0,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Dense shaping in ``[0, 1]``: each jaw tip must approach its own trailing-edge target."""
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    gate_l = _trailing_edge_along_gate(geom["along_l"], near_along_m)
    gate_r = _trailing_edge_along_gate(geom["along_r"], near_along_m)
    sig = float(sigma_m) + 1e-9
    prox_l = gate_l * torch.exp(-geom["dist_l"] / sig)
    prox_r = gate_r * torch.exp(-geom["dist_r"] / sig)
    return torch.minimum(prox_l, prox_r)


# Per-env previous 3D distances to per-finger trailing-edge targets.
_EE_TRAILING_EDGE_PREV_DIST_L: torch.Tensor | None = None
_EE_TRAILING_EDGE_PREV_DIST_R: torch.Tensor | None = None


def gripper_trailing_edge_approach_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    near_along_m: float = 0.030,
    max_step_m: float = 0.008,
    width_weight: float = 3.0,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Progress reward for each jaw tip moving toward its own trailing-edge target."""
    global _EE_TRAILING_EDGE_PREV_DIST_L, _EE_TRAILING_EDGE_PREV_DIST_R

    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    dist_l = geom["dist_l"]
    dist_r = geom["dist_r"]
    gate_l = _trailing_edge_along_gate(geom["along_l"], near_along_m)
    gate_r = _trailing_edge_along_gate(geom["along_r"], near_along_m)

    if (
        _EE_TRAILING_EDGE_PREV_DIST_L is None
        or _EE_TRAILING_EDGE_PREV_DIST_L.shape[0] != dist_l.shape[0]
        or _EE_TRAILING_EDGE_PREV_DIST_L.device != dist_l.device
    ):
        _EE_TRAILING_EDGE_PREV_DIST_L = dist_l.clone()
        _EE_TRAILING_EDGE_PREV_DIST_R = dist_r.clone()
        return torch.zeros_like(dist_l)

    first_step = env.episode_length_buf == 1
    _EE_TRAILING_EDGE_PREV_DIST_L = torch.where(first_step, dist_l, _EE_TRAILING_EDGE_PREV_DIST_L)
    _EE_TRAILING_EDGE_PREV_DIST_R = torch.where(first_step, dist_r, _EE_TRAILING_EDGE_PREV_DIST_R)
    prog_l = (_EE_TRAILING_EDGE_PREV_DIST_L - dist_l).clamp(0.0, float(max_step_m))
    prog_r = (_EE_TRAILING_EDGE_PREV_DIST_R - dist_r).clamp(0.0, float(max_step_m))
    _EE_TRAILING_EDGE_PREV_DIST_L = dist_l.clone()
    _EE_TRAILING_EDGE_PREV_DIST_R = dist_r.clone()
    scale = float(max_step_m) + 1e-9
    return gate_l * prog_l / scale + gate_r * prog_r / scale


def gripper_trailing_edge_xy_proximity_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    sigma_m: float = 0.10,
    near_along_m: float = 0.030,
    thick_couple_sigma_m: float = 0.025,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Deprecated — use :func:`gripper_trailing_edge_proximity_shaping` (3D target)."""
    return gripper_trailing_edge_proximity_shaping(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        sigma_m=sigma_m,
        near_along_m=near_along_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def gripper_trailing_edge_thickness_descent_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.10,
    near_along_m: float = 0.030,
    near_xy_min: float = 0.35,
    max_step_m: float = 0.006,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Deprecated — use :func:`gripper_trailing_edge_approach_progress` (3D target)."""
    return gripper_trailing_edge_approach_progress(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        near_along_m=near_along_m,
        max_step_m=max_step_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def grasp_edge_center_achieved(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    open_width_m: float,
    max_gripper_gap_m: float = 0.00055,
    gate_dist_m: float = 0.06,
    width_frac: float = 0.10,
    min_pinch_ready: float = 0.55,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    pcb_half_thickness_m: float = 0.00125,
    min_straddle_sep_m: float = 0.0012,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """True when the gripper has a valid closed pinch on the trailing short-edge centre.

    Closedness is ``left_carriage_joint < max_gripper_gap_m`` (typically ``PCB_Z * 1.1``).
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        width_weight=width_weight,
        pcb_half_thickness_m=pcb_half_thickness_m,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    closed = _gripper_gap_below_threshold(gq, max_gripper_gap_m)
    near = dist < float(gate_dist_m)
    centered = (torch.abs(geom["width_l"]) < float(half_width_m) * float(width_frac)) & (
        torch.abs(geom["width_r"]) < float(half_width_m) * float(width_frac)
    )
    pinch = gripper_pinch_readiness(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
        push_axis_world=push_axis_world,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )
    w_left, w_right = _finger_thickness_offsets(env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m=tip_offset_m, wrist_body_cfg=wrist_body_cfg)
    straddled = (w_left * w_right < 0) & (torch.abs(w_left - w_right) >= float(min_straddle_sep_m))
    return closed & near & centered & (pinch >= float(min_pinch_ready)) & straddled



def grasp_success_bonus_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    open_width_m: float,
    max_gripper_gap_m: float = 0.00055,
    gate_dist_m: float = 0.06,
    width_frac: float = 0.10,
    min_pinch_ready: float = 0.55,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    pcb_half_thickness_m: float = 0.00125,
    min_straddle_sep_m: float = 0.0012,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Bonus (1.0) on steps where a valid edge-centre grasp is achieved; for grasp-only training."""
    achieved = grasp_edge_center_achieved(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        half_width_m,
        open_width_m,
        max_gripper_gap_m=max_gripper_gap_m,
        gate_dist_m=gate_dist_m,
        width_frac=width_frac,
        min_pinch_ready=min_pinch_ready,
        width_weight=width_weight,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        min_straddle_sep_m=min_straddle_sep_m,
        push_axis_world=push_axis_world,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )
    bonus = achieved.to(dtype=env.scene[pcb_cfg.name].data.root_pos_w.dtype)
    return bonus


# Deprecated alias — kept for old env cfgs / checkpoints logging names.
_EE_XY_APPROACH_PREV_DIST: torch.Tensor | None = None


def ee_xy_approach_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_step_m: float = 0.05,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Deprecated — use :func:`gripper_trailing_edge_approach_progress` (3D target)."""
    return gripper_trailing_edge_approach_progress(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        max_step_m=max_step_m,
        width_weight=width_weight,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def gripper_pinch_readiness(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Soft readiness in ``[0, 1]`` for closing: thickness + jaw rail + wrist→carriage alignment.

    ``sep_ok`` uses the **thickness-axis straddle gap** ``|w_left - w_right|`` (jaw separation
    along the PCB normal), not the raw 3D distance between the jaw body origins.  A genuine grip
    on a 2.5 mm board drives the body-origin distance small, so the old ``n > min_finger_sep_m``
    raw check spuriously failed (``sep_ok = 0``) exactly when the gripper was correctly closed.
    The straddle gap stays ~board-thickness while straddling, so it is the correct, geometry-
    independent anti-degenerate signal.

    ``wc_align`` is floored so a non-ideal wrist→carriage heading cannot by itself zero out the
    readiness; it still contributes a continuous bonus toward the push-axis-aligned pose.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    thickness_ok = torch.exp(-torch.abs(w) / thickness_sigma_m)

    rail_align_z = gripper_rail_align_world_z(env, left, right)
    wc_align = gripper_wrist_carriage_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        tip_offset_m,
        push_axis_world,
    )
    # Floor wc_align at 0.5 so a non-ideal heading reduces but never zeroes the readiness.
    wc_align = 0.5 + 0.5 * wc_align

    # Anti-degenerate: jaws straddle the board faces with a real gap along the thickness axis.
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    straddle_gap = torch.abs(w_left - w_right)
    sep_ok = (straddle_gap > float(min_finger_sep_m)).to(dtype=thickness_ok.dtype)
    return thickness_ok * rail_align_z * wc_align * sep_ok


def pcb_object_gripper_mid_distance(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward reaching the PCB with the jaw midpoint (Isaac Lab ``object_ee_distance`` / lift task).

    See ``isaaclab_tasks/.../lift/mdp/rewards.py::object_ee_distance``.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos_w = pcb.data.root_pos_w
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)
    distance = torch.norm(pcb_pos_w - mid, dim=-1)
    return 1.0 - torch.tanh(distance / (float(std) + 1e-9))


def pcb_between_gripper_fingers(
    env: ManagerBasedRLEnv,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.0005,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
    width_sigma_m: float = 0.010,
) -> torch.Tensor:
    """Straddle-quality reward: how well the PCB is positioned between the jaws (position only).

    Four multiplicative factors — purely geometric, **no closedness** (closing is handled by
    the separate ``gripper_closing_reward`` term so the closing gradient is not diluted or
    suppressed by these position factors):

    1. ``is_graspable``  — hard gate: one jaw on each face of the PCB (thickness axis sign).
    2. ``between_jaws``  — hard gate: PCB centre within the jaw-span projection.
    3. ``prox``          — ``min(exp(-dist_l/σ), exp(-dist_r/σ))`` per-finger proximity to the
                           trailing-edge ±half-thickness targets (the real board faces).
                           σ=proximity_sigma_m is wide enough that the gradient spans the full
                           approach distance.
    4. ``width_centre``  — ``exp(-|mean_Y_err|/σ_w)`` — jaws centred along the PCB short edge.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )

    # Factor 1: one jaw on each face of the board along the thickness axis.
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    is_graspable = (w_left * w_right < 0.0).to(left.dtype)

    # Factor 2: PCB centre lies within the jaw-span projection.
    span = right - left
    span_len_sq = torch.sum(span * span, dim=-1).clamp_min(1e-12)
    span_frac = torch.sum((pcb_pos - left) * span, dim=-1) / span_len_sq
    margin = float(min_span_frac)
    between_jaws = ((span_frac >= margin) & (span_frac <= (1.0 - margin))).to(left.dtype)

    # Factor 3: per-finger proximity to trailing-edge ±half-thickness targets (real board faces).
    geom = _fingers_trailing_edge_geometry(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg,
        half_length_m, pcb_half_thickness_m,
        tip_offset_m=tip_offset_m, wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(proximity_sigma_m) + 1e-9
    prox = torch.minimum(
        torch.exp(-geom["dist_l"] / sig),
        torch.exp(-geom["dist_r"] / sig),
    )

    # Factor 4: width-centring along PCB Y-axis.
    mean_width_err = 0.5 * (geom["width_l"] + geom["width_r"])
    wsig = float(width_sigma_m) + 1e-9
    width_centre = torch.exp(-torch.abs(mean_width_err) / wsig)

    return is_graspable * between_jaws * prox * width_centre


# Consecutive env steps with straddle quality + gripper closedness above hold thresholds.
_GRASP_HOLD_STEPS: torch.Tensor | None = None


def pcb_between_gripper_fingers_hold_reward(
    env: ManagerBasedRLEnv,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gripper_joint_cfg: SceneEntityCfg,
    open_width_m: float,
    closed_target_m: float = 0.00125,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.05,
    width_sigma_m: float = 0.020,
    hold_threshold: float = 0.25,
    min_closedness: float = 0.85,
    max_hold_steps: int = 80,
) -> torch.Tensor:
    """Sustained **grasp** bonus: ramps while straddle quality stays high **and** the gripper closes.

    Straddle-only hold would let the policy earn bonus with open jaws.  The hold counter runs only
    when **both**:

    * ``quality >= hold_threshold`` — PCB between jaws (``pcb_between_gripper_fingers``).
    * ``closedness >= min_closedness`` — carriage has closed enough to pinch the board.

    Reward each step: ``quality × closedness × (hold_steps / max_hold_steps)``.  Counter resets
    when either condition fails or the episode restarts.

    Sequence: approach → straddle → close → **hold closed grasp** (this term).
    """
    global _GRASP_HOLD_STEPS
    quality = pcb_between_gripper_fingers(
        env,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m,
        wrist_body_cfg,
        min_span_frac,
        width_sigma_m,
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    closedness = _gripper_closedness_to_target(gq, open_width_m, closed_target_m)

    first_step = env.episode_length_buf == 1
    grasping = (quality >= float(hold_threshold)) & (closedness >= float(min_closedness))
    if (
        _GRASP_HOLD_STEPS is None
        or _GRASP_HOLD_STEPS.shape[0] != env.num_envs
        or _GRASP_HOLD_STEPS.device != env.device
    ):
        _GRASP_HOLD_STEPS = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    _GRASP_HOLD_STEPS = torch.where(
        first_step | (~grasping),
        torch.zeros_like(_GRASP_HOLD_STEPS),
        _GRASP_HOLD_STEPS + 1,
    )
    hold_frac = (
        _GRASP_HOLD_STEPS.to(dtype=quality.dtype) / float(max(max_hold_steps, 1))
    ).clamp(0.0, 1.0)
    return torch.where(grasping, quality * closedness * hold_frac, torch.zeros_like(quality))


def gripper_closing_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float,
    closed_target_m: float = 0.00125,
    pcb_cfg: SceneEntityCfg | None = None,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Closing reward **gated on straddle**: only pays when one jaw is on each face of the board.

    Without the straddle gate the policy collapses to a degenerate strategy: rush to the PCB
    trailing edge, close immediately on top of the board face (not straddling), and collect the
    closing reward without ever achieving a proper grasp.  Gating on ``is_graspable`` (one jaw
    above, one below the board centre along the PCB thickness axis) forces the sequence:

    1. Approach with open jaws (``finger_proximity`` + ``pcb_between_fingers`` drive this).
    2. Achieve straddle (``w_left * w_right < 0``).
    3. Only then does ``gripper_closing_reward`` become non-zero → close on the straddled board.

    If ``pcb_cfg`` / finger cfgs are None the gate is skipped (ungated, for debugging).
    """
    robot = env.scene[asset_cfg.name]
    gq = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    closedness = _gripper_closedness_to_target(gq, open_width_m, closed_target_m)

    if pcb_cfg is None or left_finger_cfg is None or right_finger_cfg is None:
        return closedness

    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    is_straddled = (w_left * w_right < 0.0).to(closedness.dtype)
    return is_straddled * closedness


def premature_close_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    closed_target_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Returns ``closedness`` when NOT straddling, 0 otherwise.

    Use with a **negative weight** to penalise the policy for closing the gripper before
    the PCB is between the jaws.  Without this, the policy receives no gradient signal about
    the gripper state during the approach phase (``pcb_between_fingers = 0`` when not
    straddling), so it may randomly close — and PPO can lock in that habit.

    Penalty value:
        ``(1 - is_straddled) × closedness``
    = 0 when jaws straddle the PCB (correct time to close)
    = closedness when approaching but not yet straddling (wrong time to close)

    The straddle gate is ``w_left * w_right < 0``: one jaw above, one below the PCB
    thickness-axis centre.
    """
    robot = env.scene[asset_cfg.name]
    gq = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    closedness = _gripper_closedness_to_target(gq, open_width_m, closed_target_m)
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    is_straddled = (w_left * w_right < 0.0).to(closedness.dtype)
    return (1.0 - is_straddled) * closedness


def pcb_finger_object_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.00125,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Per-finger tanh proximity to opposite-side trailing-edge targets.

    Ungated dense gradient (active from far away) so each jaw is pulled toward its own target —
    left → trailing-edge centre ``+`` half-thickness, right → centre ``-`` half-thickness. The two
    targets sit on opposite faces of the board, so maximising this term already steers the jaws
    toward a straddle without any hard ``between`` gate.
    """
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    left_tgt, right_tgt = _finger_trailing_edge_grasp_targets(
        env, pcb_cfg, half_length_m, pcb_half_thickness_m
    )
    lfinger_dist = torch.norm(left - left_tgt, dim=-1)
    rfinger_dist = torch.norm(right - right_tgt, dim=-1)
    sig = float(std) + 1e-9
    # min over fingers: both jaws must be close (no farming one finger only).
    return torch.minimum(
        1.0 - torch.tanh(lfinger_dist / sig),
        1.0 - torch.tanh(rfinger_dist / sig),
    )


def gripper_jaw_belt_corridor_penalty(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    corridor_x_min_env: float,
    corridor_x_max_env: float,
    belt_surface_z_env: float,
    height_z_max_env: float,
    height_z_min_env: float | None = None,
    wrist_body_cfg: SceneEntityCfg | None = None,
    corridor_tip_offset_m: float = 0.0,
    jaw_lateral_half_width_m: float = 0.010,
    wrist_lateral_half_width_m: float = 0.012,
    overflow_sigma_m: float = 0.008,
) -> torch.Tensor:
    """Penalty when gripper collision volume protrudes outside the side-belt corridor in world X.

    Uses jaw bodies (optional distal offset), ``link_6``, and lateral half-width margins so
    the reward tracks collision meshes — not just ``gripper_left`` / ``gripper_right`` origins.
    Active in a vertical band over the belt gap. Returns a value in ``[0, 1]``.
    """
    origins = env.scene.env_origins
    x_lo, x_hi, z_left, z_right, z_wrist = _gripper_belt_corridor_x_bounds_env(
        env,
        left_finger_cfg,
        right_finger_cfg,
        origins,
        wrist_body_cfg=wrist_body_cfg,
        corridor_tip_offset_m=corridor_tip_offset_m,
        jaw_lateral_half_width_m=jaw_lateral_half_width_m,
        wrist_lateral_half_width_m=wrist_lateral_half_width_m,
    )
    overflow = torch.relu(float(corridor_x_min_env) - x_lo) + torch.relu(
        x_hi - float(corridor_x_max_env)
    )
    corridor_violation = 1.0 - torch.exp(-overflow / (float(overflow_sigma_m) + 1e-9))

    z_min = float(height_z_min_env) if height_z_min_env is not None else float(belt_surface_z_env) - 0.015
    z_max = float(height_z_max_env)

    def _in_height_band(z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((z - z_min) / 0.008) * (1.0 - torch.sigmoid((z - z_max) / 0.012))

    in_height = torch.maximum(
        _in_height_band(z_left),
        torch.maximum(_in_height_band(z_right), _in_height_band(z_wrist)),
    )

    return corridor_violation * in_height



def gripper_mid_thickness_offset_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    scale_m: float = 0.012,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Obs: signed offset along PCB thickness axis (jaw mid vs board center), scaled to ~[-1, 1]."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    return torch.clamp(w / (scale_m + 1e-6), -1.0, 1.0).unsqueeze(-1)


def gripper_trailing_edge_error_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    scale_along_m: float = 0.12,
    scale_width_m: float = 0.04,
    scale_thick_m: float = 0.05,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Obs: jaw-mid error vs trailing short-edge centre in PCB frame, scaled to ~[-1, 1].

    Components are ``along`` (long axis), ``width`` (short edge), ``thick`` (board thickness).
    """
    along, width, thick, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, **_gripper_tip_params(tip_offset_m, wrist_body_cfg)
    )
    sa = float(scale_along_m) + 1e-6
    sw = float(scale_width_m) + 1e-6
    st = float(scale_thick_m) + 1e-6
    return torch.stack(
        [
            torch.clamp(along / sa, -1.0, 1.0),
            torch.clamp(width / sw, -1.0, 1.0),
            torch.clamp(thick / st, -1.0, 1.0),
        ],
        dim=-1,
    )



def gripper_jaw_rail_vertical_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Shaping ``[0, 1]``: reward jaw rail parallel to world +Z (top/bottom thickness close).

    Carriage must **not** stay level (∥ XY); the left↔right rail is rolled vertical so fingers
    straddle PCB thickness.
    """
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
        min_finger_sep_m,
        left,
        right,
    )
    rail_z = gripper_rail_align_world_z(env, left, right)
    return rail_z * sep_ok * near


def gripper_wrist_carriage_push_axis_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Shaping ``[0, 1]``: reward wrist (``link_6``) → carriage mid parallel to push axis (+Y)."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
        min_finger_sep_m,
        left,
        right,
    )
    wc_y = gripper_wrist_carriage_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        tip_offset_m,
        push_axis_world,
    )
    return wc_y * sep_ok * near


def gripper_jaw_rail_horizontal_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalty ``[0, 1]``: jaw rail lying in the XY plane (level carriage / width-pinch pose)."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
        min_finger_sep_m,
        left,
        right,
    )
    horiz = gripper_rail_horizontal_component(env, left, right)
    return horiz * sep_ok * near


# Backward-compatible alias (deprecated name — use split terms above).
def gripper_fingers_perpendicular_to_trailing_edge_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
    width_weight: float = 3.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Combined orientation: ``jaw_rail_vertical * wrist_carriage_push`` (legacy single term)."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
        min_finger_sep_m,
        left,
        right,
    )
    rail_z = gripper_rail_align_world_z(env, left, right)
    wc_y = gripper_wrist_carriage_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        tip_offset_m,
        push_axis_world,
    )
    return rail_z * wc_y * sep_ok * near


def gripper_pinch_orientation_cos_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    min_finger_sep_m: float = 0.006,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Two scalars in ``[0, 1]``: jaw rail ∥ +Z, wrist→carriage ∥ push (+Y); sep-scaled."""
    left, right = gripper_finger_tips_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )
    _, n = _gripper_rail_unit_lr(left, right)
    rail_z = gripper_rail_align_world_z(env, left, right)
    wc_y = gripper_wrist_carriage_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        tip_offset_m,
        push_axis_world,
    )
    sep_soft = torch.clamp(n / (float(min_finger_sep_m) + 1e-9), 0.0, 1.0)
    return torch.stack([rail_z * sep_soft, wc_y * sep_soft], dim=-1)


def gripper_opening_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float = 0.044,
    gripper_joint_sign: float = 1.0,
) -> torch.Tensor:
    """Scalar opening in ``[0, 1]`` (0 = closed, 1 = fully open) for the gripper drive joint.

    Use ``gripper_joint_sign=-1`` when larger joint values mean *more closed* (e.g. Viola ``joint7_left``
    open toward negative limits).
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    eff = gripper_joint_sign * q
    return torch.clamp(eff / open_width_m, 0.0, 1.0).unsqueeze(-1)


def pcb_lin_vel_y_toward_lead_target_y(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
) -> torch.Tensor:
    """``relu((y_target - lead_y) * v_y)`` with lead = root + half_length * body +X in env frame.

    Rewards world-Y linear velocity only when it **reduces** the Y gap to the slot (push into +Y if
    the mouth is still ahead, or -Y if the board has overshot). Complements a flat ``relu(v_y)`` term
    by not paying for +Y motion after the lead has passed the target Y.
    """
    pcb = env.scene[pcb_cfg.name]
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    err_y = float(target_lead_y_env) - lead_y
    v_y = pcb.data.root_lin_vel_w[:, 1]
    return torch.relu(err_y * v_y)


def pcb_long_axis_parallel_to_push_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Shaped in ``[0, 1]``: PCB body +X (long edge) aligned with the insertion direction.

    1.0 when the long axis is parallel to ``axis_world`` (same or opposite direction).
    """
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    a = torch.tensor(axis_world, device=env.device, dtype=x_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    c = torch.abs(torch.sum(x_w * a.unsqueeze(0).expand_as(x_w), dim=-1))
    return torch.square(torch.clamp(c, max=1.0))


def pcb_leading_edge_insertion_proximity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xyz_env: tuple[float, float, float],
    sigma_m: float = 0.12,
) -> torch.Tensor:
    """Dense reward: PCB **leading** point near a fixed slot-mouth pose in env-local frame.

    Leading point: ``pcb_center_w + half_length * body+X_world`` (same convention as insertion tasks).
    ``target_lead_xyz_env`` is the env-local position (relative to ``env_origins``) for that point —
    typically the **slot entrance center** you measured in Isaac Sim.

    Returns ``exp(-‖lead_env - target‖ / sigma_m)`` (L2 distance, tunable kernel width ``sigma_m``).
    """
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_env = lead_w - env.scene.env_origins[:, :3]
    tgt = torch.tensor(target_lead_xyz_env, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0).expand(
        env.num_envs, -1
    )
    dist = torch.norm(lead_env - tgt, dim=-1)
    return torch.exp(-dist / (float(sigma_m) + 1e-9))



def pcb_insertion_depth_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    max_depth_m: float = 0.20,
) -> torch.Tensor:
    """Reward for **PCB depth inside the slot**: leading edge past the slot mouth in +Y.

    Zero while the leading edge has not yet crossed ``slot_mouth_y_env``.
    Linearly increases up to ``max_depth_m`` of penetration (returns 1.0 at full insertion).
    Use a positive weight; combine with ``insert_y_toward_slot`` which only fires before the mouth.
    """
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    depth = torch.clamp(lead_y - float(slot_mouth_y_env), min=0.0, max=float(max_depth_m))
    return depth / float(max_depth_m)


def pcb_slot_mouth_milestone_bonus(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    state_attr: str = "_insert_milestone_slot_mouth",
) -> torch.Tensor:
    """One-shot sparse bonus when the PCB leading edge first crosses the slot mouth (+Y).

    Fires once per episode the first time ``lead_y >= slot_mouth_y_env``.  State is cleared on
    insert reset via :func:`_store_insert_progress_baselines`.
    """
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    crossed = lead_y >= float(slot_mouth_y_env)

    if not hasattr(env, state_attr):
        setattr(
            env,
            state_attr,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
    flag: torch.Tensor = getattr(env, state_attr)
    newly = crossed & ~flag
    flag |= crossed
    return newly.float()


def _pcb_rail_parallel_quality(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    rail_center_x_env: float,
    max_lateral_x_m: float,
    min_flatness: float,
    min_long_align: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Soft quality in ``[0, 1]`` for conveyor-rail-parallel push (on-lane, flat, +Y aligned)."""
    pcb = env.scene[pcb_cfg.name]
    pos_env = pcb.data.root_pos_w - env.scene.env_origins[:, :3]
    lateral_err = torch.abs(pos_env[:, 0] - float(rail_center_x_env))
    q_lane = (1.0 - lateral_err / (float(max_lateral_x_m) + 1e-9)).clamp(0.0, 1.0)

    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    up = torch.tensor(world_up, device=env.device, dtype=x_w.dtype)
    up = up / torch.norm(up).clamp_min(1e-9)
    push = torch.tensor(axis_world, device=env.device, dtype=x_w.dtype)
    push = push / torch.norm(push).clamp_min(1e-9)

    flat = torch.abs(torch.sum(z_w * up.unsqueeze(0), dim=-1))
    long_a = torch.abs(torch.sum(x_w * push.unsqueeze(0), dim=-1))
    q_flat = ((flat - float(min_flatness)) / (1.0 - float(min_flatness) + 1e-9)).clamp(0.0, 1.0)
    q_long = ((long_a - float(min_long_align)) / (1.0 - float(min_long_align) + 1e-9)).clamp(0.0, 1.0)
    return q_lane * q_flat * q_long


def pcb_rail_parallel_approach_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    slot_mouth_y_env: float,
    rail_center_x_env: float,
    max_lateral_x_m: float = 0.030,
    min_flatness: float = 0.92,
    min_long_align: float = 0.85,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.010,
) -> torch.Tensor:
    """Per-step +Y progress toward the slot mouth, gated on rail-parallel pose (pre-mouth only).

    Credits ``Δproj · quality`` where ``quality`` rewards staying on the conveyor lane (X),
    flat (thickness ∥ world +Z), and long axis ∥ push (+Y).  Zero after the PCB centre
    projection reaches ``slot_mouth_y_env``.
    """
    global _INSERT_PREV_CENTER_PROJ

    pcb = env.scene[pcb_cfg.name]
    pos_env = pcb.data.root_pos_w - env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=env.device, dtype=pos_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(pos_env * a.unsqueeze(0), dim=-1)

    before_mouth = proj < float(slot_mouth_y_env)
    quality = _pcb_rail_parallel_quality(
        env,
        pcb_cfg,
        rail_center_x_env,
        max_lateral_x_m,
        min_flatness,
        min_long_align,
        axis_world,
    )

    if (
        _INSERT_PREV_CENTER_PROJ is None
        or _INSERT_PREV_CENTER_PROJ.shape[0] != proj.shape[0]
        or _INSERT_PREV_CENTER_PROJ.device != proj.device
    ):
        _INSERT_PREV_CENTER_PROJ = proj.clone()
        return torch.zeros_like(proj)

    first_step = env.episode_length_buf == 1
    _INSERT_PREV_CENTER_PROJ = torch.where(first_step, proj, _INSERT_PREV_CENTER_PROJ)
    delta = (proj - _INSERT_PREV_CENTER_PROJ).clamp(min=0.0, max=float(max_step_m))
    _INSERT_PREV_CENTER_PROJ = proj.clone()

    step_reward = (delta / (float(max_step_m) + 1e-9)) * quality
    return torch.where(before_mouth, step_reward, torch.zeros_like(step_reward))


def pcb_rail_parallel_approach_milestones(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    slot_mouth_y_env: float,
    rail_center_x_env: float,
    tier_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_lateral_x_m: float = 0.030,
    min_flatness: float = 0.92,
    min_long_align: float = 0.85,
    min_quality: float = 0.50,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    state_attr: str = "_insert_rail_milestone_mask",
) -> torch.Tensor:
    """One-shot bonuses at fractional progress toward the slot mouth (pre-mouth, rail-parallel).

    Progress is measured from the per-episode start projection (``env._insert_start_center_proj``)
    to ``slot_mouth_y_env``.  Each tier in ``tier_fractions`` fires once when
    ``progress_frac >= tier`` and rail-parallel ``quality >= min_quality``.
    Returns the count of newly achieved tiers this step (0, 1, 2, …).
    """
    pcb = env.scene[pcb_cfg.name]
    device = env.device
    pos_env = pcb.data.root_pos_w - env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=device, dtype=pos_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(pos_env * a.unsqueeze(0), dim=-1)

    if hasattr(env, "_insert_start_center_proj"):
        start = env._insert_start_center_proj
    else:
        start = proj.detach()
    mouth = float(slot_mouth_y_env)
    denom = (mouth - start).clamp_min(1e-6)
    frac = ((proj - start) / denom).clamp(0.0, 1.0)

    before_mouth = proj < mouth
    quality = _pcb_rail_parallel_quality(
        env,
        pcb_cfg,
        rail_center_x_env,
        max_lateral_x_m,
        min_flatness,
        min_long_align,
        axis_world,
    )
    qualified = before_mouth & (quality >= float(min_quality))

    n_tiers = len(tier_fractions)
    if not hasattr(env, state_attr):
        setattr(env, state_attr, torch.zeros(env.num_envs, n_tiers, device=device, dtype=torch.bool))
    mask: torch.Tensor = getattr(env, state_attr)
    if mask.shape[1] != n_tiers:
        mask = torch.zeros(env.num_envs, n_tiers, device=device, dtype=torch.bool)
        setattr(env, state_attr, mask)

    reward = torch.zeros(env.num_envs, device=device, dtype=proj.dtype)
    tiers = torch.tensor(tier_fractions, device=device, dtype=proj.dtype)
    for i in range(n_tiers):
        reached = qualified & (frac >= tiers[i])
        newly = reached & ~mask[:, i]
        reward = reward + newly.float()
        mask[:, i] = mask[:, i] | reached
    return reward


def pcb_horizontal_velocity_perpendicular_to_axis_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Squared horizontal speed **orthogonal** to the insertion axis (lateral skidding off-axis)."""
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w.clone()
    v[:, 2] = 0.0
    a = torch.tensor(axis_world, device=env.device, dtype=v.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    a3 = a.unsqueeze(0).expand(v.shape[0], -1)
    v_para = torch.sum(v * a3, dim=-1, keepdim=True) * a3
    v_perp = v - v_para
    return torch.sum(torch.square(v_perp), dim=-1)


# ---------------------------------------------------------
# 보상 (Penalties - Regularization)
# ---------------------------------------------------------
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    행동(Action) 변화량 패널티: 
    이전 프레임의 명령과 현재 프레임의 명령 차이가 클수록 패널티를 줍니다.
    로봇 팔이 급격하게 방향을 틀거나 덜덜 떠는 현상을 방지합니다.
    """
    # Penalize abrupt action changes: smoother control -> less jitter at grasp/insert.
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def pcb_push_axis_displacement_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    initial_y_env: float,
    max_displacement_m: float = 0.02,
) -> torch.Tensor:
    """Penalty when the PCB root shifts beyond ``max_displacement_m`` toward +Y (push direction).

    Grasp phase should pinch the trailing edge without sliding the board toward the slot.
    Uses env-local Y displacement from the spawn position (``initial_y_env``).

    Returns ``relu(dy - max_displacement_m)`` — pair with a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    y_env = pcb.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    dy = y_env - float(initial_y_env)
    return torch.clamp(dy - float(max_displacement_m), min=0.0)


def pcb_x_displacement_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    reference_x_env: float,
    max_displacement_m: float = 0.01,
) -> torch.Tensor:
    """Penalty when the PCB drifts too far in the X-axis from a reference X position.

    During insertion the PCB should travel straight along world +Y (into the slot).
    Any lateral X displacement means the PCB is being dragged sideways, risking
    mis-alignment with the slot opening.

    Returns ``relu(|dx| - max_displacement_m)`` — pair with a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    x_env = pcb.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    dx = torch.abs(x_env - float(reference_x_env))
    return torch.clamp(dx - float(max_displacement_m), min=0.0)


def pcb_x_lane_boundary_exponential_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    lane_center_x_env: float,
    inner_half_width_m: float,
    exponential_scale_m: float = 0.004,
    max_excess_m: float = 0.020,
) -> torch.Tensor:
    """Exponential penalty when the PCB centre leaves the conveyor / rail lane in env-local X.

    Inside ``|x - lane_center| <= inner_half_width_m`` the penalty is zero.  Beyond that
    boundary, ``excess = |x - centre| - inner_half_width`` drives ``expm1(excess / scale)``
    (capped at ``max_excess_m`` for value stability).  The return value is normalized to
    approximately ``[0, 1]``.  Pair with a **negative** weight.

    Use this to discourage lateral skidding off the belt / guide rails while the policy
    chases dense +Y push rewards.
    """
    pcb = env.scene[pcb_cfg.name]
    x_env = pcb.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    lateral = torch.abs(x_env - float(lane_center_x_env))
    excess = torch.clamp(
        lateral - float(inner_half_width_m),
        min=0.0,
        max=float(max_excess_m),
    )
    scale = float(exponential_scale_m) + 1e-9
    cap = float(max_excess_m)
    raw = torch.expm1(excess / scale)
    norm = torch.expm1(torch.tensor(cap / scale, device=excess.device, dtype=excess.dtype)) + 1e-9
    return raw / norm


def pcb_velocity_y_purity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_speed_m_s: float = 0.002,
) -> torch.Tensor:
    """Reward the fraction of PCB velocity that is directed along world +Y (insertion axis).

    Returns ``relu(v_y) / (||v|| + ε)`` when the PCB is moving faster than ``min_speed_m_s``,
    and 0 otherwise.  This peaks at 1.0 only when motion is purely in +Y, and is 0 when moving
    in −Y or when the PCB is stationary.  Use with a positive weight to enforce >95% Y-purity.
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    v_y = v[:, 1]
    speed = torch.norm(v, dim=-1)
    moving = speed > float(min_speed_m_s)
    purity = torch.relu(v_y) / (speed + 1e-6)
    return torch.where(moving, purity, torch.zeros_like(purity))


def pcb_z_displacement_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    reference_z_env: float,
    max_displacement_m: float = 0.010,
) -> torch.Tensor:
    """Penalty when the PCB drifts more than ``max_displacement_m`` above or below its grasp height.

    During insertion the PCB should stay at the same Z as when it was grasped — any vertical
    drift means the policy is lifting or dragging the board, risking contact with slot walls.

    Returns ``relu(|dz| - max_displacement_m)`` — pair with a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    z_env = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    dz = torch.abs(z_env - float(reference_z_env))
    return torch.clamp(dz - float(max_displacement_m), min=0.0)


def pcb_height_below_reference(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_height: float = 0.04,
) -> torch.Tensor:
    """Penalty when PCB center is below a reference height (env-local Z).

    Fights dragging the board on the floor / rail plane.
    Uses env-local height: ``pcb_z - env_origin_z``.

    Returns ``relu(min_height - pcb_height_env)`` — use a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    h = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(min_height - h, min=0.0)


def slide_mouth_reached(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    margin_m: float = 0.008,
) -> torch.Tensor:
    """True when the PCB **leading** short-edge centre reaches the slot mouth plane (+Y)."""
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    return lead_y >= float(slot_mouth_y_env) - float(margin_m)


def slide_success(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    margin_m: float = 0.008,
    min_episode_steps: int = 2,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_tilt_penalty: float = 0.08,
    max_long_axis_abs_z: float = 0.15,
    min_long_axis_xy_align: float = 0.85,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    max_gripper_gap_m: float = 0.001,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_straddle_sep_m: float = 0.0,
) -> torch.Tensor:
    """Slide-phase success: mouth reached + PCB parallel to the world XY plane + grasp hold.

    Combines:

    * **Mouth** — leading short-edge centre Y ≥ slot mouth − margin.
    * **Thickness flatness** — ``pcb_thickness_axis_tilt_penalty`` ≤ ``max_tilt_penalty``
      (default 0.08 ⇒ |dot(z_body, up)| ≥ 0.92, matches rail-parallel reward flatness).
    * **Long-axis horizontal** — |world-Z component of body +X| ≤ ``max_long_axis_abs_z``
      (long edge lies in the XY plane, not wedged edge-on).
    * **Long-axis XY alignment** — body +X projected into XY aligns with ``axis_world``
      (default +Y) by at least ``min_long_axis_xy_align`` (default 0.85).
    * **Gripper closed** — ``left_carriage_joint`` < ``max_gripper_gap_m`` when
      ``gripper_joint_cfg`` is set.
    * **Straddle** — jaws on opposite faces of the PCB thickness axis when finger cfgs are set.
    """
    reached = slide_mouth_reached(
        env, pcb_cfg, half_length_m, slot_mouth_y_env, margin_m=margin_m,
    )
    tilt_ok = pcb_thickness_axis_tilt_penalty(env, pcb_cfg, world_up) <= float(max_tilt_penalty)

    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    long_horizontal_ok = torch.abs(x_w[:, 2]) <= float(max_long_axis_abs_z)

    x_xy = x_w.clone()
    x_xy[:, 2] = 0.0
    x_xy = x_xy / torch.norm(x_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    a = torch.tensor(axis_world, device=x_w.device, dtype=x_w.dtype)
    a_xy = a.clone()
    a_xy[2] = 0.0
    a_xy = a_xy / torch.norm(a_xy).clamp_min(1e-6)
    a_xy = a_xy.unsqueeze(0).expand_as(x_xy)
    long_align = torch.abs(torch.sum(x_xy * a_xy, dim=-1))
    long_align_ok = long_align >= float(min_long_axis_xy_align)

    success = reached & tilt_ok & long_horizontal_ok & long_align_ok

    if gripper_joint_cfg is not None:
        robot = env.scene[gripper_joint_cfg.name]
        gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
        gripper_ok = gq < float(max_gripper_gap_m)
        success = success & gripper_ok

    if left_finger_cfg is not None and right_finger_cfg is not None:
        w_left, w_right = _finger_thickness_offsets(
            env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
        )
        straddle_ok = (w_left * w_right < 0.0) & (
            torch.abs(w_left - w_right) >= float(min_straddle_sep_m)
        )
        success = success & straddle_ok

    if min_episode_steps > 0:
        ready = env.episode_length_buf > min_episode_steps
        success = success & ready
    return success


def slide_success_bonus_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    margin_m: float = 0.008,
    min_episode_steps: int = 2,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_tilt_penalty: float = 0.08,
    max_long_axis_abs_z: float = 0.15,
    min_long_axis_xy_align: float = 0.85,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    max_gripper_gap_m: float = 0.001,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_straddle_sep_m: float = 0.0,
) -> torch.Tensor:
    """Bonus (1.0) on steps where slide success criteria are met; mirrors :func:`slide_success`."""
    achieved = slide_success(
        env,
        pcb_cfg,
        half_length_m,
        slot_mouth_y_env,
        margin_m=margin_m,
        min_episode_steps=min_episode_steps,
        world_up=world_up,
        axis_world=axis_world,
        max_tilt_penalty=max_tilt_penalty,
        max_long_axis_abs_z=max_long_axis_abs_z,
        min_long_axis_xy_align=min_long_axis_xy_align,
        gripper_joint_cfg=gripper_joint_cfg,
        max_gripper_gap_m=max_gripper_gap_m,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        min_straddle_sep_m=min_straddle_sep_m,
    )
    return achieved.to(dtype=env.scene[pcb_cfg.name].data.root_pos_w.dtype)


def insert_success(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    magazine_center_y_env: float,
    margin_m: float = 0.01,
) -> torch.Tensor:
    """True when the PCB long-edge centre is within ``margin_m`` of the magazine Y centre.

    The "long-edge centre" of the PCB is its root position along the push axis (world +Y).
    Success is declared when the PCB centre has reached the magazine's Y midpoint, which
    means the entire PCB footprint (XY-plane) is seated inside the magazine area.

    Args:
        pcb_cfg:              Scene entity for the PCB rigid body.
        magazine_center_y_env: Target Y position of the magazine centre in env-local frame.
        margin_m:             Half-width of the acceptance window in metres (default 0.01 m).
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_y_env = pcb.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (pcb_y_env - magazine_center_y_env).abs() < margin_m


def pcb_root_height_below_env_minimum(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_height_env: float,
) -> torch.Tensor:
    """Episode done when PCB root height (env-local Z) is below ``min_height_env``.

    Same convention as ``pcb_height_below_reference`` (``root_pos_w[:,2] - env_origins[:,2]``).
    Set ``min_height_env`` just below the guide-rail top so resting on the rail is OK, but falling to
    the table / floor ends the episode.
    """
    pcb = env.scene[pcb_cfg.name]
    h = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return h < min_height_env


def pcb_thickness_axis_tilt_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Penalize PCB not lying flat (local +Z should align with world up).

    Cuboid PCB uses local Z as thickness; for a horizontal board the body +Z axis should match
    world +Z. Returns ``1 - |dot(z_body_w, up)|`` in [0, 1] — use a **negative** weight.

    Reduces the "PCB standing on edge" failure mode from bad grasps / pushing.
    """
    pcb = env.scene[pcb_cfg.name]
    q = pcb.data.root_quat_w
    up = torch.tensor(world_up, device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    up = up / torch.norm(up, dim=-1, keepdim=True).clamp_min(1e-6)
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    z_w = math_utils.quat_apply(q, local_z)
    align = torch.abs(torch.sum(z_w * up, dim=-1))
    return 1.0 - torch.clamp(align, max=1.0)


def pcb_xy_plane_parallel_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    flat_coef: float = 0.50,
    long_horizontal_coef: float = 0.30,
    long_align_coef: float = 0.20,
) -> torch.Tensor:
    """Shaped reward in ``[0, 1]`` for keeping the PCB parallel to the world XY plane.

    Three components (coefficients should sum to 1):

    * **Flatness** — body +Z (thickness) aligned with ``world_up``; board not edge-on.
    * **Long axis horizontal** — body +X has small world-Z component (long edge lies in XY).
    * **Insertion alignment** — body +X projected into XY aligns with ``axis_world`` (+Y push).

    Use a **positive** weight.  Pair with :func:`pcb_thickness_axis_tilt_penalty` (negative weight)
    if a sharper flatness gradient is needed.
    """
    pcb = env.scene[pcb_cfg.name]
    q = pcb.data.root_quat_w
    device = env.device
    dtype = q.dtype

    up = torch.tensor(world_up, device=device, dtype=dtype)
    up = up / torch.norm(up).clamp_min(1e-9)
    up = up.unsqueeze(0).expand(q.shape[0], -1)
    local_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(q.shape[0], -1)
    z_w = math_utils.quat_apply(q, local_z)
    flat = torch.abs(torch.sum(z_w * up, dim=-1))

    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    long_horizontal = 1.0 - torch.clamp(torch.abs(x_w[:, 2]), max=1.0)

    x_xy = x_w.clone()
    x_xy[:, 2] = 0.0
    x_xy = x_xy / torch.norm(x_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    a = torch.tensor(axis_world, device=device, dtype=dtype)
    a_xy = a.clone()
    a_xy[2] = 0.0
    a_xy = a_xy / torch.norm(a_xy).clamp_min(1e-9)
    a_xy = a_xy.unsqueeze(0).expand_as(x_xy)
    long_align = torch.abs(torch.sum(x_xy * a_xy, dim=-1))

    return (
        float(flat_coef) * flat
        + float(long_horizontal_coef) * long_horizontal
        + float(long_align_coef) * long_align
    ).clamp(0.0, 1.0)


def pcb_tilt_beyond_limit(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_tilt_penalty: float = 0.2,
) -> torch.Tensor:
    """Terminate when the board is too far from flat (same metric as ``pcb_thickness_axis_tilt_penalty``).

    A tilted / edge-on PCB can keep its **root height** near the rail plane, so height-only
    terminations never fire; this catches wedged-on-rail and floor-leaning failures.
    """
    t = pcb_thickness_axis_tilt_penalty(env, pcb_cfg, world_up)
    return t > max_tilt_penalty


def pcb_long_axis_vertical_component_exceeds(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    max_abs_z: float = 0.17,
) -> torch.Tensor:
    """True when body +X (long / insertion axis) has too much world-Z component.

    On a flat board in the horizontal plane, ``x_w`` lies in the XY plane. Wedged / slipped boards pick
    up a significant Z component before thickness-axis tilt alone crosses its limit.
    """
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    return torch.abs(x_w[:, 2]) > max_abs_z


def pcb_long_axis_xy_rotation_exceeds(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_xy_alignment: float = 0.995,
) -> torch.Tensor:
    """True when body +X (long axis) is rotated too far in the horizontal (XY) plane.

    Ideal grasp spawn has long axis parallel to ``axis_world`` (world +Y). Measures alignment of
    the long-axis XY projection with ``axis_world`` (ignores Z tilt component).
    """
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    x_xy = x_w.clone()
    x_xy[:, 2] = 0.0
    x_xy = x_xy / torch.norm(x_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    a = torch.tensor(axis_world, device=x_w.device, dtype=x_w.dtype)
    a_xy = a.clone()
    a_xy[2] = 0.0
    a_xy = a_xy / torch.norm(a_xy).clamp_min(1e-6)
    a_xy = a_xy.unsqueeze(0).expand_as(x_xy)
    align = torch.abs(torch.sum(x_xy * a_xy, dim=-1))
    return align < float(min_xy_alignment)


def _grasp_not_yet_achieved(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    open_width_m: float,
    max_gripper_gap_m: float = 0.00055,
    gate_dist_m: float = 0.06,
    width_frac: float = 0.10,
    min_pinch_ready: float = 0.40,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    pcb_half_thickness_m: float = 0.00125,
    min_straddle_sep_m: float = 0.0012,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """True while a valid edge-centre grasp has **not** been achieved."""
    return ~grasp_edge_center_achieved(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        half_width_m,
        open_width_m,
        max_gripper_gap_m=max_gripper_gap_m,
        gate_dist_m=gate_dist_m,
        width_frac=width_frac,
        min_pinch_ready=min_pinch_ready,
        width_weight=width_weight,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        min_straddle_sep_m=min_straddle_sep_m,
        push_axis_world=push_axis_world,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )


def pcb_tilt_before_grasp_termination(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    open_width_m: float,
    max_gripper_gap_m: float = 0.00055,
    gate_dist_m: float = 0.06,
    width_frac: float = 0.30,
    min_pinch_ready: float = 0.40,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    pcb_half_thickness_m: float = 0.00125,
    min_straddle_sep_m: float = 0.0012,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    max_tilt_penalty: float = 0.01,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Terminate on excessive thickness-axis tilt (board not flat) **before** grasp success."""
    tilt_fail = pcb_tilt_beyond_limit(env, pcb_cfg, world_up=world_up, max_tilt_penalty=max_tilt_penalty)
    pre_grasp = _grasp_not_yet_achieved(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        half_width_m,
        open_width_m,
        max_gripper_gap_m=max_gripper_gap_m,
        gate_dist_m=gate_dist_m,
        width_frac=width_frac,
        min_pinch_ready=min_pinch_ready,
        width_weight=width_weight,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        min_straddle_sep_m=min_straddle_sep_m,
        push_axis_world=push_axis_world,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )
    return tilt_fail & pre_grasp


def pcb_xy_plane_rotation_before_grasp_termination(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    open_width_m: float,
    max_gripper_gap_m: float = 0.00055,
    gate_dist_m: float = 0.06,
    width_frac: float = 0.30,
    min_pinch_ready: float = 0.40,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    pcb_half_thickness_m: float = 0.00125,
    min_straddle_sep_m: float = 0.0012,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_xy_alignment: float = 0.97,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Terminate on excessive long-axis yaw in the XY plane **before** grasp success."""
    yaw_fail = pcb_long_axis_xy_rotation_exceeds(
        env, pcb_cfg, axis_world=axis_world, min_xy_alignment=min_xy_alignment
    )
    pre_grasp = _grasp_not_yet_achieved(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        half_width_m,
        open_width_m,
        max_gripper_gap_m=max_gripper_gap_m,
        gate_dist_m=gate_dist_m,
        width_frac=width_frac,
        min_pinch_ready=min_pinch_ready,
        width_weight=width_weight,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        min_straddle_sep_m=min_straddle_sep_m,
        push_axis_world=push_axis_world,
        **_gripper_tip_params(tip_offset_m, wrist_body_cfg),
    )
    return yaw_fail & pre_grasp


# Counts consecutive env steps where the PCB moves backward (−Y).
_PCB_BACKWARD_COUNT: torch.Tensor | None = None


def reset_robot_joints_to_values(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_positions: dict[str, float],
    velocity_scale: float = 0.0,
    use_current_joint_pos: bool = False,
) -> None:
    """Set listed articulation joints to fixed positions.

    If ``use_current_joint_pos`` is False, other joints keep scene defaults. If True, starts from
    current sim joint positions (for tightening gripper after snap without moving the arm).
    """
    robot = env.scene[asset_cfg.name]
    if use_current_joint_pos:
        joint_pos = robot.data.joint_pos[env_ids].clone()
    else:
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone() * velocity_scale
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    for name, val in joint_positions.items():
        joint_pos[:, name_to_idx[name]] = val
    lim = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(lim[..., 0], lim[..., 1])
    vlim = robot.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp(-vlim, vlim)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # Refresh link poses so a following reset term reads current FK (same reset cycle, no physics step yet).
    robot.update(0.0)


def reset_robot_joints_to_values_randomized(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_positions: dict[str, float],
    joint_position_ranges: dict[str, tuple[float, float]],
    velocity_scale: float = 0.0,
) -> None:
    """Set listed joints to nominal positions plus uniform per-env offsets (domain randomization).

    For each joint in ``joint_positions``, samples
    ``nominal + Uniform(lo, hi)`` independently per environment.  Joints omitted from
    ``joint_position_ranges`` default to zero offset.  Use ``(0.0, 0.0)`` to keep a joint
    fixed (e.g. gripper open at reset).

    Values are clamped to soft joint limits before writing to the simulator.
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone() * velocity_scale
    name_to_idx = {n: i for i, n in enumerate(robot.joint_names)}
    n = len(env_ids)
    device = env.device
    dtype = joint_pos.dtype

    for name, nominal in joint_positions.items():
        idx = name_to_idx[name]
        lo, hi = joint_position_ranges.get(name, (0.0, 0.0))
        if abs(lo) < 1e-12 and abs(hi) < 1e-12:
            joint_pos[:, idx] = float(nominal)
        else:
            offset = torch.empty(n, device=device, dtype=dtype).uniform_(float(lo), float(hi))
            joint_pos[:, idx] = float(nominal) + offset

    lim = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(lim[..., 0], lim[..., 1])
    vlim = robot.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp(-vlim, vlim)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.update(0.0)


def snap_pcb_root_to_short_edge_grasp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    rot_wxyz: tuple[float, float, float, float] | None = None,
    velocity_scale: float = 0.0,
    center_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_center_z_env_local: float | None = None,
    max_center_z_env_local: float | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> None:
    """Place PCB root so the push-face center matches the jaw midpoint (kinematic grasp contact).

    Uses the same geometry as :func:`pcb_trailing_short_edge_center_w`. Solving
    ``trailing = jaw_mid`` gives ``pcb_center = jaw_mid + sign * half_length * body+X_world``
    where ``sign = sign(dot(body+X, push_axis))``.

    ``center_offset_body_m`` nudges the root in **PCB body** axes (tune if the USD finger origins
    sit on carriage housing so the analytic mid misses the actual pad gap).

    Vertical alignment: PCB centre Z is set to the jaw midpoint Z (top/bottom pinch) so the board
    stays kinematically attached to the fingers.

    .. warning::
        Do **not** pass a large ``min_center_z_env_local`` during insert-phase reset.
        Values such as ``_PCB_CENTER_Z_ENV`` (+70 mm above the conveyor) raise the PCB centre
        above the jaw midpoint, breaking the grasp and leaving the board overlapping the guide
        rails where PhysX friction pins it in place.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    robot.update(0.0)
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)[env_ids]
    n = len(env_ids)
    dtype = mid.dtype
    device = env.device
    if rot_wxyz is not None:
        q = torch.tensor(rot_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    elif hasattr(env, "_sampled_pcb_quat"):
        q = env._sampled_pcb_quat[env_ids].to(device=device, dtype=dtype)
    else:
        raise RuntimeError(
            "snap_pcb_root_to_short_edge_grasp requires rot_wxyz or a prior "
            "reset_from_grasp_states call that sets env._sampled_pcb_quat."
        )
    local_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    x_w = math_utils.quat_apply(q, local_x)
    push = torch.tensor(_DEFAULT_PUSH_AXIS_WORLD, device=device, dtype=dtype)
    push = push / torch.norm(push).clamp_min(1e-9)
    sign = torch.sign(torch.sum(x_w * push.unsqueeze(0).expand(n, -1), dim=-1))
    sign = torch.where(torch.abs(sign) < 1e-6, torch.ones_like(sign), sign)
    ob = torch.tensor(center_offset_body_m, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    off_w = math_utils.quat_apply(q, ob)
    center_w = mid + sign.unsqueeze(-1) * float(half_length_m) * x_w + off_w
    center_w[:, 2] = mid[:, 2]
    origins_z = env.scene.env_origins[env_ids, 2]
    if min_center_z_env_local is not None:
        min_cz = origins_z + float(min_center_z_env_local)
        center_w[:, 2] = torch.maximum(center_w[:, 2], min_cz)
    if max_center_z_env_local is not None:
        max_cz = origins_z + float(max_center_z_env_local)
        center_w[:, 2] = torch.minimum(center_w[:, 2], max_cz)
    root_pose = torch.cat([center_w, q], dim=-1)

    default_root_state = pcb.data.default_root_state[env_ids].clone()
    root_vel = default_root_state[:, 7:13] * velocity_scale
    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    pcb.update(0.0)

    _store_insert_progress_baselines(env, env_ids, pcb_cfg, half_length_m)


def _store_insert_progress_baselines(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
) -> None:
    """Cache per-episode Y-progress baselines after the PCB root pose is written."""
    pcb = env.scene[pcb_cfg.name]
    device = env.device
    dtype = pcb.data.root_pos_w.dtype
    center_env = pcb.data.root_pos_w[env_ids, :3] - env.scene.env_origins[env_ids, :3]
    push = torch.tensor(_DEFAULT_PUSH_AXIS_WORLD, device=device, dtype=dtype)
    push = push / torch.norm(push).clamp_min(1e-9)
    if not hasattr(env, "_insert_start_center_proj"):
        env._insert_start_center_proj = torch.zeros(env.num_envs, device=device, dtype=dtype)
    if not hasattr(env, "_insert_start_center_z"):
        env._insert_start_center_z = torch.zeros(env.num_envs, device=device, dtype=dtype)
    if not hasattr(env, "_insert_start_center_env"):
        env._insert_start_center_env = torch.zeros(env.num_envs, 3, device=device, dtype=dtype)
    if not hasattr(env, "_insert_start_lead_proj"):
        env._insert_start_lead_proj = torch.zeros(env.num_envs, device=device, dtype=dtype)
    if not hasattr(env, "_insert_start_lead_z"):
        env._insert_start_lead_z = torch.zeros(env.num_envs, device=device, dtype=dtype)
    env._insert_start_center_env[env_ids] = center_env
    env._insert_start_center_proj[env_ids] = torch.sum(center_env * push.unsqueeze(0), dim=-1)
    env._insert_start_center_z[env_ids] = center_env[:, 2]
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)[env_ids]
    lead_env = lead_w - env.scene.env_origins[env_ids, :3]
    env._insert_start_lead_proj[env_ids] = torch.sum(lead_env * push.unsqueeze(0), dim=-1)
    env._insert_start_lead_z[env_ids] = lead_env[:, 2]

    if not hasattr(env, "_insert_milestone_slot_mouth"):
        env._insert_milestone_slot_mouth = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    env._insert_milestone_slot_mouth[env_ids] = False

    if hasattr(env, "_insert_rail_milestone_mask"):
        env._insert_rail_milestone_mask[env_ids] = False

    if hasattr(env, "_insert_backward_step_count"):
        env._insert_backward_step_count[env_ids] = 0.0


def reset_pcb_on_guide_rails(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    pos_env_local: tuple[float, float, float],
    rot_wxyz: tuple[float, float, float, float],
    velocity_scale: float = 0.0,
) -> None:
    """Place PCB root at the scene-configured rail pose (env-local pos + env_origins → world).

    Use the same ``pos`` / ``rot`` as ``RigidObjectCfg.init_state`` so the rigid body default and
    reset stay consistent — PCB rests on the kinematic rails instead of floating at the gripper.
    """
    reset_pcb_on_guide_rails_randomized(
        env,
        env_ids,
        pcb_cfg,
        pos_env_local,
        rot_wxyz,
        pos_offset_ranges={},
        yaw_offset_range=(0.0, 0.0),
        velocity_scale=velocity_scale,
    )


def reset_pcb_on_guide_rails_randomized(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    pos_env_local: tuple[float, float, float],
    rot_wxyz: tuple[float, float, float, float],
    pos_offset_ranges: dict[str, tuple[float, float]] | None = None,
    yaw_offset_range: tuple[float, float] = (0.0, 0.0),
    velocity_scale: float = 0.0,
) -> None:
    """Place PCB root at nominal rail pose plus uniform XY offsets and world-Z yaw (domain rand).

    ``pos_offset_ranges`` keys ``"x"`` / ``"y"`` give per-env uniform offsets in env-local axes.
    ``yaw_offset_range`` is a uniform world +Z rotation (radians) applied on top of ``rot_wxyz``.
    """
    pcb = env.scene[pcb_cfg.name]
    n = len(env_ids)
    device = env.device
    dtype = pcb.data.root_pos_w.dtype
    ranges = pos_offset_ranges or {}

    pl = torch.tensor(pos_env_local, device=device, dtype=dtype).unsqueeze(0).expand(n, -1).clone()
    lo_x, hi_x = ranges.get("x", (0.0, 0.0))
    lo_y, hi_y = ranges.get("y", (0.0, 0.0))
    if abs(lo_x) > 1e-12 or abs(hi_x) > 1e-12:
        pl[:, 0] += torch.empty(n, device=device, dtype=dtype).uniform_(float(lo_x), float(hi_x))
    if abs(lo_y) > 1e-12 or abs(hi_y) > 1e-12:
        pl[:, 1] += torch.empty(n, device=device, dtype=dtype).uniform_(float(lo_y), float(hi_y))

    target_pos = pl + env.scene.env_origins[env_ids]

    q_base = torch.tensor(rot_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    yaw_lo, yaw_hi = yaw_offset_range
    if abs(yaw_lo) > 1e-12 or abs(yaw_hi) > 1e-12:
        yaw = torch.empty(n, device=device, dtype=dtype).uniform_(float(yaw_lo), float(yaw_hi))
        z_axis = torch.tensor((0.0, 0.0, 1.0), device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
        q_yaw = math_utils.quat_from_angle_axis(yaw, z_axis)
        q = math_utils.quat_mul(q_yaw, q_base)
    else:
        q = q_base

    root_pose = torch.cat([target_pos, q], dim=-1)

    default_root_state = pcb.data.default_root_state[env_ids].clone()
    root_vel = default_root_state[:, 7:13] * velocity_scale

    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def pcb_moving_backward_termination(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    backward_vel_threshold: float = -0.03,
    min_steps: int = 20,
) -> torch.Tensor:
    """Terminate when the PCB sustains **−Y (backward) linear velocity** for too long.

    Counts consecutive env steps where ``v_y < backward_vel_threshold`` (negative = moving away from
    the slot). Resets the counter whenever the PCB moves forward or the episode restarts.
    Fires after ``min_steps`` consecutive backward-moving steps to avoid cutting on transient bounces.
    """
    global _PCB_BACKWARD_COUNT
    pcb = env.scene[pcb_cfg.name]
    v_y = pcb.data.root_lin_vel_w[:, 1]          # world +Y is toward the slot
    moving_backward = v_y < float(backward_vel_threshold)
    first_step = env.episode_length_buf == 1
    if (
        _PCB_BACKWARD_COUNT is None
        or _PCB_BACKWARD_COUNT.shape[0] != env.num_envs
        or _PCB_BACKWARD_COUNT.device != env.device
    ):
        _PCB_BACKWARD_COUNT = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    _PCB_BACKWARD_COUNT = torch.where(
        first_step | (~moving_backward),
        torch.zeros_like(_PCB_BACKWARD_COUNT),
        _PCB_BACKWARD_COUNT + 1,
    )
    return _PCB_BACKWARD_COUNT >= int(min_steps)


def pcb_detached_from_gripper(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_edge_dist_m: float = 0.040,
    max_finger_dist_m: float = 0.030,
    max_edge_along_m: float | None = None,
    max_edge_in_plane_m: float | None = None,
    min_straddle_sep_m: float = 0.0,
    pcb_half_thickness_m: float = 0.00025,
    width_weight: float = 3.0,
    min_height_env: float | None = None,
    min_episode_steps: int = 2,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True when the PCB is no longer kinematically held by the gripper.

    Uses the same trailing-edge grasp geometry as :func:`grasp_edge_center_achieved`, but with
    **looser** detach thresholds so minor insertion wobble does not false-trigger while a real
    slip / drop does.

    Detach is declared when any of the following hold (after ``min_episode_steps``):
    * jaw midpoint is too far from the trailing short-edge centre;
    * either jaw tip is too far from its trailing-edge grasp target;
    * the board is no longer straddled between the jaws — sign(w_left)*sign(w_right) must differ,
      AND |w_left - w_right| >= min_straddle_sep_m (set 0.0 to use sign check only, which is
      required when PCB_Z < 1 mm since the max physical separation equals the board thickness);
    * optional: PCB root height (env-local Z) falls below ``min_height_env``.
    """
    along, _, _, in_plane, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    if max_edge_along_m is not None or max_edge_in_plane_m is not None:
        lost_edge = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if max_edge_along_m is not None:
            lost_edge = lost_edge | (torch.abs(along) > float(max_edge_along_m))
        if max_edge_in_plane_m is not None:
            lost_edge = lost_edge | (in_plane > float(max_edge_in_plane_m))
    else:
        lost_edge = edge_dist > float(max_edge_dist_m)
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )

    lost_fingers = torch.maximum(geom["dist_l"], geom["dist_r"]) > float(max_finger_dist_m)
    straddled = (w_left * w_right < 0.0) & (torch.abs(w_left - w_right) >= float(min_straddle_sep_m))
    lost_straddle = ~straddled
    detached = lost_edge | lost_fingers | lost_straddle

    if min_height_env is not None:
        pcb = env.scene[pcb_cfg.name]
        h = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        detached = detached | (h < float(min_height_env))

    if min_episode_steps > 0:
        ready = env.episode_length_buf > min_episode_steps
        detached = detached & ready
    return detached


def pcb_extreme_drift_from_gripper(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_extra_sep_m: float = 0.045,
    max_finger_dist_m: float = 0.070,
    max_edge_dist_m: float = 0.090,
    max_perp_drift_m: float = 0.045,
    max_vertical_sep_m: float = 0.040,
    max_lin_speed_m_s: float = 1.2,
    pcb_half_thickness_m: float = 0.00025,
    width_weight: float = 3.0,
    min_episode_steps: int = 20,
    check_perp_drift: bool = True,
    check_vertical_sep: bool = True,
    check_flying: bool = True,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True when the PCB has clearly escaped the gripper (large gap, slide, or tumbling).

    Complements :func:`pcb_detached_from_gripper` with thresholds aimed at **extreme** failures
    (board on the table beside the jaws, flying after contact explosions).  Fires on any of:

    * jaw-mid ↔ PCB-root separation exceeds ``half_length_m + max_extra_sep_m``;
    * either jaw tip is farther than ``max_finger_dist_m`` from its grasp target;
    * jaw-mid ↔ trailing-edge centre gap exceeds ``max_edge_dist_m``;
    * optional: PCB root drift perpendicular to the push axis exceeds ``max_perp_drift_m``;
    * optional: vertical separation between jaw mid and PCB root exceeds ``max_vertical_sep_m``;
    * optional: PCB root linear speed exceeds ``max_lin_speed_m_s``.

    The optional checks are prone to false positives during normal +Y insertion pushes
    (arm kinematics, brief contact impulses).  Disable them via ``check_*`` flags and rely on
    :func:`pcb_detached_from_gripper` for moderate slip.
    """
    pcb = env.scene[pcb_cfg.name]
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)
    mid_root_dist = torch.norm(pcb.data.root_pos_w - mid, dim=-1)
    extreme_sep = mid_root_dist > (float(half_length_m) + float(max_extra_sep_m))

    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        tip_offset_m,
        wrist_body_cfg,
    )
    finger_gap = torch.maximum(geom["dist_l"], geom["dist_r"]) > float(max_finger_dist_m)

    *_, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    edge_lost = edge_dist > float(max_edge_dist_m)

    perp_drift = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if check_perp_drift and hasattr(env, "_insert_start_center_env"):
        pos_env = pcb.data.root_pos_w - env.scene.env_origins[:, :3]
        delta = pos_env - env._insert_start_center_env
        push = torch.tensor(_DEFAULT_PUSH_AXIS_WORLD, device=env.device, dtype=pos_env.dtype)
        push = push / torch.norm(push).clamp_min(1e-9)
        along = torch.sum(delta * push.unsqueeze(0), dim=-1, keepdim=True) * push.unsqueeze(0)
        perp = delta - along
        perp_drift = torch.norm(perp, dim=-1) > float(max_perp_drift_m)

    vertical_sep = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if check_vertical_sep:
        vertical_sep = torch.abs(mid[:, 2] - pcb.data.root_pos_w[:, 2]) > float(max_vertical_sep_m)

    flying = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if check_flying:
        flying = torch.norm(pcb.data.root_lin_vel_w, dim=-1) > float(max_lin_speed_m_s)

    drifted = extreme_sep | finger_gap | edge_lost | perp_drift | vertical_sep | flying
    if min_episode_steps > 0:
        ready = env.episode_length_buf > min_episode_steps
        drifted = drifted & ready
    return drifted


def pcb_dropped_from_gripper(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    expected_center_distance: float = 0.12,
    expected_offset_world: tuple[float, float, float] | None = None,
    distance_tolerance: float = 0.08,
    min_height: float = 0.02,
    check_grasp_geometry: bool = True,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    half_length_m: float = 0.12,
    max_edge_dist_m: float = 0.040,
    max_finger_dist_m: float = 0.030,
    min_height_env: float | None = None,
) -> torch.Tensor:
    """Return True if PCB is considered dropped from the gripper.

    When ``check_grasp_geometry`` is True (Insert phase), delegates to
    :func:`pcb_detached_from_gripper` which measures trailing-edge / finger-target separation
    instead of root-to-midpoint distance (the latter stays ~``half_length_m`` even after a slip).

    Set ``check_grasp_geometry=False`` when episodes start with the PCB on guide rails (not in-hand).
    """
    if check_grasp_geometry:
        return pcb_detached_from_gripper(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m=half_length_m,
            max_edge_dist_m=max_edge_dist_m,
            max_finger_dist_m=max_finger_dist_m,
            min_height_env=min_height_env,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
        )

    pcb = env.scene[pcb_cfg.name]
    pcb_height = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return pcb_height < min_height


# ---------------------------------------------------------------------------
# Policy-chaining reset: sample terminal states from Phase-1 buffer
# (Sequential Dexterity, Chen et al. CoRL 2023, §3.2)
# ---------------------------------------------------------------------------

# Module-level cache — loaded once on first call.
_GRASP_STATE_BUFFER: dict | None = None
_GRASP_BUFFER_PATH: str | None = None


def _load_grasp_state_buffer(path: str) -> dict:
    """Load (or re-use cached) grasp terminal state .npz file."""
    global _GRASP_STATE_BUFFER, _GRASP_BUFFER_PATH
    if _GRASP_STATE_BUFFER is not None and _GRASP_BUFFER_PATH == path:
        return _GRASP_STATE_BUFFER
    data = np.load(path, allow_pickle=True)
    _GRASP_STATE_BUFFER = {
        "joint_pos":   torch.from_numpy(data["joint_pos"].astype(np.float32)),
        "pcb_pos_env": torch.from_numpy(data["pcb_pos_env"].astype(np.float32)),
        "pcb_quat":    torch.from_numpy(data["pcb_quat"].astype(np.float32)),
        "joint_names": list(data["joint_names"]),
    }
    _GRASP_BUFFER_PATH = path
    n = _GRASP_STATE_BUFFER["joint_pos"].shape[0]
    print(f"[GraspStateBuffer] Loaded {n} terminal states from '{path}'")
    return _GRASP_STATE_BUFFER


def hold_gripper_closed(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_name: str = "left_carriage_joint",
    closed_target_m: float = 0.00025,
    match_sim_state: bool = False,
    store_target: bool = False,
) -> None:
    """Command the parallel gripper closed when it is excluded from the action space.

    Insert training only actuates the arm.  Without this, ``joint_pos_target`` for the carriage
    joint stays at its init value (0) while the reset pose is closed — the implicit PD then
    drives the jaws open and the PCB slips.

    At reset, pass ``match_sim_state=True`` and ``store_target=True`` so the PD target matches
    the buffer pinch pose but is tightened to ``closed_target_m`` when the buffer row is looser.
    Per-env targets are cached on ``env._gripper_hold_target_m`` for re-application during
    the episode (``RelativeJointPositionActionWithGripperHold`` calls this after each arm
    command; contact forces can otherwise drift the implicit target open).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_name)
    jid = joint_ids[0]
    closed_val = float(closed_target_m)
    if match_sim_state:
        target = robot.data.joint_pos[env_ids, jid].clone()
        # Smaller carriage joint value = tighter pinch; clamp looser buffer poses closed.
        target = torch.minimum(
            target,
            torch.full_like(target, closed_val),
        )
    elif hasattr(env, "_gripper_hold_target_m"):
        target = env._gripper_hold_target_m[env_ids].clone()
    else:
        target = torch.full(
            (len(env_ids),),
            closed_val,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )

    if store_target or not hasattr(env, "_gripper_hold_target_m"):
        if not hasattr(env, "_gripper_hold_target_m"):
            env._gripper_hold_target_m = torch.full(
                (env.num_envs,),
                closed_val,
                device=env.device,
                dtype=robot.data.joint_pos.dtype,
            )
        env._gripper_hold_target_m[env_ids] = target

    target = target.unsqueeze(-1)
    zeros = torch.zeros_like(target)
    robot.set_joint_position_target(target, joint_ids=[jid], env_ids=env_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=[jid], env_ids=env_ids)


class RelativeJointPositionActionWithGripperHold(joint_actions.RelativeJointPositionAction):
    """Relative arm deltas, then re-command the gripper closed after each action write.

    Isaac Lab applies interval events after the physics substep loop; arm ``apply_action``
    runs inside that loop and can leave the carriage joint target stale/open.  Re-applying
    ``hold_gripper_closed`` here keeps the PD target closed on every physics substep.
    """

    cfg: "RelativeJointPositionActionWithGripperHoldCfg"

    def apply_actions(self) -> None:
        super().apply_actions()
        hold_gripper_closed(
            self._env,
            None,
            self.cfg.gripper_hold_asset_cfg,
            joint_name=self.cfg.gripper_joint_name,
            closed_target_m=self.cfg.gripper_closed_target_m,
        )


@configclass
class RelativeJointPositionActionWithGripperHoldCfg(RelativeJointPositionActionCfg):
    """Arm-only relative deltas with post-action gripper hold (Slide / Insert phases)."""

    class_type: type[ActionTerm] = RelativeJointPositionActionWithGripperHold
    gripper_hold_asset_cfg: SceneEntityCfg = MISSING
    gripper_joint_name: str = "left_carriage_joint"
    gripper_closed_target_m: float = 0.00025


def _sample_grasp_buffer_indices(
    pcb_pos_env: torch.Tensor,
    n_samples: int,
    rail_center_z: float,
    max_z_delta_m: float,
) -> torch.Tensor:
    """Sample buffer rows whose PCB centre Z is near the guide-rail contact height."""
    z = pcb_pos_env[:, 2]
    valid = torch.nonzero(torch.abs(z - float(rail_center_z)) <= float(max_z_delta_m), as_tuple=False).view(-1)
    if valid.numel() == 0:
        return torch.randint(0, pcb_pos_env.shape[0], (n_samples,), device="cpu")
    pick = torch.randint(0, valid.numel(), (n_samples,), device="cpu")
    return valid[pick]


def reset_from_grasp_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    grasp_states_path: str,
    velocity_scale: float = 0.0,
    gripper_joint_name: str = "left_carriage_joint",
    gripper_closed_target_m: float = 0.00025,
    rail_center_z_env: float | None = None,
    max_rail_z_delta_m: float = 0.015,
) -> None:
    """Reset **robot joints only** by sampling from the saved grasp terminal-state buffer.

    Implements the Phase-2 initial-state distribution from Sequential Dexterity
    (Chen et al. CoRL 2023): the terminal state distribution of Phase 1 (Grasp)
    becomes the initial state distribution of Phase 2 (Insert).

    Stores ``env._grasp_buffer_idx`` so :func:`reset_pcb_from_grasp_states` can load
    the matching ``pcb_pos_env`` / ``pcb_quat`` from the same buffer row.

    Parameters
    ----------
    grasp_states_path:
        Path to the .npz produced by ``scripts/collect_grasp_states.py``.
    """
    buf = _load_grasp_state_buffer(grasp_states_path)
    n_buf = buf["joint_pos"].shape[0]
    n_reset = len(env_ids)
    device = env.device
    dtype = torch.float32

    # Prefer grasp rows with PCB Z near the rail contact height (reduces airborne / shear slip).
    if rail_center_z_env is not None:
        idx = _sample_grasp_buffer_indices(
            buf["pcb_pos_env"],
            n_reset,
            float(rail_center_z_env),
            float(max_rail_z_delta_m),
        )
    else:
        idx = torch.randint(0, n_buf, (n_reset,), device="cpu")
    if not hasattr(env, "_grasp_buffer_idx"):
        env._grasp_buffer_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._grasp_buffer_idx[env_ids] = idx.to(device=env.device)

    # ── Robot joints ──────────────────────────────────────────────────────────
    robot = env.scene[asset_cfg.name]
    joint_pos_buf = buf["joint_pos"][idx].to(device=device, dtype=dtype)   # (n_reset, n_joints)

    buf_names: list[str] = buf["joint_names"]
    robot_names: list[str] = robot.joint_names
    name_to_buf_col = {n: i for i, n in enumerate(buf_names)}

    # Map buffer columns → robot joint indices (buffer may be a subset of joints).
    joint_pos_new = robot.data.default_joint_pos[env_ids].clone()
    for robot_col, robot_name in enumerate(robot_names):
        if robot_name in name_to_buf_col:
            buf_col = name_to_buf_col[robot_name]
            joint_pos_new[:, robot_col] = joint_pos_buf[:, buf_col]

    # Clamp to soft limits.
    lim = robot.data.soft_joint_pos_limits[env_ids]
    joint_pos_new = joint_pos_new.clamp(lim[..., 0], lim[..., 1])

    joint_vel_new = torch.zeros_like(joint_pos_new) * velocity_scale
    robot.write_joint_state_to_sim(joint_pos_new, joint_vel_new, env_ids=env_ids)
    robot.update(0.0)

    # Insert obs: joint_pos relative to this reset pose (not HOME default).
    if not hasattr(env, "_insert_reset_joint_pos"):
        env._insert_reset_joint_pos = torch.zeros(
            (env.num_envs, robot.num_joints), device=device, dtype=dtype
        )
    env._insert_reset_joint_pos[env_ids] = joint_pos_new.clone()

    # Match PD target to the buffer pinch pose; tighten if looser than ``gripper_closed_target_m``.
    hold_gripper_closed(
        env,
        env_ids,
        asset_cfg,
        joint_name=gripper_joint_name,
        closed_target_m=gripper_closed_target_m,
        match_sim_state=True,
        store_target=True,
    )

    # Legacy: snap path reads quat from here; buffer PCB reset uses the same buffer row.
    pcb_quat_buf = buf["pcb_quat"][idx].to(device=device, dtype=dtype)
    if not hasattr(env, "_sampled_pcb_quat"):
        env._sampled_pcb_quat = torch.zeros((env.num_envs, 4), device=device, dtype=dtype)
    env._sampled_pcb_quat[env_ids] = pcb_quat_buf


def reset_pcb_from_grasp_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    grasp_states_path: str,
    half_length_m: float,
    velocity_scale: float = 0.0,
    rail_center_z_env: float | None = None,
    snap_z_to_rail: bool = False,
    snap_z_max_delta_m: float = 0.002,
) -> None:
    """Place the PCB at the Phase-1 terminal pose stored in the grasp buffer.

    Must run **after** :func:`reset_from_grasp_states` so ``env._grasp_buffer_idx``
    points to the same buffer row as the robot joints.

    When ``snap_z_to_rail`` is True, centre Z is set to ``rail_center_z_env`` only when the
    buffer Z is already within ``snap_z_max_delta_m`` of that height.  Larger gaps keep buffer
    Z to avoid a vertical teleport that overlaps rails/fingers and causes depenetration bounce.
    """
    if not hasattr(env, "_grasp_buffer_idx"):
        raise RuntimeError(
            "reset_pcb_from_grasp_states requires a prior reset_from_grasp_states call "
            "that sets env._grasp_buffer_idx."
        )
    buf = _load_grasp_state_buffer(grasp_states_path)
    pcb = env.scene[pcb_cfg.name]
    device = env.device
    dtype = torch.float32

    idx = env._grasp_buffer_idx[env_ids].cpu()
    pos_env = buf["pcb_pos_env"][idx].to(device=device, dtype=dtype)
    if snap_z_to_rail and rail_center_z_env is not None:
        rail_z = float(rail_center_z_env)
        z_buf = pos_env[:, 2]
        close = torch.abs(z_buf - rail_z) <= float(snap_z_max_delta_m)
        pos_env[:, 2] = torch.where(
            close,
            torch.full_like(z_buf, rail_z),
            z_buf,
        )
    quat = buf["pcb_quat"][idx].to(device=device, dtype=dtype)
    origins = env.scene.env_origins[env_ids, :3]
    pos_w = pos_env + origins
    root_pose = torch.cat([pos_w, quat], dim=-1)

    default_root_state = pcb.data.default_root_state[env_ids].clone()
    root_vel = default_root_state[:, 7:13] * velocity_scale
    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    pcb.update(0.0)

    _store_insert_progress_baselines(env, env_ids, pcb_cfg, half_length_m)


# ---------------------------------------------------------------------------
# SDF-based dense insertion reward (IndustReal, Tang et al. RSS 2023, §3.2)
# ---------------------------------------------------------------------------

def pcb_to_target_error_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_xyz_env: tuple[float, float, float],
    scale_xyz_m: tuple[float, float, float] = (0.05, 0.40, 0.05),
    half_length_m: float | None = None,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Obs: env-local error vector from a PCB reference point to the target, scaled to ~[-1, 1].

    When ``half_length_m`` is set, the reference is the **leading short-edge face centre**
    (slot-side edge); otherwise the rigid-body root is used.  Components are
    ``(dx, dy, dz) = target - pcb_pos`` in the env-local frame, each divided by a per-axis scale.
    """
    if half_length_m is not None:
        pcb_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    else:
        pcb = env.scene[pcb_cfg.name]
        pcb_env = pcb.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    tgt = torch.tensor(target_xyz_env, device=env.device, dtype=pcb_env.dtype).unsqueeze(0)
    err = tgt - pcb_env
    sc = torch.tensor(scale_xyz_m, device=env.device, dtype=pcb_env.dtype).unsqueeze(0)
    return err / (sc + 1e-6)


def pcb_insertion_orientation_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    align_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Obs: two PCB insertion-orientation cosines, each in ``[-1, 1]``.

    * ``cos(long axis +X, insertion axis +Y)`` — 1.0 when the board points lengthwise into
      the slot (correct orientation for entry).
    * ``cos(thickness axis +Z, world up)`` — 1.0 when the board is lying flat.

    These let the policy detect and correct a mis-oriented grasp before reaching the slot,
    which is essential when the grasp terminal states have varied PCB orientations.
    """
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    a = torch.tensor(align_axis_world, device=env.device, dtype=x_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    up = torch.tensor(world_up, device=env.device, dtype=z_w.dtype)
    up = up / torch.norm(up).clamp_min(1e-9)
    cos_long = torch.sum(x_w * a.unsqueeze(0), dim=-1, keepdim=True)
    cos_flat = torch.sum(z_w * up.unsqueeze(0), dim=-1, keepdim=True)
    return torch.cat([cos_long, cos_flat], dim=-1)


def pcb_y_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    initial_y_env: float,
    target_y_env: float,
) -> torch.Tensor:
    """Dense reward for cumulative **world +Y** progress (PCB root, not long axis).

    .. note::
        This measures displacement along **world +Y**, which equals the PCB long-axis
        direction only at the nominal spawn orientation (body +X || world +Y).
        For insert phase with varied grasp orientations, prefer
        :func:`pcb_push_axis_progress_reward` instead.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_y = pcb.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    total = float(target_y_env) - float(initial_y_env) + 1e-9
    progress = (pcb_y - float(initial_y_env)) / total
    return progress.clamp(0.0, 1.0)


def pcb_push_axis_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_proj_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    half_length_m: float | None = None,
    use_episode_start: bool = True,
    fallback_initial_proj_env: float | None = None,
) -> torch.Tensor:
    """Fraction of travel along the **slot insertion axis** (default world +Y).

    **Not** the PCB body long axis.  The magazine opens toward +Y; success is defined
    when the PCB *centre* reaches the magazine centre along +Y.  PCB long-axis alignment
    is handled separately by :func:`pcb_long_axis_parallel_to_push_reward`.

    By default (``half_length_m is None``) the PCB **root centre** is projected onto
    ``axis_world`` from the per-episode start pose (``env._insert_start_center_proj``,
    set at snap reset) to ``target_proj_env`` (typically ``_MAG_CENTER_Y_ENV``).

    Passing ``half_length_m`` switches to the leading short-edge point and
    ``env._insert_start_lead_proj`` instead (for depth-style metrics only).

    Range: [0, 1]. Use a positive weight.
    """
    pcb = env.scene[pcb_cfg.name]
    if half_length_m is not None:
        pos_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
        start_attr = "_insert_start_lead_proj"
    else:
        pos_w = pcb.data.root_pos_w
        start_attr = "_insert_start_center_proj"

    pos_env = pos_w - env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=env.device, dtype=pos_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(pos_env * a.unsqueeze(0), dim=-1)

    if use_episode_start and hasattr(env, start_attr):
        start_proj = getattr(env, start_attr)
    elif fallback_initial_proj_env is not None:
        start_proj = torch.full_like(proj, float(fallback_initial_proj_env))
    else:
        start_proj = proj.detach()

    total = float(target_proj_env) - start_proj
    # Require positive travel distance toward the target; otherwise return zero progress.
    valid = total > 1e-6
    total_safe = torch.where(valid, total, torch.ones_like(total))
    progress = (proj - start_proj) / total_safe
    progress = torch.where(valid, progress, torch.zeros_like(progress))
    return torch.nan_to_num(progress.clamp(0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


_INSERT_PREV_CENTER_PROJ: torch.Tensor | None = None
_INSERT_PREV_LEAD_PROJ: torch.Tensor | None = None


def pcb_push_axis_approach_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_proj_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.010,
) -> torch.Tensor:
    """Per-step reward for PCB centre motion toward ``target_proj_env`` along the push axis.

    Credits ``clamp(proj_t - proj_{t-1}, 0, max_step)`` so the policy gets signal for the
    **first** millimetres of +Y push (Grasp-style Δdistance shaping).
    """
    global _INSERT_PREV_CENTER_PROJ

    pcb = env.scene[pcb_cfg.name]
    pos_env = pcb.data.root_pos_w - env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=env.device, dtype=pos_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(pos_env * a.unsqueeze(0), dim=-1)

    if (
        _INSERT_PREV_CENTER_PROJ is None
        or _INSERT_PREV_CENTER_PROJ.shape[0] != proj.shape[0]
        or _INSERT_PREV_CENTER_PROJ.device != proj.device
    ):
        _INSERT_PREV_CENTER_PROJ = proj.clone()
        return torch.zeros_like(proj)

    first_step = env.episode_length_buf == 1
    _INSERT_PREV_CENTER_PROJ = torch.where(first_step, proj, _INSERT_PREV_CENTER_PROJ)
    delta = (proj - _INSERT_PREV_CENTER_PROJ).clamp(min=0.0, max=float(max_step_m))
    _INSERT_PREV_CENTER_PROJ = proj.clone()
    return delta / (float(max_step_m) + 1e-9)


def pcb_push_axis_approach_progress_gated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_proj_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.010,
    max_off_axis_speed_m_s: float = 0.008,
) -> torch.Tensor:
    """Like :func:`pcb_push_axis_approach_progress`, zero when PCB skids or lifts (X/Z speed)."""
    step = pcb_push_axis_approach_progress(
        env, pcb_cfg, target_proj_env, axis_world, max_step_m
    )
    pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_off_axis_speed_m_s)
    return torch.where(pure, step, torch.zeros_like(step))


def pcb_leading_edge_push_axis_approach_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
) -> torch.Tensor:
    """Per-step +Y progress of the leading short-edge centre along the push axis.

    Credits ``clamp(proj_t - proj_{t-1}, 0, max_step) / max_step`` so the policy gets
    immediate signal for millimetre-scale rail slide (unlike cumulative state progress).
    """
    global _INSERT_PREV_LEAD_PROJ

    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)

    if (
        _INSERT_PREV_LEAD_PROJ is None
        or _INSERT_PREV_LEAD_PROJ.shape[0] != proj.shape[0]
        or _INSERT_PREV_LEAD_PROJ.device != proj.device
    ):
        _INSERT_PREV_LEAD_PROJ = proj.clone()
        return torch.zeros_like(proj)

    first_step = env.episode_length_buf == 1
    _INSERT_PREV_LEAD_PROJ = torch.where(first_step, proj, _INSERT_PREV_LEAD_PROJ)
    delta = (proj - _INSERT_PREV_LEAD_PROJ).clamp(min=0.0, max=float(max_step_m))
    _INSERT_PREV_LEAD_PROJ = proj.clone()
    return delta / (float(max_step_m) + 1e-9)


def _insert_straddle_gate_mask(
    env: ManagerBasedRLEnv,
    min_straddle_quality: float,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.010,
) -> torch.Tensor:
    """``1.0`` where :func:`pcb_between_gripper_fingers` exceeds ``min_straddle_quality``, else ``0.0``."""
    q = pcb_between_gripper_fingers(
        env,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
    )
    return (q > float(min_straddle_quality)).to(dtype=q.dtype)


def _apply_insert_straddle_gate(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    min_straddle_quality: float,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg | None,
    right_finger_cfg: SceneEntityCfg | None,
    half_length_m: float,
    pcb_half_thickness_m: float,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.010,
) -> torch.Tensor:
    if float(min_straddle_quality) <= 0.0 or left_finger_cfg is None or right_finger_cfg is None:
        return reward
    gate = _insert_straddle_gate_mask(
        env,
        min_straddle_quality,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
    )
    return reward * gate


def pcb_leading_edge_push_axis_approach_progress_gated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    max_off_axis_speed_m_s: float = 0.04,
    min_straddle_quality: float = 0.0,
    proximity_sigma_m: float = 0.050,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    pcb_half_thickness_m: float = 0.00075,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
) -> torch.Tensor:
    """Like :func:`pcb_leading_edge_push_axis_approach_progress`, zero when PCB skids or lifts."""
    step = pcb_leading_edge_push_axis_approach_progress(
        env, pcb_cfg, half_length_m, axis_world, max_step_m
    )
    pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_off_axis_speed_m_s)
    out = torch.where(pure, step, torch.zeros_like(step))
    return _apply_insert_straddle_gate(
        env,
        out,
        min_straddle_quality,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
    )


def pcb_push_axis_progress_reward_gated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_proj_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    half_length_m: float | None = None,
    use_episode_start: bool = True,
    fallback_initial_proj_env: float | None = None,
    max_off_axis_speed_m_s: float = 0.008,
    min_straddle_quality: float = 0.0,
    proximity_sigma_m: float = 0.050,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    pcb_half_thickness_m: float = 0.00075,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
) -> torch.Tensor:
    """Like :func:`pcb_push_axis_progress_reward`, zero while PCB moves in X or Z."""
    progress = pcb_push_axis_progress_reward(
        env,
        pcb_cfg,
        target_proj_env,
        axis_world,
        half_length_m,
        use_episode_start,
        fallback_initial_proj_env,
    )
    pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_off_axis_speed_m_s)
    out = torch.where(pure, progress, torch.zeros_like(progress))
    if half_length_m is None:
        return out
    return _apply_insert_straddle_gate(
        env,
        out,
        min_straddle_quality,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
    )


def pcb_rail_parallel_approach_progress_straddle_gated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    slot_mouth_y_env: float,
    rail_center_x_env: float,
    max_lateral_x_m: float = 0.030,
    min_flatness: float = 0.92,
    min_long_align: float = 0.85,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.010,
    min_straddle_quality: float = 0.0,
    proximity_sigma_m: float = 0.050,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    half_length_m: float = 0.12,
    pcb_half_thickness_m: float = 0.00075,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
) -> torch.Tensor:
    """Per-step +Y progress gated on lane-parallel pose and optional straddle quality."""
    step = pcb_rail_parallel_approach_progress(
        env,
        pcb_cfg,
        slot_mouth_y_env,
        rail_center_x_env,
        max_lateral_x_m,
        min_flatness,
        min_long_align,
        axis_world,
        max_step_m,
    )
    return _apply_insert_straddle_gate(
        env,
        step,
        min_straddle_quality,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
    )


def pcb_leading_edge_z_lift_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    max_lift_m: float = 0.007,
    max_penalty_excess_m: float = 0.02,
    use_episode_start: bool = True,
    reference_z_env: float | None = None,
) -> torch.Tensor:
    """Bounded penalty when the leading short-edge centre rises above the episode start height.

    ``excess = relu(lead_z - start_z - max_lift_m)`` clamped to ``max_penalty_excess_m``, then
    squared and normalised to ``[0, 1]``.  Avoids unbounded ``expm1`` blow-ups that destabilise
    value learning.  Pair with a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_z = lead_w[:, 2] - env.scene.env_origins[:, 2]
    if use_episode_start and hasattr(env, "_insert_start_lead_z"):
        ref_z = env._insert_start_lead_z
    elif reference_z_env is not None:
        ref_z = torch.full_like(lead_z, float(reference_z_env))
    else:
        ref_z = lead_z.detach()
    excess = torch.clamp(lead_z - ref_z - float(max_lift_m), min=0.0)
    cap = max(float(max_penalty_excess_m), 1e-9)
    excess = torch.clamp(excess, max=cap)
    return torch.square(excess / cap)


def pcb_leading_edge_z_lift_exponential_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    max_lift_m: float = 0.007,
    exponential_scale_m: float = 0.005,
    use_episode_start: bool = True,
    reference_z_env: float | None = None,
) -> torch.Tensor:
    """Deprecated — use :func:`pcb_leading_edge_z_lift_penalty` (bounded quadratic)."""
    return pcb_leading_edge_z_lift_penalty(
        env,
        pcb_cfg,
        half_length_m,
        max_lift_m=max_lift_m,
        max_penalty_excess_m=max(float(exponential_scale_m) * 4.0, 0.02),
        use_episode_start=use_episode_start,
        reference_z_env=reference_z_env,
    )


def pcb_root_linear_speed_excess_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    max_speed_m_s: float = 0.01,
) -> torch.Tensor:
    """Penalty when PCB root linear speed exceeds ``max_speed_m_s``.

    Returns ``relu(speed - max_speed_m_s)`` — pair with a **negative** weight to encourage
    slow, controlled sliding along the rails.
    """
    pcb = env.scene[pcb_cfg.name]
    speed = torch.norm(pcb.data.root_lin_vel_w[:, :3], dim=-1)
    return torch.clamp(speed - float(max_speed_m_s), min=0.0)


def pcb_root_velocity_xz_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    z_scale: float = 0.5,
) -> torch.Tensor:
    """Squared off-insertion-axis root velocity: ``v_x² + z_scale · v_z²``.

    Insertion is along world +Y.  Penalises lateral skidding (±X) and lift/drop (Z)
    in one term.  ``z_scale`` sets Z severity relative to X (default 0.5 reproduces
    separate weights of −1000 on X and −500 on Z).  Pair with a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    return torch.square(v[:, 0]) + float(z_scale) * torch.square(v[:, 2])


def pcb_root_velocity_x_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Deprecated — use :func:`pcb_root_velocity_xz_penalty`."""
    return pcb_root_velocity_xz_penalty(env, pcb_cfg, z_scale=0.0)


def pcb_root_velocity_z_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Deprecated — use :func:`pcb_root_velocity_xz_penalty`."""
    pcb = env.scene[pcb_cfg.name]
    return torch.square(pcb.data.root_lin_vel_w[:, 2])


def pcb_z_height_band_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    max_displacement_m: float = 0.005,
    use_episode_start: bool = True,
    reference_z_env: float | None = None,
) -> torch.Tensor:
    """Symmetric penalty when PCB Z drifts above **or** below the grasp height.

    Unlike :func:`pcb_height_below_reference` (grasp phase only punishes low Z and
    therefore *rewards lifting*), this keeps the board at the snapped grasp height.
    Reference Z defaults to ``env._insert_start_center_z`` from snap reset.
    """
    pcb = env.scene[pcb_cfg.name]
    z_env = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if use_episode_start and hasattr(env, "_insert_start_center_z"):
        ref_z = env._insert_start_center_z
    elif reference_z_env is not None:
        ref_z = torch.full_like(z_env, float(reference_z_env))
    else:
        ref_z = z_env.detach()
    dz = torch.abs(z_env - ref_z)
    return torch.clamp(dz - float(max_displacement_m), min=0.0)


def pcb_push_axis_velocity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_push_speed_m_s: float = 0.002,
) -> torch.Tensor:
    """Reward PCB root linear speed along the push axis (default world +Y).

    Returns ``v_push`` when ``v_push > min_push_speed_m_s``; otherwise 0.  No credit for
    −push, X, or Z motion.  Pair with :func:`pcb_push_axis_progress_reward` (state).
    Use a **positive** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    a = torch.tensor(axis_world, device=env.device, dtype=v.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    v_push = torch.sum(v * a.unsqueeze(0), dim=-1)
    moving = v_push > float(min_push_speed_m_s)
    return torch.where(moving, v_push, torch.zeros_like(v_push))


def pcb_gated_y_velocity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    max_off_axis_speed_m_s: float = 0.008,
    min_y_speed_m_s: float = 0.002,
) -> torch.Tensor:
    """Reward +Y linear velocity only when the PCB is **not** moving in X or Z.

    Prevents the policy from earning insertion credit while lifting (Z) or skidding (X).
    Returns ``relu(v_y)`` when ``sqrt(v_x² + v_z²) < max_off_axis`` and ``|v_y| > min_y``;
    otherwise 0.  Use a **positive** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    off_axis = torch.sqrt(torch.square(v[:, 0]) + torch.square(v[:, 2]))
    pure_y = off_axis < float(max_off_axis_speed_m_s)
    moving_y = v[:, 1] > float(min_y_speed_m_s)
    reward = torch.relu(v[:, 1])
    return torch.where(pure_y & moving_y, reward, torch.zeros_like(reward))


def pcb_push_axis_negative_velocity_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_backward_speed_m_s: float = 0.002,
) -> torch.Tensor:
    """Penalty for PCB root velocity along **−world Y** (away from the slot).

    Returns ``relu(-v_y)`` when ``v_y < -min_backward_speed_m_s``; otherwise 0.
    Pair with a **negative** weight.
    """
    v_y = env.scene[pcb_cfg.name].data.root_lin_vel_w[:, 1]
    backward = v_y < -float(min_backward_speed_m_s)
    penalty = torch.relu(-v_y)
    return torch.where(backward, penalty, torch.zeros_like(penalty))


def pcb_push_axis_sustained_backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_backward_speed_m_s: float = 0.003,
    min_consecutive_steps: int = 3,
) -> torch.Tensor:
    """Penalty for **sustained** −Y root velocity (ignores brief slip transients).

    Counts consecutive steps with ``v_y < -min_backward_speed_m_s``; applies ``relu(-v_y)``
    only once the streak reaches ``min_consecutive_steps``.  Pair with a **negative** weight.
    """
    v_y = env.scene[pcb_cfg.name].data.root_lin_vel_w[:, 1]
    threshold = float(min_backward_speed_m_s)
    backward = v_y < -threshold

    if not hasattr(env, "_insert_backward_step_count"):
        env._insert_backward_step_count = torch.zeros(env.num_envs, device=env.device, dtype=v_y.dtype)
    count = env._insert_backward_step_count
    reset = env.episode_length_buf <= 1
    count = torch.where(reset, torch.zeros_like(count), count)
    count = torch.where(backward, count + 1.0, torch.zeros_like(count))
    env._insert_backward_step_count = count

    sustained = count >= float(min_consecutive_steps)
    penalty = torch.relu(-v_y)
    return torch.where(sustained & backward, penalty, torch.zeros_like(penalty))


def gripper_mid_gated_push_axis_velocity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_pcb_off_axis_speed_m_s: float = 0.008,
    max_ee_off_axis_speed_m_s: float = 0.015,
    min_push_speed_m_s: float = 0.002,
) -> torch.Tensor:
    """Reward jaw-mid velocity along the push axis when PCB and EE stay in the insertion plane.

    Credits ``relu(v_push)`` only when PCB root off-axis speed and EE mid off-axis speed
    are below their thresholds — blocks lift-heavy joint2/3 shortcuts that move the arm
    without a clean +Y slide.
    """
    v_mid = gripper_midpoint_lin_vel_world(env, left_finger_cfg, right_finger_cfg)
    a = torch.tensor(axis_world, device=env.device, dtype=v_mid.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    v_push = torch.sum(v_mid * a.unsqueeze(0), dim=-1)
    ee_off_axis = torch.sqrt(torch.square(v_mid[:, 0]) + torch.square(v_mid[:, 2]))
    pcb_pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_pcb_off_axis_speed_m_s)
    ee_pure = ee_off_axis < float(max_ee_off_axis_speed_m_s)
    moving = v_push > float(min_push_speed_m_s)
    reward = torch.relu(v_push)
    return torch.where(pcb_pure & ee_pure & moving, reward, torch.zeros_like(reward))


def pcb_insertion_sdf_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_center_xyz_env: tuple[float, float, float],
    slot_half_dims_xyz: tuple[float, float, float],
    slot_mouth_y_env: float,
    align_axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
    pos_sigma_m: float = 0.15,
    align_coef: float = 0.20,
    depth_coef: float = 0.60,
) -> torch.Tensor:
    """SDF-inspired dense reward for PCB-slot insertion (IndustReal §3.2).

    Three components combined into one reward in ``[0, 1]``:

    1. **Proximity** — ``exp(-dist / sigma)`` from PCB leading-edge to slot
       center.  ``sigma`` must be set to ~½ the approach distance so the
       gradient is non-zero well before contact.

    2. **Alignment** — ``|cos θ|`` between the PCB long axis and the insertion
       direction. Incentivises keeping the board parallel to +Y.

    3. **Depth** — Linear fraction of penetration past the slot mouth,
       normalised by ``slot_depth = 2 × slot_half_dims_xyz[1]`` (not by
       ``half_length_m``). Ramps 0→1 over the actual slot depth.

    Parameters
    ----------
    slot_mouth_y_env:
        Env-local Y of the slot entrance plane.  Leading edge must cross this
        before depth is non-zero.
    pos_sigma_m:
        Gaussian width for proximity.  Should be ~½ the distance from the PCB
        start position to the slot center so the gradient reaches the start.
    align_coef, depth_coef:
        Component weights summing to ≤ 1 (remainder goes to proximity).
    """
    pcb = env.scene[pcb_cfg.name]
    device = env.device
    dtype = pcb.data.root_pos_w.dtype

    # ── Leading edge position (env-local) ─────────────────────────────────────
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_env = lead_w - env.scene.env_origins[:, :3]

    slot_ctr = torch.tensor(slot_center_xyz_env, device=device, dtype=dtype)
    slot_half = torch.tensor(slot_half_dims_xyz, device=device, dtype=dtype)

    # ── 1. Proximity: Gaussian on distance to slot center ────────────────────
    dist = torch.norm(lead_env - slot_ctr.unsqueeze(0), dim=-1)
    prox = torch.exp(-dist / (float(pos_sigma_m) + 1e-9))

    # ── 2. Alignment: PCB long axis ∥ insertion axis ─────────────────────────
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    a = torch.tensor(align_axis_world, device=device, dtype=dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    align = torch.abs(torch.sum(x_w * a.unsqueeze(0).expand_as(x_w), dim=-1))

    # ── 3. Depth: penetration past slot mouth, normalised by slot depth ───────
    slot_depth_m = 2.0 * float(slot_half[1])   # actual slot depth in metres
    lead_y = lead_env[:, 1]
    depth_frac = torch.clamp(
        (lead_y - float(slot_mouth_y_env)) / (slot_depth_m + 1e-9),
        min=0.0, max=1.0,
    )

    prox_coef = 1.0 - float(align_coef) - float(depth_coef)
    reward = prox_coef * prox + float(align_coef) * align + float(depth_coef) * depth_frac
    return reward.clamp(0.0, 1.0)