"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/slide shaping, regularization, rail reset, and drop detection.

Push and slide are separate registered envs; each uses its own reward config with no in-episode phase gating.
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
# Slide-phase actions — relative joint deltas with position-target clamps
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


def _finger_jaw_opening_gaps(
    left: torch.Tensor,
    right: torch.Tensor,
    center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clearances (m) from ``center`` to each jaw pad along the left→right jaw axis."""
    jaw = right - left
    u = jaw / torch.norm(jaw, dim=-1, keepdim=True).clamp_min(1e-9)
    gap_left = torch.sum((center - left) * u, dim=-1).clamp(min=0.0)
    gap_right = torch.sum((right - center) * u, dim=-1).clamp(min=0.0)
    return gap_left, gap_right


# wxai follower: physical jaw span (m) ≈ ``left_carriage_joint`` × scale (q=0.02 → 40 mm).
_GRIPPER_CARRIAGE_JOINT_SPAN_SCALE = 2.0


def _gripper_jaw_span_from_joint(gq: torch.Tensor) -> torch.Tensor:
    """Physical pad gap (m) from ``left_carriage_joint`` (not 2× joint)."""
    return gq * _GRIPPER_CARRIAGE_JOINT_SPAN_SCALE


def _left_carriage_half_gap_m(robot: Articulation, gripper_joint_cfg: SceneEntityCfg) -> torch.Tensor:
    """Half the physical pad gap (m) from carriage joint."""
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]].clamp(min=0.0)
    return gq * (_GRIPPER_CARRIAGE_JOINT_SPAN_SCALE * 0.5)


def _gripper_tip_offset_direction_w(
    robot: Articulation,
    left: torch.Tensor,
    right: torch.Tensor,
    wrist_body_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """Unit vector from wrist toward jaw bodies (distal pad-tip direction)."""
    mid = 0.5 * (left + right)
    if wrist_body_cfg is not None:
        wrist_id = _resolve_first_body_id(robot, wrist_body_cfg)
        wrist = robot.data.body_pos_w[:, wrist_id]
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


def gripper_jaw_pad_tips_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of left/right **contact pad tips** (not finger-body origins).

    Lateral: carriage span about the jaw midpoint (``left_carriage_joint × scale``).
    Distal: each pad tip is offset from the jaw-axis pad centre toward the PCB by
    ``tip_offset_m`` along wrist→jaw (body origin sits proximal to the actual contact point).
    """
    robot = env.scene[left_finger_cfg.name]
    left_id = _resolve_first_body_id(robot, left_finger_cfg)
    right_id = _resolve_first_body_id(robot, right_finger_cfg)
    left_h = robot.data.body_pos_w[:, left_id]
    right_h = robot.data.body_pos_w[:, right_id]
    jaw = right_h - left_h
    u = jaw / torch.norm(jaw, dim=-1, keepdim=True).clamp_min(1e-9)
    mid = 0.5 * (left_h + right_h)
    half_gap = _left_carriage_half_gap_m(robot, gripper_joint_cfg)
    left = mid - u * half_gap.unsqueeze(-1)
    right = mid + u * half_gap.unsqueeze(-1)
    if float(tip_offset_m) > 0.0:
        fwd = _gripper_tip_offset_direction_w(robot, left_h, right_h, wrist_body_cfg)
        off = float(tip_offset_m) * fwd
        left = left + off
        right = right + off
    return left, right


def _trailing_edge_jaw_opening_gaps(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jaw-axis clearances (m) from trailing-edge centre to each pad (joint-based span)."""
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    return _finger_jaw_opening_gaps(left, right, center)


def straddle_midpoint_target_offset_w(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    offset_m: float,
) -> torch.Tensor:
    """World offset for asymmetric approach — trailing centre shifted toward ``gripper_right`` (−body +Z)."""
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    return -z_w * float(offset_m)


# Pinch-ready containment: each jaw tip within this many × PCB half-thickness of centre.
# Open-straddle gate / closing-ready omit this; grasp success keeps the strict multiplier.
_JAW_CONTAINED_HALF_THICKNESS_MULT = 1.5
_PINCH_READY_JAW_CONTAINED_MULT = _JAW_CONTAINED_HALF_THICKNESS_MULT


def _resolve_first_body_id(robot: Articulation, body_cfg: SceneEntityCfg) -> int:
    """Return first body id from a SceneEntityCfg, resolving lazy/slice ids via names."""
    body_ids = body_cfg.body_ids
    if isinstance(body_ids, torch.Tensor):
        if body_ids.numel() > 0:
            return int(body_ids.flatten()[0].item())
    elif isinstance(body_ids, (list, tuple)):
        if len(body_ids) > 0:
            return int(body_ids[0])
    elif isinstance(body_ids, int):
        return int(body_ids)
    # Action-term cfgs can carry unresolved ids (e.g. slice(None)); resolve by body_names.
    if body_cfg.body_names is None:
        raise ValueError(f"Cannot resolve body id for SceneEntityCfg(name={body_cfg.name!r})")
    ids, _ = robot.find_bodies(body_cfg.body_names)
    if len(ids) == 0:
        raise ValueError(
            f"No bodies matched names={body_cfg.body_names!r} for asset {body_cfg.name!r}"
        )
    return int(ids[0])



def gripper_finger_tips_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of left/right gripper finger bodies."""
    robot = env.scene[left_finger_cfg.name]
    left_id = _resolve_first_body_id(robot, left_finger_cfg)
    right_id = _resolve_first_body_id(robot, right_finger_cfg)
    left = robot.data.body_pos_w[:, left_id]
    right = robot.data.body_pos_w[:, right_id]
    return left, right


def gripper_midpoint_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """World position at the midpoint between finger bodies."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
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
        wrist_body_cfg=wrist_body_cfg,
    )
    near = torch.exp(-dist / (float(gate_dist_m) + 1e-9))
    return near, sep_ok


def gripper_wrist_carriage_align_axis(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """``|dot(u_wc, axis_world)|`` in ``[0, 1]`` — wrist→carriage mid parallel to push axis (+Y)."""
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    mid = 0.5 * (left + right)
    d = mid - wrist
    u_wc = d / torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
    axis = _world_axis_batch(env, axis_world, u_wc.dtype)
    return torch.abs(torch.sum(u_wc * axis, dim=-1))


def gripper_wrist_carriage_yaw_align_axis(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """``|dot(u_wc_xy, push_xy)|`` in ``[0, 1]`` — wrist→carriage heading in XY only (yaw); Z pitch free."""
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)
    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    mid = 0.5 * (left + right)
    d_xy = mid[:, :2] - wrist[:, :2]
    n_xy = torch.norm(d_xy, dim=-1)
    axis = torch.tensor(push_axis_world[:2], device=env.device, dtype=d_xy.dtype)
    axis = axis / torch.norm(axis).clamp(min=1e-9)
    u_xy = d_xy / n_xy.unsqueeze(-1).clamp(min=1e-6)
    align = torch.abs(torch.sum(u_xy * axis.unsqueeze(0), dim=-1))
    return torch.where(n_xy > 1e-4, align, torch.zeros_like(align))


def gripper_wrist_carriage_yaw_pitch_limited_align(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_pitch_deg: float = 25.0,
    pitch_soft_deg: float = 10.0,
) -> torch.Tensor:
    """Yaw alignment in XY with a soft cap on wrist pitch (elevation of wrist→jaw from horizontal).

    For unit wrist→carriage direction ``u_wc``, ``|u_wc_z|`` is ``sin(pitch)``. Credit is full while
  ``|u_wc_z| <= sin(max_pitch_deg)`` and decays smoothly beyond that over ``pitch_soft_deg``.
    """
    yaw = gripper_wrist_carriage_yaw_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        push_axis_world,
    )
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return yaw

    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    mid = 0.5 * (left + right)
    u_wc = mid - wrist
    u_wc = u_wc / torch.norm(u_wc, dim=-1, keepdim=True).clamp(min=1e-6)
    z_abs = torch.abs(u_wc[:, 2])

    max_z = float(np.sin(np.radians(max_pitch_deg)))
    soft_z = max(float(np.sin(np.radians(pitch_soft_deg))), 1e-6)
    excess = torch.clamp(z_abs - max_z, min=0.0)
    pitch_gate = torch.exp(-excess / soft_z)
    return yaw * pitch_gate


def gripper_wrist_carriage_target_pitch_shaping(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    target_pitch_deg: float = 20.0,
    pitch_sigma_deg: float = 5.0,
) -> torch.Tensor:
    """Yaw alignment in XY plus soft target on wrist→jaw pitch (elevation from horizontal push plane).

    ``|u_wc_z|`` is ``sin(pitch)``.  Credit peaks near ``target_pitch_deg`` (e.g. 20° for camera
    clearance) with width ``pitch_sigma_deg`` (≈15–25° band at σ=5°).
    """
    yaw = gripper_wrist_carriage_yaw_align_axis(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        push_axis_world,
    )
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return yaw

    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    mid = 0.5 * (left + right)
    u_wc = mid - wrist
    u_wc = u_wc / torch.norm(u_wc, dim=-1, keepdim=True).clamp(min=1e-6)
    z_abs = torch.abs(u_wc[:, 2])
    target_z = float(np.sin(np.radians(target_pitch_deg)))
    sigma_z = max(float(np.sin(np.radians(pitch_sigma_deg))), 1e-6)
    pitch_q = torch.exp(-torch.abs(z_abs - target_z) / sigma_z)
    return yaw * pitch_q


def _gripper_belt_corridor_x_bounds_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    origins: torch.Tensor,
    wrist_body_cfg: SceneEntityCfg | None = None,
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
) -> torch.Tensor:
    """Env-local position at the midpoint between finger bodies."""
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
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
    reset_joint_pos_attr: str = "_slide_reset_joint_pos",
) -> torch.Tensor:
    """Joint positions relative to the robot pose stored at slide reset (straddle buffer row).

    Unlike ``joint_pos_rel`` (vs USD default / HOME), this zeros at the actual Phase-1
    terminal pose so the policy sees deltas from the straddle configuration.
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


def pcb_trailing_short_edge_center_lin_vel_world(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """World linear velocity at the trailing (rear) short-edge face centre."""
    pcb = env.scene[pcb_cfg.name]
    pos = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m, axis_world)
    r = pos - pcb.data.root_pos_w
    return pcb.data.root_lin_vel_w + torch.cross(pcb.data.root_ang_vel_w, r, dim=-1)


def _gripper_mid_trailing_edge_errors(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    target_offset_w: torch.Tensor | None = None,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCB-frame errors vs the trailing short-edge **face centre** target.

    Returns ``along, width, thick, in_plane, edge_dist`` where width is body +Y (short-edge width).
    """
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
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


def _one_sided_trailing_finger_dists(
    geom: dict[str, torch.Tensor],
    width_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-jaw 3-D distance to trailing-edge targets, one-sided along the long axis.

    ``along > 0`` (jaw past trailing face) does not increase distance — only width/thickness
   errors count once the jaw has entered the PCB body region.
    """
    ww = float(width_weight)
    along_l_err = torch.clamp(-geom["along_l"], min=0.0)
    along_r_err = torch.clamp(-geom["along_r"], min=0.0)
    dist_l = torch.sqrt(along_l_err**2 + (ww * geom["width_l"]) ** 2 + geom["thick_l"] ** 2 + 1e-6)
    dist_r = torch.sqrt(along_r_err**2 + (ww * geom["width_r"]) ** 2 + geom["thick_r"] ** 2 + 1e-6)
    return dist_l, dist_r


def _fingers_trailing_edge_geometry(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.00125,
    width_weight: float = 3.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> dict[str, torch.Tensor]:
    """Per-jaw geometry vs opposite-side trailing-edge targets (not the jaw midpoint)."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
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


def _both_jaws_near_trailing_edge(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float,
    pcb_half_thickness_m: float = 0.00125,
    width_weight: float = 3.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """True when both pad tips are within ``gate_dist_m`` of trailing-edge grasp targets."""
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        wrist_body_cfg,
    )
    max_dist = torch.maximum(geom["dist_l"], geom["dist_r"])
    return max_dist < float(gate_dist_m)


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
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed offset of each jaw tip along PCB body +Z from the board centre (thickness axis)."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
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
        wrist_body_cfg,
    )
    dist_l, dist_r = _one_sided_trailing_finger_dists(geom, width_weight)
    return torch.maximum(dist_l, dist_r)


def _trailing_edge_xy_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    width_weight: float = 3.0,
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
        wrist_body_cfg=wrist_body_cfg,
    )


def _straddle_asymmetric_parts(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float,
    proximity_sigma_m: float,
    pcb_half_thickness_m: float = 0.0005,
    width_gap_target_left_m: float = 0.012,
    width_gap_target_right_m: float = 0.003,
    gap_tolerance_m: float = 0.004,
    min_open_width_m: float | None = None,
    open_width_tolerance_m: float = 0.002,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.0005,
    min_span_frac: float = 0.01,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> dict[str, torch.Tensor]:
    """Per-env straddle gate booleans plus jaw-axis gap measurements (m)."""
    del proximity_sigma_m, open_width_m, min_open_width_m

    left, right = gripper_jaw_pad_tips_world(env, left_finger_cfg, right_finger_cfg, gripper_joint_cfg)
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg=wrist_body_cfg,
    )
    along_min = float(min_along_m)
    jaws_past = (geom["along_l"] >= along_min) & (geom["along_r"] >= along_min)
    between_jaws = _pcb_between_jaws_mask(
        env, pcb_cfg, left, right, min_span_frac, min_straddle_sep_m
    )
    z_straddled = _width_straddle_ready_mask(
        env, pcb_cfg, left, right, min_straddle_sep_m, min_span_frac
    )
    gap_left, gap_right = _finger_jaw_opening_gaps(left, right, center)
    jaw_span = gap_left + gap_right
    target_span = float(width_gap_target_left_m) + float(width_gap_target_right_m)
    tol = float(gap_tolerance_m)
    gap_ok = (
        (torch.abs(gap_left - float(width_gap_target_left_m)) <= tol)
        & (torch.abs(gap_right - float(width_gap_target_right_m)) <= tol)
    )
    span_tol = float(open_width_tolerance_m)
    open_ok = torch.abs(jaw_span - target_span) <= span_tol
    achieved = jaws_past & between_jaws & z_straddled & gap_ok & open_ok
    return {
        "jaws_past": jaws_past.reshape(env.num_envs),
        "between_jaws": between_jaws.reshape(env.num_envs),
        "z_straddled": z_straddled.reshape(env.num_envs),
        "gap_left": gap_left.reshape(env.num_envs),
        "gap_right": gap_right.reshape(env.num_envs),
        "jaw_span": jaw_span.reshape(env.num_envs),
        "gap_ok": gap_ok.reshape(env.num_envs),
        "open_ok": open_ok.reshape(env.num_envs),
        "achieved": achieved.reshape(env.num_envs),
    }


def straddle_asymmetric_achieved(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float | None = None,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    open_width_m: float | None = None,
    proximity_sigma_m: float | None = None,
    pcb_half_thickness_m: float | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    gap_tolerance_m: float | None = None,
    min_open_width_m: float | None = None,
    open_width_tolerance_m: float | None = None,
    min_along_m: float | None = None,
    min_straddle_sep_m: float | None = None,
    min_span_frac: float | None = None,
    min_between_quality: float | None = None,
    width_gap_sigma_m: float | None = None,
    pcb_half_width_m: float | None = None,
    width_weight: float | None = None,
    width_sigma_m: float | None = None,
    jaw_thick_gate_std_m: float | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for :func:`straddle_finger_target_success`."""
    sig = float(
        std
        if std is not None
        else (proximity_sigma_m if proximity_sigma_m is not None else 0.005)
    )
    del (
        open_width_m,
        proximity_sigma_m,
        pcb_half_thickness_m,
        width_gap_target_left_m,
        width_gap_target_right_m,
        gap_tolerance_m,
        min_open_width_m,
        open_width_tolerance_m,
        min_along_m,
        min_straddle_sep_m,
        min_span_frac,
        min_between_quality,
        width_gap_sigma_m,
        pcb_half_width_m,
        width_weight,
        width_sigma_m,
        jaw_thick_gate_std_m,
    )
    return straddle_finger_target_success(
        env,
        sig,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        closedness_threshold=closedness_threshold,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def straddle_success_bonus_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    state_attr: str = "_straddle_success_bonus_paid",
) -> torch.Tensor:
    """One-shot bonus (1.0) the first time finger-target closedness crosses ``closedness_threshold``."""
    if not hasattr(env, state_attr):
        setattr(env, state_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    paid: torch.Tensor = getattr(env, state_attr)
    if paid.shape[0] != env.num_envs:
        paid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, state_attr, paid)

    paid[env.episode_length_buf == 1] = False

    achieved = straddle_finger_target_success(
        env,
        std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        closedness_threshold=closedness_threshold,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    newly = achieved & (~paid)
    paid[:] = paid | achieved
    return newly.to(dtype=torch.float32)




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
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
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
        push_axis_world,
    )
    # Floor wc_align at 0.5 so a non-ideal heading reduces but never zeroes the readiness.
    wc_align = 0.5 + 0.5 * wc_align

    # Anti-degenerate: jaws straddle the board faces with a real gap along the thickness axis.
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, wrist_body_cfg
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
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward reaching the PCB with the jaw midpoint (Isaac Lab ``object_ee_distance`` / lift task).

    See ``isaaclab_tasks/.../lift/mdp/rewards.py::object_ee_distance``.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos_w = pcb.data.root_pos_w
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
    distance = torch.norm(pcb_pos_w - mid, dim=-1)
    return 1.0 - torch.tanh(distance / (float(std) + 1e-9))


def _open_straddle_ready_mask(
    w_left: torch.Tensor,
    w_right: torch.Tensor,
    min_straddle_sep_m: float,
) -> torch.Tensor:
    """True when pad tips are on opposite PCB faces with minimum thickness-axis separation."""
    straddle_gap = torch.abs(w_left - w_right)
    separated = straddle_gap >= float(min_straddle_sep_m)
    return (w_left * w_right < 0.0) & separated


def _width_straddle_ready_mask(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left: torch.Tensor,
    right: torch.Tensor,
    min_straddle_sep_m: float,
    min_span_frac: float = 0.01,
    min_y_align: float = 0.85,
) -> torch.Tensor:
    """True when jaws span the PCB along body +Y (78.5 mm trailing edge) with PCB centred in span."""
    between_jaws = _pcb_between_jaws_mask(
        env, pcb_cfg, left, right, min_span_frac, min_straddle_sep_m
    )
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    jaw = left - right
    u = jaw / torch.norm(jaw, dim=-1, keepdim=True).clamp_min(1e-9)
    y_align = torch.abs(torch.sum(u * y_w, dim=-1)) >= float(min_y_align)
    span_ok = torch.norm(jaw, dim=-1) >= float(min_straddle_sep_m)
    return between_jaws & y_align & span_ok


def _pcb_between_jaws_mask(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left: torch.Tensor,
    right: torch.Tensor,
    min_span_frac: float,
    min_straddle_sep_m: float,
) -> torch.Tensor:
    """True when the PCB centre lies inside the open jaw span along the thickness axis."""
    pcb_pos = env.scene[pcb_cfg.name].data.root_pos_w
    span = right - left
    span_len_sq = torch.sum(span * span, dim=-1).clamp_min(1e-12)
    span_len = torch.sqrt(span_len_sq)
    span_frac = torch.sum((pcb_pos - left) * span, dim=-1) / span_len_sq
    margin = float(min_span_frac)
    return (
        (span_len >= float(min_straddle_sep_m))
        & (span_frac >= margin)
        & (span_frac <= (1.0 - margin))
    )


def pcb_open_straddle_gate_quality(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.002,
    width_weight: float = 3.0,
    jaw_thick_gate_std_m: float | None = None,
) -> torch.Tensor:
    """Open-jaw straddle signal for the effort gate (no ``jaw_contained``).

    Returns ``1.0`` when pad tips are on opposite PCB faces, the board centre lies
    between the jaw span, and both tips are past the trailing short-edge face.
    """
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_weight,
        wrist_body_cfg=wrist_body_cfg,
    )
    w_left, w_right = _finger_thickness_offsets(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, wrist_body_cfg
    )
    z_open = _open_straddle_ready_mask(w_left, w_right, min_straddle_sep_m)
    along_min = float(min_along_m)
    jaws_past_edge = (geom["along_l"] >= along_min) & (geom["along_r"] >= along_min)
    between_jaws = _pcb_between_jaws_mask(
        env, pcb_cfg, left, right, min_span_frac, min_straddle_sep_m
    )
    ready = z_open & jaws_past_edge & between_jaws
    quality = ready.to(left.dtype)
    return quality


def _midpoint_trailing_edge_soft_gate(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    std_m: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Soft gate ≈1 when the gripper midpoint is within ``std_m`` of the trailing-edge face centre."""
    _, _, _, _, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
    )
    sig = float(std_m) + 1e-9
    return 1.0 - torch.tanh(edge_dist / sig)


def straddle_finger_pcb_z_vertical_distances(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute PCB-thickness-axis offset (m) of each jaw tip from its trailing-edge face target.

    When the board lies flat, this matches the vertical (world +Z) clearance to the nearest
    PCB top/bottom face at the trailing short edge.
    """
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg=wrist_body_cfg,
    )
    return torch.abs(geom["thick_l"]), torch.abs(geom["thick_r"])


def _pcb_open_straddle_ready_parts(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.0005,
) -> dict[str, torch.Tensor]:
    """Per-env booleans for each hard straddle gate term plus combined ``ready``."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg=wrist_body_cfg,
    )
    jaws_past = (geom["along_l"] >= float(min_along_m)) & (geom["along_r"] >= float(min_along_m))
    between_jaws = _pcb_between_jaws_mask(
        env, pcb_cfg, left, right, min_span_frac, min_straddle_sep_m
    )
    z_straddled = _width_straddle_ready_mask(
        env, pcb_cfg, left, right, min_straddle_sep_m, min_span_frac
    )
    ready = jaws_past & between_jaws & z_straddled
    return {
        "jaws_past": jaws_past.reshape(env.num_envs),
        "between_jaws": between_jaws.reshape(env.num_envs),
        "z_straddled": z_straddled.reshape(env.num_envs),
        "ready": ready.reshape(env.num_envs),
    }


def pcb_open_straddle_ready(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.0005,
) -> torch.Tensor:
    """Hard boolean gate: Z-straddle, jaws past trailing face, PCB in jaw span."""
    return _pcb_open_straddle_ready_parts(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        wrist_body_cfg=wrist_body_cfg,
        min_span_frac=min_span_frac,
        min_along_m=min_along_m,
        min_straddle_sep_m=min_straddle_sep_m,
    )["ready"]


# Per-env episode accumulators for push gripper debug logging.
_STRADDLE_DEBUG_JOINT_SUM: torch.Tensor | None = None
_STRADDLE_DEBUG_READY_SUM: torch.Tensor | None = None
_STRADDLE_DEBUG_JAWS_PAST_SUM: torch.Tensor | None = None
_STRADDLE_DEBUG_BETWEEN_JAWS_SUM: torch.Tensor | None = None
_STRADDLE_DEBUG_Z_STRADDLED_SUM: torch.Tensor | None = None
_STRADDLE_DEBUG_STEP_COUNT: torch.Tensor | None = None
_STRADDLE_DEBUG_LAST_GQ: torch.Tensor | None = None
_STRADDLE_DEBUG_LAST_READY: torch.Tensor | None = None
_STRADDLE_DEBUG_LAST_JAWS_PAST: torch.Tensor | None = None
_STRADDLE_DEBUG_LAST_BETWEEN_JAWS: torch.Tensor | None = None
_STRADDLE_DEBUG_LAST_Z_STRADDLED: torch.Tensor | None = None
_STRADDLE_Z_LEFT_SUM: torch.Tensor | None = None
_STRADDLE_Z_RIGHT_SUM: torch.Tensor | None = None
_STRADDLE_ACHIEVED_STEP_COUNT: torch.Tensor | None = None
_STRADDLE_Z_LEFT_LAST: torch.Tensor | None = None
_STRADDLE_Z_RIGHT_LAST: torch.Tensor | None = None
_STRADDLE_GAP_LEFT_SUM: torch.Tensor | None = None
_STRADDLE_GAP_RIGHT_SUM: torch.Tensor | None = None
_STRADDLE_GAP_LEFT_LAST: torch.Tensor | None = None
_STRADDLE_GAP_RIGHT_LAST: torch.Tensor | None = None
_STRADDLE_CLOSEDNESS_SUM: torch.Tensor | None = None
_STRADDLE_CLOSEDNESS_LAST: torch.Tensor | None = None
_STRADDLE_CLOSEDNESS_TIGHT_SUM: torch.Tensor | None = None
_STRADDLE_CLOSEDNESS_TIGHT_LAST: torch.Tensor | None = None
_STRADDLE_CLOSEDNESS_EP_MAX: torch.Tensor | None = None
_STRADDLE_DIST_L_SUM: torch.Tensor | None = None
_STRADDLE_DIST_R_SUM: torch.Tensor | None = None
_STRADDLE_DIST_L_LAST: torch.Tensor | None = None
_STRADDLE_DIST_R_LAST: torch.Tensor | None = None
_STRADDLE_PUSH_PROGRESS_SUM: torch.Tensor | None = None
_STRADDLE_PUSH_PROGRESS_LAST: torch.Tensor | None = None
_STRADDLE_PUSH_GATE_OPEN_SUM: torch.Tensor | None = None
_STRADDLE_PUSH_GATE_OPEN_LAST: torch.Tensor | None = None
_STRADDLE_LEAD_Y_SUM: torch.Tensor | None = None
_STRADDLE_LEAD_Y_LAST: torch.Tensor | None = None
_STRADDLE_LEAD_VY_SUM: torch.Tensor | None = None
_STRADDLE_LEAD_VY_LAST: torch.Tensor | None = None
_STRADDLE_BETWEEN_FINGERS_SUM: torch.Tensor | None = None
_STRADDLE_BETWEEN_FINGERS_LAST: torch.Tensor | None = None


def _push_gripper_debug_ensure_buffers(env: ManagerBasedEnv) -> None:
    global _STRADDLE_DEBUG_JOINT_SUM, _STRADDLE_DEBUG_READY_SUM, _STRADDLE_DEBUG_STEP_COUNT
    global _STRADDLE_DEBUG_JAWS_PAST_SUM, _STRADDLE_DEBUG_BETWEEN_JAWS_SUM, _STRADDLE_DEBUG_Z_STRADDLED_SUM
    global _STRADDLE_DEBUG_LAST_GQ, _STRADDLE_DEBUG_LAST_READY
    global _STRADDLE_DEBUG_LAST_JAWS_PAST, _STRADDLE_DEBUG_LAST_BETWEEN_JAWS, _STRADDLE_DEBUG_LAST_Z_STRADDLED
    global _STRADDLE_Z_LEFT_SUM, _STRADDLE_Z_RIGHT_SUM, _STRADDLE_ACHIEVED_STEP_COUNT
    global _STRADDLE_Z_LEFT_LAST, _STRADDLE_Z_RIGHT_LAST
    global _STRADDLE_GAP_LEFT_SUM, _STRADDLE_GAP_RIGHT_SUM
    global _STRADDLE_GAP_LEFT_LAST, _STRADDLE_GAP_RIGHT_LAST
    global _STRADDLE_CLOSEDNESS_SUM, _STRADDLE_CLOSEDNESS_LAST
    global _STRADDLE_CLOSEDNESS_TIGHT_SUM, _STRADDLE_CLOSEDNESS_TIGHT_LAST, _STRADDLE_CLOSEDNESS_EP_MAX
    global _STRADDLE_DIST_L_SUM, _STRADDLE_DIST_R_SUM, _STRADDLE_DIST_L_LAST, _STRADDLE_DIST_R_LAST
    global _STRADDLE_PUSH_PROGRESS_SUM, _STRADDLE_PUSH_PROGRESS_LAST
    global _STRADDLE_PUSH_GATE_OPEN_SUM, _STRADDLE_PUSH_GATE_OPEN_LAST
    global _STRADDLE_LEAD_Y_SUM, _STRADDLE_LEAD_Y_LAST
    global _STRADDLE_LEAD_VY_SUM, _STRADDLE_LEAD_VY_LAST
    global _STRADDLE_BETWEEN_FINGERS_SUM, _STRADDLE_BETWEEN_FINGERS_LAST
    n = env.num_envs
    if (
        _STRADDLE_DEBUG_JOINT_SUM is None
        or _STRADDLE_DEBUG_JOINT_SUM.shape[0] != n
        or _STRADDLE_DEBUG_JOINT_SUM.device != env.device
    ):
        _STRADDLE_DEBUG_JOINT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_READY_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_JAWS_PAST_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_BETWEEN_JAWS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_Z_STRADDLED_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_STEP_COUNT = torch.zeros(n, device=env.device, dtype=torch.long)
        _STRADDLE_DEBUG_LAST_GQ = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DEBUG_LAST_READY = torch.zeros(n, device=env.device, dtype=torch.bool)
        _STRADDLE_DEBUG_LAST_JAWS_PAST = torch.zeros(n, device=env.device, dtype=torch.bool)
        _STRADDLE_DEBUG_LAST_BETWEEN_JAWS = torch.zeros(n, device=env.device, dtype=torch.bool)
        _STRADDLE_DEBUG_LAST_Z_STRADDLED = torch.zeros(n, device=env.device, dtype=torch.bool)
        _STRADDLE_Z_LEFT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_Z_RIGHT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_ACHIEVED_STEP_COUNT = torch.zeros(n, device=env.device, dtype=torch.long)
        _STRADDLE_Z_LEFT_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_Z_RIGHT_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_GAP_LEFT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_GAP_RIGHT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_GAP_LEFT_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_GAP_RIGHT_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_TIGHT_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_TIGHT_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_EP_MAX = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DIST_L_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DIST_R_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DIST_L_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_DIST_R_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PUSH_PROGRESS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PUSH_PROGRESS_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PUSH_GATE_OPEN_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PUSH_GATE_OPEN_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_LEAD_Y_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_LEAD_Y_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_LEAD_VY_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_LEAD_VY_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_BETWEEN_FINGERS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_BETWEEN_FINGERS_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)


def _read_straddle_debug_sample(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    proximity_sigma_m: float,
    pcb_half_thickness_m: float = 0.0005,
    width_gap_target_left_m: float = 0.012,
    width_gap_target_right_m: float = 0.003,
    gap_tolerance_m: float = 0.004,
    min_open_width_m: float | None = None,
    open_width_tolerance_m: float = 0.002,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.0005,
    min_span_frac: float = 0.01,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(z_left_m, z_right_m, gap_left_m, gap_right_m, straddle_achieved)`` per env."""
    z_left, z_right = straddle_finger_pcb_z_vertical_distances(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        asset_cfg,
        half_length_m,
    )
    parts = _straddle_asymmetric_parts(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        asset_cfg,
        half_length_m,
        open_width_m,
        proximity_sigma_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        gap_tolerance_m=gap_tolerance_m,
        min_open_width_m=min_open_width_m,
        open_width_tolerance_m=open_width_tolerance_m,
        min_along_m=min_along_m,
        min_straddle_sep_m=min_straddle_sep_m,
        min_span_frac=min_span_frac,
        wrist_body_cfg=wrist_body_cfg,
    )
    return z_left, z_right, gap_left, gap_right, parts["achieved"]


def _read_straddle_z_debug_sample(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-compatible wrapper — Z distances and achieved mask only."""
    z_left, z_right, _, _, achieved = _read_straddle_debug_sample(*args, **kwargs)
    return z_left, z_right, achieved


def _push_gripper_debug_accumulate_step(
    gap_left: torch.Tensor,
    gap_right: torch.Tensor,
    achieved: torch.Tensor,
    closedness_prox: torch.Tensor,
    closedness_tight: torch.Tensor | None = None,
    dist_left: torch.Tensor | None = None,
    dist_right: torch.Tensor | None = None,
) -> None:
    """Accumulate per-step jaw gaps and closedness for episode-mean TensorBoard scalars."""
    global _STRADDLE_GAP_LEFT_SUM, _STRADDLE_GAP_RIGHT_SUM
    global _STRADDLE_GAP_LEFT_LAST, _STRADDLE_GAP_RIGHT_LAST
    global _STRADDLE_ACHIEVED_STEP_COUNT, _STRADDLE_DEBUG_STEP_COUNT
    global _STRADDLE_CLOSEDNESS_SUM, _STRADDLE_CLOSEDNESS_LAST
    global _STRADDLE_CLOSEDNESS_TIGHT_SUM, _STRADDLE_CLOSEDNESS_TIGHT_LAST, _STRADDLE_CLOSEDNESS_EP_MAX
    global _STRADDLE_DIST_L_SUM, _STRADDLE_DIST_R_SUM, _STRADDLE_DIST_L_LAST, _STRADDLE_DIST_R_LAST
    gap_left_f = gap_left.detach().to(dtype=torch.float32)
    gap_right_f = gap_right.detach().to(dtype=torch.float32)
    _STRADDLE_GAP_LEFT_SUM += gap_left_f
    _STRADDLE_GAP_RIGHT_SUM += gap_right_f
    _STRADDLE_GAP_LEFT_LAST = gap_left_f
    _STRADDLE_GAP_RIGHT_LAST = gap_right_f
    _STRADDLE_ACHIEVED_STEP_COUNT += achieved.detach().to(dtype=torch.long)
    _STRADDLE_DEBUG_STEP_COUNT += 1
    prox_f = closedness_prox.detach().to(dtype=torch.float32)
    _STRADDLE_CLOSEDNESS_SUM += prox_f
    _STRADDLE_CLOSEDNESS_LAST = prox_f
    _STRADDLE_CLOSEDNESS_EP_MAX = torch.maximum(_STRADDLE_CLOSEDNESS_EP_MAX, prox_f)
    if closedness_tight is not None:
        tight_f = closedness_tight.detach().to(dtype=torch.float32)
        _STRADDLE_CLOSEDNESS_TIGHT_SUM += tight_f
        _STRADDLE_CLOSEDNESS_TIGHT_LAST = tight_f
    if dist_left is not None:
        dist_l_f = dist_left.detach().to(dtype=torch.float32)
        _STRADDLE_DIST_L_SUM += dist_l_f
        _STRADDLE_DIST_L_LAST = dist_l_f
    if dist_right is not None:
        dist_r_f = dist_right.detach().to(dtype=torch.float32)
        _STRADDLE_DIST_R_SUM += dist_r_f
        _STRADDLE_DIST_R_LAST = dist_r_f


def _push_debug_accumulate_step(
    push_progress: torch.Tensor,
    push_gate_open: torch.Tensor,
    lead_y_env: torch.Tensor,
    lead_vy: torch.Tensor,
    between_fingers_q: torch.Tensor,
) -> None:
    """Accumulate per-step +Y push diagnostics for episode-mean TensorBoard scalars."""
    global _STRADDLE_PUSH_PROGRESS_SUM, _STRADDLE_PUSH_PROGRESS_LAST
    global _STRADDLE_PUSH_GATE_OPEN_SUM, _STRADDLE_PUSH_GATE_OPEN_LAST
    global _STRADDLE_LEAD_Y_SUM, _STRADDLE_LEAD_Y_LAST
    global _STRADDLE_LEAD_VY_SUM, _STRADDLE_LEAD_VY_LAST
    global _STRADDLE_BETWEEN_FINGERS_SUM, _STRADDLE_BETWEEN_FINGERS_LAST
    push_f = push_progress.detach().to(dtype=torch.float32)
    gate_f = push_gate_open.detach().to(dtype=torch.float32)
    lead_y_f = lead_y_env.detach().to(dtype=torch.float32)
    lead_vy_f = lead_vy.detach().to(dtype=torch.float32)
    between_f = between_fingers_q.detach().to(dtype=torch.float32)
    _STRADDLE_PUSH_PROGRESS_SUM += push_f
    _STRADDLE_PUSH_PROGRESS_LAST = push_f
    _STRADDLE_PUSH_GATE_OPEN_SUM += gate_f
    _STRADDLE_PUSH_GATE_OPEN_LAST = gate_f
    _STRADDLE_LEAD_Y_SUM += lead_y_f
    _STRADDLE_LEAD_Y_LAST = lead_y_f
    _STRADDLE_LEAD_VY_SUM += lead_vy_f
    _STRADDLE_LEAD_VY_LAST = lead_vy_f
    _STRADDLE_BETWEEN_FINGERS_SUM += between_f
    _STRADDLE_BETWEEN_FINGERS_LAST = between_f


def _push_gripper_debug_accumulate_straddle_metrics(
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    gap_left: torch.Tensor,
    gap_right: torch.Tensor,
    achieved: torch.Tensor,
) -> None:
    """Accumulate finger–PCB Z and lateral gap samples while straddle is held."""
    global _STRADDLE_Z_LEFT_SUM, _STRADDLE_Z_RIGHT_SUM, _STRADDLE_ACHIEVED_STEP_COUNT
    global _STRADDLE_Z_LEFT_LAST, _STRADDLE_Z_RIGHT_LAST
    global _STRADDLE_GAP_LEFT_SUM, _STRADDLE_GAP_RIGHT_SUM
    global _STRADDLE_GAP_LEFT_LAST, _STRADDLE_GAP_RIGHT_LAST
    mask = achieved.to(dtype=torch.bool)
    z_left_f = z_left.detach().to(dtype=torch.float32)
    z_right_f = z_right.detach().to(dtype=torch.float32)
    gap_left_f = gap_left.detach().to(dtype=torch.float32)
    gap_right_f = gap_right.detach().to(dtype=torch.float32)
    _STRADDLE_Z_LEFT_SUM += torch.where(mask, z_left_f, torch.zeros_like(z_left_f))
    _STRADDLE_Z_RIGHT_SUM += torch.where(mask, z_right_f, torch.zeros_like(z_right_f))
    _STRADDLE_GAP_LEFT_SUM += torch.where(mask, gap_left_f, torch.zeros_like(gap_left_f))
    _STRADDLE_GAP_RIGHT_SUM += torch.where(mask, gap_right_f, torch.zeros_like(gap_right_f))
    _STRADDLE_ACHIEVED_STEP_COUNT += mask.to(dtype=torch.long)
    _STRADDLE_Z_LEFT_LAST = torch.where(mask, z_left_f, _STRADDLE_Z_LEFT_LAST)
    _STRADDLE_Z_RIGHT_LAST = torch.where(mask, z_right_f, _STRADDLE_Z_RIGHT_LAST)
    _STRADDLE_GAP_LEFT_LAST = torch.where(mask, gap_left_f, _STRADDLE_GAP_LEFT_LAST)
    _STRADDLE_GAP_RIGHT_LAST = torch.where(mask, gap_right_f, _STRADDLE_GAP_RIGHT_LAST)


def _push_gripper_debug_accumulate_straddle_z(
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    achieved: torch.Tensor,
) -> None:
    """Deprecated alias — gaps omitted."""
    zeros = torch.zeros_like(z_left)
    _push_gripper_debug_accumulate_straddle_metrics(z_left, z_right, zeros, zeros, achieved)


def _read_straddle_gripper_debug_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float,
    min_along_m: float,
    min_straddle_sep_m: float,
    min_span_frac: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``(left_carriage_joint, straddle_ready, ready_parts)`` per env."""
    robot = env.scene[asset_cfg.name]
    gq = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    parts = _pcb_open_straddle_ready_parts(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        wrist_body_cfg=wrist_body_cfg,
        min_span_frac=min_span_frac,
        min_along_m=min_along_m,
        min_straddle_sep_m=min_straddle_sep_m,
    )
    ready = parts["ready"]
    return gq, ready, parts


def _push_gripper_debug_accumulate_tensors(
    gq: torch.Tensor, ready: torch.Tensor, parts: dict[str, torch.Tensor]
) -> None:
    """Add one control-step sample to per-env episode accumulators."""
    global _STRADDLE_DEBUG_JOINT_SUM, _STRADDLE_DEBUG_READY_SUM, _STRADDLE_DEBUG_STEP_COUNT
    global _STRADDLE_DEBUG_JAWS_PAST_SUM, _STRADDLE_DEBUG_BETWEEN_JAWS_SUM, _STRADDLE_DEBUG_Z_STRADDLED_SUM
    global _STRADDLE_DEBUG_LAST_GQ, _STRADDLE_DEBUG_LAST_READY
    global _STRADDLE_DEBUG_LAST_JAWS_PAST, _STRADDLE_DEBUG_LAST_BETWEEN_JAWS, _STRADDLE_DEBUG_LAST_Z_STRADDLED
    _STRADDLE_DEBUG_JOINT_SUM += gq.detach()
    _STRADDLE_DEBUG_READY_SUM += ready.to(dtype=torch.float32).detach()
    _STRADDLE_DEBUG_JAWS_PAST_SUM += parts["jaws_past"].to(dtype=torch.float32).detach()
    _STRADDLE_DEBUG_BETWEEN_JAWS_SUM += parts["between_jaws"].to(dtype=torch.float32).detach()
    _STRADDLE_DEBUG_Z_STRADDLED_SUM += parts["z_straddled"].to(dtype=torch.float32).detach()
    _STRADDLE_DEBUG_STEP_COUNT += 1
    _STRADDLE_DEBUG_LAST_GQ = gq.detach()
    _STRADDLE_DEBUG_LAST_READY = ready.detach().bool()
    _STRADDLE_DEBUG_LAST_JAWS_PAST = parts["jaws_past"].detach().bool()
    _STRADDLE_DEBUG_LAST_BETWEEN_JAWS = parts["between_jaws"].detach().bool()
    _STRADDLE_DEBUG_LAST_Z_STRADDLED = parts["z_straddled"].detach().bool()


def push_gripper_debug_accumulate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Zero-weight reward hook: accumulate gap / closedness stats before episode reset."""
    del asset_cfg
    _push_gripper_debug_ensure_buffers(env)
    gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    closedness = straddle_finger_target_closedness(
        env,
        std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    achieved = closedness >= float(closedness_threshold)
    _push_gripper_debug_accumulate_step(
        gap_left, gap_right, achieved, closedness, closedness_tight=closedness
    )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def push_gripper_debug_step(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    asset_cfg: SceneEntityCfg,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    print_every_control_steps: int = 32,
    print_env_id: int = 0,
    enable_print: bool = False,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    max_off_axis_speed_m_s: float = 0.020,
    min_straddle_quality: float = 0.3,
    proximity_sigma_m: float = 0.050,
    pcb_half_thickness_m: float = 0.00075,
    width_sigma_m: float = 0.025,
    min_closedness_for_push: float = 0.0,
    proximity_std_m: float = 0.035,
) -> None:
    """Accumulate gap / closedness / +Y push stats each step; optional console line for play."""
    del env_ids, asset_cfg

    _push_gripper_debug_ensure_buffers(env)
    gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    dist_left, dist_right, along_l, along_r, thick_l, thick_r = _straddle_width_target_tip_dists(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    closedness_tight = straddle_finger_target_closedness(
        env,
        std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    closedness_prox = straddle_finger_target_closedness(
        env,
        proximity_std_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    achieved = closedness_prox >= float(closedness_threshold)
    _push_gripper_debug_accumulate_step(
        gap_left,
        gap_right,
        achieved,
        closedness_prox,
        closedness_tight=closedness_tight,
        dist_left=dist_left,
        dist_right=dist_right,
    )

    push_thresh = (
        float(min_closedness_for_push)
        if float(min_closedness_for_push) > 0.0
        else float(closedness_threshold)
    )

    between_q = pcb_between_gripper_fingers(
        env,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
    )
    push_step = pcb_leading_edge_push_axis_approach_progress(
        env, pcb_cfg, half_length_m, axis_world, max_step_m
    )
    pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_off_axis_speed_m_s)
    push_progress = torch.where(pure, push_step, torch.zeros_like(push_step))
    push_gate = closedness_prox >= push_thresh
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    lead_y_env = lead_env[:, 1]
    pcb = env.scene[pcb_cfg.name]
    a = torch.tensor(axis_world, device=env.device, dtype=pcb.data.root_lin_vel_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    lead_vy = torch.sum(pcb.data.root_lin_vel_w * a.unsqueeze(0), dim=-1)
    _push_debug_accumulate_step(push_progress, push_gate, lead_y_env, lead_vy, between_q)

    do_print = enable_print or env.num_envs <= 8
    if not do_print:
        return
    if int(env.common_step_counter) % int(print_every_control_steps) != 0:
        return
    eid = int(print_env_id) % env.num_envs
    print(
        f"[push] step={int(env.common_step_counter)} env={eid} "
        f"dist_L={dist_left[eid].item()*1000:.1f}mm dist_R={dist_right[eid].item()*1000:.1f}mm "
        f"along_L={along_l[eid].item()*1000:.1f}mm along_R={along_r[eid].item()*1000:.1f}mm "
        f"thick_L={thick_l[eid].item()*1000:.1f}mm thick_R={thick_r[eid].item()*1000:.1f}mm "
        f"gap_L={gap_left[eid].item()*1000:.1f}mm gap_R={gap_right[eid].item()*1000:.1f}mm "
        f"closedness={closedness_prox[eid].item():.3f} "
        f"closedness_tight={closedness_tight[eid].item():.3f} "
        f"between_q={between_q[eid].item():.3f} "
        f"push_gate={bool(push_gate[eid].item())} "
        f"push_prog={push_progress[eid].item():.4f} "
        f"lead_y={lead_y_env[eid].item()*1000:.1f}mm "
        f"lead_vy={lead_vy[eid].item()*1000:.1f}mm/s "
        f"success={bool(achieved[eid].item())} "
        f"(std={float(std)*1000:.1f}mm gate_q>{float(min_straddle_quality):.2f})",
        flush=True,
    )


def push_gripper_debug_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    max_off_axis_speed_m_s: float = 0.020,
    min_straddle_quality: float = 0.3,
    proximity_sigma_m: float = 0.050,
    pcb_half_thickness_m: float = 0.00075,
    width_sigma_m: float = 0.025,
    min_closedness_for_push: float = 0.0,
    proximity_std_m: float = 0.035,
) -> dict[str, float]:
    """Log episode closedness / gap / push means to TensorBoard via ``Curriculum/push_gripper_debug/*``."""
    del (
        asset_cfg,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        std,
        finger_offset_m,
        closedness_threshold,
        tip_offset_m,
        wrist_body_cfg,
        axis_world,
        max_step_m,
        max_off_axis_speed_m_s,
        min_straddle_quality,
        proximity_sigma_m,
        pcb_half_thickness_m,
        width_sigma_m,
        min_closedness_for_push,
        proximity_std_m,
    )
    global _STRADDLE_DEBUG_STEP_COUNT, _STRADDLE_ACHIEVED_STEP_COUNT
    global _STRADDLE_GAP_LEFT_SUM, _STRADDLE_GAP_RIGHT_SUM
    global _STRADDLE_GAP_LEFT_LAST, _STRADDLE_GAP_RIGHT_LAST
    global _STRADDLE_CLOSEDNESS_SUM, _STRADDLE_CLOSEDNESS_LAST
    global _STRADDLE_CLOSEDNESS_TIGHT_SUM, _STRADDLE_CLOSEDNESS_TIGHT_LAST, _STRADDLE_CLOSEDNESS_EP_MAX
    global _STRADDLE_DIST_L_SUM, _STRADDLE_DIST_R_SUM, _STRADDLE_DIST_L_LAST, _STRADDLE_DIST_R_LAST
    global _STRADDLE_PUSH_PROGRESS_SUM, _STRADDLE_PUSH_PROGRESS_LAST
    global _STRADDLE_PUSH_GATE_OPEN_SUM, _STRADDLE_PUSH_GATE_OPEN_LAST
    global _STRADDLE_LEAD_Y_SUM, _STRADDLE_LEAD_Y_LAST
    global _STRADDLE_LEAD_VY_SUM, _STRADDLE_LEAD_VY_LAST
    global _STRADDLE_BETWEEN_FINGERS_SUM, _STRADDLE_BETWEEN_FINGERS_LAST
    _push_gripper_debug_ensure_buffers(env)
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    else:
        ids = env_ids.to(device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return {}

    counts = _STRADDLE_DEBUG_STEP_COUNT[ids].to(dtype=torch.float32).clamp(min=1.0)
    gap_left_mean = (_STRADDLE_GAP_LEFT_SUM[ids] / counts).mean()
    gap_right_mean = (_STRADDLE_GAP_RIGHT_SUM[ids] / counts).mean()
    gap_left_live = _STRADDLE_GAP_LEFT_LAST[ids].mean()
    gap_right_live = _STRADDLE_GAP_RIGHT_LAST[ids].mean()
    closedness_mean = (_STRADDLE_CLOSEDNESS_SUM[ids] / counts).mean()
    closedness_live = _STRADDLE_CLOSEDNESS_LAST[ids].mean()
    closedness_tight_mean = (_STRADDLE_CLOSEDNESS_TIGHT_SUM[ids] / counts).mean()
    closedness_tight_live = _STRADDLE_CLOSEDNESS_TIGHT_LAST[ids].mean()
    closedness_ep_max = _STRADDLE_CLOSEDNESS_EP_MAX[ids].mean()
    dist_l_mm_mean = (_STRADDLE_DIST_L_SUM[ids] / counts).mean() * 1000.0
    dist_r_mm_mean = (_STRADDLE_DIST_R_SUM[ids] / counts).mean() * 1000.0
    dist_l_mm_live = _STRADDLE_DIST_L_LAST[ids].mean() * 1000.0
    dist_r_mm_live = _STRADDLE_DIST_R_LAST[ids].mean() * 1000.0
    success_frac = (_STRADDLE_ACHIEVED_STEP_COUNT[ids] / counts).mean()
    push_progress_mean = (_STRADDLE_PUSH_PROGRESS_SUM[ids] / counts).mean()
    push_progress_live = _STRADDLE_PUSH_PROGRESS_LAST[ids].mean()
    push_gate_open_frac = (_STRADDLE_PUSH_GATE_OPEN_SUM[ids] / counts).mean()
    push_gate_open_live = _STRADDLE_PUSH_GATE_OPEN_LAST[ids].mean()
    lead_y_mean = (_STRADDLE_LEAD_Y_SUM[ids] / counts).mean()
    lead_y_live = _STRADDLE_LEAD_Y_LAST[ids].mean()
    lead_vy_mean = (_STRADDLE_LEAD_VY_SUM[ids] / counts).mean()
    lead_vy_live = _STRADDLE_LEAD_VY_LAST[ids].mean()
    between_q_mean = (_STRADDLE_BETWEEN_FINGERS_SUM[ids] / counts).mean()
    between_q_live = _STRADDLE_BETWEEN_FINGERS_LAST[ids].mean()

    _STRADDLE_DEBUG_STEP_COUNT[ids] = 0
    _STRADDLE_GAP_LEFT_SUM[ids] = 0.0
    _STRADDLE_GAP_RIGHT_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_TIGHT_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_EP_MAX[ids] = 0.0
    _STRADDLE_DIST_L_SUM[ids] = 0.0
    _STRADDLE_DIST_R_SUM[ids] = 0.0
    _STRADDLE_ACHIEVED_STEP_COUNT[ids] = 0
    _STRADDLE_PUSH_PROGRESS_SUM[ids] = 0.0
    _STRADDLE_PUSH_GATE_OPEN_SUM[ids] = 0.0
    _STRADDLE_LEAD_Y_SUM[ids] = 0.0
    _STRADDLE_LEAD_VY_SUM[ids] = 0.0
    _STRADDLE_BETWEEN_FINGERS_SUM[ids] = 0.0

    return {
        # Proximity σ — same as ``finger_proximity`` reward (rises during approach).
        "closedness_mean": float(closedness_mean.item()),
        "closedness_live": float(closedness_live.item()),
        "closedness_ep_max": float(closedness_ep_max.item()),
        # Tight success σ (20 mm) — stricter placement at ±20 mm width targets.
        "closedness_tight_mean": float(closedness_tight_mean.item()),
        "closedness_tight_live": float(closedness_tight_live.item()),
        "dist_l_mm_mean": float(dist_l_mm_mean.item()),
        "dist_r_mm_mean": float(dist_r_mm_mean.item()),
        "dist_l_mm_live": float(dist_l_mm_live.item()),
        "dist_r_mm_live": float(dist_r_mm_live.item()),
        "success_frac": float(success_frac.item()),
        "gap_left_m_mean": float(gap_left_mean.item()),
        "gap_right_m_mean": float(gap_right_mean.item()),
        "gap_left_m_live": float(gap_left_live.item()),
        "gap_right_m_live": float(gap_right_live.item()),
        "push_progress_mean": float(push_progress_mean.item()),
        "push_progress_live": float(push_progress_live.item()),
        "push_gate_open_frac": float(push_gate_open_frac.item()),
        "push_gate_open_live": float(push_gate_open_live.item()),
        "lead_y_env_mean": float(lead_y_mean.item()),
        "lead_y_env_live": float(lead_y_live.item()),
        "lead_vy_mean": float(lead_vy_mean.item()),
        "lead_vy_live": float(lead_vy_live.item()),
        "between_fingers_q_mean": float(between_q_mean.item()),
        "between_fingers_q_live": float(between_q_live.item()),
    }




def _trailing_edge_finger_pcb_width_gaps(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_width_m: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jaw-axis clearances (m) from trailing-edge centre to each pad (joint-based span)."""
    del pcb_half_width_m
    return _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
    )


def _straddle_jaw_span_gap_quality(
    gap_left: torch.Tensor,
    gap_right: torch.Tensor,
    target_jaw_span_m: float,
    target_left_m: float,
    target_right_m: float,
    jaw_span_sigma_m: float,
    gap_sigma_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(span_q, gap_q, span_q * gap_q)`` for asymmetric open-straddle shaping."""
    jaw_span = gap_left + gap_right
    span_sig = float(jaw_span_sigma_m) + 1e-9
    span_q = torch.exp(-torch.abs(jaw_span - float(target_jaw_span_m)) / span_sig)
    gap_q = _asymmetric_width_gap_quality(
        gap_left, gap_right, float(target_left_m), float(target_right_m), float(gap_sigma_m)
    )
    return span_q, gap_q, span_q * gap_q


def straddle_predeep_open_gap_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    target_jaw_span_m: float,
    width_gap_target_left_m: float,
    width_gap_target_right_m: float,
    jaw_span_sigma_m: float,
    width_gap_sigma_m: float,
    predeep_along_gate_m: float = 0.0,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    height_gate_std_m: float | None = None,
) -> torch.Tensor:
    """Reward open asymmetric span + gaps while approaching **before** deep along (``along < gate``).

    Encourages the policy to reach ``target_jaw_span_m`` (e.g. 20 mm) with the correct
    left/right clearances while still behind the trailing face, so ``jaw_along_deep`` can
    take over only after span and gap are aligned.
    """
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg,
    )
    along_min = torch.minimum(geom["along_l"], geom["along_r"])
    mask = (along_min < float(predeep_along_gate_m)).to(geom["along_l"].dtype)
    gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
    )
    _, _, quality = _straddle_jaw_span_gap_quality(
        gap_left,
        gap_right,
        float(target_jaw_span_m),
        float(width_gap_target_left_m),
        float(width_gap_target_right_m),
        float(jaw_span_sigma_m),
        float(width_gap_sigma_m),
    )
    rew = mask * quality
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        rew = rew * height_gate
    return rew


def straddle_lateral_gap_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    width_gap_target_left_m: float,
    width_gap_target_right_m: float,
    width_gap_sigma_m: float,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Dense reward for symmetric jaw-axis clearances (±20 mm at trailing edge, 40 mm span)."""
    gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return _asymmetric_width_gap_quality(
        gap_left,
        gap_right,
        float(width_gap_target_left_m),
        float(width_gap_target_right_m),
        float(width_gap_sigma_m),
    )


def _asymmetric_width_gap_quality(
    gap_left: torch.Tensor,
    gap_right: torch.Tensor,
    target_left_m: float,
    target_right_m: float,
    sigma_m: float,
) -> torch.Tensor:
    """Soft quality in ``[0, 1]`` when each side matches its target lateral clearance."""
    sig = float(sigma_m) + 1e-9
    ql = torch.exp(-torch.abs(gap_left - float(target_left_m)) / sig)
    qr = torch.exp(-torch.abs(gap_right - float(target_right_m)) / sig)
    return torch.sqrt(ql * qr)


def pcb_between_gripper_fingers(
    env: ManagerBasedRLEnv,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
    width_sigma_m: float = 0.010,
    min_along_m: float = 0.0,
    min_straddle_sep_m: float = 0.0005,
    width_weight: float = 3.0,
    jaw_thick_gate_std_m: float | None = 0.003,
    ready_aligned_graspable: bool = False,
    pcb_half_width_m: float | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    width_gap_sigma_m: float | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
) -> torch.Tensor:
    """Straddle-quality reward: how well the PCB is positioned between the jaws (position only).

    Four multiplicative geometric factors (no gripper closedness):

    1. ``is_graspable``  — hard gate: ``jaws_past_edge`` only when ``ready_aligned_graspable``;
                           otherwise Z-straddle × height factor × jaws past trailing face.
    2. ``between_jaws``  — hard gate: PCB centre within jaw-span projection **and** jaw span
                           wide enough to physically contain the board (not closed in empty air).
    3. ``prox``          — ``exp(-max(dist_l, dist_r)/σ)`` — both jaws near trailing-edge
                           targets (one-sided along). σ=proximity_sigma_m spans the approach.
    4. ``width_centre``  — symmetric ``exp(-|mean_Y_err|/σ_w)``, **or** asymmetric lateral
                           gap quality when ``width_gap_target_*`` are set (trailing-edge
                           centre biased toward ``gripper_right`` with unequal clearances).
    """
    if gripper_joint_cfg is not None:
        left, right = gripper_jaw_pad_tips_world(
            env,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
        )
    else:
        left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
        if float(tip_offset_m) > 0.0:
            robot = env.scene[left_finger_cfg.name]
            fwd = _gripper_tip_offset_direction_w(robot, left, right, wrist_body_cfg)
            off = float(tip_offset_m) * fwd
            left = left + off
            right = right + off

    geom = _fingers_trailing_edge_geometry(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg,
        half_length_m, pcb_half_thickness_m,
        width_weight, wrist_body_cfg=wrist_body_cfg,
    )

    along_min = float(min_along_m)
    jaws_past_edge = (geom["along_l"] >= along_min) & (geom["along_r"] >= along_min)
    if ready_aligned_graspable:
        is_graspable = jaws_past_edge.to(left.dtype)
    elif (
        width_gap_target_left_m is not None
        and width_gap_target_right_m is not None
        and gripper_joint_cfg is not None
    ):
        is_graspable = (
            _width_straddle_ready_mask(
                env, pcb_cfg, left, right, min_straddle_sep_m, min_span_frac
            )
            & jaws_past_edge
        ).to(left.dtype)
    else:
        w_left, w_right = _finger_thickness_offsets(
            env, pcb_cfg, left_finger_cfg, right_finger_cfg, wrist_body_cfg
        )
        straddle_gap = torch.abs(w_left - w_right)
        separated = straddle_gap >= float(min_straddle_sep_m)
        z_straddled = (w_left * w_right < 0.0) & separated
        is_graspable = (z_straddled & jaws_past_edge).to(left.dtype)

    # Factor 2: PCB centre in jaw span along thickness — requires open span, not closed-in-air.
    between_jaws = _pcb_between_jaws_mask(
        env, pcb_cfg, left, right, min_span_frac, min_straddle_sep_m
    ).to(left.dtype)

    # Factor 3: per-finger proximity to trailing-edge ±half-thickness targets (real board faces).
    sig = float(proximity_sigma_m) + 1e-9
    dist_l_os, dist_r_os = _one_sided_trailing_finger_dists(geom, width_weight=width_weight)
    prox = torch.exp(-torch.maximum(dist_l_os, dist_r_os) / sig)

    # Factor 4: width shaping along PCB short-edge axis (body +Y).
    if (
        pcb_half_width_m is not None
        and width_gap_target_left_m is not None
        and width_gap_target_right_m is not None
        and gripper_joint_cfg is not None
    ):
        gap_l, gap_r = _trailing_edge_finger_pcb_width_gaps(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            float(pcb_half_width_m),
            wrist_body_cfg,
        )
        gap_sig = float(width_gap_sigma_m if width_gap_sigma_m is not None else width_sigma_m)
        width_centre = _asymmetric_width_gap_quality(
            gap_l,
            gap_r,
            float(width_gap_target_left_m),
            float(width_gap_target_right_m),
            gap_sig,
        )
    else:
        mean_width_err = 0.5 * (geom["width_l"] + geom["width_r"])
        wsig = float(width_sigma_m) + 1e-9
        width_centre = torch.exp(-torch.abs(mean_width_err) / wsig)

    return is_graspable * between_jaws * prox * width_centre


def pcb_finger_object_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.00125,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Per-finger tanh proximity to opposite-side trailing-edge targets.

    Ungated dense gradient (active from far away) so each jaw is pulled toward its own target —
    left → trailing-edge centre ``+`` half-thickness, right → centre ``-`` half-thickness. The two
    targets sit on opposite faces of the board, so maximising this term already steers the jaws
    toward a straddle without any hard ``between`` gate.
    """
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
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


def _straddle_width_face_targets(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trailing short-edge face targets at mid-thickness, ±offset along body +Y (width).

    Both targets lie on the trailing face centre plane (not PCB top/bottom faces) so pads
    straddle the 78.5 mm edge at mid-height for a forward push.
    """
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    off = float(finger_offset_m)
    return center + y_w * off, center - y_w * off, center


def _straddle_width_target_tip_dists(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-jaw distance (m) to trailing-face ±width targets plus mid-thickness offsets.

    Returns ``(dist_l, dist_r, along_l_mm, along_r_mm, thick_l_mm, thick_r_mm)`` for debug.
    Uses symmetric PCB-frame error to each target (penalises pads on the PCB top as well as
    behind the trailing face).
    """
    left_tgt, right_tgt, center = _straddle_width_face_targets(
        env, pcb_cfg, half_length_m, finger_offset_m
    )
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )

    def _frame_dists(tip: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = tip - tgt
        along = torch.sum(delta * x_w, dim=-1)
        width = torch.sum(delta * y_w, dim=-1)
        thick_tgt = torch.sum(delta * z_w, dim=-1)
        thick_mid = torch.sum((tip - center) * z_w, dim=-1)
        dist = torch.sqrt(along * along + width * width + thick_tgt * thick_tgt + 1e-6)
        return dist, along, thick_tgt, thick_mid

    dist_l, along_l, thick_l_tgt, thick_l_mid = _frame_dists(left, left_tgt)
    dist_r, along_r, thick_r_tgt, thick_r_mid = _frame_dists(right, right_tgt)
    return dist_l, dist_r, along_l, along_r, thick_l_mid, thick_r_mid


def straddle_finger_target_closedness(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Composite closedness in ``[0, 1]`` from per-jaw distance to trailing-edge ±offset targets.

    Each jaw: ``q = 1 - tanh(dist / std)``.  Returns ``0.5 * (q_left + q_right)``.
    Distances use **contact pad tips** (body origin + ``tip_offset_m`` distal), not link centres.
    """
    lfinger_dist, rfinger_dist, _, _, _, _ = _straddle_width_target_tip_dists(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    left_q = 1.0 - torch.tanh(lfinger_dist / sig)
    right_q = 1.0 - torch.tanh(rfinger_dist / sig)
    return 0.5 * (left_q + right_q)


def straddle_tip_mid_thickness_shaping(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Pull each contact pad tip to the PCB mid-thickness plane (not the finger-body centre)."""
    _, _, _, _, thick_l, thick_r = _straddle_width_target_tip_dists(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    left_q = 1.0 - torch.tanh(torch.abs(thick_l) / sig)
    right_q = 1.0 - torch.tanh(torch.abs(thick_r) / sig)
    return 0.5 * (left_q + right_q)


def straddle_trailing_face_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    height_gate_std_m: float | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """One-sided reward for pads advancing to the trailing short-edge face (width straddle).

    Unlike :func:`jaw_along_approach_reward` (grasp targets on ±thickness faces), both pads
    should reach the trailing **centre plane** so they can flank the 78.5 mm edge at mid-height.
    """
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )

    def _behind_face_err(tip: torch.Tensor) -> torch.Tensor:
        along = torch.sum((tip - center) * x_w, dim=-1)
        return torch.clamp(-along, min=0.0)

    sig = float(std) + 1e-9
    left_rew = 1.0 - torch.tanh(_behind_face_err(left) / sig)
    right_rew = 1.0 - torch.tanh(_behind_face_err(right) / sig)
    rew = torch.minimum(left_rew, right_rew)
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        rew = rew * height_gate
    return rew


def straddle_trailing_face_overshoot_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    overshoot_std_m: float,
    target_along_m: float = 0.0,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Shaping in ``[0, 1]`` that decays when pads move past the trailing face (overshoot).

    Signed along offset vs trailing-face centre (body +X / push axis): negative = behind the
    face, ``target_along_m`` = desired stop (0 = on the face).  Returns
    ``min_jaw (1 - tanh(relu(along - target) / overshoot_std))`` so credit falls once either
    pad crosses the face toward the slot (+Y).  Pair with :func:`straddle_trailing_face_approach_reward`
    or use :func:`straddle_trailing_face_bounded_approach_reward` for a single bell-shaped term.
    """
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(overshoot_std_m) + 1e-9
    tgt = float(target_along_m)

    def _overshoot_factor(tip: torch.Tensor) -> torch.Tensor:
        along = torch.sum((tip - center) * x_w, dim=-1)
        overshoot_err = torch.clamp(along - tgt, min=0.0)
        return 1.0 - torch.tanh(overshoot_err / sig)

    return torch.minimum(_overshoot_factor(left), _overshoot_factor(right))


def straddle_trailing_face_bounded_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    approach_std_m: float,
    overshoot_std_m: float,
    target_along_m: float = 0.0,
    height_gate_std_m: float | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Bell-shaped trailing-face approach: rises from behind, peaks at ``target_along_m``, decays past."""
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    along_l = torch.sum((left - center) * x_w, dim=-1)
    along_r = torch.sum((right - center) * x_w, dim=-1)
    left_rew = _jaw_along_deep_single_reward(along_l, target_along_m, approach_std_m, overshoot_std_m)
    right_rew = _jaw_along_deep_single_reward(along_r, target_along_m, approach_std_m, overshoot_std_m)
    rew = torch.minimum(left_rew, right_rew)
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        rew = rew * height_gate
    return rew


def straddle_finger_target_success(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float = 0.020,
    closedness_threshold: float = 0.5,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Episode success when :func:`straddle_finger_target_closedness` ≥ ``closedness_threshold``."""
    closedness = straddle_finger_target_closedness(
        env,
        std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return closedness >= float(closedness_threshold)


def straddle_finger_trailing_width_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Per-jaw proximity reward — same closedness index as straddle success (dense shaping)."""
    return straddle_finger_target_closedness(
        env,
        std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )


def pcb_midpoint_trailing_edge_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
    midpoint_jaw_offset_m: float = 0.0,
) -> torch.Tensor:
    """Midpoint-to-trailing-edge proximity: ``1 - tanh(edge_dist / std)``.

    Uses the **gripper midpoint** vs the PCB trailing short-edge target (face centre plus optional
    ``midpoint_jaw_offset_m`` toward ``gripper_right`` along body −Z when jaws are thickness-aligned).

    Contrast with :func:`pcb_finger_object_proximity` which targets each jaw to ±half_thickness —
    that per-jaw Z gradient inadvertently pulls the jaws *closed* during the approach phase.
    """
    target_offset_w = straddle_midpoint_target_offset_w(env, pcb_cfg, midpoint_jaw_offset_m)
    _, _, _, _, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        target_offset_w=target_offset_w,
        wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    return 1.0 - torch.tanh(edge_dist / sig)


def jaw_thickness_height_alignment(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Dense reward for each jaw tip being at its correct HEIGHT target along the PCB thickness axis.

    Left jaw target: PCB top face  (+half_thickness from PCB centre).
    Right jaw target: PCB bottom face (-half_thickness from PCB centre).

    Only the thickness (Z-body) component of each jaw's position error is used — lateral (along / width)
    errors are ignored.  This creates an independent gradient that pulls each jaw to the correct
    height BEFORE lateral approach, preventing the lower jaw from approaching from above and becoming
    blocked by the PCB top surface.

    The reward is the minimum over both jaws (tanh-softened) so both must reach the correct height.
    ``std`` should be comparable to PCB half-thickness (e.g. 0.003 m for a 1 mm PCB) so that a
    0.5 mm error gives a meaningful reward difference from the correctly-positioned state.
    A std of 0.02 m is too loose — tanh(0.5 mm / 20 mm) ≈ 0.025, giving near-zero gradient.
    """
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    left_rew = 1.0 - torch.tanh(torch.abs(geom["thick_l"]) / sig)
    right_rew = 1.0 - torch.tanh(torch.abs(geom["thick_r"]) / sig)
    return torch.minimum(left_rew, right_rew)


def _gripper_jaw_axis_pcb_body_alignment(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    body_axis_fn,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cosine-similarity reward: jaw-separation vector aligned with a PCB body axis."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    jaw_vec = left - right
    jaw_norm = jaw_vec / jaw_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    axis_w = body_axis_fn(env, pcb_cfg)
    return torch.abs(torch.sum(jaw_norm * axis_w, dim=-1))


def gripper_jaw_axis_pcb_z_alignment(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cosine-similarity reward: jaw-separation vector aligned with PCB thickness (Z-body) axis."""
    return _gripper_jaw_axis_pcb_body_alignment(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, pcb_body_axis_z_world, wrist_body_cfg
    )


def gripper_jaw_axis_pcb_y_alignment(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cosine-similarity reward: jaw axis parallel to PCB body +Y (78.5 mm trailing short edge)."""
    return _gripper_jaw_axis_pcb_body_alignment(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, pcb_body_axis_y_world, wrist_body_cfg
    )


def gripper_midpoint_pcb_center_height(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Dense reward for the gripper midpoint Z height matching the PCB trailing-edge centre Z.

    The midpoint of the two jaw tips should be at the same height as the PCB centre (Z-body axis)
    so that, when the jaws are opened correctly along Z, one jaw lands above and one below the PCB.
    If the midpoint is too high, the lower jaw hits the PCB top face before being able to slide
    under it.

    Uses the PCB-body Z projection so it works even if the PCB is slightly tilted.
    ``std`` should be chosen to provide gradient from the typical approach distance (e.g. 0.02 m
    for 2 cm far-field shaping).  This is the far-field complement to ``jaw_thickness_height_alignment``
    (which is local, std ≈ PCB half-thickness) and active from any arm height during approach.
    """
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    midpoint = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    dz = torch.abs(torch.sum((midpoint - center) * z_w, dim=-1))
    return 1.0 - torch.tanh(dz / (float(std) + 1e-9))


def jaw_along_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    height_gate_std_m: float | None = None,
) -> torch.Tensor:
    """Reward for jaw tips advancing along the PCB long axis to reach the trailing edge.

    ``along_l / along_r`` is the signed error of each jaw tip vs its target along the PCB body
    X axis (push / slide direction ≈ world Y).  Negative values mean the jaw has not yet
    reached the trailing edge; zero means it is exactly at the edge; positive is impossible
    in practice (PCB collision blocks the jaw from passing through the board face).

    This is a ONE-SIDED reward: only penalises being BEHIND the trailing edge (along < 0).
    Full reward (1.0) is given when the jaw tip has reached the trailing edge (along ≥ 0).
    The reward is the minimum over both jaws so BOTH must advance together.

    ``std`` sets the length-scale; 0.05 m (5 cm) gives gradient from ~15 cm behind the edge —
    active from the very start of the episode so the arm has a continuous incentive to
    advance in +Y toward the PCB, independently of the Z-height and orientation corrections.
    """
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    # clamp(min=0): zero error when jaw is at or past trailing edge; positive error when behind.
    left_err = torch.clamp(-geom["along_l"], min=0.0)
    right_err = torch.clamp(-geom["along_r"], min=0.0)
    left_rew = 1.0 - torch.tanh(left_err / sig)
    right_rew = 1.0 - torch.tanh(right_err / sig)
    along_rew = torch.minimum(left_rew, right_rew)
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        along_rew = along_rew * height_gate
    return along_rew


def _jaw_along_deep_single_reward(
    along: torch.Tensor,
    target: float,
    approach_std_m: float,
    overshoot_std_m: float,
) -> torch.Tensor:
    """Bell-ish along reward: rises toward ``target``, peaks there, decays when past (overshoot)."""
    app_sig = float(approach_std_m) + 1e-9
    over_sig = float(overshoot_std_m) + 1e-9
    behind_err = torch.clamp(float(target) - along, min=0.0)
    approach_rew = 1.0 - torch.tanh(behind_err / app_sig)
    overshoot_err = torch.clamp(along - float(target), min=0.0)
    overshoot_factor = 1.0 - torch.tanh(overshoot_err / over_sig)
    return approach_rew * overshoot_factor


def jaw_along_deep_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    target_along_m: float,
    std: float,
    pcb_half_thickness_m: float = 0.0005,
    wrist_body_cfg: SceneEntityCfg | None = None,
    overshoot_std_m: float = 0.005,
    height_gate_std_m: float | None = None,
    predeep_jaw_span_m: float | None = None,
    predeep_gap_left_m: float | None = None,
    predeep_gap_right_m: float | None = None,
    predeep_span_sigma_m: float = 0.004,
    predeep_gap_sigma_m: float = 0.004,
    gripper_joint_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Deep along reward: peak at ``target_along_m`` past the trailing face, decay beyond (overshoot).

    Before the target the reward rises one-sided (same as legacy deep approach).  Past the target
    an overshoot factor ``1 - tanh((along - target) / overshoot_std_m)`` decays credit so the
    policy does not keep driving into the PCB after straddle.  Both jaws must satisfy (min).

    When ``predeep_jaw_span_m`` is set, credit is multiplied by soft span+gap quality so deep
    advance only pays after the 20 mm asymmetric open is established during pre-deep approach.
    """
    geom = _fingers_trailing_edge_geometry(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        1.0,
        wrist_body_cfg,
    )
    target = float(target_along_m)
    left_rew = _jaw_along_deep_single_reward(geom["along_l"], target, std, overshoot_std_m)
    right_rew = _jaw_along_deep_single_reward(geom["along_r"], target, std, overshoot_std_m)
    deep_rew = torch.minimum(left_rew, right_rew)
    if (
        predeep_jaw_span_m is not None
        and predeep_gap_left_m is not None
        and predeep_gap_right_m is not None
        and gripper_joint_cfg is not None
    ):
        gap_left, gap_right = _trailing_edge_jaw_opening_gaps(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            wrist_body_cfg,
        )
        _, _, predeep_q = _straddle_jaw_span_gap_quality(
            gap_left,
            gap_right,
            float(predeep_jaw_span_m),
            float(predeep_gap_left_m),
            float(predeep_gap_right_m),
            float(predeep_span_sigma_m),
            float(predeep_gap_sigma_m),
        )
        deep_rew = deep_rew * predeep_q
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        deep_rew = deep_rew * height_gate
    return deep_rew


def gripper_midpoint_along_approach_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    std: float,
    wrist_body_cfg: SceneEntityCfg | None = None,
    height_gate_std_m: float | None = None,
    midpoint_jaw_offset_m: float = 0.0,
) -> torch.Tensor:
    """Reward for the gripper midpoint advancing along the PCB long axis to the trailing edge.

    Complements ``jaw_along_approach_reward`` (per-jaw tips): the arm often moves the wrist
    midpoint first while jaws are still open, so this provides a direct +Y advance signal before
    the jaw tips reach the trailing-edge face targets.
    """
    target_offset_w = straddle_midpoint_target_offset_w(env, pcb_cfg, midpoint_jaw_offset_m)
    along, _, _, _, _ = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        target_offset_w=target_offset_w,
        wrist_body_cfg=wrist_body_cfg,
    )
    sig = float(std) + 1e-9
    err = torch.clamp(-along, min=0.0)
    along_rew = 1.0 - torch.tanh(err / sig)
    if height_gate_std_m is not None:
        height_gate = gripper_midpoint_pcb_center_height(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            half_length_m,
            float(height_gate_std_m),
            wrist_body_cfg,
        )
        along_rew = along_rew * height_gate
    return along_rew




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
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Obs: signed offset along PCB thickness axis (jaw mid vs board center), scaled to ~[-1, 1]."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
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
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Obs: jaw-mid error vs trailing short-edge centre in PCB frame, scaled to ~[-1, 1].

    Components are ``along`` (long axis), ``width`` (short edge), ``thick`` (board thickness).
    """
    along, width, thick, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
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
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Shaping ``[0, 1]``: reward jaw rail parallel to world +Z (top/bottom thickness close).

    Carriage must **not** stay level (∥ XY); the left↔right rail is rolled vertical so fingers
    straddle PCB thickness.
    """
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    yaw_only: bool = True,
    max_pitch_deg: float | None = None,
    pitch_soft_deg: float = 10.0,
) -> torch.Tensor:
    """Shaping ``[0, 1]``: wrist (``link_6``) → carriage mid aligned with push axis in XY (yaw).

    With ``yaw_only=True`` (default), only the horizontal (+Y) heading is rewarded. When
    ``max_pitch_deg`` is set, wrist pitch is softly limited (not fully free, not full 3D lock).
    With ``yaw_only=False``, full 3D alignment with the push axis is used.
    """
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
        wrist_body_cfg,
        min_finger_sep_m,
        left,
        right,
    )
    if yaw_only:
        if max_pitch_deg is not None:
            wc_align = gripper_wrist_carriage_yaw_pitch_limited_align(
                env,
                left_finger_cfg,
                right_finger_cfg,
                wrist_body_cfg,
                push_axis_world,
                max_pitch_deg,
                pitch_soft_deg,
            )
        else:
            wc_align = gripper_wrist_carriage_yaw_align_axis(
                env,
                left_finger_cfg,
                right_finger_cfg,
                wrist_body_cfg,
        push_axis_world,
            )
    else:
        wc_align = gripper_wrist_carriage_align_axis(
            env,
            left_finger_cfg,
            right_finger_cfg,
            wrist_body_cfg,
            push_axis_world,
        )
    return wc_align * sep_ok * near


def gripper_jaw_rail_horizontal_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
    width_weight: float = 3.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalty ``[0, 1]``: jaw rail lying in the XY plane (level carriage / width-pinch pose)."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Combined orientation: ``jaw_rail_vertical * wrist_carriage_push`` (legacy single term)."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    near, sep_ok = _gripper_top_bottom_near_gate(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        gate_dist_m,
        width_weight,
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
        push_axis_world,
    )
    return rail_z * wc_y * sep_ok * near


def gripper_pinch_orientation_cos_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    min_finger_sep_m: float = 0.006,
    wrist_body_cfg: SceneEntityCfg | None = None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    yaw_only_wrist_align: bool = False,
) -> torch.Tensor:
    """Two scalars in ``[0, 1]``: jaw rail ∥ +Z; wrist→carriage push alignment; sep-scaled."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    _, n = _gripper_rail_unit_lr(left, right)
    rail_z = gripper_rail_align_world_z(env, left, right)
    if yaw_only_wrist_align:
        wc_y = gripper_wrist_carriage_yaw_align_axis(
            env,
            left_finger_cfg,
            right_finger_cfg,
            wrist_body_cfg,
        push_axis_world,
        )
    else:
        wc_y = gripper_wrist_carriage_align_axis(
            env,
            left_finger_cfg,
            right_finger_cfg,
            wrist_body_cfg,
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


def slide_mouth_lead_y_reached(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    min_lead_y_env: float,
) -> torch.Tensor:
    """True when the PCB leading short-edge centre Y ≥ ``min_lead_y_env`` (env-local)."""
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    return lead_y >= float(min_lead_y_env)


def slide_mouth_leading_edge_near_xy(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_x_env: float,
    slot_mouth_y_env: float,
    tolerance_m: float = 0.010,
) -> torch.Tensor:
    """True when the PCB leading short-edge centre is within ``tolerance_m`` of slot mouth (X, Y)."""
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_env = lead_w - env.scene.env_origins[:, :3]
    dx = torch.abs(lead_env[:, 0] - float(slot_mouth_x_env))
    dy = torch.abs(lead_env[:, 1] - float(slot_mouth_y_env))
    tol = float(tolerance_m)
    return (dx <= tol) & (dy <= tol)


def slide_mouth_reached(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    margin_m: float = 0.008,
    slot_mouth_x_env: float | None = None,
    tolerance_m: float | None = None,
) -> torch.Tensor:
    """Legacy Y-only mouth check; prefer :func:`slide_mouth_leading_edge_near_xy`."""
    if slot_mouth_x_env is not None and tolerance_m is not None:
        return slide_mouth_leading_edge_near_xy(
            env,
            pcb_cfg,
            half_length_m,
            slot_mouth_x_env,
            slot_mouth_y_env,
            tolerance_m=tolerance_m,
        )
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    return lead_y >= float(slot_mouth_y_env) - float(margin_m)


def _slide_success_sustain_count_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Consecutive control steps satisfying mouth + gripper + low-speed criteria."""
    if not hasattr(env, "_slide_success_sustain_count"):
        env._slide_success_sustain_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    count = env._slide_success_sustain_count
    if count.shape[0] != env.num_envs or count.device != env.device:
        count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._slide_success_sustain_count = count
    return count


def _advance_slide_success_sustain(
    env: ManagerBasedRLEnv,
    frame: torch.Tensor,
    min_sustained_steps: int,
) -> torch.Tensor:
    """Update sustain counter once per control step; return success mask."""
    count = _slide_success_sustain_count_tensor(env)
    first_step = env.episode_length_buf == 1

    if not hasattr(env, "_slide_success_sustain_step_buf"):
        env._slide_success_sustain_step_buf = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
    step_buf = env._slide_success_sustain_step_buf
    if step_buf.shape[0] != env.num_envs or step_buf.device != env.device:
        step_buf = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._slide_success_sustain_step_buf = step_buf

    ep_step = env.episode_length_buf
    already_counted = step_buf == ep_step
    new_count = torch.where(first_step | (~frame), torch.zeros_like(count), count + 1)
    count = torch.where(already_counted, count, new_count)
    env._slide_success_sustain_count = count
    env._slide_success_sustain_step_buf = torch.where(already_counted, step_buf, ep_step)

    sustained = int(min_sustained_steps)
    if sustained <= 1:
        return frame
    return count >= sustained


def slide_leading_edge_in_target_xy_range(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xy_env: tuple[float, float],
    tolerance_xy_m: tuple[float, float] = (0.003, 0.020),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """True when leading short-edge centre X/Y are inside ``target_lead_xy_env ± tolerance``."""
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    tgt = torch.tensor(target_lead_xy_env, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0)
    tol = torch.tensor(tolerance_xy_m, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0)
    delta = torch.abs(lead_env[:, :2] - tgt)
    return (delta[:, 0] <= tol[:, 0]) & (delta[:, 1] <= tol[:, 1])


def slide_leading_edge_in_target_xyz_range(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xyz_env: tuple[float, float, float],
    tolerance_xyz_m: tuple[float, float, float] = (0.003, 0.020, 0.003),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """True when the leading short-edge centre is inside a per-axis box around ``target_lead_xyz_env``."""
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    tgt = torch.tensor(target_lead_xyz_env, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0)
    tol = torch.tensor(tolerance_xyz_m, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0)
    delta = torch.abs(lead_env - tgt)
    return (delta[:, 0] <= tol[:, 0]) & (delta[:, 1] <= tol[:, 1]) & (delta[:, 2] <= tol[:, 2])


def _slide_success_gripper_closed_ok(
    env: ManagerBasedRLEnv,
    gripper_joint_cfg: SceneEntityCfg | None,
    max_gripper_gap_m: float,
    require_gripper_closed: bool,
) -> torch.Tensor:
    """True when ``left_carriage_joint`` gap is below ``max_gripper_gap_m``."""
    if not require_gripper_closed or gripper_joint_cfg is None:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    return gq < float(max_gripper_gap_m)


def _slide_success_in_range(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xy_env: tuple[float, float],
    tolerance_xy_m: tuple[float, float],
    min_episode_steps: int,
    axis_world: tuple[float, float, float],
    gripper_joint_cfg: SceneEntityCfg | None,
    max_gripper_gap_m: float,
    require_gripper_closed: bool,
) -> torch.Tensor:
    """Shared mask for ``slide_success`` termination and ``slide_success_bonus`` reward."""
    in_range = slide_leading_edge_in_target_xy_range(
        env,
        pcb_cfg,
        half_length_m,
        target_lead_xy_env,
        tolerance_xy_m,
        axis_world,
    )
    gripper_ok = _slide_success_gripper_closed_ok(
        env, gripper_joint_cfg, max_gripper_gap_m, require_gripper_closed
    )
    in_range = in_range & gripper_ok
    if min_episode_steps > 0:
        in_range = in_range & (env.episode_length_buf > min_episode_steps)
    return in_range


def slide_success(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xy_env: tuple[float, float] = (0.056, 0.190),
    tolerance_xy_m: tuple[float, float] = (0.003, 0.020),
    min_episode_steps: int = 0,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    max_gripper_gap_m: float = 0.0015,
    require_gripper_closed: bool = True,
) -> torch.Tensor:
    """Slide success: leading-edge X/Y in target box and gripper closed (no Z / velocity gates)."""
    return _slide_success_in_range(
        env,
        pcb_cfg,
        half_length_m,
        target_lead_xy_env,
        tolerance_xy_m,
        min_episode_steps,
        axis_world,
        gripper_joint_cfg,
        max_gripper_gap_m,
        require_gripper_closed,
    )


def _slide_lead_pose_ok(
    env: ManagerBasedRLEnv,
    lead_env: torch.Tensor,
    max_lead_x_drift_m: float,
    belt_center_z_env: float,
    max_lead_z_drift_m: float,
) -> torch.Tensor:
    """True when leading-edge X stays near spawn and Z is near belt-top centre height."""
    if hasattr(env, "_insert_start_lead_env"):
        delta_x = lead_env[:, 0] - env._insert_start_lead_env[:, 0]
        x_ok = torch.abs(delta_x) <= float(max_lead_x_drift_m)
    else:
        x_ok = torch.ones(lead_env.shape[0], device=lead_env.device, dtype=torch.bool)
    ref_z = float(belt_center_z_env)
    z_ok = torch.abs(lead_env[:, 2] - ref_z) <= float(max_lead_z_drift_m)
    return x_ok & z_ok


def slide_leading_edge_travel_milestone_bonus(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
    milestone_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    state_attr: str = "_slide_travel_milestone_paid",
    max_lead_x_drift_m: float = 0.003,
    belt_center_z_env: float = 0.10175,
    max_lead_z_drift_m: float = 0.005,
) -> torch.Tensor:
    """One-shot sparse bonus each time leading-edge +Y travel crosses a milestone fraction.

    Progress is measured from ``env._insert_start_lead_proj`` (set at slide reset) to
    ``target_lead_y_env``.  Milestone credit requires leading-edge lane X and belt Z pose
    (flat push on the conveyor).  Returns the count of newly crossed tiers this step.
    """
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    pose_ok = _slide_lead_pose_ok(
        env, lead_env, max_lead_x_drift_m, belt_center_z_env, max_lead_z_drift_m
    )
    a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)

    if hasattr(env, "_insert_start_lead_proj"):
        start_proj = env._insert_start_lead_proj
    else:
        start_proj = proj.detach()

    total = float(target_lead_y_env) - start_proj
    valid = total > 1e-6
    total_safe = torch.where(valid, total, torch.ones_like(total))
    frac = torch.where(valid, (proj - start_proj) / total_safe, torch.zeros_like(proj))
    frac = frac.clamp(0.0, 1.0)

    tiers = tuple(milestone_fractions)
    n_tiers = len(tiers)
    if not hasattr(env, state_attr):
        setattr(
            env,
            state_attr,
            torch.zeros(env.num_envs, n_tiers, device=env.device, dtype=torch.bool),
        )
    paid: torch.Tensor = getattr(env, state_attr)
    if paid.shape[0] != env.num_envs or paid.shape[1] != n_tiers:
        paid = torch.zeros(env.num_envs, n_tiers, device=env.device, dtype=torch.bool)
        setattr(env, state_attr, paid)

    bonus = torch.zeros(env.num_envs, device=env.device, dtype=frac.dtype)
    for i, mf in enumerate(tiers):
        crossed = frac >= float(mf)
        newly = crossed & (~paid[:, i]) & pose_ok
        paid[:, i] = paid[:, i] | newly
        bonus = bonus + newly.to(dtype=frac.dtype)
    return bonus


def slide_success_bonus_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xy_env: tuple[float, float] = (0.056, 0.190),
    tolerance_xy_m: tuple[float, float] = (0.003, 0.020),
    min_episode_steps: int = 0,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    max_gripper_gap_m: float = 0.0015,
    require_gripper_closed: bool = True,
) -> torch.Tensor:
    """Bonus (1.0) when leading-edge X/Y box and gripper closed match :func:`slide_success`."""
    achieved = _slide_success_in_range(
        env,
        pcb_cfg,
        half_length_m,
        target_lead_xy_env,
        tolerance_xy_m,
        min_episode_steps,
        axis_world,
        gripper_joint_cfg,
        max_gripper_gap_m,
        require_gripper_closed,
    )
    return achieved.to(dtype=env.scene[pcb_cfg.name].data.root_pos_w.dtype)










def _pcb_yaw_xy_signed_sin_cos(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed yaw ``sin`` and ``|cos|`` in XY between PCB body +X and ``axis_world``."""
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    a = torch.tensor(axis_world, device=x_w.device, dtype=x_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    x_xy = x_w.clone()
    x_xy[:, 2] = 0.0
    x_xy = x_xy / torch.norm(x_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    a_xy = a.clone()
    a_xy[2] = 0.0
    a_xy = a_xy / torch.norm(a_xy).clamp_min(1e-9)
    cos_align = torch.abs(torch.sum(x_xy * a_xy.unsqueeze(0), dim=-1)).clamp(0.0, 1.0)
    sin_yaw = a_xy[0] * x_xy[:, 1] - a_xy[1] * x_xy[:, 0]
    return sin_yaw, cos_align


def slide_pcb_yaw_xy_alignment_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Reward PCB long-axis alignment with the slide (+Y) direction in the horizontal plane."""
    _, cos_align = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    return cos_align


def slide_pcb_yaw_sin_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    scale: float = 0.15,
) -> torch.Tensor:
    """Obs: signed yaw error ``sin(θ)`` between PCB long axis and push axis (XY), scaled."""
    sin_yaw, _ = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    return (sin_yaw / (float(scale) + 1e-9)).unsqueeze(-1)


def slide_finger_push_axis_delta_obs(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    scale_m: float = 0.010,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Obs: left-minus-right finger projection on the push axis (env frame), scaled."""
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    origins = env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=left.device, dtype=left.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    push_l = torch.sum((left - origins) * a.unsqueeze(0), dim=-1)
    push_r = torch.sum((right - origins) * a.unsqueeze(0), dim=-1)
    return ((push_l - push_r) / (float(scale_m) + 1e-9)).unsqueeze(-1)


def slide_finger_push_axis_y_sync_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    yaw_good_cos: float = 0.995,
    sync_std_m: float = 0.004,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward matched finger push-axis depth when yaw is good (symmetric +Y slide).

    When PCB yaw is off, shaping is disabled so the policy may advance one jaw ahead of the
    other for corrective pushing.
    """
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    origins = env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=left.device, dtype=left.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    push_l = torch.sum((left - origins) * a.unsqueeze(0), dim=-1)
    push_r = torch.sum((right - origins) * a.unsqueeze(0), dim=-1)
    sync = torch.exp(-torch.abs(push_l - push_r) / (float(sync_std_m) + 1e-9))
    _, cos_align = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    gate = (cos_align >= float(yaw_good_cos)).to(dtype=sync.dtype)
    return sync * gate


def slide_yaw_corrective_asymmetric_push_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    yaw_bad_cos: float = 0.970,
    asym_std_m: float = 0.006,
    asym_gain_m: float = 0.015,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """When yaw is off, reward finger push-axis differential that opposes the yaw error.

    ``asym = proj_left - proj_right`` along the push axis; desired ``asym ≈ -gain * sin(yaw)``.
    """
    sin_yaw, cos_align = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    origins = env.scene.env_origins[:, :3]
    a = torch.tensor(axis_world, device=left.device, dtype=left.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    push_l = torch.sum((left - origins) * a.unsqueeze(0), dim=-1)
    push_r = torch.sum((right - origins) * a.unsqueeze(0), dim=-1)
    asym = push_l - push_r
    desired = -float(asym_gain_m) * sin_yaw
    quality = torch.exp(-torch.abs(asym - desired) / (float(asym_std_m) + 1e-9))
    active = cos_align < float(yaw_bad_cos)
    return torch.where(active, quality, torch.ones_like(quality))


def slide_gripper_span_yaw_recovery_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    nominal_span_m: float = 0.040,
    max_open_mult: float = 1.35,
    span_sigma_m: float = 0.006,
    yaw_bad_cos: float = 0.970,
) -> torch.Tensor:
    """Nominal 40 mm span when yaw is good; allow wider opening when yaw is bad to re-seat the board."""
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]].clamp(min=0.0)
    span = _gripper_jaw_span_from_joint(gq)
    _, cos_align = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    nominal = float(nominal_span_m)
    yaw_ok = cos_align >= float(yaw_bad_cos)
    ok_rew = torch.exp(-torch.abs(span - nominal) / (float(span_sigma_m) + 1e-9))
    max_open = nominal * float(max_open_mult)
    open_frac = torch.clamp((span - nominal) / (max_open - nominal + 1e-9), 0.0, 1.0)
    bad_rew = open_frac * cos_align
    return torch.where(yaw_ok, ok_rew, bad_rew.clamp(0.0, 1.0))


def gripper_gap_excess_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_gripper_gap_m: float,
) -> torch.Tensor:
    """Penalty when ``left_carriage_joint`` exceeds ``max_gripper_gap_m`` (linear excess)."""
    robot = env.scene[asset_cfg.name]
    gq = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    return torch.relu(gq - float(max_gripper_gap_m))


def pcb_root_center_z_excess_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    reference_z_env: float,
    max_excess_m: float = 0.005,
) -> torch.Tensor:
    """Penalty when PCB root centre Z rises more than ``max_excess_m`` above ``reference_z_env``."""
    pcb = env.scene[pcb_cfg.name]
    z_env = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    ref = torch.full_like(z_env, float(reference_z_env))
    return torch.relu(z_env - ref - float(max_excess_m))








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




def _store_insert_progress_baselines(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float = 0.198,
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
    if not hasattr(env, "_insert_start_lead_env"):
        env._insert_start_lead_env = torch.zeros(env.num_envs, 3, device=device, dtype=dtype)
    env._insert_start_center_env[env_ids] = center_env
    env._insert_start_center_proj[env_ids] = torch.sum(center_env * push.unsqueeze(0), dim=-1)
    env._insert_start_center_z[env_ids] = center_env[:, 2]
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)[env_ids]
    lead_env = lead_w - env.scene.env_origins[env_ids, :3]
    env._insert_start_lead_env[env_ids] = lead_env
    env._insert_start_lead_proj[env_ids] = torch.sum(lead_env * push.unsqueeze(0), dim=-1)
    env._insert_start_lead_z[env_ids] = lead_env[:, 2]

    if not hasattr(env, "_insert_milestone_slot_mouth"):
        env._insert_milestone_slot_mouth = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    env._insert_milestone_slot_mouth[env_ids] = False

    if hasattr(env, "_insert_rail_milestone_mask"):
        env._insert_rail_milestone_mask[env_ids] = False

    if hasattr(env, "_insert_backward_step_count"):
        env._insert_backward_step_count[env_ids] = 0.0

    if not hasattr(env, "_insert_mouth_stall_count"):
        env._insert_mouth_stall_count = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    if not hasattr(env, "_insert_mouth_prev_depth"):
        env._insert_mouth_prev_depth = torch.zeros(env.num_envs, device=device, dtype=dtype)
    env._insert_mouth_stall_count[env_ids] = 0
    penetration = torch.clamp(lead_env[:, 1] - float(slot_mouth_y_env), min=0.0)
    env._insert_mouth_prev_depth[env_ids] = penetration

    if hasattr(env, "_slide_travel_milestone_paid"):
        env._slide_travel_milestone_paid[env_ids] = False

    if hasattr(env, "_slide_success_sustain_count"):
        env._slide_success_sustain_count[env_ids] = 0

    if hasattr(env, "_slide_success_sustain_step_buf"):
        env._slide_success_sustain_step_buf[env_ids] = -1


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


def pcb_gripper_straddled_mask(
    env: ManagerBasedEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    min_straddle_sep_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_span_frac: float = 0.01,
) -> torch.Tensor:
    """True when the PCB is centred in an open width-axis jaw span (trailing-edge push straddle)."""
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    return _width_straddle_ready_mask(
        env, pcb_cfg, left, right, min_straddle_sep_m, min_span_frac
    )


def pcb_gripper_straddle_lost(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    min_straddle_sep_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_episode_steps: int = 0,
) -> torch.Tensor:
    """Terminate when the PCB is no longer straddled between the gripper jaws."""
    lost = ~pcb_gripper_straddled_mask(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        min_straddle_sep_m=min_straddle_sep_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    if min_episode_steps > 0:
        lost = lost & (env.episode_length_buf > min_episode_steps)
    return lost


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
    wrist_body_cfg: SceneEntityCfg | None = None,
    check_straddle: bool = True,
) -> torch.Tensor:
    """True when the PCB is no longer kinematically held by the gripper.

    Uses trailing-edge geometry relative to the gripper midpoint, but with
    **looser** detach thresholds so minor insertion wobble does not false-trigger while a real
    slip / drop does.

    Detach is declared when any of the following hold (after ``min_episode_steps``):
    * jaw midpoint is too far from the trailing short-edge centre;
    * either jaw tip is too far from its trailing-edge grasp target;
    * optionally: the board is no longer straddled (when ``check_straddle`` is True);
    * optional: PCB root height (env-local Z) falls below ``min_height_env``.
    """
    along, _, _, in_plane, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
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
        wrist_body_cfg,
    )
    lost_fingers = torch.maximum(geom["dist_l"], geom["dist_r"]) > float(max_finger_dist_m)
    detached = lost_edge | lost_fingers
    if check_straddle:
        lost_straddle = ~pcb_gripper_straddled_mask(
            env,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            min_straddle_sep_m=min_straddle_sep_m,
            wrist_body_cfg=wrist_body_cfg,
        )
        detached = detached | lost_straddle

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
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
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
        wrist_body_cfg,
    )
    finger_gap = torch.maximum(geom["dist_l"], geom["dist_r"]) > float(max_finger_dist_m)

    *_, edge_dist = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
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
_STRADDLE_STATE_BUFFER: dict | None = None
_STRADDLE_BUFFER_PATH: str | None = None


def _buffer_gripper_closed_mask(buf: dict, max_gripper_gap_m: float) -> torch.Tensor | None:
    """True for buffer rows whose saved carriage joint indicates a closed pinch."""
    names = buf["joint_names"]
    if "left_carriage_joint" not in names:
        return None
    col = list(names).index("left_carriage_joint")
    return buf["joint_pos"][:, col] < float(max_gripper_gap_m)


def _compose_buffer_row_mask(buf: dict, path: str, max_gripper_gap_m: float = 0.001) -> torch.Tensor | None:
    """Row mask for buffer sampling: closed gripper (insert can push without straddle)."""
    return _buffer_gripper_closed_mask(buf, max_gripper_gap_m)


def _load_straddle_state_buffer(path: str, max_gripper_gap_m: float = 0.001) -> dict:
    """Load (or re-use cached) straddle terminal state .npz file."""
    global _STRADDLE_STATE_BUFFER, _STRADDLE_BUFFER_PATH
    if _STRADDLE_STATE_BUFFER is not None and _STRADDLE_BUFFER_PATH == path:
        return _STRADDLE_STATE_BUFFER
    data = np.load(path, allow_pickle=True)
    buffer: dict = {
        "joint_pos": torch.from_numpy(data["joint_pos"].astype(np.float32)),
        "pcb_pos_env": torch.from_numpy(data["pcb_pos_env"].astype(np.float32)),
        "pcb_quat": torch.from_numpy(data["pcb_quat"].astype(np.float32)),
        "joint_names": list(data["joint_names"]),
    }
    if "is_straddled" in data:
        buffer["is_straddled"] = torch.from_numpy(data["is_straddled"].astype(bool))
    n = buffer["joint_pos"].shape[0]
    buffer["row_mask"] = _compose_buffer_row_mask(buffer, path, max_gripper_gap_m)
    if buffer["row_mask"] is not None:
        n_ok = int(buffer["row_mask"].sum().item())
        print(f"[StraddleStateBuffer] {n_ok}/{n} rows pass slide sampling mask")
        if n_ok == 0:
            raise RuntimeError(
                f"No valid rows in straddle state buffer '{path}'. "
                "Re-collect slide_terminal_states.npz with collect_slide_states.py."
            )
    _STRADDLE_STATE_BUFFER = buffer
    _STRADDLE_BUFFER_PATH = path
    print(f"[StraddleStateBuffer] Loaded {n} terminal states from '{path}'")
    return _STRADDLE_STATE_BUFFER


def sync_gripper_position_target_to_sim(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_name: str = "left_carriage_joint",
    open_width_m: float | None = None,
) -> None:
    """Set gripper PD ``joint_pos_target`` after reset.

    When ``open_width_m`` is set (grasp reset), command that explicit open width so the
  target is not inherited from the previous episode's closed pinch (~``PCB_Z * 0.5``).
    When omitted, sync to the current sim joint position (legacy behaviour).
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
    if open_width_m is not None:
        ow = float(open_width_m)
        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_vel = robot.data.joint_vel[env_ids].clone()
        joint_pos[:, jid] = ow
        joint_vel[:, jid] = 0.0
        lim = robot.data.soft_joint_pos_limits[env_ids, jid]
        joint_pos[:, jid] = joint_pos[:, jid].clamp(lim[:, 0], lim[:, 1])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        robot.update(0.0)
        target = torch.full(
            (len(env_ids), 1),
            ow,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )
    else:
        target = robot.data.joint_pos[env_ids, jid].unsqueeze(-1)
    zeros = torch.zeros_like(target)
    robot.set_joint_position_target(target, joint_ids=[jid], env_ids=env_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=[jid], env_ids=env_ids)




def hold_gripper_open(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_name: str,
    open_width_m: float,
    match_sim_state: bool = False,
    store_target: bool = False,
) -> None:
    """Command the parallel gripper to stay open (Phase 1 straddle / open-gripper slide)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_name)
    jid = joint_ids[0]
    open_val = float(open_width_m)

    if match_sim_state:
        target = robot.data.joint_pos[env_ids, jid].unsqueeze(-1)
    else:
        target = torch.full(
            (len(env_ids), 1),
            open_val,
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        )

    if store_target:
        if not hasattr(env, "_gripper_hold_target_m"):
            env._gripper_hold_target_m = torch.full(
                (env.num_envs,),
                open_val,
                device=env.device,
                dtype=robot.data.joint_pos.dtype,
            )
        env._gripper_hold_target_m[env_ids] = target.squeeze(-1)

    zeros = torch.zeros_like(target)
    robot.set_joint_position_target(target, joint_ids=[jid], env_ids=env_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=[jid], env_ids=env_ids)

    # Snap sim state on reset only; interval steps rely on PD to avoid fighting contacts.
    if not match_sim_state and store_target:
        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_pos[:, jid] = target.squeeze(-1)
        joint_vel = robot.data.joint_vel[env_ids].clone()
        joint_vel[:, jid] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _pcb_tilt_penalty_from_quat_batch(
    quat_wxyz: torch.Tensor,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> torch.Tensor:
    """Thickness-axis tilt penalty ``1 - |dot(z_body, up)|`` for buffer quaternions ``(N, 4)`` wxyz."""
    device = quat_wxyz.device
    dtype = quat_wxyz.dtype
    up = torch.tensor(world_up, device=device, dtype=dtype)
    up = up / torch.norm(up).clamp_min(1e-9)
    local_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    z_w = math_utils.quat_apply(
        quat_wxyz,
        local_z.unsqueeze(0).expand(quat_wxyz.shape[0], -1),
    )
    align = torch.abs(torch.sum(z_w * up.unsqueeze(0), dim=-1))
    return 1.0 - torch.clamp(align, max=1.0)


def _sample_straddle_buffer_indices(
    pcb_pos_env: torch.Tensor,
    n_samples: int,
    pcb_z_reference: float,
    max_z_delta_m: float,
    pcb_quat: torch.Tensor | None = None,
    max_tilt_penalty: float | None = None,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample buffer rows near ``pcb_z_reference``; optionally reject high tilt / bad rows."""
    z = pcb_pos_env[:, 2]
    valid = torch.abs(z - float(pcb_z_reference)) <= float(max_z_delta_m)
    if max_tilt_penalty is not None and pcb_quat is not None:
        tilt = _pcb_tilt_penalty_from_quat_batch(pcb_quat)
        valid = valid & (tilt <= float(max_tilt_penalty))
    if row_mask is not None:
        valid = valid & row_mask.bool()
    valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        if row_mask is not None:
            fallback = torch.nonzero(row_mask.bool(), as_tuple=False).view(-1)
            if fallback.numel() > 0:
                pick = torch.randint(0, fallback.numel(), (n_samples,), device="cpu")
                return fallback[pick]
            raise RuntimeError(
                "Straddle state buffer row_mask is empty — cannot sample a valid terminal state."
            )
        return torch.randint(0, pcb_pos_env.shape[0], (n_samples,), device="cpu")
    pick = torch.randint(0, valid_idx.numel(), (n_samples,), device="cpu")
    return valid_idx[pick]


def _sample_buffer_row_indices(
    buf: dict,
    n_samples: int,
    z_ref: float | None = None,
    max_z_delta_m: float = 0.025,
    pcb_quat: torch.Tensor | None = None,
    max_tilt_penalty: float | None = None,
    row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample buffer rows with optional Z / tilt / row-mask filters."""
    n_buf = buf["pcb_pos_env"].shape[0]
    if z_ref is not None:
        return _sample_straddle_buffer_indices(
            buf["pcb_pos_env"],
            n_samples,
            float(z_ref),
            float(max_z_delta_m),
            pcb_quat=pcb_quat,
            max_tilt_penalty=max_tilt_penalty,
            row_mask=row_mask,
        )
    if row_mask is not None:
        valid_idx = torch.nonzero(row_mask.bool(), as_tuple=False).view(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError(
                "Straddle state buffer row_mask is empty — cannot sample a valid terminal state."
            )
        pick = torch.randint(0, valid_idx.numel(), (n_samples,), device="cpu")
        return valid_idx[pick]
    return torch.randint(0, n_buf, (n_samples,), device="cpu")


def _gripper_midpoint_z_w(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """World Z of the gripper jaw midpoint."""
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
    return mid[env_ids, 2]


def _lift_robot_wrist_z_by_delta(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    delta_z_env: torch.Tensor,
    joint_names: tuple[str, ...] = ("joint_1", "joint_2"),
    joint_weights: tuple[float, ...] = (0.65, 0.35),
    max_iters: int = 12,
    z_tol_m: float = 0.002,
    step_gain: float = 0.9,
    m_per_rad: float = 0.12,
) -> None:
    """Raise gripper midpoint world-Z by ``delta_z_env`` using closed-loop ``joint_1``/``joint_2`` nudges."""
    if len(env_ids) == 0:
        return
    active = delta_z_env.abs() > 1e-6
    if not torch.any(active):
        return

    robot: Articulation = env.scene[robot_cfg.name]
    dtype = robot.data.joint_pos.dtype
    device = env.device
    delta_z_env = delta_z_env.to(device=device, dtype=dtype)

    joint_ids: list[int] = []
    weights = torch.tensor(joint_weights, device=device, dtype=dtype)
    for name in joint_names:
        ids, _ = robot.find_joints(name)
        if len(ids) == 0:
            return
        joint_ids.append(ids[0])

    z_now = _gripper_midpoint_z_w(
        env, env_ids, left_finger_cfg, right_finger_cfg, wrist_body_cfg
    )
    target_z = z_now + delta_z_env

    for _ in range(int(max_iters)):
        err = target_z - z_now
        err = torch.where(active, err, torch.zeros_like(err))
        if torch.all((err.abs() <= float(z_tol_m)) | ~active):
            break

        joint_pos_new = robot.data.joint_pos[env_ids].clone()
        for jid, w in zip(joint_ids, weights):
            # wxai: decreasing joint_1/2 raises the wrist at typical slide poses.
            delta_j = -float(step_gain) * err * w / float(m_per_rad)
            joint_pos_new[:, jid] = joint_pos_new[:, jid] + delta_j
            lim = robot.data.soft_joint_pos_limits[env_ids, jid]
            joint_pos_new[:, jid] = joint_pos_new[:, jid].clamp(lim[..., 0], lim[..., 1])

        joint_vel_new = torch.zeros_like(joint_pos_new)
        robot.write_joint_state_to_sim(joint_pos_new, joint_vel_new, env_ids=env_ids)
        robot.update(0.0)
        z_now = _gripper_midpoint_z_w(
            env, env_ids, left_finger_cfg, right_finger_cfg, wrist_body_cfg
        )

    if hasattr(env, "_slide_reset_joint_pos"):
        env._slide_reset_joint_pos[env_ids] = robot.data.joint_pos[env_ids].clone()


def _apply_robot_vertical_z_lift(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    delta_z_env: torch.Tensor,
    joint_name: str = "joint_1",
    lift_m_per_rad: float = 0.12,
    lift_sign: float = -1.0,
) -> None:
    """Nudge a shoulder joint after a PCB Z snap so the gripper moves up with the board."""
    if len(env_ids) == 0:
        return
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_name)
    if len(joint_ids) == 0:
        return
    jid = joint_ids[0]
    delta_joint = (
        float(lift_sign) * delta_z_env.to(device=env.device, dtype=robot.data.joint_pos.dtype) / float(lift_m_per_rad)
    )
    # Only lift envs that received a non-zero snap delta.
    delta_joint = torch.where(delta_z_env.abs() > 1e-6, delta_joint, torch.zeros_like(delta_joint))
    joint_pos_new = robot.data.joint_pos[env_ids].clone()
    joint_pos_new[:, jid] = joint_pos_new[:, jid] + delta_joint
    lim = robot.data.soft_joint_pos_limits[env_ids, jid]
    joint_pos_new[:, jid] = joint_pos_new[:, jid].clamp(lim[..., 0], lim[..., 1])
    joint_vel_new = torch.zeros_like(joint_pos_new)
    robot.write_joint_state_to_sim(joint_pos_new, joint_vel_new, env_ids=env_ids)
    robot.update(0.0)
    if hasattr(env, "_slide_reset_joint_pos"):
        env._slide_reset_joint_pos[env_ids] = joint_pos_new.clone()


def _zero_entity_velocities_after_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg | None = None,
    robot_cfg: SceneEntityCfg | None = None,
) -> None:
    """Zero root / joint velocities after a teleport reset to avoid launch impulses."""
    if pcb_cfg is not None:
        pcb = env.scene[pcb_cfg.name]
        zeros = torch.zeros((len(env_ids), 6), device=env.device, dtype=pcb.data.root_vel_w.dtype)
        pcb.write_root_velocity_to_sim(zeros, env_ids=env_ids)
        pcb.update(0.0)
    if robot_cfg is not None:
        robot = env.scene[robot_cfg.name]
        joint_zeros = torch.zeros_like(robot.data.joint_vel[env_ids])
        robot.write_joint_velocity_to_sim(joint_zeros, env_ids=env_ids)
        robot.update(0.0)




def reset_from_straddle_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    straddle_states_path: str,
    velocity_scale: float = 0.0,
    gripper_joint_name: str = "left_carriage_joint",
    gripper_closed_target_m: float = 0.00025,
    gripper_open_target_m: float | None = None,
    gripper_hold_open: bool = False,
    pcb_z_filter_env: float | None = None,
    max_pcb_z_delta_m: float = 0.025,
    max_buffer_tilt_penalty: float | None = None,
    rail_center_z_env: float | None = None,
    max_rail_z_delta_m: float | None = None,
    apply_gripper_hold_on_reset: bool = True,
) -> None:
    """Reset **robot joints only** by sampling from the saved straddle terminal-state buffer.

    Implements the Phase-2 initial-state distribution from Sequential Dexterity
    (Chen et al. CoRL 2023): the terminal state distribution of Phase 1 (Straddle)
    becomes the initial state distribution of Phase 2 (Slide).

    Stores ``env._straddle_buffer_idx`` so :func:`reset_pcb_from_straddle_states` can load
    the matching ``pcb_pos_env`` / ``pcb_quat`` from the same buffer row.

    Parameters
    ----------
    straddle_states_path:
        Path to the .npz produced by ``scripts/collect_straddle_states.py`` (or legacy
        ``collect_grasp_states.py``).
    gripper_hold_open:
        When True, PD-hold the gripper at ``gripper_open_target_m`` (open-gripper slide).
    """
    buf = _load_straddle_state_buffer(straddle_states_path)
    n_buf = buf["joint_pos"].shape[0]
    n_reset = len(env_ids)
    device = env.device
    dtype = torch.float32

    z_ref = pcb_z_filter_env if pcb_z_filter_env is not None else rail_center_z_env
    z_delta = max_pcb_z_delta_m if max_rail_z_delta_m is None else max_rail_z_delta_m

    row_mask = buf.get("row_mask")

    idx = _sample_buffer_row_indices(
        buf,
        n_reset,
        z_ref=z_ref,
        max_z_delta_m=float(z_delta),
        pcb_quat=buf["pcb_quat"] if max_buffer_tilt_penalty is not None else None,
        max_tilt_penalty=max_buffer_tilt_penalty,
        row_mask=row_mask,
    )
    if not hasattr(env, "_straddle_buffer_idx"):
        env._straddle_buffer_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._straddle_buffer_idx[env_ids] = idx.to(device=env.device)

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

    # Slide obs: joint_pos relative to this reset pose (not HOME default).
    if not hasattr(env, "_slide_reset_joint_pos"):
        env._slide_reset_joint_pos = torch.zeros(
            (env.num_envs, robot.num_joints), device=device, dtype=dtype
        )
    env._slide_reset_joint_pos[env_ids] = joint_pos_new.clone()

    if apply_gripper_hold_on_reset:
        gripper_ids, _ = robot.find_joints(gripper_joint_name)
        if gripper_hold_open:
            if gripper_open_target_m is not None:
                open_target = float(gripper_open_target_m)
            elif len(gripper_ids) > 0:
                open_target = float(joint_pos_new[:, gripper_ids[0]].mean().item())
            else:
                open_target = 0.015
            hold_gripper_open(
                env,
                env_ids,
                asset_cfg,
                joint_name=gripper_joint_name,
                open_width_m=open_target,
                match_sim_state=True,
                store_target=True,
            )
        else:
            joint_ids, _ = robot.find_joints(gripper_joint_name)
            if len(joint_ids) > 0:
                jid = joint_ids[0]
                grip_target = joint_pos_new[:, jid].unsqueeze(-1)
                robot.set_joint_position_target(grip_target, joint_ids=[jid], env_ids=env_ids)
                zeros = torch.zeros_like(grip_target)
                robot.set_joint_velocity_target(zeros, joint_ids=[jid], env_ids=env_ids)
    else:
        joint_ids, _ = robot.find_joints(gripper_joint_name)
        if len(joint_ids) > 0:
            jid = joint_ids[0]
            grip_target = joint_pos_new[:, jid].unsqueeze(-1)
            robot.set_joint_position_target(grip_target, joint_ids=[jid], env_ids=env_ids)
            zeros = torch.zeros_like(grip_target)
            robot.set_joint_velocity_target(zeros, joint_ids=[jid], env_ids=env_ids)

    # Legacy: snap path reads quat from here; buffer PCB reset uses the same buffer row.
    pcb_quat_buf = buf["pcb_quat"][idx].to(device=device, dtype=dtype)
    if not hasattr(env, "_sampled_pcb_quat"):
        env._sampled_pcb_quat = torch.zeros((env.num_envs, 4), device=device, dtype=dtype)
    env._sampled_pcb_quat[env_ids] = pcb_quat_buf


def reset_pcb_from_straddle_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    straddle_states_path: str,
    half_length_m: float,
    velocity_scale: float = 0.0,
    rail_center_z_env: float | None = None,
    snap_z_to_rail: bool = False,
    snap_z_max_delta_m: float = 0.002,
    rail_z_clearance_m: float = 0.0,
    flatten_pcb_orientation: bool = False,
    flat_rot_wxyz: tuple[float, float, float, float] | None = None,
    lift_robot_with_snap: bool = False,
    robot_asset_cfg: SceneEntityCfg | None = None,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    wrist_body_cfg: SceneEntityCfg | None = None,
    wrist_lift_joint_names: tuple[str, ...] = ("joint_1", "joint_2"),
    wrist_lift_joint_weights: tuple[float, ...] = (0.65, 0.35),
    wrist_lift_max_iters: int = 12,
    wrist_lift_z_tol_m: float = 0.002,
    # Legacy heuristic lift (used when finger cfgs are omitted).
    vertical_lift_joint_name: str = "joint_1",
    vertical_lift_joint_name_2: str | None = "joint_2",
    vertical_lift_m_per_rad: float = 0.12,
    vertical_lift_m_per_rad_2: float = 0.10,
    vertical_lift_sign: float = -1.0,
    vertical_lift_split: float = 0.65,
    slot_mouth_y_env: float = 0.198,
) -> None:
    """Place the PCB at the Phase-1 terminal pose stored in the grasp buffer.

    Must run **after** :func:`reset_from_straddle_states` so ``env._straddle_buffer_idx``
    points to the same buffer row as the robot joints.

    When ``snap_z_to_rail`` is True, centre Z is set to ``rail_center_z_env`` (+ optional
    clearance) when the buffer Z is within ``snap_z_max_delta_m``.  With ``lift_robot_with_snap``,
    the arm wrist midpoint is raised to match the PCB Z delta before teleport.  Optional
    ``flatten_pcb_orientation`` replaces buffer quaternions with ``flat_rot_wxyz``.
    """
    if not hasattr(env, "_straddle_buffer_idx"):
        raise RuntimeError(
            "reset_pcb_from_straddle_states requires a prior reset_from_straddle_states call "
            "that sets env._straddle_buffer_idx."
        )
    buf = _load_straddle_state_buffer(straddle_states_path)
    pcb = env.scene[pcb_cfg.name]
    device = env.device
    dtype = torch.float32

    idx = env._straddle_buffer_idx[env_ids].cpu()
    pos_env = buf["pcb_pos_env"][idx].to(device=device, dtype=dtype)
    delta_z_env = torch.zeros(len(env_ids), device=device, dtype=dtype)
    if snap_z_to_rail and rail_center_z_env is not None:
        rail_z = float(rail_center_z_env) + float(rail_z_clearance_m)
        z_buf = pos_env[:, 2]
        dz = rail_z - z_buf
        snap = torch.abs(dz) <= float(snap_z_max_delta_m)
        delta_z_env = torch.where(snap, dz, torch.zeros_like(dz))
        pos_env[:, 2] = torch.where(snap, torch.full_like(z_buf, rail_z), z_buf)

    # Lift wrist to match PCB Z snap before teleport (closed-loop on gripper midpoint Z).
    if lift_robot_with_snap and robot_asset_cfg is not None and torch.any(delta_z_env.abs() > 1e-6):
        if left_finger_cfg is not None and right_finger_cfg is not None:
            _lift_robot_wrist_z_by_delta(
                env,
                env_ids,
                robot_asset_cfg,
                left_finger_cfg,
                right_finger_cfg,
                wrist_body_cfg,
                delta_z_env,
                joint_names=wrist_lift_joint_names,
                joint_weights=wrist_lift_joint_weights,
                max_iters=wrist_lift_max_iters,
                z_tol_m=wrist_lift_z_tol_m,
                m_per_rad=vertical_lift_m_per_rad,
            )
        else:
            split = float(vertical_lift_split)
            _apply_robot_vertical_z_lift(
                env,
                env_ids,
                robot_asset_cfg,
                delta_z_env * split,
                joint_name=vertical_lift_joint_name,
                lift_m_per_rad=vertical_lift_m_per_rad,
                lift_sign=vertical_lift_sign,
            )
            if vertical_lift_joint_name_2 is not None:
                _apply_robot_vertical_z_lift(
                    env,
                    env_ids,
                    robot_asset_cfg,
                    delta_z_env * (1.0 - split),
                    joint_name=vertical_lift_joint_name_2,
                    lift_m_per_rad=vertical_lift_m_per_rad_2,
                    lift_sign=vertical_lift_sign,
                )

    quat = buf["pcb_quat"][idx].to(device=device, dtype=dtype)
    if flatten_pcb_orientation and flat_rot_wxyz is not None:
        flat_q = torch.tensor(flat_rot_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(len(env_ids), -1)
        quat = flat_q.clone()
    if hasattr(env, "_sampled_pcb_quat"):
        env._sampled_pcb_quat[env_ids] = quat.clone()

    origins = env.scene.env_origins[env_ids, :3]
    pos_w = pos_env + origins
    root_pose = torch.cat([pos_w, quat], dim=-1)

    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    root_vel = torch.zeros((len(env_ids), 6), device=device, dtype=dtype)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    pcb.update(0.0)
    if robot_asset_cfg is not None:
        _zero_entity_velocities_after_reset(env, env_ids, robot_cfg=robot_asset_cfg)

    _store_insert_progress_baselines(env, env_ids, pcb_cfg, half_length_m, slot_mouth_y_env)


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
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.010,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.010,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    min_closedness_for_push: float = 0.0,
    closedness_std: float = 0.020,
    finger_offset_m: float = 0.020,
) -> torch.Tensor:
    """Gate push rewards on straddle closedness (preferred) or ``pcb_between_gripper_fingers`` quality."""
    if (
        float(min_closedness_for_push) > 0.0
        and gripper_joint_cfg is not None
        and left_finger_cfg is not None
        and right_finger_cfg is not None
    ):
        closedness = straddle_finger_target_closedness(
            env,
            closedness_std,
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            finger_offset_m=finger_offset_m,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
        )
        gate = (closedness >= float(min_closedness_for_push)).to(dtype=reward.dtype)
        return reward * gate
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    min_closedness_for_push: float = 0.0,
    closedness_std: float = 0.020,
    finger_offset_m: float = 0.020,
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        min_closedness_for_push=min_closedness_for_push,
        closedness_std=closedness_std,
        finger_offset_m=finger_offset_m,
    )


def pcb_push_axis_velocity_reward_gated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_push_speed_m_s: float = 0.002,
    max_off_axis_speed_m_s: float = 0.020,
    min_straddle_quality: float = 0.0,
    proximity_sigma_m: float = 0.050,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    half_length_m: float = 0.060,
    pcb_half_thickness_m: float = 0.00075,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    min_closedness_for_push: float = 0.0,
    closedness_std: float = 0.020,
    finger_offset_m: float = 0.020,
) -> torch.Tensor:
    """Reward +push-axis PCB velocity when straddle is held and the board is not skidding."""
    vel = pcb_push_axis_velocity_reward(
        env, pcb_cfg, axis_world, min_push_speed_m_s=min_push_speed_m_s
    )
    pure = _pcb_off_axis_speed(env, pcb_cfg) < float(max_off_axis_speed_m_s)
    out = torch.where(pure, vel, torch.zeros_like(vel))
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        min_closedness_for_push=min_closedness_for_push,
        closedness_std=closedness_std,
        finger_offset_m=finger_offset_m,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
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
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_sigma_m: float = 0.025,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
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
        wrist_body_cfg=wrist_body_cfg,
        width_sigma_m=width_sigma_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
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
    max_lift_m: float = 0.003,
    exponential_scale_m: float = 0.003,
    max_excess_m: float = 0.02,
    use_episode_start: bool = True,
    reference_z_env: float | None = None,
) -> torch.Tensor:
    """Exponential penalty when the leading short-edge centre rises above the reference Z.

    Below ``ref_z + max_lift_m`` the penalty is zero. Beyond it,
    ``excess = lead_z - ref_z - max_lift_m`` drives ``expm1(excess / scale)`` (capped at
    ``max_excess_m``).  Return is normalized to approximately ``[0, 1]``.
    Pair with a **negative** weight.
    """
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_z = lead_w[:, 2] - env.scene.env_origins[:, 2]
    if use_episode_start and hasattr(env, "_insert_start_lead_z"):
        ref_z = env._insert_start_lead_z
    elif reference_z_env is not None:
        ref_z = torch.full_like(lead_z, float(reference_z_env))
    else:
        ref_z = lead_z.detach()
    excess = torch.clamp(
        lead_z - ref_z - float(max_lift_m),
        min=0.0,
        max=float(max_excess_m),
    )
    scale = float(exponential_scale_m) + 1e-9
    cap = float(max_excess_m)
    raw = torch.expm1(excess / scale)
    norm = torch.expm1(torch.tensor(cap / scale, device=excess.device, dtype=excess.dtype)) + 1e-9
    return raw / norm


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


def pcb_trailing_edge_z_velocity_exponential_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    min_z_speed_m_s: float = 0.003,
    exponential_scale_m: float = 0.003,
    max_excess_m_s: float = 0.02,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Exponential penalty on trailing (rear) short-edge centre |v_z| above ``min_z_speed_m_s``.

    Below the threshold the penalty is zero. Beyond it,
    ``excess = |v_z| - min_z_speed`` drives ``expm1(excess / scale)`` (capped at
    ``max_excess_m_s``).  Return is normalized to approximately ``[0, 1]``.
    Pair with a **negative** weight.
    """
    v = pcb_trailing_short_edge_center_lin_vel_world(
        env, pcb_cfg, half_length_m, axis_world
    )
    excess = torch.clamp(
        torch.abs(v[:, 2]) - float(min_z_speed_m_s),
        min=0.0,
        max=float(max_excess_m_s),
    )
    scale = float(exponential_scale_m) + 1e-9
    cap = float(max_excess_m_s)
    raw = torch.expm1(excess / scale)
    norm = torch.expm1(torch.tensor(cap / scale, device=excess.device, dtype=excess.dtype)) + 1e-9
    return raw / norm


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


def pcb_slide_axis_sustained_backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_backward_speed_m_s: float = 0.003,
    min_consecutive_steps: int = 3,
) -> torch.Tensor:
    """Slide-phase alias for :func:`pcb_push_axis_sustained_backward_velocity_penalty`."""
    return pcb_push_axis_sustained_backward_velocity_penalty(
        env,
        pcb_cfg,
        min_backward_speed_m_s=min_backward_speed_m_s,
        min_consecutive_steps=min_consecutive_steps,
    )


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


