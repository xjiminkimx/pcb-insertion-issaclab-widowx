"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/grasp shaping, regularization, rail reset, and drop detection.

Grasp and insert are separate registered envs (``Isaac-WidowX-PCB-Grasp-v0``,
``Isaac-WidowX-PCB-Insert-v0``); each uses its own reward config with no in-episode phase gating.
"""

import torch
import isaaclab.utils.math as math_utils
from dataclasses import MISSING
from collections.abc import Sequence

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


def _gripper_tip_params(
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> dict:
    """Kwargs bundle for pad-tip offset (passed from env cfg)."""
    return {"tip_offset_m": tip_offset_m, "wrist_body_cfg": wrist_body_cfg}


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
    closed_threshold: float = 0.35,
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
    """True when the gripper has a valid closed pinch on the trailing short-edge centre."""
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
    closed = gq < float(closed_threshold) * float(open_width_m)
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
    closed_threshold: float = 0.35,
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
        closed_threshold=closed_threshold,
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


def gripper_closing_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float,
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
    closedness = (1.0 - gq / float(open_width_m)).clamp(0.0, 1.0)

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
    closedness = (1.0 - gq / float(open_width_m)).clamp(0.0, 1.0)
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
    closed_threshold: float = 0.35,
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
        closed_threshold=closed_threshold,
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
    closed_threshold: float = 0.35,
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
        closed_threshold=closed_threshold,
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
    closed_threshold: float = 0.35,
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
        closed_threshold=closed_threshold,
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


def snap_pcb_root_to_short_edge_grasp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    rot_wxyz: tuple[float, float, float, float],
    velocity_scale: float = 0.0,
    center_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_center_z_env_local: float | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> None:
    """Place PCB root so the push-face center matches the jaw midpoint (kinematic grasp contact).

    Uses the same geometry as :func:`pcb_trailing_short_edge_center_w`. Solving
    ``trailing = jaw_mid`` gives ``pcb_center = jaw_mid + sign * half_length * body+X_world``
    where ``sign = sign(dot(body+X, push_axis))``.

    ``center_offset_body_m`` nudges the root in **PCB body** axes (tune if the USD finger origins
    sit on carriage housing so the analytic mid misses the actual pad gap).

    Vertical alignment: PCB center Z is set to the jaw midpoint Z (top/bottom pinch). If
    ``min_center_z_env_local`` is set, Z is floored to ``env_origin_z + min`` so the board is not
    spawned below the analytic rail plane when the arm pose is too low.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    robot.update(0.0)
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg)[env_ids]
    n = len(env_ids)
    dtype = mid.dtype
    device = env.device
    q = torch.tensor(rot_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
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
    if min_center_z_env_local is not None:
        min_cz = env.scene.env_origins[env_ids, 2] + float(min_center_z_env_local)
        center_w[:, 2] = torch.maximum(center_w[:, 2], min_cz)
    root_pose = torch.cat([center_w, q], dim=-1)

    default_root_state = pcb.data.default_root_state[env_ids].clone()
    root_vel = default_root_state[:, 7:13] * velocity_scale
    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    pcb.update(0.0)


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
    pcb = env.scene[pcb_cfg.name]
    N = len(env_ids)
    dtype = pcb.data.root_pos_w.dtype
    pl = torch.tensor(pos_env_local, device=env.device, dtype=dtype).unsqueeze(0).expand(N, -1)
    target_pos = pl + env.scene.env_origins[env_ids]
    q = torch.tensor(rot_wxyz, device=env.device, dtype=dtype).unsqueeze(0).expand(N, -1)
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
) -> torch.Tensor:
    """Return True if PCB is considered dropped from the gripper.

    Drop is detected if either:
    - (when ``check_grasp_geometry``) PCB deviates from expected grasp geometry vs jaw midpoint, or
    - PCB drops below a minimum height near table level.

    Set ``check_grasp_geometry=False`` when episodes start with the PCB on guide rails (not in-hand).
    """
    pcb = env.scene[pcb_cfg.name]  # grasped object

    pcb_pos = pcb.data.root_pos_w
    gripper_center = gripper_midpoint_world(
        env, left_finger_cfg, right_finger_cfg, tip_offset_m, wrist_body_cfg
    )

    pcb_height = pcb_pos[:, 2] - env.scene.env_origins[:, 2]
    is_low = pcb_height < min_height

    if not check_grasp_geometry:
        return is_low

    if expected_offset_world is not None:
        off = torch.tensor(expected_offset_world, device=env.device, dtype=pcb_pos.dtype).unsqueeze(0).expand_as(pcb_pos)
        expected_pos = gripper_center + off
        dist_to_grasp = torch.norm(pcb_pos - expected_pos, dim=-1)
        is_far = dist_to_grasp > distance_tolerance
    else:
        dist_to_gripper = torch.norm(pcb_pos - gripper_center, dim=-1)
        geom_error = torch.abs(dist_to_gripper - expected_center_distance)
        is_far = geom_error > distance_tolerance
    return is_far | is_low