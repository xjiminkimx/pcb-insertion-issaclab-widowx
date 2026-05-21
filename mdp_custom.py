"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/grasp shaping, regularization, rail reset, and drop detection.
"""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# ---------------------------------------------------------
# 관측 (Observations)
# ---------------------------------------------------------
def gripper_midpoint_position_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Env-local position at the midpoint between the two fingertip bodies (true grasp center).

    JetCobot mounts ``camera_link`` on ``link_6``; using ``link_6`` alone for EE pulls the policy toward the wrist /
    camera instead of the jaws. Prefer this term when fingertip link names are known.
    """
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
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


def gripper_mid_central_face_contact_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_extent_x_m: float,
    half_extent_y_m: float,
    half_extent_z_m: float,
    edge_fraction: float = 0.2,
    surface_band_m: float = 0.022,
    soft_sharpness: float = 40.0,
    linear_gate_blend: float = 0.92,
) -> torch.Tensor:
    """Penalty when the gripper midpoint is near the **top/bottom faces** over the **inner** board area.

    The large faces are normal to body ±Z. Each in-plane axis (X, Y) keeps an ``edge_fraction``
    strip at both ends (20 % of board length from each end along that axis); the remaining inner
    rectangle is the “central surface” where contact is discouraged (top/bottom crush onto the
    board instead of grasping the perimeter / short edge).

    Uses mostly **linear** gates so that dead-center pressing yields ~1.0 (not ~0.5–0.8 from
    sigmoid products). A small **sigmoid** blend keeps gradients usable just outside the band.
    """
    pcb = env.scene[pcb_cfg.name]
    if edge_fraction >= 0.5:
        return torch.zeros(env.num_envs, device=env.device, dtype=pcb.data.root_pos_w.dtype)

    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)

    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    d = mid - pcb_pos
    u = torch.sum(d * x_w, dim=-1)
    v = torch.sum(d * y_w, dim=-1)
    w = torch.sum(d * z_w, dim=-1)

    # Inner half-extents: forbidden when |coord| < limit (central 60 % × 60 % for edge_fraction=0.2).
    u_lim = half_extent_x_m * (1.0 - 2.0 * edge_fraction)
    v_lim = half_extent_y_m * (1.0 - 2.0 * edge_fraction)
    eps = 1e-5
    gu_lin = torch.clamp((u_lim - torch.abs(u)) / (u_lim + eps), 0.0, 1.0)
    gv_lin = torch.clamp((v_lim - torch.abs(v)) / (v_lim + eps), 0.0, 1.0)
    gu_soft = torch.sigmoid(soft_sharpness * (u_lim - torch.abs(u)))
    gv_soft = torch.sigmoid(soft_sharpness * (v_lim - torch.abs(v)))
    b = linear_gate_blend
    gu = b * gu_lin + (1.0 - b) * gu_soft
    gv = b * gv_lin + (1.0 - b) * gv_soft

    # Near top or bottom face: distance of |w| to the outer face offset hz.
    dist_from_face_plane = torch.abs(torch.abs(w) - half_extent_z_m)
    gw_lin = torch.clamp((surface_band_m - dist_from_face_plane) / (surface_band_m + eps), 0.0, 1.0)
    gw_soft = torch.sigmoid(soft_sharpness * (surface_band_m - dist_from_face_plane))
    gw = b * gw_lin + (1.0 - b) * gw_soft

    return gu * gv * gw


def gripper_mid_to_pcb_trailing_edge_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
) -> torch.Tensor:
    """Distance from jaw midpoint to the **push-face center** (robot-side end of the board).

    The push face is perpendicular to body +X (long axis). Its center lies on the **short edge**
    (length = cuboid local Y / ``PCB_Y``): the narrow end cap you grasp before sliding toward the
    magazine. Target: ``pcb_center - half_length * body_x_world``.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    long_axis = pcb_body_axis_x_world(env, pcb_cfg)
    trailing = pcb_pos - half_length_m * long_axis
    return torch.norm(mid - trailing, dim=-1)


def gripper_mid_to_pcb_hover_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    hover_height_m: float = 0.05,
) -> torch.Tensor:
    """Distance from jaw midpoint to a **pre-grasp hover point** above the PCB trailing edge.

    The hover point is the trailing-edge face center offset upward by ``hover_height_m`` along the
    world Z axis. The approach reward should first pull the EE to this hover point (from any
    direction without rail obstruction), then a separate descent reward brings it down to the face.

    This enables a **top-down grasp** trajectory:
    1. EE moves to hover point (above the trailing edge).
    2. EE descends vertically onto the 2.5 mm board face.
    3. Gripper closes on the board thickness from above/below.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    long_axis = pcb_body_axis_x_world(env, pcb_cfg)
    trailing = pcb_pos - half_length_m * long_axis
    # Offset upward in world Z (hover point above the edge)
    hover_offset = torch.zeros_like(trailing)
    hover_offset[:, 2] = float(hover_height_m)
    hover_point = trailing + hover_offset
    return torch.norm(mid - hover_point, dim=-1)


# Per-env previous EE distance for hover approach (reset each episode).
_EE_HOVER_PREV_DIST: torch.Tensor | None = None


def ee_hover_approach_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    hover_height_m: float = 0.05,
    max_step_m: float = 0.05,
) -> torch.Tensor:
    """Progress reward for the **hover-approach phase** of a top-down grasp.

    Rewards the EE moving closer to the pre-grasp hover point above the PCB trailing edge.
    Returns ``clamp(prev_dist - curr_dist, 0, max_step_m) / max_step_m`` ∈ [0, 1].
    Hovering or moving away returns exactly 0 — no hover-exploit.

    Pair this with ``ee_descent_progress_reward`` (or ``grasp_short_edge_closure_reward``) so the
    arm is first drawn above the board then rewarded for descending and closing.
    """
    global _EE_HOVER_PREV_DIST

    dist = gripper_mid_to_pcb_hover_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, hover_height_m
    )

    if (
        _EE_HOVER_PREV_DIST is None
        or _EE_HOVER_PREV_DIST.shape[0] != dist.shape[0]
        or _EE_HOVER_PREV_DIST.device != dist.device
    ):
        _EE_HOVER_PREV_DIST = dist.clone()
        return torch.zeros_like(dist)

    first_step = env.episode_length_buf == 1
    _EE_HOVER_PREV_DIST = torch.where(first_step, dist, _EE_HOVER_PREV_DIST)
    progress = (_EE_HOVER_PREV_DIST - dist).clamp(0.0, float(max_step_m))
    _EE_HOVER_PREV_DIST = dist.clone()
    return progress / float(max_step_m)


def gripper_pinch_readiness(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """Soft readiness in ``[0, 1]`` for closing: thickness alignment × pinch orientation."""
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    thickness_ok = torch.exp(-torch.abs(w) / thickness_sigma_m)

    v = right - left
    n = torch.norm(v, dim=-1)
    safe_n = n.clamp(min=1e-6).unsqueeze(-1)
    u_lr = v / safe_n
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    open_est = torch.cross(x_w, u_lr, dim=-1)
    no = torch.norm(open_est, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    open_est = open_est / no
    align_open = torch.abs(torch.sum(open_est * z_w, dim=-1))
    align_finger = torch.abs(torch.sum(u_lr * y_w, dim=-1))
    sep_ok = (n > min_finger_sep_m).to(dtype=v.dtype)
    return thickness_ok * align_open * align_finger * sep_ok


def grasp_short_edge_closure_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float = 0.044,
    gate_dist_m: float = 0.07,
    pcb_half_thickness_m: float = 0.00125,
    above_sigma_m: float = 0.004,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """Reward **closing** only when near the trailing edge, above the board, and pinch-ready.

    Returns ``closure * gate * above_gate * pinch_ready`` in ``[0, 1]``.
    Closure is not rewarded until jaws are aligned to straddle the short edge — this prevents
    closing while still approaching or digging under the PCB.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    closure = torch.clamp(1.0 - gq / open_width_m, 0.0, 1.0)
    gate = torch.exp(-dist / gate_dist_m)

    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    above_gate = torch.sigmoid((w + pcb_half_thickness_m) / above_sigma_m)

    pinch_ready = gripper_pinch_readiness(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
    )

    return closure * gate * above_gate * pinch_ready


def gripper_mid_thickness_plane_alignment_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    sigma_m: float = 0.012,
    half_length_m: float | None = None,
    gate_dist_m: float | None = None,
) -> torch.Tensor:
    """Shaping: 1 when jaw midpoint lies on the PCB **mid-thickness** plane (ideal top/bottom pinch).

    Parallel jaws should straddle the thin board; ``dot(mid - pcb_center, body+Z)`` should be ~0.

    Optional trailing-edge gate: when ``half_length_m`` and ``gate_dist_m`` are provided the
    reward is multiplied by ``exp(-d / gate_dist_m)`` where ``d`` is the distance to the trailing
    edge face center.  This prevents the large PCB top surface from becoming an attractive well —
    thickness alignment only pays off near the short edge where we actually want to pinch.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    alignment = torch.exp(-torch.abs(w) / sigma_m)

    if half_length_m is not None and gate_dist_m is not None:
        dist = gripper_mid_to_pcb_trailing_edge_distance(
            env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
        )
        alignment = alignment * torch.exp(-dist / gate_dist_m)

    return alignment


def gripper_open_push_face_rub_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float = 0.044,
    plane_band_m: float = 0.032,
    thickness_band_m: float = 0.004,
    pcb_half_thickness_m: float = 0.00125,
    above_sigma_m: float = 0.003,
) -> torch.Tensor:
    """Penalty for **open** gripper side-rubbing the push face at rail height (not top-down descent).

    Suppressed when the jaw midpoint is above the PCB top face so top-down open approach is not
    penalised.  High when: open, near push-face plane, near mid-thickness, and not above the board.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w_coord = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    d_plane = torch.abs(torch.sum((mid - pcb_pos) * x_w, dim=-1) + half_length_m)
    gq = env.scene[gripper_joint_cfg.name].data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / open_width_m, 0.0, 1.0)
    rub_plane = torch.exp(-d_plane / plane_band_m)
    rub_thick = torch.exp(-torch.abs(w_coord) / (thickness_band_m + 1e-6))
    # No side-rub penalty while descending from above the board top.
    not_top_down = 1.0 - torch.sigmoid((w_coord - pcb_half_thickness_m) / above_sigma_m)
    return open_norm * rub_plane * rub_thick * not_top_down


def gripper_open_above_edge_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float = 0.010,
    gate_dist_m: float = 0.10,
    pcb_half_thickness_m: float = 0.00125,
    above_margin_m: float = 0.002,
    above_sigma_m: float = 0.004,
    min_open_fraction: float = 0.65,
) -> torch.Tensor:
    """Reward a **wide-open** gripper while approaching the trailing edge from above the PCB.

    Returns ``open_norm * near_gate * above_gate`` in ``[0, 1]``.
    ``above_gate`` requires the jaw midpoint to sit above the board top (top-down pre-grasp).
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / open_width_m, 0.0, 1.0)
    # Soft preference for fully open (not just slightly open).
    open_ok = torch.clamp((open_norm - min_open_fraction) / (1.0 - min_open_fraction + 1e-6), 0.0, 1.0)
    near_gate = torch.exp(-dist / gate_dist_m)

    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    above_gate = torch.sigmoid((w - pcb_half_thickness_m - above_margin_m) / above_sigma_m)

    return open_ok * near_gate * above_gate


def premature_close_at_edge_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float = 0.010,
    gate_dist_m: float = 0.10,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    partial_close_threshold: float = 0.35,
) -> torch.Tensor:
    """Penalty for closing (or partially closing) near the edge before pinch geometry is ready.

    Returns ``closed * near_gate * (1 - pinch_ready)`` in ``[0, 1]``.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / open_width_m, 0.0, 1.0)
    closed = torch.clamp(1.0 - open_norm / (partial_close_threshold + 1e-6), 0.0, 1.0)
    near_gate = torch.exp(-dist / gate_dist_m)
    pinch_ready = gripper_pinch_readiness(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
    )
    return closed * near_gate * (1.0 - pinch_ready)


def closed_below_pcb_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    pcb_half_thickness_m: float = 0.00125,
    open_width_m: float = 0.010,
    sigma_below_m: float = 0.005,
) -> torch.Tensor:
    """Penalty when the gripper is **closed** and the jaw midpoint is **below the PCB bottom face**.

    Directly penalises the "close and dig under" failure mode. The penalty scales with how
    closed the gripper is and how far the jaw has sunk below the PCB bottom surface.
    Returns a value in ``[0, 1]``.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)

    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    # depth_below > 0 only when jaw mid sinks below the PCB bottom face
    depth_below = torch.relu(-(w + pcb_half_thickness_m))
    below_signal = 1.0 - torch.exp(-depth_below / sigma_below_m)

    robot_art = env.scene[gripper_joint_cfg.name]
    gq = robot_art.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    closed_norm = torch.clamp(1.0 - gq / open_width_m, 0.0, 1.0)

    return closed_norm * below_signal


def gripper_mid_thickness_offset_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    scale_m: float = 0.012,
) -> torch.Tensor:
    """Obs: signed offset along PCB thickness axis (jaw mid vs board center), scaled to ~[-1, 1]."""
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    w = torch.sum((mid - pcb_pos) * z_w, dim=-1)
    return torch.clamp(w / (scale_m + 1e-6), -1.0, 1.0).unsqueeze(-1)


def gripper_pinch_orientation_flat_edge_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.12,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """Near the short-edge / push-face target, reward a **rail-parallel pinch** pose (top/bottom grasp).

    Ideal geometry (board flat, long axis ∥ rail / body +X):
    - **Finger span** ``normalize(right - left)`` ∥ PCB **short edge** (body +Y) so pads straddle the
      thin board in thickness.
    - **Opening direction** estimated as ``normalize(cross(body+X, finger_span))`` ∥ body +Z so jaws
      open along thickness (same as ``pcb_body_axis_z_world``).

    Gated by distance to the push-face center so the whole-arm approach is not over-constrained early.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    v = right - left
    n = torch.norm(v, dim=-1)
    safe_n = n.clamp(min=1e-6).unsqueeze(-1)
    u_lr = v / safe_n
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    open_est = torch.cross(x_w, u_lr, dim=-1)
    no = torch.norm(open_est, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    open_est = open_est / no
    align_open = torch.abs(torch.sum(open_est * z_w, dim=-1))
    align_finger = torch.abs(torch.sum(u_lr * y_w, dim=-1))
    sep_ok = (n > min_finger_sep_m).to(dtype=v.dtype)
    gate = torch.exp(-dist / gate_dist_m)
    return align_open * align_finger * sep_ok * gate


def gripper_pinch_orientation_cos_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """Two cosines in [0, 1]: opening∥thickness, finger-span∥short-edge (zeroed if fingers coincide)."""
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    v = right - left
    n = torch.norm(v, dim=-1)
    safe_n = n.clamp(min=1e-6).unsqueeze(-1)
    u_lr = v / safe_n
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    y_w = pcb_body_axis_y_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    open_est = torch.cross(x_w, u_lr, dim=-1)
    no = torch.norm(open_est, dim=-1).clamp(min=1e-6).unsqueeze(-1)
    open_est = open_est / no
    align_open = torch.abs(torch.sum(open_est * z_w, dim=-1))
    align_finger = torch.abs(torch.sum(u_lr * y_w, dim=-1))
    m = (n > min_finger_sep_m).to(dtype=v.dtype)
    return torch.stack([align_open * m, align_finger * m], dim=-1)


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


def gripper_closure_early_episode_shaping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    open_width_m: float = 0.044,
    max_episode_length_steps: int = 120,
    min_gate: float = 0.0,
    gripper_joint_sign: float = 1.0,
) -> torch.Tensor:
    """Keep the gripper **closed** (or near the reset opening) at the **start** of the episode.

    This task often resets with the PCB already pinched on the short edge. Nothing in insertion /
    push reward explicitly discourages the first action from **opening** the tool and dropping the
    board. Returns ``closure * gate`` with ``closure = 1 - (sign * q)/open_max`` and ``gate`` linearly
    decaying from 1 to ``min_gate`` over the first ``max_episode_length_steps`` **env** (control)
    steps so the policy can re-open later for re-grasps if needed.
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    eff = gripper_joint_sign * q
    closure = torch.clamp(1.0 - eff / open_width_m, 0.0, 1.0)
    t = env.episode_length_buf.to(dtype=closure.dtype)
    m = float(max(1, max_episode_length_steps))
    progress = torch.clamp(t / m, 0.0, 1.0)
    gate = 1.0 - (1.0 - min_gate) * progress
    return closure * gate


def pcb_linear_velocity_along_world_axis(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Scalar: PCB linear velocity projected onto a **world-frame** push direction (unit vector).

    Use this when the fixture is an ``AssetBaseCfg`` / Xform without ``.data`` (no rigid root state).
    Set ``axis_world`` to the direction you want the board to move (e.g. ``(0, 1, 0)`` toward +Y).
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    ax = torch.tensor(axis_world, device=env.device, dtype=v.dtype)
    ax = ax / torch.norm(ax).clamp_min(1e-9)
    ax = ax.unsqueeze(0).expand(v.shape[0], -1)
    return torch.sum(v * ax, dim=-1)


def pcb_forward_velocity_along_world_axis(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Same projection as ``pcb_linear_velocity_along_world_axis``, but **only forward** (relu).

    Backward motion along the push axis contributes **0** instead of a negative reward. That avoids the
    push term fighting the approach phase and makes small forward slides strictly preferable to
    standing still when the policy has already reached the trailing edge.
    """
    v = pcb_linear_velocity_along_world_axis(env, pcb_cfg, axis_world)
    return torch.relu(v)


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
    p_w = pcb.data.root_pos_w
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    lead_w = p_w + float(half_length_m) * x_w
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
    pcb = env.scene[pcb_cfg.name]
    p_w = pcb.data.root_pos_w
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    lead_w = p_w + float(half_length_m) * x_w
    lead_env = lead_w - env.scene.env_origins[:, :3]
    tgt = torch.tensor(target_lead_xyz_env, device=lead_env.device, dtype=lead_env.dtype).unsqueeze(0).expand(
        env.num_envs, -1
    )
    dist = torch.norm(lead_env - tgt, dim=-1)
    return torch.exp(-dist / (float(sigma_m) + 1e-9))


def ee_approach_pcb_trailing_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    sigma_m: float = 0.08,
) -> torch.Tensor:
    """Approach shaping: reward EE getting close to the PCB **trailing (push-face) edge**.

    Returns ``exp(-d / sigma_m)`` in (0, 1]. Pulls the arm toward the board in the early
    phase before grasp and insertion rewards dominate. Set a modest positive weight
    (e.g. 4.0) so it does not compete with the grasp/push terms.

    .. warning::
        This Gaussian form gives reward for *being* near the PCB, which can be
        exploited by hovering. Prefer :func:`ee_approach_progress_reward` to prevent
        the hover-exploit cycle.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    return torch.exp(-dist / (float(sigma_m) + 1e-9))


# Per-env previous EE distance used by the progress reward (reset each episode).
_EE_APPROACH_PREV_DIST: torch.Tensor | None = None


def ee_approach_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_step_m: float = 0.05,
) -> torch.Tensor:
    """**Progress-only** approach reward: fires only when the EE is getting *closer* to the PCB.

    Returns ``clamp(prev_dist - curr_dist, 0, max_step_m) / max_step_m`` ∈ [0, 1].
    When hovering (distance unchanged) or moving away the reward is exactly 0, which
    prevents the hover-exploit cycle that the Gaussian form is susceptible to.

    The previous distance is reset to the current distance on the first step of each
    episode (``episode_length_buf == 1``), so resets are handled cleanly.
    """
    global _EE_APPROACH_PREV_DIST

    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )

    if (
        _EE_APPROACH_PREV_DIST is None
        or _EE_APPROACH_PREV_DIST.shape[0] != dist.shape[0]
        or _EE_APPROACH_PREV_DIST.device != dist.device
    ):
        _EE_APPROACH_PREV_DIST = dist.clone()
        return torch.zeros_like(dist)

    first_step = env.episode_length_buf == 1
    # On the first step of an episode, reset prev_dist so we don't reward the teleport.
    _EE_APPROACH_PREV_DIST = torch.where(first_step, dist, _EE_APPROACH_PREV_DIST)

    progress = (_EE_APPROACH_PREV_DIST - dist).clamp(0.0, float(max_step_m))
    _EE_APPROACH_PREV_DIST = dist.clone()
    return progress / float(max_step_m)


def grasp_and_push_bonus(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float,
    closed_threshold: float = 0.3,
    gate_dist_m: float = 0.08,
) -> torch.Tensor:
    """Combined bonus: fires **only** when the gripper is closed AND the PCB is moving in +Y.

    This breaks the "descend + partial-close + stay" local optimum by rewarding the
    transition from grasping to pushing. The policy must commit to both actions simultaneously
    to earn this reward.

    Returns ``closure_gate × velocity_relu`` where:
    - ``closure_gate = 1`` when gripper joint < ``closed_threshold × open_width_m``
      (i.e. gripper is actually closed, not just partially closed)
    - ``velocity_relu = relu(v_y)`` — world +Y velocity of the PCB

    Result is in [0, ∞) proportional to how fast the PCB moves while grasped.
    """
    # Closure gate: 1 when closed enough, 0 when open
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    is_closed = (gq < float(closed_threshold) * float(open_width_m)).to(dtype=gq.dtype)

    # Distance gate: only fire when near the push face
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    dist_gate = torch.exp(-dist / float(gate_dist_m))

    # PCB +Y velocity (toward slot)
    pcb = env.scene[pcb_cfg.name]
    v_y = torch.relu(pcb.data.root_lin_vel_w[:, 1])

    return is_closed * dist_gate * v_y


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
    Use a positive weight; combine with ``push_y_toward_slot`` which only fires before the mouth.
    """
    pcb = env.scene[pcb_cfg.name]
    p_w = pcb.data.root_pos_w
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    lead_w = p_w + float(half_length_m) * x_w
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


# Counts consecutive env steps with near-zero arm joint velocity (one buffer per training process).
_ARM_VEL_IDLE_COUNT: torch.Tensor | None = None
# Counts consecutive env steps where the PCB moves backward (−Y).
_PCB_BACKWARD_COUNT: torch.Tensor | None = None


def arm_joints_velocity_idle_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_abs_vel_rad_s: float = 0.03,
    min_idle_steps: int = 100,
) -> torch.Tensor:
    """Terminate when monitored joints stay essentially still for ``min_idle_steps`` control steps.

    Uses ``episode_length_buf == 1`` to reset the counter at each new episode, and resets whenever
    ``max |qdot|`` exceeds ``max_abs_vel_rad_s``. Restrict ``asset_cfg`` to **arm joints only** so a
    closing gripper alone does not count as "moving the arm" if you want pure arm-idle detection.

    Note:
        If you add a **success** state where the policy legitimately holds still, increase
        ``min_idle_steps`` or disable this term for those envs.
    """
    global _ARM_VEL_IDLE_COUNT
    robot = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        j_vel = robot.data.joint_vel
    else:
        j_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    max_v = torch.max(torch.abs(j_vel), dim=1).values
    moving = max_v > max_abs_vel_rad_s
    first_step = env.episode_length_buf == 1
    if (
        _ARM_VEL_IDLE_COUNT is None
        or _ARM_VEL_IDLE_COUNT.shape[0] != env.num_envs
        or _ARM_VEL_IDLE_COUNT.device != env.device
    ):
        _ARM_VEL_IDLE_COUNT = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    _ARM_VEL_IDLE_COUNT = torch.where(
        first_step | moving,
        torch.zeros_like(_ARM_VEL_IDLE_COUNT),
        _ARM_VEL_IDLE_COUNT + 1,
    )
    return _ARM_VEL_IDLE_COUNT >= min_idle_steps


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
) -> None:
    """Place PCB root so the push-face center matches the jaw midpoint (kinematic grasp contact).

    Uses the same geometry as ``gripper_mid_to_pcb_trailing_edge_distance``: trailing center is
    ``pcb_center - half_length * body+X``. Solving ``trailing = jaw_mid`` gives
    ``pcb_center = jaw_mid + half_length * body+X_world``.

    ``center_offset_body_m`` nudges the root in **PCB body** axes (tune if the USD finger origins
    sit on carriage housing so the analytic mid misses the actual pad gap).

    Vertical alignment: PCB center Z is set to the jaw midpoint Z (top/bottom pinch). If
    ``min_center_z_env_local`` is set, Z is floored to ``env_origin_z + min`` so the board is not
    spawned below the analytic rail plane when the arm pose is too low.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    robot.update(0.0)
    left = robot.data.body_pos_w[env_ids][:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[env_ids][:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    n = len(env_ids)
    dtype = mid.dtype
    device = env.device
    q = torch.tensor(rot_wxyz, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    local_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    x_w = math_utils.quat_apply(q, local_x)
    ob = torch.tensor(center_offset_body_m, device=device, dtype=dtype).unsqueeze(0).expand(n, -1)
    off_w = math_utils.quat_apply(q, ob)
    center_w = mid + float(half_length_m) * x_w + off_w
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
) -> torch.Tensor:
    """Return True if PCB is considered dropped from the gripper.

    Drop is detected if either:
    - (when ``check_grasp_geometry``) PCB deviates from expected grasp geometry vs jaw midpoint, or
    - PCB drops below a minimum height near table level.

    Set ``check_grasp_geometry=False`` when episodes start with the PCB on guide rails (not in-hand).
    """
    pcb = env.scene[pcb_cfg.name]  # grasped object
    robot = env.scene[left_finger_cfg.name]

    pcb_pos = pcb.data.root_pos_w
    left_pos = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right_pos = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    gripper_center = 0.5 * (left_pos + right_pos)

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