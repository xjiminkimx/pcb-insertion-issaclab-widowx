"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/grasp shaping, regularization, rail reset, and drop detection.

Two-phase task support (grasp edge centre → push +Y into slot):
  - Standalone grasp / push envs use separate reward configs (no gating).
  - Full env gates rewards with ``task_phase_gate`` (``"grasp"`` | ``"push"``) and
    :func:`task_phase_transition_step` to advance after :func:`grasp_edge_center_achieved`.
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


# World push axis for slot insertion (must match ``PUSH_AXIS_WORLD`` in env cfg).
_DEFAULT_PUSH_AXIS_WORLD = (0.0, 1.0, 0.0)


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


# ---------------------------------------------------------------------------
# Two-phase task state (grasp → push), per parallel env
# ---------------------------------------------------------------------------
_TASK_PHASE: torch.Tensor | None = None  # 0 = grasp, 1 = push


def _ensure_task_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    global _TASK_PHASE
    if _TASK_PHASE is None or _TASK_PHASE.shape[0] != env.num_envs or _TASK_PHASE.device != env.device:
        _TASK_PHASE = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return _TASK_PHASE


def reset_task_phase_on_reset(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event (mode=reset): set task phase back to grasp for reset envs."""
    phase = _ensure_task_phase(env)
    phase[env_ids] = 0


def _apply_task_phase_gate(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    task_phase_gate: str | None,
) -> torch.Tensor:
    if task_phase_gate is None:
        return reward
    phase = _ensure_task_phase(env)
    phase_id = 0 if task_phase_gate == "grasp" else 1
    return reward * (phase == phase_id).to(reward.dtype)


def _gripper_mid_trailing_edge_errors(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    target_offset_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCB-frame errors vs the trailing short-edge **face centre** target.

    Returns ``along, width, thick, in_plane, edge_dist`` where width is body +Y (short-edge width).
    """
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
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


def _trailing_edge_along_gate(
    along: torch.Tensor,
    along_sigma_m: float = 0.025,
) -> torch.Tensor:
    """≈1 at the **trailing** short edge (|along| small); ≈0 at the leading (slot-side) edge."""
    return torch.exp(-torch.abs(along) / (float(along_sigma_m) + 1e-9))


def gripper_leading_edge_grasp_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.10,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for the jaw midpoint near the **leading** (slot-side) short edge during grasp."""
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    lead = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    dist = torch.norm(mid - lead, dim=-1)
    near = torch.exp(-dist / (float(gate_dist_m) + 1e-9))
    return _apply_task_phase_gate(env, near, task_phase_gate)


def gripper_mid_to_pcb_trailing_edge_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    width_weight: float = 3.0,
) -> torch.Tensor:
    """Distance to the trailing short-edge **face centre** (width-weighted vs corners)."""
    along, width, thick, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    ww = float(width_weight)
    return torch.sqrt(along * along + (ww * width) * (ww * width) + thick * thick + 1e-12)


def gripper_short_edge_width_centering_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    sigma_frac: float = 0.22,
    gate_dist_m: float = 0.10,
    max_thick_m: float = 0.012,
    thick_sigma_m: float = 0.006,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Shaping in ``[0, 1]``: jaw centred on the short edge along body +Y."""
    _, width, thick, _, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    sigma_w = float(half_width_m) * float(sigma_frac) + 1e-9
    center = torch.exp(-torch.square(width / sigma_w))
    near = torch.exp(-edge_dist / (float(gate_dist_m) + 1e-9))
    height_gate = torch.exp(-torch.relu(torch.abs(thick) - float(max_thick_m)) / (float(thick_sigma_m) + 1e-9))
    return _apply_task_phase_gate(env, center * near * height_gate, task_phase_gate)


def gripper_short_edge_corner_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    corner_frac: float = 0.45,
    gate_dist_m: float = 0.10,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty in ``[0, 1]`` for hugging a short-edge corner (large ``|width|``)."""
    _, width, _, _, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    hw = float(half_width_m) + 1e-9
    near = torch.exp(-edge_dist / (float(gate_dist_m) + 1e-9))
    corner = torch.relu(torch.abs(width) / hw - float(corner_frac))
    out = near * torch.clamp(corner / (1.0 - float(corner_frac) + 1e-9), 0.0, 1.0)
    return _apply_task_phase_gate(env, out, task_phase_gate)


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
    width_frac: float = 0.30,
    min_pinch_ready: float = 0.55,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """True when the gripper has a valid closed pinch on the trailing short-edge centre."""
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        width_weight=width_weight,
    )
    _, width, _, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    closed = gq < float(closed_threshold) * float(open_width_m)
    near = dist < float(gate_dist_m)
    centered = torch.abs(width) < float(half_width_m) * float(width_frac)
    pinch = gripper_pinch_readiness(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
    )
    return closed & near & centered & (pinch >= float(min_pinch_ready))


def task_phase_transition_step(
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
    min_pinch_ready: float = 0.55,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
) -> torch.Tensor:
    """Side-effect reward (return 0): advance grasp → push when :func:`grasp_edge_center_achieved`."""
    phase = _ensure_task_phase(env)
    first = env.episode_length_buf == 1
    phase[:] = torch.where(first, torch.zeros_like(phase), phase)
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
    )
    phase[:] = torch.where((phase == 0) & achieved, torch.ones_like(phase), phase)
    return torch.zeros(env.num_envs, device=env.device, dtype=env.scene[pcb_cfg.name].data.root_pos_w.dtype)


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
    width_frac: float = 0.30,
    min_pinch_ready: float = 0.55,
    width_weight: float = 3.0,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    task_phase_gate: str | None = None,
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
    )
    bonus = achieved.to(dtype=env.scene[pcb_cfg.name].data.root_pos_w.dtype)
    return _apply_task_phase_gate(env, bonus, task_phase_gate)


# Per-env previous XY distance to trailing edge (ignores thickness).
_EE_XY_APPROACH_PREV_DIST: torch.Tensor | None = None


def ee_xy_approach_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_step_m: float = 0.05,
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Progress reward for moving closer in **XY only** to the trailing short-edge centre.

    Ignores thickness offset so Z descent is not double-counted with ``ee_thickness_descent_progress_reward``.
    """
    global _EE_XY_APPROACH_PREV_DIST

    along, width, _, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    ww = float(width_weight)
    dist = torch.sqrt(along * along + (ww * width) * (ww * width) + 1e-12)

    if (
        _EE_XY_APPROACH_PREV_DIST is None
        or _EE_XY_APPROACH_PREV_DIST.shape[0] != dist.shape[0]
        or _EE_XY_APPROACH_PREV_DIST.device != dist.device
    ):
        _EE_XY_APPROACH_PREV_DIST = dist.clone()
        return torch.zeros_like(dist)

    first_step = env.episode_length_buf == 1
    _EE_XY_APPROACH_PREV_DIST = torch.where(first_step, dist, _EE_XY_APPROACH_PREV_DIST)
    progress = (_EE_XY_APPROACH_PREV_DIST - dist).clamp(0.0, float(max_step_m))
    _EE_XY_APPROACH_PREV_DIST = dist.clone()
    return _apply_task_phase_gate(env, progress / float(max_step_m), task_phase_gate)


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
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Reward **closing** when near the trailing edge and above the board.

    Returns ``closure * gate * above_gate * pinch_soft`` in ``[0, 1]``.
    ``pinch_soft = 0.25 + 0.75 * pinch_ready`` so partial closure is rewarded near the edge
    even before perfect pinch alignment.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, width_weight=width_weight
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
    pinch_soft = 0.25 + 0.75 * pinch_ready

    return _apply_task_phase_gate(env, closure * gate * above_gate * pinch_soft, task_phase_gate)


def gripper_mid_thickness_plane_alignment_shaping(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    sigma_m: float = 0.012,
    half_length_m: float | None = None,
    gate_dist_m: float | None = None,
    max_thick_m: float = 0.010,
    thick_sigma_m: float = 0.006,
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
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
            env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, width_weight=width_weight
        )
        alignment = alignment * torch.exp(-dist / gate_dist_m)
        _, _, thick, _, _ = _gripper_mid_trailing_edge_errors(
            env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
        )
        height_gate = torch.exp(-torch.relu(torch.abs(thick) - float(max_thick_m)) / (float(thick_sigma_m) + 1e-9))
        alignment = alignment * height_gate

    return _apply_task_phase_gate(env, alignment, task_phase_gate)


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
    task_phase_gate: str | None = None,
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
    sign = pcb_body_x_push_sign(env, pcb_cfg)
    d_plane = torch.abs(torch.sum((mid - pcb_pos) * x_w, dim=-1) + sign * float(half_length_m))
    gq = env.scene[gripper_joint_cfg.name].data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / open_width_m, 0.0, 1.0)
    rub_plane = torch.exp(-d_plane / plane_band_m)
    rub_thick = torch.exp(-torch.abs(w_coord) / (thickness_band_m + 1e-6))
    # No side-rub penalty while descending from above the board top.
    not_top_down = 1.0 - torch.sigmoid((w_coord - pcb_half_thickness_m) / above_sigma_m)
    out = open_norm * rub_plane * rub_thick * not_top_down
    return _apply_task_phase_gate(env, out, task_phase_gate)


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
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for closing (or partially closing) near the edge before pinch geometry is ready.

    Returns ``closed * near_gate * (1 - pinch_ready)`` in ``[0, 1]``.
    """
    dist = gripper_mid_to_pcb_trailing_edge_distance(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, width_weight=width_weight
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
    return _apply_task_phase_gate(env, closed * near_gate * (1.0 - pinch_ready), task_phase_gate)


# Per-env previous |thick| for thickness descent progress (reset each episode).
_EE_THICK_PREV: torch.Tensor | None = None


def ee_thickness_descent_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    near_in_plane_m: float = 0.035,
    near_along_m: float = 0.025,
    max_step_m: float = 0.004,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Progress reward for lowering the jaw toward the short-edge pinch plane when XY-near the edge.

    Gated on ``|along|`` so descent is **not** rewarded at the leading (slot-side) short edge —
    ``in_plane`` alone is ~0 at both short edges.
    """
    global _EE_THICK_PREV

    along, _, thick, in_plane, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    abs_thick = torch.abs(thick)
    along_gate = _trailing_edge_along_gate(along, near_along_m)
    near = along_gate * torch.exp(-in_plane / (float(near_in_plane_m) + 1e-9))

    if (
        _EE_THICK_PREV is None
        or _EE_THICK_PREV.shape[0] != abs_thick.shape[0]
        or _EE_THICK_PREV.device != abs_thick.device
    ):
        _EE_THICK_PREV = abs_thick.clone()
        return torch.zeros_like(abs_thick)

    first_step = env.episode_length_buf == 1
    _EE_THICK_PREV = torch.where(first_step, abs_thick, _EE_THICK_PREV)
    progress = (_EE_THICK_PREV - abs_thick).clamp(0.0, float(max_step_m))
    _EE_THICK_PREV = abs_thick.clone()
    return _apply_task_phase_gate(env, near * progress / float(max_step_m), task_phase_gate)


def gripper_top_face_strike_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float,
    near_in_plane_m: float = 0.035,
    near_along_m: float = 0.025,
    gate_dist_m: float = 0.10,
    strike_thick_m: float = 0.008,
    strike_sigma_m: float = 0.004,
    min_open_fraction: float = 0.65,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for an **open** gripper hovering or tapping over the board top near the trailing edge."""
    along, _, thick, in_plane, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    along_gate = _trailing_edge_along_gate(along, near_along_m)
    near_xy = along_gate * torch.exp(-in_plane / (float(near_in_plane_m) + 1e-9))
    near_edge = torch.exp(-edge_dist / (float(gate_dist_m) + 1e-9))
    above_pinch = 1.0 - torch.exp(
        -torch.relu(torch.abs(thick) - float(strike_thick_m)) / (float(strike_sigma_m) + 1e-9)
    )

    robot = env.scene[gripper_joint_cfg.name]
    gq = robot.data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / float(open_width_m), 0.0, 1.0)
    open_ok = (open_norm >= float(min_open_fraction)).to(dtype=open_norm.dtype)

    out = near_xy * near_edge * above_pinch * open_ok
    return _apply_task_phase_gate(env, out, task_phase_gate)


def gripper_vertical_bounce_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    open_width_m: float,
    near_in_plane_m: float = 0.035,
    near_along_m: float = 0.025,
    gate_dist_m: float = 0.10,
    speed_scale_m_s: float = 0.06,
    min_open_fraction: float = 0.65,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for large vertical jaw speed while open and XY-near the trailing edge (Z oscillation)."""
    along, _, _, in_plane, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    along_gate = _trailing_edge_along_gate(along, near_along_m)
    near_xy = along_gate * torch.exp(-in_plane / (float(near_in_plane_m) + 1e-9))
    near_edge = torch.exp(-edge_dist / (float(gate_dist_m) + 1e-9))

    robot = env.scene[left_finger_cfg.name]
    left_vel = robot.data.body_lin_vel_w[:, left_finger_cfg.body_ids[0]]
    right_vel = robot.data.body_lin_vel_w[:, right_finger_cfg.body_ids[0]]
    mid_vel = 0.5 * (left_vel + right_vel)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    v_thick = torch.abs(torch.sum(mid_vel * z_w, dim=-1))
    speed_norm = torch.clamp(v_thick / (float(speed_scale_m_s) + 1e-9), 0.0, 1.0)

    gq = env.scene[gripper_joint_cfg.name].data.joint_pos[:, gripper_joint_cfg.joint_ids[0]]
    open_norm = torch.clamp(gq / float(open_width_m), 0.0, 1.0)
    open_ok = (open_norm >= float(min_open_fraction)).to(dtype=open_norm.dtype)

    out = near_xy * near_edge * speed_norm * open_ok
    return _apply_task_phase_gate(env, out, task_phase_gate)


def closed_below_pcb_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    pcb_half_thickness_m: float = 0.00125,
    open_width_m: float = 0.010,
    sigma_below_m: float = 0.005,
    task_phase_gate: str | None = None,
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

    return _apply_task_phase_gate(env, closed_norm * below_signal, task_phase_gate)


def gripper_along_board_slip_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    near_cross_m: float = 0.025,
    near_along_m: float = 0.025,
    along_sigma_m: float = 0.035,
    near_edge_gate_m: float = 0.14,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for sliding along the PCB long axis near the trailing-edge line (Y scrape between rails)."""
    along, _, _, in_plane, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    on_trailing = _trailing_edge_along_gate(along, near_along_m)
    on_board_line = on_trailing * torch.exp(-in_plane / (near_cross_m + 1e-9))
    along_slip = 1.0 - torch.exp(-torch.abs(along) / (along_sigma_m + 1e-9))
    near_edge = torch.exp(-edge_dist / (near_edge_gate_m + 1e-9))
    return _apply_task_phase_gate(env, on_board_line * along_slip * near_edge, task_phase_gate)


def gripper_mid_long_axis_speed_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    gate_dist_m: float = 0.14,
    speed_scale_m_s: float = 0.08,
    thickness_sigma_m: float = 0.006,
    min_finger_sep_m: float = 0.006,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Penalty for jaw midpoint speed along the PCB long axis before pinch-ready (Y swing)."""
    robot = env.scene[left_finger_cfg.name]
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    left_vel = robot.data.body_lin_vel_w[:, left_finger_cfg.body_ids[0]]
    right_vel = robot.data.body_lin_vel_w[:, right_finger_cfg.body_ids[0]]
    mid_vel = 0.5 * (left_vel + right_vel)
    long_axis = pcb_body_axis_x_world(env, pcb_cfg)
    v_along = torch.abs(torch.sum(mid_vel * long_axis, dim=-1))

    _, _, _, _, edge_dist = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    near = torch.exp(-edge_dist / (gate_dist_m + 1e-9))
    ready = gripper_pinch_readiness(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        thickness_sigma_m=thickness_sigma_m,
        min_finger_sep_m=min_finger_sep_m,
    )
    speed_norm = torch.clamp(v_along / (speed_scale_m_s + 1e-9), 0.0, 1.0)
    return _apply_task_phase_gate(env, speed_norm * near * (1.0 - ready), task_phase_gate)


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
    max_thick_m: float = 0.010,
    thick_sigma_m: float = 0.006,
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
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
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, width_weight=width_weight
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
    _, _, thick, _, _ = _gripper_mid_trailing_edge_errors(
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m
    )
    height_gate = torch.exp(-torch.relu(torch.abs(thick) - float(max_thick_m)) / (float(thick_sigma_m) + 1e-9))
    return _apply_task_phase_gate(env, align_open * align_finger * sep_ok * gate * height_gate, task_phase_gate)


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


def pcb_lin_vel_y_toward_lead_target_y(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
    task_phase_gate: str | None = None,
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
    return _apply_task_phase_gate(env, torch.relu(err_y * v_y), task_phase_gate)


def pcb_long_axis_parallel_to_push_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Shaped in ``[0, 1]``: PCB body +X (long edge) aligned with the insertion direction.

    1.0 when the long axis is parallel to ``axis_world`` (same or opposite direction).
    """
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    a = torch.tensor(axis_world, device=env.device, dtype=x_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    c = torch.abs(torch.sum(x_w * a.unsqueeze(0).expand_as(x_w), dim=-1))
    return _apply_task_phase_gate(env, torch.square(torch.clamp(c, max=1.0)), task_phase_gate)


def pcb_leading_edge_insertion_proximity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xyz_env: tuple[float, float, float],
    sigma_m: float = 0.12,
    task_phase_gate: str | None = None,
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
    return _apply_task_phase_gate(env, torch.exp(-dist / (float(sigma_m) + 1e-9)), task_phase_gate)


# Per-env previous EE distance used by the progress reward (reset each episode).
_EE_APPROACH_PREV_DIST: torch.Tensor | None = None


def ee_approach_progress_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    max_step_m: float = 0.05,
    width_weight: float = 3.0,
    task_phase_gate: str | None = None,
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
        env, pcb_cfg, left_finger_cfg, right_finger_cfg, half_length_m, width_weight=width_weight
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
    return _apply_task_phase_gate(env, progress / float(max_step_m), task_phase_gate)


def pcb_insertion_depth_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    slot_mouth_y_env: float,
    max_depth_m: float = 0.20,
    task_phase_gate: str | None = None,
) -> torch.Tensor:
    """Reward for **PCB depth inside the slot**: leading edge past the slot mouth in +Y.

    Zero while the leading edge has not yet crossed ``slot_mouth_y_env``.
    Linearly increases up to ``max_depth_m`` of penetration (returns 1.0 at full insertion).
    Use a positive weight; combine with ``push_y_toward_slot`` which only fires before the mouth.
    """
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m)
    lead_y = (lead_w - env.scene.env_origins[:, :3])[:, 1]
    depth = torch.clamp(lead_y - float(slot_mouth_y_env), min=0.0, max=float(max_depth_m))
    return _apply_task_phase_gate(env, depth / float(max_depth_m), task_phase_gate)


def pcb_horizontal_velocity_perpendicular_to_axis_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
    task_phase_gate: str | None = None,
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
    return _apply_task_phase_gate(env, torch.sum(torch.square(v_perp), dim=-1), task_phase_gate)


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
    left = robot.data.body_pos_w[env_ids][:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[env_ids][:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
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