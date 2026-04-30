"""Custom MDP terms for the WidowX PCB insertion task.

This module contains:
- observation helpers (e.g., end-effector position),
- reward helpers (distance/height/regularization),
- reset helpers (spawn PCB in a stable in-gripper pose),
- termination helpers (detect dropped PCB),
- target-slot geometry helpers.

Design note:
The insertion target is represented as a local offset from the magazine root.
At runtime we rotate this offset by magazine orientation so the slot target stays
correct even if the magazine rotates.
"""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------
# 관측 (Observations)
# ---------------------------------------------------------
def ee_position_env(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return selected end-effector body position in env-local coordinates.

    Why env-local:
    - This subtracts `env_origins` so each parallel environment has a consistent
      coordinate frame regardless of tiling offset in the world.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0]] - env.scene.env_origins


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
def object_ee_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ee_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distance between PCB center and selected end-effector body."""
    pcb = env.scene[asset_cfg.name]
    robot = env.scene[ee_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    ee_pos = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    return torch.norm(pcb_pos - ee_pos, dim=-1)


def object_gripper_midpoint_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Distance from PCB root to the midpoint between fingertip bodies (matches grasp-centric resets)."""
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    return torch.norm(pcb_pos - mid, dim=-1)


def pcb_grasp_deviation(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    expected_distance: float | None = None,
    expected_offset_world: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Penalize deviation from the nominal grasp pose (PCB center relative to jaw midpoint).

    Prefer ``expected_offset_world``: vector from jaw midpoint to PCB center in **world** frame
    (e.g. ``(0, PCB_X/2, dz)`` for trailing grasp with a vertical correction ``dz`` when link COMs
    sit above the real pads).

    If only ``expected_distance`` is set (scalar), falls back to ``|‖pcb−mid‖ − expected_distance|``
    (1D radius — wrong when a Z correction is needed).
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[left_finger_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    left = robot.data.body_pos_w[:, left_finger_cfg.body_ids[0]]
    right = robot.data.body_pos_w[:, right_finger_cfg.body_ids[0]]
    mid = 0.5 * (left + right)
    if expected_offset_world is not None:
        off = torch.tensor(expected_offset_world, device=env.device, dtype=pcb_pos.dtype).unsqueeze(0).expand_as(pcb_pos)
        expected_pos = mid + off
        return torch.norm(pcb_pos - expected_pos, dim=-1)
    actual = torch.norm(pcb_pos - mid, dim=-1)
    exp = expected_distance if expected_distance is not None else 0.12
    return torch.abs(actual - exp)


def object_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return PCB height above per-env origin (z-axis)."""
    pcb = env.scene[asset_cfg.name]
    return pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

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

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    조인트 속도 패널티: 
    관절이 너무 빠른 속도로 회전하는 것을 방지하여 에너지를 절약하고 
    실제 하드웨어(WidowX)의 모터 마모를 줄입니다.
    """
    asset = env.scene[asset_cfg.name]  # robot articulation
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def pcb_to_slot_distance(env: ManagerBasedRLEnv, pcb_cfg: SceneEntityCfg, magazine_cfg: SceneEntityCfg) -> torch.Tensor:
    """Legacy helper: distance from PCB to magazine root + fixed offset.

    Note:
    - Kept for compatibility/reference.
    - Prefer `pcb_to_target_slot_distance()` for current training terms.
    """
    pcb = env.scene[pcb_cfg.name]
    magazine = env.scene[magazine_cfg.name]
    
    pcb_pos = pcb.data.root_pos_w
    mag_pos = magazine.data.root_pos_w
    
    # ✅ 핵심: 매거진 중심(Root)에서 슬롯 중심까지의 거리를 입력합니다. (단위: 미터)
    # GUI의 Slot_01 Translate 값이 만약 (X: 0, Y: 0, Z: 5cm) 라면 [0.0, 0.0, 0.05] 로 적습니다.
    # (GUI 값이 0, 0, 0 이라면 그냥 [0.0, 0.0, 0.0] 으로 두시면 됩니다.)
    slot_offset = torch.tensor([0.0, 0.0, 0.0], device=env.device)
    
    # 환경(64개)의 모든 매거진 좌표에 오프셋을 일괄적으로 더해 최종 목표 좌표를 만듭니다.
    target_slot_pos = mag_pos + slot_offset
    
    return torch.norm(target_slot_pos - pcb_pos, dim=-1)


def target_slot_position(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Compute world-frame slot center from magazine root + local slot offset.

    Args:
        target_cfg: The target asset (magazine root body).
        slot_offset: Slot center offset in target asset *local frame* (meters).
    """
    target_asset = env.scene[target_cfg.name]  # magazine rigid object
    target_pos = target_asset.data.root_pos_w
    target_quat = target_asset.data.root_quat_w
    offset_local = torch.tensor(slot_offset, device=env.device, dtype=target_pos.dtype).unsqueeze(0).repeat(target_pos.shape[0], 1)
    return target_pos + math_utils.quat_apply(target_quat, offset_local)


def pcb_to_target_slot_distance(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Distance between PCB center and configured slot center (L2 norm)."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    return torch.norm(slot_pos - pcb_pos, dim=-1)


def pcb_vertical_gap_to_slot(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Absolute vertical (world Z) gap between PCB center and slot target.

    Use with a *negative* reward weight so the policy is pushed to match slot height
    (reduces "hover above magazine" local optima from 3D distance alone).
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    return torch.abs(pcb_pos[:, 2] - slot_pos[:, 2])


def pcb_xy_distance_to_slot(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Horizontal (XY) distance from PCB to slot — encourages plan-view alignment before / during insert."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    delta = pcb_pos - slot_pos
    return torch.norm(delta[:, :2], dim=-1)


def _magazine_insertion_axis_w(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    insertion_axis_local: tuple[float, float, float],
) -> torch.Tensor:
    """World-frame unit vector along insertion direction from magazine orientation."""
    target_asset = env.scene[target_cfg.name]
    q = target_asset.data.root_quat_w
    axis_local = torch.tensor(insertion_axis_local, device=env.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    axis_w = math_utils.quat_apply(q, axis_local)
    return axis_w / torch.norm(axis_w, dim=-1, keepdim=True).clamp_min(1e-6)


def pcb_perpendicular_distance_to_insertion_axis(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    insertion_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Lateral error: norm of PCB-to-slot vector projected onto plane perpendicular to insertion axis.

    Use with **negative** reward weight. Minimizing this aligns the PCB with the slot opening when the
    magazine is tilted or rotated — unlike world-Z or XY-only terms, which assume a vertical slot.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    delta = pcb_pos - slot_pos
    axis_w = _magazine_insertion_axis_w(env, target_cfg, insertion_axis_local)
    parallel = torch.sum(delta * axis_w, dim=-1, keepdim=True) * axis_w
    perp = delta - parallel
    return torch.norm(perp, dim=-1)


def pcb_parallel_distance_along_insertion_axis(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    insertion_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> torch.Tensor:
    """Depth along insertion axis: |dot(pcb - slot, axis_w)| — distance to slot center measured along insert direction.

    Use with **negative** weight to reward sliding in along the pocket axis (reduces hovering off-axis).
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    delta = pcb_pos - slot_pos
    axis_w = _magazine_insertion_axis_w(env, target_cfg, insertion_axis_local)
    return torch.abs(torch.sum(delta * axis_w, dim=-1))


def pcb_insertion_wrong_side_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    insertion_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    lateral_scale: float = 8.0,
) -> torch.Tensor:
    """Penalize PCB on the *wrong* side of the slot along the insertion axis.

    Convention (matches scene comments in ``widowx_pcb_env_cfg``):
    - ``insertion_axis_local`` is rotated to world ``axis_w`` = direction the PCB **travels to enter**
      the magazine (into the slot).
    - While approaching correctly, ``dot(pcb - slot, axis_w) < 0`` (PCB still "before" the slot).
    - ``dot(pcb - slot, axis_w) > 0`` can mean past the opening **or** past the slot *center* when
      fully inserted; we scale by lateral misalignment so on-axis insertion is not over-penalized.

    Returns ``relu(axial) * (1 + lateral_scale * perp)`` with ``perp`` = perpendicular distance to
    the insertion line — use a **negative** reward weight.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    delta = pcb_pos - slot_pos
    axis_w = _magazine_insertion_axis_w(env, target_cfg, insertion_axis_local)
    axial = torch.sum(delta * axis_w, dim=-1)
    parallel = torch.sum(delta * axis_w, dim=-1, keepdim=True) * axis_w
    perp = delta - parallel
    perp_norm = torch.norm(perp, dim=-1)
    wrong = torch.clamp(axial, min=0.0)
    return wrong * (1.0 + lateral_scale * perp_norm)


def pcb_height_below_reference(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    min_height: float = 0.04,
) -> torch.Tensor:
    """Penalty when PCB center is below a reference height (env-local Z).

    Fights dragging the board on the floor / magazine base to minimize 3D distance.
    Same height convention as ``object_height``: ``pcb_z - env_origin_z``.

    Returns ``relu(min_height - pcb_height_env)`` — use a **negative** weight.
    """
    pcb = env.scene[pcb_cfg.name]
    h = pcb.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(min_height - h, min=0.0)


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


def pcb_horizontal_offset_perpendicular_to_insertion(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    insertion_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Plan-view lateral error: in-plane distance perpendicular to insertion axis only.

    Unlike ``pcb_xy_distance_to_slot``, this does **not** penalize offset *along* the insertion
    direction in the horizontal plane (e.g. world +Y when ``axis_w`` has no Z). That avoids
    fighting the depth term while the PCB slides into the slot.

    Projects ``(pcb - slot)`` onto the horizontal plane (Z=0 in world), removes the component
    along the insertion axis, and returns the norm of the remainder.
    """
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    delta = pcb_pos - slot_pos
    delta_h = delta.clone()
    delta_h[:, 2] = 0.0
    axis_w = _magazine_insertion_axis_w(env, target_cfg, insertion_axis_local)
    axis_h = axis_w.clone()
    axis_h[:, 2] = 0.0
    axis_h_norm = torch.norm(axis_h, dim=-1, keepdim=True).clamp_min(1e-6)
    axis_h = axis_h / axis_h_norm
    parallel_h = torch.sum(delta_h * axis_h, dim=-1, keepdim=True) * axis_h
    perp_h = delta_h - parallel_h
    return torch.norm(perp_h, dim=-1)


def is_success_and_stable(
    env: "ManagerBasedRLEnv",
    pcb_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    dist_threshold: float = 0.015,  # 1.5cm 이내
    vel_threshold: float = 0.05,    # 0.05 m/s 이하 (정지 상태)
) -> torch.Tensor:
    """
    Check whether PCB is inserted and stable.
    
    Args:
        env: Isaac Lab 환경 객체
        pcb_cfg: PCB 자산(Asset) 설정
        target_cfg: 목표 매거진(Magazine) 자산 설정
        dist_threshold: 삽입 성공으로 간주할 최대 거리 (미터 단위)
        vel_threshold: 정지 상태로 간주할 최대 선속도 (m/s)
        
    Returns:
        torch.Tensor: 각 환경별 성공 여부 (Boolean Tensor)
    """
    # 1) Read current PCB state.
    pcb_asset = env.scene[pcb_cfg.name]
    # PCB의 월드 좌표와 속도
    pcb_pos = pcb_asset.data.root_pos_w  # [num_envs, 3]
    pcb_vel = pcb_asset.data.root_lin_vel_w  # [num_envs, 3]

    # 매거진 슬롯의 월드 좌표 (Target)
    target_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)

    # 2) Compute insertion distance to designated slot center.
    distance = torch.norm(pcb_pos - target_pos, dim=-1)

    # 3) Compute translational speed magnitude.
    velocity_mag = torch.norm(pcb_vel, dim=-1)

    # 4) Success requires both near-slot and low-speed conditions.
    is_near = distance < dist_threshold
    is_static = velocity_mag < vel_threshold

    # 두 조건이 모두 만족될 때 True 반환
    return is_near & is_static


def reset_pcb_in_gripper(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.01),
    edge_to_center_offset: float = 0.0,
    edge_sign: float = 1.0,
    velocity_scale: float = 0.0,
    # --- Fixed-orientation mode (recommended for on-rail grasps) ---
    # When set, the PCB is always spawned with this exact world-frame quaternion (w,x,y,z).
    # The gripper closing axis is then IGNORED for orientation; only EE XY is used for position.
    # PCB local +X = long edge (PCB_X), local +Z = thickness.
    # For a flat board with long axis along world +Y: (0.7071068, 0, 0, 0.7071068).
    fixed_world_quat_wxyz: tuple[float, float, float, float] | None = None,
    # World Z of PCB *center* to snap to after XY placement (= rail_top + PCB_Z/2).
    # Prevents the board from floating or burying because the EE is at a wrong height.
    snap_center_z: float | None = None,
    # Added in world frame after XY/long-axis placement (and after optional snap_center_z on Z).
    # Finger-link origins in USD are often above the real jaw pads → use negative Z to drop the PCB onto the pads.
    world_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    # --- Legacy / insertion-axis alignment (only used when fixed_world_quat_wxyz is None) ---
    insertion_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0),
    insertion_target_cfg: SceneEntityCfg | None = None,
) -> None:
    """Reset PCB to a pose at the gripper with configurable orientation.

    **Recommended path (fixed-orientation)**:
    Pass ``fixed_world_quat_wxyz`` (e.g. flat board, long axis along insertion) and
    ``snap_center_z`` (= rail top + PCB_Z/2).  The board will always be flat and at the
    correct height; only the gripper XY is used to place the leading edge.

    **Legacy path** (``fixed_world_quat_wxyz is None``):
    Derives PCB orientation from the gripper closing axis.  The result depends on
    arm joint angles — if the gripper axis is not world +Z the board stands on its edge.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[ee_cfg.name]

    ee_pos = robot.data.body_pos_w[env_ids, ee_cfg.body_ids[0]]
    ee_quat = robot.data.body_quat_w[env_ids, ee_cfg.body_ids[0]]

    if fixed_world_quat_wxyz is not None:
        # ---- Fixed-orientation path ----------------------------------------
        # Orientation is fully specified; only EE XY position feeds placement.
        N = len(env_ids)
        q = torch.tensor(fixed_world_quat_wxyz, device=env.device, dtype=ee_pos.dtype)
        target_quat = q.unsqueeze(0).expand(N, -1)

        # Use gripper midpoint if fingers provided (better XY accuracy), else EE link.
        if left_finger_cfg is not None and right_finger_cfg is not None:
            left_pos = robot.data.body_pos_w[env_ids, left_finger_cfg.body_ids[0]]
            right_pos = robot.data.body_pos_w[env_ids, right_finger_cfg.body_ids[0]]
            ee_pos = 0.5 * (left_pos + right_pos)

        # PCB long axis in world = local +X rotated by fixed_world_quat_wxyz.
        local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=ee_pos.dtype)
        local_x = local_x.unsqueeze(0).expand(N, -1)
        pcb_long_axis_w = math_utils.quat_apply(target_quat, local_x)
        pcb_long_axis_w = pcb_long_axis_w / torch.norm(pcb_long_axis_w, dim=-1, keepdim=True).clamp_min(1e-6)

        # Leading edge at EE, shift center back along the long axis.
        target_pos = ee_pos.clone()
        if edge_to_center_offset != 0.0:
            target_pos = target_pos - edge_sign * edge_to_center_offset * pcb_long_axis_w

        # Snap PCB center to exact rail height if provided.
        if snap_center_z is not None:
            target_pos = target_pos.clone()
            target_pos[:, 2] = snap_center_z

        ox, oy, oz = world_position_offset
        if abs(ox) + abs(oy) + abs(oz) > 1e-9:
            woff = torch.tensor(world_position_offset, device=env.device, dtype=target_pos.dtype).unsqueeze(0).expand(N, -1)
            target_pos = target_pos + woff

    else:
        # ---- Legacy / grip-axis path ----------------------------------------
        grip_axis_w = None
        pcb_long_axis_w = None

        if left_finger_cfg is not None and right_finger_cfg is not None:
            left_pos = robot.data.body_pos_w[env_ids, left_finger_cfg.body_ids[0]]
            right_pos = robot.data.body_pos_w[env_ids, right_finger_cfg.body_ids[0]]
            ee_pos = 0.5 * (left_pos + right_pos)
            grip_axis_w = right_pos - left_pos
            grip_axis_w = grip_axis_w / torch.norm(grip_axis_w, dim=-1, keepdim=True).clamp_min(1e-6)

            ee_forward_w = math_utils.quat_apply(
                ee_quat,
                torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=ee_pos.dtype).unsqueeze(0).repeat(len(env_ids), 1),
            )
            pcb_long_axis_w = ee_forward_w - torch.sum(ee_forward_w * grip_axis_w, dim=-1, keepdim=True) * grip_axis_w
            fallback = math_utils.quat_apply(
                ee_quat,
                torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=ee_pos.dtype).unsqueeze(0).repeat(len(env_ids), 1),
            )
            fallback = fallback - torch.sum(fallback * grip_axis_w, dim=-1, keepdim=True) * grip_axis_w
            use_fallback = torch.norm(pcb_long_axis_w, dim=-1, keepdim=True) < 1e-6
            pcb_long_axis_w = torch.where(use_fallback, fallback, pcb_long_axis_w)
            pcb_long_axis_w = pcb_long_axis_w / torch.norm(pcb_long_axis_w, dim=-1, keepdim=True).clamp_min(1e-6)

            if insertion_target_cfg is not None:
                insert_axis_w = _magazine_insertion_axis_w(env, insertion_target_cfg, insertion_axis_local)[env_ids]
                projected = insert_axis_w - torch.sum(insert_axis_w * grip_axis_w, dim=-1, keepdim=True) * grip_axis_w
                proj_len = torch.norm(projected, dim=-1, keepdim=True)
                projected = projected / proj_len.clamp_min(1e-6)
                pcb_long_axis_w = torch.where(proj_len < 1e-5, pcb_long_axis_w, projected)

        offset = torch.tensor(pos_offset, device=env.device, dtype=ee_pos.dtype).unsqueeze(0).repeat(len(env_ids), 1)
        target_pos = ee_pos + math_utils.quat_apply(ee_quat, offset)
        if pcb_long_axis_w is not None and edge_to_center_offset != 0.0:
            target_pos = target_pos - edge_sign * edge_to_center_offset * pcb_long_axis_w

        if snap_center_z is not None:
            target_pos = target_pos.clone()
            target_pos[:, 2] = snap_center_z

        if grip_axis_w is not None and pcb_long_axis_w is not None:
            pcb_width_axis_w = torch.linalg.cross(grip_axis_w, pcb_long_axis_w)
            pcb_width_axis_w = pcb_width_axis_w / torch.norm(pcb_width_axis_w, dim=-1, keepdim=True).clamp_min(1e-6)
            rot_mats = torch.stack([pcb_long_axis_w, pcb_width_axis_w, grip_axis_w], dim=-1)
            target_quat = math_utils.quat_from_matrix(rot_mats)
        else:
            target_quat = ee_quat

    default_root_state = pcb.data.default_root_state[env_ids].clone()
    root_pose = torch.cat([target_pos, target_quat], dim=-1)
    root_vel = default_root_state[:, 7:13] * velocity_scale

    pcb.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    pcb.write_root_velocity_to_sim(root_vel, env_ids=env_ids)


def reset_pcb_on_guide_rails(
    env: "ManagerBasedRLEnv",
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


def pcb_dropped_from_gripper(
    env: "ManagerBasedRLEnv",
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


def no_progress_termination(
    env: "ManagerBasedRLEnv",
    pcb_cfg: SceneEntityCfg,
    ee_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    slot_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_target_distance: float = 0.08,
    ee_speed_threshold: float = 0.01,
    pcb_speed_threshold: float = 0.01,
    min_episode_steps: int = 60,
) -> torch.Tensor:
    """Terminate episodes that are effectively dead/stuck.

    Condition (all must hold simultaneously):
    - at least ``min_episode_steps`` have elapsed since reset — prevents firing at t=0
      when the robot and PCB are both stationary immediately after initialization;
    - still far from target slot; and
    - both EE and PCB speeds are near zero.
    """
    pcb = env.scene[pcb_cfg.name]
    robot = env.scene[ee_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    slot_pos = target_slot_position(env, target_cfg=target_cfg, slot_offset=slot_offset)
    dist_to_target = torch.norm(pcb_pos - slot_pos, dim=-1)

    ee_vel = robot.data.body_lin_vel_w[:, ee_cfg.body_ids[0]]
    ee_speed = torch.norm(ee_vel, dim=-1)
    pcb_speed = torch.norm(pcb.data.root_lin_vel_w, dim=-1)

    # episode_length_buf counts policy steps since last reset (incremented before termination check).
    past_warmup = env.episode_length_buf >= min_episode_steps

    return past_warmup & (dist_to_target > min_target_distance) & (ee_speed < ee_speed_threshold) & (pcb_speed < pcb_speed_threshold)