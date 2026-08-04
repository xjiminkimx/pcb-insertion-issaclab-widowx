"""Custom MDP terms for the WidowX PCB on-rail task.

Observation helpers, push/insert shaping, regularization, rail reset, and drop detection.

Approach and Insert are separate registered envs; each uses its own reward config with no in-episode phase gating.
"""
from __future__ import annotations
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
_DEFAULT_PUSH_AXIS_WORLD = (0.0, 1.0, 0.0)
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg  # noqa: E402
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg  # noqa: E402
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction  # noqa: E402
class JointVariableImpedanceAction(ActionTerm):
    """RL action term wrapping :class:`JointImpedanceController` (variable K / ζ).

    Policy output layout (``impedance_mode="variable"``, 6 arm joints → 18 dims):

    - ``[0:6]``   relative joint position deltas (rad), scaled by ``position_scale``
    - ``[6:12]``  stiffness K (mapped from [-1, 1] → ``stiffness_limits``)
    - ``[12:18]`` damping ratio ζ (mapped from [-1, 1] → ``damping_ratio_limits``)

    Computed torques are sent with :meth:`Articulation.set_joint_effort_target`; arm actuators
    should keep ``stiffness=0`` so the implicit actuator does not fight the VIC torques.
    """

    cfg: JointVariableImpedanceActionCfg

    def __init__(self, cfg: JointVariableImpedanceActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if cfg.impedance_mode == "variable":
            self._blocks = 3
        elif cfg.impedance_mode == "variable_kp":
            self._blocks = 2
        else:
            raise ValueError(
                f"JointVariableImpedanceAction supports impedance_mode 'variable' or 'variable_kp',"
                f" got {cfg.impedance_mode!r}."
            )

        self._pos_scale = self._resolve_per_joint_scale(cfg.position_scale)
        k_lo, k_hi = cfg.stiffness_limits
        d_lo, d_hi = cfg.damping_ratio_limits
        self._stiffness_min = float(k_lo)
        self._stiffness_span = float(k_hi) - float(k_lo)
        self._damping_min = float(d_lo)
        self._damping_span = float(d_hi) - float(d_lo)

        dof_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :].clone()
        ctrl_cfg = JointImpedanceControllerCfg(
            command_type=cfg.command_type,
            impedance_mode=cfg.impedance_mode,
            stiffness=cfg.default_stiffness,
            damping_ratio=cfg.default_damping_ratio,
            stiffness_limits=cfg.stiffness_limits,
            damping_ratio_limits=cfg.damping_ratio_limits,
            gravity_compensation=cfg.gravity_compensation,
        )
        self._controller = JointImpedanceController(
            ctrl_cfg, env.num_envs, dof_limits, device=self.device
        )
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._command_buf = torch.zeros(
            self.num_envs, self._controller.num_actions, device=self.device
        )
        self._stiffness_cmd = torch.full(
            (self.num_envs, self._num_joints),
            float(cfg.default_stiffness),
            device=self.device,
            dtype=torch.float32,
        )
        self._damping_ratio_cmd = torch.full(
            (self.num_envs, self._num_joints),
            float(cfg.default_damping_ratio),
            device=self.device,
            dtype=torch.float32,
        )
        # Absolute posture setpoint (p_abs): latched to the joint pose at reset, then nudged by the
        # policy's scaled position deltas.  This gives K a real anchor so the arm actively holds its
        # height (no droop), unlike p_rel where the error re-references the current pose every substep
        # and the stiffness term vanishes at Delta q = 0.
        self._pos_setpoint = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._setpoint_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._dof_pos_limits = dof_limits  # (num_envs, num_joints, 2)
        self._max_setpoint_dev = (
            None if cfg.max_setpoint_deviation is None else float(cfg.max_setpoint_deviation)
        )

    @property
    def stiffness_cmd(self) -> torch.Tensor:
        """Current stiffness K (N·m/rad) per env × joint."""
        return self._stiffness_cmd

    @property
    def damping_ratio_cmd(self) -> torch.Tensor:
        """Current damping ratio ζ per env × joint."""
        return self._damping_ratio_cmd

    def stiffness_normalized(self) -> torch.Tensor:
        """Map K to [-1, 1] using ``stiffness_limits``."""
        return (
            2.0 * (self._stiffness_cmd - self._stiffness_min) / (self._stiffness_span + 1e-9) - 1.0
        )

    def damping_normalized(self) -> torch.Tensor:
        """Map ζ to [-1, 1] using ``damping_ratio_limits``."""
        return 2.0 * (self._damping_ratio_cmd - self._damping_min) / (self._damping_span + 1e-9) - 1.0

    def _resolve_per_joint_scale(self, scale: float | dict[str, float]) -> torch.Tensor:
        out = torch.ones(self.num_envs, self._num_joints, device=self.device)
        if isinstance(scale, (float, int)):
            out[:] = float(scale)
            return out
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            scale, self._joint_names, preserve_order=self.cfg.preserve_order
        )
        out[:, index_list] = torch.tensor(value_list, device=self.device)
        return out

    @staticmethod
    def _map_symmetric_to_range(actions: torch.Tensor, lo: float, span: float) -> torch.Tensor:
        """Map policy actions in [-1, 1] linearly to [lo, lo + span]."""
        return lo + 0.5 * (actions + 1.0) * span

    @property
    def action_dim(self) -> int:
        return self._num_joints * self._blocks

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical VIC command buffer: [Δq, K, (ζ)] per joint."""
        return self._processed_actions

    def _write_controller_command(self) -> None:
        """Push clipped commands into the impedance controller (bypasses buggy ``set_command`` clip)."""
        n = self._num_joints
        ctrl = self._controller
        k_lo, k_hi = self.cfg.stiffness_limits
        pos_cmd = self._command_buf[:, :n]
        stiff_cmd = self._command_buf[:, n : 2 * n].clamp(float(k_lo), float(k_hi))
        ctrl._dof_pos_target[:] = pos_cmd
        ctrl._p_gains[:] = stiff_cmd
        self._stiffness_cmd[:] = stiff_cmd
        if self._blocks == 3:
            d_lo, d_hi = self.cfg.damping_ratio_limits
            damp_cmd = self._command_buf[:, 2 * n : 3 * n].clamp(float(d_lo), float(d_hi))
            ctrl._d_gains[:] = 2.0 * torch.sqrt(stiff_cmd.clamp(min=1e-9)) * damp_cmd
            self._damping_ratio_cmd[:] = damp_cmd
        else:
            ctrl._d_gains[:] = 2.0 * torch.sqrt(stiff_cmd.clamp(min=1e-9))

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        n = self._num_joints
        pos_a = actions[:, :n].clamp(-1.0, 1.0)
        stiff_a = actions[:, n : 2 * n].clamp(-1.0, 1.0)
        pos_delta = pos_a * self._pos_scale
        stiff_cmd = self._map_symmetric_to_range(stiff_a, self._stiffness_min, self._stiffness_span)
        if self.cfg.command_type == "p_abs":
            # Latch the setpoint to the current (reset) posture the first control step after a reset,
            # then integrate the policy deltas.  Clamp to soft joint limits and (optionally) to a max
            # deviation from the live pose to avoid integral wind-up when contact blocks the arm.
            cur_q = self._asset.data.joint_pos[:, self._joint_ids]
            new_env = ~self._setpoint_initialized
            if bool(new_env.any()):
                self._pos_setpoint[new_env] = cur_q[new_env]
                self._setpoint_initialized[new_env] = True
            self._pos_setpoint += pos_delta
            if self._max_setpoint_dev is not None:
                self._pos_setpoint.clamp_(
                    cur_q - self._max_setpoint_dev, cur_q + self._max_setpoint_dev
                )
            self._pos_setpoint.clamp_(
                self._dof_pos_limits[..., 0], self._dof_pos_limits[..., 1]
            )
            pos_cmd = self._pos_setpoint
        else:
            pos_cmd = pos_delta
        self._command_buf[:, :n] = pos_cmd
        self._command_buf[:, n : 2 * n] = stiff_cmd
        if self._blocks == 3:
            damp_a = actions[:, 2 * n : 3 * n].clamp(-1.0, 1.0)
            damp_cmd = self._map_symmetric_to_range(damp_a, self._damping_min, self._damping_span)
            self._command_buf[:, 2 * n : 3 * n] = damp_cmd
        self._processed_actions[:] = self._command_buf
        self._write_controller_command()

    def apply_actions(self) -> None:
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        gravity = None
        if self.cfg.gravity_compensation:
            gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        torques = self._controller.compute(joint_pos, joint_vel, gravity=gravity)
        self._asset.set_joint_effort_target(torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._stiffness_cmd[:] = float(self.cfg.default_stiffness)
            self._damping_ratio_cmd[:] = float(self.cfg.default_damping_ratio)
            self._setpoint_initialized[:] = False
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._raw_actions[ids] = 0.0
            self._processed_actions[ids] = 0.0
            self._stiffness_cmd[ids] = float(self.cfg.default_stiffness)
            self._damping_ratio_cmd[ids] = float(self.cfg.default_damping_ratio)
            self._setpoint_initialized[ids] = False
        self._controller.reset_idx(
            None if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
def get_joint_variable_impedance_action(
    env: ManagerBasedEnv,
    action_name: str = "arm_action",
) -> JointVariableImpedanceAction | None:
    """Return the VIC action term if ``action_name`` is a :class:`JointVariableImpedanceAction`."""
    if not hasattr(env, "action_manager"):
        return None
    term = env.action_manager.get_term(action_name)
    if isinstance(term, JointVariableImpedanceAction):
        return term
    return None
def get_task_space_impedance_action(
    env: ManagerBasedEnv,
    action_name: str = "arm_action",
) -> OperationalSpaceControllerAction | None:
    """Return task-space OSC action term (``WidowXTaskSpaceImpedanceAction`` or base OSC)."""
    if not hasattr(env, "action_manager"):
        return None
    term = env.action_manager.get_term(action_name)
    if isinstance(term, OperationalSpaceControllerAction):
        return term
    return None
def vic_arm_stiffness_normalized_obs(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Policy obs: stiffness command in [-1, 1] (6 joints or 6 task axes)."""
    vic_term = get_joint_variable_impedance_action(env, action_name)
    if vic_term is not None:
        return vic_term.stiffness_normalized()
    osc_term = get_task_space_impedance_action(env, action_name)
    if osc_term is not None and osc_term._stiffness_idx is not None:
        idx = osc_term._stiffness_idx
        return osc_term.raw_actions[:, idx : idx + 6].clamp(-1.0, 1.0)
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)
def vic_arm_damping_normalized_obs(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Policy obs: damping ratio command in [-1, 1] (6 joints or 6 task axes)."""
    vic_term = get_joint_variable_impedance_action(env, action_name)
    if vic_term is not None:
        return vic_term.damping_normalized()
    osc_term = get_task_space_impedance_action(env, action_name)
    if osc_term is not None and osc_term._damping_ratio_idx is not None:
        idx = osc_term._damping_ratio_idx
        return osc_term.raw_actions[:, idx : idx + 6].clamp(-1.0, 1.0)
    return torch.zeros(env.num_envs, 6, device=env.device, dtype=torch.float32)
class JointVariableImpedanceActionCfg(ActionTermCfg):
    """Variable joint impedance: relative Δq, stiffness K, damping ratio ζ per controlled joint."""

    class_type: type[ActionTerm] = JointVariableImpedanceAction
    joint_names: list[str] = MISSING
    preserve_order: bool = True
    command_type: str = "p_rel"
    """``p_rel``: target = current q + Δq command; ``p_abs``: absolute joint targets."""
    impedance_mode: str = "variable"
    """``variable`` → action = [Δq, K, ζ] per joint (3×DoF dims). ``variable_kp`` → [Δq, K]."""
    position_scale: float | dict[str, float] = 0.05
    """Scale policy position block (rad) after clipping to [-1, 1]."""
    stiffness_limits: tuple[float, float] = (20.0, 150.0)
    """Map stiffness block action ∈ [-1, 1] linearly to [min, max] (N·m/rad)."""
    damping_ratio_limits: tuple[float, float] = (0.7, 1.5)
    """Map damping block action ∈ [-1, 1] linearly to [min, max] (ζ)."""
    default_stiffness: float = 60.0
    default_damping_ratio: float = 1.0
    gravity_compensation: bool = False
    max_setpoint_deviation: float | None = None
    """(``p_abs`` only) Clamp the integrated setpoint to ±this many rad around the live joint pose.

    Prevents integral wind-up when contact blocks the arm during the push.  ``None`` = no cap
    (only soft joint limits apply)."""
class WidowXTaskSpaceImpedanceAction(OperationalSpaceControllerAction):
    """Task-space OSC with RL-friendly [-1, 1] → physical mapping for stiffness and damping.

    Isaac Lab's default :class:`OperationalSpaceControllerAction` clamps stiffness/damping
    actions directly to physical limits.  This subclass linearly maps the policy blocks
    from [-1, 1] to ``motion_stiffness_limits_task`` / ``motion_damping_ratio_limits_task``
    (same convention as :class:`JointVariableImpedanceAction`).

  Optional ``motion_stiffness_limits_per_axis`` on :class:`WidowXTaskSpaceImpedanceActionCfg`
  sets per task-axis K clamps (translation vs rotation) on the underlying OSC.
    """

    cfg: "WidowXTaskSpaceImpedanceActionCfg"

    def __init__(self, cfg: "WidowXTaskSpaceImpedanceActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        per_axis = cfg.motion_stiffness_limits_per_axis
        if per_axis is not None:
            if len(per_axis) != 6:
                raise ValueError("motion_stiffness_limits_per_axis must have 6 (lo, hi) pairs.")
            lim = self._osc._motion_p_gains_limits
            for i, pair in enumerate(per_axis):
                lim[:, i, 0] = float(pair[0])
                lim[:, i, 1] = float(pair[1])
            lo = torch.tensor([float(p[0]) for p in per_axis], device=self.device, dtype=torch.float32)
            hi = torch.tensor([float(p[1]) for p in per_axis], device=self.device, dtype=torch.float32)
            self._stiffness_lo_per_axis = lo
            self._stiffness_span_per_axis = hi - lo
        else:
            self._stiffness_lo_per_axis = None
            self._stiffness_span_per_axis = None
        self._task_push_axis_w: torch.Tensor | None = None
        self._task_lateral_axis_w: torch.Tensor | None = None
        self._task_vertical_axis_w: torch.Tensor | None = None
        if getattr(cfg, "task_position_box_enabled", False):
            self._task_push_axis_w = self._unit_axis_tensor(cfg.push_axis_world, self.device)
            self._task_lateral_axis_w = self._unit_axis_tensor(cfg.lateral_axis_world, self.device)
            self._task_vertical_axis_w = self._unit_axis_tensor(cfg.vertical_axis_world, self.device)
        # Cumulative commanded EE translation (world frame) since the last reset.  Used by
        # ``_clamp_pose_rel_to_reset_position_box`` to turn ``pose_rel`` into an absolute-from-reset
        # target -- without this, gravity residual sinks the arm and the sunk pose becomes the new
        # setpoint (the "쳐짐" ratchet).  See the method docstring.
        self._cum_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Cumulative axis-angle rotation (small-angle, additive) commanded since the last reset,
        # per task rotation axis (rx, ry, rz).  With ``pose_rel`` the per-step target is always
        # "current + delta" -- there is no absolute orientation anchor, so a persistent policy
        # bias (or unlucky exploration) can walk the roll/yaw target arbitrarily far over an
        # episode with nothing pulling it back (unlike stiffness, which only resists *sudden*
        # deviation from the *current* target, not slow drift).  This buffer tracks that sum so
        # ``_clamp_pose_rel_rotation_box`` can cap it, the rotational analogue of
        # ``task_position_box_enabled`` above.
        self._cum_rot_vec = torch.zeros(self.num_envs, 3, device=self.device)
        # Per-axis (lo, hi) bounds on that cumulative rotation.  A single symmetric scalar cannot
        # express what Slide needs: the wrist must be free to pitch ~25-30 deg DOWN (rx) for rail
        # clearance while roll (ry) and yaw (rz) stay pinned within a few degrees of the reset pose.
        # Defaults to the symmetric ``±orientation_max_dev_rad`` on every axis.
        max_dev = float(getattr(cfg, "orientation_max_dev_rad", 0.26))
        per_axis = getattr(cfg, "orientation_dev_limits_per_axis", None)
        bounds = [(-max_dev, max_dev)] * 3 if per_axis is None else [tuple(b) for b in per_axis]
        self._rot_box_lo = torch.tensor([[b[0] for b in bounds]], device=self.device, dtype=torch.float32)
        self._rot_box_hi = torch.tensor([[b[1] for b in bounds]], device=self.device, dtype=torch.float32)
        self._home_q: torch.Tensor | None = None
        if getattr(cfg, "use_home_joint_posture_hold", False):
            home_map = cfg.home_joint_pos or {}
            home_vec = torch.zeros(self._num_DoF, device=self.device)
            for i, name in enumerate(self._joint_names):
                if name in home_map:
                    home_vec[i] = float(home_map[name])
            self._home_q = home_vec.unsqueeze(0)

    @staticmethod
    def _unit_axis_tensor(axis: Sequence[float], device: torch.device | None = None) -> torch.Tensor:
        a = torch.tensor(axis, dtype=torch.float32, device=device)
        return a / torch.norm(a).clamp_min(1e-9)

    def _clamp_pose_rel_to_reset_position_box(self) -> None:
        """Anchor ``pose_rel`` translation to the reset EE pose, then clamp the cumulative offset.

        WHY THIS IS NOT A SIMPLE CLIP OF ``current + delta``: Isaac Lab's OSC with
        ``target_types=["pose_rel"]`` sets ``desired = current_ee + delta`` every step.  Gravity-
        compensation residual (and any soft-impedance droop) therefore sinks the arm AND the
        setpoint together -- a ratchet.  The previous implementation computed
        ``offset = current + delta - p0``, clamped that, and wrote ``delta = (p0 + offset) - current``.
        With zero action that algebra reduces to ``delta = 0`` for every pose still *inside* the
        box, so the arm free-falls until it hits the box floor (measured: -29 mm in 1.2 s with a
        ±60 mm box; the image "쳐짐" is this continuing past the floor when the reset-pose event
        was missing).  Stiffness cannot fix a ratchet whose target is the already-sunk pose.

        FIX: treat the policy's scaled ``delta`` as an *increment to a cumulative absolute offset
        from the reset pose* (same pattern as ``_cum_rot_vec`` for orientation).  Clamp that
        cumulative offset along push / lateral / vertical, then set this step's ``delta`` so the
        OSC target equals ``p0 + clamped_offset``.  Zero action then springs the EE back to the
        reset pose; a sustained policy command walks the absolute setpoint within the box.
        """
        if self._pose_rel_idx is None or self._task_push_axis_w is None:
            return
        env = self._env
        if not hasattr(env, "_insert_reset_ee_pos_w"):
            return

        self._compute_ee_pose()
        p0_w = env._insert_reset_ee_pos_w
        p_cur_w = self._ee_pose_w[:, :3]
        idx = self._pose_rel_idx
        delta_b = self._processed_actions[:, idx : idx + 3]
        base_quat = self._asset.data.root_quat_w
        delta_w = math_utils.quat_apply(base_quat, delta_b)

        # Accumulate the *policy* delta (before we overwrite it with the absolute correction).
        prospective = self._cum_pos_w + delta_w
        push = self._task_push_axis_w.unsqueeze(0)
        lat = self._task_lateral_axis_w.unsqueeze(0)
        vert = self._task_vertical_axis_w.unsqueeze(0)

        s_push = torch.sum(prospective * push, dim=-1, keepdim=True)
        s_lat = torch.sum(prospective * lat, dim=-1, keepdim=True)
        s_vert = torch.sum(prospective * vert, dim=-1, keepdim=True)

        lat_lim = float(self.cfg.lateral_half_range_m)
        vert_lim = float(self.cfg.vertical_half_range_m)
        s_push = s_push.clamp(float(self.cfg.push_offset_min_m), float(self.cfg.push_offset_max_m))
        s_lat = s_lat.clamp(-lat_lim, lat_lim)
        s_vert = s_vert.clamp(-vert_lim, vert_lim)

        # Leash the push target to what the arm has actually achieved.  The box alone only bounds
        # where the target may END UP, not how far AHEAD of the arm it may sit: with
        # position_scale=0.04 the policy saturates a 0.60 m push box in ~15 steps, and the target
        # then sits a quarter of a metre beyond the fingertips for the rest of the episode.  The
        # OSC reads that as a constant maximal position error on the stiffest axis and hauls the
        # arm out to its reach boundary, where a 6-DoF arm cannot satisfy position and orientation
        # at once -- position wins (ty stiffness 250-2500 vs rx 30-150) and the wrist collapses.
        # Measured with a constant full push command: target +600 mm vs achieved +366 mm, EE lifted
        # +72 mm through a +/-10 mm vertical box, wrist pitch -13 deg -> -75 deg.  Bounding the lead
        # makes the setpoint advance only as fast as the arm follows, which also caps the push force
        # at ``K_push * push_lead_max_m`` instead of ``K_push * position_scale`` every step.
        lead = float(getattr(self.cfg, "push_lead_max_m", 0.0) or 0.0)
        if lead > 0.0:
            achieved_push = torch.sum((p_cur_w - p0_w) * push, dim=-1, keepdim=True)
            s_push = torch.min(s_push, achieved_push + lead)

        offset_c = s_push * push + s_lat * lat + s_vert * vert
        self._cum_pos_w[:] = offset_c
        # Absolute target from reset; rewrite pose_rel delta so OSC desired = p_des.
        p_des_w = p0_w + offset_c
        delta_w_new = p_des_w - p_cur_w
        delta_b_new = math_utils.quat_apply(math_utils.quat_inv(base_quat), delta_w_new)
        self._processed_actions[:, idx : idx + 3] = delta_b_new

    def _clamp_pose_rel_rotation_box(self) -> None:
        """Anchor ``pose_rel`` orientation to the reset EE quat, then clamp the cumulative delta.

        Same gravity/contact ratchet as translation: with ``desired = current ⊕ delta`` and zero
        action, any wrist roll from coupling becomes the new setpoint.  Measured symptom after the
        position-only absolute fix: EE tip Z settled (~-13 mm) but ``joint_4`` still drifted
        +0.17 rad and the pads dropped another ~15 mm.  Fix mirrors the position box -- accumulate
        an absolute axis-angle offset from the reset orientation, clamp per component, and rewrite
        this step's rotational delta so the OSC target equals ``reset ⊕ cum``.
        """
        if self._pose_rel_idx is None:
            return
        env = self._env
        idx = self._pose_rel_idx
        delta = self._processed_actions[:, idx + 3 : idx + 6]
        prospective = self._cum_rot_vec + delta
        clamped = torch.max(torch.min(prospective, self._rot_box_hi), self._rot_box_lo)
        self._cum_rot_vec[:] = clamped

        if not hasattr(env, "_insert_reset_ee_quat_w"):
            # No reset quat stored -- fall back to relative (bounded) increments only.
            self._processed_actions[:, idx + 3 : idx + 6] = clamped - (prospective - delta)
            return

        self._compute_ee_pose()
        q0_w = env._insert_reset_ee_quat_w
        p0_w = env._insert_reset_ee_pos_w if hasattr(env, "_insert_reset_ee_pos_w") else self._ee_pose_w[:, :3]
        q_cur_w = self._ee_pose_w[:, 3:7]
        # Desired orientation = reset ⊕ cum (same left-multiply convention as ``apply_delta_pose``).
        zeros = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        delta_pose = torch.cat([zeros, clamped], dim=-1)
        _, q_des_w = math_utils.apply_delta_pose(p0_w, q0_w, delta_pose)
        # pose_rel rot delta such that ``quat_mul(δq, q_cur) = q_des``.
        q_err = math_utils.quat_mul(q_des_w, math_utils.quat_inv(q_cur_w))
        self._processed_actions[:, idx + 3 : idx + 6] = math_utils.axis_angle_from_quat(q_err)

    def _arm_joint_reference(self) -> torch.Tensor | None:
        """Buffer straddle joint targets stored at insert reset (``_insert_reset_joint_pos``)."""
        env = self._env
        if not hasattr(env, "_insert_reset_joint_pos"):
            return None
        ref_all = env._insert_reset_joint_pos
        if isinstance(self._joint_ids, slice):
            return ref_all[:, : self._num_DoF]
        return ref_all[:, self._joint_ids]

    def _map_stiffness_action(self, stiff_a: torch.Tensor) -> torch.Tensor:
        """Remap raw [-1,1] action so the resulting K fraction-of-range is clamped to [floor, 1].

        BUG FIX (2026-07-24): this used to return ``floor + (1-floor)*0.5*(a+1)`` directly, i.e. a
        value already expressed as a *fraction of [0,1]*.  But the caller (``_preprocess_actions``)
        feeds this straight back into the SAME ``lo + 0.5*(x+1)*span`` formula used for the
        unfloored path, which expects ``x`` in **[-1, 1]**, not already a [0,1] fraction.  Passing a
        [floor,1] value through ``0.5*(x+1)`` a second time silently squashed the *effective* floor
        to ``floor + (1-floor)*0.5`` (e.g. floor=0.5 actually enforced a minimum of 75% of the K
        range, not 50%) -- on EVERY task axis, including the ones deliberately tuned "soft"
        (tz/rx/ry in Slide). That meant the policy could never actually command a soft touch: every
        contact (including light exploratory ones) landed near-max stiffness, which for a ~100g PCB
        is enough to launch it on first contact ("PCB flies forward when pushed" symptom). Fix:
        return the value in the SAME [-1,1] domain the downstream formula expects, by pre-applying
        the inverse of that formula so the two compositions cancel out to exactly ``[floor, 1]``
        fraction-of-range.
        """
        floor = float(getattr(self.cfg, "stiffness_action_floor", 0.0) or 0.0)
        if floor > 0.0:
            frac = floor + (1.0 - floor) * 0.5 * (stiff_a + 1.0)  # fraction-of-range in [floor, 1]
            stiff_a = 2.0 * frac - 1.0  # back to [-1, 1] so the caller's 0.5*(x+1) recovers `frac`
        return stiff_a

    def _map_damping_action(self, damp_a: torch.Tensor) -> torch.Tensor:
        """See ``_map_stiffness_action`` bug-fix note -- identical issue, identical fix."""
        floor = float(getattr(self.cfg, "damping_action_floor", 0.0) or 0.0)
        if floor > 0.0:
            frac = floor + (1.0 - floor) * 0.5 * (damp_a + 1.0)
            damp_a = 2.0 * frac - 1.0
        return damp_a

    def _joint_posture_hold_torques(self) -> torch.Tensor | None:
        """Additive joint PD toward buffer straddle q (6-DoF arm — OSC nullspace unavailable)."""
        if not getattr(self.cfg, "use_buffer_joint_posture_hold", False):
            return None
        q_ref = self._arm_joint_reference()
        if q_ref is None:
            return None
        kp = float(self.cfg.joint_posture_hold_stiffness)
        kd = float(self.cfg.joint_posture_hold_damping)
        return kp * (q_ref - self._joint_pos) - kd * self._joint_vel

    def _home_posture_hold_torques(self) -> torch.Tensor | None:
        """Additive joint PD toward a fixed ``home_joint_pos`` (regulates the null space that
        Isaac Lab's OSC ``nullspace_control`` cannot reach on a 6-DoF arm; see cfg docstring)."""
        if self._home_q is None:
            return None
        kp = float(self.cfg.home_joint_posture_hold_stiffness)
        kd = float(self.cfg.home_joint_posture_hold_damping)
        return kp * (self._home_q - self._joint_pos) - kd * self._joint_vel

    def _nullspace_joint_target(self) -> torch.Tensor | None:
        """Per-env nullspace joint target (requires >6 arm joints in Isaac Lab OSC)."""
        if getattr(self.cfg, "use_buffer_nullspace_target", False):
            q_ref = self._arm_joint_reference()
            if q_ref is not None:
                return q_ref
        return self._nullspace_joint_pos_target

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)

    def apply_actions(self) -> None:
        """OSC torques; optional additive joint PD holds buffer straddle joints on 6-DoF arms."""
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()

        ns_target = None
        if self.cfg.controller_cfg.nullspace_control != "none":
            ns_target = self._nullspace_joint_target()

        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=ns_target,
        )
        tau_hold = self._joint_posture_hold_torques()
        if tau_hold is not None:
            self._joint_efforts[:] = self._joint_efforts + tau_hold
        tau_home = self._home_posture_hold_torques()
        if tau_home is not None:
            self._joint_efforts[:] = self._joint_efforts + tau_home
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def _preprocess_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions
        if self._pose_abs_idx is not None:
            pose_a = self._raw_actions[:, self._pose_abs_idx : self._pose_abs_idx + 7].clamp(-1.0, 1.0)
            self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3] = (
                pose_a[:, :3] * self._position_scale
            )
            self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7] = (
                pose_a[:, 3:7] * self._orientation_scale
            )
        if self._pose_rel_idx is not None:
            pose_a = self._raw_actions[:, self._pose_rel_idx : self._pose_rel_idx + 6].clamp(-1.0, 1.0)
            self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] = (
                pose_a[:, :3] * self._position_scale
            )
            self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] = (
                pose_a[:, 3:6] * self._orientation_scale
            )
        if self._wrench_abs_idx is not None:
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] = (
                self._raw_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6].clamp(-1.0, 1.0)
                * self._wrench_scale
            )
        if self._stiffness_idx is not None:
            stiff_a = self._map_stiffness_action(
                self._raw_actions[:, self._stiffness_idx : self._stiffness_idx + 6].clamp(-1.0, 1.0)
            )
            if self._stiffness_lo_per_axis is not None:
                k_cmd = self._stiffness_lo_per_axis + 0.5 * (stiff_a + 1.0) * self._stiffness_span_per_axis
                self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] = k_cmd
            else:
                k_lo, k_hi = self.cfg.controller_cfg.motion_stiffness_limits_task
                span = float(k_hi) - float(k_lo)
                self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] = (
                    float(k_lo) + 0.5 * (stiff_a + 1.0) * span
                )
        if self._damping_ratio_idx is not None:
            damp_a = self._map_damping_action(
                self._raw_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6].clamp(-1.0, 1.0)
            )
            d_lo, d_hi = self.cfg.controller_cfg.motion_damping_ratio_limits_task
            span = float(d_hi) - float(d_lo)
            self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6] = (
                float(d_lo) + 0.5 * (damp_a + 1.0) * span
            )
        if getattr(self.cfg, "task_position_box_enabled", False):
            self._clamp_pose_rel_to_reset_position_box()
        if getattr(self.cfg, "task_orientation_box_enabled", False):
            self._clamp_pose_rel_rotation_box()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._cum_pos_w[:] = 0.0
            self._cum_rot_vec[:] = 0.0
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._cum_pos_w[ids] = 0.0
            self._cum_rot_vec[ids] = 0.0

@configclass
class WidowXTaskSpaceImpedanceActionCfg(OperationalSpaceControllerActionCfg):
    """Approach task-space action: ``pose_rel`` (6) + task stiffness (6) + damping ratio (6) → 18 dims."""

    class_type: type[ActionTerm] = WidowXTaskSpaceImpedanceAction
    motion_stiffness_limits_per_axis: Sequence[tuple[float, float]] | None = None
    """Optional per task-axis ``(K_min, K_max)``; patches OSC clamps (e.g. softer rotation)."""
    warmup_hold_steps: int = 0
    use_buffer_nullspace_target: bool = False
    """OSC nullspace target from buffer (requires >6 controlled joints — not WidowX arm)."""
    use_buffer_joint_posture_hold: bool = False
    """Additive joint PD toward ``_insert_reset_joint_pos`` (insert straddle hold on 6-DoF arm)."""
    joint_posture_hold_stiffness: float = 120.0
    joint_posture_hold_damping: float = 8.0
    use_home_joint_posture_hold: bool = False
    """Additive joint PD toward a fixed ``home_joint_pos`` target (any name/count of arm joints).

    A 6-DoF arm doing < 6-DoF task-space control (e.g. translation-only) has uncontrolled
    null-space DOF that Isaac Lab's OSC ``nullspace_control`` cannot regulate (it requires > 6
    joints).  Jacobian-TRANSPOSE motion control is also not a minimum-norm IK solve, so those
    null-space joints (typically the shoulder, which carries the largest lever arm / gravity
    torque) can be driven to extreme values purely as a side-effect of reaching a translation
    target, independent of (and not fixed by) gravity compensation.  This adds a light joint-space
    spring-damper toward ``home_joint_pos`` to keep the arm's posture sane without fighting the
    primary task (keep the gains well below the task-space stiffness).
    """
    home_joint_pos: dict[str, float] | None = None
    home_joint_posture_hold_stiffness: float = 15.0
    home_joint_posture_hold_damping: float = 3.0
    stiffness_action_floor: float = 0.0
    """Remap policy K block from [-1,1] to [floor,1] before physical limits."""
    damping_action_floor: float = 0.0
    task_position_box_enabled: bool = False
    """Anchor ``pose_rel`` translation to ``_insert_reset_ee_pos_w`` and clamp the cumulative offset.

    See ``WidowXTaskSpaceImpedanceAction._clamp_pose_rel_to_reset_position_box``: this is what stops
    the pose_rel gravity ratchet ("쳐짐").  Requires the ``store_reset_ee_pose`` reset event.
    """
    task_orientation_box_enabled: bool = False
    """Clamp cumulative commanded rotation (per axis-angle component) since the last reset.

    With ``target_types=["pose_rel"]`` the per-step orientation target is always "current +
    delta" -- there is no absolute anchor, so the impedance spring only resists a *sudden* jump
    away from the *current* running target, not a slow, persistent drift of that target itself.
    A policy that (even slightly) over-rotates one rotation axis on average will walk the
    commanded orientation arbitrarily far off level over an episode, independent of stiffness.
    This caps that drift to ``±orientation_max_dev_rad`` per component (rx, ry, rz in the task
    frame) -- the rotational analogue of ``task_position_box_enabled``.
    """
    orientation_max_dev_rad: float = 0.26
    """Max cumulative rotation (rad, ≈15°) per axis-angle component when the box is enabled."""
    orientation_dev_limits_per_axis: Sequence[tuple[float, float]] | None = None
    """Optional per-axis ``(lo, hi)`` cumulative rotation bounds (rad) for ``(rx, ry, rz)``.

    Overrides the symmetric ``±orientation_max_dev_rad``.  Needed whenever one rotation axis must
    stay free while the others are pinned -- Slide's wrist has to pitch tens of degrees DOWN about
    ``rx`` for rail-guide clearance, but must not roll (``ry``) or yaw (``rz``) away from the
    straddle orientation it was reset into.
    """
    push_axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    lateral_axis_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    vertical_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lateral_half_range_m: float = 0.01
    vertical_half_range_m: float = 0.01
    push_offset_min_m: float = -0.05
    push_offset_max_m: float = 0.60
    push_lead_max_m: float = 0.0
    """Max distance the push target may sit AHEAD of the EE pose actually achieved (0 = disabled).

    Without this the position box bounds only the target's endpoint, so a saturating policy parks
    the setpoint a quarter of a metre past the fingertips and the OSC drags the arm to its reach
    boundary, where orientation control collapses.  See
    ``_clamp_pose_rel_to_reset_position_box``.  Doubles as a push-force cap: the steady-state
    contact force becomes ``K_push * push_lead_max_m``.
    """
def store_insert_reset_ee_pose_w(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str = "arm_action",
    ee_z_bias_m: float = 0.0,
) -> None:
    """Record OSC EE world pose at reset for the absolute-from-reset position/orientation boxes.

    ``ee_z_bias_m`` shifts only the *anchor* upward.  With ``cum = 0`` the OSC then pulls the EE
    toward that higher setpoint, so a small positive bias is a cheap "start already lifted" nudge
    without changing the Approach handoff joints or the pitch box.  Tips still have to stay seated
    via ``tip_mid_thickness`` / the seating factor on ``jaw_rail_clearance``.
    """
    term = get_task_space_impedance_action(env, action_name)
    if term is None:
        return
    term._compute_ee_pose()
    if not hasattr(env, "_insert_reset_ee_pos_w"):
        env._insert_reset_ee_pos_w = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
    if not hasattr(env, "_insert_reset_ee_quat_w"):
        env._insert_reset_ee_quat_w = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.float32)
        env._insert_reset_ee_quat_w[:, 0] = 1.0
    env._insert_reset_ee_pos_w[env_ids] = term._ee_pose_w[env_ids, :3].clone()
    if ee_z_bias_m:
        env._insert_reset_ee_pos_w[env_ids, 2] += float(ee_z_bias_m)
    env._insert_reset_ee_quat_w[env_ids] = term._ee_pose_w[env_ids, 3:7].clone()
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
_GRIPPER_CARRIAGE_JOINT_SPAN_SCALE = 2.0
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
def gripper_jaw_pad_midpoint_world(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """World position at the midpoint between left/right **contact pad tips**."""
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return 0.5 * (left + right)
def gripper_jaw_pad_midpoint_position_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Env-local position at the midpoint between contact pad tips (matches finger_proximity reward)."""
    mid = gripper_jaw_pad_midpoint_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return mid - env.scene.env_origins
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
def _pcb_off_axis_speed(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """``sqrt(v_x² + v_z²)`` of the PCB root (insertion off-axis speed)."""
    v = env.scene[pcb_cfg.name].data.root_lin_vel_w
    return torch.sqrt(torch.square(v[:, 0]) + torch.square(v[:, 2]))
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
def gripper_wrist_carriage_tip_down_pitch_shaping(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    target_pitch_down_deg: float = 20.0,
    max_pitch_down_deg: float = 60.0,
) -> torch.Tensor:
    """Yaw alignment in XY times a SIGNED, tip-DOWN-only, MONOTONE ramp on wrist->jaw pitch.

    ``gripper_wrist_carriage_target_pitch_shaping`` uses ``|u_wc_z|`` (absolute value), so it is
    direction-agnostic: a wrist tilted UP (jaw above wrist) and one tilted DOWN by the same angle
    (jaw below wrist) score identically. That ambiguity meant the policy had no actual incentive to
    pick the tip-DOWN configuration needed to clear the rail guide, and converged to a near-flat
    posture instead (see chat 2026-07-23 clearance discussion).

    This variant uses the SIGNED z-component of the wrist->jaw unit vector (world frame, +Z up), so
    ONLY jaw-below-wrist (tip-down) earns credit: ``down = -u_wc_z`` is ``sin(pitch_down)`` when the
    jaw droops below the wrist.

    SHAPE (2026-07-27 fix): this used to be a NARROW SYMMETRIC PEAK,
    ``exp(-|u_wc_z - target_z| / sin(pitch_sigma_deg))`` with a 6 deg sigma.  Measured by FK at the
    side-base reset posture the wrist->jaw pitch is only ``-2.8 deg``, i.e. ~27 deg away from the
    30 deg target, so that form evaluated to ``exp(-4.3) ~ 0.013`` of full credit with a gradient of
    ~0.3% of the term's weight per degree -- a flat dead zone the policy could never climb, and it
    also PENALISED going deeper than the target even though any angle at or past the clearance
    requirement is equally acceptable (usually better).

    Now the pitch factor is a monotone ramp that is linear in ``sin`` space:

    * ``0`` while flat or tip-UP (no credit for the wrong sign),
    * rising linearly to ``1`` at ``target_pitch_down_deg`` -- a real gradient from the reset pose,
    * held at ``1`` through ``max_pitch_down_deg`` (so "target or steeper" is free), then
    * decaying back toward 0 past ``max_pitch_down_deg``, which guards against degenerate
      near-vertical postures that would swing the pads off the trailing face.
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
    # +1 = jaw straight below the wrist; <=0 = flat or jaw above the wrist (tip-up).
    down = -u_wc[:, 2]
    target_down = max(float(np.sin(np.radians(target_pitch_down_deg))), 1e-6)
    max_down = float(np.sin(np.radians(max_pitch_down_deg)))
    ramp = (down / target_down).clamp(0.0, 1.0)
    if max_down > target_down:
        # Linear decay from full credit at ``max_down`` to zero at straight-down (down = 1).
        excess = (down - max_down).clamp(min=0.0)
        ramp = ramp * (1.0 - excess / max(1.0 - max_down, 1e-6)).clamp(0.0, 1.0)
    return yaw * ramp
def gripper_wrist_carriage_tip_down_pitch_shaping_gated(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
    pcb_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    closedness_std: float,
    push_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    target_pitch_down_deg: float = 20.0,
    max_pitch_down_deg: float = 60.0,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    gate_start: float = 0.20,
    gate_full: float = 0.40,
    tip_mid_std: float | None = None,
    tip_mid_gate_start: float = 0.30,
    tip_mid_gate_full: float = 0.70,
) -> torch.Tensor:
    """Tip-down shaping that only pays after trailing-edge closedness has started to rise.

    Ungated tip-down (weight 80) let Approach farm a deep wrist pitch while ``closedness_tight``
    sat at ~0.10-0.15 — well below the 0.55 success bar — because the OSC rotates about the wrist
    and pitching early pulls the pads off the trailing edge.  Measured run: ep~20 had the best
    tight closedness (0.22) with tip-down still weak; once tip-down saturated (~ep 60+) tight
    closedness never recovered and success stayed ~0.

    Gate is a soft ramp on the SAME tight closedness index the success termination uses
    (``closedness_std`` should be ``_APPROACH_SUCCESS_STD_M``): 0 below ``gate_start``, full
    credit from ``gate_full`` upward.  Tip-down then reinforces the seated posture instead of
    competing with seating.

    Setting ``tip_mid_std`` adds a SECOND, multiplicative ramp on the mid-thickness index.  Lateral
    closedness alone does not pin the pads to the trailing edge along the board's thickness, so with
    only the closedness gate the policy can still buy pitch by dropping the pads under the board --
    which is exactly how the 2026-07-29 handoff ended up 12 mm low.  Because this reward converts
    pitch into rail clearance through the 60 mm pad lever, it is only meaningful when that lever is
    actually anchored at the board edge, and this gate is what enforces it.
    """
    tip = gripper_wrist_carriage_tip_down_pitch_shaping(
        env,
        left_finger_cfg,
        right_finger_cfg,
        wrist_body_cfg,
        push_axis_world=push_axis_world,
        target_pitch_down_deg=target_pitch_down_deg,
        max_pitch_down_deg=max_pitch_down_deg,
    )
    closedness = straddle_finger_target_closedness(
        env,
        float(closedness_std),
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    start = float(gate_start)
    full = float(gate_full)
    if full <= start:
        gate = (closedness >= start).to(dtype=tip.dtype)
    else:
        gate = ((closedness - start) / (full - start)).clamp(0.0, 1.0)
    if tip_mid_std is not None:
        tip_mid = straddle_tip_mid_thickness_shaping(
            env,
            float(tip_mid_std),
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            finger_offset_m=finger_offset_m,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
            width_gap_target_left_m=width_gap_target_left_m,
            width_gap_target_right_m=width_gap_target_right_m,
        )
        mid_start = float(tip_mid_gate_start)
        mid_full = float(tip_mid_gate_full)
        if mid_full <= mid_start:
            mid_gate = (tip_mid >= mid_start).to(dtype=tip.dtype)
        else:
            mid_gate = ((tip_mid - mid_start) / (mid_full - mid_start)).clamp(0.0, 1.0)
        gate = gate * mid_gate
    return tip * gate
def gripper_wrist_pitch_deg_signed_obs(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    wrist_body_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
    """Debug scalar: signed wrist->jaw pitch in degrees (world +Z up; negative = tip-down).

    Not part of the task reward on its own -- wire with a near-zero weight (like
    ``approach_gripper_debug_monitor``) purely so TensorBoard shows the actual achieved pitch sign
    and magnitude, to sanity-check ``gripper_wrist_carriage_tip_down_pitch_shaping`` convergence.
    """
    if wrist_body_cfg is None or len(wrist_body_cfg.body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    robot = env.scene[left_finger_cfg.name]
    wrist = robot.data.body_pos_w[:, wrist_body_cfg.body_ids[0]]
    left, right = gripper_finger_tips_world(env, left_finger_cfg, right_finger_cfg)
    mid = 0.5 * (left + right)
    u_wc = mid - wrist
    u_wc = u_wc / torch.norm(u_wc, dim=-1, keepdim=True).clamp(min=1e-6)
    return torch.asin(u_wc[:, 2].clamp(-1.0, 1.0)) * (180.0 / float(np.pi))
def gripper_midpoint_position_env(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Env-local position at the midpoint between finger bodies."""
    mid = gripper_midpoint_world(env, left_finger_cfg, right_finger_cfg)
    return mid - env.scene.env_origins
def pcb_leading_short_edge_center_env(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Env-local position of the leading short-edge face centre ``(N, 3)``."""
    lead_w = pcb_leading_short_edge_center_w(env, pcb_cfg, half_length_m, axis_world)
    return lead_w - env.scene.env_origins[:, :3]
def pcb_trailing_short_edge_center_env(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Env-local position of the trailing short-edge face centre ``(N, 3)``."""
    trail_w = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m, axis_world)
    return trail_w - env.scene.env_origins[:, :3]
def joint_pos_rel_episode_reset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reset_joint_pos_attr: str = "_insert_reset_joint_pos",
) -> torch.Tensor:
    """Joint positions relative to the robot pose stored at insert reset (straddle buffer row).

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
    wrist_body_cfg: SceneEntityCfg | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCB-frame errors vs the trailing short-edge **face centre** target.

    Returns ``along, width, thick, in_plane, edge_dist`` where width is body +Y (short-edge width).
    When ``gripper_joint_cfg`` is set, uses the jaw **pad-tip midpoint** (same frame as finger_proximity).
    """
    if gripper_joint_cfg is not None:
        mid = gripper_jaw_pad_midpoint_world(
            env,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
        )
    else:
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
_EE_TRAILING_EDGE_PREV_DIST_L: torch.Tensor | None = None
_EE_TRAILING_EDGE_PREV_DIST_R: torch.Tensor | None = None
def approach_success_bonus_reward(
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    tip_mid_thickness_std: float | None = None,
    tip_mid_thickness_threshold: float = 0.5,
    min_tip_down_deg: float | None = None,
    state_attr: str = "_approach_success_bonus_paid",
) -> torch.Tensor:
    """One-shot bonus (1.0) the first time :func:`approach_finger_target_success` is achieved."""
    if not hasattr(env, state_attr):
        setattr(env, state_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    paid: torch.Tensor = getattr(env, state_attr)
    if paid.shape[0] != env.num_envs:
        paid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        setattr(env, state_attr, paid)

    paid[env.episode_length_buf == 1] = False

    achieved = approach_finger_target_success(
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        tip_mid_thickness_std=tip_mid_thickness_std,
        tip_mid_thickness_threshold=tip_mid_thickness_threshold,
        min_tip_down_deg=min_tip_down_deg,
    )
    newly = achieved & (~paid)
    paid[:] = paid | achieved
    return newly.to(dtype=torch.float32)
_EE_XY_APPROACH_PREV_DIST: torch.Tensor | None = None
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
def _debug_buf_matches_env(buf: torch.Tensor | None, env: ManagerBasedEnv) -> bool:
    """True when ``buf`` is sized for ``env`` and on the same device (str-safe)."""
    if buf is None:
        return False
    return buf.shape[0] == env.num_envs and str(buf.device) == str(env.device)
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
_INSERT_SUCCESS_STEP_COUNT: torch.Tensor | None = None
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
_STRADDLE_PITCH_DEG_SUM: torch.Tensor | None = None
_STRADDLE_PITCH_DEG_LAST: torch.Tensor | None = None
_STRADDLE_PITCH_DEG_EP_MIN: torch.Tensor | None = None  # most tip-down (most negative) this episode
_INSERT_TRAVEL_FRAC_MAX: torch.Tensor | None = None
_INSERT_TRAVEL_FRAC_LAST: torch.Tensor | None = None
_INSERT_MILESTONE_POSE_OK_SUM: torch.Tensor | None = None
_INSERT_MILESTONE_BONUS_SUM: torch.Tensor | None = None
_INSERT_MILESTONE_DEBUG_STEPS: torch.Tensor | None = None
_VIC_STIFFNESS_SUM: torch.Tensor | None = None
_VIC_STIFFNESS_LAST: torch.Tensor | None = None
_VIC_DAMPING_SUM: torch.Tensor | None = None
_VIC_DAMPING_LAST: torch.Tensor | None = None
def _insert_milestone_debug_ensure_buffers(env: ManagerBasedEnv) -> None:
    """Per-env buffers for slide travel-fraction / milestone TensorBoard scalars."""
    global _INSERT_TRAVEL_FRAC_MAX, _INSERT_TRAVEL_FRAC_LAST
    global _INSERT_MILESTONE_POSE_OK_SUM, _INSERT_MILESTONE_BONUS_SUM, _INSERT_MILESTONE_DEBUG_STEPS
    n = env.num_envs
    if not _debug_buf_matches_env(_INSERT_TRAVEL_FRAC_MAX, env):
        _INSERT_TRAVEL_FRAC_MAX = torch.zeros(n, device=env.device, dtype=torch.float32)
        _INSERT_TRAVEL_FRAC_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _INSERT_MILESTONE_POSE_OK_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _INSERT_MILESTONE_BONUS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _INSERT_MILESTONE_DEBUG_STEPS = torch.zeros(n, device=env.device, dtype=torch.long)
def _insert_milestone_debug_accumulate(
    env: ManagerBasedEnv,
    frac: torch.Tensor,
    pose_ok: torch.Tensor,
    bonus: torch.Tensor,
) -> None:
    """Accumulate travel fraction and milestone hits for episode-end curriculum logging."""
    global _INSERT_TRAVEL_FRAC_MAX, _INSERT_TRAVEL_FRAC_LAST
    global _INSERT_MILESTONE_POSE_OK_SUM, _INSERT_MILESTONE_BONUS_SUM, _INSERT_MILESTONE_DEBUG_STEPS
    _insert_milestone_debug_ensure_buffers(env)
    frac_f = frac.detach().to(dtype=torch.float32)
    pose_f = pose_ok.detach().to(dtype=torch.float32)
    bonus_f = bonus.detach().to(dtype=torch.float32)
    _INSERT_TRAVEL_FRAC_MAX = torch.maximum(_INSERT_TRAVEL_FRAC_MAX, frac_f)
    _INSERT_TRAVEL_FRAC_LAST = frac_f
    _INSERT_MILESTONE_POSE_OK_SUM += pose_f
    _INSERT_MILESTONE_BONUS_SUM += bonus_f
    _INSERT_MILESTONE_DEBUG_STEPS += 1
def _insert_milestone_debug_clear(env_ids: torch.Tensor) -> None:
    """Reset milestone debug accumulators for finished episodes."""
    global _INSERT_TRAVEL_FRAC_MAX, _INSERT_TRAVEL_FRAC_LAST
    global _INSERT_MILESTONE_POSE_OK_SUM, _INSERT_MILESTONE_BONUS_SUM, _INSERT_MILESTONE_DEBUG_STEPS
    if _INSERT_TRAVEL_FRAC_MAX is None:
        return
    _INSERT_TRAVEL_FRAC_MAX[env_ids] = 0.0
    _INSERT_TRAVEL_FRAC_LAST[env_ids] = 0.0
    _INSERT_MILESTONE_POSE_OK_SUM[env_ids] = 0.0
    _INSERT_MILESTONE_BONUS_SUM[env_ids] = 0.0
    _INSERT_MILESTONE_DEBUG_STEPS[env_ids] = 0
def _vic_impedance_debug_accumulate(
    stiffness_mean: torch.Tensor,
    damping_mean: torch.Tensor,
) -> None:
    """Accumulate per-env mean K / ζ for episode-mean TensorBoard scalars."""
    global _VIC_STIFFNESS_SUM, _VIC_STIFFNESS_LAST, _VIC_DAMPING_SUM, _VIC_DAMPING_LAST
    k_f = stiffness_mean.detach().to(dtype=torch.float32)
    z_f = damping_mean.detach().to(dtype=torch.float32)
    _VIC_STIFFNESS_SUM += k_f
    _VIC_STIFFNESS_LAST = k_f
    _VIC_DAMPING_SUM += z_f
    _VIC_DAMPING_LAST = z_f
def _approach_gripper_debug_ensure_buffers(env: ManagerBasedEnv) -> None:
    global _STRADDLE_DEBUG_JOINT_SUM, _STRADDLE_DEBUG_READY_SUM, _STRADDLE_DEBUG_STEP_COUNT
    global _STRADDLE_DEBUG_JAWS_PAST_SUM, _STRADDLE_DEBUG_BETWEEN_JAWS_SUM, _STRADDLE_DEBUG_Z_STRADDLED_SUM
    global _STRADDLE_DEBUG_LAST_GQ, _STRADDLE_DEBUG_LAST_READY
    global _STRADDLE_DEBUG_LAST_JAWS_PAST, _STRADDLE_DEBUG_LAST_BETWEEN_JAWS, _STRADDLE_DEBUG_LAST_Z_STRADDLED
    global _STRADDLE_Z_LEFT_SUM, _STRADDLE_Z_RIGHT_SUM, _STRADDLE_ACHIEVED_STEP_COUNT, _INSERT_SUCCESS_STEP_COUNT
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
    global _STRADDLE_PITCH_DEG_SUM, _STRADDLE_PITCH_DEG_LAST, _STRADDLE_PITCH_DEG_EP_MIN
    global _VIC_STIFFNESS_SUM, _VIC_STIFFNESS_LAST, _VIC_DAMPING_SUM, _VIC_DAMPING_LAST
    n = env.num_envs
    if not _debug_buf_matches_env(_STRADDLE_DEBUG_JOINT_SUM, env):
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
        _INSERT_SUCCESS_STEP_COUNT = torch.zeros(n, device=env.device, dtype=torch.long)
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
        _STRADDLE_PITCH_DEG_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PITCH_DEG_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        # Init to +inf so the first sample wins the min (most tip-down) reduction.
        _STRADDLE_PITCH_DEG_EP_MIN = torch.full((n,), float("inf"), device=env.device, dtype=torch.float32)
        _VIC_STIFFNESS_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _VIC_STIFFNESS_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _VIC_DAMPING_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _VIC_DAMPING_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
    # Pitch buffers may be missing if the process already allocated the older buffer set.
    if not _debug_buf_matches_env(_STRADDLE_PITCH_DEG_SUM, env):
        _STRADDLE_PITCH_DEG_SUM = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PITCH_DEG_LAST = torch.zeros(n, device=env.device, dtype=torch.float32)
        _STRADDLE_PITCH_DEG_EP_MIN = torch.full((n,), float("inf"), device=env.device, dtype=torch.float32)
def _approach_gripper_debug_accumulate_step(
    gap_left: torch.Tensor,
    gap_right: torch.Tensor,
    achieved: torch.Tensor,
    closedness_prox: torch.Tensor,
    closedness_tight: torch.Tensor | None = None,
    dist_left: torch.Tensor | None = None,
    dist_right: torch.Tensor | None = None,
    pitch_deg: torch.Tensor | None = None,
) -> None:
    """Accumulate per-step jaw gaps and closedness for episode-mean TensorBoard scalars."""
    global _STRADDLE_GAP_LEFT_SUM, _STRADDLE_GAP_RIGHT_SUM
    global _STRADDLE_GAP_LEFT_LAST, _STRADDLE_GAP_RIGHT_LAST
    global _STRADDLE_ACHIEVED_STEP_COUNT, _STRADDLE_DEBUG_STEP_COUNT
    global _STRADDLE_CLOSEDNESS_SUM, _STRADDLE_CLOSEDNESS_LAST
    global _STRADDLE_CLOSEDNESS_TIGHT_SUM, _STRADDLE_CLOSEDNESS_TIGHT_LAST, _STRADDLE_CLOSEDNESS_EP_MAX
    global _STRADDLE_DIST_L_SUM, _STRADDLE_DIST_R_SUM, _STRADDLE_DIST_L_LAST, _STRADDLE_DIST_R_LAST
    global _STRADDLE_PITCH_DEG_SUM, _STRADDLE_PITCH_DEG_LAST, _STRADDLE_PITCH_DEG_EP_MIN
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
    if pitch_deg is not None:
        pitch_f = pitch_deg.detach().to(dtype=torch.float32)
        _STRADDLE_PITCH_DEG_SUM += pitch_f
        _STRADDLE_PITCH_DEG_LAST = pitch_f
        _STRADDLE_PITCH_DEG_EP_MIN = torch.minimum(_STRADDLE_PITCH_DEG_EP_MIN, pitch_f)
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
def _push_debug_accumulate_insert_success(insert_success: torch.Tensor) -> None:
    """Accumulate per-step ``insert_success`` mask for episode ``success_frac`` in TensorBoard."""
    global _INSERT_SUCCESS_STEP_COUNT
    _INSERT_SUCCESS_STEP_COUNT += insert_success.detach().to(dtype=torch.long)
def _approach_gripper_debug_accumulate_all(
    env: ManagerBasedEnv,
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    target_lead_xy_env: tuple[float, float] = (0.056, 0.457),
    tolerance_xy_m: tuple[float, float] = (0.003, 0.015),
    max_gripper_gap_m: float = 0.002,
    require_gripper_closed: bool = False,
    insert_success_min_episode_steps: int = 0,
) -> dict[str, torch.Tensor]:
    """Accumulate per-step push / straddle debug scalars (TensorBoard curriculum hooks)."""
    del asset_cfg, min_straddle_quality
    _approach_gripper_debug_ensure_buffers(env)
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    achieved = closedness_prox >= float(closedness_threshold)
    pitch_deg = gripper_wrist_pitch_deg_signed_obs(env, left_finger_cfg, right_finger_cfg, wrist_body_cfg)
    _approach_gripper_debug_accumulate_step(
        gap_left,
        gap_right,
        achieved,
        closedness_prox,
        closedness_tight=closedness_tight,
        dist_left=dist_left,
        dist_right=dist_right,
        pitch_deg=pitch_deg,
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
        # Without width-gap targets the graspable gate falls back to Z-straddle (opposite
        # thickness faces).  Open-jaw trailing-edge approach keeps both pads at mid-thickness,
        # so that gate is almost always False and between_fingers_q stays ~0.  Pass the same
        # ±width targets as closedness so the metric uses the width-straddle path.
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    push_step = pcb_leading_edge_push_axis_approach_progress(
        env, pcb_cfg, half_length_m, axis_world, max_step_m, update_prev=False
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

    insert_ok = _insert_success_in_range(
        env,
        pcb_cfg,
        half_length_m,
        target_lead_xy_env,
        tolerance_xy_m,
        insert_success_min_episode_steps,
        axis_world,
        gripper_joint_cfg,
        max_gripper_gap_m,
        require_gripper_closed,
    )
    _push_debug_accumulate_insert_success(insert_ok)

    vic_term = get_joint_variable_impedance_action(env, action_name="arm_action")
    if vic_term is not None:
        _vic_impedance_debug_accumulate(
            vic_term.stiffness_cmd.mean(dim=-1),
            vic_term.damping_ratio_cmd.mean(dim=-1),
        )
    else:
        osc_term = get_task_space_impedance_action(env, action_name="arm_action")
        if osc_term is not None and osc_term._stiffness_idx is not None:
            ks = osc_term._stiffness_idx
            zs = osc_term._damping_ratio_idx
            k = osc_term.processed_actions[:, ks : ks + 6].mean(dim=-1)
            z = (
                osc_term.processed_actions[:, zs : zs + 6].mean(dim=-1)
                if zs is not None
                else torch.zeros_like(k)
            )
            _vic_impedance_debug_accumulate(k, z)
    return {
        "gap_left": gap_left,
        "gap_right": gap_right,
        "dist_left": dist_left,
        "dist_right": dist_right,
        "along_l": along_l,
        "along_r": along_r,
        "thick_l": thick_l,
        "thick_r": thick_r,
        "closedness_prox": closedness_prox,
        "closedness_tight": closedness_tight,
        "between_q": between_q,
        "push_gate": push_gate,
        "push_progress": push_progress,
        "lead_y_env": lead_y_env,
        "lead_vy": lead_vy,
        "achieved": achieved,
        "insert_success": insert_ok,
        "pitch_deg": pitch_deg,
    }
def approach_gripper_debug_monitor_reward(
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
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    max_off_axis_speed_m_s: float = 0.020,
    min_straddle_quality: float = 0.3,
    proximity_sigma_m: float = 0.050,
    pcb_half_thickness_m: float = 0.00075,
    width_sigma_m: float = 0.025,
    min_closedness_for_push: float = 0.0,
    proximity_std_m: float = 0.035,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    target_lead_xy_env: tuple[float, float] = (0.056, 0.457),
    tolerance_xy_m: tuple[float, float] = (0.003, 0.015),
    max_gripper_gap_m: float = 0.002,
    require_gripper_closed: bool = False,
    insert_success_min_episode_steps: int = 0,
) -> torch.Tensor:
    """Near-zero reward hook so debug accumulators run during ``reward_manager.compute`` (before reset)."""
    _approach_gripper_debug_accumulate_all(
        env,
        asset_cfg,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        std,
        finger_offset_m=finger_offset_m,
        closedness_threshold=closedness_threshold,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        axis_world=axis_world,
        max_step_m=max_step_m,
        max_off_axis_speed_m_s=max_off_axis_speed_m_s,
        min_straddle_quality=min_straddle_quality,
        proximity_sigma_m=proximity_sigma_m,
        pcb_half_thickness_m=pcb_half_thickness_m,
        width_sigma_m=width_sigma_m,
        min_closedness_for_push=min_closedness_for_push,
        proximity_std_m=proximity_std_m,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        target_lead_xy_env=target_lead_xy_env,
        tolerance_xy_m=tolerance_xy_m,
        max_gripper_gap_m=max_gripper_gap_m,
        require_gripper_closed=require_gripper_closed,
        insert_success_min_episode_steps=insert_success_min_episode_steps,
    )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
def approach_gripper_debug_curriculum(
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    target_lead_xy_env: tuple[float, float] = (0.056, 0.457),
    tolerance_xy_m: tuple[float, float] = (0.003, 0.015),
    max_gripper_gap_m: float = 0.002,
    require_gripper_closed: bool = False,
    insert_success_min_episode_steps: int = 0,
) -> dict[str, float]:
    """Log episode closedness / gap / push means to TensorBoard via ``Curriculum/approach_gripper_debug/*``."""
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
        width_gap_target_left_m,
        width_gap_target_right_m,
        proximity_std_m,
        target_lead_xy_env,
        tolerance_xy_m,
        max_gripper_gap_m,
        require_gripper_closed,
        insert_success_min_episode_steps,
    )
    global _STRADDLE_DEBUG_STEP_COUNT, _STRADDLE_ACHIEVED_STEP_COUNT, _INSERT_SUCCESS_STEP_COUNT
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
    global _STRADDLE_PITCH_DEG_SUM, _STRADDLE_PITCH_DEG_LAST, _STRADDLE_PITCH_DEG_EP_MIN
    global _VIC_STIFFNESS_SUM, _VIC_STIFFNESS_LAST, _VIC_DAMPING_SUM, _VIC_DAMPING_LAST
    if _STRADDLE_DEBUG_STEP_COUNT is None:
        _approach_gripper_debug_ensure_buffers(env)
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    else:
        ids = env_ids.to(device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return {}

    counts = _STRADDLE_DEBUG_STEP_COUNT[ids].to(dtype=torch.float32)
    valid = counts > 0
    if not bool(valid.any()):
        return {}
    counts_safe = counts.clamp(min=1.0)
    gap_left_mean = (_STRADDLE_GAP_LEFT_SUM[ids] / counts_safe)[valid].mean()
    gap_right_mean = (_STRADDLE_GAP_RIGHT_SUM[ids] / counts_safe)[valid].mean()
    gap_left_live = _STRADDLE_GAP_LEFT_LAST[ids][valid].mean()
    gap_right_live = _STRADDLE_GAP_RIGHT_LAST[ids][valid].mean()
    closedness_mean = (_STRADDLE_CLOSEDNESS_SUM[ids] / counts_safe)[valid].mean()
    closedness_live = _STRADDLE_CLOSEDNESS_LAST[ids][valid].mean()
    closedness_tight_mean = (_STRADDLE_CLOSEDNESS_TIGHT_SUM[ids] / counts_safe)[valid].mean()
    closedness_tight_live = _STRADDLE_CLOSEDNESS_TIGHT_LAST[ids][valid].mean()
    closedness_ep_max = _STRADDLE_CLOSEDNESS_EP_MAX[ids][valid].mean()
    dist_l_mm_mean = (_STRADDLE_DIST_L_SUM[ids] / counts_safe)[valid].mean() * 1000.0
    dist_r_mm_mean = (_STRADDLE_DIST_R_SUM[ids] / counts_safe)[valid].mean() * 1000.0
    dist_l_mm_live = _STRADDLE_DIST_L_LAST[ids][valid].mean() * 1000.0
    dist_r_mm_live = _STRADDLE_DIST_R_LAST[ids][valid].mean() * 1000.0
    straddle_achieved_frac = (_STRADDLE_ACHIEVED_STEP_COUNT[ids].to(dtype=torch.float32) / counts_safe)[
        valid
    ].mean()
    success_frac = (_INSERT_SUCCESS_STEP_COUNT[ids].to(dtype=torch.float32) / counts_safe)[valid].mean()
    push_progress_mean = (_STRADDLE_PUSH_PROGRESS_SUM[ids] / counts_safe)[valid].mean()
    push_progress_live = _STRADDLE_PUSH_PROGRESS_LAST[ids][valid].mean()
    push_gate_open_frac = (_STRADDLE_PUSH_GATE_OPEN_SUM[ids] / counts_safe)[valid].mean()
    push_gate_open_live = _STRADDLE_PUSH_GATE_OPEN_LAST[ids][valid].mean()
    lead_y_mean = (_STRADDLE_LEAD_Y_SUM[ids] / counts_safe)[valid].mean()
    lead_y_live = _STRADDLE_LEAD_Y_LAST[ids][valid].mean()
    lead_vy_mean = (_STRADDLE_LEAD_VY_SUM[ids] / counts_safe)[valid].mean()
    lead_vy_live = _STRADDLE_LEAD_VY_LAST[ids][valid].mean()
    between_q_mean = (_STRADDLE_BETWEEN_FINGERS_SUM[ids] / counts_safe)[valid].mean()
    between_q_live = _STRADDLE_BETWEEN_FINGERS_LAST[ids][valid].mean()
    pitch_deg_mean = (_STRADDLE_PITCH_DEG_SUM[ids] / counts_safe)[valid].mean()
    pitch_deg_live = _STRADDLE_PITCH_DEG_LAST[ids][valid].mean()
    pitch_ep_min = _STRADDLE_PITCH_DEG_EP_MIN[ids][valid]
    # Replace +inf (never sampled) with 0 so the mean stays finite.
    pitch_ep_min = torch.where(torch.isfinite(pitch_ep_min), pitch_ep_min, torch.zeros_like(pitch_ep_min))
    pitch_deg_ep_min = pitch_ep_min.mean()
    vic_stiffness_mean = (_VIC_STIFFNESS_SUM[ids] / counts_safe)[valid].mean()
    vic_stiffness_live = _VIC_STIFFNESS_LAST[ids][valid].mean()
    vic_damping_mean = (_VIC_DAMPING_SUM[ids] / counts_safe)[valid].mean()
    vic_damping_live = _VIC_DAMPING_LAST[ids][valid].mean()

    _STRADDLE_DEBUG_STEP_COUNT[ids] = 0
    _STRADDLE_GAP_LEFT_SUM[ids] = 0.0
    _STRADDLE_GAP_RIGHT_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_TIGHT_SUM[ids] = 0.0
    _STRADDLE_CLOSEDNESS_EP_MAX[ids] = 0.0
    _STRADDLE_DIST_L_SUM[ids] = 0.0
    _STRADDLE_DIST_R_SUM[ids] = 0.0
    _STRADDLE_ACHIEVED_STEP_COUNT[ids] = 0
    _INSERT_SUCCESS_STEP_COUNT[ids] = 0
    _STRADDLE_PUSH_PROGRESS_SUM[ids] = 0.0
    _STRADDLE_PUSH_GATE_OPEN_SUM[ids] = 0.0
    _STRADDLE_LEAD_Y_SUM[ids] = 0.0
    _STRADDLE_LEAD_VY_SUM[ids] = 0.0
    _STRADDLE_BETWEEN_FINGERS_SUM[ids] = 0.0
    _STRADDLE_PITCH_DEG_SUM[ids] = 0.0
    _STRADDLE_PITCH_DEG_EP_MIN[ids] = float("inf")
    _VIC_STIFFNESS_SUM[ids] = 0.0
    _VIC_DAMPING_SUM[ids] = 0.0

    milestone_out: dict[str, float] = {}
    if _INSERT_TRAVEL_FRAC_MAX is not None:
        m_steps = _INSERT_MILESTONE_DEBUG_STEPS[ids].to(dtype=torch.float32)
        m_valid = m_steps > 0
        if bool(m_valid.any()):
            m_steps_safe = m_steps.clamp(min=1.0)
            milestone_out["travel_frac_ep_max"] = float(_INSERT_TRAVEL_FRAC_MAX[ids][m_valid].mean().item())
            milestone_out["travel_frac_end"] = float(_INSERT_TRAVEL_FRAC_LAST[ids][m_valid].mean().item())
            milestone_out["milestone_pose_ok_frac"] = float(
                (_INSERT_MILESTONE_POSE_OK_SUM[ids][m_valid] / m_steps_safe[m_valid]).mean().item()
            )
            milestone_out["milestone_bonus_ep"] = float(_INSERT_MILESTONE_BONUS_SUM[ids][m_valid].mean().item())
            if hasattr(env, "_insert_milestone_tier_hits_ep"):
                hits = env._insert_milestone_tier_hits_ep[ids][m_valid].to(dtype=torch.float32)
                for i in range(hits.shape[1]):
                    milestone_out[f"milestone_tier_{i}_hit_frac"] = float(hits[:, i].mean().item())
        _insert_milestone_debug_clear(ids)

    return {
        # Short aliases (README / dashboards) plus explicit *_mean / *_live keys.
        "closedness": float(closedness_mean.item()),
        "closedness_tight": float(closedness_tight_mean.item()),
        "closedness_peak": float(closedness_ep_max.item()),
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
        "straddle_achieved_frac": float(straddle_achieved_frac.item()),
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
        # Signed wrist->pad-tip pitch in degrees (negative = tip-down).  Use these, NOT the
        # ``Episode_Reward/wrist_pitch_deg_debug`` term (weight 1e-10 zeroes it out in TB).
        "pitch_deg_mean": float(pitch_deg_mean.item()),
        "pitch_deg_live": float(pitch_deg_live.item()),
        "pitch_deg_ep_min": float(pitch_deg_ep_min.item()),
        "vic_stiffness_mean": float(vic_stiffness_mean.item()),
        "vic_stiffness_live": float(vic_stiffness_live.item()),
        "vic_damping_mean": float(vic_damping_mean.item()),
        "vic_damping_live": float(vic_damping_live.item()),
        **milestone_out,
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-jaw distance (m) to the trailing face using jaw-axis lateral gaps.

    Returns ``(dist_l, dist_r, along_l, along_r, thick_l, thick_r)`` for debug.

    Lateral error is measured along the live jaw axis (``left→right`` pad line), not PCB body +Y,
    so wrist roll/yaw does not move the reward target away from the pads.  Along / thickness
    use the trailing-face centre in the PCB long / thickness axes.
    """
    center = pcb_trailing_short_edge_center_w(env, pcb_cfg, half_length_m)
    x_w = pcb_body_axis_x_world(env, pcb_cfg)
    z_w = pcb_body_axis_z_world(env, pcb_cfg)
    left, right = gripper_jaw_pad_tips_world(
        env,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    gap_left, gap_right = _finger_jaw_opening_gaps(left, right, center)
    tgt_left = float(width_gap_target_left_m if width_gap_target_left_m is not None else finger_offset_m)
    tgt_right = float(width_gap_target_right_m if width_gap_target_right_m is not None else finger_offset_m)

    def _jaw_aligned_dist(
        tip: torch.Tensor, gap: torch.Tensor, gap_target: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = tip - center
        along = torch.sum(delta * x_w, dim=-1)
        thick = torch.sum(delta * z_w, dim=-1)
        gap_err = gap - gap_target
        dist = torch.sqrt(along * along + thick * thick + gap_err * gap_err + 1e-6)
        return dist, along, thick

    dist_l, along_l, thick_l_mid = _jaw_aligned_dist(left, gap_left, tgt_left)
    dist_r, along_r, thick_r_mid = _jaw_aligned_dist(right, gap_right, tgt_right)
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
) -> torch.Tensor:
    """Composite closedness in ``[0, 1]`` from per-jaw distance to trailing-edge targets.

    Each jaw: ``q = 1 - tanh(dist / std)``.  Returns ``min(q_left, q_right)``.
    Lateral error uses jaw-axis gaps; along / thickness use the trailing-face centre.
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    sig = float(std) + 1e-9
    left_q = 1.0 - torch.tanh(lfinger_dist / sig)
    right_q = 1.0 - torch.tanh(rfinger_dist / sig)
    return torch.minimum(left_q, right_q)
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """Pull each contact pad tip to the PCB mid-thickness plane (not the finger-body centre).

    ``thickness_target_offset_m`` shifts the attractor along the board's thickness axis (PCB body
    +Z; world +Z when the board is flat).  Positive = above the mid-plane.  Insert uses a small
    positive offset so the pads ride slightly above the 1 mm edge rather than straddling its centre.
    """
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    tgt = float(thickness_target_offset_m)
    sig = float(std) + 1e-9
    left_q = 1.0 - torch.tanh(torch.abs(thick_l - tgt) / sig)
    right_q = 1.0 - torch.tanh(torch.abs(thick_r - tgt) / sig)
    return 0.5 * (left_q + right_q)
def straddle_tip_mid_thickness_shaping_gated(
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
    min_closedness: float = 0.3,
    closedness_std: float = 0.10,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """Like :func:`straddle_tip_mid_thickness_shaping`, zero until jaws are near the trailing edge."""
    reward = straddle_tip_mid_thickness_shaping(
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        thickness_target_offset_m=thickness_target_offset_m,
    )
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    gate = (closedness >= float(min_closedness)).to(dtype=reward.dtype)
    return reward * gate
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
def approach_finger_target_success(
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    tip_mid_thickness_std: float | None = None,
    tip_mid_thickness_threshold: float = 0.5,
    min_tip_down_deg: float | None = None,
) -> torch.Tensor:
    """Episode success when finger-target closedness AND tip mid-thickness index both clear their thresholds.

    ``closedness`` (:func:`straddle_finger_target_closedness`) must be ``>= closedness_threshold``.
    If ``tip_mid_thickness_std`` is given, the tip mid-thickness index
    (:func:`straddle_tip_mid_thickness_shaping`, also in ``[0, 1]``) must additionally be
    ``>= tip_mid_thickness_threshold``.  Leaving ``tip_mid_thickness_std=None`` reproduces the old
    closedness-only behaviour.

    ``min_tip_down_deg`` additionally requires the wrist->pad-tip line to be pitched at least that
    far below horizontal.  This is a Insert feasibility gate, not an Approach objective: this
    termination is what ``scripts/collect_approach_states.py`` filters on, so whatever posture is
    admitted here becomes the entire starting distribution of the Insert phase.  With the pad tips
    pinned to the trailing edge, the finger bodies clear the board-support rails by roughly
    ``tip_offset_m * sin(pitch)``, so a shallow terminal pose leaves the lower finger inside the rail
    and the slide jams a few centimetres in no matter what the Insert policy does.
    """
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    achieved = closedness >= float(closedness_threshold)
    if tip_mid_thickness_std is not None:
        tip_mid_thickness = straddle_tip_mid_thickness_shaping(
            env,
            float(tip_mid_thickness_std),
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            finger_offset_m=finger_offset_m,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
            width_gap_target_left_m=width_gap_target_left_m,
            width_gap_target_right_m=width_gap_target_right_m,
        )
        achieved = achieved & (tip_mid_thickness >= float(tip_mid_thickness_threshold))
    if min_tip_down_deg is not None:
        pitch_deg = gripper_wrist_pitch_deg_signed_obs(env, left_finger_cfg, right_finger_cfg, wrist_body_cfg)
        achieved = achieved & (pitch_deg <= -float(min_tip_down_deg))
    return achieved
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
def approach_near_success_shaping_scale(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """Scale in ``[min_scale, 1]`` that fades dense shaping as tight closedness nears success.

    Uses the same tight-σ closedness as the success termination.  Below ``fade_start`` the scale
    is 1 (full shaping credit while approaching).  Between ``fade_start`` and ``fade_end`` it
    linearly falls to ``min_scale``, so lingering near-success no longer farms more return than
    terminating with the success bonus (the collapse mode seen around epoch 10 → 100).

    ``min_tip_down_deg`` must mirror the success termination's pitch gate whenever that gate is in
    use.  The fade only makes sense once staying put is genuinely worse than terminating, and that
    is false while the pose is still pitch-ineligible: closedness alone can sit past ``fade_end``
    with the wrist too flat to ever trigger success, which would strand the policy holding position
    on 5% shaping with no way to bank the bonus.  Envs that fail the pitch gate keep full shaping.

    ``fade_min_tip_mid``/``fade_tip_mid_std`` apply the identical exemption to the success
    termination's mid-thickness gate.  Every conjunct of the success condition needs its own
    exemption here, otherwise the fade punishes closedness progress that cannot yet be cashed in:
    the policy's best response is to park closedness just under ``fade_start`` and trade the
    remaining error between axes, which keeps closedness flat forever.
    """
    closedness_tight = straddle_finger_target_closedness(
        env,
        success_std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    start = float(fade_start)
    end = float(fade_end)
    lo = float(min_scale)
    if end <= start:
        return torch.ones_like(closedness_tight)
    t = ((closedness_tight - start) / (end - start)).clamp(0.0, 1.0)
    if min_tip_down_deg is not None:
        pitch_deg = gripper_wrist_pitch_deg_signed_obs(env, left_finger_cfg, right_finger_cfg, wrist_body_cfg)
        t = torch.where(pitch_deg <= -float(min_tip_down_deg), t, torch.zeros_like(t))
    if fade_min_tip_mid is not None and fade_tip_mid_std is not None:
        tip_mid = straddle_tip_mid_thickness_shaping(
            env,
            float(fade_tip_mid_std),
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            half_length_m,
            finger_offset_m=finger_offset_m,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
            width_gap_target_left_m=width_gap_target_left_m,
            width_gap_target_right_m=width_gap_target_right_m,
        )
        t = torch.where(tip_mid >= float(fade_min_tip_mid), t, torch.zeros_like(t))
    return 1.0 - t * (1.0 - lo)
def _apply_near_success_fade(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    *,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    success_std: float,
    fade_start: float,
    fade_end: float,
    min_scale: float,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    scale = approach_near_success_shaping_scale(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
    )
    return reward * scale
def straddle_finger_trailing_width_proximity_fade_near_success(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """:func:`straddle_finger_trailing_width_proximity` with near-success shaping fade."""
    reward = straddle_finger_trailing_width_proximity(
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    return _apply_near_success_fade(
        env,
        reward,
        pcb_cfg=pcb_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        half_length_m=half_length_m,
        success_std=success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
    )
def straddle_tip_mid_thickness_shaping_gated_fade_near_success(
    env: ManagerBasedRLEnv,
    std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    min_closedness: float = 0.3,
    closedness_std: float = 0.10,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """:func:`straddle_tip_mid_thickness_shaping_gated` with near-success shaping fade."""
    reward = straddle_tip_mid_thickness_shaping_gated(
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
        min_closedness=min_closedness,
        closedness_std=closedness_std,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    return _apply_near_success_fade(
        env,
        reward,
        pcb_cfg=pcb_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        half_length_m=half_length_m,
        success_std=success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
    )
def straddle_trailing_face_bounded_approach_reward_fade_near_success(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    approach_std_m: float,
    overshoot_std_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    target_along_m: float = 0.0,
    height_gate_std_m: float | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    finger_offset_m: float = 0.020,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """:func:`straddle_trailing_face_bounded_approach_reward` with near-success shaping fade."""
    reward = straddle_trailing_face_bounded_approach_reward(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        approach_std_m,
        overshoot_std_m,
        target_along_m=target_along_m,
        height_gate_std_m=height_gate_std_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return _apply_near_success_fade(
        env,
        reward,
        pcb_cfg=pcb_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        half_length_m=half_length_m,
        success_std=success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
    )
def pcb_between_gripper_fingers_fade_near_success(
    env: ManagerBasedRLEnv,
    proximity_sigma_m: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    width_sigma_m: float = 0.025,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    width_gap_sigma_m: float | None = None,
    pcb_half_width_m: float | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    finger_offset_m: float = 0.020,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """:func:`pcb_between_gripper_fingers` with near-success shaping fade."""
    if gripper_joint_cfg is None:
        raise ValueError("gripper_joint_cfg is required for near-success fade (tight closedness).")
    reward = pcb_between_gripper_fingers(
        env,
        proximity_sigma_m,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        pcb_half_thickness_m,
        width_sigma_m=width_sigma_m,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        width_gap_sigma_m=width_gap_sigma_m,
        pcb_half_width_m=pcb_half_width_m,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return _apply_near_success_fade(
        env,
        reward,
        pcb_cfg=pcb_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        half_length_m=half_length_m,
        success_std=success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
    )
def straddle_lateral_gap_shaping_fade_near_success(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    width_gap_target_left_m: float,
    width_gap_target_right_m: float,
    width_gap_sigma_m: float,
    success_std: float,
    fade_start: float = 0.40,
    fade_end: float = 0.50,
    min_scale: float = 0.05,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    finger_offset_m: float = 0.020,
    min_tip_down_deg: float | None = None,
    fade_tip_mid_std: float | None = None,
    fade_min_tip_mid: float | None = None,
) -> torch.Tensor:
    """:func:`straddle_lateral_gap_shaping` with near-success shaping fade."""
    reward = straddle_lateral_gap_shaping(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        width_gap_target_left_m,
        width_gap_target_right_m,
        width_gap_sigma_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
    )
    return _apply_near_success_fade(
        env,
        reward,
        pcb_cfg=pcb_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        half_length_m=half_length_m,
        success_std=success_std,
        fade_start=fade_start,
        fade_end=fade_end,
        min_scale=min_scale,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        min_tip_down_deg=min_tip_down_deg,
        fade_tip_mid_std=fade_tip_mid_std,
        fade_min_tip_mid=fade_min_tip_mid,
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
def gripper_mid_thickness_offset_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    scale_m: float = 0.012,
    wrist_body_cfg: SceneEntityCfg | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
) -> torch.Tensor:
    """Obs: signed offset along PCB thickness axis (jaw mid vs board center), scaled to ~[-1, 1]."""
    pcb = env.scene[pcb_cfg.name]
    pcb_pos = pcb.data.root_pos_w
    if gripper_joint_cfg is not None:
        mid = gripper_jaw_pad_midpoint_world(
            env,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
        )
    else:
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
    gripper_joint_cfg: SceneEntityCfg | None = None,
    tip_offset_m: float = 0.0,
) -> torch.Tensor:
    """Obs: jaw-mid error vs trailing short-edge centre in PCB frame, scaled to ~[-1, 1].

    Components are ``along`` (long axis), ``width`` (short edge), ``thick`` (board thickness).
    """
    along, width, thick, _, _ = _gripper_mid_trailing_edge_errors(
        env,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        half_length_m,
        wrist_body_cfg=wrist_body_cfg,
        gripper_joint_cfg=gripper_joint_cfg,
        tip_offset_m=tip_offset_m,
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
def straddle_finger_target_closedness_obs(
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
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
) -> torch.Tensor:
    """Policy obs: trailing-edge finger-target closedness in ``[0, 1]`` (shape ``[N, 1]``).

    Same index as ``finger_proximity`` reward and push-gate closedness (``min(q_left, q_right)``).
    On hardware, swap this term for a vision-derived estimate of the same scalar.
    """
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    return closedness.unsqueeze(-1)
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    행동(Action) 변화량 패널티: 
    이전 프레임의 명령과 현재 프레임의 명령 차이가 클수록 패널티를 줍니다.
    로봇 팔이 급격하게 방향을 틀거나 덜덜 떠는 현상을 방지합니다.
    """
    # Penalize abrupt action changes: smoother control -> less jitter at grasp/insert.
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
def pcb_forward_push_displacement_indicator(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    initial_y_env: float,
    max_displacement_m: float = 0.005,
) -> torch.Tensor:
    """Binary 1.0 when the PCB has been pushed +Y past ``max_displacement_m`` from spawn.

    Approach must straddle the trailing edge **without** sliding the board toward the magazine.
    Pair with a **negative** weight (e.g. -50) for a flat per-step penalty while the board
    remains forward of the allowed tolerance.
    """
    pcb = env.scene[pcb_cfg.name]
    y_env = pcb.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    dy = y_env - float(initial_y_env)
    return (dy > float(max_displacement_m)).to(dtype=y_env.dtype)
def insert_leading_edge_in_target_xy_range(
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
def _insert_success_gripper_closed_ok(
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
def _insert_success_in_range(
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
    """Shared mask for ``insert_success`` termination and ``insert_success_bonus`` reward."""
    in_range = insert_leading_edge_in_target_xy_range(
        env,
        pcb_cfg,
        half_length_m,
        target_lead_xy_env,
        tolerance_xy_m,
        axis_world,
    )
    gripper_ok = _insert_success_gripper_closed_ok(
        env, gripper_joint_cfg, max_gripper_gap_m, require_gripper_closed
    )
    in_range = in_range & gripper_ok
    if min_episode_steps > 0:
        in_range = in_range & (env.episode_length_buf > min_episode_steps)
    return in_range
def insert_success(
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
    """Insert success: leading-edge X/Y in target box and gripper closed (no Z / velocity gates)."""
    return _insert_success_in_range(
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


def _insert_debug_episode_id(env: ManagerBasedEnv, eid: int, ep_step: int) -> int:
    """Per-env episode counter for play debug (``ManagerBasedRLEnv`` has no ``episode_count_buf``)."""
    if not hasattr(env, "_insert_debug_episode_id"):
        env._insert_debug_episode_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    if ep_step <= 1:
        env._insert_debug_episode_id[eid] += 1
    return int(env._insert_debug_episode_id[eid].item())


def insert_success_debug_step(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | None,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_xy_env: tuple[float, float],
    tolerance_xy_m: tuple[float, float],
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    target_lead_y_env: float = 0.422,
    slot_mouth_y_env: float = 0.207,
    mag_far_y_env: float = 0.472,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    max_gripper_gap_m: float = 0.002,
    require_gripper_closed: bool = False,
    insert_success_min_episode_steps: int = 0,
    print_every_control_steps: int = 32,
    print_env_id: int = 0,
    enable_print: bool = False,
) -> None:
    """Play console hook: leading-edge pose vs ``insert_success`` XY box (enable via ``--debug`` in play_insert.sh)."""
    del env_ids
    do_print = enable_print or env.num_envs <= 8
    if not do_print:
        return

    eid = int(print_env_id) % env.num_envs
    ep_step = int(env.episode_length_buf[eid].item())
    tgt_x, tgt_y = float(target_lead_xy_env[0]), float(target_lead_xy_env[1])
    tol_x, tol_y = float(tolerance_xy_m[0]), float(tolerance_xy_m[1])

    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    lead_x = float(lead_env[eid, 0].item())
    lead_y = float(lead_env[eid, 1].item())
    lead_z = float(lead_env[eid, 2].item())

    dx = lead_x - tgt_x
    dy = lead_y - tgt_y
    in_x = abs(dx) <= tol_x
    in_y = abs(dy) <= tol_y
    insert_ok = bool(
        _insert_success_in_range(
            env,
            pcb_cfg,
            half_length_m,
            target_lead_xy_env,
            tolerance_xy_m,
            insert_success_min_episode_steps,
            axis_world,
            gripper_joint_cfg,
            max_gripper_gap_m,
            require_gripper_closed,
        )[eid].item()
    )

    travel_mm = 0.0
    total_mm = max((tgt_y - lead_y) * 1000.0, 1.0)
    if hasattr(env, "_insert_start_lead_proj"):
        a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
        a = a / torch.norm(a).clamp_min(1e-9)
        proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)
        start_proj = float(env._insert_start_lead_proj[eid].item())
        travel_mm = float((proj[eid] - env._insert_start_lead_proj[eid]).item()) * 1000.0
        total_mm = max((float(target_lead_y_env) - start_proj) * 1000.0, 1.0)

    pcb = env.scene[pcb_cfg.name]
    a = torch.tensor(axis_world, device=env.device, dtype=pcb.data.root_lin_vel_w.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    lead_vy = float(torch.sum(pcb.data.root_lin_vel_w[eid] * a, dim=-1).item()) * 1000.0

    if ep_step <= 1:
        if hasattr(env, "_insert_start_lead_env"):
            start_x = float(env._insert_start_lead_env[eid, 0].item())
            start_y = float(env._insert_start_lead_env[eid, 1].item())
        else:
            start_x, start_y = lead_x, lead_y
        print(
            f"[insert-debug] ep={_insert_debug_episode_id(env, eid, ep_step)} reset "
            f"lead=({start_x * 1000:.1f}, {start_y * 1000:.1f}) mm "
            f"success_box X=[{(tgt_x - tol_x) * 1000:.1f}, {(tgt_x + tol_x) * 1000:.1f}] "
            f"Y=[{(tgt_y - tol_y) * 1000:.1f}, {(tgt_y + tol_y) * 1000:.1f}] "
            f"slot_mouth_y={slot_mouth_y_env * 1000:.1f} far_y={mag_far_y_env * 1000:.1f} "
            f"travel_to_success≈{(tgt_y - start_y) * 1000:.1f} mm",
            flush=True,
        )

    if insert_ok:
        last_ok = getattr(env, "_insert_debug_last_success_ep", None)
        cur_ep = _insert_debug_episode_id(env, eid, ep_step)
        if last_ok is None or last_ok[eid] != cur_ep:
            if not hasattr(env, "_insert_debug_last_success_ep"):
                env._insert_debug_last_success_ep = torch.full(
                    (env.num_envs,), -1, device=env.device, dtype=torch.long
                )
            env._insert_debug_last_success_ep[eid] = cur_ep
            print(
                f"[insert-debug] *** INSERT SUCCESS *** step={int(env.common_step_counter)} "
                f"ep_step={ep_step} lead=({lead_x * 1000:.1f}, {lead_y * 1000:.1f}) mm "
                f"dX={dx * 1000:+.1f} dY={dy * 1000:+.1f} travel={travel_mm:.1f} mm",
                flush=True,
            )
        return

    if int(env.common_step_counter) % int(print_every_control_steps) != 0:
        return

    print(
        f"[insert-debug] step={int(env.common_step_counter)} ep_step={ep_step} "
        f"lead=({lead_x * 1000:.1f}, {lead_y * 1000:.1f}) mm "
        f"dX={dx * 1000:+.1f} dY={dy * 1000:+.1f} "
        f"travel={travel_mm:.1f}/{total_mm:.0f} mm "
        f"vy={lead_vy:+.1f} mm/s in_x={int(in_x)} in_y={int(in_y)} success=0 "
        f"(Z={lead_z * 1000:.1f} mm)",
        flush=True,
    )


def _insert_lead_pose_ok(
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
def _insert_leading_edge_travel_frac(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading-edge env pose, normalized +Y travel fraction ``[0, 1]``, and valid-travel mask."""
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)

    if hasattr(env, "_insert_start_lead_proj"):
        start_proj = env._insert_start_lead_proj
    else:
        # Reset must call ``_store_insert_progress_baselines``; without it frac stays ~0.
        start_proj = proj.detach()

    total = float(target_lead_y_env) - start_proj
    valid = total > 1e-6
    total_safe = torch.where(valid, total, torch.ones_like(total))
    frac = torch.where(valid, (proj - start_proj) / total_safe, torch.zeros_like(proj))
    frac = frac.clamp(0.0, 1.0)
    return lead_env, frac, valid
def insert_leading_edge_travel_milestone_bonus(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
    milestone_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    state_attr: str = "_insert_travel_milestone_paid",
    max_lead_x_drift_m: float = 0.003,
    belt_center_z_env: float = 0.10175,
    max_lead_z_drift_m: float = 0.005,
) -> torch.Tensor:
    """One-shot sparse bonus each time leading-edge +Y travel crosses a milestone fraction.

    Progress is measured from ``env._insert_start_lead_proj`` (set at PCB reset) to
    ``target_lead_y_env``.  Milestone credit requires leading-edge lane X and belt Z pose
    (flat push on the conveyor).  Returns the count of newly crossed tiers this step.
    """
    lead_env, frac, _ = _insert_leading_edge_travel_frac(
        env, pcb_cfg, half_length_m, target_lead_y_env, axis_world
    )
    pose_ok = _insert_lead_pose_ok(
        env, lead_env, max_lead_x_drift_m, belt_center_z_env, max_lead_z_drift_m
    )

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

    if not hasattr(env, "_insert_milestone_tier_hits_ep"):
        env._insert_milestone_tier_hits_ep = torch.zeros(
            env.num_envs, n_tiers, device=env.device, dtype=torch.bool
        )
    tier_hits: torch.Tensor = env._insert_milestone_tier_hits_ep
    if tier_hits.shape[0] != env.num_envs or tier_hits.shape[1] != n_tiers:
        tier_hits = torch.zeros(env.num_envs, n_tiers, device=env.device, dtype=torch.bool)
        env._insert_milestone_tier_hits_ep = tier_hits

    bonus = torch.zeros(env.num_envs, device=env.device, dtype=frac.dtype)
    for i, mf in enumerate(tiers):
        crossed = frac >= float(mf)
        newly = crossed & (~paid[:, i]) & pose_ok
        paid[:, i] = paid[:, i] | newly
        tier_hits[:, i] = tier_hits[:, i] | newly
        bonus = bonus + newly.to(dtype=frac.dtype)

    _insert_milestone_debug_accumulate(env, frac, pose_ok, bonus)
    return bonus
def insert_success_bonus_reward(
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
    """Bonus (1.0) when leading-edge X/Y box and gripper closed match :func:`insert_success`."""
    achieved = _insert_success_in_range(
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
def insert_pcb_yaw_sin_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    scale: float = 0.15,
) -> torch.Tensor:
    """Obs: signed yaw error ``sin(θ)`` between PCB long axis and push axis (XY), scaled."""
    sin_yaw, _ = _pcb_yaw_xy_signed_sin_cos(env, pcb_cfg, axis_world)
    return (sin_yaw / (float(scale) + 1e-9)).unsqueeze(-1)
def insert_finger_push_axis_delta_obs(
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

    if hasattr(env, "_insert_travel_milestone_paid"):
        env._insert_travel_milestone_paid[env_ids] = False
    if hasattr(env, "_insert_milestone_tier_hits_ep"):
        env._insert_milestone_tier_hits_ep[env_ids] = False

    if hasattr(env, "_insert_success_sustain_count"):
        env._insert_success_sustain_count[env_ids] = 0

    if hasattr(env, "_insert_success_sustain_step_buf"):
        env._insert_success_sustain_step_buf[env_ids] = -1
def reset_pcb_on_guide_rails_randomized(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pcb_cfg: SceneEntityCfg,
    pos_env_local: tuple[float, float, float],
    rot_wxyz: tuple[float, float, float, float],
    pos_offset_ranges: dict[str, tuple[float, float]] | None = None,
    yaw_offset_range: tuple[float, float] = (0.0, 0.0),
    velocity_scale: float = 0.0,
    half_length_m: float | None = None,
    slot_mouth_y_env: float = 0.198,
) -> None:
    """Place PCB root at nominal rail pose plus uniform XY offsets and world-Z yaw (domain rand).

    ``pos_offset_ranges`` keys ``"x"`` / ``"y"`` give per-env uniform offsets in env-local axes.
    ``yaw_offset_range`` is a uniform world +Z rotation (radians) applied on top of ``rot_wxyz``.
    When ``half_length_m`` is set, caches leading-edge travel baselines for insert milestones.
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
    pcb.update(0.0)
    if half_length_m is not None:
        _store_insert_progress_baselines(
            env, env_ids, pcb_cfg, float(half_length_m), float(slot_mouth_y_env)
        )
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
_STRADDLE_STATE_BUFFER: dict | None = None
_STRADDLE_BUFFER_PATH: str | None = None
def _load_straddle_state_buffer(path: str) -> dict:
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
    _STRADDLE_STATE_BUFFER = buffer
    _STRADDLE_BUFFER_PATH = path
    print(f"[StraddleStateBuffer] Loaded {n} terminal states from '{path}'")
    return _STRADDLE_STATE_BUFFER
def hold_gripper_open(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_name: str,
    open_width_m: float,
    match_sim_state: bool = False,
    store_target: bool = False,
) -> None:
    """Command the parallel gripper to stay open (Phase 1 straddle / open-gripper insert)."""
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
            # wxai: decreasing joint_1/2 raises the wrist at typical insert poses.
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

    if hasattr(env, "_insert_reset_joint_pos"):
        env._insert_reset_joint_pos[env_ids] = robot.data.joint_pos[env_ids].clone()
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
    if hasattr(env, "_insert_reset_joint_pos"):
        env._insert_reset_joint_pos[env_ids] = joint_pos_new.clone()
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
    apply_gripper_hold_on_reset: bool = True,
) -> None:
    """Reset **robot joints only** by sampling from the saved straddle terminal-state buffer.

    Implements the Phase-2 initial-state distribution from Sequential Dexterity
    (Chen et al. CoRL 2023): the terminal state distribution of Phase 1 (Straddle)
    becomes the initial state distribution of Phase 2 (Insert).

    Stores ``env._straddle_buffer_idx`` so :func:`reset_pcb_from_straddle_states` can load
    the matching ``pcb_pos_env`` / ``pcb_quat`` from the same buffer row.

    Parameters
    ----------
    straddle_states_path:
        Path to the .npz produced by ``scripts/collect_straddle_states.py`` (or legacy
        ``collect_grasp_states.py``).
    gripper_hold_open:
        When True, PD-hold the gripper at ``gripper_open_target_m`` (open-gripper insert).
    """
    buf = _load_straddle_state_buffer(straddle_states_path)
    n_buf = buf["joint_pos"].shape[0]
    n_reset = len(env_ids)
    device = env.device
    dtype = torch.float32

    idx = torch.randint(0, n_buf, (n_reset,), device="cpu")
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

    # Insert obs: joint_pos relative to this reset pose (not HOME default).
    if not hasattr(env, "_insert_reset_joint_pos"):
        env._insert_reset_joint_pos = torch.zeros(
            (env.num_envs, robot.num_joints), device=device, dtype=dtype
        )
    env._insert_reset_joint_pos[env_ids] = joint_pos_new.clone()

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
_INSERT_PREV_CENTER_PROJ: torch.Tensor | None = None
_INSERT_PREV_LEAD_PROJ: torch.Tensor | None = None
def pcb_leading_edge_push_axis_approach_progress(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    update_prev: bool = True,
) -> torch.Tensor:
    """Per-step +Y progress of the leading short-edge centre along the push axis.

    Credits ``clamp(proj_t - proj_{t-1}, 0, max_step) / max_step`` so the policy gets
    immediate signal for millimetre-scale rail slide (unlike cumulative state progress).

    Set ``update_prev=False`` for read-only sampling (e.g. TensorBoard debug) so a prior
    call in the same control step does not zero the reward term's delta.
    """
    global _INSERT_PREV_LEAD_PROJ

    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)

    if (
        _INSERT_PREV_LEAD_PROJ is None
        or _INSERT_PREV_LEAD_PROJ.shape[0] != proj.shape[0]
        or str(_INSERT_PREV_LEAD_PROJ.device) != str(proj.device)
    ):
        if update_prev:
            _INSERT_PREV_LEAD_PROJ = proj.clone()
        return torch.zeros_like(proj)

    first_step = env.episode_length_buf == 1
    prev = torch.where(first_step, proj, _INSERT_PREV_LEAD_PROJ)
    delta = (proj - prev).clamp(min=0.0, max=float(max_step_m))
    if update_prev:
        _INSERT_PREV_LEAD_PROJ = torch.where(first_step, proj, _INSERT_PREV_LEAD_PROJ)
        _INSERT_PREV_LEAD_PROJ = proj.clone()
    return delta / (float(max_step_m) + 1e-9)
def _apply_tip_mid_seated_push_gate(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    min_tip_mid_for_push: float,
    tip_mid_std: float,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    clearance_reference_z_env: float | None = None,
    clearance_gate_start_m: float | None = None,
    clearance_gate_full_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """Zero push credit when pad tips leave the PCB mid-thickness band.

    When the ``clearance_*`` geometry is supplied the credit is additionally scaled by a linear
    ramp on jaw-body height over the rail plane, ``start -> full``.  Clearing the conveyor
    supports is a physical PRECONDITION for pushing, but nothing in the reward said so: the push
    terms paid the same whether the lane ahead was open or the carriage was jammed against a
    rail, so a policy that stalled at 15 mm of clearance still collected push income and the only
    thing asking it to climb was ``jaw_rail_clearance``, worth ~1/s over the band that matters.
    Gating here rather than raising that weight keeps the fix out of the STATIC income ledger
    documented on ``alive_penalty`` -- this multiplies a term that is already zero while idle.

    The ramp is deliberately smooth and starts BELOW the height policies currently reach.  A hard
    step (as used for the mid-thickness gate, whose threshold the handover pose clears by
    construction) would zero all push credit at the current operating point and collapse the
    phase back to freezing.
    """
    mid = straddle_tip_mid_thickness_shaping(
        env,
        float(tip_mid_std),
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        thickness_target_offset_m=thickness_target_offset_m,
    )
    gate = (mid >= float(min_tip_mid_for_push)).to(dtype=reward.dtype)
    if (
        clearance_reference_z_env is not None
        and clearance_gate_start_m is not None
        and clearance_gate_full_m is not None
    ):
        clearance = _jaw_rail_clearance_m(
            env, left_finger_cfg, right_finger_cfg, clearance_reference_z_env
        )
        start = float(clearance_gate_start_m)
        span = max(float(clearance_gate_full_m) - start, 1e-9)
        gate = gate * torch.clamp((clearance - start) / span, min=0.0, max=1.0)
    return reward * gate
def pcb_leading_edge_push_axis_approach_progress_seated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    max_step_m: float = 0.005,
    min_tip_mid_for_push: float = 0.45,
    tip_mid_std: float = 0.008,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    clearance_reference_z_env: float | None = None,
    clearance_gate_start_m: float | None = None,
    clearance_gate_full_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """Leading-edge +Y progress, zero while tips are off the mid-thickness plane."""
    step = pcb_leading_edge_push_axis_approach_progress(
        env, pcb_cfg, half_length_m, axis_world, max_step_m
    )
    if (
        left_finger_cfg is None
        or right_finger_cfg is None
        or gripper_joint_cfg is None
    ):
        return step
    return _apply_tip_mid_seated_push_gate(
        env,
        step,
        min_tip_mid_for_push,
        tip_mid_std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        clearance_reference_z_env=clearance_reference_z_env,
        clearance_gate_start_m=clearance_gate_start_m,
        clearance_gate_full_m=clearance_gate_full_m,
        thickness_target_offset_m=thickness_target_offset_m,
    )
def pcb_push_axis_velocity_reward_seated(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_push_speed_m_s: float = 0.002,
    ref_speed_m_s: float | None = None,
    min_tip_mid_for_push: float = 0.45,
    tip_mid_std: float = 0.008,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    half_length_m: float = 0.060,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    clearance_reference_z_env: float | None = None,
    clearance_gate_start_m: float | None = None,
    clearance_gate_full_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """+push-axis PCB velocity, zero while tips are off the mid-thickness plane."""
    vel = pcb_push_axis_velocity_reward(
        env,
        pcb_cfg,
        axis_world,
        min_push_speed_m_s=min_push_speed_m_s,
        ref_speed_m_s=ref_speed_m_s,
    )
    if (
        left_finger_cfg is None
        or right_finger_cfg is None
        or gripper_joint_cfg is None
    ):
        return vel
    return _apply_tip_mid_seated_push_gate(
        env,
        vel,
        min_tip_mid_for_push,
        tip_mid_std,
        pcb_cfg,
        left_finger_cfg,
        right_finger_cfg,
        gripper_joint_cfg,
        half_length_m,
        finger_offset_m=finger_offset_m,
        tip_offset_m=tip_offset_m,
        wrist_body_cfg=wrist_body_cfg,
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
        clearance_reference_z_env=clearance_reference_z_env,
        clearance_gate_start_m=clearance_gate_start_m,
        clearance_gate_full_m=clearance_gate_full_m,
        thickness_target_offset_m=thickness_target_offset_m,
    )
def pcb_edge_axis_parallel_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    half_width_m: float,
    max_offset_m: float = 0.001,
    max_penalty_excess_m: float = 0.012,
    max_lift_m: float = 0.001,
    max_penalty_lift_m: float = 0.008,
    reference_z_env: float | None = None,
    use_episode_start: bool = True,
) -> torch.Tensor:
    """Penalty when the PCB stops lying square and flat in the lane, as ``in-plane + lift``.

    The board may start the episode tilted or rotated, but to enter the slot it has to end up
    square with the lane AND still on the belt.  Both halves are read off the same four
    mid-thickness corners, so the term states the geometric condition directly rather than as a
    quaternion error, and it stays valid whichever way the board's body frame happens to be signed:

    * **in-plane** -- each short edge's two corners must share a world **Y**, and each long edge's
      two corners must share a world **X**.
    * **lift** -- no corner may rise above the reference belt plane in world **+Z**.  Taking the max
      over corners rather than the centre catches a single pried-up corner, not just bulk lift.

    Each half is the excess past its dead band, capped and squared into ``[0, 1]``; the return is
    their **sum**, so ``[0, 2]``.  Pair with a **negative** weight.

    The dead bands are deliberately tight.  The long edge binds first in-plane: at
    ``half_length_m = 0.12`` the 1 mm dead band is ~0.24 deg of yaw and the 12 mm cap is ~2.9 deg,
    so the penalty is already saturated at an error that a loose 40 mm cap barely registered.
    """
    pcb = env.scene[pcb_cfg.name]
    center = pcb.data.root_pos_w
    dx = float(half_length_m) * pcb_body_axis_x_world(env, pcb_cfg)
    dy = float(half_width_m) * pcb_body_axis_y_world(env, pcb_cfg)
    corner_pp = center + dx + dy
    corner_pm = center + dx - dy
    corner_mp = center - dx + dy
    corner_mm = center - dx - dy

    short_edge_dev = torch.maximum(
        torch.abs(corner_pp[:, 1] - corner_pm[:, 1]),
        torch.abs(corner_mp[:, 1] - corner_mm[:, 1]),
    )
    long_edge_dev = torch.maximum(
        torch.abs(corner_pp[:, 0] - corner_mp[:, 0]),
        torch.abs(corner_pm[:, 0] - corner_mm[:, 0]),
    )
    dev = torch.maximum(short_edge_dev, long_edge_dev)
    plane_cap = max(float(max_penalty_excess_m), 1e-9)
    plane_excess = torch.clamp(dev - float(max_offset_m), min=0.0, max=plane_cap)
    plane_term = torch.square(plane_excess / plane_cap)

    corner_z_env = torch.stack(
        (corner_pp[:, 2], corner_pm[:, 2], corner_mp[:, 2], corner_mm[:, 2]), dim=-1
    ) - env.scene.env_origins[:, 2].unsqueeze(-1)
    top_z = torch.amax(corner_z_env, dim=-1)
    if use_episode_start and hasattr(env, "_insert_start_lead_z"):
        ref_z = env._insert_start_lead_z
    elif reference_z_env is not None:
        ref_z = torch.full_like(top_z, float(reference_z_env))
    else:
        ref_z = top_z.detach()
    lift_cap = max(float(max_penalty_lift_m), 1e-9)
    lift_excess = torch.clamp(top_z - ref_z - float(max_lift_m), min=0.0, max=lift_cap)
    lift_term = torch.square(lift_excess / lift_cap)

    return plane_term + lift_term
def _jaw_rail_clearance_m(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    reference_z_env: float,
) -> torch.Tensor:
    """Height of the LOWER jaw body above the rail plane, in metres.

    ``min`` over the two jaws because the inherited jaw roll skews them by ~±9 mm about their
    mean and it is the lower one that catches on the conveyor supports.
    """
    robot = env.scene[left_finger_cfg.name]
    left_id = _resolve_first_body_id(robot, left_finger_cfg)
    right_id = _resolve_first_body_id(robot, right_finger_cfg)
    origin_z = env.scene.env_origins[:, 2]
    left_z = robot.data.body_pos_w[:, left_id, 2] - origin_z
    right_z = robot.data.body_pos_w[:, right_id, 2] - origin_z
    return torch.minimum(left_z, right_z) - float(reference_z_env)
def gripper_jaw_rail_clearance_shaping(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    reference_z_env: float,
    target_clearance_m: float = 0.018,
    pcb_cfg: SceneEntityCfg | None = None,
    gripper_joint_cfg: SceneEntityCfg | None = None,
    half_length_m: float | None = None,
    seat_tip_mid_std: float | None = None,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    thickness_target_offset_m: float = 0.0,
    travel_target_lead_y_env: float | None = None,
    travel_axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    travel_gate_start_m: float | None = None,
    travel_gate_full_m: float | None = None,
    travel_gate_floor: float = 0.25,
) -> torch.Tensor:
    """Reward the LOWER jaw body for riding clear above the conveyor's rail plane.

    This states the insert phase's actual clearance requirement directly, instead of via wrist pitch.  Pitch
    was the wrong proxy: measured from the Insert reset, commanding more tip-down moved the jaw bodies
    from +10 mm above the board plane DOWN to -7 mm -- into the support rails -- because the OSC
    rotates about the wrist, so pitching swings the whole carriage down rather than lifting it.  What
    separates a clear lane from a jam is jaw-body HEIGHT: at +17.9 mm the nearest fixture ahead is the
    magazine 331 mm away, at +10.0 mm it is a support rail 25 mm away.

    ``min`` over the two jaws because the inherited jaw roll skews them by ~±9 mm about their mean and
    it is the lower one that catches.  Linear ramp to full credit at ``target_clearance_m``; pair with
    a **positive** weight.

    Supplying the ``seat_tip_mid_std`` geometry multiplies the ramp by the pads' mid-thickness index,
    which closes the term's one loophole.  With the tips on the board plane the clearance is fixed at
    ``tip_offset * sin(pitch)``, so height and pitch are the same number -- but that identity only
    holds while the pads stay on the plane.  When the carriage fouls the rail guide the braced tip
    becomes the pivot, and rotating the jaw down then levers the body UP: the raw ramp pays for that,
    and ``gripper_tip_under_pcb_penalty`` reads exactly zero until the tip is already under the board,
    so nothing opposes the rotation until the damage is done.  The seating factor removes the payout
    the instant the pads leave the plane, leaving the coordinated lift as the only way to earn it.

    Optional travel gate (``travel_gate_*``): scales the payout from ``travel_gate_floor`` up to 1.0
    as leading-edge travel goes from ``start`` to ``full``.  Approach hands over a shallow pitch;
    Insert is meant to deepen clearance *while pushing*, not farm height at the reset pose.  The floor
    keeps a weak early gradient so the arm starts lifting before the ~25-30 mm jam zone.
    """
    clearance = _jaw_rail_clearance_m(
        env, left_finger_cfg, right_finger_cfg, reference_z_env
    )
    target = max(float(target_clearance_m), 1e-9)
    ramp = torch.clamp(clearance / target, min=0.0, max=1.0)
    if seat_tip_mid_std is not None and pcb_cfg is not None and gripper_joint_cfg is not None:
        ramp = ramp * straddle_tip_mid_thickness_shaping(
            env,
            float(seat_tip_mid_std),
            pcb_cfg,
            left_finger_cfg,
            right_finger_cfg,
            gripper_joint_cfg,
            float(half_length_m),
            finger_offset_m=finger_offset_m,
            tip_offset_m=tip_offset_m,
            wrist_body_cfg=wrist_body_cfg,
            width_gap_target_left_m=width_gap_target_left_m,
            width_gap_target_right_m=width_gap_target_right_m,
            thickness_target_offset_m=thickness_target_offset_m,
        )
    if (
        travel_gate_start_m is not None
        and travel_gate_full_m is not None
        and travel_target_lead_y_env is not None
        and pcb_cfg is not None
        and half_length_m is not None
    ):
        lead_env, _, _ = _insert_leading_edge_travel_frac(
            env,
            pcb_cfg,
            float(half_length_m),
            float(travel_target_lead_y_env),
            travel_axis_world,
        )
        a = torch.tensor(travel_axis_world, device=env.device, dtype=lead_env.dtype)
        a = a / torch.norm(a).clamp_min(1e-9)
        proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)
        if hasattr(env, "_insert_start_lead_proj"):
            travel_m = proj - env._insert_start_lead_proj
        else:
            travel_m = torch.zeros_like(proj)
        start = float(travel_gate_start_m)
        span = max(float(travel_gate_full_m) - start, 1e-9)
        floor = float(travel_gate_floor)
        travel_scale = floor + (1.0 - floor) * torch.clamp(
            (travel_m - start) / span, min=0.0, max=1.0
        )
        ramp = ramp * travel_scale
    return ramp
def jaw_rail_clearance_mm_obs(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    reference_z_env: float,
) -> torch.Tensor:
    """Debug readout: lower-jaw height above the rail plane, in MILLIMETRES.

    Pair with a 1e-10 weight so the logged ``Episode_Reward`` value divided by 1e-10 reads directly
    in mm.  ``jaw_rail_clearance`` itself logs ``weight * ramp * seat``, which needs the seating
    index from a second term to invert -- three quantities the run cannot separate after the fact.
    """
    return _jaw_rail_clearance_m(env, left_finger_cfg, right_finger_cfg, reference_z_env) * 1000.0
def insert_leading_edge_travel_mm_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    target_lead_y_env: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Debug readout: leading-edge push-axis travel since the insert reset, in MILLIMETRES.

    The push terms only ever log a *normalised* rate, so a run cannot tell "slow but arriving" from
    "nowhere near the goal" -- the distinction that decides whether the milestones are reachable
    inside ``episode_length_s`` at all.
    """
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    a = torch.tensor(axis_world, device=env.device, dtype=lead_env.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    proj = torch.sum(lead_env * a.unsqueeze(0), dim=-1)
    if hasattr(env, "_insert_start_lead_proj"):
        return (proj - env._insert_start_lead_proj) * 1000.0
    return torch.zeros_like(proj)
def insert_leading_edge_lane_drift_mm_obs(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    half_length_m: float,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
) -> torch.Tensor:
    """Debug readout: leading-edge lane (X) drift since the insert reset, in MILLIMETRES.

    Signed, so the sign says which way the board is being steered.  This is the gate that silently
    withholds ``insert_travel_milestone`` credit (see ``_insert_lead_pose_ok``), and a run where the
    board travels far enough but the milestone stays zero is otherwise indistinguishable from one
    where it never travelled.
    """
    lead_env = pcb_leading_short_edge_center_env(env, pcb_cfg, half_length_m, axis_world)
    if hasattr(env, "_insert_start_lead_env"):
        return (lead_env[:, 0] - env._insert_start_lead_env[:, 0]) * 1000.0
    return torch.zeros_like(lead_env[:, 0])
def gripper_tip_under_pcb_penalty(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_joint_cfg: SceneEntityCfg,
    half_length_m: float,
    pcb_half_thickness_m: float = 0.0005,
    finger_offset_m: float = 0.020,
    tip_offset_m: float = 0.0,
    wrist_body_cfg: SceneEntityCfg | None = None,
    width_gap_target_left_m: float | None = None,
    width_gap_target_right_m: float | None = None,
    max_penalty_excess_m: float = 0.010,
    thickness_target_offset_m: float = 0.0,
) -> torch.Tensor:
    """Penalty when a pad tip leaves the PCB's trailing-edge thickness band (over OR under).

    Symmetric guard for mid-thickness seating.  The old one-sided ``under`` form left climbing *over*
    the board unpunished: ``tip_under_penalty`` read exactly zero while the tip rode the top face,
    push rewards kept paying, and the only attractor was the weak ``tip_mid_thickness`` shaping term.
    Inflated contact offsets can also pop the tip up over the 1 mm edge during collision resolution,
    which looks like adaptive contact but is just the physics shell riding the corner.

    ``thick`` is the signed offset of each tip from the trailing-face centre along the board's
    thickness axis.  With ``thickness_target_offset_m`` the allowed band is centred on that offset
    (Slide's +5 mm "ride above the edge" target); ``relu(abs(thick - offset) - half)`` is how far a
    tip has left the band on either side.

    ``excess`` is squared and normalised to ``[0, 1]``.  Pair with a **negative** weight.
    """
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
        width_gap_target_left_m=width_gap_target_left_m,
        width_gap_target_right_m=width_gap_target_right_m,
    )
    half = float(pcb_half_thickness_m)
    tgt = float(thickness_target_offset_m)
    off_l = torch.clamp(torch.abs(thick_l - tgt) - half, min=0.0)
    off_r = torch.clamp(torch.abs(thick_r - tgt) - half, min=0.0)
    excess = torch.maximum(off_l, off_r)
    cap = max(float(max_penalty_excess_m), 1e-9)
    excess = torch.clamp(excess, min=0.0, max=cap)
    return torch.square(excess / cap)
def pcb_push_axis_velocity_reward(
    env: ManagerBasedRLEnv,
    pcb_cfg: SceneEntityCfg,
    axis_world: tuple[float, float, float] = _DEFAULT_PUSH_AXIS_WORLD,
    min_push_speed_m_s: float = 0.002,
    ref_speed_m_s: float | None = None,
) -> torch.Tensor:
    """Reward PCB root linear speed along the push axis (default world +Y).

    Returns ``v_push`` when ``v_push > min_push_speed_m_s``; otherwise 0.  No credit for
    −push, X, or Z motion.  Pair with :func:`pcb_push_axis_progress_reward` (state).
    Use a **positive** weight.

    ``ref_speed_m_s`` normalises the result to ``clamp(v_push / ref, 0, 1)`` so the term is a
    [0, 1] index like the rest of the reward set and its weight means what it looks like.  Left
    at ``None`` the raw m/s value is returned, which makes any weight read several orders of
    magnitude larger than it pays.
    """
    pcb = env.scene[pcb_cfg.name]
    v = pcb.data.root_lin_vel_w
    a = torch.tensor(axis_world, device=env.device, dtype=v.dtype)
    a = a / torch.norm(a).clamp_min(1e-9)
    v_push = torch.sum(v * a.unsqueeze(0), dim=-1)
    moving = v_push > float(min_push_speed_m_s)
    if ref_speed_m_s is not None:
        v_push = (v_push / (float(ref_speed_m_s) + 1e-9)).clamp(0.0, 1.0)
    return torch.where(moving, v_push, torch.zeros_like(v_push))
push_gripper_debug_monitor_reward = approach_gripper_debug_monitor_reward
push_gripper_debug_accumulate = _approach_gripper_debug_accumulate_all
push_gripper_debug_curriculum = approach_gripper_debug_curriculum
