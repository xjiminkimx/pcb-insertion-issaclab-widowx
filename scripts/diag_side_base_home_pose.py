#!/usr/bin/env python3
"""Search a reset (home) arm posture for the SIDE base placement, by batched FK in sim.

Why: moving the base back beside the belt (identity rotation, arm reaching world +X) rotates the
whole arm -90 deg about Z relative to the old behind-the-belt (+90 deg yaw) placement, so the
gripper's jaw-opening axis no longer lies along the PCB width (world X).  The Approach action term
caps CUMULATIVE commanded rotation at ``_ARM_TASK_ORIENTATION_MAX_DEV_RAD`` (~8.6 deg), so the
policy cannot re-orient the wrist by 90 deg during an episode -- the reset posture must already be
correct.

Method: each parallel env is one FK sample.  Random joint configs (round 1) then Gaussian
refinement around the best (round 2+) are written straight to the articulation, and the resulting
body poses are scored against the straddle geometry the Approach rewards expect:
  * pad midpoint near the trailing-edge standoff point (lane X, trailing edge Y - standoff, mid-thickness Z)
  * jaw-opening axis parallel to world X (the PCB width direction) and horizontal
  * wrist -> pad direction pointing along +Y (straight-in approach onto the trailing face)

Usage (from workspace root)::

    python scripts/diag_side_base_home_pose.py --num_envs 4096 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_WORKSPACE, "..", "..", "..", "..", ".."))

if _ISAACLAB_ROOT not in sys.path:
    sys.path.insert(0, _ISAACLAB_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Search a home posture for the side base placement.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel FK samples per round.")
parser.add_argument("--rounds", type=int, default=6, help="Refinement rounds after the random round.")
parser.add_argument(
    "--standoff_m",
    type=float,
    default=0.06,
    help="How far before the trailing face (-Y) the pad midpoint should start.",
)
parser.add_argument(
    "--min_standoff_m",
    type=float,
    default=0.05,
    help="Hard minimum clearance behind the trailing face, so zero-action drift cannot lean the "
    "gripper on the board and pre-push the PCB toward the slot.",
)
parser.add_argument("--gui", action="store_true", help="Show the viewport (default: headless).")
parser.add_argument(
    "--legacy_behind_base",
    action="store_true",
    help="Score the OLD behind-the-belt base (pos (0.06,-0.30), +90 deg yaw) instead of searching. "
    "Sanity check: the shipped home posture should score |jaw.X| ~ 1 there.",
)
parser.add_argument(
    "--inspect",
    action="store_true",
    help="Only report the geometry of --base / --base_yaw_deg / --home (no search).",
)
parser.add_argument(
    "--smoke_steps",
    type=int,
    default=0,
    help="Instead of searching, reset the real env and step it with zero actions, reporting PCB "
    "displacement and pad drift.  Checks the placement does not clip the conveyor or nudge the PCB.",
)
parser.add_argument(
    "--vert_box_m",
    type=float,
    default=None,
    help="Smoke test: override the EE vertical box half-range, to check how tightly it bounds droop.",
)
parser.add_argument("--pad_z_min", type=float, default=None, help="Search: min pad-midpoint height (env Z).")
parser.add_argument("--pad_z_max", type=float, default=None, help="Search: max pad-midpoint height (env Z).")
parser.add_argument("--base", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--base_yaw_deg", type=float, default=0.0, help="Base yaw about world +Z (deg).")
parser.add_argument(
    "--home",
    type=float,
    nargs=6,
    default=None,
    metavar=("J0", "J1", "J2", "J3", "J4", "J5"),
    help="Arm joint values to inspect (default: the cfg's _ROBOT_HOME_JOINT_POS).",
)
parser.add_argument("--base_y", type=float, default=None, help="Search a single base Y instead of sweeping.")
parser.add_argument("--base_x", type=float, default=None, help="Override base X (default: cfg _ROBOT_BASE_POS).")
parser.add_argument(
    "--pos_weight",
    type=float,
    default=0.5,
    help="Weight on pad-midpoint position error (orientation terms are fixed at 3.0 because the "
    "policy cannot correct orientation past the ~8.6 deg cumulative box, but can translate freely).",
)
parser.add_argument(
    "--restarts",
    type=int,
    default=1,
    help="Independent search restarts per base Y (the refinement is stochastic and does get stuck).",
)
parser.add_argument(
    "--target",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Pad-midpoint target in env frame (default: lane X, standoff before the trailing face, "
    "PCB mid-thickness Z).  Pass the legacy home's measured pad midpoint to start Approach from the "
    "same geometry the behind-the-belt policy learned from.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.gui:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.manager_based.widowx_pcb import widowx_pcb_env_cfg as cfg  # noqa: E402

_ARM_JOINTS = [f"joint_{i}" for i in range(6)]

# Orientation constraints, measured off the proven behind-the-belt home posture
# (|jaw.X| = 1.000, |jaw_z| = 0.000, approach.Y = +0.936).
_JAW_ALONG_X_MIN = 0.999
_JAW_TILT_MAX = 0.01
_APPROACH_ALONG_Y_MIN = 0.93
# Reset height band for the pad midpoint (env-frame world Z).  Belt top is 0.150 m; stay clear of it
# and of the guide rails, but well below the 0.425 m the old side-base home used.
_PAD_Z_MIN_W = 0.19
_PAD_Z_MAX_W = 0.28


def _target_pad_midpoint_env() -> tuple[float, float, float]:
    """Straddle start point in env frame: lane X, ``standoff`` before the trailing face, mid-thickness Z."""
    if args.target is not None:
        return tuple(float(v) for v in args.target)  # type: ignore[return-value]
    trailing_edge_y = cfg._PCB_INIT_POS[1] - cfg.PCB_X * 0.5
    return (
        cfg._CONVEYOR_CENTER_X_ENV,
        trailing_edge_y - float(args.standoff_m),
        cfg._PCB_CENTER_Z_ENV,
    )


def _score(env: ManagerBasedRLEnv, target_xyz: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-env FK metrics for the current joint configuration.

    Everything is measured RELATIVE TO THE ROBOT BASE, so ``target_xyz`` is a base-relative offset.
    That makes the metrics independent of where the base actually sits, which is what lets the base
    placement be swept analytically (and lets the robot be spawned up in free space, see ``main``).
    """
    robot = env.scene["robot"]
    origins = robot.data.root_pos_w

    left_idx = robot.body_names.index("gripper_left")
    right_idx = robot.body_names.index("gripper_right")
    wrist_idx = robot.body_names.index(cfg._EE_OSC_BODY_NAME)

    p_left = robot.data.body_pos_w[:, left_idx] - origins
    p_right = robot.data.body_pos_w[:, right_idx] - origins
    p_wrist = robot.data.body_pos_w[:, wrist_idx] - origins

    pad_mid = 0.5 * (p_left + p_right)
    jaw = p_right - p_left
    jaw = jaw / jaw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    reach = (pad_mid - p_wrist)
    reach = reach / reach.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    pos_err = (pad_mid - target_xyz).norm(dim=-1)
    # Jaw axis must lie along the PCB width (world X) and be horizontal (both pads same height).
    jaw_along_x = jaw[:, 0].abs()
    jaw_tilt = jaw[:, 2].abs()
    # Wrist -> pads should point at the trailing face (+Y), i.e. a straight-in approach.
    approach_along_y = reach[:, 1]

    # HORIZONTAL PLACEMENT IS THE OBJECTIVE; EVERYTHING ELSE IS A CONSTRAINT.
    #   * Orientation: the Approach action term caps CUMULATIVE commanded rotation at ~8.6 deg
    #     (``_ARM_TASK_ORIENTATION_MAX_DEV_RAD``), so orientation error left at reset is permanent.
    #     Thresholds are what the proven behind-the-belt home posture actually achieves.
    #   * Height: must clear the belt/guide rails (the unconstrained optimum happily parks the pads
    #     below the belt top) but need not start at board height -- the policy descends.
    #   * Standoff behind the trailing face: under zero actions the arm free-drifts 10-15 mm/s, so a
    #     pose parked ~30 mm behind the board leans on it and pre-pushes the PCB toward the slot.
    # Translation is not boxed (``task_position_box_enabled=False``), so a residual horizontal gap is
    # the one error the policy can close by itself -- hence it is the objective, not a constraint.
    pad_z_w = pad_mid[:, 2] + cfg._ROBOT_BASE_POS[2]
    xy_err = (pad_mid[:, :2] - target_xyz[:, :2]).norm(dim=-1)
    # Shift the env-frame standoff limit into the same base-relative frame as ``target_xyz``.
    trailing_edge_y = cfg._PCB_INIT_POS[1] - cfg.PCB_X * 0.5
    pad_y_max = target_xyz[:, 1] + (trailing_edge_y - args.min_standoff_m - _target_pad_midpoint_env()[1])
    violation = (
        (_JAW_ALONG_X_MIN - jaw_along_x).clamp_min(0.0)
        + (jaw_tilt - _JAW_TILT_MAX).clamp_min(0.0)
        + (_APPROACH_ALONG_Y_MIN - approach_along_y).clamp_min(0.0)
        + (_PAD_Z_MIN_W - pad_z_w).clamp_min(0.0)
        + (pad_z_w - _PAD_Z_MAX_W).clamp_min(0.0)
        + (pad_mid[:, 1] - pad_y_max).clamp_min(0.0)
    )
    joint_margin = _joint_limit_margin_penalty(robot)
    cost = float(args.pos_weight) * xy_err + 50.0 * violation + joint_margin
    return {
        "cost": cost,
        "pos_err": pos_err,
        "xy_err": xy_err,
        "pad_mid": pad_mid,
        "jaw_along_x": jaw_along_x,
        "jaw_tilt": jaw_tilt,
        "approach_along_y": approach_along_y,
        "joint_margin": joint_margin,
        # How far the pads sit AHEAD of the wrist along +Y.  The arm only has to reach the WRIST, so
        # the reach needed at the far (insert) end of the slide is this much less than the pad travel.
        "pad_lead_y": pad_mid[:, 1] - p_wrist[:, 1],
        "jaw_vec": jaw,
        "approach_vec": reach,
    }


def _joint_limit_margin_penalty(robot, margin_rad: float = 0.20) -> torch.Tensor:
    """Penalise postures parked against a joint limit (leaves the policy no room to move)."""
    arm_ids = [robot.joint_names.index(n) for n in _ARM_JOINTS]
    q = robot.data.joint_pos[:, arm_ids]
    limits = robot.data.soft_joint_pos_limits[:, arm_ids]
    slack = torch.minimum(q - limits[..., 0], limits[..., 1] - q)
    return (margin_rad - slack).clamp_min(0.0).sum(dim=-1)


def _target_offset_for_base(base_xy: tuple[float, float], device: torch.device) -> torch.Tensor:
    """Straddle target as an offset FROM THE BASE, for a hypothetical base at ``base_xy``.

    The base placement is swept analytically this way (rather than by teleporting the articulation)
    because the reachable set simply translates with the base, and because random joint teleports
    inside the conveyor/magazine meshes corrupt the PhysX GPU scene.
    """
    tx, ty, tz = _target_pad_midpoint_env()
    bx, by = base_xy
    return torch.tensor([tx - bx, ty - by, tz - cfg._ROBOT_BASE_POS[2]], device=device).unsqueeze(0)


def _write_arm_joints(env: ManagerBasedRLEnv, q_arm: torch.Tensor, arm_ids: list[int]) -> None:
    """Write arm joint positions (zero velocity) and settle one physics step for FK."""
    robot = env.scene["robot"]
    q = robot.data.joint_pos.clone()
    q[:, arm_ids] = q_arm
    dq = torch.zeros_like(q)
    robot.write_joint_state_to_sim(q, dq)
    robot.write_data_to_sim()
    env.sim.step(render=False)
    robot.update(env.sim.get_physics_dt())


def _search_posture(
    env: ManagerBasedRLEnv,
    target: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    arm_ids: list[int],
    rounds: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Random + Gaussian-refinement posture search; returns best joints and its metrics."""
    q = lo + (hi - lo) * torch.rand(env.num_envs, 6, device=env.device)
    _write_arm_joints(env, q, arm_ids)
    m = _score(env, target)
    i = int(m["cost"].argmin())
    q_best, best_cost = q[i : i + 1].clone(), float(m["cost"][i])

    sigma = 0.40
    for _ in range(rounds):
        q = (q_best + sigma * torch.randn(env.num_envs, 6, device=env.device)).clamp(lo, hi)
        q[0] = q_best[0]
        _write_arm_joints(env, q, arm_ids)
        m = _score(env, target)
        i = int(m["cost"].argmin())
        if float(m["cost"][i]) < best_cost:
            best_cost, q_best = float(m["cost"][i]), q[i : i + 1].clone()
        sigma *= 0.6

    _write_arm_joints(env, q_best.repeat(env.num_envs, 1), arm_ids)
    m = _score(env, target)
    metrics = {
        "cost": best_cost,
        "pos_err": float(m["pos_err"][0]),
        "xy_err": float(m["xy_err"][0]),
        "jaw_along_x": float(m["jaw_along_x"][0]),
        "jaw_tilt": float(m["jaw_tilt"][0]),
        "approach_along_y": float(m["approach_along_y"][0]),
        "joint_margin": float(m["joint_margin"][0]),
        "pad_mid": [float(v) for v in m["pad_mid"][0]],
        "pad_lead_y": float(m["pad_lead_y"][0]),
    }
    return q_best, metrics


def _smoke_test() -> None:
    """Reset the real env (collisions on, real base) and hold still, to check the placement."""
    env_cfg = cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.vert_box_m is not None:
        env_cfg.actions.arm_action.vertical_half_range_m = float(args.vert_box_m)
        print(f"[smoke] EE vertical box half-range overridden to {args.vert_box_m * 1000:.0f} mm")
    if args.legacy_behind_base:
        env_cfg.scene.robot.init_state.pos = (0.06, -0.30, 0.0025)
        env_cfg.scene.robot.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
        env_cfg.scene.robot.init_state.joint_pos = {
            "joint_0": 0.0, "joint_1": 0.0, "joint_2": 0.35, "joint_3": -0.7,
            "joint_4": 0.0, "joint_5": 0.0,
            "left_carriage_joint": cfg._APPROACH_OPEN_WIDTH_M,
        }
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot, pcb = env.scene["robot"], env.scene["pcb"]
    left_idx = robot.body_names.index("gripper_left")
    right_idx = robot.body_names.index("gripper_right")
    pad0 = 0.5 * (robot.data.body_pos_w[:, left_idx] + robot.data.body_pos_w[:, right_idx])
    pcb0 = pcb.data.root_pos_w.clone()
    q0 = robot.data.joint_pos.clone()

    # Which links sit closest to the board?  A reset posture that rests on the PCB pre-moves it
    # before the policy acts at all, which is exactly the "PCB drifts toward the slot by itself"
    # failure mode -- so report the clearances rather than trusting the pad positions alone.
    board_half = torch.tensor(
        [cfg.PCB_Y * 0.5, cfg.PCB_X * 0.5, cfg.PCB_Z * 0.5], device=env.device
    )  # board long axis (PCB_X) lies along world Y
    body_pos = robot.data.body_pos_w - env.scene.env_origins.unsqueeze(1)
    delta = (body_pos - pcb0.unsqueeze(1) + env.scene.env_origins.unsqueeze(1)).abs() - board_half
    gap = delta.clamp_min(0.0).norm(dim=-1).mean(dim=0)  # per body, mean over envs
    order = torch.argsort(gap)[:6]
    print("\n  closest links to the PCB at reset (gap to board box, env 0 position):")
    for i in order.tolist():
        p = body_pos[0, i]
        print(
            f"    {robot.body_names[i]:>16s}  gap {float(gap[i]) * 1000:7.1f} mm   "
            f"pos ({float(p[0]):+.3f}, {float(p[1]):+.3f}, {float(p[2]):+.3f})"
        )

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    terminated_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for step in range(args.smoke_steps):
        _, _, terminated, truncated, _ = env.step(actions)
        terminated_any |= terminated | truncated
        if (step + 1) % 25 == 0:
            pad = 0.5 * (robot.data.body_pos_w[:, left_idx] + robot.data.body_pos_w[:, right_idx])
            d = pad - pad0
            print(
                f"  step {step + 1:4d} | PCB moved {(pcb.data.root_pos_w - pcb0).norm(dim=-1).max() * 1000:6.2f} mm"
                f" (max) | pad dz {d[:, 2].mean() * 1000:+7.2f} mm  dxy {d[:, :2].norm(dim=-1).mean() * 1000:6.2f} mm"
                f" | terminated {int(terminated_any.sum())}/{env.num_envs}"
            )

    pad = 0.5 * (robot.data.body_pos_w[:, left_idx] + robot.data.body_pos_w[:, right_idx])
    label = "LEGACY behind-belt base" if args.legacy_behind_base else "cfg base"
    d_pcb = pcb.data.root_pos_w - pcb0
    print(f"\n===== SMOKE TEST ({label}, zero actions, real collisions) =====")
    print(f"  PCB displacement  : mean {d_pcb.norm(dim=-1).mean() * 1000:.2f} mm"
          f"  max {d_pcb.norm(dim=-1).max() * 1000:.2f} mm")
    print(f"    per axis (mean) : dx {d_pcb[:, 0].mean() * 1000:+.2f}  dy {d_pcb[:, 1].mean() * 1000:+.2f}"
          f"  dz {d_pcb[:, 2].mean() * 1000:+.2f} mm  (dy>0 = drifting toward the slot)")
    print(f"  pad drift         : mean {(pad - pad0).norm(dim=-1).mean() * 1000:.2f} mm"
          f"  max {(pad - pad0).norm(dim=-1).max() * 1000:.2f} mm")
    print(f"    vertical (sag)  : mean {(pad[:, 2] - pad0[:, 2]).mean() * 1000:+.2f} mm"
          f"  worst {(pad[:, 2] - pad0[:, 2]).min() * 1000:+.2f} mm  (negative = drooping)")
    print(f"  episodes ended    : {int(terminated_any.sum())}/{env.num_envs}")

    # Which joint gives way?  Sag shows up as drift in the joints carrying the gravity torque.
    arm_ids = [robot.joint_names.index(n) for n in _ARM_JOINTS]
    dq = (robot.data.joint_pos[:, arm_ids] - q0[:, arm_ids]).mean(dim=0)
    print("  joint drift (mean):  " + "  ".join(f"{n}={float(v):+.4f}" for n, v in zip(_ARM_JOINTS, dq)))
    env.close()


def main() -> None:
    if args.smoke_steps > 0:
        _smoke_test()
        return

    env_cfg = cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # FK-only diagnostic.  Random joint teleports drive links deep into the conveyor/magazine meshes
    # (and into the arm itself), which corrupts the PhysX GPU scene ("Scene state is corrupted") and
    # then silently FREEZES all further joint writes -- every sweep row reports the same stale
    # posture.  Collisions are irrelevant to forward kinematics, so remove both sources: disable
    # self-collision, and (search mode only) spawn the arm 2 m up in free space.  All metrics are
    # base-relative, so lifting the base changes nothing that is measured.
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.legacy_behind_base:
        env_cfg.scene.robot.init_state.pos = (0.06, -0.30, 0.0025)
        env_cfg.scene.robot.init_state.rot = (0.7071068, 0.0, 0.0, 0.7071068)
    elif args.inspect:
        env_cfg.scene.robot.init_state.pos = tuple(args.base or cfg._ROBOT_BASE_POS)
        half = math.radians(args.base_yaw_deg) * 0.5
        env_cfg.scene.robot.init_state.rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    else:
        bx, by, bz = cfg._ROBOT_BASE_POS
        env_cfg.scene.robot.init_state.pos = (bx, by, bz + 2.0)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    device = env.device
    arm_ids = [robot.joint_names.index(n) for n in _ARM_JOINTS]

    limits = robot.data.soft_joint_pos_limits[0, arm_ids]  # (6, 2)
    lo, hi = limits[:, 0], limits[:, 1]
    base_x, base_y_cfg, base_z = cfg._ROBOT_BASE_POS
    if args.base_x is not None:
        base_x = args.base_x
    if args.legacy_behind_base:
        target = _target_offset_for_base((0.06, -0.30), device)
    elif args.inspect:
        insp = tuple(args.base or cfg._ROBOT_BASE_POS)
        target = _target_offset_for_base((insp[0], insp[1]), device)
    else:
        target = _target_offset_for_base((base_x, base_y_cfg), device)

    print(f"[INFO] base pos {cfg._ROBOT_BASE_POS}  rot {cfg._ROBOT_BASE_ROT_WXYZ}")
    print(f"[INFO] arm joint limits (rad):")
    for name, l, h in zip(_ARM_JOINTS, lo.tolist(), hi.tolist()):
        print(f"         {name}: [{l:+.3f}, {h:+.3f}]")
    print(f"[INFO] target pad midpoint (env frame): {_target_pad_midpoint_env()}")

    # Reference: the posture currently in the cfg, at the cfg's own base Y.
    home_vals = args.home if args.home is not None else [cfg._ROBOT_HOME_JOINT_POS[n] for n in _ARM_JOINTS]
    q_home = torch.tensor([[float(v) for v in home_vals]], device=device).repeat(env.num_envs, 1)
    _write_arm_joints(env, q_home, arm_ids)
    m = _score(env, target)
    label = "LEGACY behind-belt base" if args.legacy_behind_base else "INSPECT" if args.inspect else "CURRENT cfg home"
    base_used = (0.06, -0.30, 0.0025) if args.legacy_behind_base else tuple(args.base or cfg._ROBOT_BASE_POS)
    pad_env = [float(v) + b for v, b in zip(m["pad_mid"][0], base_used)]
    print(f"\n[{label}] base={base_used}  yaw={args.base_yaw_deg if args.inspect else 'cfg'} deg")
    print(f"  joints              : {[round(float(v), 4) for v in home_vals]}")
    print(f"  pad midpoint (env)  : {[round(v, 4) for v in pad_env]}")
    print(f"  jaw axis (world)    : {[round(float(v), 3) for v in m['jaw_vec'][0]]}")
    print(f"  approach dir (world): {[round(float(v), 3) for v in m['approach_vec'][0]]}")
    print(
        f"  pos_err={float(m['pos_err'][0]):.4f} m  |jaw·X|={float(m['jaw_along_x'][0]):.3f}  "
        f"|jaw_z|={float(m['jaw_tilt'][0]):.3f}  approach·Y={float(m['approach_along_y'][0]):+.3f}"
    )
    if args.legacy_behind_base or args.inspect:
        env.close()
        return

    # The wrist sits ~0.1-0.15 m BEHIND the pads (along -Y) whenever the jaws open across the PCB
    # width, so the base Y that makes the approach start reachable is NOT the pad-travel midpoint.
    # Sweep base Y and report what posture quality is achievable at each.
    lane_x = cfg._CONVEYOR_CENTER_X_ENV
    insert_end_y = cfg._MAG_Y_NEAR_FACE_ENV
    print("\n===== base Y sweep (base X fixed at %.3f) =====" % base_x)
    print("  base_y | cost  | xy_err  |jaw·X| |jaw_z| appr·Y | jlim | reach_start reach_end | pad z")
    results: list[tuple[float, torch.Tensor, dict[str, float]]] = []
    candidates = [args.base_y] if args.base_y is not None else [round(-0.28 + 0.04 * i, 3) for i in range(12)]
    for base_y in candidates:
        target_off = _target_offset_for_base((base_x, base_y), device)
        q_best, met = min(
            (_search_posture(env, target_off, lo, hi, arm_ids, args.rounds) for _ in range(args.restarts)),
            key=lambda r: r[1]["cost"],
        )
        # Metrics are base-relative; express the pad midpoint in the env frame for reporting.
        met["pad_mid"] = [
            met["pad_mid"][0] + base_x,
            met["pad_mid"][1] + base_y,
            met["pad_mid"][2] + base_z,
        ]
        # Reach is what the WRIST must span, not the pads (the pads lead the wrist by ~0.15 m along +Y).
        wrist_start_y = met["pad_mid"][1] - met["pad_lead_y"]
        reach_start = float(((met["pad_mid"][0] - base_x) ** 2 + (wrist_start_y - base_y) ** 2) ** 0.5)
        reach_end = float(((lane_x - base_x) ** 2 + (insert_end_y - met["pad_lead_y"] - base_y) ** 2) ** 0.5)
        results.append((base_y, q_best, {**met, "reach_start": reach_start, "reach_end": reach_end}))
        print(
            f"  {base_y:+.3f} | {met['cost']:.3f} | {met['xy_err'] * 1000:6.1f}mm "
            f"{met['jaw_along_x']:.3f}  {met['jaw_tilt']:.3f}  {met['approach_along_y']:+.3f} | "
            f"{met['joint_margin']:.2f} | {reach_start:.3f} m    {reach_end:.3f} m | z={met['pad_mid'][2]:.3f}"
        )

    base_y, q_best, met = min(results, key=lambda r: r[2]["cost"])
    print("\n===== BEST (side base placement) =====")
    print(f"  _ROBOT_BASE_POS = ({base_x}, {base_y}, {cfg._ROBOT_BASE_POS[2]})")
    print(f"  pad midpoint (env)  : {[round(v, 4) for v in met['pad_mid']]}")
    print(f"  target       (env)  : {[round(v, 4) for v in _target_pad_midpoint_env()]}")
    print(f"  horizontal error    : {met['xy_err'] * 1000:.1f} mm  (total {met['pos_err'] * 1000:.1f} mm incl. Z)")
    print(f"  |jaw · worldX|      : {met['jaw_along_x']:.3f}  (1.0 = jaws open along PCB width)")
    print(f"  |jaw_z| (tilt)      : {met['jaw_tilt']:.3f}  (0.0 = both pads same height)")
    print(f"  approach · worldY   : {met['approach_along_y']:+.3f}  (1.0 = straight into trailing face)")
    print(f"  joint-limit penalty : {met['joint_margin']:.3f}  (0.0 = all joints >0.20 rad from a limit)")
    print(f"  reach start / end   : {met['reach_start']:.3f} m / {met['reach_end']:.3f} m  (arm max ~0.769 m)")
    print("\n  _ROBOT_HOME_JOINT_POS = {")
    for name, v in zip(_ARM_JOINTS, q_best[0].tolist()):
        print(f'      "{name}": {v:+.4f},')
    print('      "left_carriage_joint": _APPROACH_OPEN_WIDTH_M,')
    print("  }")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
