"""Name the fixture part each gripper-side body will run into during the insert.

The board itself is not what jams: the pads sit at the 1 mm board plane, but the carriage and wrist
ride 7-35 mm ABOVE it, and that is the band the conveyor's end structure lives in.  An earlier scan
that only looked at the board plane found nothing but the magazine and drew the wrong conclusion.

For every gripper-side body this walks the fixture's collision prims, keeps the ones that overlap
that body's own X and Z footprint, and reports the nearest one ahead of it along the push axis --
i.e. the part it is about to hit and how much room is left.

Run:
    python -u scripts/diag_insert_obstacle.py --headless
    python -u scripts/diag_insert_obstacle.py --headless --pitch_cmd -1   # hold max tip-down first
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--approach",
    action="store_true",
    help="Measure the Approach env instead of Insert, to check what wrist pitch the Approach "
    "orientation box actually allows -- Insert inherits whatever posture Approach hands over.",
)
parser.add_argument(
    "--rot_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("RX", "RY", "RZ"),
    help="Hold a constant rotation action for --settle steps before measuring, so the clearance "
    "is reported for the posture the policy would actually slide in.",
)
parser.add_argument(
    "--tz_max",
    type=float,
    default=None,
    help="Override the vertical stiffness ceiling (N/m) to test how much of the sink is just "
    "static gravity sag against a soft vertical spring.",
)
parser.add_argument(
    "--ry_box",
    type=float,
    default=None,
    help="Widen the Insert roll (ry) cumulative bound to +/- this many rad.  The reset pose carries "
    "~13 deg of inherited jaw roll and the shipped +/-0.10 rad box cannot undo it.",
)
parser.add_argument(
    "--pos_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("TX", "TY", "TZ"),
    help="Hold a constant translation action while settling.  Use +Z to test whether lifting the "
    "whole EE pulls the buried pad tips back to the board plane AND lifts the carriage clear.",
)
parser.add_argument(
    "--tz_box",
    type=float,
    default=None,
    help="Override the vertical half-range (m) of the cumulative EE position box, which is what "
    "caps how far the EE can climb away from the (possibly too-low) reset anchor.",
)
parser.add_argument("--settle", type=int, default=150, help="Steps to settle before measuring.")
parser.add_argument(
    "--stiff_cmd",
    type=float,
    default=0.0,
    help="Hold this stiffness action on every axis while settling.  Matters a lot: a slack hold "
    "lets the whole EE sink into the rails even though the reset pose clears them.",
)
parser.add_argument(
    "--joint_delta",
    nargs=2,
    action="append",
    default=None,
    metavar=("NAME", "RAD"),
    help="Offset a joint by RAD right after reset and re-measure with pure kinematics (no policy "
    "stepping), to find what levels the jaw.  Repeatable, e.g. --joint_delta joint_5 0.2.",
)
parser.add_argument(
    "--xz_tol_mm",
    type=float,
    default=25.0,
    help="Half-size of the footprint assumed around each body origin when testing overlap.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks.manager_based.widowx_pcb.widowx_pcb_env_cfg as cfg  # noqa: E402

_BODY_HINTS = ("gripper", "carriage", "link_6", "link_5")


def _fixture_boxes(env) -> list[tuple[str, tuple[float, float, float, float, float, float]]]:
    """World-frame AABBs of every fixture collision prim, expressed in env-0 frame."""
    origin = env.scene.env_origins[0].tolist()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    boxes = []
    for prim in env.sim.stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/env_0") or "/PCB" in path or "/Robot" in path:
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        boxes.append(
            (
                path.split("/env_0/")[-1],
                (
                    lo[0] - origin[0],
                    hi[0] - origin[0],
                    lo[1] - origin[1],
                    hi[1] - origin[1],
                    lo[2] - origin[2],
                    hi[2] - origin[2],
                ),
            )
        )
    return boxes


def _wrist_pitch_deg(env) -> float:
    """Signed wrist->pad-tip pitch of env 0, in degrees.  Negative is tip-down."""
    import copy

    from isaaclab_tasks.manager_based.widowx_pcb.mdp_custom import gripper_wrist_pitch_deg_signed_obs

    left, right, wrist = (copy.deepcopy(c) for c in (cfg._LEFT_FINGER, cfg._RIGHT_FINGER, cfg._WRIST_BODY))
    for c in (left, right, wrist):
        c.resolve(env.scene)
    return float(gripper_wrist_pitch_deg_signed_obs(env, left, right, wrist).mean())


def _tip_thickness_mm(env) -> tuple[float, float]:
    """Signed pad-tip offset (mm) from the trailing-face centre along the board thickness axis.

    This is the quantity ``gripper_tip_under_pcb_penalty`` scores: positive is above the mid plane,
    below ``-half_thickness`` means the tip has gone under the board and will shovel it.
    """
    import copy

    from isaaclab_tasks.manager_based.widowx_pcb.mdp_custom import _straddle_width_target_tip_dists

    ents = {
        k: copy.deepcopy(v)
        for k, v in (
            ("pcb_cfg", cfg._PCB_ENT),
            ("left_finger_cfg", cfg._LEFT_FINGER),
            ("right_finger_cfg", cfg._RIGHT_FINGER),
            ("gripper_joint_cfg", cfg._GRIPPER_JOINT),
            ("wrist_body_cfg", cfg._WRIST_BODY),
        )
    }
    for c in ents.values():
        c.resolve(env.scene)
    *_, thick_l, thick_r = _straddle_width_target_tip_dists(
        env,
        half_length_m=cfg._HALF_LENGTH_M,
        finger_offset_m=cfg._APPROACH_FINGER_OFFSET_M,
        tip_offset_m=cfg._GRIPPER_TIP_OFFSET_M,
        width_gap_target_left_m=cfg._APPROACH_GAP_LEFT_M,
        width_gap_target_right_m=cfg._APPROACH_GAP_RIGHT_M,
        **ents,
    )
    return float(thick_l.mean()) * 1000.0, float(thick_r.mean()) * 1000.0


def main() -> None:
    env_cfg = cfg.WidowXPcbApproachEnvCfg() if args.approach else cfg.WidowXPcbInsertEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.physx.gpu_collision_stack_size = 2**26
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**17
    if args.tz_max is not None:
        lo, hi = env_cfg.actions.arm_action.controller_cfg.motion_stiffness_limits_task
        env_cfg.actions.arm_action.controller_cfg.motion_stiffness_limits_task = (
            lo,
            max(float(hi), float(args.tz_max)),
        )
        a = [list(x) for x in env_cfg.actions.arm_action.motion_stiffness_limits_per_axis]
        a[2][1] = float(args.tz_max)
        env_cfg.actions.arm_action.motion_stiffness_limits_per_axis = tuple(tuple(x) for x in a)
    if args.tz_box is not None:
        env_cfg.actions.arm_action.vertical_half_range_m = float(args.tz_box)
    if args.ry_box is not None:
        b = [list(x) for x in env_cfg.actions.arm_action.orientation_dev_limits_per_axis]
        b[1] = [-abs(args.ry_box), abs(args.ry_box)]
        env_cfg.actions.arm_action.orientation_dev_limits_per_axis = tuple(tuple(x) for x in b)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot0 = env.scene["robot"]
    if args.joint_delta:
        q = robot0.data.joint_pos.clone()
        for name, rad in args.joint_delta:
            q[:, robot0.joint_names.index(name)] += float(rad)
        robot0.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.step(render=False)
        robot0.update(env.sim.get_physics_dt())
        args.settle = 0

    term = env.action_manager.get_term("arm_action")
    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    if term._stiffness_idx is not None and args.stiff_cmd != 0.0:
        action[:, term._stiffness_idx : term._stiffness_idx + 6] = args.stiff_cmd
    if args.rot_cmd is not None:
        action[:, term._pose_rel_idx + 3 : term._pose_rel_idx + 6] = torch.tensor(
            args.rot_cmd, device=env.device
        )
    if args.pos_cmd is not None:
        action[:, term._pose_rel_idx : term._pose_rel_idx + 3] = torch.tensor(
            args.pos_cmd, device=env.device
        )
    for _ in range(args.settle):
        env.step(action)

    robot = env.scene["robot"]
    origin = env.scene.env_origins[0]
    board_z = float(env.scene["pcb"].data.root_pos_w[0, 2] - origin[2])
    boxes = _fixture_boxes(env)
    tol = args.xz_tol_mm * 0.001

    print(f"\nboard plane env Z = {board_z:.4f}    fixture prims scanned = {len(boxes)}")
    print(
        f"rotation = {args.rot_cmd}  translation = {args.pos_cmd}  stiffness = {args.stiff_cmd}  "
        f"tz_box = {env_cfg.actions.arm_action.vertical_half_range_m}  settle = {args.settle}"
    )
    print(f"wrist->pad-tip pitch = {_wrist_pitch_deg(env):+.2f} deg  (negative = tip-down)")
    thick_l, thick_r = _tip_thickness_mm(env)
    half_mm = cfg.PCB_Z * 0.5 * 1000.0
    print(
        f"pad tip vs trailing-face mid plane: L {thick_l:+.1f} mm  R {thick_r:+.1f} mm   "
        f"(under the board below {-half_mm:+.2f} mm)\n"
    )
    print("  fixture parts sitting in or just under the board plane (the rails the board rides on):")
    for part, (x0, x1, y0, y1, z0, z1) in sorted(boxes, key=lambda b: b[1][0]):
        if z1 < board_z - 0.06 or z0 > board_z + 0.06:
            continue
        print(
            f"    X[{x0:+.4f},{x1:+.4f}]  Y[{y0:+.4f},{y1:+.4f}]  Z[{z0:+.4f},{z1:+.4f}]  {part}"
        )
    pcb = env.scene["pcb"]
    print(f"    board root X = {float(pcb.data.root_pos_w[0, 0] - origin[0]):+.4f}\n")
    # Per-step reward contribution of each term at THIS pose, i.e. what the policy is being paid to
    # hold.  Term values are already weighted and scaled by ``step_dt``, so dividing by dt gives the
    # per-second income used in the static-income ledger on ``alive_penalty``.
    rm = env.reward_manager
    print("  reward term                     per second at this pose")
    total = 0.0
    for name, term_cfg in zip(rm.active_terms, rm._term_cfgs):
        value = float(term_cfg.func(env, **term_cfg.params).mean()) * float(term_cfg.weight)
        total += value
        print(f"  {name:30s} {value:+10.2f}")
    print(f"  {'TOTAL':30s} {total:+10.2f}\n")
    # Averaged over every env, not env 0.  Each env replays a DIFFERENT handover state out of the
    # 500 in the buffer, so single-env readings carry the spread of the buffer and cannot be
    # compared across configurations -- an A/B of two control settings on env 0 alone is mostly
    # measuring which two states happened to be sampled.
    print(
        "  body              env X      env Y     dZ vs board (mean +/- sd, min)   "
        "nearest part ahead        gap +Y"
    )
    # Per-env origins, not env 0's: the envs sit on a grid, so subtracting a single origin from
    # every env's world position would put the mean X/Y somewhere in the middle of the grid and
    # silently break the corridor lookup below.
    origins = env.scene.env_origins
    all_z = (robot.data.body_pos_w[:, :, 2] - origins[:, 2:3] - board_z) * 1000.0
    for i, name in enumerate(robot.body_names):
        if not any(h in name for h in _BODY_HINTS):
            continue
        p = robot.data.body_pos_w[:, i] - origins
        bx, by, bz = float(p[:, 0].mean()), float(p[:, 1].mean()), float(p[:, 2].mean())
        dz = all_z[:, i]
        ahead = [
            (y0 - by, part)
            for part, (x0, x1, y0, y1, z0, z1) in boxes
            if x0 - tol <= bx <= x1 + tol and z0 - tol <= bz <= z1 + tol and y1 > by
        ]
        ahead.sort()
        gap, part = ahead[0] if ahead else (float("nan"), "-- nothing in this corridor --")
        print(
            f"  {name:16s} {bx:+8.4f}  {by:+8.4f}   "
            f"{float(dz.mean()):+7.1f} +/-{float(dz.std()):4.1f}  min {float(dz.min()):+7.1f} mm   "
            f"{part:26s} {gap * 1000:+8.1f} mm"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
