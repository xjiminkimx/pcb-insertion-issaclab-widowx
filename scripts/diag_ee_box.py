"""Instrument the Approach EE translation box: is it called, and what is it anchored to?

The arm sinks ~70 mm per episode with ZERO actions.  ``task_position_box_enabled`` was switched on to
bound that, but the sag did not change even with a 10 mm box -- so either the clamp never runs, or its
anchor (``env._slide_reset_ee_pos_w``, written by the ``store_reset_ee_pose`` reset event) is not the
home pose.  This traces both: call count, the anchor, and the vertical offset before/after clamping.

Run:
    python -u scripts/diag_ee_box.py --num_envs 4 --steps 120 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--vert_box_m", type=float, default=None, help="Override the vertical box half-range.")
parser.add_argument(
    "--push_lead_m",
    type=float,
    default=None,
    help="Override push_lead_max_m.  This trades the two failure modes off against each other: too "
    "large and the OSC drags the arm to its reach boundary and the wrist collapses, too small and "
    "the push force never breaks the board's static friction.",
)
parser.add_argument(
    "--stiff_cmd",
    type=float,
    default=0.0,
    help="Constant normalised stiffness action on every task axis (-1 = softest, +1 = stiffest).",
)
parser.add_argument(
    "--axis_max",
    type=float,
    nargs=2,
    action="append",
    default=None,
    metavar=("AXIS", "K_MAX"),
    help="Override one task axis' stiffness ceiling, e.g. --axis_max 2 1000 for tz.  Axis order is "
    "tx ty tz rx ry rz.  Repeatable.",
)
parser.add_argument(
    "--rot_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("RX", "RY", "RZ"),
    help="Hold a constant normalised rotation action instead of zeros, to check which sign of each "
    "task rotation axis does what and where the orientation box stops it.",
)
parser.add_argument(
    "--slide",
    action="store_true",
    help="Instrument the SLIDE phase instead of Approach, and also report the wrist->jaw tip-down "
    "pitch and the ``wrist_tip_down`` reward each sample (checks the pitch shaping is not a dead "
    "zone at the replayed straddle reset pose).",
)
parser.add_argument(
    "--pos_cmd",
    type=float,
    nargs=3,
    default=None,
    metavar=("TX", "TY", "TZ"),
    help="Hold a constant normalised translation action.  Use --pos_cmd 0 1 0 to drive the push "
    "axis to the far end of the position box and check whether the wrist can still hold its "
    "commanded orientation once the target leaves the arm's comfortable reach.",
)
parser.add_argument(
    "--pcb_dy",
    type=float,
    default=None,
    help="Shift the PCB start along the push axis (m).",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=0,
    help="Run this many back-to-back episodes and, at every reset, compare the stored box anchor "
    "against the EE pose actually measured after that reset.  Every other mode here samples a "
    "single episode, which cannot see an anchor that inherits the previous episode's drooped pose "
    "and therefore compounds the droop episode over episode.",
)
parser.add_argument(
    "--episode_s",
    type=float,
    default=None,
    help="Override episode_length_s.  In --episodes mode it defaults to 1.0 s so many resets "
    "happen quickly; in GUI inspection it can be raised so a jammed board stays jammed on screen "
    "instead of being reset away.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks.manager_based.widowx_pcb.widowx_pcb_env_cfg as cfg  # noqa: E402


_PITCH_ENTS: list = []


def _wrist_pitch_deg_env(env, env_id: int = 0) -> float:
    """Signed wrist->jaw pitch of one env in degrees (negative = tip-down)."""
    import copy

    import isaaclab_tasks.manager_based.widowx_pcb.mdp_custom as mdp_custom

    if not _PITCH_ENTS:
        # The module-level SceneEntityCfgs are unresolved templates (``body_ids`` is still a slice).
        for ent in (cfg._LEFT_FINGER, cfg._RIGHT_FINGER, cfg._WRIST_BODY):
            resolved = copy.deepcopy(ent)
            resolved.resolve(env.scene)
            _PITCH_ENTS.append(resolved)
    return float(mdp_custom.gripper_wrist_pitch_deg_signed_obs(env, *_PITCH_ENTS)[env_id])


_GRIPPER_BODY_HINTS = ("gripper", "carriage", "link_6", "link_5")


def _print_clearance(env, label: str) -> None:
    """Rail-guide clearance proxy: gripper-side body heights relative to the board plane.

    The pads must stay at the PCB mid-thickness to push the trailing edge, so the tip-down pitch is
    only worth having if it lifts the surrounding hardware (carriage, wrist) away from the rails.
    Print it at reset and again after the run so the sign and size of that effect are measured, not
    assumed.
    """
    robot = env.scene["robot"]
    pcb = env.scene["pcb"]
    board_top_z = float(pcb.data.root_pos_w[0, 2]) + cfg.PCB_Z * 0.5
    print(f"\n[clearance {label}]  board top Z = {board_top_z:.4f}  pitch = {_wrist_pitch_deg_env(env):+.2f} deg")
    for i, name in enumerate(robot.body_names):
        if not any(h in name for h in _GRIPPER_BODY_HINTS):
            continue
        dz = float(robot.data.body_pos_w[0, i, 2]) - board_top_z
        print(f"    {name:24s} {dz * 1000:+8.1f} mm vs board top")


def _print_stall_report(env) -> None:
    """When the push stops advancing, is the ARM out of torque or is the BOARD jammed?

    Both look identical from the EE trace (target pinned at the leash limit, nothing moving), but
    they need opposite fixes: actuator/reach limits mean the task geometry is wrong, while a jam
    means the board is binding on the rails and the lateral/yaw control is at fault.
    """
    robot = env.scene["robot"]
    pcb = env.scene["pcb"]
    tau = robot.data.applied_torque[0]
    lim = robot.data.joint_effort_limits[0]
    print("\n[stall] arm joint torque vs limit (env 0)")
    for i, name in enumerate(robot.joint_names):
        if not name.startswith("joint_"):
            continue
        frac = abs(float(tau[i])) / max(float(lim[i]), 1e-9)
        flag = "  <-- SATURATED" if frac > 0.95 else ""
        print(f"    {name:10s} {float(tau[i]):+8.2f} / {float(lim[i]):6.2f} Nm  ({frac * 100:5.1f}%){flag}")
    import isaaclab.utils.math as math_utils

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(pcb.data.root_quat_w[:1])
    print(
        f"[stall] PCB yaw = {float(yaw[0]) * 57.2958:+.2f} deg   lateral X drift ="
        f" {float(pcb.data.root_pos_w[0, 0] - env.scene.env_origins[0, 0]) * 1000:+.1f} mm"
    )


def _reset_anchor_probe(env, term, action) -> None:
    """Across several resets: does the box anchor match the pose the arm was actually reset into?

    The anchor is written by the ``store_*_reset_ee_pose`` event, which reads the EE body pose
    during the reset event.  If that read is stale -- i.e. it returns the pose from the END of the
    previous episode rather than the freshly replayed straddle pose -- then the "absolute" box is
    anchored to an already-drooped pose, the OSC happily drives the arm there, and the next reset
    anchors to something even lower.  That compounding is invisible to a single-episode sample and
    is the only mechanism that can take the wrist past the box limit to a fully collapsed jaw.
    """
    print(f"\n[probe] {args.episodes} episodes @ episode_length_s={args.episode_s}")
    print("   ep | pitch @ end | pitch @ reset | anchor-vs-live pos err | anchor z | live z")
    steps_per_ep = int(args.episode_s / env.step_dt) + 2
    for ep in range(args.episodes):
        pitch_end = float("nan")
        for _ in range(steps_per_ep):
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            if bool(done.any()):
                break
            pitch_end = _wrist_pitch_deg_env(env, 0)
        # env.step() already reset the finished envs, so this reads the post-reset state.
        term._compute_ee_pose()
        live = term._ee_pose_w[0, :3]
        anchor = env._slide_reset_ee_pos_w[0]
        err_mm = float((live - anchor).norm()) * 1000.0
        print(
            f"  {ep:3d} | {pitch_end:+11.2f} | {_wrist_pitch_deg_env(env, 0):+13.2f} |"
            f" {err_mm:19.2f} mm | {float(anchor[2]):8.4f} | {float(live[2]):6.4f}"
        )


def main() -> None:
    env_cfg = cfg.WidowXPcbSlideEnvCfg() if args.slide else cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.episodes > 0 and args.episode_s is None:
        args.episode_s = 1.0
    if args.episode_s is not None:
        env_cfg.episode_length_s = float(args.episode_s)
    # The shipped PhysX GPU buffers are sized for thousands of envs (1 GB collision stack alone).
    # A handful of diagnostic envs does not need them, and asking for them while a training or play
    # session holds most of the card makes ``PxgCudaDeviceMemoryAllocator`` fail -- after which
    # PhysX cannot launch its narrowphase kernels and the articulation silently FREEZES, reporting
    # the same stale pose every step (which looks exactly like a perfectly-held arm, not an error).
    env_cfg.sim.physx.gpu_collision_stack_size = 2**26
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**17
    if args.vert_box_m is not None:
        env_cfg.actions.arm_action.vertical_half_range_m = float(args.vert_box_m)
    if args.push_lead_m is not None:
        env_cfg.actions.arm_action.push_lead_max_m = float(args.push_lead_m)
    if args.pcb_dy is not None:
        # Shifting the board's start separates the two reasons a push can stop: an obstacle stops it
        # at the same ABSOLUTE Y no matter where it started, while an arm/friction limit lets it
        # cover the same DISTANCE from wherever it starts.
        p = list(env_cfg.scene.pcb.init_state.pos)
        p[1] += float(args.pcb_dy)
        env_cfg.scene.pcb.init_state.pos = tuple(p)
        print(f"[cfg ] PCB start Y shifted by {args.pcb_dy:+.3f} m -> {p[1]:+.4f}")
    if args.axis_max:
        limits = [list(b) for b in env_cfg.actions.arm_action.motion_stiffness_limits_per_axis]
        for axis, k_max in args.axis_max:
            limits[int(axis)][1] = float(k_max)
        env_cfg.actions.arm_action.motion_stiffness_limits_per_axis = tuple(tuple(b) for b in limits)
        print(f"[cfg ] stiffness ceilings overridden -> {env_cfg.actions.arm_action.motion_stiffness_limits_per_axis}")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    term = env.action_manager.get_term("arm_action")
    print(f"[cfg ] task_position_box_enabled = {getattr(term.cfg, 'task_position_box_enabled', None)}")
    print(f"[cfg ] vertical_half_range_m     = {getattr(term.cfg, 'vertical_half_range_m', None)}")
    print(f"[term] _pose_rel_idx             = {term._pose_rel_idx}")
    print(f"[term] _task_vertical_axis_w     = {term._task_vertical_axis_w}")
    print(f"[env ] has _slide_reset_ee_pos_w = {hasattr(env, '_slide_reset_ee_pos_w')}")
    if hasattr(env, "_slide_reset_ee_pos_w"):
        print(f"[env ] anchor p0 (env 0)         = {env._slide_reset_ee_pos_w[0].tolist()}")
    term._compute_ee_pose()
    print(f"[term] live EE pose  (env 0)     = {term._ee_pose_w[0, :3].tolist()}")

    if args.slide:
        # Does the home pose's orientation error survive into the replayed straddle states?  Slide
        # never reads ``_ROBOT_HOME_JOINT_POS``, but the Approach policy that produced these states
        # started from it and could only correct ~8.6 deg of it (the orientation box), so any jaw
        # roll baked into the home pose is inherited here.  Same metrics as
        # ``diag_side_base_home_pose.py``: |jaw.X| ~ 1 and |jaw_z| ~ 0 is a level jaw rail.
        robot = env.scene["robot"]
        li = robot.body_names.index("gripper_left")
        ri = robot.body_names.index("gripper_right")
        jaw = robot.data.body_pos_w[:, ri] - robot.data.body_pos_w[:, li]
        jaw = jaw / jaw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        print("\n[slide reset geometry, per env]  (home pose measures |jaw.X|=0.974 |jaw_z|=0.137)")
        for e in range(env.num_envs):
            print(
                f"    env {e}: |jaw.X|={float(jaw[e, 0].abs()):.3f}  |jaw_z|={float(jaw[e, 2].abs()):.3f}"
                f"  pitch={_wrist_pitch_deg_env(env, e):+.2f} deg"
            )

    stats = {"calls": 0, "pcb_y0": None}
    original = term._clamp_pose_rel_to_reset_position_box
    trace: list[tuple[int, float, float, float, float]] = []

    def wrapped() -> None:
        stats["calls"] += 1
        idx = term._pose_rel_idx
        term._compute_ee_pose()
        z_cur = float(term._ee_pose_w[0, 2])
        z0 = float(env._slide_reset_ee_pos_w[0, 2]) if hasattr(env, "_slide_reset_ee_pos_w") else float("nan")
        original()
        dz_cmd = float(term._processed_actions[0, idx + 2])
        push_w = term._task_push_axis_w
        # Commanded push offset (inside the box) vs the one the arm actually achieved.  A gap here
        # means the OSC target has left the reachable workspace and the arm is being dragged.
        push_cmd = float((term._cum_pos_w[0] * push_w).sum())
        push_act = float(((term._ee_pose_w[0, :3] - env._slide_reset_ee_pos_w[0]) * push_w).sum())
        # EE travel is not board travel: if the pads ride up over the 1 mm edge the arm keeps
        # advancing while the PCB stays put, which looks like progress in every EE-side metric.
        pcb_y = float((env.scene["pcb"].data.root_pos_w[0] * push_w).sum())
        if stats.get("pcb_y0") is None:
            stats["pcb_y0"] = pcb_y
        pcb_travel = pcb_y - stats["pcb_y0"]
        stats["pcb_y_abs"] = pcb_y - float((env.scene.env_origins[0] * push_w).sum())
        trace.append(
            (
                stats["calls"],
                z0,
                z_cur,
                dz_cmd,
                _wrist_pitch_deg_env(env) if args.slide else float("nan"),
                float(term._cum_rot_vec[0, 0]),
                push_cmd,
                push_act,
                pcb_travel,
            )
        )

    term._clamp_pose_rel_to_reset_position_box = wrapped

    if args.slide:
        _print_clearance(env, "at reset")

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    if args.rot_cmd is not None:
        idx = term._pose_rel_idx
        action[:, idx + 3 : idx + 6] = torch.tensor(args.rot_cmd, device=env.device)
        print(f"\n[cmd ] constant rotation action (rx, ry, rz) = {args.rot_cmd}")
    if args.pos_cmd is not None:
        idx = term._pose_rel_idx
        action[:, idx : idx + 3] = torch.tensor(args.pos_cmd, device=env.device)
        print(f"[cmd ] constant translation action (tx, ty, tz) = {args.pos_cmd}")
    if term._stiffness_idx is not None and args.stiff_cmd != 0.0:
        action[:, term._stiffness_idx : term._stiffness_idx + 6] = args.stiff_cmd
    print(
        f"[cmd ] push_lead_max_m = {getattr(term.cfg, 'push_lead_max_m', None)}"
        f"  stiffness action = {args.stiff_cmd}"
    )
    if args.episodes > 0:
        _reset_anchor_probe(env, term, action)
        env.close()
        return

    for _ in range(args.steps):
        env.step(action)

    print(f"\n[trace] clamp called {stats['calls']} times in {args.steps} steps")
    print(f"[box  ] rot lo = {term._rot_box_lo[0].tolist()}  hi = {term._rot_box_hi[0].tolist()}")
    print(
        "  step | vert (mm) | pitch (deg) | push cmd (mm) | EE push (mm) | lag (mm) |"
        " PCB travel (mm)"
    )
    for i, z0, z_cur, _dz, pitch, _crx, p_cmd, p_act, pcb in trace[:: max(1, len(trace) // 20)]:
        print(
            f"  {i:4d} | {(z_cur - z0) * 1000:+9.1f} | {pitch:+11.2f} | {p_cmd * 1000:+13.1f} |"
            f" {p_act * 1000:+12.1f} | {(p_cmd - p_act) * 1000:+8.1f} | {pcb * 1000:+15.1f}"
        )
    if args.slide:
        _print_clearance(env, "after")
        print(
            f"[stall] PCB stopped at env Y = {stats['pcb_y_abs']:+.4f}"
            f"   after travelling {(stats['pcb_y_abs'] - (stats['pcb_y0'] - float(env.scene.env_origins[0, 1]))) * 1000:+.1f} mm"
        )
        _print_stall_report(env)
        rew = env.reward_manager._episode_sums
        print("\n[slide] episode reward sums (env 0), most negative/positive first:")
        for name, val in sorted(rew.items(), key=lambda kv: float(kv[1][0])):
            print(f"    {name:34s} {float(val[0]):+12.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
