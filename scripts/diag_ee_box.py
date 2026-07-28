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


def main() -> None:
    env_cfg = cfg.WidowXPcbSlideEnvCfg() if args.slide else cfg.WidowXPcbApproachEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
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

    stats = {"calls": 0}
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
        trace.append(
            (
                stats["calls"],
                z0,
                z_cur,
                dz_cmd,
                _wrist_pitch_deg_env(env) if args.slide else float("nan"),
                float(term._cum_rot_vec[0, 0]),
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
    for _ in range(args.steps):
        env.step(action)

    print(f"\n[trace] clamp called {stats['calls']} times in {args.steps} steps")
    print(f"[box  ] rot lo = {term._rot_box_lo[0].tolist()}  hi = {term._rot_box_hi[0].tolist()}")
    print("  step |   anchor z0 |    live z   | offset (mm) | commanded dz_b (mm) | pitch (deg) | cum rx")
    for i, z0, z_cur, dz, pitch, crx in trace[:: max(1, len(trace) // 20)]:
        print(
            f"  {i:4d} | {z0:11.4f} | {z_cur:11.4f} | {(z_cur - z0) * 1000:+11.2f} | {dz * 1000:+11.3f}"
            f" | {pitch:+11.2f} | {crx:+7.3f}"
        )
    if args.slide:
        _print_clearance(env, "after")
        rew = env.reward_manager._episode_sums
        print("\n[slide] episode reward sums (env 0), most negative/positive first:")
        for name, val in sorted(rew.items(), key=lambda kv: float(kv[1][0])):
            print(f"    {name:34s} {float(val[0]):+12.3f}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
