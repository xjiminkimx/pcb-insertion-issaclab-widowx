"""Measure the true wrist->pad-tip distance and compare it to ``_GRIPPER_TIP_OFFSET_M``.

Every straddle reward locates the contact point as ``pad_midpoint + _GRIPPER_TIP_OFFSET_M * fwd``,
where ``fwd`` is the unit vector from ``link_6`` toward the jaw pad midpoint (see
``gripper_jaw_pad_tips_world`` / ``_gripper_tip_offset_direction_w`` in mdp_custom.py).  That 60 mm
is a hand-entered constant, and it is load-bearing: at the ~21 deg tip-down pose the Slide phase
inherits, the assumed lever alone accounts for the full ~12 mm of "pad tip is under the board" that
``gripper_tip_under_pcb_penalty`` reports.  If the real pads are shorter, the rewards are scoring a
phantom point in free space.

This walks the actual jaw meshes, projects every vertex onto ``fwd``, and reports how far the most
distal real geometry sits from the pad midpoint -- i.e. what the constant should have been.

Run:
    python -u scripts/diag_gripper_tip_offset.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--approach",
    action="store_true",
    help="Measure in the Approach env instead of Slide (the geometry is identical; this only "
    "changes the pose the projection is reported at).",
)
parser.add_argument("--settle", type=int, default=60, help="Steps to settle before measuring.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import copy  # noqa: E402

import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import isaaclab_tasks.manager_based.widowx_pcb.widowx_pcb_env_cfg as cfg  # noqa: E402
from isaaclab_tasks.manager_based.widowx_pcb.mdp_custom import (  # noqa: E402
    _gripper_tip_offset_direction_w,
    _resolve_first_body_id,
    gripper_jaw_pad_tips_world,
)

# Jaw-side prim subtrees to measure.  ``carriage`` is included because gripper_left/right share an
# origin with it, so it is worth seeing which subtree actually carries the distal pad geometry.
_JAW_HINTS = ("gripper_left", "gripper_right", "carriage_left", "carriage_right")


def _mesh_points_in_body_frame(prim: Usd.Prim, body_prim: Usd.Prim) -> torch.Tensor | None:
    """Vertices of ``prim`` expressed in ``body_prim``'s frame.

    The USD stage is NOT written back from the GPU physics pipeline in headless runs, so a mesh's
    ``ComputeLocalToWorldTransform`` reports the AUTHORED pose, not the simulated one.  Mixing that
    with a live body pose silently produces nonsense.  The mesh-to-body transform, however, is rigid
    and authored, so it is safe to read from USD and then compose with the live body pose.
    """
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return None
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    time = Usd.TimeCode.Default()
    m_mesh = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
    m_body = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(time)
    to_body = m_mesh * m_body.GetInverse()
    return torch.tensor([tuple(to_body.Transform(p)) for p in points], dtype=torch.float32)


def main() -> None:
    env_cfg = cfg.WidowXPcbApproachEnvCfg() if args.approach else cfg.WidowXPcbSlideEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.physx.gpu_collision_stack_size = 2**26
    env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**20
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**17
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.settle):
        env.step(action)

    robot = env.scene["robot"]
    left_cfg, right_cfg, wrist_cfg, joint_cfg = (
        copy.deepcopy(c)
        for c in (cfg._LEFT_FINGER, cfg._RIGHT_FINGER, cfg._WRIST_BODY, cfg._GRIPPER_JOINT)
    )
    for c in (left_cfg, right_cfg, wrist_cfg, joint_cfg):
        c.resolve(env.scene)

    left_id = _resolve_first_body_id(robot, left_cfg)
    right_id = _resolve_first_body_id(robot, right_cfg)
    wrist_id = _resolve_first_body_id(robot, wrist_cfg)
    left_h = robot.data.body_pos_w[:, left_id]
    right_h = robot.data.body_pos_w[:, right_id]
    wrist_w = robot.data.body_pos_w[0, wrist_id]

    fwd = _gripper_tip_offset_direction_w(robot, left_h, right_h, wrist_cfg)[0]
    pad_mid = 0.5 * (left_h[0] + right_h[0])

    # The point the rewards actually score, for reference.
    tip_l, tip_r = gripper_jaw_pad_tips_world(
        env, left_cfg, right_cfg, joint_cfg, tip_offset_m=cfg._GRIPPER_TIP_OFFSET_M, wrist_body_cfg=wrist_cfg
    )

    fwd_np = fwd.tolist()
    pad_mid_np = pad_mid.tolist()

    def _proj(p: tuple[float, float, float]) -> float:
        return sum((p[i] - pad_mid_np[i]) * fwd_np[i] for i in range(3))

    print(f"\nconfigured _GRIPPER_TIP_OFFSET_M = {cfg._GRIPPER_TIP_OFFSET_M:.4f} m")
    print(f"wrist (link_6) world      = {[round(v, 4) for v in wrist_w.tolist()]}")
    print(f"pad midpoint world        = {[round(v, 4) for v in pad_mid_np]}")
    print(f"fwd (link_6 -> pad mid)   = {[round(v, 4) for v in fwd_np]}")
    print(f"|wrist -> pad midpoint|   = {float(torch.norm(pad_mid - wrist_w)):.4f} m")
    print(f"scored tip L/R world Z    = {float(tip_l[0, 2]):.4f} / {float(tip_r[0, 2]):.4f}\n")

    board_z = float(env.scene["pcb"].data.root_pos_w[0, 2])
    half_thick_mm = cfg.PCB_Z * 0.5 * 1000.0
    print(f"  board mid-thickness plane world Z = {board_z:.4f}  (half thickness {half_thick_mm:.2f} mm)")
    print("  distal extent along fwd (vs pad midpoint), and lowest real vertex (vs board plane):")
    print("  prim                                              verts    min       max     lowest dZ")
    overall_max = None
    lowest_dz = None
    fwd_t = fwd.detach().cpu()
    pad_mid_t = pad_mid.detach().cpu()
    for body_name in _JAW_HINTS:
        if body_name not in robot.body_names:
            continue
        body_idx = robot.body_names.index(body_name)
        body_pos = robot.data.body_pos_w[0, body_idx].detach().cpu()
        body_quat = robot.data.body_quat_w[0, body_idx].detach().cpu()
        body_prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body_name}")
        if not body_prim.IsValid():
            print(f"  [WARN] no prim for body '{body_name}'")
            continue
        for prim in Usd.PrimRange(body_prim):
            path = str(prim.GetPath())
            # Visuals and collisions are the same STL; measure collisions only (what physics uses).
            if "/collisions/" not in path:
                continue
            pts_body = _mesh_points_in_body_frame(prim, body_prim)
            if pts_body is None:
                continue
            pts_w = body_pos + quat_apply(body_quat.expand(pts_body.shape[0], 4), pts_body)
            projs = (pts_w - pad_mid_t) @ fwd_t
            lo, hi = float(projs.min()), float(projs.max())
            overall_max = hi if overall_max is None else max(overall_max, hi)
            dz = (float(pts_w[:, 2].min()) - board_z) * 1000.0
            lowest_dz = dz if lowest_dz is None else min(lowest_dz, dz)
            print(f"  {body_name:48s} {pts_body.shape[0]:6d}  {lo:+7.4f}  {hi:+7.4f}   {dz:+8.1f} mm")

    if overall_max is None:
        print("\n  [WARN] no jaw meshes found -- prim paths may not match _JAW_HINTS")
    else:
        print(
            f"\n  TRUE distal reach along fwd = {overall_max:+.4f} m "
            f"vs configured {cfg._GRIPPER_TIP_OFFSET_M:.4f} m "
            f"(error {(cfg._GRIPPER_TIP_OFFSET_M - overall_max) * 1000:+.1f} mm)"
        )
        print(
            f"  lowest real jaw vertex sits {lowest_dz:+.1f} mm from the board mid-thickness plane "
            f"-- negative means real geometry IS under the board"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
