from __future__ import annotations

"""Environment configuration for WidowX PCB insertion.

High-level flow:
1) Reset places PCB in a stable in-gripper grasp pose.
2) Policy moves PCB toward a designated slot on the magazine.
3) Rewards favor approaching/inserting the PCB into the slot target.
4) Terminations cut failed episodes early (drop/fall) and mark success.
"""

import math
import os
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from .mdp_custom import gripper_midpoint_position_env, object_gripper_midpoint_distance, pcb_grasp_deviation, object_height, \
    action_rate_l2, joint_vel_l2, pcb_to_target_slot_distance, pcb_vertical_gap_to_slot, \
    pcb_perpendicular_distance_to_insertion_axis, pcb_parallel_distance_along_insertion_axis, \
    pcb_insertion_wrong_side_penalty, pcb_height_below_reference, pcb_thickness_axis_tilt_penalty, \
    pcb_horizontal_offset_perpendicular_to_insertion, \
    target_slot_position, is_success_and_stable, reset_pcb_on_guide_rails, pcb_dropped_from_gripper, no_progress_termination

import torch
from isaaclab.envs import ManagerBasedRLEnv

# Conversion: mm to meters
PCB_X = 240.0 * 0.001
PCB_Y = 77.5 * 0.001
PCB_Z = 0.005
# IMPORTANT: set this to your real slot center in magazine local frame.
# Example: if Slot_01 is 20 mm forward and 8 mm up from magazine root:
# TARGET_SLOT_OFFSET = (0.02, 0.0, 0.008)
TARGET_SLOT_OFFSET = (0.0, 0.0, 0.0)
# PCB enters along the magazine LONG axis (local ±X). With _MAG_ROT_WXYZ = +90° about world Z,
# local +X → world +Y (slide toward +Y into the slot in the default layout).
# Use (-1, 0, 0) if the mouth is on the opposite end (e.g. “right‑to‑left” in your USD vs ours).
# Wrong-side and depth rewards use this; flip the sign if the policy hugs the wrong face.
INSERTION_AXIS_LOCAL = (1.0, 0.0, 0.0)
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

# Magazine root pose (shared with guide rail math below).
# Place on world X-axis (y=0) and rotate +90° about world Z.
_MAG_POS = (0.25, 0.35, 0.05)
_MAG_ROT_WXYZ = (0.7071068, 0.0, 0.0, 0.7071068)


def _quat_apply_wxyz(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate vector v by unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    cx = y * tz - z * ty
    cy = z * tx - x * tz
    cz = x * ty - y * tx
    return (vx + w * tx + cx, vy + w * ty + cy, vz + w * tz + cz)


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    L = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if L < 1e-9:
        return (0.0, 1.0, 0.0)
    return (v[0] / L, v[1] / L, v[2] / L)


def _add3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(s: float, v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (s * v[0], s * v[1], s * v[2])


# --- Kinematic guide rails: ABSOLUTE world placement ----------------------------
# Magazine long axis = world +Y (local X after +90° Z rotation).
# PCB slides in world +Y direction (INSERTION_AXIS_LOCAL = (1,0,0) in mag local).
# Guide rails flank the PCB along world X, supporting its left/right edges from below.
#
# Top view (XY plane, typical layout with magazine in +X/+Y):
#
#   world X →
#        rail-L ════════════════╡ magazine (slot mouth at +Y face)
#   +Y   [PCB on rails, slides in +Y into slot]
#        rail-R ════════════════╡
#        robot @ origin reaches into +X/+Y
#
# rails are unrotated in world frame (identity); size = (X-thin, Y-long, Z-height).
_GUIDE_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)
# Rail length (Y) is set to 0.28 m to span most of the PCB (240 mm) approach path.
_GUIDE_RAIL_SIZE_LOCAL = (0.03, 0.28, 0.01)   # thin in X, long in Y (insertion), height Z

# Half-channel in X: PCB width (PCB_Y = 77.5 mm) / 2 + 3 mm clearance gap + 3 mm half-wall.
_GUIDE_HALF_CHAN_X = PCB_Y * 0.5 + 0.003 + 0.003

# PCB on rails: long axis ∥ world +Y (insertion). Leading edge at slot mouth plane Y = MAG_POS[1].
# Center sits halfway back so the board is supported along the rail length.
_PCB_CENTER_Y = _MAG_POS[1] - (PCB_X + 0.03)
# Guide rail top (world Z) = rail center Z + half rail height. Cuboid PCB root is at its geometric
# center, so PCB bottom = center_z - PCB_Z/2 must equal rail top → center_z = rail_top + PCB_Z/2.
_GUIDE_RAIL_TOP_Z = _MAG_POS[2] + _GUIDE_RAIL_SIZE_LOCAL[2] * 0.5
_PCB_INIT_Z = _GUIDE_RAIL_TOP_Z + PCB_Z * 0.5
_PCB_INIT_POS = (_MAG_POS[0], _PCB_CENTER_Y, _PCB_INIT_Z)
# Initial orientation: cuboid local +X (long / PCB_X) ∥ magazine insertion in world (+Y with default
# mag yaw). Rotation +90° about world Z: body X → world Y. Thickness local +Z stays world +Z (flat on rails).
# If you set ``INSERTION_AXIS_LOCAL`` to (-1,0,0), use (0.7071068, 0.0, 0.0, -0.7071068) instead.
_PCB_INIT_ROT_WXYZ = (0.7071068, 0.0, 0.0, 0.7071068)
# wxai_follower: gripper_left/right link origins sit above the real jaw pads (near camera / carriage rail).
# Negative Z shifts spawn onto the pads; tune ± few cm if the board still floats or penetrates.
_PCB_GRASP_Z_WORLD_CORRECTION = -0.045
# Nominal PCB-center pose relative to jaw midpoint (world): +Y along board (trailing grasp), Z corrected.
_PCB_GRASP_EXPECTED_OFFSET_W = (
    0.0,
    PCB_X * 0.5,
    _PCB_GRASP_Z_WORLD_CORRECTION,
)
# Floor penalty reference: penalise if PCB drops below rail-top level (env-local Z ≈ rail top Z).
_MIN_PCB_HEIGHT_ENV = _GUIDE_RAIL_TOP_Z - 0.005   # 5 mm tolerance below rail top

# Rails share PCB center Y so left/right short edges rest on both rails (see your layout).
# Rail center Z = magazine root Z; tops at MAG_Z + half rail height.
_GUIDE_LEFT_POS = (_MAG_POS[0] - _GUIDE_HALF_CHAN_X, _PCB_CENTER_Y , _MAG_POS[2])
_GUIDE_RIGHT_POS = (_MAG_POS[0] + _GUIDE_HALF_CHAN_X, _PCB_CENTER_Y , _MAG_POS[2])

# Pre-insertion arm: reach toward the *leading* PCB edge at the slot mouth (+X, +Y from base).
# Goal EE orientation: tool axis ≈ world +Y (arm reaches HORIZONTALLY toward the magazine).
#   With a horizontal tool axis and joint_5 = 0 the gripper jaws open along world +Z —
#   i.e. they clamp the PCB's TOP and BOTTOM faces (across the 2 mm thickness axis). This is
#   exactly perpendicular to both the PCB face and the insertion direction.
#
# WidowX AI (wxai_follower) arm joint pre-insertion pose.
# joint_0: base yaw toward magazine center.
# joint_1 ≥ 0 (shoulder always positive on WidowX AI hardware).
# joints 1-3 net pitch ≈ 0 → tool axis horizontal toward magazine (+Y).
# Fine-tune in sim if link_6 does not align with slot mouth.
_ARM_JOINT0 = math.atan2(_MAG_POS[1], _MAG_POS[0])  # base yaw toward magazine
_ARM_JOINT1 = 0.95   # shoulder pitch (positive only)
_ARM_JOINT2 = 1.15   # elbow
_ARM_JOINT3 = -0.35  # wrist pitch
_ARM_JOINT4 = 0.00   # wrist yaw
_ARM_JOINT5 = 0.00   # wrist roll: 0 = jaws open along world +Z (grips PCB top/bottom face)

@configclass
class WidowXPcbSceneCfg(InteractiveSceneCfg):
    """Scene assets for the WidowX PCB insertion task.

    Contains:
    - static world assets (ground, light),
    - robot articulation,
    - grasped object (PCB),
    -     insertion target object (magazine),
    optional kinematic guide rails for narrow-slot insertion.
    """
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    light = AssetBaseCfg(prim_path="/World/defaultLight", spawn=sim_utils.DistantLightCfg(intensity=3000.0))
    
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # Trossen AI WidowX Follower USD (stiffness/damping already baked in).
            usd_path=os.path.join(ASSET_DIR, "trossen_ai_isaac", "assets", "robots", "wxai", "wxai_follower.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # WidowX AI joints: joint_0 (Z yaw), joint_1-3 (Y pitch), joint_4 (Z yaw), joint_5 (X roll)
                # Pre-insertion: reach toward magazine at _MAG_POS with arm horizontal.
                "joint_0": _ARM_JOINT0,  # base yaw toward magazine
                "joint_1": _ARM_JOINT1,  # shoulder pitch
                "joint_2": _ARM_JOINT2,  # elbow pitch
                "joint_3": _ARM_JOINT3,  # wrist pitch
                "joint_4": _ARM_JOINT4,  # wrist yaw
                "joint_5": _ARM_JOINT5,  # wrist roll
                # Gripper closed to hold PCB. left_carriage_joint range: 0.0 (closed) → 0.044 (open)
                "left_carriage_joint": 0.0,
            },
        ),
        # wxai_follower.usd has PD gains baked in — use None to inherit from USD.
        actuators={
            "wxai_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[0-5]"],
                stiffness=None,
                damping=None,
            ),
            # right_carriage_joint is a mimic joint in the USD (driven by left_carriage_joint).
            "wxai_gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_carriage_joint"],
                stiffness=None,
                damping=None,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    
    pcb = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PCB",
        spawn=sim_utils.CuboidCfg(
            size=(PCB_X, PCB_Y, PCB_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=12,
                max_depenetration_velocity=0.5,
                max_linear_velocity=25.0,
                max_angular_velocity=720.0,
                linear_damping=0.25,
                angular_damping=0.35,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                # Slightly thicker contact band reduces tunneling vs magazine meshes without huge ghost gaps.
                contact_offset=0.003,
                rest_offset=0.0005,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=2.0,
                dynamic_friction=1.6,
                restitution=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.1)),
        ),
        # Default pose: flat on kinematic guide rails (same pose used by ``reset_pcb_on_guide_rails``).
        init_state=RigidObjectCfg.InitialStateCfg(pos=_PCB_INIT_POS, rot=_PCB_INIT_ROT_WXYZ),
    )


    magazine = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Magazine",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_DIR, "usd_model", "magazine.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=12,
                max_depenetration_velocity=0.5,
            ),
            # Apply consistent collision offsets under magazine meshes (helps PCB vs slot penetration).
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0005,
            ),
        ), 
        # Magazine pose: centered on world X-axis (y=0), then yaw +90° about world Z.
        # Guide rails use the same `_MAG_ROT_WXYZ` and are kept in front of the slot mouth.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_MAG_POS,
            rot=_MAG_ROT_WXYZ,
        ),
    )

    # Guide rails: top/bottom lips in front of magazine mouth (not side walls).
    guide_rail_left = RigidObjectCfg(  # top front rail
        prim_path="{ENV_REGEX_NS}/GuideRailLeft",
        spawn=sim_utils.CuboidCfg(
            size=_GUIDE_RAIL_SIZE_LOCAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0003),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35,
                dynamic_friction=0.30,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.48), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_GUIDE_LEFT_POS, rot=_GUIDE_ROT_WXYZ),
    )
    guide_rail_right = RigidObjectCfg(  # bottom front rail
        prim_path="{ENV_REGEX_NS}/GuideRailRight",
        spawn=sim_utils.CuboidCfg(
            size=_GUIDE_RAIL_SIZE_LOCAL,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0003),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35,
                dynamic_friction=0.30,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.48), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_GUIDE_RIGHT_POS, rot=_GUIDE_ROT_WXYZ),
    )

    # # 매거진 내부에 생성한 특정 슬롯(예: Slot_01)의 위치를 추적할 프림 경로 추가
    # target_slot = SceneEntityCfg("magazine", body_names="Slot_01")


@configclass
class ActionsCfg:
    """Policy action space configuration."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[0-5]"],
        scale=0.1,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups exposed to the policy."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Per-step policy observations.

        Includes robot state, PCB position, slot target position, and EE position.
        """
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        
        # ✅ 물체 위치는 Isaac Lab 내장 함수인 root_pos_w를 사용합니다.
        object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("pcb")})
        target_pos = ObservationTermCfg(
            func=target_slot_position,
            params={"target_cfg": SceneEntityCfg("magazine"), "slot_offset": TARGET_SLOT_OFFSET},
        )
        
        # EE: midpoint of WidowX AI jaw tips (gripper_left/right) — actual contact surface.
        ee_pos = ObservationTermCfg(
            func=gripper_midpoint_position_env,
            params={
                "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
                "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            },
        )
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward shaping terms used during training.

    Mixes dense shaping (distance/depth) and sparse success bonuses.

    Reward convention (Isaac Lab): total reward is sum of ``weight * term_value``.
    For distance-like terms, use **negative** weights so smaller distance -> higher reward.

    Insertion is decomposed along ``INSERTION_AXIS_LOCAL`` so the policy cannot minimize plain
    3D distance by orbiting the magazine side face or dragging on the floor.
    """
    # In-hand grasp shaping (disabled while PCB starts on rails). Re-enable with negative weight if you
    # switch back to ``reset_pcb_in_gripper`` and ``expected_offset_world`` / in-hand termination.
    grasp_deviation = RewardTermCfg(
        func=pcb_grasp_deviation,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
            "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            "expected_offset_world": _PCB_GRASP_EXPECTED_OFFSET_W,
        },
        weight=0.0,
    )
    # Small bonus for height — large values encourage hovering instead of inserting.
    lifting_object = RewardTermCfg(func=object_height, params={"asset_cfg": SceneEntityCfg("pcb")}, weight=0.25)

    action_rate_penalty = RewardTermCfg(func=action_rate_l2, weight=-0.0)
    joint_vel_penalty = RewardTermCfg(
        func=joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=0.0,
    )

    # Weak 3D distance — alone it rewards sliding along the magazine shell; pair with axis terms below.
    inserting_pcb = RewardTermCfg(
        func=pcb_to_target_slot_distance,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
        },
        weight=-4.0,
    )

    # Match slot height (world Z gap vs target).
    slot_height_alignment = RewardTermCfg(
        func=pcb_vertical_gap_to_slot,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
        },
        weight=-22.0,
    )

    # Horizontal lateral error only *perpendicular* to insertion — does not fight sliding along the slot axis
    # (unlike raw XY distance, which penalizes progress along world Y when insertion is +Y).
    slot_horizontal_lateral = RewardTermCfg(
        func=pcb_horizontal_offset_perpendicular_to_insertion,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "insertion_axis_local": INSERTION_AXIS_LOCAL,
        },
        weight=-20.0,
    )

    # Full 3D lateral error perpendicular to insertion (includes Z component off the rail plane).
    slot_insertion_lateral = RewardTermCfg(
        func=pcb_perpendicular_distance_to_insertion_axis,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "insertion_axis_local": INSERTION_AXIS_LOCAL,
        },
        weight=-24.0,
    )

    # Depth along insertion axis toward slot center.
    slot_insertion_depth = RewardTermCfg(
        func=pcb_parallel_distance_along_insertion_axis,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "insertion_axis_local": INSERTION_AXIS_LOCAL,
        },
        weight=-28.0,
    )

    # Penalize being on the wrong side of the slot along the insert ray (orbit / wrong face cheat).
    insertion_wrong_side = RewardTermCfg(
        func=pcb_insertion_wrong_side_penalty,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "insertion_axis_local": INSERTION_AXIS_LOCAL,
        },
        weight=-35.0,
    )

    # Penalize dragging the PCB on the floor / magazine base (low env-local Z).
    above_floor = RewardTermCfg(
        func=pcb_height_below_reference,
        params={"pcb_cfg": SceneEntityCfg("pcb"), "min_height": _MIN_PCB_HEIGHT_ENV},
        weight=-18.0,
    )

    # Keep board flat (local +Z ∥ world up); reduces edge-on pushing against the magazine wall.
    pcb_flat = RewardTermCfg(
        func=pcb_thickness_axis_tilt_penalty,
        params={"pcb_cfg": SceneEntityCfg("pcb")},
        weight=-12.0,
    )

    success_bonus = RewardTermCfg(
        func=is_success_and_stable,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "dist_threshold": 0.015,
            "vel_threshold": 0.05,
        },
        weight=1000.0,
    )

    # Duplicate of inserting_pcb removed; single 3D distance term avoids conflicting gradients.

    insertion_hold_reward = RewardTermCfg(
        func=is_success_and_stable,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "dist_threshold": 0.015,
            "vel_threshold": 0.05,
        },
        weight=10.0,
    )
    

@configclass
class EventCfg:
    # Place PCB on guide rails (matches ``pcb`` init_state); avoids floating/in-gripper teleport.
    reset_pcb_on_rails = EventTermCfg(
        func=reset_pcb_on_guide_rails,
        mode="reset",
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "pos_env_local": _PCB_INIT_POS,
            "rot_wxyz": _PCB_INIT_ROT_WXYZ,
            "velocity_scale": 0.0,
        },
    )
    # Keep gripper command at default after rail pose (optional; tune open vs closed for pick-from-rail).
    reset_gripper_closed = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_carriage_joint"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions.

    Purpose:
    - end failed episodes early (drop/fall/no-progress),
    - detect successful insertion stabilization.
    """
    pass
    # ⚠️ 수정: time_limit 삭제, time_out=True (타임아웃 플래그) 추가
    # time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    object_falling = TerminationTermCfg(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("pcb")},
    )
    pcb_dropped = TerminationTermCfg(
        func=pcb_dropped_from_gripper,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
            "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            "check_grasp_geometry": False,
            "min_height": 0.02,
        },
    )
    no_progress = TerminationTermCfg(
        func=no_progress_termination,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "ee_cfg": SceneEntityCfg("robot", body_names="link_6"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "min_target_distance": 0.08,
            "ee_speed_threshold": 0.01,
            "pcb_speed_threshold": 0.01,
            # Grace period: do not fire at t=0 (arm + PCB are stationary right after reset).
            # 60 policy steps × decimation(4) × dt(0.002) = 0.48 s warmup.
            "min_episode_steps": 60,
        },
    )

    # 성공 조건: 거리가 1cm 미만이고, 속도가 거의 0일 때
    # (커스텀 함수 success_check가 정의되어 있다는 가정)
    success = TerminationTermCfg(
        func=is_success_and_stable, 
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "target_cfg": SceneEntityCfg("magazine"),
            "slot_offset": TARGET_SLOT_OFFSET,
            "dist_threshold": 0.01,
            "vel_threshold": 0.05, # 거의 멈춘 상태
        }
    )


@configclass
class WidowXPcbEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level RL environment configuration for this task."""
    scene: WidowXPcbSceneCfg = WidowXPcbSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
    )

    # ✅ 수정 3: EnvCfg에 누락되어 있던 관측(observations)과 보상(rewards)을 연결
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    actions: ActionsCfg = ActionsCfg()   
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Simulation Settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.002,  # 500Hz로 상향 (터널링 방지)
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            # Thin PCB vs magazine: reduce tunneling / visible penetration.
            enable_ccd=True,
            # Resolve rigid contacts after articulation constraints (better grasp + PCB wedged in slot).
            solve_articulation_contact_last=True,
            # 5.1 버전에서 에러가 난다면 이 아래의 정밀도 설정은 일단 생략해도
            # dt=0.002 만으로도 충분히 강력합니다.
            # Headroom for 2048 parallel envs (many articulated + rigid contacts).
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**19,
        ),
    )

    def __post_init__(self):
        """Finalize runtime settings after dataclass initialization."""
        # Frame env 0: robot near origin, magazine ~ (_MAG_POS). Helps debugging when assets stay local.
        self.viewer.eye = (0.95, 0.95, 0.65)
        self.viewer.lookat = (0.25, 0.35, 0.08)
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 15.0