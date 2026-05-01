from __future__ import annotations

"""Environment configuration for WidowX PCB slot insertion.

1) Reset: PCB starts **already grasped** on the short edge (snap + gripper close on thickness).
2) Rewards: align long axis with ``PUSH_AXIS_WORLD``, drive the **leading** board corner toward a fixed
   **slot mouth** pose in env-local coordinates, forward velocity along the insertion axis, and mild
   lateral-velocity regularization (no guide-rail corridor terms).
3) Terminations: timeout, arm idle, PCB tilt / height / drop heuristics.
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
from .mdp_custom import (
    gripper_midpoint_position_env,
    gripper_opening_normalized,
    pcb_forward_velocity_along_world_axis,
    pcb_long_axis_parallel_to_push_reward,
    pcb_leading_edge_insertion_proximity_reward,
    pcb_horizontal_velocity_perpendicular_to_axis_penalty,
    action_rate_l2,
    pcb_height_below_reference,
    pcb_thickness_axis_tilt_penalty,
    reset_pcb_on_guide_rails,
    reset_robot_joints_to_values,
    snap_pcb_root_to_short_edge_grasp,
    pcb_dropped_from_gripper,
    pcb_root_height_below_env_minimum,
    pcb_tilt_beyond_limit,
    pcb_long_axis_vertical_component_exceeds,
    gripper_mid_thickness_offset_obs,
    gripper_pinch_orientation_cos_obs,
    arm_joints_velocity_idle_termination,
)

# Conversion: mm to meters
PCB_X = 240.0 * 0.001
PCB_Y = 77.5 * 0.001
PCB_Z = 0.003
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
# target =  (x = 0.243 , y = 0.7, z = 0.041 )
# Magazine root pose (shared with guide rail math below).
# Place on world X-axis (y=0) and rotate +90° about world Z.

_MAG_POS = (0.20, 0.6, 0.10)
_MAG_ROT_WXYZ = (0.7071068, 0.0, 0.0, 0.7071068)

# World-frame unit direction for “push forward” velocity reward. Default: +Y (matches PCB long axis
# when ``_PCB_INIT_ROT_WXYZ`` maps body +X → world +Y). Flip sign if your rail runs the other way.
PUSH_AXIS_WORLD = (0.0, 1.0, 0.0)

# --- PCB-on-rail placement (matches integrated magazine + rail USD when offsets are tuned) -----
# With _MAG_ROT_WXYZ = +90° about world Z, magazine local +X maps to world +Y (typical slide direction).
# Half-channel and rail tops position the cuboid PCB flat on the rails.
#
# Top view (XY plane, typical layout):
#
#   world X →
#        rail-L ════════════════╡ magazine / rail assembly
#   +Y   [PCB on rails]
#        rail-R ════════════════╡
#        robot @ origin reaches into +X/+Y
#
# Rail length (Y) is set to 0.28 m to span most of the PCB (240 mm) approach path.
_GUIDE_RAIL_SIZE_LOCAL = (0.03, 0.28, 0.01)   # thin in X, long in Y (insertion), height Z

# PCB on rails: long axis ∥ world +Y. Center Y is derived from magazine pose and board length.
_PCB_CENTER_Y = _MAG_POS[1] - (PCB_X + 0.03)
# Guide rail top (world Z) = rail center Z + half rail height. Cuboid PCB root is at its geometric
# center, so PCB bottom = center_z - PCB_Z/2 must equal rail top → center_z = rail_top + PCB_Z/2.
_GUIDE_RAIL_TOP_Z = _MAG_POS[2] + _GUIDE_RAIL_SIZE_LOCAL[2] * 0.5
# ``magazine.usd`` rail collision often sits **above** this analytic plane; without bias the PCB can
# spawn interpenetrating the mesh (visual overlap). Tune ± after inspecting contacts in Isaac Sim.
_PCB_SPAWN_Z_BIAS = 0.002
_EFFECTIVE_RAIL_TOP_Z = _GUIDE_RAIL_TOP_Z + _PCB_SPAWN_Z_BIAS
_PCB_INIT_Z = _EFFECTIVE_RAIL_TOP_Z + PCB_Z * 0.5
_PCB_INIT_POS = (_MAG_POS[0], _PCB_CENTER_Y, _PCB_INIT_Z)
# Initial orientation: cuboid local +X (long / PCB_X) ∥ world +Y. Rotation +90° about world Z:
# body X → world Y. Thickness local +Z stays world +Z (flat on rails).
_PCB_INIT_ROT_WXYZ = (0.7071068, 0.0, 0.0, 0.7071068)
# Floor penalty / fall detection use the same effective rail plane as spawn (see ``_PCB_SPAWN_Z_BIAS``).
_MIN_PCB_HEIGHT_ENV = _EFFECTIVE_RAIL_TOP_Z - 0.005   # 5 mm tolerance below rail top
_PCB_TERMINATE_MIN_HEIGHT_ENV = _EFFECTIVE_RAIL_TOP_Z - 0.025
# Env-local pose for the **leading** point of the board (root + half length along body +X) at the
# **slot mouth** (user-measured in Isaac Sim).
_SLOT_MOUTH_LEAD_TARGET_XYZ_ENV = (0.243, _MAG_POS[1], 0.041)
_INSERT_DISTANCE_SIGMA_M = 0.12
# Arm appears "dead": no meaningful joint motion for this many env steps (after ``decimation``).
_ARM_IDLE_MAX_ABS_VEL_RAD_S = 0.03
_ARM_IDLE_MIN_STEPS = 100

# --- Episode start: short-edge pinch (push-face center = jaw mid after snap; tune arm in sim) ---
_PUSH_FACE_CENTER_XY = (_PCB_INIT_POS[0], _PCB_INIT_POS[1] - PCB_X * 0.5)
_ARM_JOINT0_EDGE = math.atan2(_PUSH_FACE_CENTER_XY[1], _PUSH_FACE_CENTER_XY[0])
_ARM_JOINT0_EDGE = 0.0
_ARM_JOINT1_EDGE = 1.5
_ARM_JOINT2_EDGE = 0.0
_ARM_JOINT3_EDGE = 1.5
_ARM_JOINT4_EDGE = -1.5
_ARM_JOINT5_EDGE = 1.5

# left_carriage_joint: larger = more open. Pre-snap must clear PCB **thickness** with margin so pads
# straddle the board; too-small “closed” collapses fingers and wedges the PCB on the housing.
_GRIPPER_OPENING_PRE_SNAP = max(PCB_Z * 2.5, 0.003)
# Stay slightly **wider** than board thickness so physics keeps the board between pads (not metal-metal shut).
_GRIPPER_OPENING_CLOSED_EDGE = max(PCB_Z * 1.45, 0.003)
# Fine-tune snap if gripper_left/right frames sit on carriages: body +X long, +Y short, +Z thickness.
_SNAP_PCB_CENTER_OFFSET_BODY_M = (0.030, 0.0, 0.0)
# Floor for PCB root Z (env-local): stay at/above nominal on-rail center height if the arm FK is low.
_SNAP_PCB_MIN_CENTER_Z_ENV_LOCAL = _PCB_INIT_POS[2]

@configclass
class WidowXPcbSceneCfg(InteractiveSceneCfg):
    """Scene assets for the WidowX PCB task.

    Contains ground, light, robot, PCB cuboid, and the static magazine + rail USD (single asset).
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
                # Matches reset events (post-snap gripper = ``_GRIPPER_OPENING_CLOSED_EDGE``).
                "joint_0": _ARM_JOINT0_EDGE,
                "joint_1": _ARM_JOINT1_EDGE,
                "joint_2": _ARM_JOINT2_EDGE,
                "joint_3": _ARM_JOINT3_EDGE,
                "joint_4": _ARM_JOINT4_EDGE,
                "joint_5": _ARM_JOINT5_EDGE,
                "left_carriage_joint": _GRIPPER_OPENING_CLOSED_EDGE,
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
                # Larger contact band + small rest separation vs kinematic meshes reduces tunneling and
                # visible z-fighting / overlap with ``magazine.usd`` rails.
                contact_offset=0.004,
                rest_offset=0.0012,
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


    magazine = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Magazine",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_DIR, "usd_model", "magazine.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=12,
                max_depenetration_velocity=0.5,
            ),
            # Match PCB offsets so contacts resolve with less interpenetration vs thin PCBs.
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.0012,
            ),
        ), 
        # Fixture pose: adjust `_MAG_POS` / `_MAG_ROT_WXYZ` to match the imported USD in world frame.
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_MAG_POS,
            rot=_MAG_ROT_WXYZ,
        ),
    )

@configclass
class ActionsCfg:
    """Policy action space: arm + parallel gripper (so the policy can close on the PCB edge)."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[0-5]"],
        scale=1.0,
        use_default_offset=True,
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_carriage_joint"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups exposed to the policy."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Per-step policy observations: robot joints, PCB pose, EE (grip midpoint)."""
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)

        object_pos = ObservationTermCfg(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("pcb")})

        # EE: midpoint of WidowX AI jaw tips (gripper_left/right) — actual contact surface.
        ee_pos = ObservationTermCfg(
            func=gripper_midpoint_position_env,
            params={
                "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
                "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            },
        )
        # 0 = closed, 1 = open — explicit signal to learn approach-then-close-then-push.
        gripper_opening = ObservationTermCfg(
            func=gripper_opening_normalized,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_carriage_joint"])},
        )
        # Jaw mid vs PCB center along thickness (pinch alignment); helps learn close from top/bottom.
        ee_thickness_offset = ObservationTermCfg(
            func=gripper_mid_thickness_offset_obs,
            params={
                "pcb_cfg": SceneEntityCfg("pcb"),
                "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
                "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
                "scale_m": 0.012,
            },
        )
        # |cos|: jaw opening axis ∥ PCB thickness; finger line ∥ PCB short edge (pinch 단변).
        pinch_orientation_cos = ObservationTermCfg(
            func=gripper_pinch_orientation_cos_obs,
            params={
                "pcb_cfg": SceneEntityCfg("pcb"),
                "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
                "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
                "min_finger_sep_m": 0.006,
            },
        )
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Insertion: long-axis ∥ ``PUSH_AXIS_WORLD``, 3D proximity of leading point to slot mouth, push velocity."""

    # Long edge ∥ ``PUSH_AXIS_WORLD`` (insertion / yaw alignment).
    insert_axis_parallel_long_axis = RewardTermCfg(
        func=pcb_long_axis_parallel_to_push_reward,
        params={"pcb_cfg": SceneEntityCfg("pcb"), "axis_world": PUSH_AXIS_WORLD},
        weight=6.0,
    )

    # Leading point → ``_SLOT_MOUTH_LEAD_TARGET_XYZ_ENV`` (env-local).
    insertion_proximity = RewardTermCfg(
        func=pcb_leading_edge_insertion_proximity_reward,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "half_length_m": PCB_X * 0.5,
            "target_lead_xyz_env": _SLOT_MOUTH_LEAD_TARGET_XYZ_ENV,
            "sigma_m": _INSERT_DISTANCE_SIGMA_M,
        },
        weight=12.0,
    )

    # Forward-only velocity along push axis (relu).
    push_velocity = RewardTermCfg(
        func=pcb_forward_velocity_along_world_axis,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "axis_world": PUSH_AXIS_WORLD,
        },
        weight=14.0,
    )

    # Penalize horizontal velocity orthogonal to the insertion axis.
    lateral_slide_penalty = RewardTermCfg(
        func=pcb_horizontal_velocity_perpendicular_to_axis_penalty,
        params={"pcb_cfg": SceneEntityCfg("pcb"), "axis_world": PUSH_AXIS_WORLD},
        weight=-2.5,
    )

    action_rate_penalty = RewardTermCfg(func=action_rate_l2, weight=-0.0025)

    above_floor = RewardTermCfg(
        func=pcb_height_below_reference,
        params={"pcb_cfg": SceneEntityCfg("pcb"), "min_height": _MIN_PCB_HEIGHT_ENV},
        weight=-18.0,
    )

    pcb_flat = RewardTermCfg(
        func=pcb_thickness_axis_tilt_penalty,
        params={"pcb_cfg": SceneEntityCfg("pcb")},
        weight=-12.0,
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
    # Arm to short-edge grasp pose with clearance so the PCB snap does not huge-interpenetrate.
    reset_robot_short_edge_grasp_loose = EventTermCfg(
        func=reset_robot_joints_to_values,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_positions": {
                "joint_0": _ARM_JOINT0_EDGE,
                "joint_1": _ARM_JOINT1_EDGE,
                "joint_2": _ARM_JOINT2_EDGE,
                "joint_3": _ARM_JOINT3_EDGE,
                "joint_4": _ARM_JOINT4_EDGE,
                "joint_5": _ARM_JOINT5_EDGE,
                "left_carriage_joint": _GRIPPER_OPENING_PRE_SNAP,
            },
            "velocity_scale": 0.0,
            "use_current_joint_pos": False,
        },
    )
    # Kinematic contact: push-face center = jaw midpoint (magazine pose unchanged; PCB may shift slightly in XY).
    snap_pcb_to_short_edge_grasp = EventTermCfg(
        func=snap_pcb_root_to_short_edge_grasp,
        mode="reset",
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
            "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            "half_length_m": PCB_X * 0.5,
            "rot_wxyz": _PCB_INIT_ROT_WXYZ,
            "velocity_scale": 0.0,
            "center_offset_body_m": _SNAP_PCB_CENTER_OFFSET_BODY_M,
            "min_center_z_env_local": _SNAP_PCB_MIN_CENTER_Z_ENV_LOCAL,
        },
    )
    # Re-open on the current arm pose so pads straddle the board before the final squeeze (avoids side wedge).
    reset_gripper_presnap_after_pcb = EventTermCfg(
        func=reset_robot_joints_to_values,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_positions": {"left_carriage_joint": _GRIPPER_OPENING_PRE_SNAP},
            "velocity_scale": 0.0,
            "use_current_joint_pos": True,
        },
    )
    # Light squeeze: stay wider than metal-metal shut so the cuboid stays between finger pads.
    reset_gripper_closed_on_edge = EventTermCfg(
        func=reset_robot_joints_to_values,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_positions": {"left_carriage_joint": _GRIPPER_OPENING_CLOSED_EDGE},
            "velocity_scale": 0.0,
            "use_current_joint_pos": True,
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination: timeout, arm idle, tilt / axis skew, height, drop."""

    # Marks truncated episodes when ``episode_length_s`` is reached (pairs with ``mdp.time_out``).
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # End stale episodes when joint_[0-5] stay below velocity threshold (gripper excluded).
    arm_idle = TerminationTermCfg(
        func=arm_joints_velocity_idle_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[0-5]"]),
            "max_abs_vel_rad_s": _ARM_IDLE_MAX_ABS_VEL_RAD_S,
            "min_idle_steps": _ARM_IDLE_MIN_STEPS,
        },
    )

    # Thickness axis vs world up — stricter than 0.2 so ~25–30° failures still cut (was missing ~30°).
    pcb_tilt_excessive = TerminationTermCfg(
        func=pcb_tilt_beyond_limit,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "max_tilt_penalty": 0.13,
        },
    )

    # Long axis should stay horizontal (XY); wedge / slip often rotates body +X out of the plane first.
    pcb_long_axis_not_horizontal = TerminationTermCfg(
        func=pcb_long_axis_vertical_component_exceeds,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "max_abs_z": 0.16,
        },
    )

    # Same height convention as ``above_floor`` reward — not world ``z < -0.5`` (never fired on a table).
    pcb_fallen_below_rail = TerminationTermCfg(
        func=pcb_root_height_below_env_minimum,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "min_height_env": _PCB_TERMINATE_MIN_HEIGHT_ENV,
        },
    )
    pcb_dropped = TerminationTermCfg(
        func=pcb_dropped_from_gripper,
        params={
            "pcb_cfg": SceneEntityCfg("pcb"),
            "left_finger_cfg": SceneEntityCfg("robot", body_names="gripper_left"),
            "right_finger_cfg": SceneEntityCfg("robot", body_names="gripper_right"),
            "check_grasp_geometry": True,
            "expected_center_distance": PCB_X * 0.5,
            "distance_tolerance": 0.06,
            # Slightly above floor (env-local); pairs with ``pcb_fallen_below_rail`` for clear failures.
            "min_height": 0.025,
        },
    )


@configclass
class WidowXPcbEnvCfg(ManagerBasedRLEnvCfg):
    """Top-level RL environment configuration for this task."""
    scene: WidowXPcbSceneCfg = WidowXPcbSceneCfg(
        num_envs=2048,
        env_spacing=2.5,
    )

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
            # Thin PCB vs fixture: reduce tunneling / visible penetration.
            enable_ccd=True,
            # Resolve rigid contacts after articulation constraints.
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