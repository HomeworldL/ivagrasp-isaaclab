"""Base manager-based environment config for floating-hand dexgrasp_float."""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass
import logging
from pathlib import Path
from typing import Mapping

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import CollisionPropertiesCfg, MassPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from dynamic_dexgrasp_lab.tasks.dexgrasp_float import mdp
from dynamic_dexgrasp_lab.assets.data.dataset_loader import (
    ObjectSelection,
    ObjectSetQueryCfg,
    resolve_object_selection,
)
from dynamic_dexgrasp_lab.assets.data.object_assignment import build_env_usd_path_list
from dynamic_dexgrasp_lab.model.backbone.bps import default_bps_basis_path

LOGGER = logging.getLogger(__name__)

DEFAULT_GOAL_SUCCESS_POS_THRESHOLD = 0.06
DEFAULT_GOAL_SUCCESS_ROT_THRESHOLD = math.radians(20.0)
DEFAULT_GOAL_SUCCESS_MIN_CONTACT_FINGERS = 4
DEFAULT_OBJECT_RESET_DISTANCE = 0.46
DEFAULT_OBJECT_GOAL_RESET_DISTANCE = 0.46
DEFAULT_GEOMETRY_NUM_POINTS = 512
DEFAULT_GEOMETRY_POINT_CLOUD_SOURCE = "dataset_or_usd"
DEFAULT_STATIC_FRICTION = 0.6
DEFAULT_DYNAMIC_FRICTION = 0.6
DEFAULT_DEXCUBE_OBJECT_SET_ID = "default:dexcube"
DEFAULT_DEXCUBE_OBJECT_SET_IDS = {
    "default:dexcube", "builtin:dexcube", "nucleus:dexcube",
    "default:blue_block", "default:red_block", "default:green_block",
    "default:sphere_4cm", "default:sphere_5cm",
    "default:cylinder_4cm", "default:cylinder_5cm",
    "default:cone_4cm", "default:cone_5cm",
    "default:capsule_4cm", "default:capsule_5cm",
    "default:cuboid_5cm", "default:cuboid_4x6x8cm",
}


@dataclass
class DexGraspHandProfile:
    """Per-hand runtime description used to assemble the default dexgrasp environment."""

    hand_name: str
    hand_cfg: ArticulationCfg
    hand_joint_names: tuple[str, ...]
    hand_actuated_joint_names: tuple[str, ...]
    hand_home_joint_positions: tuple[float, ...]
    hand_home_joint_pos_map: Mapping[str, float]
    hand_position_action_offset_map: Mapping[str, float]
    hand_position_action_scale_map: Mapping[str, float]
    hand_transform_pos: tuple[float, float, float]
    hand_transform_quat: tuple[float, float, float, float]
    hand_object_relative_pos: tuple[float, float, float]
    hand_object_relative_quat: tuple[float, float, float, float]
    hand_camera_object_relative_pos: tuple[float, float, float]
    hand_camera_object_relative_quat: tuple[float, float, float, float]
    hand_palm_body_name: str
    hand_geo_body_names: tuple[str, ...]
    hand_geo_body_weights: tuple[float, ...]
    hand_articulation_root_prim_path: str
    hand_contact_body_names: tuple[str, ...]
    hand_contact_finger_body_names: tuple[tuple[str, ...], ...]
    hand_contact_fingertip_body_names: tuple[str, ...]
    hand_drive_stiffness: Mapping[str, float] | None
    hand_drive_damping: Mapping[str, float] | None
    hand_drive_max_force: Mapping[str, float] | None
    hand_contact_sensor_prim_path_prefix: str

def _make_default_shape_collision_props() -> CollisionPropertiesCfg:
    return CollisionPropertiesCfg(collision_enabled=True)


def _make_default_shape_mass_props() -> MassPropertiesCfg:
    # Use explicit density for programmatic primitives so PhysX can compute stable mass/inertia.
    return MassPropertiesCfg(density=500.0)


def _make_default_shape_physics_material() -> RigidBodyMaterialCfg:
    return RigidBodyMaterialCfg(
        static_friction=DEFAULT_STATIC_FRICTION,
        dynamic_friction=DEFAULT_DYNAMIC_FRICTION,
        restitution=0.0,
    )


_DEFAULT_NUCLEUS_BLOCK_SPAWN_MAP = {
    "default:dexcube": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.9, 0.9, 0.9),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "builtin:dexcube": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.9, 0.9, 0.9),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "nucleus:dexcube": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.9, 0.9, 0.9),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "default:blue_block": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "default:red_block": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "default:green_block": lambda: UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    ),
    "default:sphere_4cm": lambda: SphereCfg(
        radius=0.04,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:sphere_5cm": lambda: SphereCfg(
        radius=0.05,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cylinder_4cm": lambda: CylinderCfg(
        radius=0.04, height=0.08,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cylinder_5cm": lambda: CylinderCfg(
        radius=0.05, height=0.10,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cone_4cm": lambda: ConeCfg(
        radius=0.04, height=0.08,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cone_5cm": lambda: ConeCfg(
        radius=0.05, height=0.10,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:capsule_4cm": lambda: CapsuleCfg(
        radius=0.04, height=0.04,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:capsule_5cm": lambda: CapsuleCfg(
        radius=0.05, height=0.05,
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cuboid_5cm": lambda: CuboidCfg(
        size=(0.05, 0.05, 0.05),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
    "default:cuboid_4x6x8cm": lambda: CuboidCfg(
        size=(0.04, 0.06, 0.08),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
        collision_props=_make_default_shape_collision_props(),
        mass_props=_make_default_shape_mass_props(),
        physics_material=_make_default_shape_physics_material(),
    ),
}


def _make_default_object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=10.0,
        disable_gravity=True,
        enable_gyroscopic_forces=True,
    )


def _make_object_spawn_cfg(usd_path: str) -> UsdFileCfg:
    return UsdFileCfg(
        usd_path=usd_path,
        scale=(1.0, 1.0, 1.0),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    )


def _make_default_dexcube_spawn_cfg() -> UsdFileCfg:
    return UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.9, 0.9, 0.9),
        activate_contact_sensors=True,
        rigid_props=_make_default_object_rigid_props(),
    )


@configclass
class DexGraspFloatSceneCfg(InteractiveSceneCfg):
    """Floating-hand scene that can swap object USDs from RL split metadata."""

    robot: ArticulationCfg = MISSING
    visual_ground = AssetBaseCfg(
        prim_path="/World/visualGround",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.22, 0.22, 0.24),
                roughness=0.95,
                metallic=0.0,
            ),
            collision_props=None,
            rigid_props=None,
            mass_props=None,
        ),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spawn=_make_default_dexcube_spawn_cfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for dexgrasp_float."""

    goal = mdp.GoalCommandCfg(
        object_asset_name="object",
        robot_asset_name="robot",
        command_name="goal",
        resampling_time_range=(float(1.0e9), float(1.0e9)),
        debug_vis=False,
        success_pos_threshold=DEFAULT_GOAL_SUCCESS_POS_THRESHOLD,
        success_rot_threshold=DEFAULT_GOAL_SUCCESS_ROT_THRESHOLD,
    )


@configclass
class ActionsCfg:
    """Action terms for dexgrasp_float."""

    # Hand-specific policy/action semantics are injected later from the active hand profile.
    floating_root = mdp.RootTwistPBVSWrenchSim2SimActionCfg(
        asset_name="robot",
    )
    hand_joints = mdp.DistanceGatedRelativeJointPositionSim2SimActionCfg(
        asset_name="robot",
        joint_names=[],
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups for dexgrasp_float."""

    @configclass
    class ActorCfg(ObsGroup):
        obj_pc_obs = None
        hand_goal_pose_rel = ObsTerm(func=mdp.hand_goal_pose_rel, params={"command_name": "goal"})
        hand_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)})
        hand_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", preserve_order=True)})
        hand_root_lin_vel = ObsTerm(func=mdp.hand_root_lin_vel, params={"robot_cfg": SceneEntityCfg("robot")})
        hand_root_ang_vel = ObsTerm(func=mdp.hand_root_ang_vel, params={"robot_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ActorCfg):
        obj_pc_obs = ObsTerm(
            func=mdp.obj_pc_full_bps_obs,
            params={"num_points": DEFAULT_GEOMETRY_NUM_POINTS, "basis_path": str(default_bps_basis_path())},
        )
        obj_pose_rel = ObsTerm(func=mdp.obj_pose_rel)
        obj_lin_vel = ObsTerm(func=mdp.obj_lin_vel)
        obj_ang_vel = ObsTerm(func=mdp.obj_ang_vel)
        # Observation-side geometry is unified to BPS; legacy hand_obj_geo_vec is not used here.
        hand_obj_geo_vec = None
        hand_contact_force = ObsTerm(
            func=mdp.hand_contact_force,
            params={"contact_sensor_names": (), "normalize_by": 500.0},
        )
        hand_joint_force = ObsTerm(func=mdp.hand_joint_force, params={"robot_cfg": SceneEntityCfg("robot", preserve_order=True)})

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DistillObservationsCfg(ObservationsCfg):
    """Observation groups for teacher-student distillation."""

    @configclass
    class TeacherCfg(ObservationsCfg.ActorCfg):
        def __post_init__(self):
            super().__post_init__()

    @configclass
    class StudentCfg(ObservationsCfg.ActorCfg):
        def __post_init__(self):
            super().__post_init__()

    teacher: TeacherCfg = TeacherCfg()
    student: StudentCfg = StudentCfg()


@configclass
class EventCfg:
    """Reset and startup events for dexgrasp_float."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (DEFAULT_STATIC_FRICTION, DEFAULT_STATIC_FRICTION),
            "dynamic_friction_range": (DEFAULT_DYNAMIC_FRICTION, DEFAULT_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    robot_collider_offsets = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets_relative,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "contact_offset_scale_range": (1.0, 1.0),
            "rest_offset_delta_range": (0.0, 0.0),
        },
    )
    palm_dynamics_scale = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),
            "mass_distribution_params": (20.0, 20.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    # object_visual_color = EventTerm(
    #     func=mdp.apply_rigid_object_preview_color,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "diffuse_color": (0.2, 0.2, 0.6),
    #         "roughness": 0.9,
    #         "metallic": 0.0,
    #         "opacity": 1.0,
    #         "material_name": "dexgrasp_object_preview_gray",
    #     },
    # )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (DEFAULT_STATIC_FRICTION, DEFAULT_STATIC_FRICTION),
            "dynamic_friction_range": (DEFAULT_DYNAMIC_FRICTION, DEFAULT_DYNAMIC_FRICTION),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    object_collider_offsets = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets_relative,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "contact_offset_scale_range": (1.0, 1.0),
            "rest_offset_delta_range": (0.0, 0.0),
        },
    )
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sample_object_initial_pose = EventTerm(
        func=mdp.sample_object_initial_pose,
        mode="reset",
        params={
            "position": (0.0, 0.0, 0.0),
            "roll_range_deg": (-180.0, 180.0),
            "pitch_range_deg": (-180.0, 180.0),
            "yaw_range_deg": (-180.0, 180.0),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    sample_hand_home_pose = EventTerm(
        func=mdp.sample_hand_home_pose_sim2sim,
        mode="reset",
        params={
            "home_joint_positions": (),
            "hand_object_relative_position": None,
            "hand_object_relative_quat": None,
            "position_jitter_std": 0.005,
            "orientation_jitter_std_deg": 4.0,
            "min_world_z": None,
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    sample_camera_home_pose = EventTerm(
        func=mdp.sample_camera_home_pose,
        mode="reset",
        params={
            "camera_object_relative_position": None,
            "camera_object_relative_quat": None,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for dexgrasp_float."""

    hand_obj_approach_penalty = RewTerm(
        func=mdp.hand_obj_approach_penalty,
        weight=-1.0,
        params={"active_phase": "grasp"},
    )
    hand_obj_approach_direction_penalty = RewTerm(
        func=mdp.hand_obj_approach_direction_penalty,
        weight=-1.0,
        params={"active_phase": "grasp"},
    )
    hand_geo_proximity_penalty = RewTerm(
        func=mdp.hand_geo_proximity_penalty,
        weight=-2.5,
        params={"active_phase": "grasp"},
    )
    obj_goal_pos_reward = RewTerm(
        func=mdp.obj_goal_pos_reward,
        weight=5.0,
        params={
            "command_name": "goal",
            "base": 1.0,
            "slope": 2.0,
            "active_phase": "both",
        },
    )
    obj_goal_rot_reward = RewTerm(
        func=mdp.obj_goal_rot_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "base": 1.0,
            "slope": 2.0 / math.pi,
            "active_phase": "grasp",
        },
    )
    hand_goal_pos_reward = RewTerm(
        func=mdp.hand_goal_pos_reward,
        weight=5.0,
        params={
            "command_name": "goal",
            "base": 1.0,
            "slope": 2.0,
            "active_phase": "grasp",
        },
    )
    hand_goal_rot_reward = RewTerm(
        func=mdp.hand_goal_rot_reward,
        weight=1.0,
        params={
            "command_name": "goal",
            "base": 1.0,
            "slope": 2.0 / math.pi,
            "active_phase": "grasp",
        },
    )
    obj_pos_success_sparse_reward = RewTerm(
        func=mdp.obj_pos_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_pos": 10.0,
            "active_phase": "grasp",
        },
    )
    obj_rot_success_sparse_reward = RewTerm(
        func=mdp.obj_rot_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_rot": 3.0,
            "active_phase": "grasp",
        },
    )
    obj_success_sparse_reward = RewTerm(
        func=mdp.obj_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_pos": 10.0,
            "k_rot": 3.0,
            "active_phase": "grasp",
        },
    )
    hand_pos_success_sparse_reward = RewTerm(
        func=mdp.hand_pos_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_pos": 10.0,
            "active_phase": "grasp",
        },
    )
    hand_rot_success_sparse_reward = RewTerm(
        func=mdp.hand_rot_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_rot": 3.0,
            "active_phase": "grasp",
        },
    )
    hand_success_sparse_reward = RewTerm(
        func=mdp.hand_success_sparse_reward,
        weight=0.0,
        params={
            "command_name": "goal",
            "k_pos": 10.0,
            "k_rot": 3.0,
            "active_phase": "grasp",
        },
    )
    hand_success_obj_pos_success_sparse_reward = RewTerm(
        func=mdp.hand_success_obj_pos_success_sparse_reward,
        weight=45.0,
        params={
            "command_name": "goal",
            "k_obj_pos": 10.0,
            "k_hand_pos": 10.0,
            "k_hand_rot": 3.0,
            "active_phase": "both",
        },
    )
    # success_low_dynamics_sparse_reward = RewTerm(
    #     func=mdp.success_low_dynamics_sparse_reward,
    #     weight=0.0,
    #     params={
    #         "command_name": "goal",
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "object_cfg": SceneEntityCfg("object"),
    #         "k_obj_pos": 10.0,
    #         "k_hand_pos": 10.0,
    #         "k_hand_rot": 3.0,
    #         "k_obj_lin_speed": 2.0,
    #         "k_obj_ang_speed": 0.5,
    #         "k_root_lin_speed": 2.0,
    #         "k_root_ang_speed": 0.5,
    #         "primary_contact_sensor_names": (),
    #         "contact_sensor_groups": (),
    #         "min_contact_fingers": DEFAULT_GOAL_SUCCESS_MIN_CONTACT_FINGERS,
    #         "active_phase": "grasp",
    #     },
    # )
    action_l2_penalty = RewTerm(func=mdp.action_l2_penalty, weight=0.0, params={"active_phase": "grasp"})
    action_rate_l2_penalty = RewTerm(func=mdp.action_rate_l2_penalty, weight=0.0, params={"active_phase": "grasp"})
    root_action_l2_penalty = RewTerm(
        func=mdp.root_action_l2_penalty,
        weight=-0.005,
        params={"active_phase": "grasp"},
    )
    root_action_rate_l2_penalty = RewTerm(
        func=mdp.root_action_rate_l2_penalty,
        weight=-0.01,
        params={"active_phase": "grasp"},
    )
    joint_action_l2_penalty = RewTerm(
        func=mdp.joint_action_l2_penalty,
        weight=-0.005,
        params={"active_phase": "both"},
    )
    joint_action_rate_l2_penalty = RewTerm(
        func=mdp.joint_action_rate_l2_penalty,
        weight=-0.01,
        params={"active_phase": "both"},
    )
    joint_pos_limits_penalty = RewTerm(
        func=mdp.joint_pos_limits_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "active_phase": "both"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for dexgrasp_float."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_too_far = DoneTerm(func=mdp.object_too_far, params={"threshold": DEFAULT_OBJECT_RESET_DISTANCE})
    object_goal_too_far = DoneTerm(
        func=mdp.object_goal_too_far,
        params={"command_name": "goal", "threshold": DEFAULT_OBJECT_GOAL_RESET_DISTANCE},
    )
    invalid_state = DoneTerm(func=mdp.invalid_state, params={"contact_sensor_names": ()})


@configclass
class CurriculumCfg:
    """Curriculum is intentionally disabled for strict baseline parity."""

    pass


@configclass
class DexGraspFloatEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment config for floating-hand dexgrasp_float."""

    scene: DexGraspFloatSceneCfg = DexGraspFloatSceneCfg(num_envs=1024, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    object_sets: ObjectSetQueryCfg = ObjectSetQueryCfg()
    hand_profile: DexGraspHandProfile | None = None
    sim_timestep: float = 1.0 / 360
    control_frequency_hz: int = 30
    render_frequency_hz: int = 30
    phase_switch_time_s: float = 1.0
    episode_length_s: float = 5.0
    geometry_num_points: int = DEFAULT_GEOMETRY_NUM_POINTS
    observation_mode: str = "full"
    target_pose_source_action: str = "full"
    target_pose_source_reward: str = "full"
    floating_root_action_family: str = "wrench_pbvs"
    student_camera_object_relative_position: tuple[float, float, float] | None = None
    student_camera_object_relative_quat: tuple[float, float, float, float] | None = None
    student_zbuf_width: int = 64
    student_zbuf_height: int = 48
    student_camera_horizontal_fov_deg: float = 90.0
    student_camera_near: float = 0.05
    student_camera_far: float = 2.0
    student_visibility_depth_margin: float = 0.005
    student_partial_pc_noise_std: float = 0.0
    object_assignment_seed: int | None = None
    geometry_point_cloud_source: str = DEFAULT_GEOMETRY_POINT_CLOUD_SOURCE
    object_assignment_verbose: bool = False
    palm_dynamics_scale: float = 20.0
    palm_dynamics_scale_range: tuple[float, float] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.sim_timestep <= 0.0:
            raise ValueError(f"sim_timestep must be positive, got {self.sim_timestep}.")
        if self.control_frequency_hz <= 0:
            raise ValueError(f"control_frequency_hz must be positive, got {self.control_frequency_hz}.")
        if self.render_frequency_hz <= 0:
            raise ValueError(f"render_frequency_hz must be positive, got {self.render_frequency_hz}.")
        if self.observation_mode not in {"partial", "full"}:
            raise ValueError(
                f"observation_mode must be 'partial' or 'full', got {self.observation_mode!r}."
            )
        if self.target_pose_source_action not in {"full", "partial"}:
            raise ValueError(
                "target_pose_source_action must be 'full' or 'partial', "
                f"got {self.target_pose_source_action!r}."
            )
        if self.target_pose_source_reward not in {"full", "partial"}:
            raise ValueError(
                "target_pose_source_reward must be 'full' or 'partial', "
                f"got {self.target_pose_source_reward!r}."
            )
        if self.floating_root_action_family not in {"wrench_pbvs", "wrench"}:
            raise ValueError(
                "floating_root_action_family must be 'wrench_pbvs' or 'wrench', "
                f"got {self.floating_root_action_family!r}."
            )
        decimation = int(round(1.0 / (self.sim_timestep * float(self.control_frequency_hz))))
        if decimation <= 0:
            raise ValueError(
                "Computed invalid decimation from sim_timestep and control_frequency_hz: "
                f"{self.sim_timestep}, {self.control_frequency_hz}."
            )
        env_step = self.sim_timestep * decimation
        target_env_step = 1.0 / float(self.control_frequency_hz)
        if abs(env_step - target_env_step) > 1e-9:
            raise ValueError(
                "sim_timestep and control_frequency_hz do not produce an integer decimation: "
                f"sim_timestep={self.sim_timestep}, control_frequency_hz={self.control_frequency_hz}, "
                f"computed_env_step={env_step}, target_env_step={target_env_step}."
            )

        self.decimation = decimation
        self.scene.replicate_physics = False
        self.scene.clone_in_fabric = False
        self.sim.dt = self.sim_timestep
        render_interval = int(round(1.0 / (self.sim_timestep * float(self.render_frequency_hz))))
        self.sim.render_interval = max(1, render_interval)
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physics_material.static_friction = DEFAULT_STATIC_FRICTION
        self.sim.physics_material.dynamic_friction = DEFAULT_DYNAMIC_FRICTION
        self.sim.physics_material.restitution = 0.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.friction_correlation_distance = 0.00625
        self._wire_hand_dependent_terms()
        self._apply_task_runtime_cfg()
        self.apply_object_selection(log_assignment=False)

    def _require_hand_profile(self) -> DexGraspHandProfile:
        if self.hand_profile is None:
            raise ValueError(f"{self.__class__.__name__} requires a hand_profile.")
        return self.hand_profile

    def _validate_hand_profile(self, profile: DexGraspHandProfile) -> None:
        if len(profile.hand_joint_names) != len(profile.hand_home_joint_positions):
            raise ValueError(
                f"{profile.hand_name}: hand_joint_names and hand_home_joint_positions length mismatch: "
                f"{len(profile.hand_joint_names)} vs {len(profile.hand_home_joint_positions)}."
            )
        if len(profile.hand_geo_body_names) != len(profile.hand_geo_body_weights):
            raise ValueError(
                f"{profile.hand_name}: hand_geo_body_names and hand_geo_body_weights length mismatch: "
                f"{len(profile.hand_geo_body_names)} vs {len(profile.hand_geo_body_weights)}."
            )
        missing_home_joints = set(profile.hand_joint_names) - set(profile.hand_home_joint_pos_map)
        if missing_home_joints:
            raise ValueError(
                f"{profile.hand_name}: hand_home_joint_pos_map is missing joints: {sorted(missing_home_joints)}."
            )
        missing_actuated_joints = set(profile.hand_actuated_joint_names) - set(profile.hand_joint_names)
        if missing_actuated_joints:
            raise ValueError(
                f"{profile.hand_name}: hand_actuated_joint_names are not a subset of hand_joint_names: "
                f"{sorted(missing_actuated_joints)}."
            )
        missing_fingertips = set(profile.hand_contact_fingertip_body_names) - set(profile.hand_contact_body_names)
        if missing_fingertips:
            raise ValueError(
                f"{profile.hand_name}: hand_contact_fingertip_body_names are not covered by hand_contact_body_names: "
                f"{sorted(missing_fingertips)}."
            )
        flat_finger_body_names = {body_name for body_names in profile.hand_contact_finger_body_names for body_name in body_names}
        missing_finger_bodies = flat_finger_body_names - set(profile.hand_contact_body_names)
        if missing_finger_bodies:
            raise ValueError(
                f"{profile.hand_name}: hand_contact_finger_body_names are not covered by hand_contact_body_names: "
                f"{sorted(missing_finger_bodies)}."
            )
        if len(profile.hand_transform_pos) != 3:
            raise ValueError(f"{profile.hand_name}: hand_transform_pos must have length 3.")
        if len(profile.hand_transform_quat) != 4:
            raise ValueError(f"{profile.hand_name}: hand_transform_quat must have length 4.")
        if len(profile.hand_object_relative_pos) != 3:
            raise ValueError(f"{profile.hand_name}: hand_object_relative_pos must have length 3.")
        if len(profile.hand_object_relative_quat) != 4:
            raise ValueError(f"{profile.hand_name}: hand_object_relative_quat must have length 4.")
        if len(profile.hand_camera_object_relative_pos) != 3:
            raise ValueError(f"{profile.hand_name}: hand_camera_object_relative_pos must have length 3.")
        if len(profile.hand_camera_object_relative_quat) != 4:
            raise ValueError(f"{profile.hand_name}: hand_camera_object_relative_quat must have length 4.")

    def _wire_hand_dependent_terms(self) -> None:
        profile = self._require_hand_profile()
        self._validate_hand_profile(profile)

        self._apply_hand_profile_constants(profile)
        self._wire_action_terms(profile)
        self._wire_observation_terms(profile)
        self._wire_event_terms()
        self._wire_reward_terms()
        self._wire_command_terms()
        self._wire_termination_terms()

    def _apply_hand_profile_constants(self, profile: DexGraspHandProfile) -> None:
        """Copy hand-profile constants into environment-level runtime fields."""
        self.scene.robot = profile.hand_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if profile.hand_articulation_root_prim_path:
            self.scene.robot.articulation_root_prim_path = profile.hand_articulation_root_prim_path
        self.scene.robot.spawn.activate_contact_sensors = True
        articulation_props = getattr(self.scene.robot.spawn, "articulation_props", None)
        if articulation_props is not None and hasattr(articulation_props, "fix_root_link"):
            articulation_props.fix_root_link = False

        self.hand_home_joint_positions = dict(profile.hand_home_joint_pos_map)
        self.hand_transform_pos = profile.hand_transform_pos
        self.hand_transform_quat = profile.hand_transform_quat
        self.hand_object_relative_position = profile.hand_object_relative_pos
        self.hand_object_relative_quat = profile.hand_object_relative_quat
        if self.student_camera_object_relative_position is None:
            self.student_camera_object_relative_position = profile.hand_camera_object_relative_pos
        if self.student_camera_object_relative_quat is None:
            self.student_camera_object_relative_quat = profile.hand_camera_object_relative_quat
        self.palm_body_name = profile.hand_palm_body_name
        self.hand_geometry_body_names = profile.hand_geo_body_names
        self.hand_geometry_body_weights = profile.hand_geo_body_weights
        self.reward_finger_contact_mode = "fingertip"
        self.observation_contact_sensor_names = tuple(
            f"{body_name}_object_s" for body_name in profile.hand_contact_body_names
        )
        self.reward_contact_sensor_names = tuple(
            f"{body_name}_object_s" for body_name in profile.hand_contact_fingertip_body_names
        )
        self.reward_contact_sensor_groups = tuple(
            tuple(f"{body_name}_object_s" for body_name in body_names)
            for body_names in profile.hand_contact_finger_body_names
        )

    def _wire_action_terms(self, profile: DexGraspHandProfile) -> None:
        """Bind action-term structure and runtime parameters to the active hand profile."""
        if self.floating_root_action_family == "wrench":
            self.actions.floating_root = mdp.RootTwistWrenchActionCfg(asset_name="robot")
        else:
            self.actions.floating_root = mdp.RootTwistPBVSWrenchSim2SimActionCfg(
                asset_name="robot",
                hand_transform_pos=profile.hand_transform_pos,
                hand_transform_quat=profile.hand_transform_quat,
            )
        self.actions.hand_joints = mdp.DistanceGatedRelativeJointPositionSim2SimActionCfg(
            asset_name="robot",
            joint_names=list(profile.hand_actuated_joint_names),
            preserve_order=True,
            use_zero_offset=True,
            home_joint_positions=dict(profile.hand_home_joint_pos_map),
            hand_transform_pos=profile.hand_transform_pos,
            hand_transform_quat=profile.hand_transform_quat,
        )
        if hasattr(self.actions.floating_root, "phase_switch_time_s"):
            self.actions.floating_root.phase_switch_time_s = self.phase_switch_time_s
        if hasattr(self.actions.floating_root, "approach_num_points"):
            self.actions.floating_root.approach_num_points = self.geometry_num_points
        if hasattr(self.actions.hand_joints, "phase_switch_time_s"):
            self.actions.hand_joints.phase_switch_time_s = self.phase_switch_time_s
        if hasattr(self.actions.hand_joints, "approach_num_points"):
            self.actions.hand_joints.approach_num_points = self.geometry_num_points
        if hasattr(self.actions.hand_joints, "standoff_distance"):
            self.actions.hand_joints.standoff_distance = 0.0
        if self.actions.hand_joints.joint_names != list(profile.hand_actuated_joint_names):
            raise ValueError("hand_joints action term joint_names do not match hand_profile.hand_actuated_joint_names.")
        if self.actions.hand_joints.preserve_order is not True:
            raise ValueError("hand_joints action term must preserve the explicit policy joint order.")

    def _wire_observation_terms(self, profile: DexGraspHandProfile) -> None:
        """Bind observation-term order, geometry, and contact sensors to the active hand profile."""
        bps_params = {"num_points": self.geometry_num_points, "basis_path": str(default_bps_basis_path())}
        if isinstance(self.observations, DistillObservationsCfg):
            # Keep the PPO-compatible actor group available in distillation envs so full-observation teacher
            # checkpoints can be evaluated under exactly the same observation contract as bucket eval.
            self.observations.actor.obj_pose_rel = ObsTerm(func=mdp.obj_pose_rel)
            self.observations.actor.obj_lin_vel = ObsTerm(func=mdp.obj_lin_vel)
            self.observations.actor.obj_ang_vel = ObsTerm(func=mdp.obj_ang_vel)
            self.observations.actor.obj_pc_obs = ObsTerm(func=mdp.obj_pc_full_bps_obs, params=dict(bps_params))
            self.observations.teacher.obj_pose_rel = ObsTerm(func=mdp.obj_pose_rel)
            self.observations.teacher.obj_lin_vel = ObsTerm(func=mdp.obj_lin_vel)
            self.observations.teacher.obj_ang_vel = ObsTerm(func=mdp.obj_ang_vel)
            self.observations.teacher.obj_pc_obs = ObsTerm(func=mdp.obj_pc_full_bps_obs, params=dict(bps_params))
            self.observations.student.obj_pc_obs = ObsTerm(func=mdp.obj_pc_part_bps_obs, params=dict(bps_params))
        elif self.observation_mode == "full":
            self.observations.actor.obj_pose_rel = ObsTerm(func=mdp.obj_pose_rel)
            self.observations.actor.obj_lin_vel = ObsTerm(func=mdp.obj_lin_vel)
            self.observations.actor.obj_ang_vel = ObsTerm(func=mdp.obj_ang_vel)
            self.observations.actor.obj_pc_obs = ObsTerm(func=mdp.obj_pc_full_bps_obs, params=dict(bps_params))
        else:
            self.observations.actor.obj_pose_rel = None
            self.observations.actor.obj_lin_vel = None
            self.observations.actor.obj_ang_vel = None
            self.observations.actor.obj_pc_obs = ObsTerm(func=mdp.obj_pc_part_bps_obs, params=dict(bps_params))

        self.observations.actor.hand_joint_pos.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
        self.observations.actor.hand_joint_vel.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
        self.observations.critic.hand_joint_pos.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
        self.observations.critic.hand_joint_vel.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
        self.observations.critic.hand_joint_force.params["robot_cfg"].joint_names = list(profile.hand_joint_names)
        if isinstance(self.observations, DistillObservationsCfg):
            self.observations.teacher.hand_joint_pos.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
            self.observations.teacher.hand_joint_vel.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
            self.observations.student.hand_joint_pos.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
            self.observations.student.hand_joint_vel.params["asset_cfg"].joint_names = list(profile.hand_joint_names)
        for body_name, sensor_name in zip(
            profile.hand_contact_body_names, self.observation_contact_sensor_names, strict=True
        ):
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=profile.hand_contact_sensor_prim_path_prefix + body_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    history_length=8,
                    update_period=0.0,
                ),
            )
        self.observations.critic.hand_contact_force.params.update(
            {"contact_sensor_names": self.observation_contact_sensor_names}
        )
        if self.hand_geometry_body_names != profile.hand_geo_body_names:
            raise ValueError("hand_geometry_body_names do not match hand_profile.hand_geo_body_names.")
        if len(self.observation_contact_sensor_names) != len(profile.hand_contact_body_names):
            raise ValueError("observation_contact_sensor_names must align one-to-one with hand_contact_body_names.")
        for obs_cfg in (
            self.observations.actor.hand_joint_pos.params["asset_cfg"],
            self.observations.actor.hand_joint_vel.params["asset_cfg"],
            self.observations.critic.hand_joint_pos.params["asset_cfg"],
            self.observations.critic.hand_joint_vel.params["asset_cfg"],
            self.observations.critic.hand_joint_force.params["robot_cfg"],
        ):
            if obs_cfg.joint_names != list(profile.hand_joint_names) or obs_cfg.preserve_order is not True:
                raise ValueError("Policy-facing joint-family observation terms must use the explicit hand_profile order.")
        if isinstance(self.observations, DistillObservationsCfg):
            for obs_cfg in (
                self.observations.teacher.hand_joint_pos.params["asset_cfg"],
                self.observations.teacher.hand_joint_vel.params["asset_cfg"],
                self.observations.student.hand_joint_pos.params["asset_cfg"],
                self.observations.student.hand_joint_vel.params["asset_cfg"],
            ):
                if obs_cfg.joint_names != list(profile.hand_joint_names) or obs_cfg.preserve_order is not True:
                    raise ValueError(
                        "Distill joint-family observation terms must use the explicit hand_profile order."
                    )

    def _log_env_object_assignment(self, env_object_names: tuple[str, ...]) -> None:
        """Log deterministic env->object mapping for startup verification."""
        summary = (
            f"Object assignment summary: num_envs={len(env_object_names)} "
            f"unique_objects={len(set(env_object_names))}"
        )
        LOGGER.info(summary)
        print(summary, flush=True)
        for env_id, object_name in enumerate(env_object_names):
            line = f"Object assignment: env_{env_id:04d} -> {object_name}"
            LOGGER.info(line)
            print(line, flush=True)

    def _set_contact_sensor_filter_expr(self, object_filter_expr: str) -> None:
        for field_name, field_value in vars(self.scene).items():
            if field_name.endswith("_object_s") and hasattr(field_value, "filter_prim_paths_expr"):
                field_value.filter_prim_paths_expr = [object_filter_expr]

    def _wire_reward_terms(self) -> None:
        """Inject runtime reward parameters from the assembled hand/environment config."""
        phase_params = {"phase_switch_time_s": self.phase_switch_time_s}
        contact_params = {
            "primary_contact_sensor_names": self.reward_contact_sensor_names,
            "contact_sensor_groups": self.reward_contact_sensor_groups,
            "finger_contact_mode": self.reward_finger_contact_mode,
            "min_contact_fingers": DEFAULT_GOAL_SUCCESS_MIN_CONTACT_FINGERS,
        }
        hand_params = {
            "palm_body_name": self.palm_body_name,
            "hand_transform_pos": self.hand_transform_pos,
            "hand_transform_quat": self.hand_transform_quat,
        }
        geometry_params = {
            "body_names": self.hand_geometry_body_names,
            "body_weights": self.hand_geometry_body_weights,
            "num_points": self.geometry_num_points,
        }

        self.rewards.hand_obj_approach_penalty.params.update({**hand_params, **phase_params})
        self.rewards.hand_obj_approach_direction_penalty.params.update({**hand_params, **phase_params})
        self.rewards.hand_geo_proximity_penalty.params.update({**geometry_params, **phase_params})

        for reward_term in (
            self.rewards.obj_goal_pos_reward,
            self.rewards.obj_goal_rot_reward,
            self.rewards.hand_goal_pos_reward,
            self.rewards.hand_goal_rot_reward,
            self.rewards.obj_pos_success_sparse_reward,
            self.rewards.obj_rot_success_sparse_reward,
            self.rewards.obj_success_sparse_reward,
            self.rewards.hand_pos_success_sparse_reward,
            self.rewards.hand_rot_success_sparse_reward,
            self.rewards.hand_success_sparse_reward,
            self.rewards.hand_success_obj_pos_success_sparse_reward,
        ):
            reward_term.params.update({**contact_params, **phase_params})

        for reward_term in (
            self.rewards.action_l2_penalty,
            self.rewards.action_rate_l2_penalty,
            self.rewards.root_action_l2_penalty,
            self.rewards.root_action_rate_l2_penalty,
            self.rewards.joint_action_l2_penalty,
            self.rewards.joint_action_rate_l2_penalty,
            self.rewards.joint_pos_limits_penalty,
        ):
            reward_term.params.update(phase_params)

        for reward_name, required_keys in (
            ("hand_obj_approach_penalty", ("palm_body_name", "hand_transform_pos", "hand_transform_quat")),
            ("hand_obj_approach_direction_penalty", ("palm_body_name", "hand_transform_pos", "hand_transform_quat")),
            ("hand_geo_proximity_penalty", ("body_names", "body_weights", "num_points")),
            (
                "hand_success_obj_pos_success_sparse_reward",
                ("primary_contact_sensor_names", "contact_sensor_groups", "finger_contact_mode", "min_contact_fingers"),
            ),
        ):
            reward_params = getattr(self.rewards, reward_name).params
            missing_keys = tuple(key for key in required_keys if key not in reward_params)
            if missing_keys:
                raise ValueError(f"{reward_name} is missing runtime reward params: {missing_keys}.")

    def _wire_event_terms(self) -> None:
        """Populate reset-event parameters that depend on the selected hand profile."""
        if self.hand_object_relative_position is None or self.hand_object_relative_quat is None:
            raise ValueError("hand object-relative pose must be injected before wiring event terms.")
        if self.student_camera_object_relative_position is None or self.student_camera_object_relative_quat is None:
            raise ValueError("student camera object-relative pose must be injected before wiring event terms.")
        if self.palm_dynamics_scale <= 0.0:
            raise ValueError(f"palm_dynamics_scale must be positive, got {self.palm_dynamics_scale}.")
        palm_dynamics_scale_range = (
            (float(self.palm_dynamics_scale), float(self.palm_dynamics_scale))
            if self.palm_dynamics_scale_range is None
            else tuple(float(value) for value in self.palm_dynamics_scale_range)
        )
        if len(palm_dynamics_scale_range) != 2:
            raise ValueError(f"palm_dynamics_scale_range must contain two values, got {palm_dynamics_scale_range}.")
        if palm_dynamics_scale_range[0] <= 0.0 or palm_dynamics_scale_range[1] < palm_dynamics_scale_range[0]:
            raise ValueError(f"Invalid palm_dynamics_scale_range: {palm_dynamics_scale_range}.")
        self.events.palm_dynamics_scale.params.update(
            {
                "asset_cfg": SceneEntityCfg("robot", body_names=(self.palm_body_name,)),
                "mass_distribution_params": palm_dynamics_scale_range,
            }
        )
        self.events.sample_hand_home_pose.params.update(
            {
                "home_joint_positions": self.hand_home_joint_positions,
                "hand_object_relative_position": self.hand_object_relative_position,
                "hand_object_relative_quat": self.hand_object_relative_quat,
            }
        )
        self.events.sample_camera_home_pose.params.update(
            {
                "camera_object_relative_position": self.student_camera_object_relative_position,
                "camera_object_relative_quat": self.student_camera_object_relative_quat,
            }
        )

    def _wire_command_terms(self) -> None:
        """Populate command-term parameters that depend on the selected hand profile."""
        self.commands.goal.hand_transform_pos = self.hand_transform_pos
        self.commands.goal.hand_transform_quat = self.hand_transform_quat
        self.commands.goal.phase_switch_time_s = self.phase_switch_time_s
        self.commands.goal.palm_body_name = self.palm_body_name
        self.commands.goal.contact_sensor_names = self.reward_contact_sensor_names
        self.commands.goal.contact_sensor_groups = self.reward_contact_sensor_groups
        self.commands.goal.finger_contact_mode = self.reward_finger_contact_mode
        self.commands.goal.geometry_debug_body_names = self.hand_geometry_body_names

    def _wire_termination_terms(self) -> None:
        """Populate termination-term parameters that depend on runtime contact sensor names."""
        self.terminations.invalid_state.params.update({"contact_sensor_names": self.observation_contact_sensor_names})

    def _apply_task_runtime_cfg(self) -> None:
        """Apply pure task/scene runtime settings that do not belong to a specific manager category."""
        if self.scene.robot.spawn.rigid_props is not None:
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        if self.scene.object.spawn.rigid_props is not None:
            self.scene.object.spawn.rigid_props.disable_gravity = True
        if hasattr(self.scene.object.spawn, "activate_contact_sensors"):
            self.scene.object.spawn.activate_contact_sensors = True

    def apply_object_selection(self, *, log_assignment: bool = True) -> ObjectSelection:
        """Resolve the configured object set and map it into the scene spawn config."""
        assignment_seed = self.seed if self.object_assignment_seed is None else self.object_assignment_seed
        if assignment_seed is None:
            assignment_seed = 0
        normalized_object_set_id = self.object_sets.object_set_id.strip().lower()
        if normalized_object_set_id in DEFAULT_DEXCUBE_OBJECT_SET_IDS:
            spawn_factory = _DEFAULT_NUCLEUS_BLOCK_SPAWN_MAP.get(
                normalized_object_set_id,
                _DEFAULT_NUCLEUS_BLOCK_SPAWN_MAP["default:dexcube"],
            )
            spawn_cfg = spawn_factory()
            if isinstance(spawn_cfg, UsdFileCfg):
                per_env_usd_paths = build_env_usd_path_list(
                    (spawn_cfg.usd_path,),
                    self.scene.num_envs,
                    seed=assignment_seed,
                )
                self._env_object_scale_keys = tuple(normalized_object_set_id for _ in range(self.scene.num_envs))
                self._env_object_usd_paths = tuple(per_env_usd_paths)
                self._env_object_dataset_pc_paths = tuple(None for _ in range(self.scene.num_envs))
                self._env_object_global_pc_paths = self._env_object_dataset_pc_paths
                self.scene.object.spawn = MultiUsdFileCfg(
                    usd_path=list(per_env_usd_paths),
                    random_choice=False,
                    scale=spawn_cfg.scale,
                    activate_contact_sensors=spawn_cfg.activate_contact_sensors,
                    rigid_props=spawn_cfg.rigid_props,
                    collision_props=spawn_cfg.collision_props,
                    mass_props=spawn_cfg.mass_props,
                    articulation_props=spawn_cfg.articulation_props,
                    fixed_tendons_props=spawn_cfg.fixed_tendons_props,
                    joint_drive_props=spawn_cfg.joint_drive_props,
                    deformable_props=spawn_cfg.deformable_props,
                    visual_material=spawn_cfg.visual_material,
                    visual_material_path=spawn_cfg.visual_material_path,
                    semantic_tags=spawn_cfg.semantic_tags,
                    copy_from_source=spawn_cfg.copy_from_source,
                )
                if log_assignment and self.object_assignment_verbose:
                    self._log_env_object_assignment(
                        tuple(f"{normalized_object_set_id}:{usd_path}" for usd_path in per_env_usd_paths)
                    )
            else:
                self.scene.object.spawn = spawn_cfg
                self._env_object_scale_keys = tuple(normalized_object_set_id for _ in range(self.scene.num_envs))
                self._env_object_usd_paths = tuple("" for _ in range(self.scene.num_envs))
                self._env_object_dataset_pc_paths = tuple(None for _ in range(self.scene.num_envs))
                self._env_object_global_pc_paths = self._env_object_dataset_pc_paths
            self._set_contact_sensor_filter_expr("{ENV_REGEX_NS}/Object")
            return ObjectSelection(split_side="train", object_set_id=self.object_sets.object_set_id, active_assets=())

        selection = resolve_object_selection(self.object_sets)
        usd_paths = tuple(
            str(asset.resolved_usd_path(self.object_sets.dataset_root))
            for asset in selection.active_assets
        )
        for usd_path in usd_paths:
            if check_file_path(usd_path):
                continue
            raise FileNotFoundError(
                "Resolved object_set_id includes a missing USD asset. "
                "Run dexgrasp_sample/prepare_object_usds.py first. "
                f"Missing path: {usd_path}"
            )

        per_env_usd_paths = build_env_usd_path_list(
            usd_paths,
            self.scene.num_envs,
            seed=assignment_seed,
        )
        key_to_asset = {asset.object_scale_key: asset for asset in selection.active_assets}
        usd_to_object_name = {
            str(asset.resolved_usd_path(self.object_sets.dataset_root)): asset.object_scale_key
            for asset in selection.active_assets
        }
        env_object_names = tuple(usd_to_object_name[path] for path in per_env_usd_paths)
        dataset_root = Path(self.object_sets.dataset_root).expanduser().resolve()
        env_dataset_pc_paths: list[str | None] = []
        for object_name in env_object_names:
            asset = key_to_asset[object_name]
            dataset_pc_path = dataset_root / asset.asset_path / "pc_warp" / "global_pc.npy"
            env_dataset_pc_paths.append(str(dataset_pc_path) if dataset_pc_path.is_file() else None)
        self._env_object_scale_keys = tuple(env_object_names)
        self._env_object_usd_paths = tuple(per_env_usd_paths)
        self._env_object_dataset_pc_paths = tuple(env_dataset_pc_paths)
        self._env_object_global_pc_paths = self._env_object_dataset_pc_paths
        self.scene.object.spawn = MultiUsdFileCfg(
            usd_path=list(per_env_usd_paths),
            random_choice=False,
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            rigid_props=_make_default_object_rigid_props(),
        )
        if log_assignment and self.object_assignment_verbose:
            self._log_env_object_assignment(env_object_names)
        dataset_object_filter_expr = "{ENV_REGEX_NS}/Object/base_link"
        self._set_contact_sensor_filter_expr(dataset_object_filter_expr)
        return selection
