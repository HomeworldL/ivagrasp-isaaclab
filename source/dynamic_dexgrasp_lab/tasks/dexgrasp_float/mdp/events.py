"""Reset events specific to the floating dexgrasp task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from dynamic_dexgrasp_lab.assets.data.sampler import sample_positions_on_spherical_shell
from dynamic_dexgrasp_lab.utils import (
    combine_frame_transforms,
    grasp_to_palm_transform,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def apply_rigid_object_preview_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diffuse_color: tuple[float, float, float] = (0.55, 0.55, 0.55),
    roughness: float = 0.9,
    metallic: float = 0.0,
    opacity: float = 1.0,
    material_name: str = "dexgrasp_object_preview_gray",
) -> None:
    """Bind a stable preview material to all environment-local object prims.

    This avoids the replicator dependency in IsaacLab's built-in visual-color
    randomizer while still keeping the visual path inside the environment event
    system shared by train/play/viewer.
    """
    del env_ids

    asset = env.scene[asset_cfg.name]
    stage = sim_utils.get_current_stage()
    material_prim_path = f"/World/Looks/{material_name}"

    if not stage.GetPrimAtPath(material_prim_path).IsValid():
        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=diffuse_color,
            roughness=roughness,
            metallic=metallic,
            opacity=opacity,
        )
        material_cfg.func(material_prim_path, material_cfg)

    object_suffix = asset.cfg.prim_path
    env_regex_prefix = f"{env.scene.env_regex_ns}/"
    if object_suffix.startswith(env_regex_prefix):
        object_suffix = object_suffix[len(env_regex_prefix) :]
    elif "{ENV_REGEX_NS}/" in object_suffix:
        object_suffix = object_suffix.split("{ENV_REGEX_NS}/", 1)[1]
    elif "{ENV_REGEX_NS}" in object_suffix:
        object_suffix = object_suffix.split("{ENV_REGEX_NS}", 1)[1].lstrip("/")
    else:
        object_suffix = object_suffix.lstrip("/")

    for env_prim_path in env.scene.env_prim_paths:
        object_prim_path = f"{env_prim_path}/{object_suffix}" if object_suffix else env_prim_path
        if stage.GetPrimAtPath(object_prim_path).IsValid():
            sim_utils.bind_visual_material(
                object_prim_path,
                material_prim_path,
                stage=stage,
                stronger_than_descendants=True,
            )


def randomize_rigid_body_collider_offsets_relative(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_offset_scale_range: tuple[float, float] = (1.0, 1.0),
    rest_offset_delta_range: tuple[float, float] = (0.0, 0.0),
    min_contact_offset: float = 1.0e-5,
    min_separation: float = 1.0e-5,
) -> None:
    """Randomize collider offsets relative to each collider's existing values.

    IsaacLab's built-in collider-offset event sets absolute offsets. For mixed
    USD assets, preserving each collider's base offset and applying a small
    relative perturbation is safer for grasp-contact semantics.
    """
    asset = env.scene[asset_cfg.name]
    if not isinstance(asset, (RigidObject, Articulation)):
        raise ValueError(
            "randomize_rigid_body_collider_offsets_relative only supports RigidObject or Articulation assets, "
            f"got {type(asset)} for {asset_cfg.name!r}."
        )

    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.long)
    else:
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)

    if env_ids_cpu.numel() == 0:
        return

    contact_low, contact_high = float(contact_offset_scale_range[0]), float(contact_offset_scale_range[1])
    rest_low, rest_high = float(rest_offset_delta_range[0]), float(rest_offset_delta_range[1])
    if contact_low <= 0.0 or contact_high < contact_low:
        raise ValueError(f"Invalid contact_offset_scale_range={contact_offset_scale_range}.")
    if rest_high < rest_low:
        raise ValueError(f"Invalid rest_offset_delta_range={rest_offset_delta_range}.")
    if min_contact_offset <= 0.0:
        raise ValueError(f"min_contact_offset must be positive, got {min_contact_offset}.")
    if min_separation < 0.0:
        raise ValueError(f"min_separation must be non-negative, got {min_separation}.")

    contact_offsets = asset.root_physx_view.get_contact_offsets().clone()
    rest_offsets = asset.root_physx_view.get_rest_offsets().clone()

    contact_selected = contact_offsets[env_ids_cpu]
    rest_selected = rest_offsets[env_ids_cpu]
    contact_scale = torch.empty_like(contact_selected).uniform_(contact_low, contact_high)
    rest_delta = torch.empty_like(rest_selected).uniform_(rest_low, rest_high)

    rest_selected = torch.clamp(rest_selected + rest_delta, min=0.0)
    contact_selected = torch.clamp(contact_selected * contact_scale, min=float(min_contact_offset))
    contact_selected = torch.maximum(contact_selected, rest_selected + float(min_separation))

    rest_offsets[env_ids_cpu] = rest_selected
    contact_offsets[env_ids_cpu] = contact_selected
    asset.root_physx_view.set_rest_offsets(rest_offsets, env_ids_cpu)
    asset.root_physx_view.set_contact_offsets(contact_offsets, env_ids_cpu)


def sample_object_initial_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_range_deg: tuple[float, float] = (-180.0, 180.0),
    pitch_range_deg: tuple[float, float] = (-180.0, 180.0),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the free object around the environment origin with random orientation."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    obj: RigidObject = env.scene[object_cfg.name]
    num_envs = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    object_pos = torch.tensor(position, dtype=torch.float32, device=env.device).unsqueeze(0).expand(num_envs, -1)
    object_pos = object_pos + env_origins

    roll = torch.empty(num_envs, device=env.device).uniform_(
        math.radians(roll_range_deg[0]), math.radians(roll_range_deg[1])
    )
    pitch = torch.empty(num_envs, device=env.device).uniform_(
        math.radians(pitch_range_deg[0]), math.radians(pitch_range_deg[1])
    )
    yaw = torch.empty(num_envs, device=env.device).uniform_(
        math.radians(yaw_range_deg[0]), math.radians(yaw_range_deg[1])
    )
    object_quat = quat_from_euler_xyz(roll, pitch, yaw)

    root_pose = torch.cat([object_pos, object_quat], dim=-1)
    root_velocity = torch.zeros(num_envs, 6, device=env.device)
    obj.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1), env_ids=env_ids)

    if not hasattr(env, "dexgrasp_float_object_anchor_pose_w"):
        env.dexgrasp_float_object_anchor_pose_w = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    env.dexgrasp_float_object_anchor_pose_w[env_ids] = root_pose


def sample_hand_home_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    home_joint_positions: tuple[float, ...] | dict[str, float] = (),
    radius_range: tuple[float, float] = (0.18, 0.24),
    azimuth_range_deg: tuple[float, float] = (-160.0, -140.0),
    elevation_range_deg: tuple[float, float] = (-30.0, -10.0),
    roll_range_deg: tuple[float, float] = (0.0, 0.0),
    position_jitter_std: float = 0.005,
    orientation_jitter_std_deg: float = 4.0,
    hand_transform_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hand_transform_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    min_world_z: float | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample a floating-hand home pose around the local object origin."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = len(env_ids)

    # ── sample palm poses from a spherical shell around the object origin ──
    roll_min_deg, roll_max_deg = roll_range_deg
    if roll_max_deg < roll_min_deg:
        raise ValueError(f"Invalid roll_range_deg: {roll_range_deg}")

    grasp_position = sample_positions_on_spherical_shell(
        num_samples=num_envs,
        device=env.device,
        radius_range=radius_range,
        azimuth_range_deg=azimuth_range_deg,
        elevation_range_deg=elevation_range_deg,
    )

    radial = torch.nn.functional.normalize(grasp_position, dim=-1)
    helper = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(num_envs, 1)
    alternate_helper = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)
    nearly_parallel = torch.abs((radial * helper).sum(dim=-1)) > 0.99
    helper = torch.where(nearly_parallel.unsqueeze(-1), alternate_helper, helper)

    x_axis = torch.nn.functional.normalize(torch.cross(helper, radial, dim=-1), dim=-1)
    y_axis = torch.nn.functional.normalize(torch.cross(radial, x_axis, dim=-1), dim=-1)
    rotation_matrix = torch.stack([x_axis, -y_axis, -radial], dim=-1)
    quat = quat_from_matrix(rotation_matrix)

    grasp_yaw = torch.deg2rad(
        torch.rand(num_envs, device=env.device) * (roll_max_deg - roll_min_deg) + roll_min_deg
    )
    quat = quat_mul(quat, quat_from_euler_xyz(torch.zeros_like(grasp_yaw), torch.zeros_like(grasp_yaw), grasp_yaw))

    grasp_to_palm_pos, grasp_to_palm_quat = grasp_to_palm_transform(
        num_samples=num_envs,
        device=env.device,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
    )
    position, quat = combine_frame_transforms(grasp_position, quat, grasp_to_palm_pos, grasp_to_palm_quat)

    if orientation_jitter_std_deg > 0.0:
        angle_std = math.radians(orientation_jitter_std_deg)
        delta_quat = quat_from_euler_xyz(
            angle_std * torch.randn(num_envs, device=env.device),
            angle_std * torch.randn(num_envs, device=env.device),
            angle_std * torch.randn(num_envs, device=env.device),
        )
        quat = quat_mul(quat, delta_quat)

    if position_jitter_std > 0.0:
        position = position + position_jitter_std * torch.randn_like(position)

    root_pose = torch.cat([position, quat], dim=-1)
    # ── end palm-pose sampling ──

    root_pose = root_pose.clone()
    root_pose[:, 0:3] += env.scene.env_origins[env_ids]
    if min_world_z is not None:
        root_pose[:, 2] = torch.clamp(root_pose[:, 2], min=min_world_z)

    root_velocity = torch.zeros(num_envs, 6, device=env.device)
    robot.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1), env_ids=env_ids)

    if isinstance(home_joint_positions, dict):
        missing = [joint_name for joint_name in robot.joint_names if joint_name not in home_joint_positions]
        if missing:
            raise ValueError(
                "home_joint_positions mapping is missing robot joints: " + ", ".join(missing)
            )
        ordered_home = [float(home_joint_positions[joint_name]) for joint_name in robot.joint_names]
        joint_pos = torch.tensor(ordered_home, dtype=torch.float32, device=env.device).unsqueeze(0).expand(num_envs, -1)
    elif len(home_joint_positions) == 0:
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
    else:
        if len(home_joint_positions) != robot.num_joints:
            raise ValueError(
                f"home_joint_positions must match the robot joint count, "
                f"got {len(home_joint_positions)} vs {robot.num_joints}."
            )
        joint_pos = (
            torch.tensor(home_joint_positions, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .expand(num_envs, -1)
        )
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_data_to_sim()

    if not hasattr(env, "dexgrasp_float_home_pose_w"):
        env.dexgrasp_float_home_pose_w = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    env.dexgrasp_float_home_pose_w[env_ids] = root_pose

    if not hasattr(env, "dexgrasp_float_home_joint_pos"):
        env.dexgrasp_float_home_joint_pos = torch.zeros(
            env.num_envs, robot.num_joints, dtype=torch.float32, device=env.device
        )
    env.dexgrasp_float_home_joint_pos[env_ids] = joint_pos
    # NOTE:
    # We intentionally do not bump a reset-aware cache epoch here. For the current
    # sim2sim migration baseline, we keep the legacy first-step geometry carry-over
    # behavior because it improves transfer stability. See docs for rationale.


def sample_camera_home_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    camera_object_relative_position: tuple[float, float, float] | None = None,
    camera_object_relative_quat: tuple[float, float, float, float] | None = None,
) -> None:
    """Store lightweight per-env virtual student camera state.

    中文说明：
    - 这里不再驱动渲染相机传感器；
    - 只记录每个环境的虚拟外部相机位姿，供 student partial point cloud 在线构造使用。
    """
    if camera_object_relative_position is None or camera_object_relative_quat is None:
        raise ValueError("sample_camera_home_pose requires camera_object_relative_position/quat to be injected at runtime.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    obj: RigidObject = env.scene["object"]
    if not hasattr(env, "dexgrasp_cam_pos_w"):
        env.dexgrasp_cam_pos_w = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    if not hasattr(env, "dexgrasp_cam_quat_w"):
        env.dexgrasp_cam_quat_w = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)

    object_pos_w = obj.data.root_pos_w[env_ids]
    camera_pos_w = (
        torch.tensor(camera_object_relative_position, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(len(env_ids), -1)
        + object_pos_w
    )
    camera_quat_w = (
        torch.tensor(camera_object_relative_quat, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(len(env_ids), -1)
    )
    env.dexgrasp_cam_pos_w[env_ids] = camera_pos_w
    env.dexgrasp_cam_quat_w[env_ids] = math_utils.quat_unique(camera_quat_w)


def sample_hand_home_pose_sim2sim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    home_joint_positions: tuple[float, ...] | dict[str, float] = (),
    hand_object_relative_position: tuple[float, float, float] | None = None,
    hand_object_relative_quat: tuple[float, float, float, float] | None = None,
    position_jitter_std: float = 0.005,
    orientation_jitter_std_deg: float = 4.0,
    min_world_z: float | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Set a sim2sim-specific fixed hand home pose with world-aligned orientation.

    The sim2sim home pose is anchored at the object's initial position, but its
    orientation is interpreted directly in the world frame rather than composed
    with the object's current quaternion. This matches the workflow where the
    object starts at the world origin and the measured ``T_hand_in_object`` is
    recorded against that world-aligned initialization frame.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if hand_object_relative_position is None or hand_object_relative_quat is None:
        raise ValueError(
            "sample_hand_home_pose_sim2sim requires hand_object_relative_position/quat to be injected at runtime."
        )
    if len(hand_object_relative_position) != 3:
        raise ValueError(
            f"hand_object_relative_position must have length 3, got {len(hand_object_relative_position)}."
        )
    if len(hand_object_relative_quat) != 4:
        raise ValueError(
            f"hand_object_relative_quat must have length 4, got {len(hand_object_relative_quat)}."
        )

    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = len(env_ids)
    if hasattr(env, "dexgrasp_float_object_anchor_pose_w"):
        object_anchor_pose_w = env.dexgrasp_float_object_anchor_pose_w[env_ids]
    else:
        obj: RigidObject = env.scene[object_cfg.name]
        object_anchor_pos_w = obj.data.default_root_state[env_ids, :3] + env.scene.env_origins[env_ids]
        object_anchor_quat_w = obj.data.default_root_state[env_ids, 3:7]
        object_anchor_pose_w = torch.cat([object_anchor_pos_w, object_anchor_quat_w], dim=-1)

    relative_pos = (
        torch.tensor(hand_object_relative_position, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )
    relative_quat = (
        torch.tensor(hand_object_relative_quat, dtype=torch.float32, device=env.device)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )
    position = object_anchor_pose_w[:, :3] + relative_pos
    quat = relative_quat.clone()

    if orientation_jitter_std_deg > 0.0:
        angle_std = math.radians(orientation_jitter_std_deg)
        delta_quat = quat_from_euler_xyz(
            angle_std * torch.randn(num_envs, device=env.device),
            angle_std * torch.randn(num_envs, device=env.device),
            angle_std * torch.randn(num_envs, device=env.device),
        )
        quat = quat_mul(quat, delta_quat)

    if position_jitter_std > 0.0:
        position = position + position_jitter_std * torch.randn_like(position)

    root_pose = torch.cat([position, quat], dim=-1)
    if min_world_z is not None:
        root_pose[:, 2] = torch.clamp(root_pose[:, 2], min=min_world_z)

    root_velocity = torch.zeros(num_envs, 6, device=env.device)
    robot.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1), env_ids=env_ids)

    if isinstance(home_joint_positions, dict):
        missing = [joint_name for joint_name in robot.joint_names if joint_name not in home_joint_positions]
        if missing:
            raise ValueError(
                "home_joint_positions mapping is missing robot joints: " + ", ".join(missing)
            )
        ordered_home = [float(home_joint_positions[joint_name]) for joint_name in robot.joint_names]
        joint_pos = torch.tensor(ordered_home, dtype=torch.float32, device=env.device).unsqueeze(0).expand(num_envs, -1)
    elif len(home_joint_positions) == 0:
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
    else:
        if len(home_joint_positions) != robot.num_joints:
            raise ValueError(
                f"home_joint_positions must match the robot joint count, "
                f"got {len(home_joint_positions)} vs {robot.num_joints}."
            )
        joint_pos = (
            torch.tensor(home_joint_positions, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .expand(num_envs, -1)
        )
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    robot.write_data_to_sim()

    if not hasattr(env, "dexgrasp_float_home_pose_w"):
        env.dexgrasp_float_home_pose_w = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    env.dexgrasp_float_home_pose_w[env_ids] = root_pose

    if not hasattr(env, "dexgrasp_float_home_joint_pos"):
        env.dexgrasp_float_home_joint_pos = torch.zeros(
            env.num_envs, robot.num_joints, dtype=torch.float32, device=env.device
        )
    env.dexgrasp_float_home_joint_pos[env_ids] = joint_pos
    # NOTE:
    # We intentionally do not bump a reset-aware cache epoch here. For the current
    # sim2sim migration baseline, we keep the legacy first-step geometry carry-over
    # behavior because it improves transfer stability. See docs for rationale.
