"""Shared task metrics for dexgrasp_float."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from dynamic_dexgrasp_lab.utils import combine_frame_transforms, quat_apply

from .geometry import obj_pc_full_sample_w, obj_pc_part_sample_w


@dataclass(frozen=True)
class HandObjectGeoFeatures:
    """Cached nearest-surface geometry for selected hand bodies."""

    geo_vec_h: torch.Tensor
    geo_vec_w: torch.Tensor
    geo_dist: torch.Tensor
    nearest_points_w: torch.Tensor


def _command_term(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def episode_step_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the per-environment episode step count."""
    return env.episode_length_buf


def episode_elapsed_time_s(env: ManagerBasedRLEnv, step_offset: int = 0) -> torch.Tensor:
    """Return per-environment elapsed episode time in seconds."""
    step_count = torch.clamp(episode_step_count(env).to(dtype=torch.float32) + float(step_offset), min=0.0)
    return step_count * float(env.step_dt)


def _phase_mask(env: ManagerBasedRLEnv, switch_time_s: float, step_offset: int = 0) -> torch.Tensor:
    """Return whether each environment is in the post-switch return phase."""
    if switch_time_s <= 0.0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return episode_elapsed_time_s(env, step_offset=step_offset) >= float(switch_time_s)


def _finite_per_env(tensor: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=-1)


def _valid_quat_per_env(quat: torch.Tensor, norm_epsilon: float) -> torch.Tensor:
    quat_norm = torch.linalg.norm(quat, dim=-1)
    finite = torch.isfinite(quat).all(dim=-1)
    valid = finite & (quat_norm > norm_epsilon)
    return valid.reshape(valid.shape[0], -1).all(dim=-1)


def _episode_flag(
    env: ManagerBasedRLEnv,
    command_name: str,
    field_name: str,
) -> torch.Tensor:
    return getattr(_command_term(env, command_name), field_name)


def _get_robot_body_ids(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, body_names: tuple[str, ...]) -> torch.Tensor:
    if len(body_names) == 0:
        return torch.zeros(0, dtype=torch.long, device=env.device)
    cache_suffix = "__".join(body_names)
    cache_name = f"_dexgrasp_float_{robot_cfg.name}_surface_body_ids_{cache_suffix}"
    if not hasattr(env, cache_name):
        robot: Articulation = env.scene[robot_cfg.name]
        body_ids, _ = robot.find_bodies(body_names, preserve_order=True)
        setattr(env, cache_name, torch.tensor(body_ids, dtype=torch.long, device=env.device))
    return getattr(env, cache_name)


def _sensor_force_w(sensor: ContactSensor, num_envs: int) -> torch.Tensor:
    has_filter = len(getattr(sensor.cfg, "filter_prim_paths_expr", [])) > 0
    if sensor.data.force_matrix_w is not None:
        return torch.nan_to_num(sensor.data.force_matrix_w, nan=0.0).view(num_envs, -1, 3)
    if has_filter:
        raise RuntimeError(
            "Contact sensor is configured with filter_prim_paths_expr, but force_matrix_w is unavailable. "
            "Refusing fallback to net_forces_w because filtered contact semantics cannot be guaranteed."
        )
    if sensor.data.net_forces_w is not None:
        return torch.nan_to_num(sensor.data.net_forces_w, nan=0.0).view(num_envs, -1, 3)
    raise RuntimeError("Contact sensor does not expose force_matrix_w or net_forces_w.")


def object_goal_errors(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object-goal position and orientation errors in world frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    command = _command_term(env, command_name)
    pos_error = torch.linalg.norm(command.object_goal_pose_w[:, :3] - obj.data.root_pos_w, dim=-1)
    rot_error = math_utils.quat_error_magnitude(obj.data.root_quat_w, command.object_goal_pose_w[:, 3:7])
    return pos_error, rot_error


def hand_goal_errors(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hand-root goal position and orientation errors in world frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = _command_term(env, command_name)
    pos_error = torch.linalg.norm(command.hand_goal_pose_w[:, :3] - robot.data.root_pos_w, dim=-1)
    rot_error = math_utils.quat_error_magnitude(robot.data.root_quat_w, command.hand_goal_pose_w[:, 3:7])
    return pos_error, rot_error


def hand_object_errors(
    env: ManagerBasedRLEnv,
    command_name: str = "goal",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return object error relative to the current grasp frame.

    中文说明：
    - 这里用于 hand-object 几何统计和 object_too_far；
    - hand 侧保持 grasp-frame 语义，不切到 approach target；
    - object 侧保持 object root 语义。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    grasp_pos_w, grasp_quat_w = hand_grasp_pose_w(
        env=env,
        hand_entity_name=robot_cfg.name,
        palm_body_name=env.cfg.palm_body_name,
        hand_transform_pos=env.cfg.hand_transform_pos,
        hand_transform_quat=env.cfg.hand_transform_quat,
    )
    pos_error = torch.linalg.norm(obj.data.root_pos_w - grasp_pos_w, dim=-1)
    rot_error = math_utils.quat_error_magnitude(obj.data.root_quat_w, grasp_quat_w)
    return pos_error, rot_error


def obj_episode_pos_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "object_episode_goal_pos_reached").bool()


def obj_episode_rot_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "object_episode_goal_rot_reached").bool()


def obj_episode_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "object_episode_goal_reached").bool()


def hand_episode_pos_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "hand_episode_goal_pos_reached").bool()


def hand_episode_rot_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "hand_episode_goal_rot_reached").bool()


def hand_episode_success_mask(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return _episode_flag(env, command_name, "hand_episode_goal_reached").bool()


def grasp_contact_active(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_contact_fingers: int = 3,
) -> torch.Tensor:
    """Return whether grasp contact is currently considered active."""
    contact_fingers = _command_term(env, command_name).metrics["contact_finger_count"]
    return (contact_fingers >= float(min_contact_fingers)).float()


def hand_object_geo_features(
    env: ManagerBasedRLEnv,
    body_names: tuple[str, ...],
    num_points: int | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> HandObjectGeoFeatures:
    """Return nearest object-surface vectors for the selected robot bodies."""
    if num_points is None:
        num_points = int(getattr(env.cfg, "geometry_num_points", 0))
    if num_points <= 0:
        raise ValueError(f"geometry_num_points must be positive, got {num_points}.")
    if len(body_names) == 0:
        empty = torch.zeros((env.num_envs, 0, 3), dtype=torch.float32, device=env.device)
        empty_dist = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)
        return HandObjectGeoFeatures(
            geo_vec_h=empty,
            geo_vec_w=empty,
            geo_dist=empty_dist,
            nearest_points_w=empty,
        )

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_ids = _get_robot_body_ids(env, robot_cfg, body_names)
    body_pos_w = robot.data.body_pos_w[:, body_ids]
    env_origins = env.scene.env_origins.unsqueeze(1)
    body_pos_env = body_pos_w - env_origins

    cache_name = f"_dexgrasp_float_geo_features_{robot_cfg.name}_{object_cfg.name}_{num_points}_{'__'.join(body_names)}"
    # NOTE:
    # For the current sim2sim baseline we intentionally keep the legacy step-scoped
    # geometry cache behavior. This preserves the historical first-step carry-over
    # that has proven beneficial for transfer, even though it is not reset-aware.
    cache_token = int(getattr(env, "common_step_counter", -1))
    if hasattr(env, cache_name):
        cached_token, cached_features = getattr(env, cache_name)
        if cached_token == cache_token:
            return cached_features

    object_points_w = obj_pc_full_sample_w(env, num_points=num_points)
    object_points_env = object_points_w - env_origins

    distance_matrix = torch.cdist(body_pos_env, object_points_env)
    nearest_indices = torch.argmin(distance_matrix, dim=-1)
    nearest_points_w = torch.gather(
        object_points_w,
        dim=1,
        index=nearest_indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    nearest_points_env = torch.gather(
        object_points_env,
        dim=1,
        index=nearest_indices.unsqueeze(-1).expand(-1, -1, 3),
    )
    geo_vec_w = nearest_points_w - body_pos_w
    geo_vec_env = nearest_points_env - body_pos_env
    repeated_root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, len(body_names), -1)
    geo_vec_h = math_utils.quat_apply_inverse(
        repeated_root_quat.reshape(-1, 4),
        geo_vec_env.reshape(-1, 3),
    ).reshape(env.num_envs, len(body_names), 3)
    geo_dist = torch.linalg.norm(geo_vec_env, dim=-1)

    features = HandObjectGeoFeatures(
        geo_vec_h=geo_vec_h,
        geo_vec_w=geo_vec_w,
        geo_dist=geo_dist,
        nearest_points_w=nearest_points_w,
    )
    setattr(env, cache_name, (cache_token, features))
    return features


def contact_force_w(env: ManagerBasedRLEnv, sensor_names: Sequence[str]) -> torch.Tensor:
    """Return concatenated per-sensor contact forces in world frame."""
    if len(sensor_names) == 0:
        raise ValueError("sensor_names must not be empty.")
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]
        force_w = _sensor_force_w(sensor, env.num_envs)
        forces.append(force_w[:, 0, :])
    return torch.stack(forces, dim=1)


def contact_force_magnitude(env: ManagerBasedRLEnv, sensor_names: Sequence[str]) -> torch.Tensor:
    """Return per-sensor contact-force magnitudes."""
    return torch.linalg.norm(contact_force_w(env, sensor_names), dim=-1)


def contact_count(
    env: ManagerBasedRLEnv,
    sensor_names: Sequence[str],
    force_threshold: float = 0.05,
) -> torch.Tensor:
    """Return the number of sensors currently in meaningful contact."""
    contact_mask = contact_force_magnitude(env, sensor_names) > force_threshold
    return torch.sum(contact_mask.float(), dim=-1)


def contact_group_count(
    env: ManagerBasedRLEnv,
    sensor_groups: Sequence[Sequence[str]],
    force_threshold: float = 0.05,
) -> torch.Tensor:
    """Return active-group count from grouped contact sensors."""
    if len(sensor_groups) == 0:
        raise ValueError("sensor_groups must not be empty.")
    group_masks = []
    for sensor_group in sensor_groups:
        if len(sensor_group) == 0:
            raise ValueError("Each sensor group must contain at least one sensor name.")
        group_force = contact_force_magnitude(env, sensor_group)
        group_masks.append(torch.any(group_force > force_threshold, dim=-1))
    group_mask = torch.stack(group_masks, dim=-1)
    return torch.sum(group_mask.float(), dim=-1)


def finger_contact_count(
    env: ManagerBasedRLEnv,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    force_threshold: float = 0.05,
    finger_contact_mode: str = "fingertip",
) -> torch.Tensor:
    """Return finger-contact count with a configurable per-finger criterion.

    Modes:
        - ``fingertip``: each fingertip sensor corresponds to one finger.
        - ``any_body``: each finger is a sensor group; any sensor in the group activates that finger.
    """
    if finger_contact_mode == "fingertip":
        if len(primary_contact_sensor_names) == 0:
            raise ValueError("primary_contact_sensor_names must not be empty when finger_contact_mode='fingertip'.")
        return contact_count(env, primary_contact_sensor_names, force_threshold=force_threshold)
    if finger_contact_mode == "any_body":
        if len(contact_sensor_groups) == 0:
            raise ValueError("contact_sensor_groups must not be empty when finger_contact_mode='any_body'.")
        return contact_group_count(env, sensor_groups=contact_sensor_groups, force_threshold=force_threshold)
    raise ValueError(f"Unsupported finger_contact_mode: {finger_contact_mode}")


def invalid_sim_state_mask(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_sensor_names: Sequence[str] = (),
    quat_norm_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return per-env invalid-state mask before bad values enter the policy."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    invalid = (
        (~_finite_per_env(robot.data.root_state_w))
        | (~_valid_quat_per_env(robot.data.root_quat_w, norm_epsilon=quat_norm_epsilon))
        | (~_finite_per_env(robot.data.body_state_w))
        | (~_valid_quat_per_env(robot.data.body_quat_w, norm_epsilon=quat_norm_epsilon))
        | (~_finite_per_env(robot.data.joint_pos))
        | (~_finite_per_env(robot.data.joint_vel))
        | (~_finite_per_env(robot.data.applied_torque))
        | (~_finite_per_env(obj.data.root_state_w))
        | (~_valid_quat_per_env(obj.data.root_quat_w, norm_epsilon=quat_norm_epsilon))
        | (~_finite_per_env(obj.data.root_lin_vel_w))
        | (~_finite_per_env(obj.data.root_ang_vel_w))
    )

    for sensor_name in contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if sensor.data.force_matrix_w is not None:
            invalid |= ~_finite_per_env(sensor.data.force_matrix_w)
        elif sensor.data.net_forces_w is not None:
            invalid |= ~_finite_per_env(sensor.data.net_forces_w)

    return invalid


def hand_grasp_pose_w(
    env: ManagerBasedEnv,
    hand_entity_name: str,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """World grasp-frame pose derived from palm body pose and T_palm_grasp."""
    cache_name = f"_{hand_entity_name}_grasp_frame_palm_body_id_{palm_body_name}"
    if not hasattr(env, cache_name):
        hand = env.scene[hand_entity_name]
        body_ids, _ = hand.find_bodies((palm_body_name,), preserve_order=True)
        setattr(env, cache_name, int(body_ids[0]))

    hand = env.scene[hand_entity_name]
    palm_body_id = getattr(env, cache_name)
    palm_pos_w = hand.data.body_link_pos_w[:, palm_body_id, :]
    palm_quat_w = hand.data.body_link_quat_w[:, palm_body_id, :]

    grasp_pos_local = torch.tensor(hand_transform_pos, dtype=torch.float32, device=env.device).unsqueeze(0).expand(
        env.num_envs, -1
    )
    grasp_quat_local = torch.tensor(hand_transform_quat, dtype=torch.float32, device=env.device).unsqueeze(0).expand(
        env.num_envs, -1
    )
    return combine_frame_transforms(palm_pos_w, palm_quat_w, grasp_pos_local, grasp_quat_local)


def build_target_grasp_pose_from_points(
    current_grasp_pos_w: torch.Tensor,
    current_grasp_quat_w: torch.Tensor,
    object_points_w: torch.Tensor | Sequence[torch.Tensor],
    standoff_distance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a grasp/approach target pose from world-frame object points.

    中文说明：
    - 这个函数只关心当前 grasp frame 和世界系物体点云；
    - 不关心这些点云来自真值 canonical geometry 还是观测/感知模块；
    - action/controller 后续应只消费这里产出的 target grasp pose。
    """
    if isinstance(object_points_w, torch.Tensor):
        env_count = object_points_w.shape[0]
        device = object_points_w.device
        centroid_w = object_points_w.mean(dim=1)
    else:
        env_count = len(object_points_w)
        device = current_grasp_pos_w.device
        centroid_w = torch.stack(
            [
                points.mean(dim=0) if points.shape[0] > 0 else current_grasp_pos_w[env_id]
                for env_id, points in enumerate(object_points_w)
            ],
            dim=0,
        )

    raw_z_axis_w = centroid_w - current_grasp_pos_w
    current_z_axis_w = quat_apply(
        current_grasp_quat_w,
        torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=device).unsqueeze(0).expand(env_count, -1),
    )
    z_norm = torch.linalg.norm(raw_z_axis_w, dim=-1, keepdim=True)
    z_axis_w = torch.where(
        z_norm > 1.0e-8,
        raw_z_axis_w / torch.clamp(z_norm, min=1.0e-8),
        current_z_axis_w,
    )

    current_x_axis_w = quat_apply(
        current_grasp_quat_w,
        torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=device).unsqueeze(0).expand(env_count, -1),
    )
    x_proj_w = current_x_axis_w - torch.sum(current_x_axis_w * z_axis_w, dim=-1, keepdim=True) * z_axis_w
    x_proj_norm = torch.linalg.norm(x_proj_w, dim=-1, keepdim=True)

    world_x = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float32, device=device).unsqueeze(0).expand(env_count, -1)
    fallback_x_w = world_x - torch.sum(world_x * z_axis_w, dim=-1, keepdim=True) * z_axis_w
    fallback_x_norm = torch.linalg.norm(fallback_x_w, dim=-1, keepdim=True)
    fallback_x_w = fallback_x_w / torch.clamp(fallback_x_norm, min=1.0e-8)

    x_axis_w = torch.where(
        x_proj_norm > 1.0e-8,
        x_proj_w / torch.clamp(x_proj_norm, min=1.0e-8),
        fallback_x_w,
    )
    y_axis_w = torch.cross(z_axis_w, x_axis_w, dim=-1)
    y_axis_w = y_axis_w / torch.clamp(torch.linalg.norm(y_axis_w, dim=-1, keepdim=True), min=1.0e-8)
    x_axis_w = torch.cross(y_axis_w, z_axis_w, dim=-1)
    x_axis_w = x_axis_w / torch.clamp(torch.linalg.norm(x_axis_w, dim=-1, keepdim=True), min=1.0e-8)

    rot_matrix_w = torch.stack((x_axis_w, y_axis_w, z_axis_w), dim=-1)
    target_quat_w = math_utils.quat_unique(math_utils.quat_from_matrix(rot_matrix_w))
    target_pos_w = centroid_w - standoff_distance * z_axis_w
    return target_pos_w, target_quat_w


def hand_target_grasp_pose_w_teacher(
    env: ManagerBasedEnv,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
    num_points: int,
    standoff_distance: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-only target grasp pose built from canonical geometry truth."""
    current_grasp_pos_w, current_grasp_quat_w = hand_grasp_pose_w(
        env=env,
        hand_entity_name=robot_cfg.name,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
    )

    object_points_w = obj_pc_full_sample_w(env, num_points=num_points)
    target_pos_w, target_quat_w = build_target_grasp_pose_from_points(
        current_grasp_pos_w=current_grasp_pos_w,
        current_grasp_quat_w=current_grasp_quat_w,
        object_points_w=object_points_w,
        standoff_distance=standoff_distance,
    )
    return target_pos_w, target_quat_w


def hand_target_grasp_pose_w_student(
    env: ManagerBasedEnv,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
    num_points: int,
    standoff_distance: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student target grasp pose built from partial point cloud observations."""
    current_grasp_pos_w, current_grasp_quat_w = hand_grasp_pose_w(
        env=env,
        hand_entity_name=robot_cfg.name,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
    )
    object_points_w = obj_pc_part_sample_w(env, num_points=num_points)
    target_pos_w, target_quat_w = build_target_grasp_pose_from_points(
        current_grasp_pos_w=current_grasp_pos_w,
        current_grasp_quat_w=current_grasp_quat_w,
        object_points_w=object_points_w,
        standoff_distance=standoff_distance,
    )
    return target_pos_w, target_quat_w

def hand_target_grasp_pose_w(
    env: ManagerBasedEnv,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
    num_points: int,
    target_pose_source: str,
    standoff_distance: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch target grasp pose to full/partial geometry providers."""
    source = target_pose_source.lower()
    if source == "full":
        return hand_target_grasp_pose_w_teacher(
            env=env,
            palm_body_name=palm_body_name,
            hand_transform_pos=hand_transform_pos,
            hand_transform_quat=hand_transform_quat,
            num_points=num_points,
            standoff_distance=standoff_distance,
            robot_cfg=robot_cfg,
            object_cfg=object_cfg,
        )
    if source == "partial":
        return hand_target_grasp_pose_w_student(
            env=env,
            palm_body_name=palm_body_name,
            hand_transform_pos=hand_transform_pos,
            hand_transform_quat=hand_transform_quat,
            num_points=num_points,
            standoff_distance=standoff_distance,
            robot_cfg=robot_cfg,
        )
    raise ValueError(f"Unsupported target_pose_source {source!r}. Expected 'full' or 'partial'.")
