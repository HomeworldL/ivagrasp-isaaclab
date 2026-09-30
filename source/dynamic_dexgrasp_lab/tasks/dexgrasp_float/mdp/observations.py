"""Observation helpers for dexgrasp_float."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .metrics import contact_force_magnitude, hand_object_geo_features


def obj_pose_rel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object pose relative to the current hand root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    env_origins = env.scene.env_origins
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w - env_origins,
        robot.data.root_quat_w,
        obj.data.root_pos_w - env_origins,
        obj.data.root_quat_w,
    )
    rel_quat = math_utils.quat_unique(rel_quat)
    return torch.cat((rel_pos, rel_quat), dim=-1)


def obj_lin_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object root-body linear velocity in the object body frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_link_lin_vel_b


def obj_ang_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object root-body angular velocity in the object body frame."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_link_ang_vel_b


def hand_root_lin_vel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return hand root-body linear velocity in the hand root-body frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_link_lin_vel_b


def hand_root_ang_vel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return hand root-body angular velocity in the hand root-body frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_link_ang_vel_b


def hand_goal_pose_rel(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return goal hand-root pose relative to the current hand root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_term(command_name)
    env_origins = env.scene.env_origins
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w - env_origins,
        robot.data.root_quat_w,
        command.hand_goal_pose_w[:, :3] - env_origins,
        command.hand_goal_pose_w[:, 3:7],
    )
    rel_quat = math_utils.quat_unique(rel_quat)
    return torch.cat((rel_pos, rel_quat), dim=-1)


def hand_contact_force(
    env: ManagerBasedRLEnv,
    contact_sensor_names: Sequence[str],
    normalize_by: float = 500.0,
) -> torch.Tensor:
    """Return normalized per-sensor contact-force magnitudes."""
    if normalize_by <= 0.0:
        raise ValueError("normalize_by must be positive.")
    return contact_force_magnitude(env, contact_sensor_names) / normalize_by


def hand_joint_force(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return applied joint torques for the hand articulation."""
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids = robot_cfg.joint_ids if robot_cfg.joint_ids is not None else slice(None)
    return robot.data.applied_torque[:, joint_ids]


def hand_obj_geo_vec(
    env: ManagerBasedRLEnv,
    body_names: tuple[str, ...],
    num_points: int = 512,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return nearest object-surface vectors for selected hand bodies in hand frame."""
    geo_features = hand_object_geo_features(
        env,
        body_names=body_names,
        num_points=num_points,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    return geo_features.geo_vec_h.reshape(geo_features.geo_vec_h.shape[0], -1)
