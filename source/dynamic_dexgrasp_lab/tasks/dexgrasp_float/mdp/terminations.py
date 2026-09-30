"""Termination terms for dexgrasp_float."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .metrics import hand_object_errors, invalid_sim_state_mask, object_goal_errors


def object_too_far(
    env: ManagerBasedRLEnv,
    threshold: float = 0.30,
    command_name: str = "goal",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object drifts too far from the hand."""
    pos_error, _ = hand_object_errors(env, command_name=command_name, robot_cfg=robot_cfg, object_cfg=object_cfg)
    return pos_error >= threshold


def object_goal_too_far(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.30,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object drifts too far from the goal pose."""
    pos_error, _ = object_goal_errors(env, command_name=command_name, object_cfg=object_cfg)
    return pos_error > threshold


def invalid_state(
    env: ManagerBasedRLEnv,
    contact_sensor_names: Sequence[str] = (),
    quat_norm_epsilon: float = 1e-6,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate before invalid simulator state can enter observations."""
    return invalid_sim_state_mask(
        env,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        contact_sensor_names=contact_sensor_names,
        quat_norm_epsilon=quat_norm_epsilon,
    )
