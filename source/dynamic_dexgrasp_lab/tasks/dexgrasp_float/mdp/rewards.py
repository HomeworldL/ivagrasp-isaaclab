"""Reward helpers for dexgrasp_float."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from dynamic_dexgrasp_lab.utils import quat_apply

from .metrics import (
    _phase_mask,
    contact_count,
    finger_contact_count,
    hand_episode_pos_success_mask,
    hand_episode_rot_success_mask,
    hand_episode_success_mask,
    hand_goal_errors,
    hand_grasp_pose_w,
    hand_object_geo_features,
    hand_target_grasp_pose_w,
    obj_episode_pos_success_mask,
    obj_episode_rot_success_mask,
    obj_episode_success_mask,
    object_goal_errors,
)


def _normalize_last_dim(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)
    return tensor / torch.clamp(norm, min=eps)


def _reward_phase_mask(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Return whether the reward term is active for each environment this step."""
    active_phase = active_phase.lower()
    if active_phase == "both" or phase_switch_time_s <= 0.0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return_phase = _phase_mask(env, phase_switch_time_s, step_offset=-1)
    if active_phase == "grasp":
        return ~return_phase
    if active_phase == "return":
        return return_phase
    raise ValueError(f"Unsupported active_phase {active_phase!r}. Expected one of: 'grasp', 'return', 'both'.")


def _contact_gate(
    env: ManagerBasedRLEnv,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 3,
    force_threshold: float = 0.05,
) -> torch.Tensor:
    if min_contact_fingers <= 0:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    count = finger_contact_count(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        force_threshold=force_threshold,
        finger_contact_mode=finger_contact_mode,
    )
    return count >= float(min_contact_fingers)


def _contact_count_bonus(
    env: ManagerBasedRLEnv,
    *,
    enable_contact_count_bonus: bool,
    contact_sensor_groups: Sequence[Sequence[str]],
    max_contact_count: float = 10.0,
    force_threshold: float = 0.05,
) -> torch.Tensor:
    """Return extra full-hand contact bonus in [0, 1]."""
    if not enable_contact_count_bonus:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if max_contact_count <= 0.0:
        raise ValueError(f"max_contact_count must be positive, got {max_contact_count}.")
    if len(contact_sensor_groups) == 0:
        raise ValueError("contact_sensor_groups must not be empty when enable_contact_count_bonus=True.")
    flat_sensor_names: list[str] = []
    for sensor_group in contact_sensor_groups:
        for sensor_name in sensor_group:
            flat_sensor_names.append(sensor_name)
    unique_sensor_names = tuple(dict.fromkeys(flat_sensor_names))
    if len(unique_sensor_names) == 0:
        raise ValueError("contact_sensor_groups resolved to zero sensor names.")
    count = contact_count(env, unique_sensor_names, force_threshold=force_threshold)
    return torch.clamp(count / float(max_contact_count), min=0.0, max=1.0)


def contact_count_bonus_reward(
    env: ManagerBasedRLEnv,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    enable_contact_count_bonus: bool = False,
    max_contact_count: float = 10.0,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Reward normalized contact count behind the same contact gate used by other terms."""
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    bonus = _contact_count_bonus(
        env,
        enable_contact_count_bonus=enable_contact_count_bonus,
        contact_sensor_groups=contact_sensor_groups,
        max_contact_count=max_contact_count,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(gate & phase_gate, bonus, torch.zeros_like(bonus))


def hand_obj_approach_penalty(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize distance from the current grasp frame to the zero-standoff target grasp frame."""
    grasp_pos_w, _ = hand_grasp_pose_w(
        env=env,
        hand_entity_name=robot_cfg.name,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
    )
    target_grasp_pos_w, _ = hand_target_grasp_pose_w(
        env=env,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
        num_points=int(getattr(env.cfg, "geometry_num_points", 0)),
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        target_pose_source=env.cfg.target_pose_source_reward,
    )
    penalty = torch.linalg.norm(target_grasp_pos_w - grasp_pos_w, dim=-1)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def hand_obj_approach_direction_penalty(
    env: ManagerBasedRLEnv,
    palm_body_name: str,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize current grasp-frame +Z misalignment with the zero-standoff target grasp frame +Z."""
    _, grasp_quat_w = hand_grasp_pose_w(
        env=env,
        hand_entity_name=robot_cfg.name,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
    )
    _, target_grasp_quat_w = hand_target_grasp_pose_w(
        env=env,
        palm_body_name=palm_body_name,
        hand_transform_pos=hand_transform_pos,
        hand_transform_quat=hand_transform_quat,
        num_points=int(getattr(env.cfg, "geometry_num_points", 0)),
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        target_pose_source=env.cfg.target_pose_source_reward,
    )
    local_z = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    grasp_z_axis_w = _normalize_last_dim(quat_apply(grasp_quat_w, local_z))
    target_z_axis_w = _normalize_last_dim(quat_apply(target_grasp_quat_w, local_z))
    alignment = torch.sum(grasp_z_axis_w * target_z_axis_w, dim=-1).clamp(min=-1.0, max=1.0)
    penalty = 1.0 - alignment
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def hand_geo_proximity_penalty(
    env: ManagerBasedRLEnv,
    body_names: tuple[str, ...],
    num_points: int,
    body_weights: tuple[float, ...] = (),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize weighted hand-object geometry distance without distance gating."""
    features = hand_object_geo_features(
        env,
        body_names=body_names,
        num_points=num_points,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )
    if features.geo_dist.shape[1] == 0:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    if len(body_weights) not in (0, len(body_names)):
        raise ValueError("body_weights must be empty or match body_names length.")
    if len(body_weights) == 0:
        penalty = torch.mean(features.geo_dist, dim=-1)
    else:
        weights = torch.tensor(body_weights, dtype=torch.float32, device=env.device)
        weights = weights / torch.clamp(torch.sum(weights), min=1e-12)
        penalty = torch.sum(features.geo_dist * weights.unsqueeze(0), dim=-1)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def obj_goal_pos_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    base: float = 1.0,
    slope: float = 2.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Reward goal proximity with optional contact gating."""
    pos_error, _ = object_goal_errors(env, command_name=command_name)
    reward = torch.clamp(base - slope * pos_error, min=0.0)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, reward, torch.zeros_like(reward))
    return reward_part


def obj_goal_rot_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    base: float = 1.0,
    slope: float = 1.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Reward goal orientation alignment with optional contact gating."""
    _, rot_error = object_goal_errors(env, command_name=command_name)
    reward = torch.clamp(base - slope * rot_error, min=0.0)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, reward, torch.zeros_like(reward))
    return reward_part


def hand_goal_pos_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    base: float = 1.0,
    slope: float = 2.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Reward hand-root goal proximity with optional contact gating."""
    pos_error, _ = hand_goal_errors(env, command_name=command_name)
    reward = torch.clamp(base - slope * pos_error, min=0.0)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, reward, torch.zeros_like(reward))
    return reward_part


def hand_goal_rot_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    base: float = 1.0,
    slope: float = 1.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Reward hand-root goal orientation alignment with optional contact gating."""
    _, rot_error = hand_goal_errors(env, command_name=command_name)
    reward = torch.clamp(base - slope * rot_error, min=0.0)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, reward, torch.zeros_like(reward))
    return reward_part


def obj_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_pos: float = 10.0,
    k_rot: float = 3.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for joint object goal success."""
    success = obj_episode_success_mask(env, command_name=command_name)
    pos_error, rot_error = object_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_pos * pos_error + k_rot * rot_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def obj_pos_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_pos: float = 10.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for object position success."""
    success = obj_episode_pos_success_mask(env, command_name=command_name)
    pos_error, _ = object_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_pos * pos_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def hand_success_obj_pos_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_obj_pos: float = 10.0,
    k_hand_pos: float = 10.0,
    k_hand_rot: float = 3.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Sparse bonus for simultaneous object-position and full hand success."""
    command = env.command_manager.get_term(command_name)
    object_at_goal_pos = command.metrics["object_at_goal_pos"].bool()
    hand_at_goal = command.metrics["hand_at_goal"].bool()
    success = object_at_goal_pos & hand_at_goal
    obj_pos_error, _ = object_goal_errors(env, command_name=command_name)
    hand_pos_error, hand_rot_error = hand_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_obj_pos * obj_pos_error + k_hand_pos * hand_pos_error + k_hand_rot * hand_rot_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))


def success_low_dynamics_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    k_obj_pos: float = 10.0,
    k_hand_pos: float = 10.0,
    k_hand_rot: float = 3.0,
    k_obj_lin_speed: float = 2.0,
    k_obj_ang_speed: float = 0.5,
    k_root_lin_speed: float = 2.0,
    k_root_ang_speed: float = 0.5,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Sparse success bonus that continuously favors low root/object residual motion."""
    command = env.command_manager.get_term(command_name)
    object_at_goal_pos = command.metrics["object_at_goal_pos"].bool()
    hand_at_goal = command.metrics["hand_at_goal"].bool()
    success = object_at_goal_pos & hand_at_goal

    obj_pos_error, _ = object_goal_errors(env, command_name=command_name)
    hand_pos_error, hand_rot_error = hand_goal_errors(env, command_name=command_name)
    goal_bonus = 1.0 / (1.0 + k_obj_pos * obj_pos_error + k_hand_pos * hand_pos_error + k_hand_rot * hand_rot_error)

    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    obj_lin_speed = torch.linalg.norm(obj.data.root_link_lin_vel_b, dim=-1)
    obj_ang_speed = torch.linalg.norm(obj.data.root_link_ang_vel_b, dim=-1)
    root_lin_speed = torch.linalg.norm(robot.data.root_link_lin_vel_b, dim=-1)
    root_ang_speed = torch.linalg.norm(robot.data.root_link_ang_vel_b, dim=-1)
    dynamics_bonus = 1.0 / (
        1.0
        + k_obj_lin_speed * obj_lin_speed
        + k_obj_ang_speed * obj_ang_speed
        + k_root_lin_speed * root_lin_speed
        + k_root_ang_speed * root_ang_speed
    )

    contact_gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    gate = contact_gate & success & phase_gate
    return torch.where(gate, goal_bonus * dynamics_bonus, torch.zeros_like(goal_bonus))


def obj_rot_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_rot: float = 3.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for object rotation success."""
    success = obj_episode_rot_success_mask(env, command_name=command_name)
    _, rot_error = object_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_rot * rot_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def hand_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_pos: float = 10.0,
    k_rot: float = 3.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for joint hand-root goal success."""
    success = hand_episode_success_mask(env, command_name=command_name)
    pos_error, rot_error = hand_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_pos * pos_error + k_rot * rot_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def hand_pos_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_pos: float = 10.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for hand-root position success."""
    success = hand_episode_pos_success_mask(env, command_name=command_name)
    pos_error, _ = hand_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_pos * pos_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def hand_rot_success_sparse_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    k_rot: float = 3.0,
    primary_contact_sensor_names: Sequence[str] = (),
    contact_sensor_groups: Sequence[Sequence[str]] = (),
    finger_contact_mode: str = "fingertip",
    min_contact_fingers: int = 0,
    force_threshold: float = 0.05,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Thresholded inverse-distance bonus for hand-root rotation success."""
    success = hand_episode_rot_success_mask(env, command_name=command_name)
    _, rot_error = hand_goal_errors(env, command_name=command_name)
    bonus = 1.0 / (1.0 + k_rot * rot_error)
    gate = _contact_gate(
        env,
        primary_contact_sensor_names=primary_contact_sensor_names,
        contact_sensor_groups=contact_sensor_groups,
        finger_contact_mode=finger_contact_mode,
        min_contact_fingers=min_contact_fingers,
        force_threshold=force_threshold,
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    reward_part = torch.where(gate & phase_gate, success.float() * bonus, torch.zeros_like(bonus))
    return reward_part


def action_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the action magnitude using squared L2 norm with clipping."""
    penalty = torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000.0, 1000.0)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def _action_term_slice(env: ManagerBasedRLEnv, action_term_name: str) -> slice:
    """Return the column slice for a named action term inside ActionManager.action."""
    action_manager = env.action_manager
    start = 0
    for name, dim in zip(action_manager.active_terms, action_manager.action_term_dim, strict=True):
        end = start + dim
        if name == action_term_name:
            return slice(start, end)
        start = end
    raise ValueError(f"Unknown action term {action_term_name!r}. Active terms: {action_manager.active_terms}")


def _term_action_l2_penalty(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    action_slice = _action_term_slice(env, action_term_name)
    penalty = torch.sum(torch.square(env.action_manager.action[:, action_slice]), dim=1).clamp(-1000.0, 1000.0)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def _term_action_rate_l2_penalty(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    action_slice = _action_term_slice(env, action_term_name)
    penalty = torch.sum(
        torch.square(env.action_manager.action[:, action_slice] - env.action_manager.prev_action[:, action_slice]),
        dim=1,
    ).clamp(-1000.0, 1000.0)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def action_rate_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the rate of change of the actions using squared L2 norm with clipping."""
    penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(
        -1000.0, 1000.0
    )
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))


def root_action_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the floating-root action magnitude using squared L2 norm."""
    return _term_action_l2_penalty(
        env, action_term_name="floating_root", phase_switch_time_s=phase_switch_time_s, active_phase=active_phase
    )


def root_action_rate_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the floating-root action rate using squared L2 norm."""
    return _term_action_rate_l2_penalty(
        env, action_term_name="floating_root", phase_switch_time_s=phase_switch_time_s, active_phase=active_phase
    )


def joint_action_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the hand-joint action magnitude using squared L2 norm."""
    return _term_action_l2_penalty(
        env, action_term_name="hand_joints", phase_switch_time_s=phase_switch_time_s, active_phase=active_phase
    )


def joint_action_rate_l2_penalty(
    env: ManagerBasedRLEnv,
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize the hand-joint action rate using squared L2 norm."""
    return _term_action_rate_l2_penalty(
        env, action_term_name="hand_joints", phase_switch_time_s=phase_switch_time_s, active_phase=active_phase
    )


def joint_pos_limits_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    phase_switch_time_s: float = 0.0,
    active_phase: str = "both",
) -> torch.Tensor:
    """Penalize violations against soft joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    lower_error = -(asset.data.joint_pos[:, joint_ids] - asset.data.soft_joint_pos_limits[:, joint_ids, 0]).clip(max=0.0)
    upper_error = (asset.data.joint_pos[:, joint_ids] - asset.data.soft_joint_pos_limits[:, joint_ids, 1]).clip(min=0.0)
    penalty = torch.sum(lower_error + upper_error, dim=1)
    phase_gate = _reward_phase_mask(env, phase_switch_time_s=phase_switch_time_s, active_phase=active_phase)
    return torch.where(phase_gate, penalty, torch.zeros_like(penalty))
