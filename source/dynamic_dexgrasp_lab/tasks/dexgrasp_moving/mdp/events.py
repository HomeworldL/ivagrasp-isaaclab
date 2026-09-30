"""Moving-task reset events that override the static dexgrasp_float defaults."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .curriculum import current_speed_curriculum_phase, linear_speed_norm_range
from dynamic_dexgrasp_lab.utils import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_linear_velocity_with_curriculum(
    *,
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    linear_velocity_x_range: tuple[float, float],
    linear_velocity_y_range: tuple[float, float],
    linear_velocity_z_range: tuple[float, float],
):
    """Sample linear reset velocities from the configured stage-local speed curriculum."""
    curriculum_cfg = getattr(env.cfg, "speed_curriculum", None)
    if curriculum_cfg is None or not curriculum_cfg.enabled:
        return None

    num_envs = int(env_ids.numel())
    if num_envs == 0:
        return torch.zeros(0, 3, dtype=torch.float32, device=env.device)

    steps_per_iteration = int(curriculum_cfg.steps_per_iteration)
    if steps_per_iteration <= 0:
        raise ValueError(
            "speed_curriculum.enabled=True requires speed_curriculum.steps_per_iteration > 0. "
            "Inject the runner's num_steps_per_env before training."
        )
    stage_max_iterations = int(curriculum_cfg.stage_max_iterations)
    if stage_max_iterations <= 0:
        raise ValueError(
            "speed_curriculum.enabled=True requires speed_curriculum.stage_max_iterations > 0. "
            "Inject the actual training max_iterations before training."
        )

    current_iteration = int(getattr(env, "common_step_counter", 0)) // steps_per_iteration
    phase_index = current_speed_curriculum_phase(
        current_iteration,
        stage_max_iterations=stage_max_iterations,
        phase_iteration_fracs=tuple(float(v) for v in curriculum_cfg.phase_iteration_fracs),
    )
    phase_weights_all = (
        curriculum_cfg.phase_weights_level1,
        curriculum_cfg.phase_weights_level2,
        curriculum_cfg.phase_weights_level3,
    )
    weights = torch.tensor(phase_weights_all[phase_index], dtype=torch.float32, device=env.device)
    weights = weights / torch.sum(weights)
    target_bin_ids = torch.multinomial(weights, num_samples=num_envs, replacement=True)

    linear_bins = tuple(tuple(float(v) for v in pair) for pair in curriculum_cfg.linear_bins)
    min_norm, max_norm = linear_speed_norm_range(
        linear_velocity_x_range,
        linear_velocity_y_range,
        linear_velocity_z_range,
    )
    unreachable_bins = []
    for bin_id, ((lower, upper), weight) in enumerate(zip(linear_bins, weights.tolist(), strict=False)):
        if weight <= 0.0:
            continue
        if upper < min_norm or lower > max_norm:
            unreachable_bins.append((bin_id, lower, upper))
    if unreachable_bins:
        raise ValueError(
            "speed_curriculum configured unreachable linear speed bins for the current reset ranges. "
            f"phase={phase_index}, x_range={linear_velocity_x_range}, y_range={linear_velocity_y_range}, "
            f"z_range={linear_velocity_z_range}, reachable_norm_range=({min_norm:.4f}, {max_norm:.4f}), "
            f"unreachable_bins={unreachable_bins}"
        )

    bin_lowers = torch.tensor([pair[0] for pair in linear_bins], dtype=torch.float32, device=env.device)
    bin_uppers = torch.tensor([pair[1] for pair in linear_bins], dtype=torch.float32, device=env.device)

    lin_vel = torch.zeros(num_envs, 3, dtype=torch.float32, device=env.device)
    unresolved = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    attempt = 0
    max_attempts = 1000
    last_pending_ids = None
    last_sampled = None
    while torch.any(unresolved):
        attempt += 1
        if attempt > max_attempts:
            break
        pending_ids = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        sample_count = int(pending_ids.numel())
        sampled = torch.empty(sample_count, 3, dtype=torch.float32, device=env.device)
        sampled[:, 0].uniform_(linear_velocity_x_range[0], linear_velocity_x_range[1])
        sampled[:, 1].uniform_(linear_velocity_y_range[0], linear_velocity_y_range[1])
        sampled[:, 2].uniform_(linear_velocity_z_range[0], linear_velocity_z_range[1])
        sampled_norm = torch.linalg.norm(sampled, dim=-1)
        last_pending_ids = pending_ids
        last_sampled = sampled

        lower = bin_lowers[target_bin_ids[pending_ids]]
        upper = bin_uppers[target_bin_ids[pending_ids]]
        upper_inclusive = target_bin_ids[pending_ids] == (len(linear_bins) - 1)
        accepted = (sampled_norm >= lower) & (
            (sampled_norm < upper) | (upper_inclusive & (sampled_norm <= upper))
        )
        if torch.any(accepted):
            accepted_ids = pending_ids[accepted]
            lin_vel[accepted_ids] = sampled[accepted]
            unresolved[accepted_ids] = False

    if torch.any(unresolved):
        if last_pending_ids is None or last_sampled is None:
            raise RuntimeError("Curriculum velocity sampling exhausted attempts before producing any samples.")
        remaining_positions = torch.nonzero(unresolved[last_pending_ids], as_tuple=False).squeeze(-1)
        remaining_ids = last_pending_ids[remaining_positions]
        lin_vel[remaining_ids] = last_sampled[remaining_positions]
        unresolved[remaining_ids] = False

    return lin_vel


def sample_object_initial_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    roll_range_deg: tuple[float, float] = (-180.0, 180.0),
    pitch_range_deg: tuple[float, float] = (-180.0, 180.0),
    yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    linear_velocity_x_range: tuple[float, float] = (0.0, 0.45),
    linear_velocity_y_range: tuple[float, float] = (-0.05, 0.05),
    linear_velocity_z_range: tuple[float, float] = (-0.05, 0.05),
    angular_velocity_x_range: tuple[float, float] = (-0.35, 0.35),
    angular_velocity_y_range: tuple[float, float] = (-0.35, 0.35),
    angular_velocity_z_range: tuple[float, float] = (-0.35, 0.35),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the free object with sampled world-frame pose and velocity."""
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

    curriculum_linear_velocity = _sample_linear_velocity_with_curriculum(
        env=env,
        env_ids=env_ids,
        linear_velocity_x_range=linear_velocity_x_range,
        linear_velocity_y_range=linear_velocity_y_range,
        linear_velocity_z_range=linear_velocity_z_range,
    )
    if curriculum_linear_velocity is None:
        lin_vel_x = torch.empty(num_envs, device=env.device).uniform_(
            linear_velocity_x_range[0], linear_velocity_x_range[1]
        )
        lin_vel_y = torch.empty(num_envs, device=env.device).uniform_(
            linear_velocity_y_range[0], linear_velocity_y_range[1]
        )
        lin_vel_z = torch.empty(num_envs, device=env.device).uniform_(
            linear_velocity_z_range[0], linear_velocity_z_range[1]
        )
    else:
        lin_vel_x = curriculum_linear_velocity[:, 0]
        lin_vel_y = curriculum_linear_velocity[:, 1]
        lin_vel_z = curriculum_linear_velocity[:, 2]
    ang_vel_x = torch.empty(num_envs, device=env.device).uniform_(
        angular_velocity_x_range[0], angular_velocity_x_range[1]
    )
    ang_vel_y = torch.empty(num_envs, device=env.device).uniform_(
        angular_velocity_y_range[0], angular_velocity_y_range[1]
    )
    ang_vel_z = torch.empty(num_envs, device=env.device).uniform_(
        angular_velocity_z_range[0], angular_velocity_z_range[1]
    )

    root_pose = torch.cat([object_pos, object_quat], dim=-1)
    root_velocity = torch.stack((lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z), dim=-1)
    obj.write_root_state_to_sim(torch.cat([root_pose, root_velocity], dim=-1), env_ids=env_ids)

    # Keep the same cache key as dexgrasp_float so the shared command/home-pose logic continues to work.
    if not hasattr(env, "dexgrasp_float_object_anchor_pose_w"):
        env.dexgrasp_float_object_anchor_pose_w = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    env.dexgrasp_float_object_anchor_pose_w[env_ids] = root_pose


__all__ = ["sample_object_initial_pose"]
