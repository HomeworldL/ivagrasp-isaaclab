"""Sampling helpers for reference-grasp ranking and object-goal sampling."""

from __future__ import annotations

import math

import numpy as np
import torch

from dynamic_dexgrasp_lab.utils import quat_from_euler_xyz


def _quat_angle_distance(
    reference_quat: np.ndarray,
    query_quat: np.ndarray,
) -> np.ndarray:
    dots = np.sum(reference_quat * query_quat[None, :], axis=-1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)
    return 2.0 * np.arccos(dots)


def rank_reference_grasps_by_home_pose(
    reference_root_pose_world: np.ndarray,
    home_root_pose_world: np.ndarray,
    rotation_weight: float = 0.15,
) -> np.ndarray:
    """Rank reference grasps by distance to the current home pose."""
    if reference_root_pose_world.ndim != 2 or reference_root_pose_world.shape[1] != 7:
        raise ValueError(
            "Expected reference_root_pose_world shape (N, 7), "
            f"got {reference_root_pose_world.shape}"
        )
    if home_root_pose_world.shape != (7,):
        raise ValueError(
            f"Expected home_root_pose_world shape (7,), got {home_root_pose_world.shape}"
        )

    pos_distance = np.linalg.norm(
        reference_root_pose_world[:, 0:3] - home_root_pose_world[None, 0:3],
        axis=-1,
    )
    rot_distance = _quat_angle_distance(
        reference_root_pose_world[:, 3:7],
        home_root_pose_world[3:7],
    )
    score = pos_distance + rotation_weight * rot_distance
    return np.argsort(score)


def sample_reference_index_from_top_k(
    ranked_indices: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    """Sample one reference index uniformly from the top-k ranked candidates."""
    if ranked_indices.ndim != 1 or ranked_indices.size == 0:
        raise ValueError("ranked_indices must be a non-empty 1D array.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    k = min(top_k, ranked_indices.size)
    topk = ranked_indices[:k]
    sampled_offset = int(rng.integers(0, k))
    return int(topk[sampled_offset])


def sample_positions_on_spherical_shell(
    num_samples: int,
    device: str | torch.device,
    radius_range: tuple[float, float],
    azimuth_range_deg: tuple[float, float],
    elevation_range_deg: tuple[float, float],
) -> torch.Tensor:
    """Sample shell positions with surface-area-uniform spherical coordinates."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    radius_min, radius_max = radius_range
    if radius_min <= 0.0 or radius_max < radius_min:
        raise ValueError(f"Invalid radius_range: {radius_range}")

    azimuth_min_deg, azimuth_max_deg = azimuth_range_deg
    if azimuth_max_deg < azimuth_min_deg:
        raise ValueError(f"Invalid azimuth_range_deg: {azimuth_range_deg}")

    elev_min_deg, elev_max_deg = elevation_range_deg
    if elev_max_deg < elev_min_deg:
        raise ValueError(f"Invalid elevation_range_deg: {elevation_range_deg}")

    radius = (
        torch.rand(num_samples, device=device) * (radius_max - radius_min) + radius_min
    )
    azimuth = torch.deg2rad(
        torch.rand(num_samples, device=device) * (azimuth_max_deg - azimuth_min_deg)
        + azimuth_min_deg
    )

    elev_min_rad = math.radians(elev_min_deg)
    elev_max_rad = math.radians(elev_max_deg)
    sin_elev_min = math.sin(elev_min_rad)
    sin_elev_max = math.sin(elev_max_rad)
    sin_elevation = (
        torch.rand(num_samples, device=device) * (sin_elev_max - sin_elev_min)
        + sin_elev_min
    )
    elevation = torch.asin(sin_elevation.clamp(-1.0, 1.0))

    xy_radius = radius * torch.cos(elevation)
    return torch.stack(
        [
            xy_radius * torch.cos(azimuth),
            xy_radius * torch.sin(azimuth),
            radius * torch.sin(elevation),
        ],
        dim=-1,
    )


def sample_object_goal_pose(
    num_samples: int,
    device: str,
    radius_range: tuple[float, float] = (0.18, 0.22),
    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0),
    elevation_range_deg: tuple[float, float] = (-90.0, 90.0),
    orientation_range_deg: tuple[float, float, float] = (30.0, 30.0, 30.0),
) -> torch.Tensor:
    """Sample goal pose offset in the current object frame."""
    position = sample_positions_on_spherical_shell(
        num_samples=num_samples,
        device=device,
        radius_range=radius_range,
        azimuth_range_deg=azimuth_range_deg,
        elevation_range_deg=elevation_range_deg,
    )

    rot_limits = tuple(math.radians(value) for value in orientation_range_deg)
    roll = (2.0 * torch.rand(num_samples, device=device) - 1.0) * rot_limits[0]
    pitch = (2.0 * torch.rand(num_samples, device=device) - 1.0) * rot_limits[1]
    yaw = (2.0 * torch.rand(num_samples, device=device) - 1.0) * rot_limits[2]
    quat = quat_from_euler_xyz(roll, pitch, yaw)
    return torch.cat([position, quat], dim=-1)
