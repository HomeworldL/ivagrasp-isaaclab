"""Pure-Python helpers for moving-task curricula."""

from __future__ import annotations

import math

def current_speed_curriculum_phase(
    current_iteration: int,
    *,
    stage_max_iterations: int,
    phase_iteration_fracs: tuple[float, float, float],
) -> int:
    """Return the current speed-curriculum phase index in {0,1,2}."""
    if stage_max_iterations <= 0:
        return 2
    frac_0, frac_1, frac_2 = phase_iteration_fracs
    total = frac_0 + frac_1 + frac_2
    if total <= 0.0:
        return 2
    norm_0 = frac_0 / total
    norm_1 = frac_1 / total
    boundary_0 = int(stage_max_iterations * norm_0)
    boundary_1 = int(stage_max_iterations * (norm_0 + norm_1))
    if current_iteration < boundary_0:
        return 0
    if current_iteration < boundary_1:
        return 1
    return 2


def _interval_abs_min(interval: tuple[float, float]) -> float:
    lo, hi = sorted((float(interval[0]), float(interval[1])))
    if lo <= 0.0 <= hi:
        return 0.0
    return min(abs(lo), abs(hi))


def _interval_abs_max(interval: tuple[float, float]) -> float:
    lo, hi = sorted((float(interval[0]), float(interval[1])))
    return max(abs(lo), abs(hi))


def linear_speed_norm_range(
    linear_velocity_x_range: tuple[float, float],
    linear_velocity_y_range: tuple[float, float],
    linear_velocity_z_range: tuple[float, float],
) -> tuple[float, float]:
    """Return the achievable norm interval for an axis-aligned linear-velocity box."""
    min_sq = (
        _interval_abs_min(linear_velocity_x_range) ** 2
        + _interval_abs_min(linear_velocity_y_range) ** 2
        + _interval_abs_min(linear_velocity_z_range) ** 2
    )
    max_sq = (
        _interval_abs_max(linear_velocity_x_range) ** 2
        + _interval_abs_max(linear_velocity_y_range) ** 2
        + _interval_abs_max(linear_velocity_z_range) ** 2
    )
    return math.sqrt(min_sq), math.sqrt(max_sq)
