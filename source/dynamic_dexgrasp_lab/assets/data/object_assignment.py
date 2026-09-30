"""Deterministic object-to-env assignment helpers."""

from __future__ import annotations

from math import gcd


def _coprime_step(num_objects: int, seed: int) -> int:
    """Return a deterministic step size coprime with ``num_objects``."""
    if num_objects <= 0:
        raise ValueError(f"num_objects must be positive, got {num_objects}.")
    if num_objects == 1:
        return 1
    candidate = (2 * (seed % num_objects) + 1) % num_objects
    if candidate == 0:
        candidate = 1
    while gcd(candidate, num_objects) != 1:
        candidate = (candidate + 2) % num_objects
        if candidate == 0:
            candidate = 1
    return candidate


def assign_objects_to_envs(
    object_ids: tuple[str, ...],
    num_envs: int,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Assign object indices to env slots with one deterministic algorithm.

    The assignment uses a modular arithmetic progression:
    ``idx_i = (start + i * step) mod num_objects``.
    ``step`` is chosen to be coprime to ``num_objects`` so the sequence traverses
    all objects before repeating.
    """
    if not object_ids:
        raise ValueError("object_ids must not be empty.")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")

    num_objects = len(object_ids)
    start = seed % num_objects
    step = _coprime_step(num_objects, seed)
    return tuple((start + env_idx * step) % num_objects for env_idx in range(num_envs))


def build_env_usd_path_list(
    object_usd_paths: tuple[str, ...],
    num_envs: int,
    *,
    seed: int = 0,
) -> tuple[str, ...]:
    """Build per-env USD path list in deterministic, reproducible order."""
    assignments = assign_objects_to_envs(object_usd_paths, num_envs, seed=seed)
    return tuple(object_usd_paths[idx] for idx in assignments)
