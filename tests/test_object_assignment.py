"""Tests for deterministic object-to-env assignment."""

from dynamic_dexgrasp_lab.assets.data.object_assignment import (
    assign_objects_to_envs,
    build_env_usd_path_list,
)


def test_assignment_is_reproducible_for_same_seed():
    object_ids = ("a", "b", "c", "d", "e")
    first = assign_objects_to_envs(object_ids, num_envs=16, seed=7)
    second = assign_objects_to_envs(object_ids, num_envs=16, seed=7)
    assert first == second


def test_assignment_balances_counts_when_envs_exceed_objects():
    object_ids = ("a", "b", "c")
    assignments = assign_objects_to_envs(object_ids, num_envs=10, seed=0)
    counts = [assignments.count(i) for i in range(len(object_ids))]
    assert max(counts) - min(counts) <= 1


def test_assignment_uses_distinct_prefix_when_objects_exceed_envs():
    object_ids = tuple(f"obj_{i}" for i in range(20))
    assignments = assign_objects_to_envs(object_ids, num_envs=8, seed=3)
    assert len(set(assignments)) == 8


def test_build_env_usd_path_list_maps_indices_consistently():
    paths = ("/tmp/a.usd", "/tmp/b.usd", "/tmp/c.usd")
    env_paths = build_env_usd_path_list(paths, num_envs=7, seed=2)
    assert len(env_paths) == 7
    assert all(path in paths for path in env_paths)
