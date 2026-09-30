"""Tests for heterogeneous geometry sampling utilities."""

from __future__ import annotations

import numpy as np
import importlib.util
from pathlib import Path

GEOMETRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dynamic_dexgrasp_lab"
    / "tasks"
    / "dexgrasp_float"
    / "mdp"
    / "geometry.py"
)
spec = importlib.util.spec_from_file_location("geometry_module_for_test", GEOMETRY_PATH)
assert spec is not None and spec.loader is not None
geometry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry_module)
_resample_points_deterministic = geometry_module._resample_points_deterministic


def test_resample_points_is_reproducible_for_same_seed():
    points = np.stack(
        [
            np.linspace(0.0, 1.0, 256, dtype=np.float32),
            np.linspace(1.0, 2.0, 256, dtype=np.float32),
            np.linspace(2.0, 3.0, 256, dtype=np.float32),
        ],
        axis=-1,
    )
    sampled_a = _resample_points_deterministic(points, num_points=128, seed=1234)
    sampled_b = _resample_points_deterministic(points, num_points=128, seed=1234)
    assert np.array_equal(sampled_a, sampled_b)


def test_resample_points_differs_for_different_seed():
    points = np.random.default_rng(0).normal(size=(512, 3)).astype(np.float32)
    sampled_a = _resample_points_deterministic(points, num_points=128, seed=1)
    sampled_b = _resample_points_deterministic(points, num_points=128, seed=2)
    assert not np.array_equal(sampled_a, sampled_b)
