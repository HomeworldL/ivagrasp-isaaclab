"""Tests for dataset-first geometry point-cloud loading."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_GEOMETRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dynamic_dexgrasp_lab"
    / "tasks"
    / "dexgrasp_float"
    / "mdp"
    / "geometry.py"
)
_GEOMETRY_SPEC = importlib.util.spec_from_file_location("dexgrasp_geometry_test_module", _GEOMETRY_PATH)
assert _GEOMETRY_SPEC is not None and _GEOMETRY_SPEC.loader is not None
geometry = importlib.util.module_from_spec(_GEOMETRY_SPEC)
_GEOMETRY_SPEC.loader.exec_module(geometry)


def test_dataset_or_usd_prefers_dataset_point_cloud(tmp_path, monkeypatch):
    dataset_pc_path = tmp_path / "global_pc.npy"
    dataset_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(dataset_pc_path, dataset_points)

    geometry.clear_surface_point_cache()
    monkeypatch.setattr(
        geometry,
        "obj_pc_resample",
        lambda points, num_points, **kwargs: points[:num_points],
    )

    def _unexpected_stage_sample(**kwargs):
        raise AssertionError("USD fallback should not run when dataset point cloud is available.")

    monkeypatch.setattr(geometry, "_sample_points_from_stage_object", _unexpected_stage_sample)

    fake_env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(
            _env_object_scale_keys=("test-object",),
            _env_object_usd_paths=("/tmp/object.usd",),
            _env_object_dataset_pc_paths=(str(dataset_pc_path),),
            geometry_point_cloud_source="dataset_or_usd",
            seed=0,
        ),
        scene={
            "object": SimpleNamespace(root_physx_view=SimpleNamespace(prim_paths=("/World/Object",))),
        },
    )

    result = geometry.obj_pc_full_sample_b(fake_env, num_points=2)

    assert np.allclose(result.numpy()[0], dataset_points[:2])


def test_dataset_or_usd_falls_back_to_usd_sampling(monkeypatch):
    usd_points = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )

    geometry.clear_surface_point_cache()
    monkeypatch.setattr(
        geometry,
        "obj_pc_resample",
        lambda points, num_points, **kwargs: points[:num_points],
    )
    monkeypatch.setattr(geometry, "_sample_points_from_stage_object", lambda **kwargs: usd_points)

    fake_env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(
            _env_object_scale_keys=("test-object-usd",),
            _env_object_usd_paths=("/tmp/object.usd",),
            _env_object_dataset_pc_paths=(None,),
            geometry_point_cloud_source="dataset_or_usd",
            seed=0,
        ),
        scene={
            "object": SimpleNamespace(root_physx_view=SimpleNamespace(prim_paths=("/World/Object",))),
        },
    )

    result = geometry.obj_pc_full_sample_b(fake_env, num_points=2)

    assert np.allclose(result.numpy()[0], usd_points)


def test_invalid_geometry_point_cloud_source_raises_value_error():
    fake_env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(
            _env_object_scale_keys=("object",),
            _env_object_usd_paths=("/tmp/object.usd",),
            _env_object_dataset_pc_paths=(None,),
            geometry_point_cloud_source="bad_source",
        ),
    )

    with pytest.raises(ValueError, match="Unsupported geometry_point_cloud_source"):
        geometry.obj_pc_full_sample_b(fake_env, num_points=16)
