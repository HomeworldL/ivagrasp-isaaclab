"""Tests for object-set resolution helpers."""

import os
from pathlib import Path

import pytest

from dynamic_dexgrasp_lab.assets.data.dataset_loader import (
    ObjectSetQueryCfg,
    resolve_object_selection,
)

RL_SPLIT_DIR = os.environ.get("DEXGRASP_TEST_RL_SPLIT_DIR", "")
POOL_SPLIT_DIR = os.environ.get("DEXGRASP_TEST_POOL_SPLIT_DIR", "")
RL_SPLIT_AVAILABLE = bool(RL_SPLIT_DIR) and Path(RL_SPLIT_DIR).is_dir()
POOL_SPLIT_AVAILABLE = bool(POOL_SPLIT_DIR) and Path(POOL_SPLIT_DIR).is_dir()


@pytest.mark.skipif(not RL_SPLIT_AVAILABLE, reason="set DEXGRASP_TEST_RL_SPLIT_DIR to run this dataset test")
def test_cluster_object_set_resolves_to_non_empty_selection():
    selection = resolve_object_selection(
        ObjectSetQueryCfg(rl_split_dir=RL_SPLIT_DIR, object_set_id="train@cluster@cluster0_topk1@s120_120")
    )

    assert selection.split_side == "train"
    assert selection.object_set_id == "train@cluster@cluster0_topk1@s120_120"
    assert len(selection.active_assets) >= 1


@pytest.mark.skipif(not RL_SPLIT_AVAILABLE, reason="set DEXGRASP_TEST_RL_SPLIT_DIR to run this dataset test")
def test_test_split_with_scale_range_resolves_to_filtered_records():
    selection = resolve_object_selection(
        ObjectSetQueryCfg(rl_split_dir=RL_SPLIT_DIR, object_set_id="test@cluster@clusterALL_topk1@s120_120")
    )

    assert selection.split_side == "test"
    assert len(selection.active_assets) >= 1
    assert all(asset.scale_tag == "scale120" for asset in selection.active_assets)


def test_bare_test_object_set_is_rejected():
    try:
        resolve_object_selection(ObjectSetQueryCfg(object_set_id="test"))
    except ValueError as exc:
        assert "Unsupported object_set_id" in str(exc)
    else:
        raise AssertionError("Expected bare 'test' object_set_id to be rejected.")


@pytest.mark.skipif(not POOL_SPLIT_AVAILABLE, reason="pool split dataset is not available on this machine")
def test_pool_split_subtest_resolves_to_heldout_objects():
    selection = resolve_object_selection(
        ObjectSetQueryCfg(
            rl_split_dir=POOL_SPLIT_DIR,
            object_set_id="subtest@cluster@clusterALL_topkALL@s120_120",
        )
    )

    assert selection.split_side == "subtest"
    assert len(selection.active_assets) == 100
    assert all(asset.scale_tag == "scale120" for asset in selection.active_assets)


@pytest.mark.skipif(not POOL_SPLIT_AVAILABLE, reason="pool split dataset is not available on this machine")
def test_pool_split_subood_resolves_to_ood_clusters():
    selection = resolve_object_selection(
        ObjectSetQueryCfg(
            rl_split_dir=POOL_SPLIT_DIR,
            object_set_id="subood@cluster@clusterALL_topkALL@s120_120",
        )
    )

    assert selection.split_side == "subood"
    assert len(selection.active_assets) == 100
    assert all(asset.cluster_id >= 20 for asset in selection.active_assets)


@pytest.mark.skipif(not POOL_SPLIT_AVAILABLE, reason="pool split dataset is not available on this machine")
def test_pool_split_ood_resolves_to_full_ood_set():
    selection = resolve_object_selection(
        ObjectSetQueryCfg(
            rl_split_dir=POOL_SPLIT_DIR,
            object_set_id="ood@cluster@clusterALL_topkALL@s120_120",
        )
    )

    assert selection.split_side == "ood"
    assert len(selection.active_assets) == 1000
    assert all(asset.cluster_id >= 20 for asset in selection.active_assets)
