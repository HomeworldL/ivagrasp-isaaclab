"""Task registration smoke test."""

import gymnasium as gym
import pytest

import dynamic_dexgrasp_lab


try:
    import pxr  # noqa: F401
except ImportError:  # pragma: no cover - integration dependency
    pytestmark = pytest.mark.skip(reason="pxr is required to import IsaacLab task configs.")


def test_allegro_task_is_registered():
    dynamic_dexgrasp_lab.register_tasks()
    spec = gym.spec("Isaac-DexGrasp-Float-Allegro-v0")
    assert spec.id == "Isaac-DexGrasp-Float-Allegro-v0"


def test_liberhand_task_is_registered():
    dynamic_dexgrasp_lab.register_tasks()
    spec = gym.spec("Isaac-DexGrasp-Float-Liberhand-v0")
    assert spec.id == "Isaac-DexGrasp-Float-Liberhand-v0"
