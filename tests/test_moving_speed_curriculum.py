from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dynamic_dexgrasp_lab"
    / "tasks"
    / "dexgrasp_moving"
    / "mdp"
    / "curriculum.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("moving_curriculum_test_module", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

current_speed_curriculum_phase = MODULE.current_speed_curriculum_phase
linear_speed_norm_range = MODULE.linear_speed_norm_range


def test_current_speed_curriculum_phase_uses_30_30_40_split() -> None:
    stage_max_iterations = 1000
    fracs = (0.3, 0.3, 0.4)

    assert current_speed_curriculum_phase(0, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 0
    assert current_speed_curriculum_phase(299, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 0
    assert current_speed_curriculum_phase(300, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 1
    assert current_speed_curriculum_phase(599, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 1
    assert current_speed_curriculum_phase(600, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 2
    assert current_speed_curriculum_phase(999, stage_max_iterations=stage_max_iterations, phase_iteration_fracs=fracs) == 2


def test_linear_speed_norm_range_for_positive_x_small_yz_box() -> None:
    min_norm, max_norm = linear_speed_norm_range((0.0, 0.5), (-0.05, 0.05), (-0.05, 0.05))
    assert min_norm == 0.0
    assert abs(max_norm - (0.5**2 + 0.05**2 + 0.05**2) ** 0.5) < 1e-9
