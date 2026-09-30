"""Thin moving-task wrappers over the current dexgrasp_float env config."""

from __future__ import annotations

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_moving import mdp as moving_mdp
from dynamic_dexgrasp_lab.tasks.dexgrasp_float import dexgrasp_float_env_cfg as float_env_cfg

DEFAULT_OBJECT_RESET_DISTANCE = 0.66
DEFAULT_OBJECT_GOAL_RESET_DISTANCE = 0.66

DexGraspHandProfile = float_env_cfg.DexGraspHandProfile
DistillObservationsCfg = float_env_cfg.DistillObservationsCfg


@configclass
class LinearSpeedCurriculumCfg:
    """Stage-local linear-speed curriculum for moving-object resets."""

    enabled: bool = False
    steps_per_iteration: int = 0
    stage_max_iterations: int = 0
    phase_iteration_fracs: tuple[float, float, float] = (0.3, 0.3, 0.4)
    linear_velocity_x_range: tuple[float, float] = (0.0, 0.45)
    linear_velocity_y_range: tuple[float, float] = (-0.05, 0.05)
    linear_velocity_z_range: tuple[float, float] = (-0.05, 0.05)
    linear_bins: tuple[tuple[float, float], ...] = (
        (0.0, 0.15),
        (0.15, 0.30),
        (0.30, 0.45),
    )
    phase_weights_level1: tuple[float, float, float] = (0.45, 0.35, 0.20)
    phase_weights_level2: tuple[float, float, float] = (0.35, 0.35, 0.30)
    phase_weights_level3: tuple[float, float, float] = (0.30, 0.35, 0.35)


@float_env_cfg.configclass
class DexGraspMovingEnvCfg(float_env_cfg.DexGraspFloatEnvCfg):
    """Moving-task wrapper over the current floating-hand dexgrasp env config."""

    speed_curriculum: LinearSpeedCurriculumCfg = LinearSpeedCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # self.phase_switch_time_s = 2.0
        # profile = self._require_hand_profile()
        # self._wire_action_terms(profile)
        # self._wire_reward_terms()
        # self._wire_command_terms()
        self.events.sample_object_initial_pose.func = moving_mdp.sample_object_initial_pose
        self.events.sample_object_initial_pose.params.update(
            {
                "position": (0.0, 0.0, 0.0),
                "roll_range_deg": (-180.0, 180.0),
                "pitch_range_deg": (-180.0, 180.0),
                "yaw_range_deg": (-180.0, 180.0),
                "linear_velocity_x_range": self.speed_curriculum.linear_velocity_x_range,
                "linear_velocity_y_range": self.speed_curriculum.linear_velocity_y_range,
                "linear_velocity_z_range": self.speed_curriculum.linear_velocity_z_range,
                "angular_velocity_x_range": (-0.35, 0.35),
                "angular_velocity_y_range": (-0.35, 0.35),
                "angular_velocity_z_range": (-0.35, 0.35),
                "object_cfg": float_env_cfg.SceneEntityCfg("object"),
            }
        )
        self.actions.floating_root.linear_velocity_scale = (1.0, 1.0, 1.0)
        self.actions.floating_root.angular_velocity_scale = (0.8, 0.8, 0.8)
        self.terminations.object_too_far.params["threshold"] = DEFAULT_OBJECT_RESET_DISTANCE
        self.terminations.object_goal_too_far.params["threshold"] = DEFAULT_OBJECT_GOAL_RESET_DISTANCE


__all__ = [
    "DexGraspHandProfile",
    "DistillObservationsCfg",
    "DexGraspMovingEnvCfg",
    "LinearSpeedCurriculumCfg",
    "DEFAULT_OBJECT_RESET_DISTANCE",
    "DEFAULT_OBJECT_GOAL_RESET_DISTANCE",
]
