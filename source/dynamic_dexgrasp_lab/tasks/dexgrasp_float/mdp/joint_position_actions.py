"""Custom joint-position action terms for dexgrasp."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import re

import torch

import isaaclab.utils.string as string_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .metrics import _phase_mask, hand_grasp_pose_w, hand_target_grasp_pose_w


class EMAJointPositionOffsetScaleAction(ActionTerm):
    """Joint position target: ``offset + scale * action`` with optional EMA smoothing."""

    cfg: EMAJointPositionOffsetScaleActionCfg

    def __init__(self, cfg: EMAJointPositionOffsetScaleActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._prev_applied_actions = torch.zeros_like(self._raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones((self.num_envs, self._num_joints), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.scale, self._joint_names, preserve_order=self.cfg.preserve_order
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros((self.num_envs, self._num_joints), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.offset, self._joint_names, preserve_order=self.cfg.preserve_order
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        if isinstance(cfg.alpha, (float, int)):
            alpha = float(cfg.alpha)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"EMA alpha must be in [0, 1], got {alpha}.")
            self._alpha = alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((self.num_envs, self._num_joints), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names, preserve_order=self.cfg.preserve_order
            )
            for name, value in zip(names_list, value_list, strict=True):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"EMA alpha for joint {name!r} must be in [0, 1], got {value}.")
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported alpha type: {type(cfg.alpha)}. Supported types are float and dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_applied_actions[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        processed = torch.clamp(actions, -1.0, 1.0)
        targets = processed * self._scale + self._offset
        if self.cfg.clip_to_joint_limits:
            targets = torch.clamp(
                targets,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )

        ema_targets = self._alpha * targets + (1.0 - self._alpha) * self._prev_applied_actions
        if self.cfg.clip_to_joint_limits:
            ema_targets = torch.clamp(
                ema_targets,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )
        self._processed_actions[:] = ema_targets
        self._prev_applied_actions[:] = ema_targets

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class EMAJointPositionOffsetScaleActionCfg(ActionTermCfg):
    """Config for :class:`EMAJointPositionOffsetScaleAction`."""

    class_type: type = EMAJointPositionOffsetScaleAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    alpha: float | dict[str, float] = 1.0
    preserve_order: bool = True
    clip_to_joint_limits: bool = True


class NearGoalRelativeJointPositionAction(ActionTerm):
    """Relative joint target with near-goal residual scaling and delta smoothing."""

    cfg: NearGoalRelativeJointPositionActionCfg

    def __init__(self, cfg: NearGoalRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._filtered_delta = torch.zeros_like(self._raw_actions)
        self._filter_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones((self.num_envs, self._num_joints), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.scale, self._joint_names, preserve_order=self.cfg.preserve_order
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros((self.num_envs, self._num_joints), device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.offset, self._joint_names, preserve_order=self.cfg.preserve_order
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        if self.cfg.use_zero_offset:
            self._offset = 0.0

        self._near_goal_action_scale = float(cfg.near_goal_action_scale)
        self._near_goal_ema_alpha = float(cfg.near_goal_ema_alpha)
        self._near_goal_obj_pos_inner = float(cfg.near_goal_object_pos_inner_threshold)
        self._near_goal_obj_pos_outer = float(cfg.near_goal_object_pos_outer_threshold)
        self._near_goal_hand_pos_inner = float(cfg.near_goal_hand_pos_inner_threshold)
        self._near_goal_hand_pos_outer = float(cfg.near_goal_hand_pos_outer_threshold)
        self._near_goal_hand_rot_inner = float(cfg.near_goal_hand_rot_inner_threshold)
        self._near_goal_hand_rot_outer = float(cfg.near_goal_hand_rot_outer_threshold)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _near_goal_progress(self, error: torch.Tensor, inner: float, outer: float) -> torch.Tensor:
        if outer <= inner:
            return (error > inner).float()
        progress = torch.clamp((error - inner) / (outer - inner), 0.0, 1.0)
        return progress * progress * (3.0 - 2.0 * progress)

    def _compute_near_goal_weight(self) -> torch.Tensor:
        try:
            command = self._env.command_manager.get_term(self.cfg.command_name)
        except Exception:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        metrics = command.metrics
        required_metrics = ("object_goal_pos_error", "hand_goal_pos_error", "hand_goal_rot_error")
        if any(metric_name not in metrics for metric_name in required_metrics):
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        obj_far = self._near_goal_progress(
            metrics["object_goal_pos_error"],
            self._near_goal_obj_pos_inner,
            self._near_goal_obj_pos_outer,
        )
        hand_pos_far = self._near_goal_progress(
            metrics["hand_goal_pos_error"],
            self._near_goal_hand_pos_inner,
            self._near_goal_hand_pos_outer,
        )
        hand_rot_far = self._near_goal_progress(
            metrics["hand_goal_rot_error"],
            self._near_goal_hand_rot_inner,
            self._near_goal_hand_rot_outer,
        )
        near_weight = 1.0 - torch.maximum(torch.maximum(obj_far, hand_pos_far), hand_rot_far)
        contact_gate = metrics.get("grasp_contact_active", None)
        if contact_gate is not None:
            near_weight = torch.where(contact_gate.bool(), near_weight, torch.zeros_like(near_weight))
        return near_weight

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._filtered_delta[env_ids] = 0.0
        self._filter_initialized[env_ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        processed = torch.clamp(actions, -1.0, 1.0)
        delta = processed * self._scale + self._offset

        near_weight = self._compute_near_goal_weight().unsqueeze(-1)
        action_scale = self._near_goal_action_scale + (1.0 - self._near_goal_action_scale) * (1.0 - near_weight)
        delta = delta * action_scale

        init_mask = ~self._filter_initialized
        if torch.any(init_mask):
            self._filtered_delta[init_mask] = delta[init_mask]
            self._filter_initialized[init_mask] = True
        keep = self._near_goal_ema_alpha * near_weight
        delta = keep * self._filtered_delta + (1.0 - keep) * delta
        self._filtered_delta[:] = delta
        self._processed_actions[:] = delta

    def apply_actions(self) -> None:
        targets = self._asset.data.joint_pos[:, self._joint_ids] + self._processed_actions
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class NearGoalRelativeJointPositionActionCfg(ActionTermCfg):
    """Config for :class:`NearGoalRelativeJointPositionAction`."""

    class_type: type = NearGoalRelativeJointPositionAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = True
    use_zero_offset: bool = True
    command_name: str = "goal"
    near_goal_action_scale: float = 0.5
    near_goal_ema_alpha: float = 0.8
    near_goal_object_pos_inner_threshold: float = 0.05
    near_goal_object_pos_outer_threshold: float = 0.10
    near_goal_hand_pos_inner_threshold: float = 0.05
    near_goal_hand_pos_outer_threshold: float = 0.10
    near_goal_hand_rot_inner_threshold: float = 0.3490658503988659
    near_goal_hand_rot_outer_threshold: float = 0.6981317007977318


class DistanceGatedRelativeJointPositionAction(NearGoalRelativeJointPositionAction):
    """Relative joint action with approach-distance release and home-pose leakage.

    This keeps the relative-action interface intact. While the selected hand
    bodies are far from the object surface, policy deltas are attenuated and a
    small delta toward the configured home pose is injected. Near the object or
    after contact, the action smoothly recovers the base relative-joint
    semantics inherited from :class:`NearGoalRelativeJointPositionAction`.
    """

    cfg: DistanceGatedRelativeJointPositionActionCfg

    def __init__(self, cfg: DistanceGatedRelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)

        self._standoff_distance = float(cfg.standoff_distance)
        self._approach_num_points = int(cfg.approach_num_points)
        self._approach_release_inner = float(cfg.approach_release_inner_distance)
        self._approach_release_outer = float(cfg.approach_release_outer_distance)
        self._approach_min_action_scale = float(cfg.approach_min_action_scale)
        self._approach_home_leak = float(cfg.approach_home_leak)
        if not 0.0 <= self._approach_min_action_scale <= 1.0:
            raise ValueError(
                f"approach_min_action_scale must be in [0, 1], got {self._approach_min_action_scale}."
            )
        if self._approach_home_leak < 0.0:
            raise ValueError(f"approach_home_leak must be non-negative, got {self._approach_home_leak}.")
        if self._approach_num_points <= 0:
            raise ValueError(f"approach_num_points must be positive, got {self._approach_num_points}.")

        self._home_joint_pos = self._resolve_home_joint_positions(cfg.home_joint_positions)
        self._last_approach_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_approach_release_weight = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

    def _resolve_home_joint_positions(self, values: float | dict[str, float]) -> torch.Tensor:
        if isinstance(values, (float, int)):
            return torch.full((self.num_envs, self._num_joints), float(values), dtype=torch.float32, device=self.device)
        if isinstance(values, dict):
            resolved = torch.zeros((self.num_envs, self._num_joints), dtype=torch.float32, device=self.device)
            missing_joint_names = []
            for joint_index, joint_name in enumerate(self._joint_names):
                matches = [
                    float(home_value)
                    for home_pattern, home_value in values.items()
                    if re.fullmatch(home_pattern, joint_name) is not None
                ]
                if len(matches) == 0:
                    missing_joint_names.append(joint_name)
                    continue
                if len(matches) > 1:
                    raise ValueError(
                        f"Multiple home_joint_positions entries match controlled joint {joint_name!r}."
                    )
                resolved[:, joint_index] = matches[0]
            if missing_joint_names:
                raise ValueError(
                    "Missing home_joint_positions entries for controlled joints: "
                    + ", ".join(missing_joint_names)
                )
            return resolved
        raise ValueError(
            f"Unsupported home_joint_positions type: {type(values)}. Supported types are float and dict."
        )

    def _compute_approach_release_weight(self) -> torch.Tensor:
        grasp_pos_w, _ = hand_grasp_pose_w(
            env=self._env,
            hand_entity_name=self.cfg.asset_name,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
        )
        target_grasp_pos_w, _ = hand_target_grasp_pose_w(
            env=self._env,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
            num_points=self._approach_num_points,
            standoff_distance=self._standoff_distance,
            target_pose_source=self._env.cfg.target_pose_source_action,
        )
        distance = torch.linalg.norm(target_grasp_pos_w - grasp_pos_w, dim=-1)
        self._last_approach_distance[:] = distance

        if self._approach_release_outer <= self._approach_release_inner:
            release = (distance <= self._approach_release_inner).float()
        else:
            release = torch.clamp(
                (self._approach_release_outer - distance)
                / (self._approach_release_outer - self._approach_release_inner),
                0.0,
                1.0,
            )
            release = release * release * (3.0 - 2.0 * release)

        self._last_approach_release_weight[:] = release
        return release

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._approach_home_leak == 0.0 and self._approach_min_action_scale >= 1.0:
            return

        release = self._compute_approach_release_weight().unsqueeze(-1)
        action_scale = self._approach_min_action_scale + (1.0 - self._approach_min_action_scale) * release

        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        home_delta = self._approach_home_leak * (self._home_joint_pos - current_joint_pos)
        self._processed_actions[:] = action_scale * self._processed_actions + (1.0 - release) * home_delta


@configclass
class DistanceGatedRelativeJointPositionActionCfg(NearGoalRelativeJointPositionActionCfg):
    """Config for :class:`DistanceGatedRelativeJointPositionAction`."""

    class_type: type = DistanceGatedRelativeJointPositionAction

    home_joint_positions: float | dict[str, float] = MISSING
    object_name: str = "object"
    hand_transform_pos: tuple[float, float, float] = MISSING
    hand_transform_quat: tuple[float, float, float, float] = MISSING
    standoff_distance: float = 0.0
    approach_num_points: int = MISSING
    approach_release_inner_distance: float = 0.08
    approach_release_outer_distance: float = 0.18
    approach_min_action_scale: float = 0.0
    approach_home_leak: float = 0.0


class DistanceGatedRelativeJointPositionSim2SimAction(RelativeJointPositionAction):
    """Relative joint action with distance gating and configurable return behavior.

    This matches IsaacLab's RelativeJointPositionAction semantics:
    ``target = current_joint_pos_at_policy_step + processed_delta``.
    The only extra behavior is an approach-distance gate that can attenuate the
    policy delta and inject a small delta toward the configured home pose while
    the grasp frame is far from the object. After the configured switch time,
    the action transitions according to ``return_mode``. The default ``hold``
    mode freezes the switch-time command target; residual return modes are kept
    as ablation options. The final absolute command target is assembled in
    ``process_actions()`` at the policy rate; ``apply_actions()`` only sends the
    latched target to the articulation.
    """

    cfg: DistanceGatedRelativeJointPositionSim2SimActionCfg
    RETURN_MODE_HOLD = "hold"
    RETURN_MODE_CONTINUOUS = "continuous"
    RETURN_MODE_CURRENT_RESIDUAL = "current_residual"
    RETURN_MODE_LATCHED_RESIDUAL = "latched_residual"
    RETURN_MODES = {
        RETURN_MODE_HOLD,
        RETURN_MODE_CONTINUOUS,
        RETURN_MODE_CURRENT_RESIDUAL,
        RETURN_MODE_LATCHED_RESIDUAL,
    }

    def __init__(self, cfg: DistanceGatedRelativeJointPositionSim2SimActionCfg, env):
        super().__init__(cfg, env)

        self._standoff_distance = float(cfg.standoff_distance)
        self._approach_num_points = int(cfg.approach_num_points)
        self._approach_release_inner = float(cfg.approach_release_inner_distance)
        self._approach_release_outer = float(cfg.approach_release_outer_distance)
        self._approach_min_action_scale = float(cfg.approach_min_action_scale)
        self._approach_home_leak = float(cfg.approach_home_leak)
        self._phase_switch_time_s = float(cfg.phase_switch_time_s)
        self._return_residual_scale = float(cfg.return_residual_scale)
        self._return_residual_clip = float(cfg.return_residual_clip)
        self._return_mode = str(cfg.return_mode)
        if not 0.0 <= self._approach_min_action_scale <= 1.0:
            raise ValueError(
                f"approach_min_action_scale must be in [0, 1], got {self._approach_min_action_scale}."
            )
        if self._approach_home_leak < 0.0:
            raise ValueError(f"approach_home_leak must be non-negative, got {self._approach_home_leak}.")
        if self._return_residual_scale < 0.0:
            raise ValueError(f"return_residual_scale must be non-negative, got {self._return_residual_scale}.")
        if self._return_residual_clip < 0.0:
            raise ValueError(f"return_residual_clip must be non-negative, got {self._return_residual_clip}.")
        if self._approach_num_points <= 0:
            raise ValueError(f"approach_num_points must be positive, got {self._approach_num_points}.")
        if self._return_mode not in self.RETURN_MODES:
            raise ValueError(
                f"Unsupported return_mode={self._return_mode!r}. "
                f"Supported modes are: {sorted(self.RETURN_MODES)}."
            )

        self._home_joint_pos = self._resolve_home_joint_positions(cfg.home_joint_positions)
        self._last_approach_distance = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_approach_release_weight = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_joint_targets = torch.zeros((self.num_envs, self._num_joints), dtype=torch.float32, device=self.device)
        self._return_hold_joint_targets = torch.zeros_like(self._last_joint_targets)
        self._command_joint_targets = torch.zeros_like(self._last_joint_targets)
        self._return_phase_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    @property
    def return_mode(self) -> str:
        """Return-phase control mode used by the joint action term."""
        return self._return_mode

    def _resolve_home_joint_positions(self, values: float | dict[str, float]) -> torch.Tensor:
        if isinstance(values, (float, int)):
            return torch.full((self.num_envs, self._num_joints), float(values), dtype=torch.float32, device=self.device)
        if isinstance(values, dict):
            resolved = torch.zeros((self.num_envs, self._num_joints), dtype=torch.float32, device=self.device)
            missing_joint_names = []
            for joint_index, joint_name in enumerate(self._joint_names):
                matches = [
                    float(home_value)
                    for home_pattern, home_value in values.items()
                    if re.fullmatch(home_pattern, joint_name) is not None
                ]
                if len(matches) == 0:
                    missing_joint_names.append(joint_name)
                    continue
                if len(matches) > 1:
                    raise ValueError(
                        f"Multiple home_joint_positions entries match controlled joint {joint_name!r}."
                    )
                resolved[:, joint_index] = matches[0]
            if missing_joint_names:
                raise ValueError(
                    "Missing home_joint_positions entries for controlled joints: "
                    + ", ".join(missing_joint_names)
                )
            return resolved
        raise ValueError(
            f"Unsupported home_joint_positions type: {type(values)}. Supported types are float and dict."
        )

    def _compute_approach_release_weight_and_distance(
        self,
        *,
        target_pose_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grasp_pos_w, _ = hand_grasp_pose_w(
            env=self._env,
            hand_entity_name=self.cfg.asset_name,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
        )
        target_grasp_pos_w, _ = hand_target_grasp_pose_w(
            env=self._env,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
            num_points=self._approach_num_points,
            standoff_distance=self._standoff_distance,
            target_pose_source=target_pose_source,
        )
        distance = torch.linalg.norm(target_grasp_pos_w - grasp_pos_w, dim=-1)

        if self._approach_release_outer <= self._approach_release_inner:
            release = (distance <= self._approach_release_inner).float()
        else:
            release = torch.clamp(
                (self._approach_release_outer - distance)
                / (self._approach_release_outer - self._approach_release_inner),
                0.0,
                1.0,
            )
            release = release * release * (3.0 - 2.0 * release)
        return release, distance

    def _compute_approach_release_weight(self) -> torch.Tensor:
        release, distance = self._compute_approach_release_weight_and_distance(
            target_pose_source=self._env.cfg.target_pose_source_action,
        )
        self._last_approach_distance[:] = distance
        self._last_approach_release_weight[:] = release
        return release

    def compute_grasp_command_target_for_label(
        self,
        actions: torch.Tensor,
        *,
        target_pose_source: str,
    ) -> torch.Tensor:
        """Compute the grasp-phase absolute command target without return-mode state."""
        release, home_delta, current_joint_pos = self.compute_grasp_command_target_context_for_label(
            target_pose_source=target_pose_source,
        )
        return self.compute_grasp_command_target_from_context_for_loss(
            actions,
            release,
            home_delta,
            current_joint_pos,
        )

    def compute_grasp_command_target_context_for_label(
        self,
        *,
        target_pose_source: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frozen state needed to reconstruct grasp-phase command targets."""
        release, _ = self._compute_approach_release_weight_and_distance(target_pose_source=target_pose_source)
        release = release.unsqueeze(-1)
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        home_delta = self._approach_home_leak * (self._home_joint_pos - current_joint_pos)
        return release.detach().clone(), home_delta.detach().clone(), current_joint_pos.detach().clone()

    def compute_grasp_command_target_from_context_for_loss(
        self,
        actions: torch.Tensor,
        release: torch.Tensor,
        home_delta: torch.Tensor,
        current_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute grasp command targets from frozen rollout context while keeping gradients."""
        processed_delta = self._compute_processed_delta(
            actions,
            release,
            home_delta,
            straight_through_action_clip=True,
        )
        return current_joint_pos + processed_delta

    def compute_processed_delta_for_label(
        self,
        actions: torch.Tensor,
        *,
        target_pose_source: str,
    ) -> torch.Tensor:
        """Compute the gated grasp-phase joint residual delta for labels."""
        release, home_delta, _ = self.compute_grasp_command_target_context_for_label(
            target_pose_source=target_pose_source,
        )
        return self._compute_processed_delta(
            actions,
            release,
            home_delta,
            straight_through_action_clip=False,
        )

    def compute_grasp_command_target_controllable_mask(
        self,
        release: torch.Tensor,
        *,
        min_action_scale: float = 1.0e-6,
    ) -> torch.Tensor:
        """Return envs where the frozen gate leaves joint command targets controllable."""
        if release.ndim == 2 and release.shape[-1] == 1:
            release = release.squeeze(-1)
        action_scale = self._approach_min_action_scale + (1.0 - self._approach_min_action_scale) * release
        return action_scale > min_action_scale

    def _compute_processed_delta(
        self,
        actions: torch.Tensor,
        release: torch.Tensor,
        home_delta: torch.Tensor,
        *,
        straight_through_action_clip: bool,
    ) -> torch.Tensor:
        """Map raw policy actions to gated joint deltas."""
        clipped_actions = torch.clamp(actions, -1.0, 1.0)
        if straight_through_action_clip:
            processed_actions = actions + (clipped_actions - actions).detach()
        else:
            processed_actions = clipped_actions
        scale, offset = self._action_affine_terms_for_batch(actions)
        residual_delta = processed_actions * scale + offset
        release_scale = self._approach_min_action_scale + (1.0 - self._approach_min_action_scale) * release
        return release_scale * residual_delta + (1.0 - release) * home_delta

    def _compute_command_targets(
        self,
        processed_delta: torch.Tensor,
        current_joint_pos: torch.Tensor,
        return_phase_mask: torch.Tensor,
        return_hold_joint_targets: torch.Tensor,
        *,
        straight_through_return_clip: bool,
    ) -> torch.Tensor:
        """Map processed joint deltas to the absolute policy-rate command target."""
        grasp_phase_targets = current_joint_pos + processed_delta
        return_residual = self._return_residual_scale * processed_delta
        if self._return_residual_clip > 0.0:
            clipped_residual = torch.clamp(
                return_residual,
                -self._return_residual_clip,
                self._return_residual_clip,
            )
            if straight_through_return_clip:
                return_residual = return_residual + (clipped_residual - return_residual).detach()
            else:
                return_residual = clipped_residual
        if self._return_mode == self.RETURN_MODE_HOLD:
            return_phase_targets = return_hold_joint_targets
        elif self._return_mode == self.RETURN_MODE_CONTINUOUS:
            return_phase_targets = grasp_phase_targets
        elif self._return_mode == self.RETURN_MODE_CURRENT_RESIDUAL:
            return_phase_targets = current_joint_pos + return_residual
        elif self._return_mode == self.RETURN_MODE_LATCHED_RESIDUAL:
            return_phase_targets = return_hold_joint_targets + return_residual
        else:
            raise RuntimeError(f"Unhandled return_mode={self._return_mode!r}.")
        return torch.where(return_phase_mask.unsqueeze(-1), return_phase_targets, grasp_phase_targets)

    def _action_affine_terms_for_batch(
        self,
        actions: torch.Tensor,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Return action scale/offset terms that broadcast over rollout minibatches."""
        scale = self._scale
        offset = self._offset
        if isinstance(scale, torch.Tensor) and scale.shape[0] != actions.shape[0]:
            scale = scale[:1]
        if isinstance(offset, torch.Tensor) and offset.shape[0] != actions.shape[0]:
            offset = offset[:1]
        return scale, offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        current_joint_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._last_joint_targets[env_ids] = current_joint_pos
        self._return_hold_joint_targets[env_ids] = current_joint_pos
        self._command_joint_targets[env_ids] = current_joint_pos
        self._return_phase_latched[env_ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        release = self._compute_approach_release_weight().unsqueeze(-1)
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        home_delta = self._approach_home_leak * (self._home_joint_pos - current_joint_pos)
        self._processed_actions[:] = self._compute_processed_delta(
            actions,
            release,
            home_delta,
            straight_through_action_clip=False,
        )

        return_phase_mask = _phase_mask(self._env, self._phase_switch_time_s, step_offset=0)
        enter_return_mask = return_phase_mask & ~self._return_phase_latched
        if torch.any(enter_return_mask):
            self._return_hold_joint_targets[enter_return_mask] = self._last_joint_targets[enter_return_mask]
            self._return_phase_latched[enter_return_mask] = True
        command_targets = self._compute_command_targets(
            self._processed_actions,
            current_joint_pos,
            return_phase_mask,
            self._return_hold_joint_targets,
            straight_through_return_clip=False,
        )
        self._command_joint_targets[:] = command_targets
        self._last_joint_targets[:] = command_targets

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._command_joint_targets, joint_ids=self._joint_ids)


@configclass
class DistanceGatedRelativeJointPositionSim2SimActionCfg(ActionTermCfg):
    """Config for :class:`DistanceGatedRelativeJointPositionSim2SimAction`."""

    class_type: type = DistanceGatedRelativeJointPositionSim2SimAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 0.5
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = True
    use_zero_offset: bool = True
    home_joint_positions: float | dict[str, float] = MISSING
    object_name: str = "object"
    hand_transform_pos: tuple[float, float, float] = MISSING
    hand_transform_quat: tuple[float, float, float, float] = MISSING
    standoff_distance: float = 0.0
    approach_num_points: int = MISSING
    approach_release_inner_distance: float = 0.12
    approach_release_outer_distance: float = 0.24
    approach_min_action_scale: float = 0.0
    approach_home_leak: float = 0.10
    phase_switch_time_s: float = MISSING
    return_mode: str = "hold"
    return_residual_scale: float = 0.2
    return_residual_clip: float = 0.08
