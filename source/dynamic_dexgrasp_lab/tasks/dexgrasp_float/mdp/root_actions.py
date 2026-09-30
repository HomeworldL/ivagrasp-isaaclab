"""Custom floating-root action term for dexgrasp."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .metrics import _phase_mask, hand_grasp_pose_w, hand_target_grasp_pose_w


def _axis_angle_delta_to_quat(local_ang_delta: torch.Tensor) -> torch.Tensor:
    """Convert local-frame axis-angle delta to a quaternion."""
    delta_angle = torch.linalg.norm(local_ang_delta, dim=-1)
    delta_axis = local_ang_delta / torch.clamp(delta_angle.unsqueeze(-1), min=1e-8)
    delta_quat = math_utils.quat_from_angle_axis(delta_angle, delta_axis)
    return torch.where(
        (delta_angle < 1e-8).unsqueeze(-1),
        delta_quat.new_tensor((1.0, 0.0, 0.0, 0.0)).expand_as(delta_quat),
        delta_quat,
    )


def _inverse_local_frame_transform(local_pos: torch.Tensor, local_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a local-frame pose transform represented in wxyz quaternion convention."""
    inv_quat = math_utils.quat_inv(local_quat)
    inv_pos = math_utils.quat_apply(inv_quat, -local_pos)
    return inv_pos, inv_quat


def _desired_root_pose_from_desired_grasp_pose(
    current_root_pos_w: torch.Tensor,
    current_root_quat_w: torch.Tensor,
    current_grasp_pos_w: torch.Tensor,
    current_grasp_quat_w: torch.Tensor,
    desired_grasp_pos_w: torch.Tensor,
    desired_grasp_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a desired grasp-frame world pose into the corresponding desired root pose.

    中文说明：
    - PBVS / wrench 控制最终作用在 hand root；
    - 因此 approach_grasp_frame 只能作为抓取参考目标，不能直接当作 desired root pose；
    - 这里先用当前状态求出 ``T_root_grasp``，再反推 ``T_grasp_root``，
      最后得到与 desired grasp pose 对应的 desired root pose。
    """
    root_to_grasp_pos, root_to_grasp_quat = math_utils.subtract_frame_transforms(
        current_root_pos_w,
        current_root_quat_w,
        current_grasp_pos_w,
        current_grasp_quat_w,
    )
    grasp_to_root_pos, grasp_to_root_quat = _inverse_local_frame_transform(root_to_grasp_pos, root_to_grasp_quat)
    desired_root_pos_w, desired_root_quat_w = math_utils.combine_frame_transforms(
        desired_grasp_pos_w,
        desired_grasp_quat_w,
        grasp_to_root_pos,
        grasp_to_root_quat,
    )
    return desired_root_pos_w, math_utils.quat_unique(desired_root_quat_w)


def _rotate_inertia_to_root_body_frame(
    root_quat_w: torch.Tensor,
    body_com_quat_w: torch.Tensor,
    body_inertia_com_principal: torch.Tensor,
) -> torch.Tensor:
    """Rotate link COM inertias from principal-axis frame into the root body frame."""
    device = root_quat_w.device
    body_com_quat_w = body_com_quat_w.to(device=device)
    body_inertia_com_principal = body_inertia_com_principal.to(device=device)
    root_quat_inv_w = math_utils.quat_inv(root_quat_w).unsqueeze(1).expand_as(body_com_quat_w)
    body_inertia_quat_root = math_utils.quat_mul(root_quat_inv_w, body_com_quat_w)
    rot_root_from_inertia = math_utils.matrix_from_quat(body_inertia_quat_root)
    return rot_root_from_inertia @ body_inertia_com_principal @ rot_root_from_inertia.transpose(-1, -2)


def _body_com_positions_in_root_body_frame(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    body_com_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Express all link COM positions in the root body frame."""
    rel_com_pos_w = body_com_pos_w - root_pos_w.unsqueeze(1)
    root_quat_inv_w = math_utils.quat_inv(root_quat_w).unsqueeze(1).expand(-1, rel_com_pos_w.shape[1], -1)
    return math_utils.quat_apply(root_quat_inv_w, rel_com_pos_w)


def _compute_subtree_com_wrench(
    approx_mass: torch.Tensor,
    approx_inertia_subtree_com_b: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    subtree_com_pos_b: torch.Tensor,
    accel_cmd_b: torch.Tensor,
    force_limit: torch.Tensor,
    torque_limit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a root-body-frame wrench around the whole-hand subtree COM."""
    subtree_com_vel_b = root_lin_vel_b + torch.cross(root_ang_vel_b, subtree_com_pos_b, dim=-1)
    subtree_com_acc_cmd_b = accel_cmd_b[:, :3] + torch.cross(accel_cmd_b[:, 3:], subtree_com_pos_b, dim=-1)

    total_force_b = approx_mass * (
        subtree_com_acc_cmd_b + torch.cross(root_ang_vel_b, subtree_com_vel_b, dim=-1)
    )

    alpha_vec_b = accel_cmd_b[:, 3:].unsqueeze(-1)
    omega_vec_b = root_ang_vel_b.unsqueeze(-1)
    inertia_alpha_b = torch.matmul(approx_inertia_subtree_com_b, alpha_vec_b).squeeze(-1)
    inertia_omega_b = torch.matmul(approx_inertia_subtree_com_b, omega_vec_b).squeeze(-1)
    torque_subtree_com_b = inertia_alpha_b + torch.cross(root_ang_vel_b, inertia_omega_b, dim=-1)
    total_torque_b = torque_subtree_com_b + torch.cross(subtree_com_pos_b, total_force_b, dim=-1)

    total_force_b = torch.clamp(total_force_b, -force_limit, force_limit)
    total_torque_b = torch.clamp(total_torque_b, -torque_limit, torque_limit)
    return total_force_b, total_torque_b


def _compute_root_body_com_wrench(
    approx_mass: torch.Tensor,
    approx_inertia_root_com_b: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    root_body_com_pos_b: torch.Tensor,
    accel_cmd_b: torch.Tensor,
    force_limit: torch.Tensor,
    torque_limit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper; the current controller interprets the COM as the subtree COM."""
    return _compute_subtree_com_wrench(
        approx_mass=approx_mass,
        approx_inertia_subtree_com_b=approx_inertia_root_com_b,
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        subtree_com_pos_b=root_body_com_pos_b,
        accel_cmd_b=accel_cmd_b,
        force_limit=force_limit,
        torque_limit=torque_limit,
    )


def _maybe_diagonalize_inertia(inertia: torch.Tensor, diagonal_only: bool) -> torch.Tensor:
    """Optionally drop off-diagonal coupling terms for diagnostics."""
    if not diagonal_only:
        return inertia
    return torch.diag_embed(torch.diagonal(inertia, dim1=-2, dim2=-1))


def _compute_subtree_com_kinematics(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    subtree_com_pos_b: torch.Tensor,
    accel_cmd_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute subtree COM velocity and commanded acceleration from root-body quantities."""
    subtree_com_vel_b = root_lin_vel_b + torch.cross(root_ang_vel_b, subtree_com_pos_b, dim=-1)
    subtree_com_acc_cmd_b = accel_cmd_b[:, :3] + torch.cross(accel_cmd_b[:, 3:], subtree_com_pos_b, dim=-1)
    return subtree_com_vel_b, subtree_com_acc_cmd_b


def _compute_root_body_com_kinematics(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    root_body_com_pos_b: torch.Tensor,
    accel_cmd_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper; the current controller interprets the COM as the subtree COM."""
    return _compute_subtree_com_kinematics(
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        subtree_com_pos_b=root_body_com_pos_b,
        accel_cmd_b=accel_cmd_b,
    )


def pbvs_calculate_velocity(
    current_pos_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    desired_pos_w: torch.Tensor,
    desired_quat_w: torch.Tensor,
    linear_kp: torch.Tensor,
    angular_kp: torch.Tensor,
    twist_limit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a simple body-frame proportional servo twist from current and desired poses."""
    pos_error_b, quat_error_b = math_utils.subtract_frame_transforms(
        current_pos_w,
        current_quat_w,
        desired_pos_w,
        desired_quat_w,
    )
    quat_error_b = math_utils.quat_unique(quat_error_b)
    rot_error_b = math_utils.axis_angle_from_quat(quat_error_b)
    twist = torch.cat((linear_kp * pos_error_b, angular_kp * rot_error_b), dim=-1)

    if twist_limit is not None:
        twist = torch.clamp(twist, -twist_limit, twist_limit)
    return twist

class RootTwistIntegralAction(ActionTerm):
    """Apply a local-frame twist command by integrating it into a root state target."""

    cfg: RootTwistIntegralActionCfg

    def __init__(self, cfg: RootTwistIntegralActionCfg, env):
        super().__init__(cfg, env)
        if self._asset.is_fixed_base:
            raise ValueError(f"Asset '{cfg.asset_name}' is fixed-base and cannot use RootTwistIntegralAction.")

        self._action_dim = 6
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(
            [*cfg.linear_velocity_scale, *cfg.angular_velocity_scale], dtype=torch.float32, device=self.device
        )
        self._physics_dt = float(env.physics_dt)
        self._target_root_pose_w = torch.zeros((self.num_envs, 7), dtype=torch.float32, device=self.device)
        self._target_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._zero_root_velocity = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self._scale

    def apply_actions(self):
        init_mask = ~self._target_initialized
        if torch.any(init_mask):
            env_ids = torch.where(init_mask)[0]
            self._target_root_pose_w[env_ids, :3] = self._asset.data.root_pos_w[env_ids]
            self._target_root_pose_w[env_ids, 3:] = self._asset.data.root_quat_w[env_ids]
            self._target_initialized[env_ids] = True

        local_lin_delta = self._processed_actions[:, :3] * self._physics_dt
        local_ang_delta = self._processed_actions[:, 3:] * self._physics_dt
        delta_quat = _axis_angle_delta_to_quat(local_ang_delta)

        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            self._target_root_pose_w[:, :3],
            self._target_root_pose_w[:, 3:],
            local_lin_delta,
            delta_quat,
        )
        self._target_root_pose_w[:, :3] = target_pos_w
        self._target_root_pose_w[:, 3:] = math_utils.quat_unique(target_quat_w)
        root_state = torch.cat((self._target_root_pose_w, self._zero_root_velocity), dim=-1)
        self._asset.write_root_state_to_sim(root_state)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._target_initialized[env_ids] = False


@configclass
class RootTwistIntegralActionCfg(ActionTermCfg):
    """Drive a floating-base articulation through local-frame twist integration."""

    class_type: type = RootTwistIntegralAction

    asset_name: str = MISSING
    linear_velocity_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    angular_velocity_scale: tuple[float, float, float] = (0.8, 0.8, 0.8)
    debug_vis: bool = False


class RootTwistAction(ActionTerm):
    """Apply a local-frame twist command by directly writing root-link velocity every dt."""

    cfg: RootTwistActionCfg

    def __init__(self, cfg: RootTwistActionCfg, env):
        super().__init__(cfg, env)
        if self._asset.is_fixed_base:
            raise ValueError(f"Asset '{cfg.asset_name}' is fixed-base and cannot use RootTwistAction.")

        self._action_dim = 6
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._twist_scale = torch.tensor(
            [*cfg.linear_velocity_scale, *cfg.angular_velocity_scale], dtype=torch.float32, device=self.device
        )
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self._twist_scale

    def apply_actions(self):
        current_quat_w = self._asset.data.root_quat_w
        target_lin_vel_w = math_utils.quat_apply(current_quat_w, self._processed_actions[:, :3])
        target_ang_vel_w = math_utils.quat_apply(current_quat_w, self._processed_actions[:, 3:])
        root_link_velocity = torch.cat((target_lin_vel_w, target_ang_vel_w), dim=-1)
        self._asset.write_root_link_velocity_to_sim(root_link_velocity)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class RootTwistActionCfg(ActionTermCfg):
    """Drive a floating-base articulation through direct local-frame velocity commands."""

    class_type: type = RootTwistAction

    asset_name: str = MISSING
    linear_velocity_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    angular_velocity_scale: tuple[float, float, float] = (0.8, 0.8, 0.8)
    debug_vis: bool = False


class RootTwistWrenchAction(ActionTerm):
    """Track a root-body local twist command with an external wrench controller.

    The controller uses root-body kinematics for the command semantics, transforms the commanded
    acceleration to the root-body COM point, and then computes the wrench required to accelerate
    the whole hand as a composite rigid body approximation. The resulting wrench is applied on the
    root body at its COM point, expressed in the root body frame.
    """

    cfg: RootTwistWrenchActionCfg

    def __init__(self, cfg: RootTwistWrenchActionCfg, env):
        super().__init__(cfg, env)
        if self._asset.is_fixed_base:
            raise ValueError(f"Asset '{cfg.asset_name}' is fixed-base and cannot use RootTwistWrenchAction.")

        self._action_dim = 6
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._filtered_twist_cmd = torch.zeros_like(self._raw_actions)
        self._filter_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._twist_scale = torch.tensor(
            [*cfg.linear_velocity_scale, *cfg.angular_velocity_scale], dtype=torch.float32, device=self.device
        )
        self._twist_limit = self._twist_scale.clone()
        self._accel_limit = torch.tensor(
            [*cfg.linear_accel_limit, *cfg.angular_accel_limit], dtype=torch.float32, device=self.device
        )
        self._kp = torch.tensor([*cfg.linear_kp, *cfg.angular_kp], dtype=torch.float32, device=self.device)
        self._kd = torch.tensor([*cfg.linear_kd, *cfg.angular_kd], dtype=torch.float32, device=self.device)
        self._force_limit = torch.tensor(cfg.force_limit, dtype=torch.float32, device=self.device)
        self._torque_limit = torch.tensor(cfg.torque_limit, dtype=torch.float32, device=self.device)
        self._target_body_id = int(cfg.body_id)
        self._approx_mass = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._approx_inertia_subtree_com_b = torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=self.device)
        self._approx_dynamics_initialized = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Keep the latest control terms available for diagnostics and state updates.
        self._last_root_twist_b = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._last_vel_error_b = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._last_accel_cmd_b = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._last_subtree_com_pos_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._last_subtree_com_vel_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._last_subtree_com_acc_cmd_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # Backwards-compatible aliases for existing inspection/demo scripts.
        self._approx_inertia_root_com_b = self._approx_inertia_subtree_com_b
        self._last_root_body_com_pos_b = self._last_subtree_com_pos_b
        self._last_root_body_com_vel_b = self._last_subtree_com_vel_b
        self._last_root_body_com_acc_cmd_b = self._last_subtree_com_acc_cmd_b
        self._last_total_force_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._last_total_torque_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._debug_zero_force = bool(cfg.zero_force_for_debug)
        self._direct_force_buffer = torch.zeros((self.num_envs, self._asset.num_bodies, 3), dtype=torch.float32, device=self.device)
        self._direct_torque_buffer = torch.zeros_like(self._direct_force_buffer)
        dr_cfg = cfg.root_tracking_dr
        self._root_tracking_dr_enabled = bool(dr_cfg.enabled)
        max_delay_steps = max(0, int(dr_cfg.delay_steps_range[1]))
        self._dr_accel_delay_buffer = torch.zeros(
            (max_delay_steps + 1, self.num_envs, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self._dr_delay_cursor = 0
        self._dr_force_scale = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._dr_torque_scale = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self._dr_delay_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0) * self._twist_scale

    def _initialize_approx_dynamics(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.cpu()

        root_pos_w = self._asset.data.root_pos_w[env_ids]
        root_quat_w = self._asset.data.root_quat_w[env_ids]
        body_com_pos_root_b = _body_com_positions_in_root_body_frame(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            body_com_pos_w=self._asset.data.body_com_pos_w[env_ids],
        )

        masses = self._asset.root_physx_view.get_masses()[env_ids_cpu].to(device=self.device).unsqueeze(-1)
        subtree_com_pos_b = (masses * body_com_pos_root_b).sum(dim=1) / torch.clamp(masses.sum(dim=1), min=1e-8)
        rel_body_com_from_subtree_com_b = body_com_pos_root_b - subtree_com_pos_b.unsqueeze(1)

        body_inertia_com_principal = (
            self._asset.root_physx_view.get_inertias()[env_ids_cpu]
            .to(device=self.device)
            .view(env_ids.shape[0], self._asset.num_bodies, 3, 3)
        )
        body_inertia_root_b = _rotate_inertia_to_root_body_frame(
            root_quat_w=root_quat_w,
            body_com_quat_w=self._asset.data.body_com_quat_w[env_ids],
            body_inertia_com_principal=body_inertia_com_principal,
        )

        rel_skew = math_utils.skew_symmetric_matrix(rel_body_com_from_subtree_com_b.reshape(-1, 3)).view(
            env_ids.shape[0], self._asset.num_bodies, 3, 3
        )
        parallel_axis = masses.unsqueeze(-1) * torch.matmul(
            rel_skew.transpose(-1, -2),
            rel_skew,
        )
        approx_inertia_subtree_com_b = (body_inertia_root_b + parallel_axis).sum(dim=1)
        approx_mass = masses.sum(dim=1)
        approx_inertia_subtree_com_b = _maybe_diagonalize_inertia(
            approx_inertia_subtree_com_b, diagonal_only=self.cfg.use_diagonal_inertia_only
        )

        self._approx_mass[env_ids] = approx_mass
        self._approx_inertia_subtree_com_b[env_ids] = approx_inertia_subtree_com_b
        self._last_subtree_com_pos_b[env_ids] = subtree_com_pos_b
        self._approx_dynamics_initialized[env_ids] = True

    def _compute_desired_accel_from_desired_twist(
        self, desired_twist_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据期望 root-body twist 计算跟踪加速度命令。

        中文说明：
        - 这一层只负责 twist 跟踪；
        - 输出 ``root_twist_b``、``vel_error`` 和 ``accel_cmd_b``，便于和部署侧的加速度接口对齐；
        - 不在这里做任何 wrench / inverse-dynamics 映射。
        """
        root_lin_vel_b = self._asset.data.root_link_lin_vel_b
        root_ang_vel_b = self._asset.data.root_link_ang_vel_b
        root_twist_b = torch.cat((root_lin_vel_b, root_ang_vel_b), dim=-1)
        vel_error = desired_twist_b - root_twist_b
        accel_cmd_b = self._kp * vel_error - self._kd * root_twist_b
        accel_cmd_b = torch.clamp(accel_cmd_b, -self._accel_limit, self._accel_limit)
        return root_twist_b, vel_error, accel_cmd_b

    def _env_ids_tensor(self, env_ids: torch.Tensor | slice | Sequence[int] | int) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def _sample_uniform_range(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = float(value_range[0]), float(value_range[1])
        if high < low:
            raise ValueError(f"Invalid root tracking DR range: {value_range}.")
        if high == low:
            return torch.full(shape, low, dtype=torch.float32, device=self.device)
        return low + (high - low) * torch.rand(shape, dtype=torch.float32, device=self.device)

    def _sample_root_tracking_domain_randomization(self, env_ids: torch.Tensor | slice) -> None:
        ids = self._env_ids_tensor(env_ids)
        if ids.numel() == 0:
            return
        if not self._root_tracking_dr_enabled:
            self._dr_force_scale[ids] = 1.0
            self._dr_torque_scale[ids] = 1.0
            self._dr_delay_steps[ids] = 0
            self._dr_accel_delay_buffer[:, ids, :] = 0.0
            return

        dr_cfg = self.cfg.root_tracking_dr
        self._dr_force_scale[ids] = self._sample_uniform_range(dr_cfg.force_scale_range, (ids.numel(), 1))
        self._dr_torque_scale[ids] = self._sample_uniform_range(dr_cfg.torque_scale_range, (ids.numel(), 1))
        delay_min, delay_max = int(dr_cfg.delay_steps_range[0]), int(dr_cfg.delay_steps_range[1])
        if delay_min < 0 or delay_max < delay_min:
            raise ValueError(f"Invalid root tracking DR delay range: {dr_cfg.delay_steps_range}.")
        if delay_max == delay_min:
            self._dr_delay_steps[ids] = delay_min
        else:
            self._dr_delay_steps[ids] = torch.randint(
                delay_min,
                delay_max + 1,
                (ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
        self._dr_accel_delay_buffer[:, ids, :] = 0.0

    def _apply_root_tracking_domain_randomization_to_accel(self, accel_cmd_b: torch.Tensor) -> torch.Tensor:
        if not self._root_tracking_dr_enabled:
            return accel_cmd_b

        dr_cfg = self.cfg.root_tracking_dr
        noise_std = accel_cmd_b.new_tensor(
            [float(dr_cfg.linear_accel_noise_std)] * 3 + [float(dr_cfg.angular_accel_noise_std)] * 3
        )
        if torch.any(noise_std > 0.0):
            accel_cmd_b = accel_cmd_b + torch.randn_like(accel_cmd_b) * noise_std.unsqueeze(0)
        accel_cmd_b = torch.clamp(accel_cmd_b, -self._accel_limit, self._accel_limit)

        if self._dr_accel_delay_buffer.shape[0] <= 1:
            return accel_cmd_b
        self._dr_delay_cursor = (self._dr_delay_cursor + 1) % self._dr_accel_delay_buffer.shape[0]
        self._dr_accel_delay_buffer[self._dr_delay_cursor] = accel_cmd_b
        read_indices = (self._dr_delay_cursor - self._dr_delay_steps) % self._dr_accel_delay_buffer.shape[0]
        env_indices = torch.arange(self.num_envs, device=self.device)
        return self._dr_accel_delay_buffer[read_indices, env_indices]

    def _apply_root_tracking_domain_randomization_to_wrench(
        self,
        total_force_b: torch.Tensor,
        total_torque_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._root_tracking_dr_enabled:
            return total_force_b, total_torque_b
        total_force_b = torch.clamp(total_force_b * self._dr_force_scale, -self._force_limit, self._force_limit)
        total_torque_b = torch.clamp(total_torque_b * self._dr_torque_scale, -self._torque_limit, self._torque_limit)
        return total_force_b, total_torque_b

    def _apply_wrench_from_desired_accel(
        self,
        accel_cmd_b: torch.Tensor,
        *,
        root_twist_b: torch.Tensor | None = None,
        vel_error_b: torch.Tensor | None = None,
    ) -> None:
        """根据期望 root-body 加速度命令计算并施加 wrench。"""
        root_lin_vel_b = self._asset.data.root_link_lin_vel_b
        root_ang_vel_b = self._asset.data.root_link_ang_vel_b
        if root_twist_b is None:
            root_twist_b = torch.cat((root_lin_vel_b, root_ang_vel_b), dim=-1)
        if vel_error_b is None:
            vel_error_b = torch.zeros_like(root_twist_b)

        accel_cmd_b = self._apply_root_tracking_domain_randomization_to_accel(accel_cmd_b)
        subtree_com_pos_b = self._last_subtree_com_pos_b
        subtree_com_vel_b, subtree_com_acc_cmd_b = _compute_subtree_com_kinematics(
            root_lin_vel_b=root_lin_vel_b,
            root_ang_vel_b=root_ang_vel_b,
            subtree_com_pos_b=subtree_com_pos_b,
            accel_cmd_b=accel_cmd_b,
        )
        total_force_b, total_torque_b = _compute_subtree_com_wrench(
            approx_mass=self._approx_mass,
            approx_inertia_subtree_com_b=self._approx_inertia_subtree_com_b,
            root_lin_vel_b=root_lin_vel_b,
            root_ang_vel_b=root_ang_vel_b,
            subtree_com_pos_b=subtree_com_pos_b,
            accel_cmd_b=accel_cmd_b,
            force_limit=self._force_limit,
            torque_limit=self._torque_limit,
        )
        total_force_b, total_torque_b = self._apply_root_tracking_domain_randomization_to_wrench(
            total_force_b,
            total_torque_b,
        )
        if self._debug_zero_force:
            total_force_b = torch.zeros_like(total_force_b)
        self._last_root_twist_b[:] = root_twist_b
        self._last_vel_error_b[:] = vel_error_b
        self._last_accel_cmd_b[:] = accel_cmd_b
        self._last_subtree_com_vel_b[:] = subtree_com_vel_b
        self._last_subtree_com_acc_cmd_b[:] = subtree_com_acc_cmd_b
        self._last_total_force_b[:] = total_force_b
        self._last_total_torque_b[:] = total_torque_b

        self._direct_force_buffer.zero_()
        self._direct_torque_buffer.zero_()
        self._direct_force_buffer[:, self._target_body_id, :] = total_force_b
        self._direct_torque_buffer[:, self._target_body_id, :] = total_torque_b
        # Direct PhysX application is required here. The permanent wrench composer path
        # did not produce reliable floating-root angular response for this articulation,
        # while direct root_physx_view application matches the expected torque tracking.
        self._asset.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self._direct_force_buffer.view(-1, 3),
            torque_data=self._direct_torque_buffer.view(-1, 3),
            position_data=None,
            indices=self._asset._ALL_INDICES,
            is_global=False,
        )

    def _filter_desired_twist(self, desired_twist_b: torch.Tensor) -> torch.Tensor:
        if self.cfg.command_ema_alpha <= 0.0:
            return desired_twist_b
        init_mask = ~self._filter_initialized
        if torch.any(init_mask):
            self._filtered_twist_cmd[init_mask] = desired_twist_b[init_mask]
            self._filter_initialized[init_mask] = True
        keep = float(self.cfg.command_ema_alpha)
        update = 1.0 - keep
        self._filtered_twist_cmd[:] = keep * self._filtered_twist_cmd + update * desired_twist_b
        return self._filtered_twist_cmd

    def apply_actions(self):
        init_env_ids = torch.where(~self._approx_dynamics_initialized)[0]
        if init_env_ids.numel() > 0:
            self._initialize_approx_dynamics(init_env_ids)
        desired_twist_b = self._filter_desired_twist(self._processed_actions)
        root_twist_b, vel_error_b, accel_cmd_b = self._compute_desired_accel_from_desired_twist(desired_twist_b)
        self._apply_wrench_from_desired_accel(
            accel_cmd_b,
            root_twist_b=root_twist_b,
            vel_error_b=vel_error_b,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._filtered_twist_cmd[env_ids] = 0.0
        self._filter_initialized[env_ids] = False
        self._approx_dynamics_initialized[env_ids] = False
        self._direct_force_buffer[env_ids] = 0.0
        self._direct_torque_buffer[env_ids] = 0.0
        self._sample_root_tracking_domain_randomization(env_ids)


@configclass
class RootTrackingDomainRandomizationCfg:
    """Per-env randomization for the low-level root wrench tracking path."""

    enabled: bool = False
    force_scale_range: tuple[float, float] = (0.8, 1.2)
    torque_scale_range: tuple[float, float] = (0.5, 1.2)
    linear_accel_noise_std: float = 0.02
    angular_accel_noise_std: float = 0.10
    delay_steps_range: tuple[int, int] = (0, 1)


@configclass
class RootTwistWrenchActionCfg(ActionTermCfg):
    """Drive a floating-base articulation through a root-body-frame external wrench controller."""

    class_type: type = RootTwistWrenchAction

    asset_name: str = MISSING
    linear_velocity_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)
    angular_velocity_scale: tuple[float, float, float] = (0.8, 0.8, 0.8)
    linear_accel_limit: tuple[float, float, float] = (5.0, 5.0, 5.0)
    angular_accel_limit: tuple[float, float, float] = (15.0, 15.0, 15.0)
    linear_kp: tuple[float, float, float] = (60.0, 60.0, 60.0)
    angular_kp: tuple[float, float, float] = (60.0, 60.0, 60.0)
    linear_kd: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_kd: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_limit: tuple[float, float, float] = (1.0e6, 1.0e6, 1.0e6)
    torque_limit: tuple[float, float, float] = (1.0e6, 1.0e6, 1.0e6)
    body_id: int = 0
    command_ema_alpha: float = 0.0
    use_diagonal_inertia_only: bool = False
    zero_force_for_debug: bool = False
    root_tracking_dr: RootTrackingDomainRandomizationCfg = RootTrackingDomainRandomizationCfg()
    debug_vis: bool = False


class RootTwistPBVSWrenchSim2SimAction(RootTwistWrenchAction):
    """PBVS-biased root wrench control with a deterministic return phase.

    During the grasp phase, it uses an approach PBVS prior plus RL residual
    twist. After the configured time switch, it ignores the RL
    residual and uses deterministic PBVS return-to-goal control only.
    """

    cfg: RootTwistPBVSWrenchSim2SimActionCfg
    RETURN_MODE_PBVS = "pbvs"
    RETURN_MODE_CONTINUOUS = "continuous"
    RETURN_MODES = {
        RETURN_MODE_PBVS,
        RETURN_MODE_CONTINUOUS,
    }

    def __init__(self, cfg: RootTwistPBVSWrenchSim2SimActionCfg, env):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name
        self._object = env.scene["object"]
        self._phase_switch_time_s = float(cfg.phase_switch_time_s)
        self._standoff_distance = float(cfg.standoff_distance)
        self._approach_num_points = int(cfg.approach_num_points)
        self._servo_kp = torch.tensor(
            [*cfg.servo_linear_kp, *cfg.servo_angular_kp],
            dtype=torch.float32,
            device=self.device,
        )
        self._servo_twist_limit = self._twist_limit
        self._rl_residual_scale = torch.tensor(
            [cfg.rl_linear_twist_scale] * 3 + [cfg.rl_angular_twist_scale] * 3,
            dtype=torch.float32,
            device=self.device,
        )
        self._return_mode = str(cfg.return_mode)
        if self._return_mode not in self.RETURN_MODES:
            raise ValueError(
                f"Unsupported root return_mode={self._return_mode!r}. "
                f"Supported modes are: {sorted(self.RETURN_MODES)}."
            )
        self._enable_motion_feedforward = bool(cfg.enable_motion_feedforward)
        self._return_phase_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._servo_twist_b = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self._desired_twist_b = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)

    @property
    def return_phase_mask(self) -> torch.Tensor:
        """Return-phase mask from the most recent action processing call."""
        return self._return_phase_mask

    @property
    def servo_twist_b(self) -> torch.Tensor:
        """PBVS twist from the most recent action processing call, expressed in the hand-root frame."""
        return self._servo_twist_b

    @property
    def desired_twist_b(self) -> torch.Tensor:
        """Desired root twist from the most recent action processing call, expressed in the hand-root frame."""
        return self._desired_twist_b

    @property
    def residual_scale(self) -> torch.Tensor:
        """Per-axis residual action scale used by the grasp-phase root twist controller."""
        return self._rl_residual_scale

    @property
    def twist_limit(self) -> torch.Tensor:
        """Per-axis desired twist limit used by the root twist controller."""
        return self._twist_limit

    @property
    def return_mode(self) -> str:
        """Return-phase root control mode."""
        return self._return_mode

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._return_phase_mask[:] = _phase_mask(self._env, self._phase_switch_time_s, step_offset=0)
        self._servo_twist_b[:] = self._compute_pbvs_servo_twist(self._return_phase_mask)
        motion_feedforward_b = 0.0
        if self._enable_motion_feedforward and self._env.cfg.target_pose_source_action == "full":
            motion_feedforward_b = self._compute_object_linear_velocity_feedforward()
        residual_twist_b = self._rl_residual_scale.unsqueeze(0) * self._processed_actions
        grasp_twist_b = torch.clamp(
            self._servo_twist_b + motion_feedforward_b + residual_twist_b,
            -self._twist_limit,
            self._twist_limit,
        )
        if self._return_mode == self.RETURN_MODE_PBVS:
            return_twist_b = self._servo_twist_b
        elif self._return_mode == self.RETURN_MODE_CONTINUOUS:
            return_twist_b = torch.clamp(
                self._servo_twist_b + residual_twist_b,
                -self._twist_limit,
                self._twist_limit,
            )
        else:
            raise RuntimeError(f"Unhandled root return_mode={self._return_mode!r}.")
        self._desired_twist_b[:] = torch.where(
            self._return_phase_mask.unsqueeze(-1),
            return_twist_b,
            grasp_twist_b,
        )

    def compute_desired_twist_b_for_label(
        self,
        actions: torch.Tensor,
        *,
        target_pose_source: str,
        enable_motion_feedforward: bool,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a desired-twist label under explicit PBVS semantics.

        This helper is used by teacher-student distillation to query what a
        teacher action would mean under full or partial target-pose semantics
        without mutating the live action buffers that drive the simulator.
        """
        if residual_scale is None:
            residual_scale = self._rl_residual_scale
        return_phase_mask = _phase_mask(self._env, self._phase_switch_time_s, step_offset=0)
        servo_twist_b = self._compute_pbvs_servo_twist(
            return_phase_mask,
            target_pose_source=target_pose_source,
        )
        motion_feedforward_b = torch.zeros_like(servo_twist_b)
        if enable_motion_feedforward:
            motion_feedforward_b = self._compute_object_linear_velocity_feedforward()
        grasp_twist_b = torch.clamp(
            servo_twist_b
            + motion_feedforward_b
            + residual_scale.unsqueeze(0) * torch.clamp(actions, -1.0, 1.0),
            -self._twist_limit,
            self._twist_limit,
        )
        return torch.where(return_phase_mask.unsqueeze(-1), servo_twist_b, grasp_twist_b)

    def compute_pbvs_twist_b_for_label(
        self,
        *,
        target_pose_source: str,
        return_phase_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a PBVS twist under explicit target-pose semantics."""
        if return_phase_mask is None:
            return_phase_mask = _phase_mask(self._env, self._phase_switch_time_s, step_offset=0)
        return self._compute_pbvs_servo_twist(
            return_phase_mask,
            target_pose_source=target_pose_source,
        )

    def compute_return_phase_mask_for_label(self) -> torch.Tensor:
        """Compute the current return-phase mask without mutating action buffers."""
        return _phase_mask(self._env, self._phase_switch_time_s, step_offset=0)

    def _compute_object_linear_velocity_feedforward(self) -> torch.Tensor:
        """Return object linear velocity expressed in the hand-root frame.

        This is a privileged motion feedforward used only when the action target
        source is full. It augments grasp-phase PBVS tracking without affecting
        the deterministic return phase.
        """
        object_lin_vel_w = self._object.data.root_lin_vel_w
        root_quat_inv_w = math_utils.quat_inv(self._asset.data.root_quat_w)
        object_lin_vel_root_b = math_utils.quat_apply(root_quat_inv_w, object_lin_vel_w)
        object_motion_ff_b = torch.zeros_like(self._servo_twist_b)
        object_motion_ff_b[:, :3] = object_lin_vel_root_b
        return object_motion_ff_b

    def _compute_pbvs_servo_twist(
        self,
        return_phase_mask: torch.Tensor,
        *,
        target_pose_source: str | None = None,
    ) -> torch.Tensor:
        command = self._env.command_manager.get_term(self._command_name)
        use_goal_pose = return_phase_mask
        if target_pose_source is None:
            target_pose_source = self._env.cfg.target_pose_source_action

        target_grasp_pos_w, target_grasp_quat_w = hand_target_grasp_pose_w(
            env=self._env,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
            standoff_distance=self._standoff_distance,
            num_points=self._approach_num_points,
            target_pose_source=target_pose_source,
        )
        current_grasp_pos_w, current_grasp_quat_w = hand_grasp_pose_w(
            env=self._env,
            hand_entity_name=self.cfg.asset_name,
            palm_body_name=self._env.cfg.palm_body_name,
            hand_transform_pos=self.cfg.hand_transform_pos,
            hand_transform_quat=self.cfg.hand_transform_quat,
        )
        target_root_pos_w, target_root_quat_w = _desired_root_pose_from_desired_grasp_pose(
            current_root_pos_w=self._asset.data.root_pos_w,
            current_root_quat_w=self._asset.data.root_quat_w,
            current_grasp_pos_w=current_grasp_pos_w,
            current_grasp_quat_w=current_grasp_quat_w,
            desired_grasp_pos_w=target_grasp_pos_w,
            desired_grasp_quat_w=target_grasp_quat_w,
        )
        desired_root_pos_w = torch.where(
            use_goal_pose.unsqueeze(-1),
            command.hand_goal_pose_w[:, :3],
            target_root_pos_w,
        )
        desired_root_quat_w = torch.where(
            use_goal_pose.unsqueeze(-1),
            command.hand_goal_pose_w[:, 3:7],
            target_root_quat_w,
        )
        desired_root_quat_w = math_utils.quat_unique(desired_root_quat_w)

        env_origins = self._env.scene.env_origins
        servo_twist_b = pbvs_calculate_velocity(
            current_pos_w=self._asset.data.root_pos_w - env_origins,
            current_quat_w=self._asset.data.root_quat_w,
            desired_pos_w=desired_root_pos_w - env_origins,
            desired_quat_w=desired_root_quat_w,
            linear_kp=self._servo_kp[:3],
            angular_kp=self._servo_kp[3:],
            twist_limit=self._servo_twist_limit,
        )

        return servo_twist_b

    def apply_actions(self):
        init_env_ids = torch.where(~self._approx_dynamics_initialized)[0]
        if init_env_ids.numel() > 0:
            self._initialize_approx_dynamics(init_env_ids)
        root_twist_b, vel_error_b, accel_cmd_b = self._compute_desired_accel_from_desired_twist(self._desired_twist_b)
        self._apply_wrench_from_desired_accel(
            accel_cmd_b,
            root_twist_b=root_twist_b,
            vel_error_b=vel_error_b,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._return_phase_mask[env_ids] = False
        self._servo_twist_b[env_ids] = 0.0
        self._desired_twist_b[env_ids] = 0.0

@configclass
class RootTwistPBVSWrenchSim2SimActionCfg(RootTwistWrenchActionCfg):
    """Config for :class:`RootTwistPBVSWrenchSim2SimAction`."""

    class_type: type = RootTwistPBVSWrenchSim2SimAction

    command_name: str = "goal"
    hand_transform_pos: tuple[float, float, float] = MISSING
    hand_transform_quat: tuple[float, float, float, float] = MISSING
    phase_switch_time_s: float = MISSING
    standoff_distance: float = 0.05
    approach_num_points: int = MISSING
    rl_linear_twist_scale: float = 0.5
    rl_angular_twist_scale: float = 0.5
    servo_linear_kp: tuple[float, float, float] = (1.5, 1.5, 1.5)
    servo_angular_kp: tuple[float, float, float] = (1.5, 1.5, 1.5)
    enable_motion_feedforward: bool = True
    return_mode: str = "pbvs"
