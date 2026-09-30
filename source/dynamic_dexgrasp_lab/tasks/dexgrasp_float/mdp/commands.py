"""Goal command terms for dexgrasp_float."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms

from .geometry import obj_pc_full_sample_w, obj_pc_part_w, obj_pc_part_sample_w
from .geometry import _ensure_camera_home_pose
from .metrics import (
    _phase_mask,
    contact_count,
    contact_force_magnitude,
    episode_elapsed_time_s,
    episode_step_count,
    finger_contact_count,
    grasp_contact_active,
    hand_goal_errors,
    hand_object_errors,
    hand_object_geo_features,
    object_goal_errors,
)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _quat_from_z_axis_to_vector(vectors: torch.Tensor) -> torch.Tensor:
    """Construct wxyz quaternions that rotate +Z to the given vectors."""
    z_axis = vectors.new_tensor((0.0, 0.0, 1.0)).expand_as(vectors)
    directions = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)
    cross = torch.cross(z_axis, directions, dim=-1)
    dot = torch.sum(z_axis * directions, dim=-1, keepdim=True)
    quat = torch.cat((1.0 + dot, cross), dim=-1)

    opposite_mask = dot.squeeze(-1) < -0.9999
    if torch.any(opposite_mask):
        quat[opposite_mask] = quat.new_tensor((0.0, 1.0, 0.0, 0.0))

    return torch.nn.functional.normalize(quat, dim=-1, eps=1e-8)


class GoalCommand(CommandTerm):
    """Maintain paired hand/object goals derived from reset poses."""

    cfg: "GoalCommandCfg"

    def __init__(self, cfg: "GoalCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object: RigidObject = env.scene[cfg.object_asset_name]
        self.robot: Articulation = env.scene[cfg.robot_asset_name]
        self.hand_goal_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.object_goal_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.hand_goal_pose_w[:, 3] = 1.0
        self.object_goal_pose_w[:, 3] = 1.0

        self.object_episode_goal_reached = torch.zeros(self.num_envs, device=self.device)
        self.object_episode_goal_pos_reached = torch.zeros(self.num_envs, device=self.device)
        self.object_episode_goal_rot_reached = torch.zeros(self.num_envs, device=self.device)
        self.hand_episode_goal_reached = torch.zeros(self.num_envs, device=self.device)
        self.hand_episode_goal_pos_reached = torch.zeros(self.num_envs, device=self.device)
        self.hand_episode_goal_rot_reached = torch.zeros(self.num_envs, device=self.device)
        self.max_contact_finger_count = torch.zeros(self.num_envs, device=self.device)

        self.metrics["object_goal_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_goal_rot_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_at_goal_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_at_goal_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_episode_goal_pos_reached"] = self.object_episode_goal_pos_reached
        self.metrics["object_episode_goal_rot_reached"] = self.object_episode_goal_rot_reached
        self.metrics["object_episode_goal_reached"] = self.object_episode_goal_reached
        self.metrics["hand_goal_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_goal_rot_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_at_goal_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_at_goal_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_episode_goal_pos_reached"] = self.hand_episode_goal_pos_reached
        self.metrics["hand_episode_goal_rot_reached"] = self.hand_episode_goal_rot_reached
        self.metrics["hand_episode_goal_reached"] = self.hand_episode_goal_reached
        self.metrics["hand_object_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_object_rot_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["grasp_contact_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_finger_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_contact_finger_count"] = self.max_contact_finger_count
        self.metrics["contact_force_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_force_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_force_nonzero_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_group_force_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_body_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["raw_contact_force_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["raw_contact_force_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_step_count"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_elapsed_time_s"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["phase_is_return"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["phase_is_grasp"] = torch.ones(self.num_envs, device=self.device)
        self._world_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._world_quat_w[:, 0] = 1.0

    @property
    def command(self) -> torch.Tensor:
        return self.hand_goal_pose_w

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "object_goal_visualizer"):
                self.object_goal_visualizer = VisualizationMarkers(self.cfg.object_goal_visualizer_cfg)
                self.object_current_visualizer = VisualizationMarkers(self.cfg.object_current_visualizer_cfg)
                self.hand_goal_visualizer = VisualizationMarkers(self.cfg.hand_goal_visualizer_cfg)
                self.hand_current_visualizer = VisualizationMarkers(self.cfg.hand_current_visualizer_cfg)
                self.world_visualizer = VisualizationMarkers(self.cfg.world_visualizer_cfg)
                self.object_surface_point_visualizer = VisualizationMarkers(self.cfg.object_surface_point_visualizer_cfg)
                self.hand_geo_segment_visualizer = VisualizationMarkers(self.cfg.hand_geo_segment_visualizer_cfg)
                self.hand_grasp_frame_visualizer = VisualizationMarkers(self.cfg.hand_grasp_frame_visualizer_cfg)
                self.student_partial_point_visualizer = VisualizationMarkers(self.cfg.student_partial_point_visualizer_cfg)
                self.camera_visualizer = VisualizationMarkers(self.cfg.camera_visualizer_cfg)
            self.object_goal_visualizer.set_visibility(True)
            self.object_current_visualizer.set_visibility(True)
            self.hand_goal_visualizer.set_visibility(True)
            self.hand_current_visualizer.set_visibility(True)
            self.world_visualizer.set_visibility(True)
            self.object_surface_point_visualizer.set_visibility(True)
            self.hand_geo_segment_visualizer.set_visibility(True)
            self.hand_grasp_frame_visualizer.set_visibility(True)
            self.camera_visualizer.set_visibility(True)
            if self._student_debug_enabled():
                self.student_partial_point_visualizer.set_visibility(True)
        else:
            if hasattr(self, "object_goal_visualizer"):
                self.object_goal_visualizer.set_visibility(False)
                self.object_current_visualizer.set_visibility(False)
                self.hand_goal_visualizer.set_visibility(False)
                self.hand_current_visualizer.set_visibility(False)
                self.world_visualizer.set_visibility(False)
                self.object_surface_point_visualizer.set_visibility(False)
                self.hand_geo_segment_visualizer.set_visibility(False)
                self.hand_grasp_frame_visualizer.set_visibility(False)
                self.camera_visualizer.set_visibility(False)
                self.student_partial_point_visualizer.set_visibility(False)

    def _student_debug_enabled(self) -> bool:
        return hasattr(self._env.cfg.observations, "student")

    def _debug_vis_callback(self, event) -> None:
        del event
        if not self.object.is_initialized:
            return
        if not hasattr(self, "object_goal_visualizer"):
            return

        self.object_goal_visualizer.visualize(self.object_goal_pose_w[:, :3], self.object_goal_pose_w[:, 3:])
        self.object_current_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        grasp_local_pos = torch.tensor(self.cfg.hand_transform_pos, dtype=torch.float32, device=self.device)
        grasp_local_pos = grasp_local_pos.unsqueeze(0).expand(self.num_envs, -1)
        grasp_local_quat = torch.tensor(self.cfg.hand_transform_quat, dtype=torch.float32, device=self.device)
        grasp_local_quat = grasp_local_quat.unsqueeze(0).expand(self.num_envs, -1)

        hand_goal_grasp_pos_w, hand_goal_grasp_quat_w = combine_frame_transforms(
            self.hand_goal_pose_w[:, :3],
            self.hand_goal_pose_w[:, 3:7],
            grasp_local_pos,
            grasp_local_quat,
        )
        self.hand_goal_visualizer.visualize(hand_goal_grasp_pos_w, hand_goal_grasp_quat_w)

        hand_current_grasp_pos_w, hand_current_grasp_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            grasp_local_pos,
            grasp_local_quat,
        )
        self.hand_current_visualizer.visualize(hand_current_grasp_pos_w, hand_current_grasp_quat_w)
        self.world_visualizer.visualize(self._env.scene.env_origins, self._world_quat_w)

        if len(self.cfg.geometry_debug_body_names) == 0:
            return

        max_envs = self.num_envs if self.cfg.geometry_debug_max_envs <= 0 else min(self.cfg.geometry_debug_max_envs, self.num_envs)
        num_points = int(getattr(self._env.cfg, "geometry_num_points", 0))
        if num_points <= 0:
            raise ValueError(f"geometry_num_points must be positive, got {num_points}.")
        object_points_w = obj_pc_full_sample_w(self._env, num_points=num_points)[:max_envs]
        object_points_vis = object_points_w.reshape(-1, 3)
        self.object_surface_point_visualizer.visualize(object_points_vis)

        geo_features = hand_object_geo_features(
            self._env,
            body_names=self.cfg.geometry_debug_body_names,
            num_points=num_points,
        )
        nearest_points_w = geo_features.nearest_points_w[:max_envs]
        body_points_w = (geo_features.nearest_points_w - geo_features.geo_vec_w)[:max_envs]
        segment_vectors = nearest_points_w - body_points_w
        segment_midpoints = 0.5 * (nearest_points_w + body_points_w)
        segment_lengths = torch.linalg.norm(segment_vectors, dim=-1, keepdim=True)
        segment_orientations = _quat_from_z_axis_to_vector(segment_vectors.reshape(-1, 3))
        segment_scales = torch.cat(
            (
                torch.full_like(segment_lengths, self.cfg.geometry_debug_line_thickness),
                torch.full_like(segment_lengths, self.cfg.geometry_debug_line_thickness),
                segment_lengths,
            ),
            dim=-1,
        ).reshape(-1, 3)
        self.hand_geo_segment_visualizer.visualize(
            translations=segment_midpoints.reshape(-1, 3),
            orientations=segment_orientations,
            scales=segment_scales,
        )

        grasp_pos_w, grasp_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            grasp_local_pos,
            grasp_local_quat,
        )
        self.hand_grasp_frame_visualizer.visualize(
            translations=grasp_pos_w[:max_envs],
            orientations=grasp_quat_w[:max_envs],
        )

        if self._student_debug_enabled():
            visible_points_world = obj_pc_part_sample_w(self._env, num_points=num_points)[:max_envs]
            if isinstance(visible_points_world, torch.Tensor):
                partial_points_world = visible_points_world.reshape(-1, 3)
            elif len(visible_points_world) > 0:
                partial_points_world = torch.cat(visible_points_world, dim=0)
            else:
                partial_points_world = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            if partial_points_world.shape[0] > 0:
                self.student_partial_point_visualizer.visualize(partial_points_world)

        _ensure_camera_home_pose(self._env)
        self.camera_visualizer.visualize(
            translations=self._env.dexgrasp_cam_pos_w[:max_envs],
            orientations=self._env.dexgrasp_cam_quat_w[:max_envs],
        )

    def _object_initial_pose_world(self, env_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self._env, "dexgrasp_float_object_anchor_pose_w"):
            return self._env.dexgrasp_float_object_anchor_pose_w[env_ids]
        pos_w = self.object.data.default_root_state[env_ids, :3] + self._env.scene.env_origins[env_ids]
        quat_w = self.object.data.default_root_state[env_ids, 3:7]
        return torch.cat([pos_w, quat_w], dim=-1)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        hand_home_pose_w = getattr(self._env, "dexgrasp_float_home_pose_w", None)
        if hand_home_pose_w is None:
            raise AttributeError("dexgrasp_float_home_pose_w is missing. sample_hand_home_pose must run first.")
        hand_goal_pose_w = hand_home_pose_w[env_ids]
        object_anchor_pose_w = self._object_initial_pose_world(env_ids)

        hand_transform_pos = torch.tensor(self.cfg.hand_transform_pos, dtype=torch.float32, device=self.device)
        hand_transform_pos = hand_transform_pos.unsqueeze(0).expand(len(env_ids), -1)
        hand_transform_quat = torch.tensor(self.cfg.hand_transform_quat, dtype=torch.float32, device=self.device)
        hand_transform_quat = hand_transform_quat.unsqueeze(0).expand(len(env_ids), -1)
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            hand_goal_pose_w[:, :3],
            hand_goal_pose_w[:, 3:7],
            hand_transform_pos,
            hand_transform_quat,
        )
        if self.cfg.object_goal_z_offset != 0.0:
            grasp_z_offset = torch.tensor(
                (0.0, 0.0, self.cfg.object_goal_z_offset),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).expand(len(env_ids), -1)
            goal_pos_w = goal_pos_w + math_utils.quat_apply(goal_quat_w, grasp_z_offset)

        self.hand_goal_pose_w[env_ids] = hand_goal_pose_w
        self.object_goal_pose_w[env_ids] = torch.cat([goal_pos_w, object_anchor_pose_w[:, 3:7]], dim=-1)
        self.object_episode_goal_reached[env_ids] = 0.0
        self.object_episode_goal_pos_reached[env_ids] = 0.0
        self.object_episode_goal_rot_reached[env_ids] = 0.0
        self.hand_episode_goal_reached[env_ids] = 0.0
        self.hand_episode_goal_pos_reached[env_ids] = 0.0
        self.hand_episode_goal_rot_reached[env_ids] = 0.0
        self.max_contact_finger_count[env_ids] = 0.0

    def _update_command(self) -> None:
        return

    def _update_metrics(self) -> None:
        object_pos_error, object_rot_error = object_goal_errors(self._env, command_name=self.cfg.command_name)
        hand_pos_error, hand_rot_error = hand_goal_errors(self._env, command_name=self.cfg.command_name)

        object_at_goal_pos = (object_pos_error <= self.cfg.success_pos_threshold).float()
        object_at_goal_rot = (object_rot_error <= self.cfg.success_rot_threshold).float()
        hand_at_goal_pos = (hand_pos_error <= self.cfg.success_pos_threshold).float()
        hand_at_goal_rot = (hand_rot_error <= self.cfg.success_rot_threshold).float()

        object_at_goal = (object_at_goal_pos.bool() & object_at_goal_rot.bool()).float()
        hand_at_goal = (hand_at_goal_pos.bool() & hand_at_goal_rot.bool()).float()

        contact_fingers = finger_contact_count(
            self._env,
            primary_contact_sensor_names=self.cfg.contact_sensor_names,
            contact_sensor_groups=self.cfg.contact_sensor_groups,
            force_threshold=self.cfg.contact_force_threshold,
            finger_contact_mode=self.cfg.finger_contact_mode,
        )
        if len(self.cfg.contact_sensor_names) > 0:
            contact_force = contact_force_magnitude(self._env, self.cfg.contact_sensor_names)
            contact_force_mean = torch.mean(contact_force, dim=-1)
            contact_force_max = torch.max(contact_force, dim=-1).values
            contact_force_nonzero_ratio = torch.mean((contact_force > self.cfg.contact_force_threshold).float(), dim=-1)

            raw_contact_force_list = []
            for sensor_name in self.cfg.contact_sensor_names:
                sensor = self._env.scene.sensors[sensor_name]
                raw_force = torch.nan_to_num(sensor.data.net_forces_w, nan=0.0).view(self.num_envs, -1, 3)
                raw_contact_force_list.append(torch.linalg.norm(raw_force[:, 0, :], dim=-1))
            raw_contact_force = torch.stack(raw_contact_force_list, dim=-1)
            raw_contact_force_mean = torch.mean(raw_contact_force, dim=-1)
            raw_contact_force_max = torch.max(raw_contact_force, dim=-1).values
        else:
            contact_force_mean = torch.zeros(self.num_envs, device=self.device)
            contact_force_max = torch.zeros(self.num_envs, device=self.device)
            contact_force_nonzero_ratio = torch.zeros(self.num_envs, device=self.device)
            raw_contact_force_mean = torch.zeros(self.num_envs, device=self.device)
            raw_contact_force_max = torch.zeros(self.num_envs, device=self.device)

        if len(self.cfg.contact_sensor_groups) > 0:
            group_force_max_list = []
            all_sensor_names: list[str] = []
            for sensor_group in self.cfg.contact_sensor_groups:
                all_sensor_names.extend(sensor_group)
                group_force = contact_force_magnitude(self._env, sensor_group)
                group_force_max_list.append(torch.max(group_force, dim=-1).values)
            contact_group_force_max = torch.max(torch.stack(group_force_max_list, dim=-1), dim=-1).values
            unique_sensor_names = tuple(dict.fromkeys(all_sensor_names))
            contact_body_count = contact_count(
                self._env,
                unique_sensor_names,
                force_threshold=self.cfg.contact_force_threshold,
            )
        else:
            contact_group_force_max = torch.zeros(self.num_envs, device=self.device)
            contact_body_count = torch.zeros(self.num_envs, device=self.device)
        hand_object_pos_error_value, hand_object_rot_error_value = hand_object_errors(
            self._env,
            command_name=self.cfg.command_name,
        )
        grasp_contact = grasp_contact_active(
            self._env,
            command_name=self.cfg.command_name,
            min_contact_fingers=self.cfg.grasp_contact_min_fingers,
        )
        step_count = episode_step_count(self._env).to(dtype=torch.float32)
        elapsed_time_s = episode_elapsed_time_s(self._env, step_offset=0)
        phase_is_return = _phase_mask(self._env, self.cfg.phase_switch_time_s, step_offset=0).float()
        phase_is_grasp = 1.0 - phase_is_return

        self.object_episode_goal_pos_reached[:] = torch.maximum(self.object_episode_goal_pos_reached, object_at_goal_pos)
        self.object_episode_goal_rot_reached[:] = torch.maximum(self.object_episode_goal_rot_reached, object_at_goal_rot)
        self.object_episode_goal_reached[:] = torch.maximum(self.object_episode_goal_reached, object_at_goal)
        self.hand_episode_goal_pos_reached[:] = torch.maximum(self.hand_episode_goal_pos_reached, hand_at_goal_pos)
        self.hand_episode_goal_rot_reached[:] = torch.maximum(self.hand_episode_goal_rot_reached, hand_at_goal_rot)
        self.hand_episode_goal_reached[:] = torch.maximum(self.hand_episode_goal_reached, hand_at_goal)
        self.max_contact_finger_count[:] = torch.maximum(self.max_contact_finger_count, contact_fingers)

        self.metrics["object_goal_pos_error"] = object_pos_error
        self.metrics["object_goal_rot_error"] = object_rot_error
        self.metrics["object_at_goal_pos"] = object_at_goal_pos
        self.metrics["object_at_goal_rot"] = object_at_goal_rot
        self.metrics["object_at_goal"] = object_at_goal
        self.metrics["object_episode_goal_pos_reached"] = self.object_episode_goal_pos_reached
        self.metrics["object_episode_goal_rot_reached"] = self.object_episode_goal_rot_reached
        self.metrics["object_episode_goal_reached"] = self.object_episode_goal_reached
        self.metrics["hand_goal_pos_error"] = hand_pos_error
        self.metrics["hand_goal_rot_error"] = hand_rot_error
        self.metrics["hand_at_goal_pos"] = hand_at_goal_pos
        self.metrics["hand_at_goal_rot"] = hand_at_goal_rot
        self.metrics["hand_at_goal"] = hand_at_goal
        self.metrics["hand_episode_goal_pos_reached"] = self.hand_episode_goal_pos_reached
        self.metrics["hand_episode_goal_rot_reached"] = self.hand_episode_goal_rot_reached
        self.metrics["hand_episode_goal_reached"] = self.hand_episode_goal_reached
        self.metrics["hand_object_pos_error"] = hand_object_pos_error_value
        self.metrics["hand_object_rot_error"] = hand_object_rot_error_value
        self.metrics["grasp_contact_active"] = grasp_contact
        self.metrics["contact_finger_count"] = contact_fingers
        self.metrics["max_contact_finger_count"] = self.max_contact_finger_count
        self.metrics["contact_force_mean"] = contact_force_mean
        self.metrics["contact_force_max"] = contact_force_max
        self.metrics["contact_force_nonzero_ratio"] = contact_force_nonzero_ratio
        self.metrics["contact_group_force_max"] = contact_group_force_max
        self.metrics["contact_body_count"] = contact_body_count
        self.metrics["raw_contact_force_mean"] = raw_contact_force_mean
        self.metrics["raw_contact_force_max"] = raw_contact_force_max
        self.metrics["episode_step_count"] = step_count
        self.metrics["episode_elapsed_time_s"] = elapsed_time_s
        self.metrics["phase_is_return"] = phase_is_return
        self.metrics["phase_is_grasp"] = phase_is_grasp


@configclass
class GoalCommandCfg(CommandTermCfg):
    """Configuration for the paired hand/object goal command."""

    class_type: type = GoalCommand

    object_asset_name: str = "object"
    robot_asset_name: str = "robot"
    command_name: str = "goal"
    contact_sensor_names: tuple[str, ...] = ()
    contact_sensor_groups: tuple[tuple[str, ...], ...] = ()
    finger_contact_mode: str = "fingertip"
    contact_force_threshold: float = 0.05
    grasp_contact_min_fingers: int = 3
    geometry_debug_max_envs: int = 0
    geometry_debug_line_thickness: float = 0.003
    geometry_debug_body_names: tuple[str, ...] = ()
    palm_body_name: str = ""
    hand_transform_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_transform_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    object_goal_z_offset: float = 0.07
    success_pos_threshold: float = 0.05
    success_rot_threshold: float = 0.3490658503988659
    phase_switch_time_s: float = 0.0
    # object_goal_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
    #     prim_path="/Visuals/DexGrasp/object_goal_pose"
    # )
    object_goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DexGrasp/object_goal_pose",
        markers={
            "origin": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.2),
                    opacity=0.95,
                ),
            )
        },
    )
    object_current_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/DexGrasp/object_current_pose"
    )
    hand_goal_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/DexGrasp/hand_goal_pose"
    )
    hand_current_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/DexGrasp/hand_current_pose"
    )
    world_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/DexGrasp/world_pose"
    )
    object_surface_point_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DexGrasp/object_surface_points",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.2),
                    opacity=0.35,
                ),
            )
        },
    )
    hand_geo_segment_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DexGrasp/hand_geo_segments",
        markers={
            "segment": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.82, 0.18),
                    opacity=0.7,
                ),
            )
        },
    )
    hand_grasp_frame_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DexGrasp/hand_grasp_frame",
        markers={
            "grasp_axis": sim_utils.CylinderCfg(
                radius=0.0015,
                height=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.6, 1.0),
                    opacity=0.95,
                ),
            )
        },
    )
    student_partial_point_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DexGrasp/student_partial_points",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 1.0),
                    opacity=0.7,
                ),
            )
        },
    )
    camera_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/DexGrasp/camera_pose"
    )
    object_current_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    hand_goal_visualizer_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    hand_current_visualizer_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    world_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    camera_visualizer_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
