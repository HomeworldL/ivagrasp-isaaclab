"""Alignment and regression checks for dexgrasp hand profiles."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "source"))


try:
    from dynamic_dexgrasp_lab.assets.hands.allegro_hand import (
        ALLEGRO_CAM_OBJECT_RELATIVE_POS,
        ALLEGRO_CAM_OBJECT_RELATIVE_QUAT,
        ALLEGRO_JOINT_NAMES,
    )
    from dynamic_dexgrasp_lab.assets.hands.inspire_hand import (
        INSPIRE_CAM_OBJECT_RELATIVE_POS,
        INSPIRE_CAM_OBJECT_RELATIVE_QUAT,
        INSPIRE_HAND_JOINT_NAMES,
    )
    from dynamic_dexgrasp_lab.assets.hands.liberhand import (
        LIBERHAND_ACTUATED_JOINT_NAMES,
        LIBERHAND_ARTICULATION_ROOT_PRIM_PATH,
        LIBERHAND_CONTACT_BODY_NAMES,
        LIBERHAND_CONTACT_FINGER_BODY_NAMES,
        LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES,
        LIBERHAND_CAM_OBJECT_RELATIVE_POS,
        LIBERHAND_CAM_OBJECT_RELATIVE_QUAT,
        LIBERHAND_DRIVE_DAMPING,
        LIBERHAND_DRIVE_MAX_FORCE,
        LIBERHAND_DRIVE_STIFFNESS,
        LIBERHAND_GEO_BODY_NAMES,
        LIBERHAND_GEO_BODY_WEIGHTS,
        LIBERHAND_HAND_TRANSFORM_POS,
        LIBERHAND_HAND_TRANSFORM_QUAT,
        LIBERHAND_HAND_OBJECT_RELATIVE_POS,
        LIBERHAND_HAND_OBJECT_RELATIVE_QUAT,
        LIBERHAND_HOME_JOINT_POSITIONS,
        LIBERHAND_HOME_JOINT_POS_MAP,
        LIBERHAND_JOINT_NAMES,
        LIBERHAND_PALM_BODY_NAME,
        LIBERHAND_POSITION_ACTION_OFFSET_MAP,
        LIBERHAND_POSITION_ACTION_SCALE_MAP,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float import dexgrasp_float_env_cfg, mdp
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float.mdp.cache_utils import _bump_reset_epoch, _env_cache_token
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float.mdp import metrics as dex_metrics
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.allegro_hand.allegro_env_cfg import (
        AllegroDexGraspFloatDistillEnvCfg,
        AllegroDexGraspFloatEnvCfg,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.inspire_hand.inspire_env_cfg import (
        InspireDexGraspFloatDistillEnvCfg,
        InspireDexGraspFloatEnvCfg,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.liberhand.liberhand_env_cfg import (
        LIBERHAND_HAND_PROFILE,
        LiberhandDexGraspFloatDistillEnvCfg,
        LiberhandDexGraspFloatEnvCfg,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_moving.config.allegro_hand.allegro_env_cfg import (
        AllegroDexGraspMovingDistillEnvCfg,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_moving.config.inspire_hand.inspire_env_cfg import (
        InspireDexGraspMovingDistillEnvCfg,
    )
    from dynamic_dexgrasp_lab.tasks.dexgrasp_moving.config.liberhand.liberhand_env_cfg import (
        LiberhandDexGraspMovingDistillEnvCfg,
    )
except ImportError:  # pragma: no cover - integration dependency
    pytest.skip("IsaacLab task config dependencies are not available.", allow_module_level=True)


def test_liberhand_profile_matches_exported_constants():
    profile = LIBERHAND_HAND_PROFILE

    assert profile.hand_name == "liberhand_right"
    assert profile.hand_joint_names == LIBERHAND_JOINT_NAMES
    assert profile.hand_actuated_joint_names == LIBERHAND_ACTUATED_JOINT_NAMES
    assert profile.hand_home_joint_positions == LIBERHAND_HOME_JOINT_POSITIONS
    assert dict(profile.hand_home_joint_pos_map) == LIBERHAND_HOME_JOINT_POS_MAP
    assert dict(profile.hand_position_action_offset_map) == LIBERHAND_POSITION_ACTION_OFFSET_MAP
    assert dict(profile.hand_position_action_scale_map) == LIBERHAND_POSITION_ACTION_SCALE_MAP
    assert profile.hand_transform_pos == LIBERHAND_HAND_TRANSFORM_POS
    assert profile.hand_transform_quat == LIBERHAND_HAND_TRANSFORM_QUAT
    assert profile.hand_object_relative_pos == LIBERHAND_HAND_OBJECT_RELATIVE_POS
    assert profile.hand_object_relative_quat == LIBERHAND_HAND_OBJECT_RELATIVE_QUAT
    assert profile.hand_camera_object_relative_pos == LIBERHAND_CAM_OBJECT_RELATIVE_POS
    assert profile.hand_camera_object_relative_quat == LIBERHAND_CAM_OBJECT_RELATIVE_QUAT
    assert profile.hand_palm_body_name == LIBERHAND_PALM_BODY_NAME
    assert profile.hand_geo_body_names == LIBERHAND_GEO_BODY_NAMES
    assert profile.hand_geo_body_weights == LIBERHAND_GEO_BODY_WEIGHTS
    assert profile.hand_articulation_root_prim_path == LIBERHAND_ARTICULATION_ROOT_PRIM_PATH
    assert profile.hand_contact_body_names == LIBERHAND_CONTACT_BODY_NAMES
    assert profile.hand_contact_finger_body_names == LIBERHAND_CONTACT_FINGER_BODY_NAMES
    assert profile.hand_contact_fingertip_body_names == LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES
    assert dict(profile.hand_drive_stiffness or {}) == LIBERHAND_DRIVE_STIFFNESS
    assert dict(profile.hand_drive_damping or {}) == LIBERHAND_DRIVE_DAMPING
    assert dict(profile.hand_drive_max_force or {}) == LIBERHAND_DRIVE_MAX_FORCE
    assert profile.hand_contact_sensor_prim_path_prefix == "{ENV_REGEX_NS}/Robot/hand_root/"


def test_liberhand_cfg_keeps_default_sim2sim_runtime_contract():
    cfg = LiberhandDexGraspFloatEnvCfg()

    assert cfg.events.sample_hand_home_pose.func is mdp.sample_hand_home_pose_sim2sim
    assert cfg.actions.floating_root.class_type is mdp.RootTwistPBVSWrenchSim2SimAction
    assert cfg.actions.hand_joints.class_type is mdp.DistanceGatedRelativeJointPositionSim2SimAction
    assert cfg.actions.floating_root.servo_linear_kp == pytest.approx((1.5, 1.5, 1.5))
    assert cfg.actions.floating_root.servo_angular_kp == pytest.approx((1.5, 1.5, 1.5))
    assert cfg.actions.hand_joints.approach_release_inner_distance == pytest.approx(0.12)
    assert cfg.actions.hand_joints.approach_release_outer_distance == pytest.approx(0.24)
    assert cfg.actions.hand_joints.approach_home_leak == pytest.approx(0.10)
    assert cfg.reward_finger_contact_mode == "fingertip"
    assert cfg.geometry_point_cloud_source == dexgrasp_float_env_cfg.DEFAULT_GEOMETRY_POINT_CLOUD_SOURCE


def test_liberhand_contact_sensor_coverage_matches_alignment_contract():
    cfg = LiberhandDexGraspFloatEnvCfg()
    expected_filter = ["{ENV_REGEX_NS}/Object/base_link"]

    assert cfg.geometry_num_points == 512
    assert len(cfg.observation_contact_sensor_names) == len(LIBERHAND_CONTACT_BODY_NAMES)
    assert len(cfg.reward_contact_sensor_names) == len(LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES)
    assert len(cfg.reward_contact_sensor_groups) == len(LIBERHAND_CONTACT_FINGER_BODY_NAMES)

    for body_name in LIBERHAND_CONTACT_BODY_NAMES:
        sensor_name = f"{body_name}_object_s"
        sensor_cfg = getattr(cfg.scene, sensor_name)
        assert sensor_cfg.filter_prim_paths_expr == expected_filter


def test_liberhand_cfg_binds_joint_family_policy_order_explicitly():
    cfg = LiberhandDexGraspFloatEnvCfg()

    assert cfg.actions.hand_joints.joint_names == list(LIBERHAND_ACTUATED_JOINT_NAMES)
    assert cfg.actions.hand_joints.preserve_order is True
    assert cfg.observations.actor.hand_joint_pos.params["asset_cfg"].joint_names == list(LIBERHAND_JOINT_NAMES)
    assert cfg.observations.actor.hand_joint_pos.params["asset_cfg"].preserve_order is True
    assert cfg.observations.actor.hand_joint_vel.params["asset_cfg"].joint_names == list(LIBERHAND_JOINT_NAMES)
    assert cfg.observations.actor.hand_joint_vel.params["asset_cfg"].preserve_order is True
    assert cfg.observations.critic.hand_joint_force.params["robot_cfg"].joint_names == list(LIBERHAND_JOINT_NAMES)
    assert cfg.observations.critic.hand_joint_force.params["robot_cfg"].preserve_order is True
    assert tuple(cfg.hand_geometry_body_names) == LIBERHAND_GEO_BODY_NAMES
    assert tuple(cfg.observations.actor.hand_obj_geo_vec.params["body_names"]) == LIBERHAND_GEO_BODY_NAMES
    assert tuple(cfg.observations.critic.hand_obj_geo_vec.params["body_names"]) == LIBERHAND_GEO_BODY_NAMES
    assert cfg.actions.floating_root.approach_num_points == cfg.geometry_num_points
    assert cfg.actions.hand_joints.approach_num_points == cfg.geometry_num_points
    assert cfg.commands.goal.phase_switch_time_s == pytest.approx(cfg.phase_switch_time_s)
    assert cfg.actions.floating_root.phase_switch_time_s == pytest.approx(cfg.phase_switch_time_s)
    assert cfg.actions.hand_joints.phase_switch_time_s == pytest.approx(cfg.phase_switch_time_s)


@pytest.mark.parametrize(
    ("cfg_type", "joint_names", "camera_pos", "camera_quat"),
    (
        (
            LiberhandDexGraspFloatDistillEnvCfg,
            LIBERHAND_JOINT_NAMES,
            LIBERHAND_CAM_OBJECT_RELATIVE_POS,
            LIBERHAND_CAM_OBJECT_RELATIVE_QUAT,
        ),
        (
            InspireDexGraspFloatDistillEnvCfg,
            INSPIRE_HAND_JOINT_NAMES,
            INSPIRE_CAM_OBJECT_RELATIVE_POS,
            INSPIRE_CAM_OBJECT_RELATIVE_QUAT,
        ),
        (
            AllegroDexGraspFloatDistillEnvCfg,
            ALLEGRO_JOINT_NAMES,
            ALLEGRO_CAM_OBJECT_RELATIVE_POS,
            ALLEGRO_CAM_OBJECT_RELATIVE_QUAT,
        ),
        (
            LiberhandDexGraspMovingDistillEnvCfg,
            LIBERHAND_JOINT_NAMES,
            LIBERHAND_CAM_OBJECT_RELATIVE_POS,
            LIBERHAND_CAM_OBJECT_RELATIVE_QUAT,
        ),
        (
            InspireDexGraspMovingDistillEnvCfg,
            INSPIRE_HAND_JOINT_NAMES,
            INSPIRE_CAM_OBJECT_RELATIVE_POS,
            INSPIRE_CAM_OBJECT_RELATIVE_QUAT,
        ),
        (
            AllegroDexGraspMovingDistillEnvCfg,
            ALLEGRO_JOINT_NAMES,
            ALLEGRO_CAM_OBJECT_RELATIVE_POS,
            ALLEGRO_CAM_OBJECT_RELATIVE_QUAT,
        ),
    ),
)
def test_distill_cfg_keeps_teacher_student_joint_family_policy_order(
    cfg_type,
    joint_names,
    camera_pos,
    camera_quat,
):
    cfg = cfg_type()

    assert cfg.target_pose_source_action == "full"
    assert cfg.target_pose_source_reward == "full"
    assert not hasattr(cfg.scene, "student_camera")
    assert cfg.student_camera_object_relative_position == camera_pos
    assert cfg.student_camera_object_relative_quat == camera_quat
    assert cfg.student_zbuf_width == 64
    assert cfg.student_zbuf_height == 48
    assert cfg.student_camera_horizontal_fov_deg == pytest.approx(70.0)
    assert cfg.student_visibility_depth_margin == pytest.approx(0.005)
    assert cfg.observations.teacher.hand_joint_pos.params["asset_cfg"].joint_names == list(joint_names)
    assert cfg.observations.teacher.hand_joint_pos.params["asset_cfg"].preserve_order is True
    assert cfg.observations.teacher.hand_joint_vel.params["asset_cfg"].joint_names == list(joint_names)
    assert cfg.observations.teacher.hand_joint_vel.params["asset_cfg"].preserve_order is True
    assert cfg.observations.student.hand_joint_pos.params["asset_cfg"].joint_names == list(joint_names)
    assert cfg.observations.student.hand_joint_pos.params["asset_cfg"].preserve_order is True
    assert cfg.observations.student.hand_joint_vel.params["asset_cfg"].joint_names == list(joint_names)
    assert cfg.observations.student.hand_joint_vel.params["asset_cfg"].preserve_order is True


def test_hand_object_geo_features_cache_keeps_legacy_step_scoped_behavior(monkeypatch):
    class FakeScene(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)
            self.env_origins = torch.zeros((1, 3), dtype=torch.float32)

    class FakeRobot:
        def __init__(self):
            self.data = SimpleNamespace(
                body_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            )

        def find_bodies(self, body_names, preserve_order=False):
            assert preserve_order is True
            return [0 for _ in body_names], list(body_names)

    class FakeObject:
        def __init__(self):
            self.data = SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            )

    monkeypatch.setattr(
        dex_metrics,
        "obj_pc_full_sample_w",
        lambda env, num_points: torch.zeros((env.num_envs, num_points, 3), dtype=torch.float32),
    )

    robot = FakeRobot()
    obj = FakeObject()
    env = SimpleNamespace(num_envs=1, device="cpu", common_step_counter=0, scene=FakeScene(robot=robot, object=obj))

    features_initial = dex_metrics.hand_object_geo_features(env, body_names=("finger",), num_points=1)
    initial_vec = features_initial.geo_vec_w.clone()
    assert torch.allclose(initial_vec, torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32))

    robot.data.body_pos_w[:] = torch.tensor([[[0.5, 0.0, 0.0]]], dtype=torch.float32)
    obj.data.root_pos_w[:] = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)

    features_same_epoch = dex_metrics.hand_object_geo_features(env, body_names=("finger",), num_points=1)
    assert torch.allclose(features_same_epoch.geo_vec_w, initial_vec)

    features_same_step = dex_metrics.hand_object_geo_features(env, body_names=("finger",), num_points=1)
    assert torch.allclose(features_same_step.geo_vec_w, initial_vec)

    env.common_step_counter += 1
    features_next_step = dex_metrics.hand_object_geo_features(env, body_names=("finger",), num_points=1)
    assert torch.allclose(features_next_step.geo_vec_w, torch.tensor([[[1.5, 0.0, 0.0]]], dtype=torch.float32))


def test_reset_cache_helpers_track_step_and_epoch():
    env = SimpleNamespace(common_step_counter=7)

    assert _env_cache_token(env) == (7, 0)
    assert _bump_reset_epoch(env) == 1
    assert _env_cache_token(env) == (7, 1)
    assert _bump_reset_epoch(env) == 2
    assert _env_cache_token(env) == (7, 2)


def test_hand_object_errors_use_grasp_frame_against_object_root(monkeypatch):
    class FakeScene(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    class FakeObject:
        def __init__(self):
            self.data = SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            )

    monkeypatch.setattr(
        dex_metrics,
        "hand_grasp_pose_w",
        lambda **kwargs: (
            torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ),
    )

    env = SimpleNamespace(
        cfg=SimpleNamespace(
            palm_body_name="palm",
            hand_transform_pos=(0.0, 0.0, 0.0),
            hand_transform_quat=(1.0, 0.0, 0.0, 0.0),
        ),
        scene=FakeScene(robot=object(), object=FakeObject()),
    )

    pos_error, rot_error = dex_metrics.hand_object_errors(env)
    assert torch.allclose(pos_error, torch.tensor([0.5], dtype=torch.float32))
    assert torch.allclose(rot_error, torch.tensor([0.0], dtype=torch.float32))


def test_teacher_target_grasp_pose_zero_standoff_uses_point_cloud_centroid(monkeypatch):
    class FakeScene(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    class FakeRobot:
        def __init__(self):
            self.data = SimpleNamespace(
                body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
                body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            )

        def find_bodies(self, body_names, preserve_order=False):
            assert preserve_order is True
            return [0 for _ in body_names], list(body_names)

    class FakeObject:
        def __init__(self):
            self.data = SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            )

    local_points = torch.tensor(
        [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        dex_metrics,
        "obj_pc_full_sample_w",
        lambda env, num_points: local_points[:, :num_points, :] + env.scene["object"].data.root_pos_w.unsqueeze(1),
    )

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=FakeScene(robot=FakeRobot(), object=FakeObject()),
    )

    pos_w, _ = dex_metrics.hand_target_grasp_pose_w_teacher(
        env=env,
        palm_body_name="palm",
        hand_transform_pos=(0.0, 0.0, 0.0),
        hand_transform_quat=(1.0, 0.0, 0.0, 0.0),
        standoff_distance=0.0,
        num_points=3,
    )

    expected_centroid = torch.tensor([[4.0, 2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(pos_w, expected_centroid)


def test_student_target_grasp_pose_zero_standoff_uses_partial_point_cloud_centroid(monkeypatch):
    class FakeScene(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    class FakeRobot:
        def __init__(self):
            self.data = SimpleNamespace(
                body_link_pos_w=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
                body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
                root_pos_w=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            )

        def find_bodies(self, body_names, preserve_order=False):
            assert preserve_order is True
            return [0 for _ in body_names], list(body_names)

    partial_points_w = [
        torch.tensor(
            [[2.0, 2.0, 3.0], [4.0, 2.0, 3.0], [6.0, 2.0, 3.0]],
            dtype=torch.float32,
        )
    ]
    monkeypatch.setattr(
        dex_metrics,
        "obj_pc_part_w",
        lambda env: [points.clone() for points in partial_points_w],
    )

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(student_partial_pc_num_points=3),
        scene=FakeScene(robot=FakeRobot()),
    )

    pos_w, _ = dex_metrics.hand_target_grasp_pose_w_student(
        env=env,
        palm_body_name="palm",
        hand_transform_pos=(0.0, 0.0, 0.0),
        hand_transform_quat=(1.0, 0.0, 0.0, 0.0),
        num_points=3,
    )

    expected_centroid = torch.tensor([[4.0, 2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(pos_w, expected_centroid)


@pytest.mark.parametrize(
    ("cfg_type", "expected_prefix"),
    (
        (LiberhandDexGraspFloatEnvCfg, "{ENV_REGEX_NS}/Robot/hand_root/"),
        (InspireDexGraspFloatEnvCfg, "{ENV_REGEX_NS}/Robot/hand_root/"),
        (AllegroDexGraspFloatEnvCfg, "{ENV_REGEX_NS}/Robot/"),
    ),
)
def test_all_hand_cfgs_use_unified_sim2sim_actions(cfg_type, expected_prefix):
    cfg = cfg_type()

    assert cfg.actions.floating_root.class_type is mdp.RootTwistPBVSWrenchSim2SimAction
    assert cfg.actions.hand_joints.class_type is mdp.DistanceGatedRelativeJointPositionSim2SimAction
    assert cfg.actions.floating_root.servo_angular_kp == pytest.approx((1.5, 1.5, 1.5))
    assert cfg.actions.hand_joints.scale == pytest.approx(0.5)
    assert cfg.hand_profile.hand_contact_sensor_prim_path_prefix == expected_prefix
