"""Shared CLI helpers for dexgrasp_float train/play/view scripts."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


DEFAULT_DATASET_ROOT = "/home/ccs/repositories/dexgrasp_sample/datasets/objdata_YCB"
DEFAULT_RL_SPLIT_DIR = (
    "/home/ccs/repositories/dexgrasp_sample/datasets/objdata_YCB/_meta/rl_split/"
    "v1_ae128_k4_seed0_test20_seed0"
)
DEFAULT_OBJECT_SET_ID = "train@cluster@cluster0_topk1@s120_120"
OBJECT_SET_ID_CLI_HELP = (
    "Object-set id examples: "
    "'train@cluster@cluster0_topk1@s120_120', "
    "'train@object@YCB_077_rubiks_cube@s120_120', "
    "'test@object@YCB_077_rubiks_cube@s080_140'. "
    "Use 'default:dexcube' to use Isaac Nucleus DexCube."
)
DEFAULT_STATIC_FRICTION = 0.6
DEFAULT_DYNAMIC_FRICTION = 0.6
DR_STATIC_FRICTION_RANGE = (0.42, 0.78)
DR_DYNAMIC_FRICTION_RANGE = (0.42, 0.78)
DR_CONTACT_OFFSET_SCALE_RANGE = (0.9, 1.1)
DR_REST_OFFSET_DELTA_RANGE = (-0.001, 0.001)
DR_MATERIAL_NUM_BUCKETS = 64


@dataclass(frozen=True)
class DexGraspHandSpec:
    hand: str
    train_task: str
    play_task: str
    distill_task: str | None = None


HAND_SPECS: dict[str, DexGraspHandSpec] = {
    "liberhand_right": DexGraspHandSpec(
        hand="liberhand_right",
        train_task="Isaac-DexGrasp-Float-Liberhand-v0",
        play_task="Isaac-DexGrasp-Float-Liberhand-Play-v0",
        distill_task="Isaac-DexGrasp-Float-Liberhand-Distill-v0",
    ),
    "allegro_right": DexGraspHandSpec(
        hand="allegro_right",
        train_task="Isaac-DexGrasp-Float-Allegro-v0",
        play_task="Isaac-DexGrasp-Float-Allegro-Play-v0",
        distill_task="Isaac-DexGrasp-Float-Allegro-Distill-v0",
    ),
    "inspire_right": DexGraspHandSpec(
        hand="inspire_right",
        train_task="Isaac-DexGrasp-Float-Inspire-v0",
        play_task="Isaac-DexGrasp-Float-Inspire-Play-v0",
        distill_task="Isaac-DexGrasp-Float-Inspire-Distill-v0",
    ),
}


def resolve_hand_spec(hand: str) -> DexGraspHandSpec:
    normalized = hand.strip().lower()
    if normalized not in HAND_SPECS:
        supported = ", ".join(sorted(HAND_SPECS))
        raise ValueError(f"Unsupported hand: {hand!r}. Supported hands: {supported}.")
    return HAND_SPECS[normalized]


def apply_dexgrasp_object_selection(
    env_cfg,
    dataset_root: str,
    rl_split_dir: str,
    object_set_id: str,
) -> object:
    env_cfg.object_sets.dataset_root = dataset_root
    env_cfg.object_sets.rl_split_dir = rl_split_dir
    env_cfg.object_sets.object_set_id = object_set_id
    return env_cfg.apply_object_selection()


def configure_dexgrasp_runtime_env_cfg(
    env_cfg,
    *,
    num_envs: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    debug_vis: bool | None = None,
    dataset_root: str | None = None,
    rl_split_dir: str | None = None,
    object_set_id: str | None = None,
):
    """Apply the common runtime env-cfg overrides used by dexgrasp scripts."""
    if num_envs is not None:
        env_cfg.scene.num_envs = int(num_envs)
    if seed is not None:
        env_cfg.seed = int(seed)
    if device is not None:
        env_cfg.sim.device = device

    env_cfg.scene.replicate_physics = False
    env_cfg.scene.clone_in_fabric = False

    if debug_vis is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "goal"):
        env_cfg.commands.goal.debug_vis = bool(debug_vis)

    if dataset_root is not None and rl_split_dir is not None and object_set_id is not None:
        return apply_dexgrasp_object_selection(
            env_cfg=env_cfg,
            dataset_root=dataset_root,
            rl_split_dir=rl_split_dir,
            object_set_id=object_set_id,
        )
    return None


def configure_dexgrasp_student_env_cfg(env_cfg, *, context: str = "student") -> None:
    """Apply deployable student action/reward semantics for a distillation env.

    Distillation envs must expose separate ``teacher`` and ``student``
    observation groups. Their full/partial observation terms are wired by
    ``DistillObservationsCfg``; this helper only sets deployable student
    action/reward semantics.
    """
    if not hasattr(env_cfg.observations, "teacher") or not hasattr(env_cfg.observations, "student"):
        raise ValueError(
            f"{context} requires a distillation env with explicit teacher and student observation groups."
        )
    env_cfg.target_pose_source_action = "partial"
    env_cfg.target_pose_source_reward = "full"


def configure_root_tracking_domain_randomization(env_cfg, *, enabled: bool) -> None:
    """Toggle root-action DR plus controlled friction/contact perturbations."""
    floating_root = getattr(getattr(env_cfg, "actions", None), "floating_root", None)
    root_tracking_dr = getattr(floating_root, "root_tracking_dr", None)
    if root_tracking_dr is not None:
        root_tracking_dr.enabled = bool(enabled)

    for event_name in ("robot_physics_material", "object_physics_material"):
        material_event = getattr(getattr(env_cfg, "events", None), event_name, None)
        if material_event is not None:
            material_event.params["static_friction_range"] = (
                DR_STATIC_FRICTION_RANGE if enabled else (DEFAULT_STATIC_FRICTION, DEFAULT_STATIC_FRICTION)
            )
            material_event.params["dynamic_friction_range"] = (
                DR_DYNAMIC_FRICTION_RANGE if enabled else (DEFAULT_DYNAMIC_FRICTION, DEFAULT_DYNAMIC_FRICTION)
            )
            material_event.params["restitution_range"] = (0.0, 0.0)
            material_event.params["num_buckets"] = DR_MATERIAL_NUM_BUCKETS if enabled else 1
            material_event.params["make_consistent"] = True

    for event_name in ("robot_collider_offsets", "object_collider_offsets"):
        collider_event = getattr(getattr(env_cfg, "events", None), event_name, None)
        if collider_event is not None:
            collider_event.params["contact_offset_scale_range"] = (
                DR_CONTACT_OFFSET_SCALE_RANGE if enabled else (1.0, 1.0)
            )
            collider_event.params["rest_offset_delta_range"] = (
                DR_REST_OFFSET_DELTA_RANGE if enabled else (0.0, 0.0)
            )

    if hasattr(env_cfg, "palm_dynamics_scale_range"):
        env_cfg.palm_dynamics_scale_range = None
        palm_event = getattr(getattr(env_cfg, "events", None), "palm_dynamics_scale", None)
        if palm_event is not None:
            palm_scale = float(env_cfg.palm_dynamics_scale)
            palm_event.params["mass_distribution_params"] = (palm_scale, palm_scale)


def configure_dexgrasp_runtime_agent_cfg(
    agent_cfg,
    *,
    seed: int,
    device: str,
    max_iterations: int | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
) -> None:
    """Apply the common runtime agent-cfg overrides used by dexgrasp scripts."""
    agent_cfg.seed = int(seed)
    agent_cfg.device = device
    if max_iterations is not None:
        agent_cfg.max_iterations = int(max_iterations)
    if experiment_name is not None:
        agent_cfg.experiment_name = experiment_name
    if run_name is not None:
        agent_cfg.run_name = run_name


def configure_torch_backends_for_training() -> None:
    """Apply backend flags aligned with upstream training defaults."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def seed_all(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
