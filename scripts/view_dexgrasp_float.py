"""Interactive viewer that reuses the training dexgrasp_float environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher
from dexgrasp_float_cli_common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OBJECT_SET_ID,
    DEFAULT_RL_SPLIT_DIR,
    OBJECT_SET_ID_CLI_HELP,
    configure_dexgrasp_runtime_env_cfg,
    configure_dexgrasp_student_env_cfg,
    configure_torch_backends_for_training,
    resolve_hand_spec,
    seed_all,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("teacher", "student"), default="teacher")
parser.add_argument(
    "--hand",
    type=str,
    default="liberhand_right",
    choices=("liberhand_right", "allegro_right", "inspire_right"),
)
parser.add_argument("--num-envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--policy", choices=("zero", "random", "oscillate"), default="oscillate")
parser.add_argument("--frame-rate", type=float, default=30.0, help="Viewer step rate cap.")
parser.add_argument("--max-steps", type=int, default=0, help="Optional max steps. Zero means run until closed.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--debug-vis",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override dexgrasp task debug visualizations. Defaults to enabled in view.",
)
parser.add_argument(
    "--dataset-root",
    type=str,
    default=DEFAULT_DATASET_ROOT,
    help="Root directory for objdata assets.",
)
parser.add_argument(
    "--rl-split-dir",
    type=str,
    default=DEFAULT_RL_SPLIT_DIR,
    help="RL split directory containing train_cluster.json / test_cluster.json.",
)
parser.add_argument(
    "--object-set-id",
    type=str,
    default=DEFAULT_OBJECT_SET_ID,
    help=OBJECT_SET_ID_CLI_HELP,
)
parser.add_argument(
    "--print-actuator-debug",
    action="store_true",
    help="Print runtime PhysX actuator stiffness/damping and first-env target/position snapshots.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
hand_spec = resolve_hand_spec(args_cli.hand)
if args_cli.mode == "student":
    if hand_spec.distill_task is None:
        raise ValueError(f"Hand {args_cli.hand!r} does not define a distill task for --mode student.")
    args_cli.task = hand_spec.distill_task
else:
    args_cli.task = hand_spec.play_task

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import dynamic_dexgrasp_lab
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

configure_torch_backends_for_training()
seed_all(args_cli.seed)


def _make_actions(policy: str, step: int, num_envs: int, action_dim: int, device: str) -> torch.Tensor:
    actions = torch.zeros(num_envs, action_dim, device=device)
    if action_dim <= 6:
        return actions
    hand_actions = actions[:, 6:]
    if policy == "random":
        hand_actions.copy_(2.0 * torch.rand_like(hand_actions) - 1.0)
    elif policy == "oscillate":
        t = float(step) * 0.05
        joint_phase = torch.linspace(0.0, 2.5, steps=hand_actions.shape[1], device=device).unsqueeze(0)
        env_phase = torch.linspace(0.0, 1.5, steps=num_envs, device=device).unsqueeze(-1)
        hand_actions.copy_(0.5 * torch.sin(t + env_phase + joint_phase))
    return actions


def _configure_goal_debug_vis(env_cfg, enabled: bool) -> None:
    env_cfg.commands.goal.debug_vis = enabled


def _print_actuator_debug(unwrapped_env, *, tag: str) -> None:
    robot = unwrapped_env.scene["robot"]
    stiff_row = None
    damp_row = None
    try:
        stiff = robot.root_physx_view.get_dof_stiffnesses()
        damp = robot.root_physx_view.get_dof_dampings()
        if stiff.ndim == 2:
            stiff_row = stiff[0]
            damp_row = damp[0]
        else:
            stiff_row = stiff
            damp_row = damp
        print(
            f"[ACT_DEBUG][{tag}] stiffness(min/max)=({float(stiff.min()):.4f}, {float(stiff.max()):.4f}) "
            f"damping(min/max)=({float(damp.min()):.4f}, {float(damp.max()):.4f})",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - debug path
        print(f"[ACT_DEBUG][{tag}] failed to query PhysX stiffness/damping: {exc}", flush=True)

    q = robot.data.joint_pos[0]
    q_target = robot.data.joint_pos_target[0]
    names = robot.joint_names
    if stiff_row is None or damp_row is None:
        stiff_row = q.new_zeros(q.shape[0])
        damp_row = q.new_zeros(q.shape[0])
    print(f"[ACT_DEBUG][{tag}] first-env joint snapshot:", flush=True)
    for i, name in enumerate(names):
        print(
            f"  {name:<20s} q={float(q[i]): .5f} target={float(q_target[i]): .5f}"
            f" stiff={float(stiff_row[i]): .5f} damp={float(damp_row[i]): .5f}",
            flush=True,
        )


def main() -> None:
    dynamic_dexgrasp_lab.register_tasks()
    t_parse_start = time.perf_counter()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print(f"[TIMING][view] parse_env_cfg={time.perf_counter() - t_parse_start:.3f}s", flush=True)
    t_apply_start = time.perf_counter()
    configure_dexgrasp_runtime_env_cfg(
        env_cfg=env_cfg,
        seed=args_cli.seed,
        debug_vis=None,
        dataset_root=args_cli.dataset_root,
        rl_split_dir=args_cli.rl_split_dir,
        object_set_id=args_cli.object_set_id,
    )
    if args_cli.mode == "student":
        configure_dexgrasp_student_env_cfg(env_cfg, context="student view")
    print(f"[TIMING][view] apply_object_selection={time.perf_counter() - t_apply_start:.3f}s", flush=True)
    _configure_goal_debug_vis(env_cfg, True if args_cli.debug_vis is None else args_cli.debug_vis)

    t_make_start = time.perf_counter()
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[TIMING][view] gym.make={time.perf_counter() - t_make_start:.3f}s", flush=True)
    try:
        t_reset_start = time.perf_counter()
        obs, _ = env.reset()
        print(f"[TIMING][view] first_reset={time.perf_counter() - t_reset_start:.3f}s", flush=True)
        del obs
        unwrapped_env = env.unwrapped
        if args_cli.print_actuator_debug:
            _print_actuator_debug(unwrapped_env, tag="after_reset_before_step")
        device = unwrapped_env.device
        num_envs = unwrapped_env.num_envs
        action_dim = unwrapped_env.action_manager.total_action_dim
        step_count = 0

        while simulation_app.is_running():
            step_start = time.perf_counter()
            actions = _make_actions(args_cli.policy, step_count, num_envs, action_dim, device)
            obs, _, terminated, truncated, _ = env.step(actions)
            del obs
            step_count += 1
            if args_cli.print_actuator_debug and step_count == 1:
                _print_actuator_debug(unwrapped_env, tag="after_first_step")
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                break

            if args_cli.frame_rate > 0.0:
                elapsed = time.perf_counter() - step_start
                time.sleep(max(0.0, (1.0 / args_cli.frame_rate) - elapsed))
    finally:
        env.close()


if __name__ == "__main__":
    main()
