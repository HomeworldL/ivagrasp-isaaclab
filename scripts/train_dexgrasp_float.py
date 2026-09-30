"""Simplified train entrypoint for dexgrasp_float tasks."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "source"
UPSTREAM_RSL_RL_DIR = Path("/home/ccs/github/IsaacLab/scripts/reinforcement_learning/rsl_rl")

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(UPSTREAM_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_RSL_RL_DIR))

from isaaclab.app import AppLauncher

from dexgrasp_float_cli_common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OBJECT_SET_ID,
    DEFAULT_RL_SPLIT_DIR,
    OBJECT_SET_ID_CLI_HELP,
    configure_dexgrasp_runtime_agent_cfg,
    configure_dexgrasp_runtime_env_cfg,
    configure_dexgrasp_student_env_cfg,
    configure_root_tracking_domain_randomization,
    configure_torch_backends_for_training,
    resolve_hand_spec,
    seed_all,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hand",
    type=str,
    default="liberhand_right",
    choices=("liberhand_right", "allegro_right", "inspire_right"),
)
parser.add_argument(
    "--mode",
    type=str,
    default="teacher",
    help="Training mode: teacher PPO or student distillation.",
)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-iterations", type=int, default=1000)
parser.add_argument(
    "--experiment-name",
    type=str,
    default="",
    help="Optional experiment folder name under logs/rsl_rl. Defaults to dexgrasp_float_<hand>.",
)
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
parser.add_argument("--rl-split-dir", type=str, default=DEFAULT_RL_SPLIT_DIR)
parser.add_argument(
    "--object-set-id",
    type=str,
    default=DEFAULT_OBJECT_SET_ID,
    help=OBJECT_SET_ID_CLI_HELP,
)
parser.add_argument("--warmup-checkpoint", type=str, default="", help="Optional checkpoint path to load before learn.")
parser.add_argument(
    "--root-tracking-dr",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable low-level floating-root tracking domain randomization. Defaults to disabled.",
)
parser.add_argument(
    "--warmup-mode",
    type=str,
    default="model_only",
    choices=("actor_only", "model_only", "full"),
    help=(
        "Checkpoint loading mode for --warmup-checkpoint. "
        "actor_only loads only the actor model, model_only loads actor and critic without optimizer/iteration, "
        "and full preserves the old full runner load behavior."
    ),
)
parser.add_argument(
    "--teacher-checkpoint",
    type=str,
    default="",
    help="Teacher checkpoint path used by --mode student.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.mode not in {"teacher", "student"}:
    raise ValueError(f"Unsupported --mode {args_cli.mode!r}. Expected 'teacher' or 'student'.")

if args_cli.mode == "student" and not args_cli.teacher_checkpoint:
    raise ValueError("--teacher-checkpoint is required when --mode student.")
if args_cli.mode != "student" and args_cli.teacher_checkpoint:
    raise ValueError("--teacher-checkpoint is only valid when --mode student.")
if args_cli.mode == "student" and args_cli.warmup_checkpoint:
    raise ValueError("--warmup-checkpoint is only supported for --mode teacher.")

if args_cli.warmup_checkpoint:
    warmup_path = Path(args_cli.warmup_checkpoint).expanduser().resolve()
    if not warmup_path.is_file():
        raise FileNotFoundError(
            f"Warmup checkpoint does not exist: {warmup_path}\n"
            "Use an existing checkpoint path before launching simulation."
        )
teacher_ckpt_path = None
if args_cli.teacher_checkpoint:
    teacher_ckpt_path = Path(args_cli.teacher_checkpoint).expanduser().resolve()
    if not teacher_ckpt_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt_path}")

hand_spec = resolve_hand_spec(args_cli.hand)
if args_cli.mode == "student":
    if hand_spec.distill_task is None:
        raise ValueError(f"Hand {args_cli.hand!r} does not provide a distillation task entry point.")
    args_cli.task = hand_spec.distill_task
    args_cli.agent = "rsl_rl_distillation_cfg_entry_point"
else:
    args_cli.task = hand_spec.train_task
    args_cli.agent = "rsl_rl_cfg_entry_point"
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import dynamic_dexgrasp_lab
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import dump_yaml

dynamic_dexgrasp_lab.register_tasks()
configure_torch_backends_for_training()
seed_all(args_cli.seed)
installed_rsl_rl_version = metadata.version("rsl-rl-lib")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    t_cfg_start = time.perf_counter()
    configure_dexgrasp_runtime_env_cfg(
        env_cfg=env_cfg,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        debug_vis=None,
        dataset_root=args_cli.dataset_root,
        rl_split_dir=args_cli.rl_split_dir,
        object_set_id=args_cli.object_set_id,
    )
    if args_cli.mode == "student":
        configure_dexgrasp_student_env_cfg(env_cfg, context="student train")
    configure_root_tracking_domain_randomization(env_cfg, enabled=args_cli.root_tracking_dr)
    print(f"[INFO] root_tracking_dr_enabled={args_cli.root_tracking_dr}", flush=True)
    print(f"[TIMING][train] cfg_setup={time.perf_counter() - t_cfg_start:.3f}s", flush=True)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    configure_dexgrasp_runtime_agent_cfg(
        agent_cfg,
        seed=args_cli.seed,
        device=args_cli.device,
        max_iterations=args_cli.max_iterations,
        experiment_name=(
            args_cli.experiment_name
            or (
                f"dexgrasp_float_student_{args_cli.hand}"
                if args_cli.mode == "student"
                else f"dexgrasp_float_{args_cli.hand}"
            )
        ),
        run_name=args_cli.run_name or args_cli.object_set_id,
    )
    if hasattr(env_cfg, "speed_curriculum"):
        # The moving-task speed curriculum is an experimental branch and must
        # stay opt-in. Default moving training should preserve the uniform
        # reset-speed baseline unless the caller explicitly exports
        # ENABLE_SPEED_CURRICULUM=1.
        enable_speed_curriculum = _env_flag("ENABLE_SPEED_CURRICULUM", False)
        env_cfg.speed_curriculum.enabled = enable_speed_curriculum
        if enable_speed_curriculum:
            env_cfg.speed_curriculum.stage_max_iterations = int(agent_cfg.max_iterations)
            env_cfg.speed_curriculum.steps_per_iteration = int(agent_cfg.num_steps_per_env)

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, timestamp + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else ""))
    env_cfg.log_dir = log_dir
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    t_make_start = time.perf_counter()
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[TIMING][train] gym.make={time.perf_counter() - t_make_start:.3f}s", flush=True)
    start_time = time.time()
    try:
        t_reset_start = time.perf_counter()
        obs, _ = env.reset()
        del obs
        print(f"[TIMING][train] first_reset={time.perf_counter() - t_reset_start:.3f}s", flush=True)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if args_cli.mode == "student":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if args_cli.mode == "student":
            runner.load(str(teacher_ckpt_path))
        elif args_cli.warmup_checkpoint:
            warmup_ckpt = str(Path(args_cli.warmup_checkpoint).expanduser().resolve())
            print(f"[INFO] Loading warmup checkpoint: {warmup_ckpt}")
            print(f"[INFO] Warmup mode: {args_cli.warmup_mode}")
            if args_cli.warmup_mode == "full":
                runner.load(warmup_ckpt)
            elif args_cli.warmup_mode == "model_only":
                runner.load(
                    warmup_ckpt,
                    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                )
            elif args_cli.warmup_mode == "actor_only":
                runner.load(
                    warmup_ckpt,
                    load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False},
                )
            else:
                raise ValueError(f"Unsupported warmup mode: {args_cli.warmup_mode}")

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"[INFO] Training time: {round(time.time() - start_time, 2)}s")
        print(f"[INFO] Log dir: {log_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
