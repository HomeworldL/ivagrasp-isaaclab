"""Simplified play entrypoint for dexgrasp_float tasks."""

from __future__ import annotations

import argparse
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
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
parser.add_argument("--rl-split-dir", type=str, default=DEFAULT_RL_SPLIT_DIR)
parser.add_argument(
    "--object-set-id",
    type=str,
    default=DEFAULT_OBJECT_SET_ID,
    help=OBJECT_SET_ID_CLI_HELP,
)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--debug-vis",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override dexgrasp goal debug visualization. Use --debug-vis or --no-debug-vis.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

hand_spec = resolve_hand_spec(args_cli.hand)
if args_cli.mode == "student":
    if hand_spec.distill_task is None:
        raise ValueError(f"Hand {args_cli.hand!r} does not define a distill task for --mode student.")
    args_cli.task = hand_spec.distill_task
    args_cli.agent = "rsl_rl_distillation_cfg_entry_point"
else:
    args_cli.task = hand_spec.play_task
    args_cli.agent = "rsl_rl_cfg_entry_point"
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import importlib.metadata as metadata
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import dynamic_dexgrasp_lab
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

dynamic_dexgrasp_lab.register_tasks()
configure_torch_backends_for_training()
seed_all(args_cli.seed)
installed_rsl_rl_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    configure_dexgrasp_runtime_env_cfg(
        env_cfg=env_cfg,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        device=args_cli.device,
        debug_vis=args_cli.debug_vis,
        dataset_root=args_cli.dataset_root,
        rl_split_dir=args_cli.rl_split_dir,
        object_set_id=args_cli.object_set_id,
    )
    if args_cli.mode == "student":
        configure_dexgrasp_student_env_cfg(env_cfg, context="student play")

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl_version)
    configure_dexgrasp_runtime_agent_cfg(agent_cfg, seed=args_cli.seed, device=args_cli.device)

    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = DistillationRunner if args_cli.mode == "student" else OnPolicyRunner
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        dt = env.unwrapped.step_dt
        obs = env.get_observations()
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
                timeout_mask = None
                if isinstance(extras, dict) and "time_outs" in extras:
                    timeout_mask = extras["time_outs"].to(dtype=torch.bool)
                elif hasattr(env.unwrapped, "reset_time_outs"):
                    timeout_mask = env.unwrapped.reset_time_outs.to(dtype=torch.bool)
                if timeout_mask is not None and torch.any(timeout_mask):
                    obs, _ = env.reset()
                    dones = torch.ones_like(dones)
                policy.reset(dones)
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
