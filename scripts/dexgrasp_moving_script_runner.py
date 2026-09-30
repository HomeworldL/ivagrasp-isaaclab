"""Utilities for running moving-specific wrappers over float script implementations."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import runpy
import sys


SCRIPT_DIR = Path(__file__).resolve().parent


def _install_module_alias(alias: str, target: str) -> None:
    sys.modules[alias] = importlib.import_module(target)


def _inject_train_experiment_name(argv: list[str]) -> list[str]:
    if "--experiment-name" in argv:
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default="teacher")
    parser.add_argument("--hand", default="liberhand_right")
    known_args, _ = parser.parse_known_args(argv)

    if known_args.mode == "student":
        experiment_name = f"dexgrasp_moving_student_{known_args.hand}"
    else:
        experiment_name = f"dexgrasp_moving_{known_args.hand}"
    return ["--experiment-name", experiment_name, *argv]


def run_float_entrypoint(script_basename: str, argv: list[str] | None = None) -> None:
    """Run the corresponding dexgrasp_float entrypoint with moving task aliases."""
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if script_basename == "train_dexgrasp_float.py":
        effective_argv = _inject_train_experiment_name(effective_argv)

    _install_module_alias("dexgrasp_float_cli_common", "dexgrasp_moving_cli_common")
    script_path = SCRIPT_DIR / script_basename
    sys.argv = [str(script_path), *effective_argv]
    runpy.run_path(str(script_path), run_name="__main__")
