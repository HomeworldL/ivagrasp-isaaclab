"""Simplified play entrypoint for dexgrasp_moving tasks."""

from __future__ import annotations

from dexgrasp_moving_script_runner import run_float_entrypoint


def main() -> None:
    run_float_entrypoint("play_dexgrasp_float.py")


if __name__ == "__main__":
    main()
