"""Interactive viewer that reuses the training dexgrasp_moving environment."""

from __future__ import annotations

from dexgrasp_moving_script_runner import run_float_entrypoint


def main() -> None:
    run_float_entrypoint("view_dexgrasp_float.py")


if __name__ == "__main__":
    main()
