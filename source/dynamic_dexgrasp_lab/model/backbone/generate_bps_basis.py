"""Generate fixed BPS basis files for student observation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .bps import (
        _DEFAULT_BPS_SEED,
        DEFAULT_BPS_NUM_BASIS,
        DEFAULT_BPS_RADIUS,
        default_bps_basis_path,
        generate_uniform_ball_basis,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from dynamic_dexgrasp_lab.model.backbone.bps import (  # type: ignore
        _DEFAULT_BPS_SEED,
        DEFAULT_BPS_NUM_BASIS,
        DEFAULT_BPS_RADIUS,
        default_bps_basis_path,
        generate_uniform_ball_basis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fixed random-uniform BPS basis.")
    parser.add_argument("--num-basis-points", type=int, default=DEFAULT_BPS_NUM_BASIS)
    parser.add_argument("--radius", type=float, default=DEFAULT_BPS_RADIUS)
    parser.add_argument("--random-seed", type=int, default=_DEFAULT_BPS_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basis = generate_uniform_ball_basis(
        num_basis_points=args.num_basis_points,
        radius=args.radius,
        random_seed=args.random_seed,
    )
    output_path = args.output
    if output_path is None:
        output_path = default_bps_basis_path(
            num_basis_points=args.num_basis_points,
            radius=args.radius,
            random_seed=args.random_seed,
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, basis.astype(np.float32))
    print(output_path)


if __name__ == "__main__":
    main()
