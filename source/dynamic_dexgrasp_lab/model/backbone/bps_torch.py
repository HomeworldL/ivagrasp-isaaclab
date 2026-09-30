"""Optional ``bps_torch`` adapter kept separate from the repo-local fixed BPS implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def encode_point_cloud_bps_dists_torch(
    point_cloud: torch.Tensor,
    *,
    basis_path: str | Path,
) -> torch.Tensor:
    """Encode ``[B, N, 3]`` point clouds with ``bps_torch`` custom-basis distances."""
    from bps_torch.bps import bps_torch

    resolved = Path(basis_path).expanduser().resolve()
    basis = np.load(resolved).astype(np.float32)
    encoder = bps_torch(
        bps_type="custom",
        n_bps_points=int(basis.shape[0]),
        radius=1.0,
        n_dims=3,
        custom_basis=basis,
    )
    encoded = encoder.encode(point_cloud, feature_type=["dists"])
    return encoded["dists"].to(device=point_cloud.device, dtype=point_cloud.dtype)
