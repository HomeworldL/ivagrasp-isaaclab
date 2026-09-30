"""Fixed BPS feature extraction for student partial point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_DEFAULT_BPS_SEED = 13
DEFAULT_BPS_NUM_BASIS = 128
DEFAULT_BPS_RADIUS = 0.45
_BPS_BASIS_DIR = Path(__file__).resolve().parent
_BPS_BASIS_CACHE: dict[Path, np.ndarray] = {}
_BPS_ENCODER_CACHE: dict[Path, "_FixedBPSDistEncoder"] = {}


def default_bps_basis_path(
    *,
    num_basis_points: int = DEFAULT_BPS_NUM_BASIS,
    radius: float = DEFAULT_BPS_RADIUS,
    random_seed: int = _DEFAULT_BPS_SEED,
) -> Path:
    """Return the default packaged BPS basis path."""
    radius_tag = str(radius).replace(".", "p")
    return _BPS_BASIS_DIR / f"bps_basis_random_uniform_r{radius_tag}_k{num_basis_points}_seed{random_seed}.npy"


def generate_uniform_ball_basis(
    *,
    num_basis_points: int,
    radius: float,
    n_dims: int = 3,
    random_seed: int = _DEFAULT_BPS_SEED,
) -> np.ndarray:
    """Match ``bps_torch.tools.sample_sphere_uniform`` for fixed basis generation."""
    if num_basis_points <= 0:
        raise ValueError(f"num_basis_points must be positive, got {num_basis_points}.")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}.")
    rng = np.random.default_rng(int(random_seed))
    samples = rng.normal(size=(int(num_basis_points), int(n_dims))).astype(np.float32)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(np.float32).eps)
    unit = samples / norms
    radii = rng.uniform(size=(int(num_basis_points), 1)).astype(np.float32)
    scale = np.power(radii, 1.0 / float(n_dims))
    return (float(radius) * unit * scale).astype(np.float32)


def _load_basis_array(basis_path: str | Path) -> np.ndarray:
    resolved = Path(basis_path).expanduser().resolve()
    if resolved not in _BPS_BASIS_CACHE:
        if not resolved.exists():
            raise FileNotFoundError(f"BPS basis file not found: {resolved}")
        basis = np.load(resolved).astype(np.float32)
        if basis.ndim != 2 or basis.shape[1] != 3:
            raise ValueError(f"BPS basis must have shape [K, 3], got {basis.shape}.")
        _BPS_BASIS_CACHE[resolved] = basis
    return _BPS_BASIS_CACHE[resolved]


class _FixedBPSDistEncoder:
    """Distance-only BPS encoder implemented directly with ``torch.cdist``."""

    def __init__(self, basis_path: str | Path) -> None:
        self.basis_path = Path(basis_path).expanduser().resolve()
        self.basis_np = _load_basis_array(self.basis_path)
        self._basis_torch: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def encode(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(f"BPS encoder expects [B, N, 3], got {tuple(point_cloud.shape)}.")
        cache_key = (point_cloud.device, point_cloud.dtype)
        if cache_key not in self._basis_torch:
            basis = torch.from_numpy(self.basis_np).to(device=point_cloud.device, dtype=point_cloud.dtype)
            self._basis_torch[cache_key] = basis.unsqueeze(0)
        basis = self._basis_torch[cache_key].expand(point_cloud.shape[0], -1, -1)
        return torch.cdist(basis, point_cloud).amin(dim=-1)


def encode_point_cloud_bps_dists(
    point_cloud: torch.Tensor,
    *,
    basis_path: str | Path,
) -> torch.Tensor:
    """Encode ``[B, N, 3]`` point clouds into fixed BPS distance features."""
    resolved = Path(basis_path).expanduser().resolve()
    encoder = _BPS_ENCODER_CACHE.get(resolved)
    if encoder is None:
        encoder = _FixedBPSDistEncoder(resolved)
        _BPS_ENCODER_CACHE[resolved] = encoder
    return encoder.encode(point_cloud)
