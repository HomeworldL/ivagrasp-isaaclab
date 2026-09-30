"""Torch-only quaternion helpers using the repository's ``wxyz`` convention."""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions in ``wxyz`` order."""
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return quaternion conjugates in ``wxyz`` order."""
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_inv(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return quaternion inverses in ``wxyz`` order."""
    conj = quat_conjugate(q)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True)
    return conj / torch.clamp(norm_sq, min=eps)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions in ``wxyz`` order."""
    q_xyz = q[..., 1:]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse quaternion."""
    return quat_apply(quat_inv(q), v)


def combine_frame_transforms(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``T_world_a`` and ``T_a_b`` into ``T_world_b``."""
    return pos_a + quat_apply(quat_a, pos_b), quat_mul(quat_a, quat_b)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Create quaternions from XYZ intrinsic Euler angles."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw

    cr = torch.cos(half_roll)
    sr = torch.sin(half_roll)
    cp = torch.cos(half_pitch)
    sp = torch.sin(half_pitch)
    cy = torch.cos(half_yaw)
    sy = torch.sin(half_yaw)

    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions in ``wxyz`` order."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3) rotation matrices, got {matrix.shape}.")

    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    quat = torch.zeros((*matrix.shape[:-2], 4), dtype=matrix.dtype, device=matrix.device)

    positive_trace = trace > 0.0
    if positive_trace.any():
        s = 2.0 * torch.sqrt(trace[positive_trace] + 1.0)
        quat[positive_trace, 0] = 0.25 * s
        quat[positive_trace, 1] = (m[positive_trace, 2, 1] - m[positive_trace, 1, 2]) / s
        quat[positive_trace, 2] = (m[positive_trace, 0, 2] - m[positive_trace, 2, 0]) / s
        quat[positive_trace, 3] = (m[positive_trace, 1, 0] - m[positive_trace, 0, 1]) / s

    cond_x = (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2]) & (~positive_trace)
    if cond_x.any():
        s = 2.0 * torch.sqrt(1.0 + m[cond_x, 0, 0] - m[cond_x, 1, 1] - m[cond_x, 2, 2])
        quat[cond_x, 0] = (m[cond_x, 2, 1] - m[cond_x, 1, 2]) / s
        quat[cond_x, 1] = 0.25 * s
        quat[cond_x, 2] = (m[cond_x, 0, 1] + m[cond_x, 1, 0]) / s
        quat[cond_x, 3] = (m[cond_x, 0, 2] + m[cond_x, 2, 0]) / s

    cond_y = (m[..., 1, 1] > m[..., 2, 2]) & (~positive_trace) & (~cond_x)
    if cond_y.any():
        s = 2.0 * torch.sqrt(1.0 + m[cond_y, 1, 1] - m[cond_y, 0, 0] - m[cond_y, 2, 2])
        quat[cond_y, 0] = (m[cond_y, 0, 2] - m[cond_y, 2, 0]) / s
        quat[cond_y, 1] = (m[cond_y, 0, 1] + m[cond_y, 1, 0]) / s
        quat[cond_y, 2] = 0.25 * s
        quat[cond_y, 3] = (m[cond_y, 1, 2] + m[cond_y, 2, 1]) / s

    cond_z = (~positive_trace) & (~cond_x) & (~cond_y)
    if cond_z.any():
        s = 2.0 * torch.sqrt(1.0 + m[cond_z, 2, 2] - m[cond_z, 0, 0] - m[cond_z, 1, 1])
        quat[cond_z, 0] = (m[cond_z, 1, 0] - m[cond_z, 0, 1]) / s
        quat[cond_z, 1] = (m[cond_z, 0, 2] + m[cond_z, 2, 0]) / s
        quat[cond_z, 2] = (m[cond_z, 1, 2] + m[cond_z, 2, 1]) / s
        quat[cond_z, 3] = 0.25 * s

    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-12)


def grasp_to_palm_transform(
    num_samples: int,
    device: str | torch.device,
    hand_transform_pos: tuple[float, float, float],
    hand_transform_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute grasp→palm transform from the hand's T_palm_grasp config.

    The hand config specifies the palm→grasp transform (position + quaternion).
    This function inverts it to produce the grasp→palm transform needed for
    home-pose sampling.
    """
    if len(hand_transform_pos) != 3:
        raise ValueError(f"hand_transform_pos must have length 3, got {len(hand_transform_pos)}")
    if len(hand_transform_quat) != 4:
        raise ValueError(f"hand_transform_quat must have length 4, got {len(hand_transform_quat)}")

    palm_to_grasp_pos = torch.tensor(hand_transform_pos, dtype=torch.float32, device=device).unsqueeze(0).expand(
        num_samples, -1
    )
    palm_to_grasp_quat = torch.tensor(hand_transform_quat, dtype=torch.float32, device=device).unsqueeze(0).expand(
        num_samples, -1
    )
    grasp_to_palm_quat = quat_inv(palm_to_grasp_quat)
    grasp_to_palm_pos = quat_apply(grasp_to_palm_quat, -palm_to_grasp_pos)
    return grasp_to_palm_pos, grasp_to_palm_quat
