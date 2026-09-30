"""Unit tests covering core MDP math: quaternions, EMA actions, home-pose transforms.

These tests use only torch and standard library — no IsaacLab runtime required.
"""

import pytest
import torch

from dynamic_dexgrasp_lab.utils.torch_quat import (
    combine_frame_transforms,
    grasp_to_palm_transform,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_inv,
    quat_mul,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _quat_from_z_axis_to_vector(vectors: torch.Tensor) -> torch.Tensor:
    """Local copy of the command helper for offline testing."""
    z_axis = vectors.new_tensor((0.0, 0.0, 1.0)).expand_as(vectors)
    directions = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)
    cross = torch.cross(z_axis, directions, dim=-1)
    dot = torch.sum(z_axis * directions, dim=-1, keepdim=True)
    quat = torch.cat((1.0 + dot, cross), dim=-1)
    opposite_mask = dot.squeeze(-1) < -0.9999
    if torch.any(opposite_mask):
        quat[opposite_mask] = quat.new_tensor((0.0, 1.0, 0.0, 0.0))
    return torch.nn.functional.normalize(quat, dim=-1, eps=1e-8)


def _ema_target(action, scale, offset, prev_applied, alpha):
    """Local copy of the EMA formula used by EMAJointPositionOffsetScaleAction."""
    target = action * scale + offset
    return alpha * target + (1.0 - alpha) * prev_applied


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

class TestQuatMul:
    def test_identity_left(self):
        q = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = quat_mul(IDENTITY.unsqueeze(0), q)
        assert torch.allclose(result, q / torch.linalg.norm(q, dim=-1))

    def test_double_cover(self):
        q = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # 180° about Y
        result = quat_mul(q, q)
        assert torch.allclose(result, -IDENTITY.unsqueeze(0))  # double cover


class TestQuatConjugate:
    def test_conj_identity_is_identity(self):
        result = quat_conjugate(IDENTITY.unsqueeze(0))
        assert torch.allclose(result, IDENTITY.unsqueeze(0))


class TestQuatInv:
    def test_roundtrip(self):
        q = quat_from_euler_xyz(torch.tensor([0.3]), torch.tensor([-0.5]), torch.tensor([1.2]))
        inv = quat_inv(q)
        prod = quat_mul(q, inv)
        assert torch.allclose(prod.abs(), IDENTITY.unsqueeze(0), atol=1e-6)


class TestQuatApply:
    def test_identity_does_not_rotate(self):
        v = torch.tensor([[1.0, 2.0, 3.0]])
        result = quat_apply(IDENTITY.unsqueeze(0), v)
        assert torch.allclose(result, v)

    def test_apply_and_inverse_roundtrip(self):
        q = quat_from_euler_xyz(torch.tensor([0.5]), torch.tensor([0.3]), torch.tensor([-0.8]))
        v = torch.tensor([[1.0, 0.0, 0.0]])
        rotated = quat_apply(q, v)
        recovered = quat_apply_inverse(q, rotated)
        assert torch.allclose(recovered, v, atol=1e-6)


class TestQuatFromEulerXYZ:
    @pytest.mark.parametrize(
        "roll,pitch,yaw",
        [(0.0, 0.0, 0.0), (1.57, 0.0, 0.0), (0.0, 1.57, 0.0), (0.0, 0.0, 1.57)],
    )
    def test_unit_length(self, roll, pitch, yaw):
        q = quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))
        assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(1))


class TestQuatFromMatrix:
    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="Expected .* rotation matrices"):
            quat_from_matrix(torch.randn(2, 4))

    def test_identity_roundtrip(self):
        mat = torch.eye(3).unsqueeze(0)
        q = quat_from_matrix(mat)
        assert torch.allclose(q.abs(), IDENTITY.unsqueeze(0), atol=1e-6)


class TestCombineFrameTransforms:
    def test_identity_identity(self):
        pos, quat = combine_frame_transforms(
            torch.zeros(1, 3), IDENTITY.unsqueeze(0),
            torch.zeros(1, 3), IDENTITY.unsqueeze(0),
        )
        assert torch.allclose(pos, torch.zeros(1, 3))
        assert torch.allclose(quat, IDENTITY.unsqueeze(0))


# ---------------------------------------------------------------------------
# _quat_from_z_axis_to_vector
# ---------------------------------------------------------------------------

class TestQuatFromZAxisToVector:
    def test_z_to_z_is_identity(self):
        q = _quat_from_z_axis_to_vector(torch.tensor([[0.0, 0.0, 1.0]]))
        assert torch.allclose(q.abs(), IDENTITY.unsqueeze(0), atol=1e-6)

    def test_z_to_neg_z_is_rot180_x(self):
        """Degenerate case: +Z → -Z uses 180° about X axis (w=0, x=1, y=0, z=0)."""
        q = _quat_from_z_axis_to_vector(torch.tensor([[0.0, 0.0, -1.0]]))
        assert torch.allclose(q.abs(), torch.tensor([[0.0, 1.0, 0.0, 0.0]]), atol=1e-6)

    def test_batch(self):
        vecs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        q = _quat_from_z_axis_to_vector(vecs)
        assert q.shape == (3, 4)
        assert torch.allclose(torch.linalg.norm(q, dim=-1), torch.ones(3))


# ---------------------------------------------------------------------------
# grasp_to_palm_transform
# ---------------------------------------------------------------------------

class TestGraspToPalmTransform:
    def test_identity_transform(self):
        pos, quat = grasp_to_palm_transform(
            2, "cpu",
            hand_transform_pos=(0.0, 0.0, 0.0),
            hand_transform_quat=(1.0, 0.0, 0.0, 0.0),
        )
        assert torch.allclose(pos, torch.zeros(2, 3))
        assert torch.allclose(quat, IDENTITY.unsqueeze(0).expand(2, -1))

    def test_bad_pos_length_raises(self):
        with pytest.raises(ValueError, match="hand_transform_pos must have length 3"):
            grasp_to_palm_transform(1, "cpu", (0.0, 0.0), (1.0, 0.0, 0.0, 0.0))

    def test_bad_quat_length_raises(self):
        with pytest.raises(ValueError, match="hand_transform_quat must have length 4"):
            grasp_to_palm_transform(1, "cpu", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# EMA action offset / scale math
# ---------------------------------------------------------------------------

class TestEMAActionMath:
    def test_zero_action_yields_offset_with_alpha_one(self):
        result = _ema_target(action=0.0, scale=0.1, offset=0.3, prev_applied=0.3, alpha=1.0)
        assert result == pytest.approx(0.3)

    def test_ema_blends_with_prev(self):
        """First step with alpha=0.95 blends 95% offset + 5% prev."""
        result = _ema_target(action=0.0, scale=0.1, offset=0.3, prev_applied=0.0, alpha=0.95)
        assert result == pytest.approx(0.285)

    def test_repeated_steps_converge_to_offset(self):
        prev = 0.0
        for _ in range(100):
            prev = _ema_target(action=0.0, scale=0.1, offset=0.3, prev_applied=prev, alpha=0.95)
        assert prev == pytest.approx(0.3, abs=1e-5)

    def test_negative_action(self):
        result = _ema_target(action=-1.0, scale=0.1, offset=0.3, prev_applied=0.3, alpha=1.0)
        assert result == pytest.approx(0.2)
