"""Convert dual Piper-X joint14 into Alpha EEF20 using the supplied FK."""

from __future__ import annotations

import numpy as np

from openwam.dataloader.transforms.piperx_fk import PiperXForwardKinematics

JOINT14_DIM = 14
EEF20_DIM = 20
_FK = PiperXForwardKinematics()


def joint14_to_eef20(values: np.ndarray) -> np.ndarray:
    """Joint radians -> base-frame flange metres, rot6d columns, native grip.

    Layout: [L joints6, L grip, R joints6, R grip] ->
    [L xyz3, L rot6d6, L grip, R xyz3, R rot6d6, R grip].
    Min-max normalization subsequently maps each gripper to [-1, +1].
    """
    joints = np.asarray(values, dtype=np.float64)
    if joints.ndim == 0 or joints.shape[-1] != JOINT14_DIM:
        raise ValueError(f"Dual Piper-X values must have shape (..., 14), got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("Dual Piper-X values contain NaN or infinity")
    arms = []
    for offset in (0, 7):
        transform = _FK.compute_fk_matrix(joints[..., offset : offset + 6])
        rot6d = np.swapaxes(transform[..., :3, :2], -2, -1).reshape(*joints.shape[:-1], 6)
        arms.append(
            np.concatenate(
                [
                    transform[..., :3, 3],
                    rot6d,
                    joints[..., offset + 6 : offset + 7],
                ],
                axis=-1,
            )
        )
    return np.concatenate(arms, axis=-1).astype(np.float32)
