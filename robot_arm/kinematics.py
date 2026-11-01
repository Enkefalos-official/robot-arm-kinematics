"""
Forward and inverse kinematics for planar serial robot arms.

Author: Akram Hssaine (Enkefalos) - MIT License
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class PlanarArm:
    """An N-link planar arm with revolute joints.

    link_lengths : length of each link (m)
    joint_limits : optional (min, max) in radians for each joint
    """

    link_lengths: Sequence[float]
    joint_limits: Sequence[tuple[float, float]] | None = None
    _L: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._L = np.asarray(self.link_lengths, dtype=float)
        if self._L.ndim != 1 or len(self._L) == 0 or np.any(self._L <= 0):
            raise ValueError("link_lengths must be a non-empty list of positive numbers")
        if self.joint_limits is not None and len(self.joint_limits) != self.dof:
            raise ValueError("joint_limits must have one (min, max) pair per joint")

    # ------------------------------------------------------------------
    @property
    def dof(self) -> int:
        return len(self._L)

    @property
    def reach(self) -> float:
        return float(self._L.sum())

    # ------------------------------------------------------------------
    def joint_positions(self, q: Sequence[float]) -> np.ndarray:
        """Return (N+1, 2) array with the base and every joint/end-effector position."""
        q = self._check_q(q)
        angles = np.cumsum(q)
        steps = np.stack([self._L * np.cos(angles), self._L * np.sin(angles)], axis=1)
        return np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])

    def forward(self, q: Sequence[float]) -> np.ndarray:
        """End-effector pose (x, y, phi)."""
        q = self._check_q(q)
        x, y = self.joint_positions(q)[-1]
        return np.array([x, y, float(np.sum(q))])

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """2xN position Jacobian of the end-effector."""
        q = self._check_q(q)
        angles = np.cumsum(q)
        J = np.zeros((2, self.dof))
        for i in range(self.dof):
            J[0, i] = -np.sum(self._L[i:] * np.sin(angles[i:]))
            J[1, i] = np.sum(self._L[i:] * np.cos(angles[i:]))
        return J

    # ------------------------------------------------------------------
    def inverse_2link(self, x: float, y: float, elbow_up: bool = True) -> np.ndarray:
        """Closed-form IK for a 2-link arm."""
        if self.dof != 2:
            raise ValueError("inverse_2link requires a 2-link arm")
        l1, l2 = self._L
        d2 = x * x + y * y
        c2 = (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if abs(c2) > 1 + 1e-9:
            raise ValueError(f"target ({x:.3f}, {y:.3f}) is out of reach")
        c2 = float(np.clip(c2, -1.0, 1.0))
        s2 = np.sqrt(1 - c2 * c2) * (1 if elbow_up else -1)
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        return np.array([wrap(q1), wrap(q2)])

    def inverse(
        self,
        target: Sequence[float],
        q0: Sequence[float] | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
        damping: float = 0.05,
    ) -> np.ndarray:
        """Numerical IK (damped least squares) for any number of links.

        Raises ValueError if the target cannot be reached within `tol`.
        """
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(target) > self.reach + 1e-9:
            raise ValueError("target is out of reach")
        q = np.full(self.dof, 0.3) if q0 is None else self._check_q(q0).copy()
        lam2 = damping ** 2
        for _ in range(max_iter):
            err = target - self.forward(q)[:2]
            if np.linalg.norm(err) < tol:
                return np.array([wrap(a) for a in q])
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(2), err)
            q = self._clamp(q + dq)
        raise ValueError("IK did not converge")

    # ------------------------------------------------------------------
    def _check_q(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"expected {self.dof} joint angles, got shape {q.shape}")
        return q

    def _clamp(self, q: np.ndarray) -> np.ndarray:
        if self.joint_limits is None:
            return q
        lo, hi = np.array(self.joint_limits).T
        return np.clip(q, lo, hi)


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)
