"""
Smooth joint-space and Cartesian trajectories.

Author: Akram Hssaine (Enkefalos) - MIT License
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .kinematics import PlanarArm


def cubic(q0: Sequence[float], q1: Sequence[float], T: float, n: int = 100):
    """Cubic polynomial joint trajectory with zero start/end velocity.

    Returns (t, q, qd): times (n,), positions (n, dof), velocities (n, dof).
    """
    if T <= 0 or n < 2:
        raise ValueError("T must be > 0 and n >= 2")
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    t = np.linspace(0.0, T, n)
    s = t / T
    pos = 3 * s**2 - 2 * s**3          # 0 -> 1
    vel = (6 * s - 6 * s**2) / T       # d/dt
    dq = q1 - q0
    return t, q0 + np.outer(pos, dq), np.outer(vel, dq)


def cartesian_line(arm: PlanarArm, p0, p1, n: int = 50, q_start=None) -> np.ndarray:
    """Joint angles that move the end-effector along a straight line p0 -> p1."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    qs = []
    q = q_start
    for s in np.linspace(0.0, 1.0, n):
        q = arm.inverse(p0 + s * (p1 - p0), q0=q)
        qs.append(q)
    return np.array(qs)
