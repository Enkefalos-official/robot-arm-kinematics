import numpy as np
import pytest

from robot_arm import PlanarArm, cartesian_line, cubic


def test_forward_straight_arm():
    arm = PlanarArm([1.0, 0.5])
    x, y, phi = arm.forward([0, 0])
    assert np.allclose([x, y, phi], [1.5, 0, 0])


def test_forward_right_angle():
    arm = PlanarArm([1.0, 1.0])
    x, y, _ = arm.forward([np.pi / 2, -np.pi / 2])
    assert np.allclose([x, y], [1.0, 1.0])


@pytest.mark.parametrize("elbow_up", [True, False])
def test_inverse_2link_roundtrip(elbow_up):
    arm = PlanarArm([0.4, 0.3])
    for target in [(0.5, 0.1), (0.2, 0.4), (-0.3, 0.3), (0.1, -0.5)]:
        q = arm.inverse_2link(*target, elbow_up=elbow_up)
        assert np.allclose(arm.forward(q)[:2], target, atol=1e-9)


def test_inverse_2link_out_of_reach():
    with pytest.raises(ValueError):
        PlanarArm([1, 1]).inverse_2link(3, 0)


def test_numeric_ik_3link():
    arm = PlanarArm([0.5, 0.4, 0.3])
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_true = rng.uniform(-1.5, 1.5, 3)
        target = arm.forward(q_true)[:2]
        q = arm.inverse(target)
        assert np.allclose(arm.forward(q)[:2], target, atol=1e-5)


def test_jacobian_matches_finite_difference():
    arm = PlanarArm([0.5, 0.4, 0.3])
    q = np.array([0.3, -0.7, 1.1])
    J = arm.jacobian(q)
    eps = 1e-6
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        fd = (arm.forward(q + dq)[:2] - arm.forward(q - dq)[:2]) / (2 * eps)
        assert np.allclose(J[:, i], fd, atol=1e-6)


def test_cubic_endpoints():
    t, q, qd = cubic([0, 0], [1, -1], T=2.0, n=50)
    assert np.allclose(q[0], [0, 0]) and np.allclose(q[-1], [1, -1])
    assert np.allclose(qd[0], 0) and np.allclose(qd[-1], 0)


def test_cartesian_line_stays_on_line():
    arm = PlanarArm([0.5, 0.4, 0.3])
    qs = cartesian_line(arm, (0.6, 0.2), (0.3, 0.6), n=20)
    pts = np.array([arm.forward(q)[:2] for q in qs])
    d = np.array([0.3, 0.6]) - np.array([0.6, 0.2])
    normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    assert np.allclose((pts - [0.6, 0.2]) @ normal, 0, atol=1e-5)
