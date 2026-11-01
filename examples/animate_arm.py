"""
Animate a 3-link arm drawing a square with its end-effector.

    python examples/animate_arm.py            # live window
    python examples/animate_arm.py --gif      # save arm.gif
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_arm import PlanarArm, cartesian_line  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true", help="save to arm.gif instead of showing")
    args = ap.parse_args()

    arm = PlanarArm([0.5, 0.4, 0.3])
    corners = [(0.7, 0.0), (0.7, 0.5), (0.2, 0.5), (0.2, 0.0), (0.7, 0.0)]
    qs, q = [], None
    for a, b in zip(corners, corners[1:]):
        seg = cartesian_line(arm, a, b, n=40, q_start=q)
        q = seg[-1]
        qs.extend(seg)
    qs = np.array(qs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.4, 1.3)
    ax.set_ylim(-0.6, 1.1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("3-link planar arm - Cartesian path")
    (links,) = ax.plot([], [], "o-", lw=4, ms=8, color="#3b82f6")
    (trail,) = ax.plot([], [], "-", lw=1.5, color="#ef4444")
    path = []

    def update(i):
        pts = arm.joint_positions(qs[i])
        links.set_data(pts[:, 0], pts[:, 1])
        path.append(pts[-1])
        p = np.array(path)
        trail.set_data(p[:, 0], p[:, 1])
        return links, trail

    anim = FuncAnimation(fig, update, frames=len(qs), interval=25, blit=True)
    if args.gif:
        anim.save("arm.gif", writer=PillowWriter(fps=30))
        print("saved arm.gif")
    else:
        plt.show()


if __name__ == "__main__":
    main()
