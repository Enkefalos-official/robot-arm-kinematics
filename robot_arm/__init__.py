"""robot_arm - kinematics and trajectory planning for planar robot arms."""
from .kinematics import PlanarArm, wrap
from .trajectory import cartesian_line, cubic

__all__ = ["PlanarArm", "wrap", "cubic", "cartesian_line"]
__version__ = "0.1.0"
