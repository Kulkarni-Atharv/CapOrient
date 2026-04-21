from dataclasses import dataclass
from typing import List


@dataclass
class CapsuleResult:
    capsule_id: int
    bounding_box: List[float]          # [x, y, width, height]
    orientation_angle_degrees: float
    sin_theta: float
    cos_theta: float
    major_axis_length: float
    minor_axis_length: float
    center: List[float]                # [x_center, y_center]
