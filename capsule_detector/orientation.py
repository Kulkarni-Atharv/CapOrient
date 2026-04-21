from typing import Optional

import cv2
import numpy as np

from capsule_detector.models import CapsuleResult


def measure_orientation(mask: np.ndarray) -> Optional[CapsuleResult]:
    """
    Fit an ellipse to the largest contour of a binary mask and return a
    CapsuleResult with orientation angle θ (from horizontal), sin(θ), cos(θ),
    axis lengths, centre, and bounding box.

    Returns None if the contour has fewer than 5 points or the ellipse fit fails.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(cnt)
    except cv2.error:
        return None

    (cx, cy), (minor_ax, major_ax), angle_cv = ellipse

    # cv2.fitEllipse returns the angle of the major axis measured from the
    # vertical (0° = vertical, 90° = horizontal). Swap axes if necessary so
    # major_ax is always the longer dimension.
    if major_ax < minor_ax:
        major_ax, minor_ax = minor_ax, major_ax
        angle_cv = (angle_cv + 90) % 180

    # Remap to θ: CCW angle from the horizontal axis, range (−90°, 90°].
    theta_deg = 90.0 - angle_cv
    theta_deg = ((theta_deg + 90) % 180) - 90

    theta_rad = np.deg2rad(theta_deg)
    sin_t = float(np.sin(theta_rad))
    cos_t = float(np.cos(theta_rad))

    x, y, bw, bh = cv2.boundingRect(cnt)

    return CapsuleResult(
        capsule_id=0,
        bounding_box=[float(x), float(y), float(bw), float(bh)],
        orientation_angle_degrees=round(theta_deg, 4),
        sin_theta=round(sin_t, 6),
        cos_theta=round(cos_t, 6),
        major_axis_length=round(float(major_ax), 2),
        minor_axis_length=round(float(minor_ax), 2),
        center=[round(float(cx), 2), round(float(cy), 2)],
    )
