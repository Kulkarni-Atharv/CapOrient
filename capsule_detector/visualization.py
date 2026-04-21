from typing import List

import cv2
import numpy as np

from capsule_detector.models import CapsuleResult


def draw_results(bgr: np.ndarray, results: List[CapsuleResult]) -> np.ndarray:
    """
    Return a copy of *bgr* annotated with bounding boxes, orientation arrows,
    and text labels for each detected capsule.
    """
    vis = bgr.copy()
    for r in results:
        _draw_capsule(vis, r)
    return vis


def _draw_capsule(vis: np.ndarray, r: CapsuleResult) -> None:
    x, y, w, h = [int(v) for v in r.bounding_box]
    cx, cy     = int(r.center[0]), int(r.center[1])
    half       = int(r.major_axis_length / 2)

    dx = int(half * r.cos_theta)
    dy = int(-half * r.sin_theta)   # image y-axis points downward

    cv2.arrowedLine(
        vis, (cx - dx, cy + dy), (cx + dx, cy - dy),
        (0, 255, 0), 2, tipLength=0.15,
    )
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)

    label = f"#{r.capsule_id}  {r.orientation_angle_degrees:.1f}deg"
    cv2.putText(
        vis, label, (x, max(y - 6, 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA,
    )
