from typing import List

from capsule_detector.models import CapsuleResult
import config


def nms(results: List[CapsuleResult]) -> List[CapsuleResult]:
    """
    Non-maximum suppression: when two detections overlap above the IoU
    threshold, keep only the one with the larger major axis.
    Re-assigns capsule_id sequentially after filtering.
    """
    kept: List[CapsuleResult] = []
    for r in sorted(results, key=lambda r: r.major_axis_length, reverse=True):
        if all(_iou(r, k) < config.NMS_IOU_THRESHOLD for k in kept):
            kept.append(r)

    for i, r in enumerate(kept, 1):
        r.capsule_id = i

    return kept


def _iou(a: CapsuleResult, b: CapsuleResult) -> float:
    ax, ay, aw, ah = a.bounding_box
    bx, by, bw, bh = b.bounding_box
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
