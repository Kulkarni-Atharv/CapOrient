"""
capsule_detector — public API
"""

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import sys

import cv2

from capsule_detector.preprocessing  import load_and_preprocess
from capsule_detector.edge_detection  import build_edge_map
from capsule_detector.segmentation    import segment_capsules
from capsule_detector.orientation     import measure_orientation
from capsule_detector.postprocessing  import nms
from capsule_detector.visualization   import draw_results
from capsule_detector.models          import CapsuleResult


def detect_capsules(image_path: str) -> List[dict]:
    """
    Detect all capsules in *image_path* and return their properties as a list
    of dicts matching the JSON output schema.
    """
    bgr, gray = load_and_preprocess(image_path)
    edge_map  = build_edge_map(gray)
    masks     = segment_capsules(gray, edge_map)

    raw: List[CapsuleResult] = [
        r for mask in masks
        for r in [measure_orientation(mask)]
        if r is not None
    ]

    return [asdict(r) for r in nms(raw)]


def detect_and_visualise(
    image_path: str,
    output_path: Optional[str] = None,
) -> List[dict]:
    """
    Detect capsules, save an annotated image, and return the same list of dicts.

    If *output_path* is None the annotated image is written next to the input
    file with ``_annotated`` appended to the stem.
    """
    bgr, gray = load_and_preprocess(image_path)
    edge_map  = build_edge_map(gray)
    masks     = segment_capsules(gray, edge_map)

    raw: List[CapsuleResult] = [
        r for mask in masks
        for r in [measure_orientation(mask)]
        if r is not None
    ]

    results = nms(raw)
    vis     = draw_results(bgr, results)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / (p.stem + "_annotated" + p.suffix))

    cv2.imwrite(output_path, vis)
    print(f"Annotated image saved → {output_path}", file=sys.stderr)

    return [asdict(r) for r in results]


__all__ = ["detect_capsules", "detect_and_visualise"]
