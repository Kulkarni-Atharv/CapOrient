from typing import List

import cv2
import numpy as np

import config


def segment_capsules(gray: np.ndarray, edge_map: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of binary masks (uint8, 0/255), one per candidate capsule.

    Pipeline:
      1. Fill edge regions to create solid blobs.
      2. Merge with Otsu and adaptive thresholds to cover low-contrast / transparent areas.
      3. Clean noise with morphological open + close.
      4. Run connected components and discard blobs that fail shape filters.
    """
    combined = _build_combined_mask(gray, edge_map)
    return _extract_valid_masks(gray, combined)


# ── helpers ──────────────────────────────────────────────────

def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    return clahe.apply(gray)


def _build_combined_mask(gray: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.SEG_FILL_KSIZE)
    filled = cv2.morphologyEx(
        edge_map, cv2.MORPH_CLOSE, kernel_fill, iterations=config.SEG_FILL_CLOSE_ITER
    )
    filled = cv2.morphologyEx(
        filled, cv2.MORPH_DILATE, kernel_fill, iterations=config.SEG_FILL_DILATE_ITER
    )

    enhanced = _enhance_contrast(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.ADAPTIVE_BLOCK_SIZE, config.ADAPTIVE_C,
    )

    combined = cv2.bitwise_or(filled, cv2.bitwise_or(otsu, adapt))

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.SEG_CLEAN_KSIZE)
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_OPEN, kernel_clean, iterations=config.SEG_CLEAN_OPEN_ITER
    )
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel_clean, iterations=config.SEG_CLEAN_CLOSE_ITER
    )
    return combined


def _extract_valid_masks(gray: np.ndarray, combined: np.ndarray) -> List[np.ndarray]:
    h, w = gray.shape
    min_area = config.MIN_AREA_FRACTION * h * w
    max_area = config.MAX_AREA_FRACTION * h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )

    masks = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        blob_mask = np.uint8(labels == lbl) * 255

        if not _passes_shape_filter(blob_mask):
            continue

        masks.append(blob_mask)

    return masks


def _passes_shape_filter(blob_mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return False

    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if rw == 0 or rh == 0:
        return False
    if max(rw, rh) / min(rw, rh) < config.MIN_ASPECT_RATIO:
        return False

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return False
    if cv2.contourArea(cnt) / hull_area < config.MIN_SOLIDITY:
        return False

    return True
