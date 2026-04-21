import cv2
import numpy as np

import config


def build_edge_map(gray: np.ndarray) -> np.ndarray:
    """
    Fuse Canny and Sobel edges, then morphologically close gaps.

    Canny captures crisp, high-contrast edges; Sobel at a lower threshold
    catches the weak gradients produced by transparent capsules.
    """
    enhanced = _enhance_contrast(gray)
    canny    = _canny_edges(enhanced)
    sobel    = _sobel_edges(enhanced)

    edge_map = cv2.bitwise_or(canny, sobel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.EDGE_CLOSE_KSIZE)
    edge_map = cv2.morphologyEx(
        edge_map, cv2.MORPH_CLOSE, kernel, iterations=config.EDGE_CLOSE_ITER
    )
    return edge_map


# ── helpers ──────────────────────────────────────────────────

def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    return clahe.apply(gray)


def _canny_edges(enhanced: np.ndarray) -> np.ndarray:
    median_val = float(np.median(enhanced))
    lower = max(0,   int(config.CANNY_LOWER_FACTOR * median_val))
    upper = min(255, int(config.CANNY_UPPER_FACTOR * median_val))
    return cv2.Canny(enhanced, lower, upper)


def _sobel_edges(enhanced: np.ndarray) -> np.ndarray:
    sx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=config.SOBEL_KSIZE)
    sy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=config.SOBEL_KSIZE)
    mag = np.sqrt(sx**2 + sy**2)
    if mag.max() == 0:
        return np.zeros_like(enhanced)
    mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
    _, thresh = cv2.threshold(mag, config.SOBEL_THRESH, 255, cv2.THRESH_BINARY)
    return thresh
