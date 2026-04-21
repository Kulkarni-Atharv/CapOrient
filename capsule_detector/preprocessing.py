from typing import Tuple

import cv2
import numpy as np

import config


def preprocess_frame(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight preprocessing for live video frames.

    Uses a cheap Gaussian blur instead of the slow NlMeans denoising so that
    per-frame CPU cost is negligible.  Returns (bgr_original, gray_smoothed).
    """
    smoothed = cv2.GaussianBlur(bgr, (5, 5), 0)
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an image from disk, denoise it, and return (bgr, gray).

    Raises FileNotFoundError if the path cannot be read by OpenCV.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None,
        config.DENOISE_H,
        config.DENOISE_H_COLOR,
        config.DENOISE_TEMPLATE_WS,
        config.DENOISE_SEARCH_WS,
    )
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    return bgr, gray
