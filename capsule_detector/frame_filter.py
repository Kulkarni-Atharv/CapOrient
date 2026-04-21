from typing import Optional

import cv2
import numpy as np

import config

# Tiny resolution used only for content-diff comparison — fast and sufficient.
_COMPARE_W, _COMPARE_H = 160, 120


class FrameFilter:
    """
    Gate that returns True from should_process() only when a new frame
    differs enough from the last processed frame to justify running the
    full detection pipeline.

    Internally it downscales and converts to grayscale before comparing,
    so the check itself is essentially free.
    """

    def __init__(self):
        self._last_gray: Optional[np.ndarray] = None

    def should_process(self, frame: np.ndarray) -> bool:
        """
        Compare *frame* to the internally stored reference.

        Returns True  (and updates the reference) when mean absolute
        pixel difference >= LIVE_FRAME_DIFF_THRESH.
        Returns False when the scene looks static.
        Always returns True on the very first call.
        """
        small = cv2.resize(frame, (_COMPARE_W, _COMPARE_H), interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._last_gray is None:
            self._last_gray = gray
            return True

        diff = float(np.mean(
            np.abs(gray.astype(np.float32) - self._last_gray.astype(np.float32))
        ))

        if diff >= config.LIVE_FRAME_DIFF_THRESH:
            self._last_gray = gray
            return True

        return False

    def reset(self) -> None:
        """Force the next call to should_process() to return True."""
        self._last_gray = None
