"""
Live capsule orientation detection — time-based trigger.

Camera backend
--------------
Uses Picamera2 (libcamera stack) which is the only correct way to access
the Raspberry Pi Camera Module on Pi OS Bullseye / Bookworm.
cv2.VideoCapture cannot open CSI cameras on these OS versions.

Install:  sudo apt install -y python3-picamera2

How it works
------------
Two threads run concurrently:

  Main thread  (display loop)
    • Reads frames from Picamera2 as fast as possible.
    • Shares each frame with the detection thread via _FrameSlot.
    • Draws the LAST STORED detection results on the freshest frame.
    • Handles keyboard: Q = quit, D = immediate detection.

  Background thread  (detection loop)
    • Spends most of its life sleeping inside threading.Event.wait().
    • Wakes up in exactly two situations:
        1. LIVE_DETECTION_INTERVAL seconds have elapsed  (timed trigger)
        2. The user pressed D                            (manual trigger)
    • Runs the full pipeline once, stores the results, resets the timer,
      then goes back to sleep.

Why Event.wait(timeout) instead of time.sleep()
------------------------------------------------
time.sleep(30) cannot be interrupted mid-sleep.
Event.wait(timeout=remaining) releases the GIL and returns instantly when
the event is set — so a D keypress always takes effect immediately
regardless of where in the 30 s window it arrives.

CPU cost while idle: essentially zero — the background thread is
blocked inside the OS wait primitive, not spinning.
"""

import sys
import threading
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2           # pip: sudo apt install python3-picamera2

import config
from capsule_detector.edge_detection import build_edge_map
from capsule_detector.models         import CapsuleResult
from capsule_detector.orientation    import measure_orientation
from capsule_detector.postprocessing import nms
from capsule_detector.preprocessing  import preprocess_frame
from capsule_detector.segmentation   import segment_capsules
from capsule_detector.visualization  import draw_results

# NOTE: FrameFilter (frame-differencing) is intentionally NOT imported.
#       Detection is now triggered purely by time or manual keypress.


# ─────────────────────────────────────────────────────────────
# Thread-safe state holders
# ─────────────────────────────────────────────────────────────

class _FrameSlot:
    """
    Holds the single most-recent camera frame.
    Overwrites on every put() so the detection thread always works on the
    latest available frame — no queue, no backlog.
    """

    def __init__(self):
        self._lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            # Return a copy so the detection thread owns its own buffer
            # and the display thread can keep writing new frames safely.
            return self._frame.copy() if self._frame is not None else None


class _ResultSlot:
    """Holds the most recent detection results."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._results: List[dict] = []

    def put(self, results: List[dict]) -> None:
        with self._lock:
            self._results = results

    def get(self) -> List[dict]:
        with self._lock:
            return list(self._results)


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class LiveDetector:
    """
    Runs capsule orientation detection on a live Picamera2 feed.
    Detection fires on a fixed time interval or on a manual D keypress.
    """

    def __init__(self, camera_index: int = config.LIVE_CAMERA_INDEX):
        # camera_index maps directly to Picamera2(camera_num=...)
        self._camera_num   = camera_index
        self._frame_slot   = _FrameSlot()
        self._result_slot  = _ResultSlot()

        # ── time-based trigger ────────────────────────────────────
        # Initialise to 0 so detection fires immediately on the first frame
        # rather than waiting a full interval before showing any result.
        self._last_detection_time: float = 0.0

        # threading.Event used for the manual D trigger.
        #   Detection thread : .wait(timeout=remaining_seconds)
        #   Display thread   : .set()  on D keypress
        #   Detection thread : .clear() immediately after waking
        self._manual_trigger = threading.Event()
        # ─────────────────────────────────────────────────────────

        self._running     = False
        self._proc_thread: Optional[threading.Thread] = None

        # Stats written by detection thread, read by display thread.
        # Simple float/bool writes are GIL-atomic; no lock needed.
        self._proc_ms:      float = 0.0   # duration of last detection pass
        self._is_detecting: bool  = False  # True while pipeline is running

    # ── public ────────────────────────────────────────────────

    def run(self) -> None:
        """
        Initialise Picamera2, start the background detection thread, and
        run the display loop.  Blocks until the user presses Q.
        """
        # ── configure Picamera2 ───────────────────────────────────
        picam2 = Picamera2(camera_num=self._camera_num)

        # RGB888 gives a plain (H, W, 3) uint8 array — easy to convert to BGR.
        # preview_configuration keeps CPU + memory usage low vs. still capture.
        cam_cfg = picam2.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
        )
        picam2.configure(cam_cfg)
        picam2.start()
        # ─────────────────────────────────────────────────────────

        self._running = True
        self._proc_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="capsule-detector",
        )
        self._proc_thread.start()

        try:
            self._display_loop(picam2)
        finally:
            self._running = False
            # Wake the sleeping detection thread so it can exit cleanly
            # instead of waiting until the next interval timeout.
            self._manual_trigger.set()
            picam2.stop()
            cv2.destroyAllWindows()

    # ── display loop  (main thread) ───────────────────────────

    def _display_loop(self, picam2: Picamera2) -> None:
        wait_ms = max(1, int(1000 / config.LIVE_DISPLAY_FPS))

        while True:
            # capture_array() grabs the latest completed frame from the
            # Picamera2 buffer — non-blocking and always fresh.
            # Picamera2 returns RGB; convert to BGR for OpenCV.
            rgb   = picam2.capture_array()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Share the latest frame with the detection thread.
            self._frame_slot.put(frame)

            # Draw last known results onto the CURRENT (freshest) frame.
            # Drawing is always fast (< 1 ms) — never blocks the display.
            results   = self._result_slot.get()
            annotated = draw_results(frame, _dicts_to_results(results))
            _draw_hud(
                annotated,
                n_capsules   = len(results),
                proc_ms      = self._proc_ms,
                is_detecting = self._is_detecting,
                next_in_secs = self._seconds_until_next(),
            )

            cv2.imshow(
                "Capsule Orientation — Live  [D = detect now | Q = quit]",
                annotated,
            )

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("d") and not self._is_detecting:
                # Manual trigger: wake the detection thread immediately.
                # The detection thread clears the event after waking.
                self._manual_trigger.set()

    # ── detection loop  (background thread) ───────────────────

    def _detection_loop(self) -> None:
        while self._running:

            # ── compute how long to sleep until next scheduled detection ──
            elapsed_since_last = time.time() - self._last_detection_time
            remaining = config.LIVE_DETECTION_INTERVAL - elapsed_since_last

            if remaining > 0:
                # Block here — zero CPU cost — until either:
                #   (a) remaining seconds elapse naturally  → timed trigger
                #   (b) display thread calls .set()         → manual trigger (D key)
                self._manual_trigger.wait(timeout=remaining)

            # Consume the event so a stale set() from a previous D keypress
            # does not cause an unexpected extra detection next iteration.
            self._manual_trigger.clear()

            # run() may have set _running=False and fired the event just to
            # unblock this thread — check before doing any real work.
            if not self._running:
                break

            # ── grab the latest frame from the display thread ─────────
            frame = self._frame_slot.get()
            if frame is None:
                # Picamera2 hasn't produced a frame yet; retry shortly.
                time.sleep(0.1)
                continue

            # ── run the full detection pipeline ───────────────────────
            self._is_detecting = True
            t0 = time.perf_counter()

            results = _run_pipeline(frame)

            self._proc_ms      = (time.perf_counter() - t0) * 1000
            self._is_detecting = False

            # Persist results and reset the interval clock.
            self._result_slot.put(results)
            self._last_detection_time = time.time()   # ← resets the 30 s timer

    # ── internal helper ───────────────────────────────────────

    def _seconds_until_next(self) -> float:
        """Seconds remaining until the next scheduled detection (minimum 0)."""
        elapsed = time.time() - self._last_detection_time
        return max(0.0, config.LIVE_DETECTION_INTERVAL - elapsed)


# ─────────────────────────────────────────────────────────────
# Detection pipeline  (module-level, stateless)
# ─────────────────────────────────────────────────────────────

def _run_pipeline(frame: np.ndarray) -> List[dict]:
    """
    Full detection pass on a single BGR frame.
    Downscales to LIVE_PROCESS_WIDTH, runs the pipeline, then scales all
    result coordinates back to the original frame dimensions.
    """
    h_orig, w_orig = frame.shape[:2]
    pw, ph = _process_size(w_orig, h_orig)

    small     = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    _, gray_p = preprocess_frame(small)   # cheap Gaussian blur + grayscale

    edge_map = build_edge_map(gray_p)
    masks    = segment_capsules(gray_p, edge_map)

    raw: List[CapsuleResult] = [
        r for mask in masks
        for r in [measure_orientation(mask)]
        if r is not None
    ]
    results = nms(raw)

    _scale_results(results, sx=w_orig / pw, sy=h_orig / ph)
    return [asdict(r) for r in results]


def _process_size(w: int, h: int) -> Tuple[int, int]:
    """Return (pw, ph) at LIVE_PROCESS_WIDTH, preserving aspect ratio."""
    pw = min(w, config.LIVE_PROCESS_WIDTH)
    ph = max(1, int(round(h * pw / w)))
    return pw, ph


def _scale_results(results: List[CapsuleResult], sx: float, sy: float) -> None:
    """Scale all spatial fields from process-space to display-space in-place."""
    s_len = (sx + sy) / 2
    for r in results:
        r.bounding_box[0] *= sx;  r.bounding_box[1] *= sy
        r.bounding_box[2] *= sx;  r.bounding_box[3] *= sy
        r.center[0]        *= sx;  r.center[1]        *= sy
        r.major_axis_length *= s_len
        r.minor_axis_length *= s_len


def _dicts_to_results(dicts: List[dict]) -> List[CapsuleResult]:
    return [CapsuleResult(**d) for d in dicts]


# ─────────────────────────────────────────────────────────────
# HUD overlay
# ─────────────────────────────────────────────────────────────

def _draw_hud(
    frame:        np.ndarray,
    n_capsules:   int,
    proc_ms:      float,
    is_detecting: bool,
    next_in_secs: float,
) -> None:
    """
    Burn a small status panel into the top-left corner of *frame* (in-place).
    Uses a dark outline so text is readable on any background colour.
    """
    if is_detecting:
        trigger_line = "Detecting...  please wait"
    else:
        trigger_line = f"Next in {next_in_secs:.0f}s  |  D = detect now"

    lines = [
        f"Capsules : {n_capsules}",
        f"Last det : {proc_ms:.0f} ms" if proc_ms > 0 else "Last det : --",
        trigger_line,
        "Q = quit",
    ]

    for i, text in enumerate(lines):
        y = 30 + i * 26
        # Black outline (thickness 3) then cyan text (thickness 1) on top.
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
