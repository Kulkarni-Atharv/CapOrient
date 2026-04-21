"""
All tuneable constants for the capsule detection pipeline.
Centralised here so nothing is hard-coded across modules.
"""

# ── Pre-processing ────────────────────────────────────────────
DENOISE_H            = 7      # luminance denoising strength
DENOISE_H_COLOR      = 7      # colour denoising strength
DENOISE_TEMPLATE_WS  = 7      # template window size
DENOISE_SEARCH_WS    = 21     # search window size

# ── CLAHE (local contrast enhancement) ───────────────────────
CLAHE_CLIP_LIMIT     = 3.0
CLAHE_TILE_GRID      = (8, 8)

# ── Edge detection ────────────────────────────────────────────
CANNY_LOWER_FACTOR   = 0.4    # lower  = CANNY_LOWER_FACTOR * median pixel value
CANNY_UPPER_FACTOR   = 1.2    # upper  = CANNY_UPPER_FACTOR * median pixel value
SOBEL_KSIZE          = 3
SOBEL_THRESH         = 20     # binarisation threshold on normalised Sobel magnitude
EDGE_CLOSE_KSIZE     = (5, 5) # morphological close kernel on the fused edge map
EDGE_CLOSE_ITER      = 2

# ── Segmentation ──────────────────────────────────────────────
SEG_FILL_KSIZE       = (7, 7)
SEG_FILL_CLOSE_ITER  = 3
SEG_FILL_DILATE_ITER = 1
SEG_CLEAN_KSIZE      = (5, 5)
SEG_CLEAN_OPEN_ITER  = 2
SEG_CLEAN_CLOSE_ITER = 3
ADAPTIVE_BLOCK_SIZE  = 51     # must be odd
ADAPTIVE_C           = 5

# ── Shape filters ─────────────────────────────────────────────
MIN_AREA_FRACTION    = 0.001  # min blob area as fraction of image area
MAX_AREA_FRACTION    = 0.40   # max blob area as fraction of image area
MIN_ASPECT_RATIO     = 1.2    # major / minor — blobs below this are too round
MIN_SOLIDITY         = 0.55   # contour area / convex-hull area

# ── Post-processing (NMS) ─────────────────────────────────────
NMS_IOU_THRESHOLD    = 0.45

# ── Live feed ─────────────────────────────────────────────────
LIVE_CAMERA_INDEX        = 0    # cv2.VideoCapture device index
LIVE_PROCESS_WIDTH       = 640  # downscale frames to this width before detection
                                 # (height is computed to maintain aspect ratio)
LIVE_DETECTION_INTERVAL  = 30   # seconds between automatic detection passes;
                                 # press D at runtime to trigger immediately
LIVE_DISPLAY_FPS         = 30   # display refresh rate (waitKey interval)
