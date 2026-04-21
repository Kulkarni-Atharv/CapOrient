"""
Capsule orientation detection — entry point.

Default behaviour  (no arguments)
----------------------------------
    python3 main.py
        → opens camera 0, runs live detection automatically.

Static image
------------
    python3 main.py image.jpg
    python3 main.py image.jpg --visualise
    python3 main.py image.jpg -v -o result.png
"""

import argparse
import json

import config
from capsule_detector.live_detection import LiveDetector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capsule orientation detection — runs live camera by default."
    )

    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to a static image. Omit to start the live camera feed.",
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=config.LIVE_CAMERA_INDEX,
        help=f"Picamera2 camera number (default: {config.LIVE_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--visualise", "-v", action="store_true",
        help="Save an annotated image alongside the JSON output (static image only)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the annotated image (only with --visualise)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image is None:
        # No image path supplied — start the live camera feed automatically.
        LiveDetector(camera_index=args.camera).run()
        return

    # Static image mode.
    from capsule_detector import detect_capsules, detect_and_visualise

    if args.visualise:
        detections = detect_and_visualise(args.image, args.output)
    else:
        detections = detect_capsules(args.image)

    print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
