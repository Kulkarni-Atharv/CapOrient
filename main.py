"""
CLI entry point for the capsule orientation detection system.

Static image
------------
    python main.py <image>
    python main.py <image> --visualise
    python main.py <image> -v -o result.png

Live camera feed
----------------
    python main.py --live
    python main.py --live --camera 1        # use device index 1
"""

import argparse
import json

import config
from capsule_detector import detect_capsules, detect_and_visualise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect pharmaceutical capsule orientations."
    )

    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to the input image (omit when using --live)",
    )
    parser.add_argument(
        "--live", "-l", action="store_true",
        help="Run on a live camera feed instead of a static image",
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=config.LIVE_CAMERA_INDEX,
        help=f"Camera device index for --live mode (default: {config.LIVE_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--visualise", "-v", action="store_true",
        help="Save an annotated image (static-image mode only)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for the annotated image (only with --visualise)",
    )

    args = parser.parse_args()

    if not args.live and args.image is None:
        parser.error("image path is required unless --live is specified.")

    return args


def main() -> None:
    args = _parse_args()

    if args.live:
        from capsule_detector.live_detection import LiveDetector
        LiveDetector(camera_index=args.camera).run()
        return

    if args.visualise:
        detections = detect_and_visualise(args.image, args.output)
    else:
        detections = detect_capsules(args.image)

    print(json.dumps(detections, indent=2))


if __name__ == "__main__":
    main()
