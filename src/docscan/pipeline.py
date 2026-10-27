"""End-to-end document photo rectification and enhancement pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .detection import PageDetection, detect_page
from .enhancement import enhance_binary, enhance_color, enhance_gray
from .geometry import order_corners, rectify_page


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write output image: {path}")


def process_document(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str = "all",
    manual_corners: np.ndarray | None = None,
    min_area_fraction: float = 0.08,
    max_preview_side: int = 1600,
    denoise: bool = True,
) -> dict:
    """Detect, rectify, and enhance one document photograph.

    Modes are ``all``, ``color``, ``gray``, and ``bw``. The color rectification
    is always saved as an intermediate result. Manual corners are ordered
    automatically and must refer to the original input image coordinates.
    """
    if mode not in {"all", "color", "gray", "bw"}:
        raise ValueError("mode must be one of: all, color, gray, bw")
    if not 0.0 < min_area_fraction < 0.98:
        raise ValueError("min_area_fraction must be greater than 0 and less than 0.98")
    if max_preview_side < 64:
        raise ValueError("max_preview_side must be at least 64 pixels")
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if manual_corners is None:
        detection: PageDetection = detect_page(image, min_area_fraction, max_preview_side)
        corners = detection.corners
        method = detection.method
        area_fraction = detection.area_fraction
    else:
        corners = order_corners(manual_corners)
        if (
            (corners[:, 0] < 0).any()
            or (corners[:, 0] >= image.shape[1]).any()
            or (corners[:, 1] < 0).any()
            or (corners[:, 1] >= image.shape[0]).any()
        ):
            raise ValueError("Manual corners must lie inside the uploaded image.")
        method = "manual"
        area_fraction = float(abs(cv2.contourArea(corners)) / (image.shape[0] * image.shape[1]))

    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (20, 180, 30), 7, cv2.LINE_AA)
    _write_image(output_dir / "boundary_preview.jpg", overlay)

    rectified, homography = rectify_page(image, corners)
    _write_image(output_dir / "rectified_color.jpg", rectified)

    outputs = ["boundary_preview.jpg", "rectified_color.jpg"]
    if mode in {"all", "color"}:
        _write_image(output_dir / "enhanced_color.jpg", enhance_color(rectified))
        outputs.append("enhanced_color.jpg")
    if mode in {"all", "gray"}:
        gray = enhance_gray(rectified, denoise=denoise)
        _write_image(output_dir / "enhanced_gray.jpg", gray)
        outputs.append("enhanced_gray.jpg")
    if mode in {"all", "bw"}:
        binary = enhance_binary(rectified)
        _write_image(output_dir / "enhanced_bw.png", binary)
        outputs.append("enhanced_bw.png")

    elapsed = time.perf_counter() - started
    report = {
        "input_file": input_path.name,
        "image_size_px": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "detection_method": method,
        "page_area_fraction": round(area_fraction, 5),
        "corners_xy": np.round(order_corners(corners), 2).tolist(),
        "rectified_size_px": {"width": int(rectified.shape[1]), "height": int(rectified.shape[0])},
        "homography": np.round(homography, 8).tolist(),
        "mode": mode,
        "processing_seconds": round(elapsed, 4),
        "outputs": outputs,
    }
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
