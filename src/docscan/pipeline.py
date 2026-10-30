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

MODES = {"all", "color", "gray", "bw"}
OUTPUT_FILENAMES = {
    "boundary": "boundary_preview.jpg",
    "rectified": "rectified_color.jpg",
    "color": "enhanced_color.jpg",
    "gray": "enhanced_gray.jpg",
    "bw": "enhanced_bw.png",
}


def full_page_corners(image: np.ndarray, inset_fraction: float = 0.0) -> np.ndarray:
    """Corners of the whole image, optionally pulled in by a fraction of each side."""
    height, width = image.shape[:2]
    dx, dy = (width - 1) * inset_fraction, (height - 1) * inset_fraction
    return np.array(
        [[dx, dy], [width - 1 - dx, dy], [width - 1 - dx, height - 1 - dy], [dx, height - 1 - dy]],
        dtype=np.float32,
    )


def _trim(image: np.ndarray, fraction: float) -> np.ndarray:
    """Cut a thin margin so background slivers at the detected edge don't show."""
    if fraction <= 0:
        return image
    height, width = image.shape[:2]
    my, mx = int(round(height * fraction)), int(round(width * fraction))
    if height - 2 * my < 16 or width - 2 * mx < 16:
        return image
    return image[my:height - my, mx:width - mx]


def process_image(
    image: np.ndarray,
    mode: str = "all",
    manual_corners: np.ndarray | None = None,
    min_area_fraction: float = 0.08,
    max_preview_side: int = 1600,
    denoise: bool = True,
    model_path: str | Path | None = None,
    trim_fraction: float = 0.008,
) -> tuple[dict[str, np.ndarray], dict]:
    """Detect, rectify and enhance an in-memory BGR image.

    Returns ``(images, report)`` where ``images`` maps output keys (see
    ``OUTPUT_FILENAMES``) to arrays. Raises ``ValueError`` if no page is found.
    """
    if mode not in MODES:
        raise ValueError("mode must be one of: all, color, gray, bw")
    if not 0.0 < min_area_fraction < 0.98:
        raise ValueError("min_area_fraction must be greater than 0 and less than 0.98")
    if max_preview_side < 64:
        raise ValueError("max_preview_side must be at least 64 pixels")
    if image is None or image.ndim != 3:
        raise ValueError("Expected a colour image.")
    started = time.perf_counter()
    if manual_corners is None:
        detection: PageDetection = detect_page(image, min_area_fraction, max_preview_side, model_path=model_path)
        corners, method, area_fraction = detection.corners, detection.method, detection.area_fraction
    else:
        corners = order_corners(manual_corners)
        height, width = image.shape[:2]
        if (corners[:, 0] < -0.5).any() or (corners[:, 0] > width - 0.5).any() \
                or (corners[:, 1] < -0.5).any() or (corners[:, 1] > height - 0.5).any():
            raise ValueError("Manual corners must lie inside the uploaded image.")
        corners = np.clip(corners, 0, [width - 1, height - 1]).astype(np.float32)
        method = "manual"
        area_fraction = float(abs(cv2.contourArea(corners)) / (height * width))

    overlay = image.copy()
    thickness = max(2, int(round(max(image.shape[:2]) / 250)))
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (20, 180, 30), thickness, cv2.LINE_AA)
    rectified, homography = rectify_page(image, corners)
    rectified = _trim(rectified, trim_fraction)
    images = {"boundary": overlay, "rectified": rectified}
    if mode in {"all", "color"}:
        images["color"] = enhance_color(rectified)
    if mode in {"all", "gray"}:
        images["gray"] = enhance_gray(rectified, denoise=denoise)
    if mode in {"all", "bw"}:
        images["bw"] = enhance_binary(rectified)
    report = {
        "image_size_px": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "detection_method": method,
        "page_area_fraction": round(area_fraction, 5),
        "corners_xy": np.round(order_corners(corners), 2).tolist(),
        "rectified_size_px": {"width": int(rectified.shape[1]), "height": int(rectified.shape[0])},
        "homography": np.round(homography, 8).tolist(),
        "mode": mode,
        "processing_seconds": round(time.perf_counter() - started, 4),
        "outputs": [OUTPUT_FILENAMES[key] for key in images],
    }
    return images, report


def encode_image(key: str, image: np.ndarray) -> bytes:
    """Encode an output array to bytes using the format implied by its filename."""
    extension = Path(OUTPUT_FILENAMES[key]).suffix
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if extension == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise OSError(f"Could not encode {OUTPUT_FILENAMES[key]}.")
    return encoded.tobytes()


def process_document(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str = "all",
    manual_corners: np.ndarray | None = None,
    min_area_fraction: float = 0.08,
    max_preview_side: int = 1600,
    denoise: bool = True,
    model_path: str | Path | None = None,
    trim_fraction: float = 0.008,
) -> dict:
    """File-based wrapper around :func:`process_image` used by the CLI."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    images, report = process_image(
        image, mode=mode, manual_corners=manual_corners, min_area_fraction=min_area_fraction,
        max_preview_side=max_preview_side, denoise=denoise, model_path=model_path,
        trim_fraction=trim_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, array in images.items():
        (output_dir / OUTPUT_FILENAMES[key]).write_bytes(encode_image(key, array))
    report = {"input_file": input_path.name, **report}
    (output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
