"""Page-boundary detection with an optional learned ONNX corner model."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .geometry import order_corners


@dataclass(frozen=True)
class PageDetection:
    corners: np.ndarray
    method: str
    area_fraction: float
    preview_scale: float


def _contours(binary: np.ndarray) -> list[np.ndarray]:
    found = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return found[-2]


@lru_cache(maxsize=2)
def _load_onnx(model_path: str, modified: float) -> "cv2.dnn.Net":
    """Load the ONNX model once; ``modified`` makes a retrained file reload."""
    return cv2.dnn.readNetFromONNX(model_path)


def _detect_with_onnx(image: np.ndarray, model_path: str | Path, min_area_fraction: float) -> PageDetection:
    """Use a user-trained corner regressor exported by scripts/train_page_model.py."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((tensor - mean) / std).transpose(2, 0, 1)[None, ...].astype(np.float32)
    path = Path(model_path)
    net = _load_onnx(str(path), path.stat().st_mtime)
    net.setInput(tensor)
    normalized = net.forward().reshape(4, 2)
    if not np.isfinite(normalized).all() or (normalized < -0.05).any() or (normalized > 1.05).any():
        raise ValueError("The trained model returned invalid page corners.")
    points = order_corners(normalized * np.array([width, height], dtype=np.float32))
    area_fraction = float(abs(cv2.contourArea(points)) / (height * width))
    if area_fraction < min_area_fraction or area_fraction > 0.99:
        raise ValueError("The trained model returned an implausible page size.")
    return PageDetection(points, "trained_model", area_fraction, 1.0)


def detect_page(
    image: np.ndarray,
    min_area_fraction: float = 0.08,
    max_preview_side: int = 1600,
    model_path: str | Path | None = None,
) -> PageDetection:
    """Try a trained ONNX model first, then use multi-threshold contours."""
    height, width = image.shape[:2]
    if model_path is not None and Path(model_path).is_file():
        try:
            return _detect_with_onnx(image, model_path, min_area_fraction)
        except (cv2.error, OSError, ValueError, RuntimeError):
            pass

    scale = min(1.0, max_preview_side / float(max(height, width)))
    if scale < 1.0:
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        preview = image.copy()
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    canny = cv2.Canny(blurred, 60, 180)
    canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)
    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    image_area = float(preview.shape[0] * preview.shape[1])
    candidates: list[tuple[float, np.ndarray, str]] = []
    for method, mask in (("canny", canny), ("otsu", otsu), ("adaptive", adaptive)):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        for contour in sorted(_contours(closed), key=cv2.contourArea, reverse=True)[:40]:
            area = float(cv2.contourArea(contour))
            fraction = area / image_area
            if fraction < min_area_fraction or fraction > 0.985:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = order_corners(polygon.reshape(4, 2))
            score = fraction - max(0.0, fraction - 0.92) * 2.0
            candidates.append((score, points, method))
    if not candidates:
        raise ValueError(
            "No page quadrilateral was found. Improve contrast around the page, "
            "enable manual corners, or add labeled examples and train a page model."
        )
    _, points, method = max(candidates, key=lambda item: item[0])
    original_points = points / scale
    area_fraction = abs(cv2.contourArea(original_points.astype(np.float32))) / float(height * width)
    return PageDetection(original_points.astype(np.float32), method, float(area_fraction), float(scale))
