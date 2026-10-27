"""Classical page-boundary detection using edges, thresholding, and contours."""

from __future__ import annotations

from dataclasses import dataclass

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


def detect_page(
    image: np.ndarray,
    min_area_fraction: float = 0.08,
    max_preview_side: int = 1600,
) -> PageDetection:
    """Find the largest plausible convex page quadrilateral.

    Detection runs on a downscaled preview for speed and maps corners back to
    the original image coordinates. Raises ValueError when no suitable quad is
    found; use the CLI's manual-corner option for difficult photographs.
    """
    height, width = image.shape[:2]
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
            # Larger plausible quadrilaterals are preferred; a slight penalty
            # avoids selecting the nearly full-frame border as the page.
            score = fraction - max(0.0, fraction - 0.92) * 2.0
            candidates.append((score, points, method))

    if not candidates:
        raise ValueError(
            "No page quadrilateral was found. Improve contrast around the page "
            "or pass four manual corners with --corners."
        )

    score, points, method = max(candidates, key=lambda item: item[0])
    original_points = points / scale
    area_fraction = abs(cv2.contourArea(original_points.astype(np.float32))) / float(height * width)
    return PageDetection(original_points.astype(np.float32), method, float(area_fraction), float(scale))
