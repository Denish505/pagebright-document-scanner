"""Page-corner ordering and perspective rectification helpers."""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return four points clockwise, starting at the top-left corner."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        raise ValueError("Provide exactly four finite page corners as x,y pairs.")
    if np.unique(pts, axis=0).shape[0] != 4:
        raise ValueError("Page corners must be four distinct points.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)

    # The angle ordering is clockwise in image coordinates. This guard handles
    # degenerate/tied point sets and keeps the expected TL, TR, BR, BL order.
    first = ordered[1] - ordered[0]
    second = ordered[2] - ordered[1]
    cross = first[0] * second[1] - first[1] * second[0]
    if cross < 0:
        ordered = ordered[[0, 3, 2, 1]]
    if not cv2.isContourConvex(ordered.reshape(-1, 1, 2)) or abs(cv2.contourArea(ordered)) < 4:
        raise ValueError("The four page corners must form a non-degenerate convex quadrilateral.")
    return ordered.astype(np.float32)


def rectify_page(image: np.ndarray, corners: np.ndarray, max_dimension: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Warp a page quadrilateral to a flat rectangle; return image and homography."""
    tl, tr, br, bl = order_corners(corners)
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        raise ValueError("Detected page corners do not define a usable rectangle.")

    scale = min(1.0, max_dimension / max(width, height))
    out_w, out_h = max(2, int(round(width * scale))), max(2, int(round(height * scale)))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), destination)
    warped = cv2.warpPerspective(
        image,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, homography
