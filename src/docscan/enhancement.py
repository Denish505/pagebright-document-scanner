"""Readability enhancement methods for rectified document images.

All three styles share one idea: estimate the paper background (the image with
the ink removed), divide by it so the paper becomes uniformly white, then
stretch the remaining ink to full contrast. Estimating the background with a
max-filter *before* blurring keeps dark text out of the estimate, which is what
prevents bright halos around text.
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(value: float, minimum: int = 3) -> int:
    value = max(minimum, int(round(value)))
    return value if value % 2 == 1 else value + 1


def estimate_background(channel: np.ndarray) -> np.ndarray:
    """Estimate paper brightness for a single 8-bit channel, ink removed."""
    height, width = channel.shape[:2]
    # Work at reduced size: the background varies slowly, and this is much faster.
    scale = min(1.0, 600.0 / max(height, width))
    small = cv2.resize(channel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else channel
    short = min(small.shape[:2])
    # Max filter wider than text strokes erases ink; median smooths blotches.
    dilated = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(short / 40), _odd(short / 40))))
    background = cv2.medianBlur(dilated, min(_odd(short / 25, minimum=5), 255))
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(1.5, short / 150))
    if scale < 1:
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    return background


def flatten_illumination(channel: np.ndarray) -> np.ndarray:
    """Divide by the background so paper becomes ~255 everywhere (float32 0..255)."""
    background = estimate_background(channel).astype(np.float32)
    return np.clip(channel.astype(np.float32) * 255.0 / np.maximum(background, 1.0), 0, 255)


def _levels(values: np.ndarray, white_point: float = 238.0) -> np.ndarray:
    """Clip near-white paper to pure white and stretch ink toward black."""
    ink = values[values < white_point]
    black_point = float(np.percentile(ink, 2)) if ink.size > 100 else 0.0
    black_point = min(black_point, 110.0)
    stretched = (values - black_point) * 255.0 / max(1.0, white_point - black_point)
    return np.clip(stretched, 0, 255).astype(np.uint8)


# Backwards-compatible name used by older code and the dossier.
def illumination_normalize(gray: np.ndarray) -> np.ndarray:
    return flatten_illumination(gray).astype(np.uint8)


def enhance_gray(image: np.ndarray, denoise: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if denoise:
        gray = cv2.bilateralFilter(gray, 5, 30, 5)
    return _levels(flatten_illumination(gray))


def enhance_color(image: np.ndarray) -> np.ndarray:
    """Whiten the paper per channel (removes shadows and paper tint), keep ink color."""
    channels = [flatten_illumination(channel) for channel in cv2.split(image)]
    flat = cv2.merge(channels)
    # One shared level stretch keeps the colour balance of the ink intact.
    luminance = cv2.cvtColor(np.clip(flat, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ink = luminance[luminance < 238]
    black_point = min(float(np.percentile(ink, 2)) if ink.size > 100 else 0.0, 110.0)
    stretched = (flat - black_point) * 255.0 / max(1.0, 238.0 - black_point)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _rejoin_ruled_lines(ink: np.ndarray) -> np.ndarray:
    """Bridge gaps in faint table/ruled lines that thresholding broke into dashes.

    Only thin, elongated fragments (dash pieces) are considered, never letters,
    and a bridged run is kept only if it ends up long and straight.
    """
    height, width = ink.shape[:2]
    thin = max(3, int(round(min(height, width) / 220)))
    gap = max(4, int(round(min(height, width) / 90)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    widths, heights = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    added = np.zeros_like(ink)
    for is_horizontal in (True, False):
        if is_horizontal:
            keep = (heights <= thin) & (widths >= 2 * heights)
            close_kernel, open_kernel = (2 * gap + 1, 1), (max(25, width // 6), 1)
        else:
            keep = (widths <= thin) & (heights >= 2 * widths)
            close_kernel, open_kernel = (1, 2 * gap + 1), (1, max(25, height // 6))
        keep[0] = False
        fragments = np.isin(labels, np.flatnonzero(keep)).astype(np.uint8) * 255
        # Long existing lines also act as anchors so a dashed tail joins them.
        joined = cv2.morphologyEx(fragments, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel))
        lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, open_kernel))
        added = cv2.bitwise_or(added, lines)
    return added


def enhance_binary(image: np.ndarray, block_size: int | None = None, constant: int = 10) -> np.ndarray:
    """Produce a clean black-and-white page.

    A pixel is ink when it is clearly darker than its neighbourhood (adaptive
    test) *and* not near-white paper (global test). The global guard removes
    speckle in empty areas; the adaptive test keeps thin, faint table lines.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = flatten_illumination(cv2.medianBlur(gray, 3)).astype(np.uint8)
    if block_size is None:
        block_size = _odd(min(flat.shape[:2]) / 25, minimum=15)
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer greater than or equal to 3.")
    adaptive_ink = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, constant
    )
    not_paper = (flat < 242).astype(np.uint8) * 255
    ink = cv2.bitwise_and(adaptive_ink, not_paper)
    ink = cv2.bitwise_or(ink, _rejoin_ruled_lines(ink))
    # Drop isolated specks of 1-2 pixels.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] <= 3)[0]
    if small.size > 1:
        ink[np.isin(labels, small[small != 0])] = 0
    return cv2.bitwise_not(ink)
