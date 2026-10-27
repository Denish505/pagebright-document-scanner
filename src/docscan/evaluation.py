"""Small evaluation helpers for annotated corners and paired image references."""

from __future__ import annotations

import math

import numpy as np

from .geometry import order_corners


def normalized_corner_error(predicted: np.ndarray, reference: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Mean corresponding-corner distance divided by the image diagonal."""
    pred = order_corners(np.asarray(predicted, dtype=np.float32))
    ref = order_corners(np.asarray(reference, dtype=np.float32))
    height, width = image_shape[:2]
    diagonal = math.hypot(width, height)
    if diagonal == 0:
        raise ValueError("Image dimensions must be nonzero.")
    return float(np.linalg.norm(pred - ref, axis=1).mean() / diagonal)


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Peak signal-to-noise ratio for same-size, aligned reference images."""
    if reference.shape != estimate.shape:
        raise ValueError("PSNR requires reference and estimate images of the same shape.")
    mse = float(np.mean((reference.astype(np.float32) - estimate.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Optional SSIM metric; requires scikit-image and aligned reference images."""
    if reference.shape != estimate.shape:
        raise ValueError("SSIM requires reference and estimate images of the same shape.")
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError("Install the optional dependency with: pip install scikit-image") from error
    if reference.ndim == 3:
        return float(structural_similarity(reference, estimate, channel_axis=2, data_range=255))
    return float(structural_similarity(reference, estimate, data_range=255))
