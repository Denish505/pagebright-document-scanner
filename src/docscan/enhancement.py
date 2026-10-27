"""Readability enhancement methods for rectified document images."""

from __future__ import annotations

import cv2
import numpy as np


def illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """Flatten slow illumination changes by dividing by a blurred background."""
    short_side = min(gray.shape[:2])
    sigma = max(12.0, min(80.0, short_side / 18.0))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = gray.astype(np.float32) / (background.astype(np.float32) + 1.0)
    normalized *= float(np.median(background))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def enhance_gray(image: np.ndarray, denoise: bool = True) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = illumination_normalize(gray)
    if denoise:
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def enhance_color(image: np.ndarray) -> np.ndarray:
    """Apply local contrast correction to luminance while keeping page color."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = illumination_normalize(lightness)
    lightness = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def enhance_binary(image: np.ndarray, block_size: int = 31, constant: int = 11) -> np.ndarray:
    """Produce a high-contrast black-and-white page using adaptive thresholding."""
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer greater than or equal to 3.")
    gray = enhance_gray(image)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant
    )
