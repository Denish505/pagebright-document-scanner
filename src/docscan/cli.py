"""Command-line interface for document image enhancement."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import process_document


def _parse_corners(value: str) -> np.ndarray:
    try:
        corners = [[float(part) for part in pair.split(",")] for pair in value.split(";")]
        result = np.asarray(corners, dtype=np.float32)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use four x,y pairs separated by semicolons.") from error
    if result.shape != (4, 2):
        raise argparse.ArgumentTypeError("Provide exactly four x,y pairs.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docscan",
        description="Detect a document page, correct perspective, and improve readability.",
    )
    parser.add_argument("input", type=Path, help="Path to a JPG, PNG, or other OpenCV-readable image")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated images and metadata")
    parser.add_argument("--mode", choices=("all", "color", "gray", "bw"), default="all")
    parser.add_argument("--min-area", type=float, default=0.08, help="Minimum detected page fraction (0-1)")
    parser.add_argument("--corners", type=_parse_corners, default=None, help="Manual corners: x1,y1;x2,y2;x3,y3;x4,y4")
    parser.add_argument("--no-denoise", action="store_true", help="Skip bilateral denoising in grayscale output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if not 0.0 < args.min_area < 0.98:
        parser.error("--min-area must be greater than 0 and less than 0.98")
    output_dir = args.output_dir or Path("outputs") / args.input.stem
    try:
        report = process_document(
            args.input,
            output_dir,
            mode=args.mode,
            manual_corners=args.corners,
            min_area_fraction=args.min_area,
            denoise=not args.no_denoise,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Processed {report['input_file']} with {report['detection_method']} page detection.")
    print(f"Saved {len(report['outputs'])} images and metadata to {output_dir.resolve()}")
    print(f"Rectified size: {report['rectified_size_px']['width']} x {report['rectified_size_px']['height']} px")
    return 0
