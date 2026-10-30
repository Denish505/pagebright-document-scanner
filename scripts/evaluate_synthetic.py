"""Synthetic benchmark for PageBright's page detector.

Each test photo is made by drawing a text page, warping it with a random
perspective onto a random background, then adding lighting, shadow, blur,
noise and JPEG compression. Because the warp is known, the true corners are
known exactly, so corner error can be measured without manual labelling.

Run:  python scripts/evaluate_synthetic.py --n 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docscan.detection import detect_page  # noqa: E402
from docscan.evaluation import normalized_corner_error  # noqa: E402

W, H = 1400, 1000  # photo size, same as the demo image


def make_page(rng: np.random.Generator) -> np.ndarray:
    pw, ph = 620, 860
    tint = rng.integers(235, 256, 3)
    page = np.full((ph, pw, 3), tint, np.uint8)
    y = 70
    cv2.putText(page, "INVOICE  No. %d" % rng.integers(100, 999), (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (20, 20, 20), 2, cv2.LINE_AA)
    y += 50
    while y < ph - 60:
        if rng.random() < 0.12:
            cv2.line(page, (50, y), (pw - 50, y), (60, 60, 60), 1)
        else:
            words = rng.integers(3, 9)
            x = 50
            for _ in range(words):
                wlen = int(rng.integers(30, 90))
                if x + wlen > pw - 50:
                    break
                cv2.rectangle(page, (x, y - 10), (x + wlen, y), (40, 40, 40), -1)
                x += wlen + 12
        y += int(rng.integers(24, 34))
    return page


def make_background(kind: str, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "dark_desk":
        base = rng.integers(40, 90, 3).astype(np.float32)
        bg = np.ones((H, W, 3), np.float32) * base
    elif kind == "wood":
        stripes = (np.sin(xx / rng.uniform(8, 20) + np.sin(yy / 60) * 2) + 1) / 2
        bg = np.dstack([60 + 50 * stripes, 90 + 60 * stripes, 130 + 70 * stripes])
    elif kind == "cluttered":
        bg = np.ones((H, W, 3), np.float32) * rng.integers(70, 140, 3)
        for _ in range(12):
            x0, y0 = int(rng.integers(0, W)), int(rng.integers(0, H))
            cv2.rectangle(bg, (x0, y0), (x0 + int(rng.integers(40, 300)), y0 + int(rng.integers(40, 300))),
                          rng.integers(20, 230, 3).tolist(), -1)
    elif kind == "light_gray":  # low contrast
        bg = np.ones((H, W, 3), np.float32) * rng.integers(165, 195)
    else:  # "white_desk": very low contrast, the hardest case
        bg = np.ones((H, W, 3), np.float32) * rng.integers(205, 225)
    bg += rng.normal(0, 6, bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def random_corners(rng: np.random.Generator) -> np.ndarray:
    cx, cy = W / 2 + rng.uniform(-150, 150), H / 2 + rng.uniform(-80, 80)
    scale = rng.uniform(0.75, 1.05)
    hw, hh = 300 * scale, 420 * scale
    angle = np.deg2rad(rng.uniform(-20, 20))
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = base @ rot.T + [cx, cy]
    pts += rng.uniform(-45, 45, pts.shape)  # perspective tilt
    pts[:, 0] = np.clip(pts[:, 0], 10, W - 10)
    pts[:, 1] = np.clip(pts[:, 1], 10, H - 10)
    return pts.astype(np.float32)


def make_photo(kind: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    page = make_page(rng)
    ph, pw = page.shape[:2]
    corners = random_corners(rng)
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    hmat = cv2.getPerspectiveTransform(src, corners)
    bg = make_background(kind, rng)
    warped = cv2.warpPerspective(page, hmat, (W, H))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), hmat, (W, H))
    photo = np.where(mask[..., None] > 127, warped, bg).astype(np.float32)
    # uneven lighting + optional hard shadow
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    light = 1 + 0.25 * (gx * (xx / W - 0.5) + gy * (yy / H - 0.5))
    if rng.random() < 0.4:
        a, b, c = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.3)
        shadow = ((a * (xx / W - 0.5) + b * (yy / H - 0.5) + c) > 0).astype(np.float32)
        light *= 1 - rng.uniform(0.25, 0.45) * cv2.GaussianBlur(shadow, (0, 0), rng.uniform(5, 25))
    photo *= light[..., None]
    if rng.random() < 0.5:
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.5, 1.8))
    photo += rng.normal(0, rng.uniform(2, 7), photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="photos per run (split across backgrounds)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="a detection counts as correct if corner error < this fraction of the diagonal")
    parser.add_argument("--json", type=Path, default=None, help="optional path to save the results")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    kinds = ["dark_desk", "wood", "cluttered", "light_gray", "white_desk"]
    rows = []
    for i in range(args.n):
        kind = kinds[i % len(kinds)]
        photo, truth = make_photo(kind, rng)
        start = time.perf_counter()
        try:
            det = detect_page(photo, min_area_fraction=0.08)
            err = normalized_corner_error(det.corners, truth, photo.shape)
            method = det.method
        except ValueError:
            err, method = None, "not_found"
        rows.append({"kind": kind, "error": err, "method": method,
                     "seconds": time.perf_counter() - start})

    def summary(sub):
        found = [r for r in sub if r["error"] is not None]
        correct = [r for r in found if r["error"] < args.threshold]
        errs = np.array([r["error"] for r in found]) if found else np.array([np.nan])
        return {
            "photos": len(sub),
            "found_%": round(100 * len(found) / len(sub), 1),
            "correct_%": round(100 * len(correct) / len(sub), 1),
            "mean_error_%diag": round(100 * float(np.mean(errs)), 2),
            "median_error_%diag": round(100 * float(np.median(errs)), 2),
            "median_ms": round(1000 * float(np.median([r["seconds"] for r in sub])), 1),
        }

    report = {"overall": summary(rows), "threshold_%diag": 100 * args.threshold,
              "by_background": {k: summary([r for r in rows if r["kind"] == k]) for k in kinds},
              "winning_method": {m: sum(r["method"] == m for r in rows) for m in sorted({r["method"] for r in rows})}}
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
