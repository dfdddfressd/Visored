"""OpenCV panel segmentation from in-memory image bytes."""

from __future__ import annotations

import hashlib
import os
from typing import Literal, TypedDict

import cv2
import numpy as np

MIN_PANEL_SIDE = 150
# If sum of detected panel areas is below this fraction of page, treat as full-bleed.
MIN_COVERAGE_FRACTION = 0.15

SegmentationMode = Literal["gutter", "legacy"]

# Gutter (recursive XY-cut) defaults
DEFAULT_GUTTER_STRENGTH = 0.92
DEFAULT_GUTTER_SMOOTH = 5
DEFAULT_GUTTER_MAX_DEPTH = 16
DEFAULT_GUTTER_MARGIN_FRAC = 0.03
DEFAULT_GUTTER_MAX_LEAVES = 64


class SegmentResult(TypedDict):
    panels_bgr: list[cv2.typing.MatLike]
    panel_count: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def segment_page_from_bytes(
    image_bytes: bytes,
    min_side: int = MIN_PANEL_SIDE,
    min_coverage_fraction: float = MIN_COVERAGE_FRACTION,
    *,
    segmentation: SegmentationMode = "gutter",
    gutter_strength: float = DEFAULT_GUTTER_STRENGTH,
    gutter_smooth: int = DEFAULT_GUTTER_SMOOTH,
    gutter_max_depth: int = DEFAULT_GUTTER_MAX_DEPTH,
    gutter_margin_frac: float = DEFAULT_GUTTER_MARGIN_FRAC,
    gutter_max_leaves: int = DEFAULT_GUTTER_MAX_LEAVES,
) -> SegmentResult:
    """Decode image and split into panel crops (gutter XY-cut or legacy contours)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed (invalid or corrupted image)")

    if segmentation == "legacy":
        return _segment_legacy_from_bgr(
            img, min_side=min_side, min_coverage_fraction=min_coverage_fraction
        )

    panels_bgr = _segment_gutter_from_bgr(
        img,
        min_side=min_side,
        gutter_strength=gutter_strength,
        gutter_smooth=gutter_smooth,
        gutter_max_depth=gutter_max_depth,
        gutter_margin_frac=gutter_margin_frac,
        gutter_max_leaves=gutter_max_leaves,
    )
    return SegmentResult(panels_bgr=panels_bgr, panel_count=len(panels_bgr))


# Ink fraction required on both sides of a seam (avoids cutting page margins).
MIN_INK_FRACTION_PER_SIDE = 0.02
# Full-row / full-column mean paper must clear this (speech balloons rarely span entire ROI).
SEAM_MIN_MEAN_PAPER = 0.86
# Seam line mean ink must stay below this (true gutters are nearly blank).
SEAM_MAX_MEAN_INK = 0.14
# require min(top-block ink, bottom-block ink) - seam_ink to qualify (rejects balloon bands).
MIN_SEAM_SPLIT_SCORE = 0.035


def _smooth_1d(a: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or a.size == 0:
        return a.astype(np.float64)
    w = max(1, int(width))
    if w % 2 == 0:
        w += 1
    if w > a.size:
        w = max(1, a.size | 1)
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(a.astype(np.float64), k, mode="same")


def _otsu_threshold_u8(gray: np.ndarray) -> float:
    """OpenCV Otsu can return 0 on some high-contrast pages; fall back to mid-intensity."""
    crop = gray
    if crop.size == 0:
        return 127.0
    t_val = float(cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    if 1.0 <= t_val <= 254.0:
        return t_val
    lo = float(np.min(crop))
    hi = float(np.max(crop))
    if hi > lo:
        return (lo + hi) / 2.0
    return 127.0


def _paper_likelihood_mask(gray: np.ndarray) -> np.ndarray:
    """Per-pixel [0,1] background score (high where paper/gutter, low on ink)."""
    h0, w0 = gray.shape
    cy0, cy1 = h0 // 4, (3 * h0) // 4
    cx0, cx1 = w0 // 4, (3 * w0) // 4
    crop = gray[cy0:cy1, cx0:cx1]
    t_val = _otsu_threshold_u8(crop if crop.size else gray)

    ink = (gray < max(0.0, t_val - 5.0)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
    paper = 1.0 - np.clip(ink.astype(np.float32) / 255.0, 0.0, 1.0)
    return paper


def _full_page_interior_gutter_score(
    bg: np.ndarray,
    gutter_smooth: int,
) -> float:
    """Max smoothed mean-paper projection in interior (matches gutter seam signal)."""
    h0, w0 = bg.shape
    if h0 < 4 or w0 < 4:
        return 0.0
    mh = max(1, int(h0 * DEFAULT_GUTTER_MARGIN_FRAC))
    mw = max(1, int(w0 * DEFAULT_GUTTER_MARGIN_FRAC))
    sub = bg[mh : h0 - mh, mw : w0 - mw]
    if sub.size == 0:
        return 0.0
    hproj = np.mean(sub, axis=1)
    vproj = np.mean(sub, axis=0)
    hs = _smooth_1d(hproj, gutter_smooth)
    vs = _smooth_1d(vproj, gutter_smooth)
    if len(hs) > 2 * mh:
        hs = hs[mh : len(hs) - mh]
    if len(vs) > 2 * mw:
        vs = vs[mw : len(vs) - mw]
    return float(max(float(np.max(hs)) if hs.size else 0.0, float(np.max(vs)) if vs.size else 0.0))


def _pick_best_seam(
    bg: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_side: int,
    gutter_strength: float,
    gutter_smooth: int,
    margin_frac: float,
    min_ink_per_side: float = MIN_INK_FRACTION_PER_SIDE,
) -> tuple[str | None, int | None, float]:
    """
    Gutter seams are full-width (or full-height) low-ink lines with white paper,
    flanked by ink-heavy panels. High percentile paper would treat speech balloons
    like gutters; we use mean paper on the line plus a split-quality score instead.
    """
    rw = x1 - x0
    rh = y1 - y0
    if rw < 2 * min_side or rh < 2 * min_side:
        return None, None, 0.0

    mx = max(1, int(rw * margin_frac))
    my = max(1, int(rh * margin_frac))

    roi = bg[y0:y1, x0:x1]
    ink_roi = 1.0 - roi
    if roi.size == 0:
        return None, None, 0.0

    # Mean ink per line (smoothed) — gutters are valleys; bubbles in-panel score poorly.
    h_ink = np.mean(ink_roi, axis=1).astype(np.float64)
    v_ink = np.mean(ink_roi, axis=0).astype(np.float64)
    h_ink_sm = _smooth_1d(h_ink, gutter_smooth)
    v_ink_sm = _smooth_1d(v_ink, gutter_smooth)

    seam_k = max(1, gutter_smooth // 2)

    def h_ink_ok(yi: int) -> bool:
        top_ink = float(np.mean(ink_roi[0:yi, :])) if yi > 0 else 0.0
        bot_ink = float(np.mean(ink_roi[yi:rh, :])) if yi < rh else 0.0
        return top_ink >= min_ink_per_side and bot_ink >= min_ink_per_side

    def v_ink_ok(xi: int) -> bool:
        left_ink = float(np.mean(ink_roi[:, 0:xi])) if xi > 0 else 0.0
        right_ink = float(np.mean(ink_roi[:, xi:rw])) if xi < rw else 0.0
        return left_ink >= min_ink_per_side and right_ink >= min_ink_per_side

    paper_floor = max(SEAM_MIN_MEAN_PAPER, gutter_strength * 0.92)
    ink_ceiling = SEAM_MAX_MEAN_INK

    y_start = max(min_side, my)
    y_end = rh - max(min_side, my)

    best_h: tuple[int, float] | None = None
    for yi in range(y_start, y_end):
        if not h_ink_ok(yi):
            continue
        line_paper = float(np.mean(roi[yi, :]))
        line_ink = float(np.mean(ink_roi[yi, :]))
        if line_paper < paper_floor or line_ink > ink_ceiling:
            continue
        y_lo = max(0, yi - seam_k)
        y_hi = min(rh, yi + seam_k + 1)
        block_top = float(np.max(h_ink_sm[0:y_lo])) if y_lo > 0 else 0.0
        block_bot = float(np.max(h_ink_sm[y_hi:rh])) if y_hi < rh else 0.0
        score = min(block_top, block_bot) - float(h_ink_sm[yi])
        if score < MIN_SEAM_SPLIT_SCORE:
            continue
        is_peak = 0 < yi < rh - 1 and (
            h_ink_sm[yi] <= h_ink_sm[yi - 1] and h_ink_sm[yi] <= h_ink_sm[yi + 1]
        )
        if not is_peak:
            continue
        if best_h is None or score > best_h[1] or (score == best_h[1] and yi < best_h[0]):
            best_h = (yi, score)

    x_start = max(min_side, mx)
    x_end = rw - max(min_side, mx)

    best_v: tuple[int, float] | None = None
    for xi in range(x_start, x_end):
        if not v_ink_ok(xi):
            continue
        line_paper = float(np.mean(roi[:, xi]))
        line_ink = float(np.mean(ink_roi[:, xi]))
        if line_paper < paper_floor or line_ink > ink_ceiling:
            continue
        x_lo = max(0, xi - seam_k)
        x_hi = min(rw, xi + seam_k + 1)
        block_left = float(np.max(v_ink_sm[0:x_lo])) if x_lo > 0 else 0.0
        block_right = float(np.max(v_ink_sm[x_hi:rw])) if x_hi < rw else 0.0
        score = min(block_left, block_right) - float(v_ink_sm[xi])
        if score < MIN_SEAM_SPLIT_SCORE:
            continue
        is_peak = 0 < xi < rw - 1 and (
            v_ink_sm[xi] <= v_ink_sm[xi - 1] and v_ink_sm[xi] <= v_ink_sm[xi + 1]
        )
        if not is_peak:
            continue
        if best_v is None or score > best_v[1] or (score == best_v[1] and xi < best_v[0]):
            best_v = (xi, score)

    if best_h is None and best_v is None:
        return None, None, 0.0
    if best_h is None:
        assert best_v is not None
        return "v", x0 + best_v[0], best_v[1]
    if best_v is None:
        return "h", y0 + best_h[0], best_h[1]
    assert best_h is not None and best_v is not None
    if best_v[1] > best_h[1]:
        return "v", x0 + best_v[0], best_v[1]
    return "h", y0 + best_h[0], best_h[1]


def _gutter_collect_leaves(
    bg: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_side: int,
    gutter_strength: float,
    gutter_smooth: int,
    margin_frac: float,
    depth: int,
    max_depth: int,
    out: list[tuple[int, int, int, int]],
) -> None:
    if depth >= max_depth:
        out.append((x0, y0, x1, y1))
        return

    axis, coord, _score = _pick_best_seam(
        bg, x0, y0, x1, y1, min_side, gutter_strength, gutter_smooth, margin_frac
    )
    if axis is None:
        out.append((x0, y0, x1, y1))
        return

    if axis == "h":
        y_split = coord
        assert y_split is not None
        _gutter_collect_leaves(
            bg,
            x0,
            y0,
            x1,
            y_split,
            min_side,
            gutter_strength,
            gutter_smooth,
            margin_frac,
            depth + 1,
            max_depth,
            out,
        )
        _gutter_collect_leaves(
            bg,
            x0,
            y_split,
            x1,
            y1,
            min_side,
            gutter_strength,
            gutter_smooth,
            margin_frac,
            depth + 1,
            max_depth,
            out,
        )
    else:
        x_split = coord
        assert x_split is not None
        _gutter_collect_leaves(
            bg,
            x0,
            y0,
            x_split,
            y1,
            min_side,
            gutter_strength,
            gutter_smooth,
            margin_frac,
            depth + 1,
            max_depth,
            out,
        )
        _gutter_collect_leaves(
            bg,
            x_split,
            y0,
            x1,
            y1,
            min_side,
            gutter_strength,
            gutter_smooth,
            margin_frac,
            depth + 1,
            max_depth,
            out,
        )


def _dedupe_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for r in rects:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _segment_gutter_from_bgr(
    img: cv2.typing.MatLike,
    *,
    min_side: int,
    gutter_strength: float,
    gutter_smooth: int,
    gutter_max_depth: int,
    gutter_margin_frac: float,
    gutter_max_leaves: int,
) -> list[cv2.typing.MatLike]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = _paper_likelihood_mask(gray)
    h, w = img.shape[:2]
    rects: list[tuple[int, int, int, int]] = []
    _gutter_collect_leaves(
        bg,
        0,
        0,
        w,
        h,
        min_side,
        gutter_strength,
        gutter_smooth,
        gutter_margin_frac,
        0,
        gutter_max_depth,
        rects,
    )
    rects = _dedupe_rects(rects)
    if len(rects) > gutter_max_leaves:
        return [img.copy()]

    page_area = float(w * h)
    areas = [(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects]
    median_area = float(np.median(areas)) if areas else 0.0

    interior_score = _full_page_interior_gutter_score(bg, gutter_smooth)

    # Fallback: no real gutters detected or absurd fragmentation
    if len(rects) == 1 and interior_score < gutter_strength * 0.85:
        return [img.copy()]
    if len(rects) > 1 and median_area < 0.01 * page_area and len(rects) > 8:
        return [img.copy()]
    if not rects:
        return [img.copy()]

    rects.sort(key=lambda r: (r[1] // max(min_side, 1), r[0]))
    panels: list[cv2.typing.MatLike] = []
    for x0, y0, x1, y1 in rects:
        panels.append(img[y0:y1, x0:x1].copy())
    return panels


def _segment_legacy_from_bgr(
    img: cv2.typing.MatLike,
    *,
    min_side: int,
    min_coverage_fraction: float,
) -> SegmentResult:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    ih, iw = img.shape[:2]
    page_area = float(iw * ih)
    boxes: list[tuple[int, int, int, int]] = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= min_side and bh >= min_side:
            boxes.append((x, y, bw, bh))

    boxes = _filter_contained_boxes(boxes)
    covered = sum(float(bw * bh) for _, _, bw, bh in boxes)

    use_full = False
    if not boxes:
        use_full = True
    elif covered < min_coverage_fraction * page_area:
        use_full = True

    if use_full:
        panels_bgr = [img.copy()]
    else:
        boxes.sort(key=lambda b: (b[1] // max(min_side, 1), b[0]))
        panels_bgr = [img[y : y + bh, x : x + bw].copy() for x, y, bw, bh in boxes]

    return SegmentResult(panels_bgr=panels_bgr, panel_count=len(panels_bgr))


def _filter_contained_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    def contains(
        outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
    ) -> bool:
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner
        return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih

    kept: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda t: t[2] * t[3], reverse=True):
        if any(contains(k, b) for k in kept):
            continue
        kept.append(b)
    return kept


def write_panel_pngs_sequential(
    panels_bgr: list[cv2.typing.MatLike],
    chapter_dir: str,
    page_index: int,
) -> int:
    """Write panel_{page_index:04d}_{sub:03d}.png for stability across pages."""
    count = 0
    os.makedirs(chapter_dir, exist_ok=True)
    for pi, panel in enumerate(panels_bgr):
        name = f"panel_{page_index:04d}_{pi:03d}.png"
        path = os.path.join(chapter_dir, name)
        ok, buf = cv2.imencode(".png", panel)
        if not ok:
            raise RuntimeError(f"cv2.imencode failed for {path}")
        with open(path, "wb") as out:
            out.write(buf.tobytes())
        count += 1
    return count
