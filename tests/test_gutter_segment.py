"""Synthetic tests for gutter and legacy segmentation."""

from __future__ import annotations

import cv2
import numpy as np

from visored.segment import segment_page_from_bytes


def _encode_png_bgr(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_gutter_stacked_full_width() -> None:
    """Two horizontal panels with a full-width white gutter band."""
    img = np.full((400, 400, 3), 255, np.uint8)
    img[0:175, :] = 0
    img[225:, :] = 0
    r = segment_page_from_bytes(
        _encode_png_bgr(img),
        min_side=50,
        segmentation="gutter",
    )
    assert r["panel_count"] == 2
    hs = sorted(p.shape[0] for p in r["panels_bgr"])
    assert hs[0] >= 150 and hs[1] >= 150


def test_gutter_two_column_full_height() -> None:
    """Two vertical panels with a full-height gutter."""
    img = np.full((400, 400, 3), 255, np.uint8)
    img[:, 0:175] = 0
    img[:, 225:] = 0
    r = segment_page_from_bytes(
        _encode_png_bgr(img),
        min_side=50,
        segmentation="gutter",
    )
    assert r["panel_count"] == 2
    ws = sorted(p.shape[1] for p in r["panels_bgr"])
    assert ws[0] >= 150 and ws[1] >= 150


def test_gutter_uniform_texture_fallback_single() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(40, 60, (300, 300, 3), dtype=np.uint8)
    r = segment_page_from_bytes(
        _encode_png_bgr(noise),
        min_side=50,
        segmentation="gutter",
    )
    assert r["panel_count"] == 1
    assert r["panels_bgr"][0].shape[:2] == (300, 300)


def test_legacy_segmentation_smoke() -> None:
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (90, 90), (0, 0, 0), -1)
    r = segment_page_from_bytes(
        _encode_png_bgr(img),
        min_side=30,
        segmentation="legacy",
    )
    assert r["panel_count"] >= 1


def test_rejects_oversized_payload() -> None:
    img = np.full((50, 50, 3), 255, np.uint8)
    blob = _encode_png_bgr(img)
    try:
        segment_page_from_bytes(blob, max_image_bytes=len(blob) - 1)
    except ValueError as e:
        assert "exceeds limit" in str(e)
    else:
        raise AssertionError("expected ValueError")
