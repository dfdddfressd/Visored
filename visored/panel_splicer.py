"""
Notes on Usage:

OpenCV contour detection (same approach as Kumiko by njean42: github.com/njean42/kumiko).
Very grateful to this repo, heavy inspiration.

Usage:
    python panel_splicer.py [--manga-id ID] [--out-dir DIR] [--quality data|data-saver]
                            [--project-url URL] [--contact-email EMAIL]
                            [--min-panel-ratio FLOAT] [--chapters N]
                            
                            
Requirements:
    pip install aiohttp aiolimiter opencv-python-headless Pillow
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

#importing from mangadex.py file, this case is Bleach (colored)

from mangadex import (
    DEFAULT_MANGA_ID,
    MangaDexClient,
    build_user_agent,
    chapter_folder_label,
    page_url,
    paginate_chapter_ids,
    unique_chapter_directory_name,
)

import aiohttp

log = logging.getLogger(__name__)

# Panel detection (Kumiko-style OpenCV contour approach)
"""
    Detect manga panels in a page image using OpenCV contour detection.

    Returns a list of (x, y, w, h) bounding boxes sorted in reading order
    (left-to-right top-to-bottom by default; right-to-left if rtl=True).
    Kumiko's approach utilizes LTR, but for manga, RTL is more appropriate.

    Parameters
    ----------
    img_bytes:
        Raw bytes of the page image (JPEG/PNG/WebP).
    min_panel_ratio:
        Panels smaller than this fraction of the page area are discarded
        (filters out tiny noise contours). Kumiko default is 1/100.
    background_tolerance:
        How far from pure white (255) a pixel may be and still count as
        background. Kumiko uses ~30.
    rtl:
        reads right-to-left, so True is correct.
"""


def detect_panels(
    img_bytes: bytes,
    *,
    min_panel_ratio: float = 0.02,
    background_tolerance: int = 30,
    rtl: bool = True,
) -> list[tuple[int, int, int, int]]:
    
    # Decode image
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cv2 could not decode image (%d bytes)", len(img_bytes))
        return []

    h, w = img.shape[:2]
    min_area = w * h * min_panel_ratio

    # Convert to grayscale and threshold: white/near-white → background (0),
    # everything else → foreground (255). This isolates the panel borders.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray,
        255 - background_tolerance,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # Dilate slightly to close gaps in panel borders (thin gutters).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find external contours — each closed contour is a candidate panel.
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    panels: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # Skip: too small, or essentially the whole page (splash noise).
        if area < min_area:
            continue
        if cw >= w * 0.95 and ch >= h * 0.95:
            continue
        panels.append((x, y, cw, ch))

    if not panels:
        # Fallback: treat the entire page as one panel (splash / full-page art).
        log.debug("No panels detected — treating full page as single panel.")
        return [(0, 0, w, h)]

    # Reading-order sort: group into rows by y-overlap, then sort each row by x.
    panels = _sort_reading_order(panels, rtl=rtl)
    return panels


def _sort_reading_order(
    panels: list[tuple[int, int, int, int]], *, rtl: bool
) -> list[tuple[int, int, int, int]]:
    """
    Sort panels into reading order using a row-grouping heuristic identical to
    Kumiko's approach: panels whose vertical centres are within the same band
    are treated as one row, then sorted horizontally within that row.
    """
    if not panels:
        return panels

    # Sort primarily by top-y.
    sorted_by_y = sorted(panels, key=lambda p: p[1])
    rows: list[list[tuple[int, int, int, int]]] = []
    current_row: list[tuple[int, int, int, int]] = [sorted_by_y[0]]

    for panel in sorted_by_y[1:]:
        px, py, pw, ph = panel
        # Compare vertical centre to the current row's average centre.
        row_centre = sum(p[1] + p[3] / 2 for p in current_row) / len(current_row)
        panel_centre = py + ph / 2
        row_height = max(p[3] for p in current_row)
        if abs(panel_centre - row_centre) < row_height * 0.5:
            current_row.append(panel)
        else:
            rows.append(current_row)
            current_row = [panel]
    rows.append(current_row)

    result: list[tuple[int, int, int, int]] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda p: p[0], reverse=rtl)
        result.extend(row_sorted)
    return result


def crop_panel(img_bytes: bytes, bbox: tuple[int, int, int, int]) -> bytes:
    """Crop a panel from raw image bytes and return JPEG bytes."""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    x, y, w, h = bbox
    crop = img[y : y + h, x : x + w]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()



# Pipeline: download pages → detect panels → save cropped panels

async def process_chapter(
    client: MangaDexClient,
    chapter: dict[str, Any],
    out_dir: Path,
    *,
    quality: str = "data",
    min_panel_ratio: float = 0.02,
    use_counts: dict[str, int],
) -> int:
    """Download all pages in a chapter, detect panels, save cropped images.

    Returns the number of panel images saved.
    """
    chapter_id: str = chapter["id"]
    attrs: dict[str, Any] = chapter.get("attributes") or {}

    base_label = chapter_folder_label(attrs, chapter_id)
    folder_name = unique_chapter_directory_name(base_label, use_counts)
    chapter_dir = out_dir / folder_name
    chapter_dir.mkdir(parents=True, exist_ok=True)

    # Fetch @Home server info for this chapter.
    try:
        server_data = await client.get_at_home_server(chapter_id)
    except Exception as exc:
        log.error("Could not get @Home server for %s: %s", chapter_id, exc)
        return 0

    base_url: str = server_data.get("baseUrl", "")
    chapter_data: dict[str, Any] = server_data.get("chapter", {})
    chapter_hash: str = chapter_data.get("hash", "")
    filenames: list[str] = chapter_data.get(
        "dataSaver" if quality == "data-saver" else "data", []
    )

    if not filenames:
        log.warning("No page files for chapter %s", chapter_id)
        return 0

    panels_saved = 0

    for page_idx, filename in enumerate(filenames, start=1):
        url = page_url(base_url, quality, chapter_hash, filename)
        try:
            img_bytes = await client.fetch_cdn_bytes(url)
        except Exception as exc:
            log.error("Failed to fetch page %d of %s: %s", page_idx, chapter_id, exc)
            continue

        # Detect panels.
        panels = detect_panels(img_bytes, min_panel_ratio=min_panel_ratio)
        log.info(
            "%s page %02d → %d panel(s)", folder_name, page_idx, len(panels)
        )

        for panel_idx, bbox in enumerate(panels, start=1):
            try:
                panel_bytes = crop_panel(img_bytes, bbox)
            except Exception as exc:
                log.error(
                    "Crop failed p%02d panel %d: %s", page_idx, panel_idx, exc
                )
                continue

            out_name = f"p{page_idx:03d}_panel{panel_idx:02d}.jpg"
            out_path = chapter_dir / out_name
            out_path.write_bytes(panel_bytes)
            panels_saved += 1

    return panels_saved


async def run(
    manga_id: str,
    out_dir: Path,
    *,
    quality: str = "data",
    min_panel_ratio: float = 0.02,
    max_chapters: int | None = None,
    user_agent: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=16)
    timeout = aiohttp.ClientTimeout(total=120, connect=15)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        client = MangaDexClient(session, user_agent)
        use_counts: dict[str, int] = {}
        total_panels = 0
        chapters_done = 0

        async for chapter in paginate_chapter_ids(client, manga_id):
            if max_chapters is not None and chapters_done >= max_chapters:
                break
            saved = await process_chapter(
                client,
                chapter,
                out_dir,
                quality=quality,
                min_panel_ratio=min_panel_ratio,
                use_counts=use_counts,
            )
            total_panels += saved
            chapters_done += 1
            log.info(
                "Chapter %d done — %d panels saved (total so far: %d)",
                chapters_done,
                saved,
                total_panels,
            )

    log.info(
        "Finished. %d chapters processed, %d panel images saved to %s",
        chapters_done,
        total_panels,
        out_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Bleach (colored) panels from MangaDex into a dataset."
    )
    p.add_argument(
        "--manga-id",
        default=os.environ.get("MANGADEX_MANGA_ID", DEFAULT_MANGA_ID),
        help="MangaDex manga UUID (default: Bleach Official Colored)",
    )
    p.add_argument(
        "--out-dir",
        default="bleach_panels",
        type=Path,
        help="Root output directory for panel images",
    )
    p.add_argument(
        "--quality",
        choices=["data", "data-saver"],
        default="data",
        help="'data' = full quality; 'data-saver' = compressed (faster, smaller)",
    )
    p.add_argument(
        "--min-panel-ratio",
        type=float,
        default=0.02,
        help="Minimum panel area as a fraction of page area (default 0.02 = 2%%)",
    )
    p.add_argument(
        "--chapters",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N chapters (omit for all)",
    )
    p.add_argument(
        "--project-url",
        default=os.environ.get("MANGADEX_PROJECT_URL", ""),
        help="Your project URL for User-Agent (required by MangaDex policy)",
    )
    p.add_argument(
        "--contact-email",
        default=os.environ.get("MANGADEX_CONTACT_EMAIL", ""),
        help="Fallback contact email for User-Agent",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        ua = build_user_agent(
            project_url=args.project_url,
            contact_email=args.contact_email,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(
        run(
            manga_id=args.manga_id,
            out_dir=args.out_dir,
            quality=args.quality,
            min_panel_ratio=args.min_panel_ratio,
            max_chapters=args.chapters,
            user_agent=ua,
        )
    )


if __name__ == "__main__":
    main()
