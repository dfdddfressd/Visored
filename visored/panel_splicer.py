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
import json
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
    """
    Detect manga panels using a dual-threshold approach that handles both
    white gutters (standard manga) and black gutters (colored Bleach pages).
 
    The original single-threshold approach only marked near-white pixels as
    background, so black borders between panels were treated as foreground
    content — causing adjacent panels to merge into one contour.
 
    Fix: run TWO threshold passes and combine them:
      1. White-gutter mask  — pixels close to pure white (original logic)
      2. Black-gutter mask  — pixels close to pure black (new)
    Any pixel that is either very white OR very black is considered a separator.
    The union of both masks is dilated and used for contour detection, giving
    the contour finder clean boundaries on both light and dark bordered pages.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("cv2 could not decode image (%d bytes)", len(img_bytes))
        return []
 
    h, w = img.shape[:2]
    min_area = w * h * min_panel_ratio
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # ── Mask 1: near-white pixels (original logic) ─────────────────────────
    # Pixels above (255 - tolerance) are background/gutter → foreground in mask
    _, white_mask = cv2.threshold(
        gray,
        255 - background_tolerance,
        255,
        cv2.THRESH_BINARY,      # white pixels → 255, everything else → 0
    )
 
    # ── Mask 2: near-black pixels (new) ───────────────────────────────────
    # Pixels below tolerance are black borders → foreground in mask
    _, black_mask = cv2.threshold(
        gray,
        background_tolerance,   # pixels darker than this → 255
        255,
        cv2.THRESH_BINARY_INV,  # invert: dark → 255, bright → 0
    )
 
    # ── Combine: anything that is a separator (white OR black) ────────────
    separator_mask = cv2.bitwise_or(white_mask, black_mask)
 
    # Invert so panel interiors are foreground (255) and separators are 0
    panel_mask = cv2.bitwise_not(separator_mask)
 
    # Dilate to close small gaps within panel content (speech bubbles, etc.)
    # Use a slightly larger kernel than before since black borders can be thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(panel_mask, kernel, iterations=2)
 
    # ── Find contours ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
 
    panels: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        if cw >= w * 0.95 and ch >= h * 0.95:
            continue
        panels.append((x, y, cw, ch))
 
    if not panels:
        log.debug("No panels detected — treating full page as single panel.")
        return [(0, 0, w, h)]
        
    # ── Full-page spread detection ───────────────────────────────────────────
    # If the total area covered by detected panels is less than 50% of the page,
    # or if removing the largest panel leaves almost nothing, the contour detector
    # has likely mis-split a full-page spread along speech bubble borders or ink
    # lines. In that case, treat the whole page as a single panel.
    if panels:
        page_area = w * h
        total_panel_area = sum(pw * ph for _, _, pw, ph in panels)
        largest_panel_area = max(pw * ph for _, _, pw, ph in panels)
    
        # Condition 1: panels collectively cover less than half the page
        # (means most of the page is being excluded as "gutter" — wrong)
        coverage_ratio = total_panel_area / page_area
        
        # Condition 2: largest single panel covers >65% of the page alone
        # (means this is probably just one big panel being detected alongside
        # small false-positive speech bubble crops)
        largest_ratio = largest_panel_area / page_area
        
        if coverage_ratio < 0.50 or largest_ratio > 0.65:
            log.debug(
                "Full-page spread detected (coverage=%.2f, largest=%.2f) "
                "— returning whole page as single panel.",
                coverage_ratio, largest_ratio,
            )
            return [(0, 0, w, h)]

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
    manga_id: str,
    *,
    quality: str = "data",
    min_panel_ratio: float = 0.02,
    use_counts: dict[str, int],
    seen_chapters: set[str],          # ← add this
) -> tuple[int, list[dict[str, Any]]]:

    chapter_id: str = chapter["id"]
    attrs: dict[str, Any] = chapter.get("attributes") or {}
    chapter_number: str = attrs.get("chapter") or "unknown"

    # DEDUPLICATION FIX — check seen_chapters, not use_counts
    if chapter_number in seen_chapters:
        log.info("Skipping duplicate chapter %s (already processed)", chapter_number)
        return 0, []
    seen_chapters.add(chapter_number)   # ← mark as seen before processing

 
    base_label = chapter_folder_label(attrs, chapter_id)
    folder_name = unique_chapter_directory_name(base_label, use_counts)
    chapter_dir = out_dir / folder_name
    chapter_dir.mkdir(parents=True, exist_ok=True)
 
    try:
        server_data = await client.get_at_home_server(chapter_id)
    except Exception as exc:
        log.error("Could not get @Home server for %s: %s", chapter_id, exc)
        return 0, []
 
    base_url: str = server_data.get("baseUrl", "")
    chapter_data: dict[str, Any] = server_data.get("chapter", {})
    chapter_hash: str = chapter_data.get("hash", "")
    filenames: list[str] = chapter_data.get(
        "dataSaver" if quality == "data-saver" else "data", []
    )
 
    if not filenames:
        log.warning("No page files for chapter %s", chapter_id)
        return 0, []
 
    panels_saved = 0
    chapter_metadata: list[dict[str, Any]] = []
 
    for page_idx, filename in enumerate(filenames, start=1):
        url = page_url(base_url, quality, chapter_hash, filename)
        try:
            img_bytes = await client.fetch_cdn_bytes(url)
        except Exception as exc:
            log.error("Failed to fetch page %d of %s: %s", page_idx, chapter_id, exc)
            continue
 
        source_filename = f"p{page_idx:03d}_source.jpg"
        source_path = chapter_dir / source_filename
        source_path.write_bytes(img_bytes)
 
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
 
            panel_filename = f"p{page_idx:03d}_panel{panel_idx:02d}.jpg"
            panel_path = chapter_dir / panel_filename
            panel_path.write_bytes(panel_bytes)
            panels_saved += 1
 
            x, y, w, h = bbox
            chapter_metadata.append({
                "manga_id": manga_id,
                "chapter_id": chapter_id,
                "chapter": chapter_number,
                "folder": folder_name,
                "page": page_idx,
                "panel": panel_idx,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "file": panel_filename,
                "source_page_file": source_filename,
            })
 
    chapter_meta_path = chapter_dir / "metadata.json"
    chapter_meta_path.write_text(
        json.dumps(chapter_metadata, indent=2), encoding="utf-8"
    )
    log.info("Wrote %s", chapter_meta_path)
 
    return panels_saved, chapter_metadata
 
 
async def run(
    manga_id: str,
    out_dir: Path,
    *,
    quality: str = "data",
    min_panel_ratio: float = 0.02,
    max_chapters: int | None = None,
    user_agent: str,
    skip_existing: bool = False,     
    language: str = "en",
    start_chapter: float | None = None,
    end_chapter: float | None = None,


) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=16)
    timeout = aiohttp.ClientTimeout(total=120, connect=15)
 
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        client = MangaDexClient(session, user_agent)
        use_counts: dict[str, int] = {}
        seen_chapters: set[str] = set()    # ← add this, was wrongly {} (a dict)
        total_panels = 0
        chapters_done = 0
        all_metadata: list[dict[str, Any]] = []

        async for chapter in paginate_chapter_ids(client, manga_id, language=language):
            if max_chapters is not None and chapters_done >= max_chapters:
                break

            attrs = chapter.get("attributes") or {}
            ch_num_str = attrs.get("chapter")
            try:
                ch_num = float(ch_num_str) if ch_num_str else None
            except ValueError:
                ch_num = None

            # ── Chapter range filtering ──────────────────────────────────────
            if start_chapter is not None and ch_num is not None and ch_num < start_chapter:
                continue
            if end_chapter is not None and ch_num is not None and ch_num > end_chapter:
                continue

            # ── Skip existing — match by chapter NUMBER, not exact folder name,
            # so "Chapter 11" and "Chapter 11 (2)" are both recognized as
            # "chapter 11 already exists" regardless of suffix ────────────────
            if skip_existing and ch_num_str:
                existing_for_this_chapter = list(out_dir.glob(f"Chapter {ch_num_str}*"))
                # Filter out false positives like "Chapter 110" matching "Chapter 11*"
                existing_for_this_chapter = [
                    d for d in existing_for_this_chapter
                    if d.name == f"Chapter {ch_num_str}" or d.name.startswith(f"Chapter {ch_num_str} (")
                ]
                if existing_for_this_chapter:
                    log.info("Skipping chapter %s — folder already exists: %s",
                            ch_num_str, existing_for_this_chapter[0].name)
                    seen_chapters.add(ch_num_str)
                    chapters_done += 1
                    continue

            saved, chapter_meta = await process_chapter(
                client, chapter, out_dir, manga_id,
                quality=quality,
                min_panel_ratio=min_panel_ratio,
                use_counts=use_counts,
                seen_chapters=seen_chapters,
            )

            if saved > 0 or chapter_meta:
                total_panels += saved
                chapters_done += 1
                all_metadata.extend(chapter_meta)
                log.info("Chapter %d done — %d panels saved (total so far: %d)",
                        chapters_done, saved, total_panels)


 
    # Write master dataset.json at the root of the output directory.
    dataset = {
        "manga_id": manga_id,
        "total_chapters": chapters_done,
        "total_panels": total_panels,
        "panels": all_metadata,
    }
    dataset_path = out_dir / "dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
 
    log.info(
        "Finished. %d chapters, %d panels saved to %s",
        chapters_done,
        total_panels,
        out_dir,
    )
    log.info("Master metadata written to %s", dataset_path)



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
        "--start-chapter",
        type=float,
        default=None,
        help="Only process chapters >= this number (for targeted re-splicing)",
    )
    p.add_argument(
        "--end-chapter",
        type=float,
        default=None,
        help="Stop processing chapters > this number"
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
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chapters whose output folder already exists on disk",
    )
    p.add_argument(
    "--language",
    default="en",
    help="Translation language code: 'en' for English, 'es' for Spanish (default: en)",
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
            skip_existing=args.skip_existing, 
            language=args.language,
            start_chapter=args.start_chapter,
            end_chapter=args.end_chapter,


        )
    )


if __name__ == "__main__":
    main()
