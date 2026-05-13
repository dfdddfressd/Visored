"""Command-line entry for Visored."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import ssl
import sys
from pathlib import Path

import aiohttp
import certifi

from visored.mangadex import DEFAULT_MANGA_ID, MangaDexClient, build_user_agent
from visored.pipeline import run_collection_pipeline
from visored.segment import (
    DEFAULT_GUTTER_MARGIN_FRAC,
    DEFAULT_GUTTER_MAX_DEPTH,
    DEFAULT_GUTTER_MAX_LEAVES,
    DEFAULT_GUTTER_SMOOTH,
    DEFAULT_GUTTER_STRENGTH,
    DEFAULT_MAX_IMAGE_BYTES,
    DEFAULT_MAX_IMAGE_SIDE,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="visored",
        description=(
            "Download English manga chapters from MangaDex and segment each "
            "page into panel PNGs via OpenCV (Visored)."
        ),
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=str(Path.home() / "Documents" / "Visored"),
        help="Root directory for Chapter # folders (default: ~/Documents/Visored)",
    )
    p.add_argument(
        "--manga-id",
        default=os.environ.get("MANGADEX_MANGA_ID", DEFAULT_MANGA_ID),
        metavar="UUID",
        help=(
            "MangaDex manga UUID (default: Bleach Official Colored; set MANGADEX_MANGA_ID; "
            "B&W main series 239d6260-d71f-43b0-afff-074e3619e3de)"
        ),
    )
    p.add_argument(
        "--quality",
        choices=("data", "data-saver"),
        default="data",
        help="at-home image quality path segment (default: data)",
    )
    p.add_argument(
        "--project-url",
        default=os.environ.get("MANGADEX_PROJECT_URL", ""),
        metavar="URL",
        help=(
            "Project or repo URL for User-Agent (preferred; or set MANGADEX_PROJECT_URL)"
        ),
    )
    p.add_argument(
        "--contact-email",
        default=os.environ.get("MANGADEX_CONTACT_EMAIL", ""),
        help="Fallback for User-Agent if --project-url is not set (or MANGADEX_CONTACT_EMAIL)",
    )
    p.add_argument(
        "--api-rps",
        type=float,
        default=5.0,
        help="Max requests per second to api.mangadex.org (default: 5)",
    )
    p.add_argument(
        "--at-home-per-minute",
        type=int,
        default=40,
        help="Max GET /at-home/server per minute (default: 40 per MangaDex)",
    )
    p.add_argument(
        "--cdn-concurrency",
        type=int,
        default=8,
        help="Max concurrent CDN image downloads (default: 8)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread/process pool workers for segmentation (default: 4)",
    )
    p.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Run OpenCV in threads (default) or processes",
    )
    p.add_argument(
        "--queue-size",
        type=int,
        default=4,
        help="Max queued pages awaiting segmentation (default: 4)",
    )
    p.add_argument(
        "--max-chapters",
        type=int,
        default=None,
        help="Stop after this many chapters (testing)",
    )
    p.add_argument(
        "--max-pages-per-chapter",
        "--max-pages",
        type=int,
        dest="max_pages_per_chapter",
        default=None,
        help="Stop after this many pages per chapter (testing)",
    )
    p.add_argument(
        "--trust-existing",
        action="store_true",
        help=(
            "Skip a page when panel PNGs already exist on disk even if the manifest "
            "does not record that page (unsafe if files are stale). When the manifest "
            "says a page is done, on-disk files are required or the page is "
            "re-fetched; if the chapter's source image name changed, panels are "
            "re-downloaded."
        ),
    )
    p.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_BYTES,
        metavar="N",
        help=(
            "Reject raw page payloads larger than this many bytes before decode "
            f"(default: {DEFAULT_MAX_IMAGE_BYTES})"
        ),
    )
    p.add_argument(
        "--max-image-side",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIDE,
        metavar="PX",
        help=(
            "Reject decoded images whose width or height exceeds this "
            f"(default: {DEFAULT_MAX_IMAGE_SIDE})"
        ),
    )
    p.add_argument(
        "--min-panel-side",
        type=int,
        default=150,
        help="Minimum width and height for a detected panel (default: 150)",
    )
    p.add_argument(
        "--segmentation",
        choices=("gutter", "legacy"),
        default="gutter",
        help=(
            "Panel split method: gutter (whitespace XY-cut, default) or legacy "
            "(ink contour bounding boxes). "
            "Spot-check: compare --segmentation gutter vs legacy on a few pages "
            "(--max-pages-per-chapter 3)."
        ),
    )
    p.add_argument(
        "--gutter-strength",
        type=float,
        default=DEFAULT_GUTTER_STRENGTH,
        metavar="F",
        help=(
            "Minimum mean background score along a candidate seam row/column (0–1; "
            "default: 0.92)"
        ),
    )
    p.add_argument(
        "--gutter-smooth",
        type=int,
        default=DEFAULT_GUTTER_SMOOTH,
        metavar="PX",
        help="Moving-average width for gutter projections (odd-ish; default: 5)",
    )
    p.add_argument(
        "--gutter-max-depth",
        type=int,
        default=DEFAULT_GUTTER_MAX_DEPTH,
        help="Maximum recursion depth for gutter XY-cut (default: 16)",
    )
    p.add_argument(
        "--gutter-margin-frac",
        type=float,
        default=DEFAULT_GUTTER_MARGIN_FRAC,
        metavar="F",
        help="Ignore seams this close to the edge of each region (default: 0.03)",
    )
    p.add_argument(
        "--gutter-max-leaves",
        type=int,
        default=DEFAULT_GUTTER_MAX_LEAVES,
        help="If gutter split yields more regions than this, use one full page (default: 64)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if int(args.max_image_bytes) < 1024:
        print("--max-image-bytes must be at least 1024", file=sys.stderr)
        return 2
    if int(args.max_image_side) < 64:
        print("--max-image-side must be at least 64", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    try:
        ua = build_user_agent(
            project_url=args.project_url,
            contact_email=args.contact_email,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    async def _run() -> None:
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, sock_read=120)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            client = MangaDexClient(
                session,
                ua,
                api_rps=float(args.api_rps),
                at_home_per_minute=int(args.at_home_per_minute),
                cdn_max_concurrent=int(args.cdn_concurrency),
            )
            await run_collection_pipeline(
                client,
                args.output_dir,
                manga_id=(str(args.manga_id).strip() or DEFAULT_MANGA_ID),
                quality=args.quality,
                queue_maxsize=int(args.queue_size),
                workers=int(args.workers),
                executor_kind=args.executor,
                min_panel_side=int(args.min_panel_side),
                segmentation=args.segmentation,
                gutter_strength=float(args.gutter_strength),
                gutter_smooth=int(args.gutter_smooth),
                gutter_max_depth=int(args.gutter_max_depth),
                gutter_margin_frac=float(args.gutter_margin_frac),
                gutter_max_leaves=int(args.gutter_max_leaves),
                max_chapters=args.max_chapters,
                max_pages_per_chapter=args.max_pages_per_chapter,
                trust_existing=bool(args.trust_existing),
                max_image_bytes=int(args.max_image_bytes),
                max_image_side=int(args.max_image_side),
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    except RuntimeError as e:
        logging.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
