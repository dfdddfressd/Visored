"""Producer-consumer pipeline: download pages, segment off the event loop."""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from tqdm import tqdm

from visored import manifest as manifest_mod
from visored.mangadex import MangaDexClient, is_trusted_mangadex_https_url, page_url
from visored.segment import (
    DEFAULT_GUTTER_MARGIN_FRAC,
    DEFAULT_GUTTER_MAX_DEPTH,
    DEFAULT_GUTTER_MAX_LEAVES,
    DEFAULT_GUTTER_SMOOTH,
    DEFAULT_GUTTER_STRENGTH,
    MIN_COVERAGE_FRACTION,
    MIN_PANEL_SIDE,
    SegmentationMode,
    segment_page_from_bytes,
    sha256_bytes,
    write_panel_pngs_sequential,
)

log = logging.getLogger(__name__)

ExecutorKind = Literal["thread", "process"]


@dataclass
class PageJob:
    chapter_dir: str
    chapter_id: str
    page_index: int
    source_filename: str
    payload_size: int
    sha256_hex: str
    image_bytes: bytes | None
    temp_path: str | None = None


def _process_page_thread(
    chapter_dir: str,
    page_index: int,
    image_bytes: bytes,
    min_side: int,
    segmentation: SegmentationMode,
    min_coverage_fraction: float,
    gutter_strength: float,
    gutter_smooth: int,
    gutter_max_depth: int,
    gutter_margin_frac: float,
    gutter_max_leaves: int,
    max_image_bytes: int,
    max_image_side: int,
) -> int:
    res = segment_page_from_bytes(
        image_bytes,
        min_side=min_side,
        min_coverage_fraction=min_coverage_fraction,
        segmentation=segmentation,
        gutter_strength=gutter_strength,
        gutter_smooth=gutter_smooth,
        gutter_max_depth=gutter_max_depth,
        gutter_margin_frac=gutter_margin_frac,
        gutter_max_leaves=gutter_max_leaves,
        max_image_bytes=max_image_bytes,
        max_image_side=max_image_side,
    )
    return write_panel_pngs_sequential(
        res["panels_bgr"], chapter_dir, page_index
    )


def _process_page_process(
    chapter_dir: str,
    page_index: int,
    temp_path: str,
    min_side: int,
    segmentation: SegmentationMode,
    min_coverage_fraction: float,
    gutter_strength: float,
    gutter_smooth: int,
    gutter_max_depth: int,
    gutter_margin_frac: float,
    gutter_max_leaves: int,
    max_image_bytes: int,
    max_image_side: int,
) -> int:
    try:
        with open(temp_path, "rb") as f:
            blob = f.read()
        res = segment_page_from_bytes(
            blob,
            min_side=min_side,
            min_coverage_fraction=min_coverage_fraction,
            segmentation=segmentation,
            gutter_strength=gutter_strength,
            gutter_smooth=gutter_smooth,
            gutter_max_depth=gutter_max_depth,
            gutter_margin_frac=gutter_margin_frac,
            gutter_max_leaves=gutter_max_leaves,
            max_image_bytes=max_image_bytes,
            max_image_side=max_image_side,
        )
        return write_panel_pngs_sequential(
            res["panels_bgr"], chapter_dir, page_index
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


async def run_collection_pipeline(
    client: MangaDexClient,
    output_dir: str,
    *,
    manga_id: str | None = None,
    quality: str = "data",
    queue_maxsize: int = 4,
    workers: int = 4,
    executor_kind: ExecutorKind = "thread",
    min_panel_side: int = MIN_PANEL_SIDE,
    segmentation: SegmentationMode = "gutter",
    min_coverage_fraction: float = MIN_COVERAGE_FRACTION,
    gutter_strength: float = DEFAULT_GUTTER_STRENGTH,
    gutter_smooth: int = DEFAULT_GUTTER_SMOOTH,
    gutter_max_depth: int = DEFAULT_GUTTER_MAX_DEPTH,
    gutter_margin_frac: float = DEFAULT_GUTTER_MARGIN_FRAC,
    gutter_max_leaves: int = DEFAULT_GUTTER_MAX_LEAVES,
    max_chapters: int | None = None,
    max_pages_per_chapter: int | None = None,
    trust_existing: bool = False,
    max_image_bytes: int | None = None,
    max_image_side: int | None = None,
    manga_feed_items: list[dict[str, Any]] | None = None,
) -> None:
    from visored.mangadex import DEFAULT_MANGA_ID, paginate_chapter_ids
    from visored.segment import DEFAULT_MAX_IMAGE_BYTES, DEFAULT_MAX_IMAGE_SIDE

    mid = manga_id if manga_id is not None else DEFAULT_MANGA_ID
    mib = int(max_image_bytes) if max_image_bytes is not None else DEFAULT_MAX_IMAGE_BYTES
    mis = int(max_image_side) if max_image_side is not None else DEFAULT_MAX_IMAGE_SIDE
    os.makedirs(output_dir, exist_ok=True)

    if manga_feed_items is None:
        manga_feed_items = []
        async for ch in paginate_chapter_ids(client, mid):
            manga_feed_items.append(ch)

    if not manga_feed_items:
        log.warning(
            "No chapters returned for manga id %s — UUID may be invalid/delisted, or "
            "there are no chapters for translatedLanguage=en. "
            "Find ids via https://api.mangadex.org/docs/ or --manga-id.",
            mid,
        )
    if max_chapters is not None:
        manga_feed_items = manga_feed_items[: max_chapters]

    total_chapters = len(manga_feed_items)
    pbar_ch = tqdm(total=total_chapters, desc="Chapters", unit="ch")
    pbar_pages = tqdm(desc="Pages processed", unit="pg", position=1, leave=True)
    pbar_panels = tqdm(desc="Panels saved", unit="pnl", position=2, leave=True)

    loop = asyncio.get_running_loop()
    if executor_kind == "process":
        executor: ThreadPoolExecutor | ProcessPoolExecutor = ProcessPoolExecutor(
            max_workers=max(1, workers)
        )
    else:
        executor = ThreadPoolExecutor(max_workers=max(1, workers))

    queue: asyncio.Queue[tuple[str, PageJob] | None] = asyncio.Queue(
        maxsize=max(1, queue_maxsize)
    )
    manifest_locks: dict[str, asyncio.Lock] = {}

    async def consumer() -> None:
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                _chapter_key, job = item
                try:
                    # ValueError, not assert: assertions are removed when Python runs with -O.
                    if executor_kind == "thread":
                        if job.image_bytes is None:
                            raise ValueError(
                                "PageJob.image_bytes is required when executor is thread"
                            )
                        fut = loop.run_in_executor(
                            executor,
                            _process_page_thread,
                            job.chapter_dir,
                            job.page_index,
                            job.image_bytes,
                            min_panel_side,
                            segmentation,
                            min_coverage_fraction,
                            gutter_strength,
                            gutter_smooth,
                            gutter_max_depth,
                            gutter_margin_frac,
                            gutter_max_leaves,
                            mib,
                            mis,
                        )
                    else:
                        if job.temp_path is None:
                            raise ValueError(
                                "PageJob requires temp_path when executor is process"
                            )
                        fut = loop.run_in_executor(
                            executor,
                            _process_page_process,
                            job.chapter_dir,
                            job.page_index,
                            job.temp_path,
                            min_panel_side,
                            segmentation,
                            min_coverage_fraction,
                            gutter_strength,
                            gutter_smooth,
                            gutter_max_depth,
                            gutter_margin_frac,
                            gutter_max_leaves,
                            mib,
                            mis,
                        )
                    n_panels = await asyncio.wait_for(fut, timeout=None)
                except Exception as e:
                    log.exception(
                        "Segment/write failed page=%s: %s",
                        job.page_index,
                        e,
                    )
                    if executor_kind == "process" and job.temp_path:
                        try:
                            os.unlink(job.temp_path)
                        except OSError:
                            pass
                    pbar_pages.update(1)
                    continue

                lock = manifest_locks.setdefault(
                    _chapter_key, asyncio.Lock()
                )
                async with lock:
                    man = manifest_mod.load_manifest(job.chapter_dir)
                    manifest_mod.mark_page_done(
                        job.chapter_dir,
                        job.chapter_id,
                        man,
                        job.page_index,
                        job.source_filename,
                        job.payload_size,
                        job.sha256_hex,
                    )
                pbar_panels.update(n_panels)
                pbar_pages.update(1)
            finally:
                queue.task_done()

    num_consumers = max(1, workers)
    consumers = [asyncio.create_task(consumer()) for _ in range(num_consumers)]

    producer_task = asyncio.create_task(
        _producer(
            client,
            output_dir,
            manga_feed_items,
            quality,
            queue,
            executor_kind,
            max_pages_per_chapter,
            trust_existing,
            mib,
            pbar_ch,
            pbar_pages,
        )
    )

    try:
        await producer_task
        await queue.join()
        for _ in consumers:
            await queue.put(None)
        await asyncio.gather(*consumers)
    except Exception:
        log.exception("Pipeline failed; cancelling consumers")
        for c in consumers:
            c.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        raise
    finally:
        executor.shutdown(wait=True)
        pbar_ch.close()
        pbar_pages.close()
        pbar_panels.close()


async def _producer(
    client: MangaDexClient,
    output_dir: str,
    manga_feed_items: list[dict[str, Any]],
    quality: str,
    queue: asyncio.Queue[tuple[str, PageJob] | None],
    executor_kind: ExecutorKind,
    max_pages_per_chapter: int | None,
    trust_existing: bool,
    max_image_bytes: int,
    pbar_ch: tqdm,
    pbar_pages: tqdm,
) -> None:
    from visored.mangadex import (
        chapter_folder_label,
        unique_chapter_directory_name,
    )

    folder_counts: dict[str, int] = {}

    for ch in manga_feed_items:
        cid = ch.get("id")
        attrs = (ch.get("attributes") or {}) if isinstance(ch, dict) else {}
        if not cid:
            pbar_ch.update(1)
            continue
        base_label = chapter_folder_label(attrs, str(cid))
        folder = unique_chapter_directory_name(base_label, folder_counts)
        chapter_dir = os.path.join(output_dir, folder)
        os.makedirs(chapter_dir, exist_ok=True)

        man = manifest_mod.load_manifest(chapter_dir)
        if man is None:
            man = manifest_mod.ChapterManifest(chapter_id=str(cid))

        try:
            at_home = await client.get_at_home_server(str(cid))
        except Exception as e:
            log.exception("at-home failed %s: %s", cid, e)
            pbar_ch.update(1)
            continue

        data = at_home.get("chapter") or {}
        base_url = at_home.get("baseUrl") or data.get("baseUrl")
        chash = data.get("hash")
        files = data.get("data") or []
        if not base_url or not chash or not files:
            log.warning(
                "Incomplete at-home payload for chapter %s — often an external "
                "or delisted chapter; try another chapter or manga.",
                cid,
            )
            pbar_ch.update(1)
            continue

        base_s = str(base_url).strip()
        if not is_trusted_mangadex_https_url(base_s):
            log.warning(
                "Skipping chapter %s: untrusted at-home baseUrl %r "
                "(expected https://*.mangadex.org)",
                cid,
                base_url,
            )
            pbar_ch.update(1)
            continue

        if max_pages_per_chapter is not None:
            files = files[: max_pages_per_chapter]

        loose_skip_warned = False
        for page_index, filename in enumerate(files):
            fname = str(filename)
            pat = os.path.join(chapter_dir, f"panel_{page_index:04d}_*.png")
            has_panels = bool(glob.glob(pat))

            rec = man.pages_done.get(str(page_index))
            manifest_agrees = rec is not None and rec.source_filename == fname

            if manifest_agrees:
                if has_panels:
                    pbar_pages.update(1)
                    continue
                log.info(
                    "%s page %s: manifest done but panel PNGs missing; re-downloading.",
                    folder,
                    page_index,
                )

            if rec is not None and rec.source_filename != fname:
                log.info(
                    "%s page %s: source file changed (%r -> %r); re-downloading.",
                    folder,
                    page_index,
                    rec.source_filename,
                    fname,
                )

            wrong_manifest = rec is not None and rec.source_filename != fname
            if (
                trust_existing
                and has_panels
                and not wrong_manifest
            ):
                if rec is None and not loose_skip_warned:
                    log.warning(
                        "%s: --trust-existing skips from disk without a manifest "
                        "record for that page; delete panels or drop the flag if "
                        "output may be stale.",
                        folder,
                    )
                    loose_skip_warned = True
                pbar_pages.update(1)
                continue

            url = page_url(base_s, quality, str(chash), fname)
            try:
                blob = await client.fetch_cdn_bytes(url)
            except Exception as e:
                log.exception("CDN %s: %s", url, e)
                pbar_pages.update(1)
                continue

            if len(blob) > max_image_bytes:
                log.error(
                    "Skipping page %s: image %s bytes exceeds limit %s",
                    page_index,
                    len(blob),
                    max_image_bytes,
                )
                pbar_pages.update(1)
                continue

            payload_size = len(blob)
            digest = sha256_bytes(blob)
            temp_path: str | None = None
            ib: bytes | None = blob
            if executor_kind == "process":
                fd, tmp = tempfile.mkstemp(
                    suffix=".img", prefix=f"p{page_index}_", dir=chapter_dir
                )
                with os.fdopen(fd, "wb") as out:
                    out.write(blob)
                ib = None
                temp_path = tmp

            job = PageJob(
                chapter_dir=chapter_dir,
                chapter_id=str(cid),
                page_index=page_index,
                source_filename=str(filename),
                payload_size=payload_size,
                sha256_hex=digest,
                image_bytes=ib,
                temp_path=temp_path,
            )
            await queue.put((folder, job))

        pbar_ch.update(1)
