"""MangaDex API client: rate limits, pagination, at-home, CDN fetch."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import aiohttp
from aiolimiter import AsyncLimiter

API_BASE = "https://api.mangadex.org"
# Default: Bleach — Official Colored (whitespace gutters read cleaner than mono scans).
# Override with --manga-id or MANGADEX_MANGA_ID. Classic B&W series id:
# 239d6260-d71f-43b0-afff-074e3619e3de
DEFAULT_MANGA_ID = "a460ab18-22c1-47eb-a08a-9ee85fe37ec8"

log = logging.getLogger(__name__)


def build_user_agent(
    *,
    project_url: str = "",
    contact_email: str = "",
    version: str = "1.0",
) -> str:
    """Non-spoofed User-Agent for MangaDex (project URL preferred; email allowed as fallback)."""
    url = project_url.strip()
    email = contact_email.strip()
    if url:
        low = url.lower()
        if not (low.startswith("http://") or low.startswith("https://")):
            raise ValueError(
                "Project URL must use http:// or https:// "
                "(MangaDex requires an honest User-Agent)."
            )
        return f"Visored/{version} (+{url})"
    if email and "@" in email:
        return f"Visored/{version} (+{email})"
    raise ValueError(
        "Set a project URL or contact for User-Agent (MangaDex policy). "
        "Use --project-url or MANGADEX_PROJECT_URL, "
        "or --contact-email / MANGADEX_CONTACT_EMAIL."
    )


class MangaDexClient:
    """Dual limiters: global API ~5/s; at-home 40/min. CDN uses separate semaphore."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        user_agent: str,
        api_rps: float = 5.0,
        at_home_per_minute: int = 40,
        cdn_max_concurrent: int = 8,
        max_retries: int = 8,
    ) -> None:
        self._session = session
        self._user_agent = user_agent
        self._api_limiter = AsyncLimiter(api_rps, time_period=1.0)
        self._at_home_limiter = AsyncLimiter(at_home_per_minute, time_period=60.0)
        self._cdn_sem = asyncio.Semaphore(cdn_max_concurrent)
        self._max_retries = max_retries

    @staticmethod
    def default_headers(user_agent: str) -> dict[str, str]:
        return {"User-Agent": user_agent, "Accept": "application/json"}

    async def _sleep_retry_after(self, response: aiohttp.ClientResponse) -> None:
        ra = response.headers.get("Retry-After")
        if ra:
            try:
                await asyncio.sleep(float(ra))
                return
            except ValueError:
                pass
        xr = response.headers.get("X-RateLimit-Retry-After")
        if xr:
            try:
                ts = float(xr)
                wait = max(0.0, ts - time.time())
                await asyncio.sleep(wait)
            except ValueError:
                pass

    async def _request_api_json(
        self,
        method: str,
        url: str,
        *,
        use_at_home_limiter: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        delay = 1.0
        last_status = 0
        headers = self.default_headers(self._user_agent)

        for _attempt in range(self._max_retries):
            async with self._api_limiter:
                if use_at_home_limiter:
                    async with self._at_home_limiter:
                        async with self._session.request(
                            method, url, headers=headers, **kwargs
                        ) as resp:
                            last_status = resp.status
                            if resp.status == 403:
                                body = await resp.text()
                                raise RuntimeError(
                                    "MangaDex returned 403 (possible IP ban); stop. "
                                    f"{url} {body[:200]}"
                                )
                            if resp.status == 429:
                                await self._sleep_retry_after(resp)
                                jitter = random.uniform(0, 0.5)
                                await asyncio.sleep(delay + jitter)
                                delay = min(delay * 2, 120.0)
                                continue
                            resp.raise_for_status()
                            return await resp.json()
                else:
                    async with self._session.request(
                        method, url, headers=headers, **kwargs
                    ) as resp:
                        last_status = resp.status
                        if resp.status == 403:
                            body = await resp.text()
                            raise RuntimeError(
                                "MangaDex returned 403 (possible IP ban); stop. "
                                f"{url} {body[:200]}"
                            )
                        if resp.status == 429:
                            await self._sleep_retry_after(resp)
                            jitter = random.uniform(0, 0.5)
                            await asyncio.sleep(delay + jitter)
                            delay = min(delay * 2, 120.0)
                            continue
                        resp.raise_for_status()
                        return await resp.json()

        raise RuntimeError(
            f"API request failed after retries: {url} last_status={last_status}"
        )

    async def get_manga_feed_page(
        self, manga_id: str, limit: int, offset: int
    ) -> dict[str, Any]:
        params = {
            "translatedLanguage[]": "en",
            "order[chapter]": "asc",
            "limit": str(limit),
            "offset": str(offset),
        }
        q = urlencode(params)
        url = f"{API_BASE}/manga/{manga_id}/feed?{q}"
        return await self._request_api_json("GET", url)

    async def get_at_home_server(self, chapter_id: str) -> dict[str, Any]:
        url = f"{API_BASE}/at-home/server/{chapter_id}"
        return await self._request_api_json(
            "GET", url, use_at_home_limiter=True
        )

    async def fetch_cdn_bytes(self, url: str) -> bytes:
        delay = 1.0
        for _attempt in range(self._max_retries):
            async with self._cdn_sem:
                try:
                    async with self._session.get(
                        url,
                        headers={"User-Agent": self._user_agent},
                    ) as resp:
                        if resp.status == 403:
                            body = await resp.text()
                            raise RuntimeError(f"CDN 403 {url} {body[:120]}")
                        if resp.status == 429:
                            await self._sleep_retry_after(resp)
                            jitter = random.uniform(0, 0.5)
                            await asyncio.sleep(delay + jitter)
                            delay = min(delay * 2, 120.0)
                            continue
                        if 500 <= resp.status < 600:
                            await asyncio.sleep(delay + random.uniform(0, 0.3))
                            delay = min(delay * 2, 120.0)
                            continue
                        resp.raise_for_status()
                        return await resp.read()
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(delay + random.uniform(0, 0.3))
                    delay = min(delay * 2, 120.0)
        raise RuntimeError(f"CDN fetch failed after retries: {url}")


def chapter_readable_on_at_home(chapter: dict[str, Any]) -> bool:
    """
    False for chapters hosted only off-site (e.g. MANGA Plus): MangaDex lists them but
    /at-home/server returns no page files. Matches the idea behind creader's
    `select(.attributes.externalUrl == null)` filter on the feed.
    """
    attrs = chapter.get("attributes") or {}
    return not attrs.get("externalUrl")


async def paginate_chapter_ids(
    client: MangaDexClient,
    manga_id: str = DEFAULT_MANGA_ID,
    page_size: int = 100,
) -> AsyncIterator[dict[str, Any]]:
    """Yield chapter resources from manga feed (skips external / non-@Home chapters)."""
    offset = 0
    while True:
        payload = await client.get_manga_feed_page(manga_id, page_size, offset)
        data = payload.get("data") or []
        if not data:
            break
        for item in data:
            if not chapter_readable_on_at_home(item):
                log.debug(
                    "Skipping chapter %s (externalUrl — no MangaDex@Home pages)",
                    item.get("id"),
                )
                continue
            yield item
        if len(data) < page_size:
            break
        offset += page_size
        if offset + page_size > 10_000:
            log.warning(
                "Feed pagination would exceed MangaDex offset+limit 10000 cap; "
                "stopping early — use a narrower query or smaller title feed."
            )
            break


def _sanitize_path_component(name: str) -> str:
    """Strip characters unsafe or awkward in folder names (Windows + POSIX)."""
    bad = '\\/:*?"<>|'
    cleaned = []
    for ch in name.strip():
        if ch in bad or ord(ch) < 32:
            cleaned.append("-")
        else:
            cleaned.append(ch)
    s = "".join(cleaned).strip(" .-") or "unknown"
    return s


def chapter_folder_label(chapter_attrs: dict[str, Any], chapter_id: str) -> str:
    """Human-readable folder label: 'Chapter {number}' from MangaDex attributes."""
    num = chapter_attrs.get("chapter")
    if num is not None and str(num).strip() != "":
        safe = _sanitize_path_component(str(num).strip())
    else:
        vol = chapter_attrs.get("volume")
        if vol not in (None, ""):
            safe = _sanitize_path_component(f"vol.{vol}")
        else:
            short = chapter_id.replace("-", "")[:8]
            safe = _sanitize_path_component(f"extra-{short}")
    return f"Chapter {safe}"


def unique_chapter_directory_name(base_label: str, use_counts: dict[str, int]) -> str:
    """
    Ensure unique folder names when MangaDex returns duplicate chapter numbers.
    First use: 'Chapter 1'; further collisions: 'Chapter 1 (2)', etc.
    """
    n = use_counts.get(base_label, 0)
    if n == 0:
        use_counts[base_label] = 1
        return base_label
    n += 1
    use_counts[base_label] = n
    return f"{base_label} ({n})"


def page_url(base_url: str, quality: str, chapter_hash: str, filename: str) -> str:
    base = base_url.rstrip("/")
    q = quality if quality == "data-saver" else "data"
    return f"{base}/{q}/{chapter_hash}/{filename}"
