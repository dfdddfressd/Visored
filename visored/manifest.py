"""Atomic read/write of per-chapter manifest for resume/idempotency."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PageRecord:
    page_index: int
    source_filename: str
    size_bytes: int
    sha256: str


@dataclass
class ChapterManifest:
    chapter_id: str
    pages_done: dict[str, PageRecord] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "pages_done": {
                k: asdict(v) for k, v in sorted(self.pages_done.items(), key=lambda x: int(x[0]))
            },
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ChapterManifest:
        pages_raw = data.get("pages_done") or {}
        pages: dict[str, PageRecord] = {}
        for key, rec in pages_raw.items():
            pages[str(key)] = PageRecord(
                page_index=int(rec["page_index"]),
                source_filename=str(rec["source_filename"]),
                size_bytes=int(rec["size_bytes"]),
                sha256=str(rec["sha256"]),
            )
        return cls(chapter_id=str(data["chapter_id"]), pages_done=pages)


def manifest_path(chapter_dir: str) -> str:
    return os.path.join(chapter_dir, ".manifest.json")


def load_manifest(chapter_dir: str) -> ChapterManifest | None:
    path = manifest_path(chapter_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ChapterManifest.from_json_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning(
            "Ignoring unreadable manifest %s (%s); delete or repair this file to "
            "resume safely.",
            path,
            e,
        )
        return None


def save_manifest_atomic(chapter_dir: str, manifest: ChapterManifest) -> None:
    os.makedirs(chapter_dir, exist_ok=True)
    path = manifest_path(chapter_dir)
    payload = json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(
        dir=chapter_dir, prefix=".manifest.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def mark_page_done(
    chapter_dir: str,
    chapter_id: str,
    manifest: ChapterManifest | None,
    page_index: int,
    source_filename: str,
    size_bytes: int,
    sha256_hex: str,
) -> ChapterManifest:
    if manifest is None:
        manifest = ChapterManifest(chapter_id=chapter_id)
    key = str(page_index)
    manifest.pages_done[key] = PageRecord(
        page_index=page_index,
        source_filename=source_filename,
        size_bytes=size_bytes,
        sha256=sha256_hex,
    )
    save_manifest_atomic(chapter_dir, manifest)
    return manifest
