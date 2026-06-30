"""
rebuild_dataset.py — Rebuild dataset.json from existing chapter metadata files.
Run this after manually deleting chapter folders to keep dataset.json in sync.

FIXED: now validates every metadata entry against the actual file on disk.
Entries pointing to jpg files that don't exist are dropped and reported,
rather than blindly trusted from metadata.json. This catches drift between
metadata.json (written once at splice time) and the actual folder contents
(which can change later — re-splices, manual deletions, etc.)

Usage:
    python rebuild_dataset.py
    python rebuild_dataset.py --panels-dir bleach_panels
"""

import argparse
import json
from pathlib import Path

DEFAULT_MANGA_ID = "a460ab18-22c1-47eb-a08a-9ee85fe37ec8"

parser = argparse.ArgumentParser()
parser.add_argument("--panels-dir", default="bleach_panels")
args = parser.parse_args()

panels_dir = Path(args.panels_dir)
all_panels = []
total_missing = 0
chapters_with_missing = []

meta_files = sorted(panels_dir.glob("*/metadata.json"))
print(f"Found {len(meta_files)} chapter folders:")

for meta_file in meta_files:
    chapter_dir = meta_file.parent
    with open(meta_file) as f:
        chapter_panels = json.load(f)

    valid_panels = []
    missing_in_chapter = []

    for p in chapter_panels:
        panel_path = chapter_dir / p["file"]
        if panel_path.exists():
            valid_panels.append(p)
        else:
            missing_in_chapter.append(p["file"])

    all_panels.extend(valid_panels)

    status = f"{len(valid_panels)} panels"
    if missing_in_chapter:
        status += f"  ⚠ {len(missing_in_chapter)} MISSING (in metadata.json but not on disk)"
        total_missing += len(missing_in_chapter)
        chapters_with_missing.append((chapter_dir.name, missing_in_chapter))

    print(f"  {chapter_dir.name}: {status}")

dataset = {
    "manga_id": DEFAULT_MANGA_ID,
    "total_chapters": len(meta_files),
    "total_panels": len(all_panels),
    "panels": all_panels,
}

out_path = panels_dir / "dataset.json"
with open(out_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\nRebuilt {out_path}")
print(f"  {dataset['total_chapters']} chapters, {dataset['total_panels']} valid panels")

if total_missing:
    print(f"\n⚠ WARNING: {total_missing} panel entries were in metadata.json but missing on disk.")
    print(f"  These were EXCLUDED from dataset.json. Affected chapters:")
    for chapter_name, missing_files in chapters_with_missing:
        print(f"    {chapter_name}: {len(missing_files)} missing")
        for f in missing_files[:5]:
            print(f"      - {f}")
        if len(missing_files) > 5:
            print(f"      ... and {len(missing_files) - 5} more")
    print(f"\n  Recommendation: these chapters' metadata.json may be stale from an")
    print(f"  earlier splice run. Consider re-splicing them to regenerate accurate metadata.")
else:
    print(f"  All panel entries verified against disk — no missing files.")