"""
rebuild_dataset.py — Rebuild dataset.json from existing chapter metadata files.
Run this after manually deleting chapter folders to keep dataset.json in sync.

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

meta_files = sorted(panels_dir.glob("*/metadata.json"))
print(f"Found {len(meta_files)} chapter folders:")

for meta_file in meta_files:
    with open(meta_file) as f:
        chapter_panels = json.load(f)
    all_panels.extend(chapter_panels)
    print(f"  {meta_file.parent.name}: {len(chapter_panels)} panels")

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
print(f"  {dataset['total_chapters']} chapters, {dataset['total_panels']} panels")
