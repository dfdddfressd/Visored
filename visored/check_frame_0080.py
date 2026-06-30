"""
check_frame_0080.py — Investigate the frame_0080 / Chapter 506 page 11 vs 12 mismatch.

Prints out:
  1. The labels.json entry for the screenshot in question
  2. The metadata.json bbox info for the labeled panel AND the panel one page ahead
  3. File existence + last-modified timestamps for both panel files
     (helps determine if one was touched/rewritten more recently than the other,
     which would point to a partial re-splice as the cause)

Run from visored root:
    python check_frame_0080.py
"""

import json
from pathlib import Path
from datetime import datetime

LABELS_FILE = Path("labels.json")
PANELS_DIR  = Path("bleach_panels")

TARGET_SCREENSHOT = "Chapter 506/frame_0080.jpg"


def main():
    pairs = json.load(open(LABELS_FILE))
    entry = next((p for p in pairs if p["anime_screenshot"] == TARGET_SCREENSHOT), None)
    if entry is None:
        print(f"No label found for {TARGET_SCREENSHOT}")
        return

    print("=== labels.json entry ===")
    for k, v in entry.items():
        print(f"  {k}: {v}")

    labeled_folder = entry["manga_panel"].split("/")[0]
    labeled_file   = entry["manga_panel"].split("/")[1]
    labeled_page   = entry["page"]
    labeled_panel  = entry["panel"]

    # Construct the "one page ahead" candidate path
    next_page_file = f"p{labeled_page + 1:03d}_panel{labeled_panel:02d}.jpg"

    print(f"\n=== Comparing labeled panel vs one page ahead ===")
    print(f"  Labeled:     {labeled_folder}/{labeled_file}  (page {labeled_page})")
    print(f"  One ahead:   {labeled_folder}/{next_page_file}  (page {labeled_page + 1})")

    for desc, fname in [("LABELED", labeled_file), ("ONE PAGE AHEAD", next_page_file)]:
        path = PANELS_DIR / labeled_folder / fname
        print(f"\n--- {desc}: {path} ---")
        if not path.exists():
            print("  DOES NOT EXIST ON DISK")
            continue
        stat = path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        print(f"  Exists: True")
        print(f"  Size: {stat.st_size} bytes")
        print(f"  Last modified: {mtime}")

    # Cross-check against metadata.json bbox info
    meta_path = PANELS_DIR / labeled_folder / "metadata.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"\n=== metadata.json bbox info ===")
        for fname in [labeled_file, next_page_file]:
            match = next((p for p in meta if p["file"] == fname), None)
            if match:
                print(f"  {fname}: page={match['page']} panel={match['panel']} bbox={match['bbox']}")
            else:
                print(f"  {fname}: NOT FOUND in metadata.json")

    meta_mtime = meta_path.stat().st_mtime if meta_path.exists() else None
    if meta_mtime:
        print(f"\nmetadata.json last modified: {datetime.fromtimestamp(meta_mtime).isoformat(timespec='seconds')}")
        print(f"Label confirmed at:           {entry['confirmed_at']}")
        print(f"(Compare these — if metadata.json was modified AFTER the label was confirmed,")
        print(f" the chapter was likely re-spliced after labeling, which would explain drift.)")


if __name__ == "__main__":
    main()