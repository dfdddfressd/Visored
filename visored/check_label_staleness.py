"""
check_label_staleness.py — Check every labeled chapter for the "labeled before
re-splice" drift bug found in Chapter 506.

For each chapter that appears in labels.json, compares each label's
confirmed_at timestamp against that chapter's metadata.json last-modified
time. If a label was confirmed BEFORE the metadata.json was last written,
the chapter was likely re-spliced after that label was made, meaning the
panel content at that path may have shifted — making the label potentially
stale/incorrect.

Run from visored root:
    python check_label_staleness.py
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

LABELS_FILE = Path("labels.json")
PANELS_DIR  = Path("bleach_panels")


def main():
    pairs = json.load(open(LABELS_FILE))

    # Group labels by manga_panel's chapter folder (not anime_screenshot folder,
    # since these can differ — e.g. Chapter 506 screenshots can map to Chapter 507+ panels)
    by_chapter = defaultdict(list)
    for p in pairs:
        chapter_folder = p["manga_panel"].split("/")[0]
        by_chapter[chapter_folder].append(p)

    print(f"Checking {len(by_chapter)} chapters referenced in labels.json...\n")

    total_stale = 0
    chapters_with_stale = []

    for chapter_folder in sorted(by_chapter.keys()):
        meta_path = PANELS_DIR / chapter_folder / "metadata.json"
        if not meta_path.exists():
            print(f"  {chapter_folder}: metadata.json MISSING — skipping")
            continue

        meta_mtime = datetime.fromtimestamp(meta_path.stat().st_mtime).isoformat(timespec="seconds")
        labels_for_chapter = by_chapter[chapter_folder]

        stale = [p for p in labels_for_chapter if p["confirmed_at"] < meta_mtime]

        if stale:
            print(f"  {chapter_folder}: {len(stale)}/{len(labels_for_chapter)} labels confirmed "
                  f"BEFORE metadata.json mtime ({meta_mtime}) — POTENTIALLY STALE")
            total_stale += len(stale)
            chapters_with_stale.append((chapter_folder, len(stale), len(labels_for_chapter), meta_mtime))
        else:
            print(f"  {chapter_folder}: {len(labels_for_chapter)} labels, all confirmed after "
                  f"metadata.json mtime ({meta_mtime}) — clean")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total potentially stale labels: {total_stale}")
    if chapters_with_stale:
        print(f"\nAffected chapters:")
        for folder, n_stale, n_total, mtime in chapters_with_stale:
            print(f"  {folder}: {n_stale}/{n_total} stale (metadata.json rewritten {mtime})")
    else:
        print("No stale labels found across any chapter.")


if __name__ == "__main__":
    main()