"""
crop_screenshots.py — Crop the bottom N% of existing screenshots in place
==========================================================================
Use this to remove subtitle bars from already-extracted frames without
affecting filenames, frame counts, or labels.json.

Run:
    # Crop all chapters with subtitles
    python crop_screenshots.py --chapters 281 365 506 566

    # Crop a single chapter
    python crop_screenshots.py --chapters 281

    # Preview how many files would be affected without actually cropping
    python crop_screenshots.py --chapters 281 365 --dry-run

    # Custom crop amount (default is 10%)
    python crop_screenshots.py --chapters 281 --crop 0.12
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SCREENSHOTS_DIR = Path("screenshots")


def crop_chapter(chapter: int, crop: float, dry_run: bool) -> int:
    chapter_dir = SCREENSHOTS_DIR / f"Chapter {chapter}"
    if not chapter_dir.exists():
        print(f"[crop] WARNING: {chapter_dir} does not exist — skipping")
        return 0

    frames = sorted(chapter_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"[crop] WARNING: no frame_*.jpg files found in {chapter_dir} — skipping")
        return 0

    print(f"[crop] Chapter {chapter} — {len(frames)} frames, cropping bottom {crop*100:.0f}%")

    if dry_run:
        # Just show what would happen on the first frame
        sample = Image.open(frames[0])
        w, h = sample.size
        new_h = int(h * (1.0 - crop))
        print(f"[crop]   DRY RUN: {w}x{h} → {w}x{new_h} (removing {h - new_h}px from bottom)")
        return len(frames)

    for frame_path in tqdm(frames, desc=f"  Chapter {chapter}", leave=False):
        with Image.open(frame_path) as img:
            w, h = img.size
            new_h = int(h * (1.0 - crop))
            cropped = img.crop((0, 0, w, new_h))
        cropped.save(frame_path, "JPEG", quality=95)

    return len(frames)


def main():
    parser = argparse.ArgumentParser(description="Crop subtitle bars from existing screenshots in place.")
    parser.add_argument("--chapters", required=True, nargs="+", type=int,
                        help="Chapter numbers to crop (e.g. --chapters 281 365 506 566)")
    parser.add_argument("--crop",     type=float, default=0.10,
                        help="Fraction of frame height to remove from bottom (default: 0.10 = 10%%)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview what would be cropped without modifying any files")
    args = parser.parse_args()

    if not 0.0 < args.crop < 1.0:
        sys.exit("[crop] --crop must be between 0.0 and 1.0")

    if args.dry_run:
        print(f"[crop] DRY RUN — no files will be modified\n")

    total = 0
    for chapter in args.chapters:
        total += crop_chapter(chapter, args.crop, args.dry_run)

    print(f"\n[crop] Done. {total} frames {'would be' if args.dry_run else ''} cropped across {len(args.chapters)} chapter(s).")
    if not args.dry_run:
        print(f"[crop] Next: re-embed with embed_dino.py to update the FAISS index.")


if __name__ == "__main__":
    main()