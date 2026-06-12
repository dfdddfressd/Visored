"""
extract_frames.py — Extract frames from anime episodes for labeling
===================================================================
Run:
    python extract_frames.py --episode episodes/episode5.mp4 --chapter 10 --interval 5
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCREENSHOTS_DIR = Path("screenshots")
FFMPEG = r"C:\Users\dfddd\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"


def extract_frames(episode_path: Path, chapter: int, interval: int):
    out_dir = SCREENSHOTS_DIR / f"Chapter {chapter}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Episode : {episode_path}")
    print(f"[extract] Output  : {out_dir}")
    print(f"[extract] Interval: one frame every {interval}s")

    existing = list(out_dir.glob("frame_*.jpg"))
    start_num = len(existing)
    print(f"[extract] {start_num} existing frames in folder — new ones will continue from there")

    output_pattern = str(out_dir / f"frame_%04d.jpg")

    cmd = [
        FFMPEG,
        "-i", str(episode_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        "-start_number", str(start_num + 1),
        output_pattern,
        "-hide_banner",
        "-loglevel", "warning",
    ]

    print(f"[extract] Running ffmpeg...\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        sys.exit("[extract] ffmpeg failed — check the episode path and that ffmpeg is in PATH")

    new_frames = list(out_dir.glob("frame_*.jpg"))
    print(f"[extract] Done. {len(new_frames)} total frames in {out_dir}")
    print(f"[extract] Next: run labeler_server.py and label the new screenshots")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from anime episodes.")
    parser.add_argument("--episode",  required=True, help="Path to episode file")
    parser.add_argument("--chapter",  required=True, type=int, help="Chapter folder to save frames into")
    parser.add_argument("--interval", type=int, default=3, help="Extract one frame every N seconds (default: 3)")
    args = parser.parse_args()

    episode_path = Path(args.episode)
    if not episode_path.exists():
        sys.exit(f"[extract] Episode not found: {episode_path}")

    extract_frames(episode_path, args.chapter, args.interval)


if __name__ == "__main__":
    main()
