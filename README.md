# Visored

Visored downloads English manga chapters from [MangaDex](https://mangadex.org/) and splits each page into individual panel images using OpenCV. It streams work through an async producer–consumer pipeline: pages are fetched from MangaDex @Home, then segmented off the event loop with a thread or process pool.

## Requirements

- **Python** 3.10 or newer  
- Dependencies: `aiohttp`, `aiolimiter`, `certifi`, `numpy`, `opencv-python-headless`, `tqdm` (see `pyproject.toml` or `requirements.txt`)

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"     # editable install + pytest
```

Or without dev extras:

```bash
pip install -e .
```

The `visored` console script is registered by the package.

## MangaDex User-Agent (required)

MangaDex expects an identifiable `User-Agent`. You **must** supply one of:

- **`--project-url`** `https://...` (preferred), or environment variable **`MANGADEX_PROJECT_URL`**
- **`--contact-email`** `you@example.com`, or **`MANGADEX_CONTACT_EMAIL`**

If neither is set, the CLI exits with an error. Use a real project or contact you control; do not spoof other clients.

## Usage

```bash
visored --project-url https://github.com/yourname/your-repo
```

Typical options:

| Option | Purpose |
|--------|---------|
| `-o`, `--output-dir` | Root folder for chapter directories (default: `~/Documents/Visored`) |
| `--manga-id` | MangaDex manga UUID (default: built-in demo title; override with `MANGADEX_MANGA_ID`) |
| `--quality` | `data` or `data-saver` (@Home image tier) |
| `--segmentation` | `gutter` (default, whitespace XY-cut) or `legacy` (ink contour boxes) |
| `--workers`, `--executor` | Segmentation parallelism (`thread` or `process`) |
| `--max-chapters`, `--max-pages-per-chapter` | Limit scope for testing |
| `--trust-existing` | Skip re-download if `panel_*` files already exist (no manifest check) |
| `-v`, `--verbose` | Debug logging |

API and CDN behavior can be tuned with `--api-rps`, `--at-home-per-minute`, and `--cdn-concurrency` to stay within [MangaDex limits](https://api.mangadex.org/docs/).

Gutter tuning (when `--segmentation gutter`): `--gutter-strength`, `--gutter-smooth`, `--gutter-max-depth`, `--gutter-margin-frac`, `--gutter-max-leaves`, and `--min-panel-side`.

Full help:

```bash
visored --help
```

You can also run the package as a module:

```bash
python -m visored --project-url https://example.com/your-project
```

## What it does

1. **Feed** — Paginates the manga chapter feed (`translatedLanguage=en`, ascending chapter order), skipping chapters that only have an external URL (no @Home files).
2. **@Home** — Resolves the image server and page file list per chapter.
3. **Download** — Fetches each page from the CDN with retries, rate limits, and optional SSL via `certifi`.
4. **Segment** — Decodes the image and runs either **gutter** segmentation (recursive horizontal/vertical cuts along low-ink “gutters”) or **legacy** segmentation (adaptive threshold + contour bounding boxes). Output is written as PNGs.
5. **Resume** — Each chapter folder gets a `.manifest.json` recording completed pages (index, source filename, size, SHA-256). Re-runs skip pages already recorded with the same source filename.

Chapter directories are named from MangaDex metadata (for example `Chapter 1`, with disambiguation if numbers collide).

## Output layout

Under your output root:

```text
<output-dir>/
  Chapter 1/
    .manifest.json
    panel_0000_000.png
    panel_0000_001.png
    ...
  Chapter 2/
    ...
```

Panel files use the pattern `panel_{page_index:04d}_{panel_index:03d}.png`.

## Development

Run tests:

```bash
pytest
```

Tests cover synthetic gutter and legacy segmentation cases in `tests/`.

## Notes

- Default manga UUID in the code is a convenience for development; set `--manga-id` to any title you are allowed to access via MangaDex. Find UUIDs in the [MangaDex API docs](https://api.mangadex.org/docs/).
- Respect MangaDex [rules and rate limits](https://api.mangadex.org/docs/). This tool is for personal, policy-compliant use of the API and hosted images.
