# Visored

visored.net - Visored is an AI-powered visual retrieval system that maps Bleach anime screenshots to their corresponding manga source panels. The core challenge is cross-domain matching: anime frames are full-color and cinematically lit, while manga panels are black-and-white line art — standard pixel-level similarity approaches fail badly across this boundary.

The system uses fine-tuned DINOv2 embeddings and FAISS similarity search to retrieve semantically matching panels from a corpus of 42,000+ images spanning 706 manga chapters, achieving **81% Recall@1** on the full index.

## Architecture

```
Anime screenshot
      ↓
DINOv2 encoder (fine-tuned, dinov2-base)
      ↓
FAISS IndexFlatIP similarity search
      ↓
Top-k manga panel candidates
```

The embedding model was trained with a contrastive objective on labeled anime-frame/manga-panel pairs. Training data was constructed manually using a custom browser-based labeling tool (`labeler.html` + `labeler_server.py`).

## Pipeline Components

| Component | Description |
|-----------|-------------|
| `mangadex_client` | Async MangaDex API client for chapter/page fetching |
| `panel_splicer` | OpenCV-based panel segmentation (gutter + contour modes) |
| `labeler` | Browser-based tool for annotating ground-truth anime→manga pairs |
| `embedder` | DINOv2 fine-tuning and inference pipeline |
| `index` | FAISS index construction and nearest-neighbor search |

## Data

- **Source:** MangaDex (English + Latin American Spanish scans)
- **Chapters:** 706
- **Panels:** 42,112 after segmentation and metadata reconciliation
- **Segmentation:** Recursive gutter XY-cut (default) or adaptive contour detection

## Results

| Model | Recall@1 |
|-------|----------|
| Zero-shot CLIP | baseline |
| Fine-tuned CLIP (ViT-L/14) | improved |
| Fine-tuned DINOv2 (dinov2-base) | **~81%** |

## Requirements

- Python 3.10+
- Dependencies: `aiohttp`, `aiolimiter`, `certifi`, `numpy`, `opencv-python-headless`, `torch`, `tqdm`

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

### Data collection & panel segmentation
```bash
visored --project-url https://github.com/yourname/visored
```

See the original pipeline docs below for full CLI options.

### Building the index
```bash
python -m visored.index --build
```

### Querying
```bash
python -m visored.index --query path/to/screenshot.png --top-k 5
```

## MangaDex User-Agent (required)

MangaDex expects an identifiable `User-Agent`. Supply one of:
- `--project-url https://...` or `MANGADEX_PROJECT_URL`
- `--contact-email you@example.com` or `MANGADEX_CONTACT_EMAIL`

## Pipeline Details

1. **Feed** — Paginates the chapter feed, skipping external-only chapters
2. **Download** — Fetches pages with retries, rate limits, and resume support via `.manifest.json`
3. **Segment** — Splits pages into panels via gutter cuts or contour detection
4. **Label** — Ground-truth pairs annotated via browser labeling tool
5. **Embed** — Fine-tuned DINOv2 encodes panels into a shared embedding space
6. **Index** — FAISS builds a searchable index over the panel corpus

## Output Layout

```text
/
  Chapter 1/
    .manifest.json
    panel_0000_000.png
    panel_0000_001.png
  Chapter 2/
    ...
```

## Development

```bash
pytest
```
- Default manga UUID in the code is a convenience for development; set `--manga-id` to any title you are allowed to access via MangaDex. Find UUIDs in the [MangaDex API docs](https://api.mangadex.org/docs/).
- Respect MangaDex [rules and rate limits](https://api.mangadex.org/docs/). This tool is for personal, policy-compliant use of the API and hosted images.
