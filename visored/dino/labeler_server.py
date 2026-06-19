"""
labeler_server.py — Visored Labeling Tool Backend (DINOv2 variant)
=====================================================================
FastAPI server that powers the labeling UI. Loads DINOv2 + FAISS on
startup, serves screenshot/panel images, and saves confirmed
anime→manga pairs.

CHANGED FROM CLIP VERSION:
- Loads DINOv2 (facebook/dinov2-base) instead of open_clip
- Reads checkpoint from index_config.json's "checkpoint" field automatically
- Removed manga_mode preprocessing entirely (color matters, established
  early in the project — anime and manga panels are both colored)
- No MODEL_REGISTRY import needed since DINOv2 isn't part of that registry

Screenshots are organized by chapter subfolder:
    screenshots/
      Chapter 1/
        frame_001.jpg
      Chapter 2/
        frame_001.jpg

Run from inside dino/:
    uvicorn labeler_server:app --reload --port 8000
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import io

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME      = "facebook/dinov2-base"
PANELS_DIR      = Path("../bleach_panels")
SCREENSHOTS_DIR = Path("../screenshots")
INDEX_DIR       = Path(".")   # looks for index.faiss etc inside dino/
LABELS_FILE     = Path("../labels.json")
SKIPS_FILE      = Path("../skips.json")
TOP_K           = 5
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Visored Labeler (DINOv2)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {
    "model":      None,
    "transform":  None,
    "device":     None,
    "index":      None,
    "meta":       None,
    "queue":      [],
    "labeled":    set(),
    "cursor":     0,
}


@app.on_event("startup")
def startup():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    state["device"] = device

    # ── Read checkpoint path from index config — keeps model and index in sync ──
    config_path = INDEX_DIR / "index_config.json"
    checkpoint_path = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            checkpoint_path = cfg.get("checkpoint")

    print(f"[labeler] Loading {MODEL_NAME} on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    if checkpoint_path:
        ckpt_full_path = Path(checkpoint_path)
        if ckpt_full_path.exists():
            print(f"[labeler] Loading fine-tuned weights from {ckpt_full_path}...")
            ckpt = torch.load(ckpt_full_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[labeler] Checkpoint: epoch {ckpt['epoch']}, "
                  f"Recall@1: {ckpt['recall_at_1']:.2%}")
        else:
            print(f"[labeler] WARNING: checkpoint path in config not found: {ckpt_full_path} — "
                  f"using zero-shot DINOv2")

    model.eval()
    model.to(device)
    state["model"] = model

    # Build the image transform matching DINOv2's expected preprocessing
    image_mean = processor.image_mean
    image_std  = processor.image_std
    image_size = processor.crop_size["height"] if hasattr(processor, "crop_size") else 224
    state["transform"] = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    index_path = INDEX_DIR / "index.faiss"
    meta_path  = INDEX_DIR / "index_meta.json"
    if not index_path.exists():
        sys.exit("[labeler] index.faiss not found — run embed_dino.py first")
    state["index"] = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        state["meta"] = json.load(f)
    print(f"[labeler] Index loaded: {state['index'].ntotal} panels")

    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            for entry in json.load(f):
                state["labeled"].add(entry["anime_screenshot"])
    if SKIPS_FILE.exists():
        with open(SKIPS_FILE) as f:
            for rel in json.load(f):
                state["labeled"].add(rel)

    _rebuild_queue()
    print(f"[labeler] Queue: {len(state['queue'])} screenshots to label")


def _rel(path: Path) -> str:
    return path.relative_to(SCREENSHOTS_DIR).as_posix()


def _rebuild_queue():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_shots = sorted(
        p for p in SCREENSHOTS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    state["queue"]  = [p for p in all_shots if _rel(p) not in state["labeled"]]
    state["cursor"] = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(img: Image.Image) -> np.ndarray:
    """
    Embed an image with DINOv2. No manga_mode parameter — color
    preprocessing was established as unnecessary early in the project
    since both domains here are colored.
    """
    tensor = state["transform"](img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        outputs = state["model"](pixel_values=tensor)
        feat = outputs.last_hidden_state[:, 0, :]   # CLS token
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def _search_vec(vec: np.ndarray) -> list[dict]:
    scores, indices = state["index"].search(vec, TOP_K)
    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        panel = state["meta"][idx]
        candidates.append({
            "index":   int(idx),
            "score":   round(float(score), 4),
            "chapter": panel.get("chapter", "?"),
            "page":    panel.get("page", "?"),
            "panel":   panel.get("panel", "?"),
            "folder":  panel.get("folder", "?"),
            "file":    panel.get("file", "?"),
            "url":     f"/panels/{panel['folder']}/{panel['file']}",
        })
    return candidates


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    total     = len(state["queue"]) + len(state["labeled"])
    labeled   = len(state["labeled"])
    remaining = len(state["queue"]) - state["cursor"]
    return {"total": total, "labeled": labeled, "remaining": remaining, "cursor": state["cursor"]}


@app.get("/screenshot_chapters")
def screenshot_chapters():
    if not SCREENSHOTS_DIR.exists():
        return {"chapters": []}
    folders = sorted(
        p.name for p in SCREENSHOTS_DIR.iterdir()
        if p.is_dir()
    )
    has_flat = any(
        p.is_file() and p.suffix.lower() in IMG_EXTS
        for p in SCREENSHOTS_DIR.iterdir()
    )
    return {"chapters": folders, "has_flat": has_flat}


@app.get("/next")
def next_screenshot():
    q = state["queue"]
    if state["cursor"] >= len(q):
        return JSONResponse({"done": True, "message": "All screenshots labeled!"})
    shot = q[state["cursor"]]
    rel  = _rel(shot)
    parent = shot.parent
    chapter_folder = parent.name if parent != SCREENSHOTS_DIR else None
    return {
        "done":           False,
        "filename":       shot.name,
        "rel":            rel,
        "url":            f"/screenshots/{rel}",
        "chapter_folder": chapter_folder,
        "index":          state["cursor"],
        "total":          len(q),
    }


@app.post("/search")
async def search(file: UploadFile = File(...), manga_mode: bool = False):
    # manga_mode kept as a parameter for frontend compatibility but ignored —
    # DINOv2 pipeline never uses grayscale preprocessing
    if state["model"] is None:
        raise HTTPException(503, "Model still loading — try again in a few seconds")
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    vec  = _embed(img)
    return {"candidates": _search_vec(vec)}


@app.post("/label")
async def save_label(payload: dict):
    entry = {
        "anime_screenshot": payload["anime_screenshot"],
        "manga_panel":      payload["manga_panel"],
        "chapter":          payload["chapter"],
        "page":             payload["page"],
        "panel":            payload["panel"],
        "score":            payload["score"],
        "manga_mode":       False,   # always False now — kept in schema for compatibility
        "manual_pick":      payload.get("manual_pick", False),
        "confirmed_at":     datetime.now().isoformat(timespec="seconds"),
    }

    labels = []
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    labels.append(entry)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)

    state["labeled"].add(payload["anime_screenshot"])
    state["cursor"] += 1
    return {"ok": True, "total_labeled": len(labels)}


@app.post("/skip")
async def skip(payload: dict):
    rel = payload["rel"]
    state["labeled"].add(rel)
    state["cursor"] += 1

    skips = []
    if SKIPS_FILE.exists():
        with open(SKIPS_FILE) as f:
            skips = json.load(f)
    if rel not in skips:
        skips.append(rel)
    with open(SKIPS_FILE, "w") as f:
        json.dump(skips, f, indent=2)

    return {"ok": True}


@app.post("/upload_screenshot")
async def upload_screenshot(file: UploadFile = File(...), chapter_folder: str = ""):
    dest_dir = SCREENSHOTS_DIR / chapter_folder if chapter_folder else SCREENSHOTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename
    data = await file.read()
    with open(dest, "wb") as f:
        f.write(data)

    rel = _rel(dest)
    if rel not in state["labeled"]:
        state["queue"].insert(state["cursor"], dest)

    img = Image.open(io.BytesIO(data)).convert("RGB")
    vec = _embed(img)
    return {
        "filename":       file.filename,
        "rel":            rel,
        "url":            f"/screenshots/{rel}",
        "chapter_folder": chapter_folder or None,
        "candidates":     _search_vec(vec),
    }


# ---------------------------------------------------------------------------
# Browse mode
# ---------------------------------------------------------------------------

@app.get("/chapters")
def list_chapters():
    if not PANELS_DIR.exists():
        return {"chapters": []}
    folders = sorted(
        p.name for p in PANELS_DIR.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )
    return {"chapters": folders}


@app.get("/chapter_panels/{folder:path}")
def chapter_panels(folder: str):
    meta_path = PANELS_DIR / folder / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, f"No metadata for folder: {folder}")
    with open(meta_path) as f:
        panels = json.load(f)
    return {
        "folder": folder,
        "panels": [
            {
                "chapter": p.get("chapter", "?"),
                "page":    p.get("page", "?"),
                "panel":   p.get("panel", "?"),
                "folder":  p.get("folder", folder),
                "file":    p.get("file", ""),
                "url":     f"/panels/{p.get('folder', folder)}/{p.get('file', '')}",
            }
            for p in panels
        ],
    }


# ---------------------------------------------------------------------------
# Static serving
# ---------------------------------------------------------------------------

@app.get("/panels/{folder}/{filename}")
def serve_panel(folder: str, filename: str):
    path = PANELS_DIR / folder / filename
    if not path.exists():
        raise HTTPException(404, f"Panel not found: {path}")
    return FileResponse(str(path))


@app.get("/screenshots/{rel_path:path}")
def serve_screenshot(rel_path: str):
    path = SCREENSHOTS_DIR / rel_path
    if not path.exists():
        raise HTTPException(404, f"Screenshot not found: {path}")
    return FileResponse(str(path))