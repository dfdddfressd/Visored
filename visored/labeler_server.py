"""
labeler_server.py — Visored Labeling Tool Backend
==================================================
FastAPI server that powers the labeling UI. Loads CLIP + FAISS on startup,
serves screenshot/panel images, and saves confirmed anime→manga pairs.
 
Screenshots are now organized by chapter subfolder:
    screenshots/
      Chapter 1/
        frame_001.jpg
      Chapter 2/
        frame_001.jpg
 
Flat screenshots/ folders (no subfolders) still work as before.
 
Run:
    uvicorn labeler_server:app --reload --port 8000
"""
 
import json
import sys
from datetime import datetime
from pathlib import Path
 
import faiss
import numpy as np
import open_clip
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, ImageFilter
import io
 
sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL_REGISTRY
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
PANELS_DIR      = Path("bleach_panels")
SCREENSHOTS_DIR = Path("screenshots")
INDEX_DIR       = Path(".")
LABELS_FILE     = Path("labels.json")
SKIPS_FILE      = Path("skips.json")
TOP_K           = 5
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".webp"}
 
# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
 
app = FastAPI(title="Visored Labeler")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
 
# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
 
state = {
    "model":      None,
    "preprocess": None,
    "device":     None,
    "index":      None,
    "meta":       None,
    # Each queue entry is a Path. We use relative-to-screenshots as the key
    # so Chapter 1/frame_001.jpg and Chapter 2/frame_001.jpg don't collide.
    "queue":      [],
    "labeled":    set(),   # relative path strings e.g. "Chapter 1/frame_001.jpg"
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
 
    config_path = INDEX_DIR / "index_config.json"
    model_name = "ViT-B-32"
    if config_path.exists():
        with open(config_path) as f:
            model_name = json.load(f).get("model", model_name)
 
    cfg = MODEL_REGISTRY[model_name]
    print(f"[labeler] Loading {model_name} on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=cfg["pretrained"]
    )
    model.eval()
    model.to(device)
    state["model"]      = model
    state["preprocess"] = preprocess
 
    index_path = INDEX_DIR / "index.faiss"
    meta_path  = INDEX_DIR / "index_meta.json"
    if not index_path.exists():
        sys.exit("[labeler] index.faiss not found — run embed.py first")
    state["index"] = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        state["meta"] = json.load(f)
    print(f"[labeler] Index loaded: {state['index'].ntotal} panels")
 
    # Build labeled set using relative paths so subfolders work correctly
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
    """Return path relative to SCREENSHOTS_DIR as a forward-slash string."""
    return path.relative_to(SCREENSHOTS_DIR).as_posix()
 
 
def _rebuild_queue():
    """
    Walk screenshots/ recursively. Collect all image files, sort them so
    subfolders come out in chapter order, filter out already-labeled ones.
    """
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
 
def _embed(img: Image.Image, manga_mode: bool = False) -> np.ndarray:
    if manga_mode:
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img = img.convert("RGB")
    tensor = state["preprocess"](img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        feat = state["model"].encode_image(tensor)
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
    """
    Return the list of chapter subfolders inside screenshots/ so the UI
    can show a chapter selector for the current labeling session.
    """
    if not SCREENSHOTS_DIR.exists():
        return {"chapters": []}
    folders = sorted(
        p.name for p in SCREENSHOTS_DIR.iterdir()
        if p.is_dir()
    )
    # Also include a sentinel for flat (no-subfolder) screenshots
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
    shot     = q[state["cursor"]]
    rel      = _rel(shot)
    # chapter_folder is the immediate parent dir name if inside a subfolder,
    # otherwise None (flat layout)
    parent = shot.parent
    chapter_folder = parent.name if parent != SCREENSHOTS_DIR else None
    return {
        "done":           False,
        "filename":       shot.name,
        "rel":            rel,                          # e.g. "Chapter 1/frame_001.jpg"
        "url":            f"/screenshots/{rel}",
        "chapter_folder": chapter_folder,               # e.g. "Chapter 1" or null
        "index":          state["cursor"],
        "total":          len(q),
    }
 
 
@app.post("/search")
async def search(file: UploadFile = File(...), manga_mode: bool = False):
    if state["model"] is None:
        raise HTTPException(503, "Model still loading — try again in a few seconds")
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    vec  = _embed(img, manga_mode=manga_mode)
    return {"candidates": _search_vec(vec)}
 
 
@app.post("/label")
async def save_label(payload: dict):
    entry = {
        "anime_screenshot": payload["anime_screenshot"],   # relative path
        "manga_panel":      payload["manga_panel"],
        "chapter":          payload["chapter"],
        "page":             payload["page"],
        "panel":            payload["panel"],
        "score":            payload["score"],
        "manga_mode":       payload.get("manga_mode", False),
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
    rel = payload["rel"]   # relative path
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
    """
    Drag-and-drop upload. If chapter_folder is provided (e.g. "Chapter 1"),
    the file is saved into screenshots/Chapter 1/. Otherwise saved flat.
    """
    dest_dir = SCREENSHOTS_DIR / chapter_folder if chapter_folder else SCREENSHOTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename
    data = await file.read()
    with open(dest, "wb") as f:
        f.write(data)
 
    rel = _rel(dest)
    if rel not in state["labeled"]:
        state["queue"].insert(state["cursor"], dest)
 
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    vec  = _embed(img, manga_mode=False)
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
    """Serve screenshots from any subfolder depth."""
    path = SCREENSHOTS_DIR / rel_path
    if not path.exists():
        raise HTTPException(404, f"Screenshot not found: {path}")
    return FileResponse(str(path))
 




