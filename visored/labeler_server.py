"""
labeler_server.py — Visored Labeling Tool Backend
==================================================
FastAPI server that powers the labeling UI. Loads CLIP + FAISS on startup,
serves screenshot/panel images, and saves confirmed anime→manga pairs.
 
Run:
    uvicorn labeler_server:app --reload --port 8000
 
Then open labeler.html in your browser.
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
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, ImageFilter
import io
 
sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL_REGISTRY
 
# ---------------------------------------------------------------------------
# Config — adjust these paths if your layout differs
# ---------------------------------------------------------------------------
 
PANELS_DIR      = Path("bleach_panels")
SCREENSHOTS_DIR = Path("screenshots")      # folder of anime screenshots to label
INDEX_DIR       = Path(".")               # where index.faiss + index_config.json live
LABELS_FILE     = Path("labels.json")
SKIPS_FILE      = Path("skips.json")
TOP_K           = 5
 
# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
 
app = FastAPI(title="Visored Labeler")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ---------------------------------------------------------------------------
# State (loaded once at startup)
# ---------------------------------------------------------------------------
 
state = {
    "model":      None,
    "preprocess": None,
    "device":     None,
    "index":      None,
    "meta":       None,
    "queue":      [],      # list of screenshot Paths not yet labeled
    "labeled":    set(),   # screenshot filenames already labeled or skipped
    "cursor":     0,       # current position in queue
}
 
 
@app.on_event("startup")
def startup():
    # ---- device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    state["device"] = device
 
    # ---- model ----
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
 
    # ---- index ----
    index_path = INDEX_DIR / "index.faiss"
    meta_path  = INDEX_DIR / "index_meta.json"
    if not index_path.exists():
        sys.exit("[labeler] index.faiss not found — run embed.py first")
    state["index"] = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        state["meta"] = json.load(f)
    print(f"[labeler] Index loaded: {state['index'].ntotal} panels")
 
    # ---- labeled/skipped sets ----
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            for entry in json.load(f):
                state["labeled"].add(Path(entry["anime_screenshot"]).name)
    if SKIPS_FILE.exists():
        with open(SKIPS_FILE) as f:
            for name in json.load(f):
                state["labeled"].add(name)
 
    # ---- screenshot queue ----
    _rebuild_queue()
    print(f"[labeler] Queue: {len(state['queue'])} screenshots to label")
 
 
def _rebuild_queue():
    """Scan screenshots/ folder and build queue of unlabeled files."""
    if not SCREENSHOTS_DIR.exists():
        SCREENSHOTS_DIR.mkdir(parents=True)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_shots = sorted(
        p for p in SCREENSHOTS_DIR.iterdir()
        if p.suffix.lower() in exts
    )
    state["queue"]  = [p for p in all_shots if p.name not in state["labeled"]]
    state["cursor"] = 0
 
 
# ---------------------------------------------------------------------------
# Image embedding helper
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
 
 
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
 
@app.get("/status")
def status():
    total    = len(state["queue"]) + len(state["labeled"])
    labeled  = len(state["labeled"])
    remaining = len(state["queue"]) - state["cursor"]
    return {
        "total":     total,
        "labeled":   labeled,
        "remaining": remaining,
        "cursor":    state["cursor"],
    }
 
 
@app.get("/next")
def next_screenshot():
    """Return the next screenshot in the folder queue."""
    q = state["queue"]
    if state["cursor"] >= len(q):
        return JSONResponse({"done": True, "message": "All screenshots labeled!"})
    shot = q[state["cursor"]]
    return {
        "done":     False,
        "filename": shot.name,
        "url":      f"/screenshots/{shot.name}",
        "index":    state["cursor"],
        "total":    len(q),
    }
 
 
@app.post("/search")
async def search(file: UploadFile = File(...), manga_mode: bool = False):
    """
    Accept an uploaded image, embed it, search the FAISS index,
    return top-K panel candidates with metadata + image URLs.
    """
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    vec  = _embed(img, manga_mode=manga_mode)
 
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
    return {"candidates": candidates}
 
 
@app.post("/label")
async def save_label(payload: dict):
    """Save a confirmed anime→manga pair to labels.json."""
    entry = {
        "anime_screenshot": payload["anime_screenshot"],
        "manga_panel":      payload["manga_panel"],
        "chapter":          payload["chapter"],
        "page":             payload["page"],
        "panel":            payload["panel"],
        "score":            payload["score"],
        "manga_mode":       payload.get("manga_mode", False),
        "confirmed_at":     datetime.now().isoformat(timespec="seconds"),
    }
 
    labels = []
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)
    labels.append(entry)
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)
 
    # Mark as labeled and advance cursor
    state["labeled"].add(Path(payload["anime_screenshot"]).name)
    state["cursor"] += 1
    return {"ok": True, "total_labeled": len(labels)}
 
 
@app.post("/skip")
async def skip(payload: dict):
    """Mark a screenshot as skipped (won't appear in queue again)."""
    name = Path(payload["filename"]).name
    state["labeled"].add(name)
    state["cursor"] += 1
 
    skips = []
    if SKIPS_FILE.exists():
        with open(SKIPS_FILE) as f:
            skips = json.load(f)
    if name not in skips:
        skips.append(name)
    with open(SKIPS_FILE, "w") as f:
        json.dump(skips, f, indent=2)
 
    return {"ok": True}
 
 
@app.post("/upload_screenshot")
async def upload_screenshot(file: UploadFile = File(...)):
    """
    Accept a drag-and-drop screenshot, save it to screenshots/,
    add it to the front of the queue, and search immediately.
    """
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = SCREENSHOTS_DIR / file.filename
    data = await file.read()
    with open(dest, "wb") as f:
        f.write(data)
 
    # Add to front of queue if not already labeled
    if file.filename not in state["labeled"]:
        state["queue"].insert(state["cursor"], dest)
 
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    vec  = _embed(img, manga_mode=False)
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
    return {
        "filename":   file.filename,
        "url":        f"/screenshots/{file.filename}",
        "candidates": candidates,
    }
 
 
# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
 
@app.get("/panels/{folder}/{filename}")
def serve_panel(folder: str, filename: str):
    path = PANELS_DIR / folder / filename
    if not path.exists():
        raise HTTPException(404, f"Panel not found: {path}")
    return FileResponse(str(path))
 
 
@app.get("/screenshots/{filename}")
def serve_screenshot(filename: str):
    path = SCREENSHOTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Screenshot not found: {path}")
    return FileResponse(str(path))
 
 


