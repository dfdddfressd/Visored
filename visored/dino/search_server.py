"""
search_server.py — Visored Public Search Server
=================================================
FastAPI server for the public-facing Visored demo. Accepts an anime
screenshot upload, embeds it with fine-tuned DINOv2, queries the FAISS
index, and returns the top 5 matching manga panels with chapter/page/panel
metadata.

No labeling, no disk writes, no queue — inference only.

Run from inside dino/:
    uvicorn search_server:app --port 8001

For production (EC2):
    uvicorn search_server:app --host 0.0.0.0 --port 8001
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME  = "facebook/dinov2-base"
PANELS_DIR  = Path("../bleach_panels")
INDEX_DIR   = Path(".")
TOP_K       = 6

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Visored Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

state = {
    "model":     None,
    "transform": None,
    "device":    None,
    "index":     None,
    "meta":      None,
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

    # Load checkpoint from index_config.json
    config_path = INDEX_DIR / "index_config.json"
    checkpoint_path = None
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            checkpoint_path = cfg.get("checkpoint")

    print(f"[search] Loading {MODEL_NAME} on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            print(f"[search] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"[search] Checkpoint epoch {ckpt['epoch']}, "
                  f"val Recall@1: {ckpt['recall_at_1']:.2%}")
        else:
            print(f"[search] WARNING: checkpoint not found at {ckpt_path} — using zero-shot")
    else:
        print(f"[search] No checkpoint config found — using zero-shot DINOv2")

    model.eval()
    model.to(device)
    state["model"] = model

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
        sys.exit("[search] index.faiss not found — run embed_dino.py first")
    state["index"] = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        state["meta"] = json.load(f)
    print(f"[search] Ready — {state['index'].ntotal} panels indexed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def embed(img: Image.Image) -> np.ndarray:
    tensor = state["transform"](img).unsqueeze(0).to(state["device"])
    with torch.no_grad():
        outputs = state["model"](pixel_values=tensor)
        feat = outputs.last_hidden_state[:, 0, :]
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)


def search(vec: np.ndarray) -> list[dict]:
    scores, indices = state["index"].search(vec, TOP_K)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        panel = state["meta"][idx]
        results.append({
            "rank":    len(results) + 1,
            "score":   round(float(score), 4),
            "chapter": panel.get("chapter", "?"),
            "page":    panel.get("page", "?"),
            "panel":   panel.get("panel", "?"),
            "folder":  panel.get("folder", "?"),
            "file":    panel.get("file", "?"),
            "url":     f"/panels/{panel['folder']}/{panel['file']}",
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path("search.html")
    if not html_path.exists():
        raise HTTPException(404, "search.html not found — place it in the dino/ folder")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/search")
async def search_endpoint(file: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(503, "Model still loading — try again in a moment")

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read image — please upload a JPEG or PNG")

    vec     = embed(img)
    results = search(vec)
    return {"results": results}


@app.get("/panels/{folder}/{filename}")
def serve_panel(folder: str, filename: str):
    path = PANELS_DIR / folder / filename
    if not path.exists():
        raise HTTPException(404, f"Panel not found: {folder}/{filename}")
    return FileResponse(str(path))


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "panels":  state["index"].ntotal if state["index"] else 0,
        "device":  state["device"],
    }