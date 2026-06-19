"""
query.py — Visored Anime-to-Manga Panel Search
===============================================
Takes an anime screenshot, embeds it with CLIP, and searches the FAISS index
for the top matching manga panels.
 
Run:
    python query.py --image screenshot.jpg
    python query.py --image screenshot.jpg --index-dir index_L14
    python query.py --image screenshot.jpg --manga-mode
    python query.py --image screenshot.jpg --index-dir index_L14 --manga-mode --top-k 10
 
--manga-mode preprocesses the anime screenshot to look more like manga before
embedding: desaturate → boost contrast → sharpen edges. This nudges the query
vector closer to the manga panel embedding space without any retraining.
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import faiss
import numpy as np
import open_clip
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
 
# Import the model registry from embed.py so model/dim are never out of sync
sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL_REGISTRY
 
 
DEFAULT_TOP_K = 5
 
 
# ---------------------------------------------------------------------------
# Manga-mode preprocessing
# ---------------------------------------------------------------------------
 
def manga_preprocess(img: Image.Image) -> Image.Image:
    """
    Transform an anime screenshot to look closer to manga line art.
 
    Steps and why each one helps:
      1. Grayscale — removes color, which differs wildly between anime and manga.
             CLIP encodes color as a strong signal; stripping it forces the model
             to focus on shape and composition instead.
      2. Autocontrast — stretches the histogram so dark lines become black and
             light backgrounds become white, matching manga's high-contrast look.
      3. Edge enhance — sharpens contour lines, making character outlines more
             prominent, similar to manga's heavy inking.
      4. Back to RGB — CLIP expects 3-channel input; grayscale is single-channel.
             Converting back to RGB duplicates the gray channel across R, G, B.
 
    This is a heuristic, not a trained transform. It may help or may not —
    that's exactly what we're testing by running both modes side by side.
    """
    img = ImageOps.grayscale(img)           # step 1: strip color
    img = ImageOps.autocontrast(img)        # step 2: maximize contrast
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)  # step 3: sharpen edges
    img = img.convert("RGB")               # step 4: back to 3 channels
    return img
 
 
# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
 
def load_model(model_name: str, device: str):
    if model_name not in MODEL_REGISTRY:
        sys.exit(f"[query] ERROR: unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")
    cfg = MODEL_REGISTRY[model_name]
    print(f"[query] Loading {model_name} ({cfg['pretrained']} weights) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=cfg["pretrained"]
    )
    model.eval()
    model.to(device)
    return model, preprocess
 
 
def load_index(index_dir: Path):
    """
    Load FAISS index + metadata. Also reads index_config.json if present —
    this tells us which model was used to build the index, so we can
    auto-select the right model without the user needing to specify it.
    """
    index_path  = index_dir / "index.faiss"
    meta_path   = index_dir / "index_meta.json"
    config_path = index_dir / "index_config.json"
 
    if not index_path.exists():
        sys.exit(f"[query] ERROR: {index_path} not found. Run embed.py first.")
    if not meta_path.exists():
        sys.exit(f"[query] ERROR: {meta_path} not found. Run embed.py first.")
 
    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        meta = json.load(f)
 
    saved_model = None
    if config_path.exists():
        with open(config_path) as f:
            saved_model = json.load(f).get("model")
 
    print(f"[query] Loaded index: {index.ntotal} panels, built with {saved_model or 'unknown model'}.")
    return index, meta, saved_model
 
 
def embed_query(image_path: Path, model, preprocess, device: str, manga_mode: bool) -> np.ndarray:
    if not image_path.exists():
        sys.exit(f"[query] ERROR: image not found at {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        sys.exit(f"[query] ERROR: could not open image — {e}")
 
    if manga_mode:
        print("[query] manga-mode ON — preprocessing screenshot to grayscale + high contrast + edge enhance.")
        img = manga_preprocess(img)
 
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)
 
 
def search(index, meta: list[dict], query_vec: np.ndarray, top_k: int) -> list[dict]:
    scores, indices = index.search(query_vec, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue
        panel = meta[idx]
        results.append({
            "rank":    rank,
            "score":   float(score),
            "chapter": panel.get("chapter", "?"),
            "page":    panel.get("page", "?"),
            "panel":   panel.get("panel", "?"),
            "folder":  panel.get("folder", "?"),
            "file":    panel.get("file", "?"),
        })
    return results
 
 
def print_results(results: list[dict], image_path: Path, model_name: str, manga_mode: bool):
    mode_tag = " [manga-mode]" if manga_mode else ""
    print(f"\n{'='*62}")
    print(f"  Query : {image_path.name}")
    print(f"  Model : {model_name}{mode_tag}")
    print(f"{'='*62}")
    print(f"  {'Rank':<6} {'Score':<8} {'Chapter':<12} {'Page':<6} {'Panel':<7} File")
    print(f"  {'-'*56}")
    for r in results:
        print(
            f"  {r['rank']:<6} "
            f"{r['score']:.4f}   "
            f"Ch {r['chapter']:<10} "
            f"p{r['page']:<5} "
            f"#{r['panel']:<6} "
            f"{r['folder']}/{r['file']}"
        )
    print(f"{'='*62}\n")
    if results:
        best = results[0]
        print(f"  Best match: Chapter {best['chapter']}, page {best['page']}, "
              f"panel {best['panel']} (similarity: {best['score']:.4f})")
        print(f"  File: {best['folder']}/{best['file']}\n")
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Search manga panels matching an anime screenshot.")
    parser.add_argument("--image",      required=True,  help="Path to the anime screenshot")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--index-dir",  default=".",
                        help="Directory with index.faiss + index_meta.json (default: .)")
    parser.add_argument("--model",      default=None, choices=list(MODEL_REGISTRY),
                        help="CLIP model — auto-detected from index_config.json if omitted")
    parser.add_argument("--manga-mode", action="store_true",
                        help="Preprocess query image to look more like manga before embedding")
    args = parser.parse_args()
 
    image_path = Path(args.image)
    index_dir  = Path(args.index_dir)
 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
 
    index, meta, saved_model = load_index(index_dir)
 
    # Resolve which model to use: CLI flag > index_config.json > default
    model_name = args.model or saved_model or "ViT-B-32"
    if args.model and saved_model and args.model != saved_model:
        print(f"[query] WARNING: --model {args.model} overrides index built with {saved_model}. "
              f"Vectors may be incompatible.")
 
    model, preprocess = load_model(model_name, device)
    query_vec = embed_query(image_path, model, preprocess, device, args.manga_mode)
    results   = search(index, meta, query_vec, args.top_k)
    print_results(results, image_path, model_name, args.manga_mode)
 
 
if __name__ == "__main__":
    main()
 


