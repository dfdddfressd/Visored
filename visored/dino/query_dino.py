"""
query_dino.py — Visored Anime-to-Manga Panel Search (DINOv2 variant)
========================================================================
Mirrors clip/query.py exactly in CLI and structure. Takes an anime
screenshot, embeds it with DINOv2, searches the FAISS index.
 
Run:
    python query_dino.py --image ../screenshots/Chapter\ 1/frame.png --index-dir index_dino
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
 
MODEL_NAME    = "facebook/dinov2-base"
DINO_DIM      = 768
DEFAULT_TOP_K = 5
 
 
def load_model(device: str, checkpoint: Path | None = None):
    print(f"[query_dino] Loading {MODEL_NAME} on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
 
    if checkpoint is not None:
        if not checkpoint.exists():
            sys.exit(f"[query_dino] ERROR: checkpoint not found at {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[query_dino] Loaded checkpoint — epoch {ckpt['epoch']}, "
              f"Recall@1: {ckpt['recall_at_1']:.2%}")
 
    model.eval()
    model.to(device)
 
    image_mean = processor.image_mean
    image_std  = processor.image_std
    image_size = processor.crop_size["height"] if hasattr(processor, "crop_size") else 224
 
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])
 
    return model, transform
 
 
def load_index(index_dir: Path):
    index_path  = index_dir / "index.faiss"
    meta_path   = index_dir / "index_meta.json"
    config_path = index_dir / "index_config.json"
 
    if not index_path.exists():
        sys.exit(f"[query_dino] ERROR: {index_path} not found. Run embed_dino.py first.")
    if not meta_path.exists():
        sys.exit(f"[query_dino] ERROR: {meta_path} not found. Run embed_dino.py first.")
 
    index = faiss.read_index(str(index_path))
    with open(meta_path) as f:
        meta = json.load(f)
 
    saved_checkpoint = None
    if config_path.exists():
        with open(config_path) as f:
            saved_checkpoint = json.load(f).get("checkpoint")
 
    print(f"[query_dino] Loaded index: {index.ntotal} panels.")
    return index, meta, saved_checkpoint
 
 
def embed_query(image_path: Path, model, transform, device: str) -> np.ndarray:
    if not image_path.exists():
        sys.exit(f"[query_dino] ERROR: image not found at {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        sys.exit(f"[query_dino] ERROR: could not open image — {e}")
 
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=tensor)
        features = outputs.last_hidden_state[:, 0, :]
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
 
 
def print_results(results: list[dict], image_path: Path):
    print(f"\n{'='*62}")
    print(f"  Query : {image_path.name}")
    print(f"  Model : DINOv2 (ViT-B/14)")
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
 
 
def main():
    parser = argparse.ArgumentParser(description="Search manga panels matching an anime screenshot (DINOv2).")
    parser.add_argument("--image",     required=True, help="Path to the anime screenshot")
    parser.add_argument("--top-k",     type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", default=".", help="Directory with index files")
    args = parser.parse_args()
 
    image_path = Path(args.image)
    index_dir  = Path(args.index_dir)
 
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
 
    index, meta, saved_checkpoint = load_index(index_dir)
    checkpoint = Path(saved_checkpoint) if saved_checkpoint else None
 
    model, transform = load_model(device, checkpoint)
    query_vec = embed_query(image_path, model, transform, device)
    results   = search(index, meta, query_vec, args.top_k)
    print_results(results, image_path)
 
 
if __name__ == "__main__":
    main()
 