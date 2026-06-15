"""
query.py — Visored Anime-to-Manga Panel Search
===============================================
Takes an anime screenshot, embeds it through CLIP + the anime projection head
(if a checkpoint is provided), and searches the FAISS index for the top
matching manga panels.

Run:
    python query.py --image screenshot.jpg --index-dir index_finetuned3
    python query.py --image screenshot.jpg --index-dir index_finetuned3 --top-k 10

The checkpoint and model are auto-detected from index_config.json.
Do not pass --manga-mode — we are working with colored manga.
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Import the model registry from embed.py so model/dim are never out of sync
sys.path.insert(0, str(Path(__file__).parent))
from embed import MODEL_REGISTRY, ProjectionHead


DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str, checkpoint: Path | None = None):
    """
    Load CLIP backbone and optionally the anime projection head.

    For querying we load the ANIME head (not manga head) — the anime head
    was trained to map anime screenshots into the same space that the manga
    head maps manga panels into. Using the wrong head would defeat the purpose.

    Returns:
        clip_model  — frozen CLIP backbone
        anime_head  — ProjectionHead if checkpoint has heads, else None
        preprocess  — image preprocessor
        embed_dim   — 256 with head, 768 without
    """
    if model_name not in MODEL_REGISTRY:
        sys.exit(f"[query] ERROR: unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    cfg      = MODEL_REGISTRY[model_name]
    clip_dim = cfg["dim"]

    print(f"[query] Loading {model_name} ({cfg['pretrained']} weights) on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=cfg["pretrained"]
    )
    clip_model.eval()
    clip_model.to(device)

    anime_head = None
    embed_dim  = clip_dim

    if checkpoint is not None:
        if not checkpoint.exists():
            sys.exit(f"[query] ERROR: checkpoint not found at {checkpoint}")

        print(f"[query] Loading projection head weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)

        if "anime_head" in ckpt:
            proj_dim   = ckpt.get("proj_dim", 256)
            anime_head = ProjectionHead(clip_dim, proj_dim).to(device)
            anime_head.load_state_dict(ckpt["anime_head"])
            anime_head.eval()
            embed_dim = proj_dim
            print(f"[query] Anime projection head loaded — epoch {ckpt['epoch']}, "
                  f"Recall@1: {ckpt['recall_at_1']:.2%}, embed_dim: {embed_dim}")
        elif "state_dict" in ckpt:
            # Legacy checkpoint — full model weights, no projection head
            print(f"[query] Legacy checkpoint (full model weights) — loading into backbone.")
            clip_model.load_state_dict(ckpt["state_dict"])
            clip_model.eval()
        else:
            sys.exit(f"[query] ERROR: unrecognized checkpoint format in {checkpoint}")

    return clip_model, anime_head, preprocess, embed_dim


def load_index(index_dir: Path):
    """
    Load FAISS index + metadata + config.
    Config tells us which model and checkpoint built this index.
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

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    saved_model      = config.get("model")
    saved_checkpoint = config.get("checkpoint")
    has_heads        = config.get("has_heads", False)

    print(f"[query] Loaded index: {index.ntotal} panels, "
          f"built with {saved_model or 'unknown model'}"
          f"{' + projection heads' if has_heads else ' (raw CLIP)'}.")

    return index, meta, saved_model, saved_checkpoint, has_heads


def embed_query(
    image_path: Path,
    clip_model,
    anime_head,
    preprocess,
    device: str,
) -> np.ndarray:
    """
    Embed a single anime screenshot.
    If anime_head is provided, passes CLIP features through it (256-dim).
    Otherwise returns raw L2-normalized CLIP features (768-dim).
    """
    if not image_path.exists():
        sys.exit(f"[query] ERROR: image not found at {image_path}")
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        sys.exit(f"[query] ERROR: could not open image — {e}")

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(tensor)
        if anime_head is not None:
            embedding = anime_head(features)
        else:
            embedding = features / features.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().astype(np.float32)


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


def print_results(results: list[dict], image_path: Path, model_name: str, has_heads: bool):
    mode_tag = " + projection heads" if has_heads else " (raw CLIP)"
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
    parser.add_argument("--image",     required=True, help="Path to the anime screenshot")
    parser.add_argument("--top-k",     type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", default=".", help="Directory with index files")
    parser.add_argument("--model",     default=None, choices=list(MODEL_REGISTRY),
                        help="Override model (auto-detected from index_config.json if omitted)")
    args = parser.parse_args()

    image_path = Path(args.image)
    index_dir  = Path(args.index_dir)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    index, meta, saved_model, saved_checkpoint, has_heads = load_index(index_dir)

    # Auto-detect model from config; CLI flag overrides if provided
    model_name = args.model or saved_model or "ViT-L-14"
    if args.model and saved_model and args.model != saved_model:
        print(f"[query] WARNING: --model {args.model} overrides index built with {saved_model}.")

    # Load checkpoint from config (same one used during embed)
    checkpoint = Path(saved_checkpoint) if saved_checkpoint else None

    clip_model, anime_head, preprocess, embed_dim = load_model(
        model_name, device, checkpoint
    )

    query_vec = embed_query(image_path, clip_model, anime_head, preprocess, device)
    results   = search(index, meta, query_vec, args.top_k)
    print_results(results, image_path, model_name, has_heads)


if __name__ == "__main__":
    main()