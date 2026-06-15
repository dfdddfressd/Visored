"""
embed.py — Visored Panel Embedding Pipeline
============================================
Reads bleach_panels/dataset.json, encodes every panel image through CLIP,
optionally passes through the manga projection head, and saves a FAISS index
+ parallel metadata file for fast nearest-neighbor search.

Run:
    python embed.py --model ViT-L-14 --out-dir index_finetuned3
    python embed.py --model ViT-L-14 --checkpoint clip_finetuned/best_checkpoint.pt --out-dir index_finetuned3

Two modes:
    No checkpoint  → raw CLIP embeddings (768-dim), zero-shot baseline
    With checkpoint → manga projection head applied (256-dim), fine-tuned retrieval

The index dimension changes between the two modes (768 vs 256), which is why
index_config.json records both the model and the checkpoint path — query.py
reads this to know which mode to use and what dim to expect.
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
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model registry — shared with query.py
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "ViT-B-32": {"pretrained": "openai", "dim": 512},
    "ViT-L-14": {"pretrained": "openai", "dim": 768},
}

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Projection head — must match finetune.py exactly
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head. Architecture must be identical to
    finetune.py's ProjectionHead — if you change one, change both.

    in_dim:  768 for ViT-L/14, 512 for ViT-B/32
    out_dim: 256 (PROJ_DIM from finetune.py)
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str, checkpoint: Path | None = None):
    """
    Load CLIP backbone and optionally the manga projection head.

    Returns:
        clip_model   — frozen CLIP backbone
        manga_head   — ProjectionHead if checkpoint provided, else None
        preprocess   — image preprocessor
        embed_dim    — output dimension (256 with head, 768 without)
    """
    if model_name not in MODEL_REGISTRY:
        sys.exit(f"[embed] ERROR: unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    cfg = MODEL_REGISTRY[model_name]
    clip_dim = cfg["dim"]

    print(f"[embed] Loading {model_name} ({cfg['pretrained']} weights) on {device}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=cfg["pretrained"]
    )
    clip_model.eval()
    clip_model.to(device)

    manga_head = None
    embed_dim  = clip_dim   # default: raw CLIP dim

    if checkpoint is not None:
        if not checkpoint.exists():
            sys.exit(f"[embed] ERROR: checkpoint not found at {checkpoint}")

        print(f"[embed] Loading projection head weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)

        # Detect checkpoint type:
        # Old format (full model state_dict) has key "state_dict"
        # New format (projection heads only) has keys "anime_head" and "manga_head"
        if "manga_head" in ckpt:
            proj_dim   = ckpt.get("proj_dim", 256)
            manga_head = ProjectionHead(clip_dim, proj_dim).to(device)
            manga_head.load_state_dict(ckpt["manga_head"])
            manga_head.eval()
            embed_dim = proj_dim
            print(f"[embed] Projection head loaded — epoch {ckpt['epoch']}, "
                  f"Recall@1: {ckpt['recall_at_1']:.2%}, embed_dim: {embed_dim}")
        elif "state_dict" in ckpt:
            # Legacy checkpoint — full model weights, no projection head
            print(f"[embed] Legacy checkpoint detected (full model weights) — "
                  f"loading into backbone. No projection head.")
            clip_model.load_state_dict(ckpt["state_dict"])
            clip_model.eval()
            print(f"[embed] Checkpoint: epoch {ckpt['epoch']}, "
                  f"Recall@1: {ckpt['recall_at_1']:.2%}")
        else:
            sys.exit(f"[embed] ERROR: unrecognized checkpoint format in {checkpoint}")

    return clip_model, manga_head, preprocess, embed_dim


def load_dataset(panels_dir: Path) -> list[dict]:
    dataset_path = panels_dir / "dataset.json"
    if not dataset_path.exists():
        sys.exit(f"[embed] ERROR: {dataset_path} not found. Run panel_splicer.py first.")
    with open(dataset_path) as f:
        data = json.load(f)
    panels = data["panels"]
    print(f"[embed] Found {len(panels)} panels across {data['total_chapters']} chapters.")
    return panels


def load_image(panels_dir: Path, entry: dict, preprocess) -> torch.Tensor | None:
    img_path = panels_dir / entry["folder"] / entry["file"]
    try:
        img = Image.open(img_path).convert("RGB")
        return preprocess(img)
    except Exception as e:
        print(f"[embed] WARNING: skipping {img_path} — {e}")
        return None


def encode_batch(
    clip_model,
    manga_head,
    batch_tensors: list[torch.Tensor],
    device: str,
) -> np.ndarray:
    """
    Encode a batch of panel images.
    If manga_head is provided, pass CLIP features through it (256-dim output).
    Otherwise return raw L2-normalized CLIP features (768-dim output).
    """
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(batch)
        if manga_head is not None:
            # Projection head handles its own L2 normalization internally
            embeddings = manga_head(features)
        else:
            # Raw CLIP — L2 normalize manually
            embeddings = features / features.norm(dim=-1, keepdim=True)
    return embeddings.cpu().numpy().astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, dim: int) -> faiss.Index:
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[embed] FAISS index built — {index.ntotal} vectors, dim={dim}.")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed Bleach manga panels with CLIP.")
    parser.add_argument("--panels-dir", default="bleach_panels")
    parser.add_argument("--out-dir",    default=".", help="Where to write index files")
    parser.add_argument("--model",      default="ViT-L-14", choices=list(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", default=None,
                        help="Path to projection head checkpoint .pt (from new finetune.py)")
    args = parser.parse_args()

    panels_dir = Path(args.panels_dir)
    out_dir    = Path(args.out_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    clip_model, manga_head, preprocess, embed_dim = load_model(
        args.model, device, checkpoint
    )
    panels = load_dataset(panels_dir)

    # Save config so query.py knows the exact setup that built this index
    config = {
        "model":      args.model,
        "dim":        embed_dim,
        "checkpoint": args.checkpoint,
        "has_heads":  manga_head is not None,
    }
    with open(out_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[embed] Saved index config → {out_dir}/index_config.json")

    # ── Encode all panels ────────────────────────────────────────────────────
    all_embeddings = []
    valid_panels   = []
    batch_tensors  = []
    batch_meta     = []

    print(f"[embed] Encoding {len(panels)} panels with {args.model} "
          f"({'+ manga head' if manga_head else 'raw CLIP'}, batch={BATCH_SIZE})...")

    for entry in tqdm(panels, unit="panel"):
        tensor = load_image(panels_dir, entry, preprocess)
        if tensor is None:
            continue
        batch_tensors.append(tensor)
        batch_meta.append(entry)

        if len(batch_tensors) == BATCH_SIZE:
            all_embeddings.append(encode_batch(clip_model, manga_head, batch_tensors, device))
            valid_panels.extend(batch_meta)
            batch_tensors = []
            batch_meta    = []

    if batch_tensors:
        all_embeddings.append(encode_batch(clip_model, manga_head, batch_tensors, device))
        valid_panels.extend(batch_meta)

    if not all_embeddings:
        sys.exit("[embed] ERROR: No panels encoded. Check your bleach_panels/ directory.")

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[embed] Encoded {len(valid_panels)} panels → shape {embeddings_matrix.shape}.")

    index = build_faiss_index(embeddings_matrix, embed_dim)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    print(f"[embed] Saved FAISS index → {out_dir}/index.faiss")

    with open(out_dir / "index_meta.json", "w") as f:
        json.dump(valid_panels, f, indent=2)
    print(f"[embed] Saved metadata  → {out_dir}/index_meta.json")

    print(f"\n[embed] Done. Query with: python query.py --image <screenshot.jpg> --index-dir {out_dir}")


if __name__ == "__main__":
    main()