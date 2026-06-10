"""
embed.py — Visored Panel Embedding Pipeline
============================================
Reads bleach_panels/dataset.json, encodes every panel image through CLIP,
and saves a FAISS index + parallel metadata file for fast nearest-neighbor search.
 
Run:
    python embed.py
    python embed.py --model ViT-L-14 --out-dir index_L14
    python embed.py --model ViT-L-14 --checkpoint clip_finetuned/best_checkpoint.pt --out-dir index_finetuned
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
 
 
# ---------------------------------------------------------------------------
# Model registry
# Centralised here so embed.py and query.py always agree on dim + weights.
# query.py imports this dict — never hardcode these values in two places.
# ---------------------------------------------------------------------------
 
MODEL_REGISTRY = {
    "ViT-B-32": {"pretrained": "openai", "dim": 512},
    "ViT-L-14": {"pretrained": "openai", "dim": 768},
}
 
BATCH_SIZE = 32   # Lower than before — L14 is heavier; 32 is safe for both models
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def load_model(model_name: str, device: str, checkpoint: Path | None = None):
    """
    Load the CLIP model and its image preprocessor.
    If checkpoint is provided, load fine-tuned weights on top of base model.
    """
    if model_name not in MODEL_REGISTRY:
        sys.exit(f"[embed] ERROR: unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")
 
    cfg = MODEL_REGISTRY[model_name]
    print(f"[embed] Loading {model_name} ({cfg['pretrained']} weights) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=cfg["pretrained"]
    )
 
    if checkpoint is not None:
        if not checkpoint.exists():
            sys.exit(f"[embed] ERROR: checkpoint not found at {checkpoint}")
        print(f"[embed] Loading fine-tuned weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[embed] Checkpoint: epoch {ckpt['epoch']}, "
              f"Recall@1: {ckpt['recall_at_1']:.2%}")
 
    model.eval()
    model.to(device)
    return model, preprocess, cfg["dim"]
 
 
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
 
 
def encode_batch(model, batch_tensors: list[torch.Tensor], device: str) -> np.ndarray:
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)   # L2 normalize
    return features.cpu().numpy().astype(np.float32)
 
 
def build_faiss_index(embeddings: np.ndarray, dim: int) -> faiss.Index:
    """
    IndexFlatIP = exact inner-product search.
    With L2-normalized vectors, inner product == cosine similarity.
    dim is passed explicitly because it varies by model (512 vs 768).
    """
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[embed] FAISS index built — {index.ntotal} vectors, dim={dim}.")
    return index
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Embed Bleach manga panels with CLIP.")
    parser.add_argument("--panels-dir", default="bleach_panels",
                        help="Path to panel_splicer output directory (default: bleach_panels)")
    parser.add_argument("--out-dir", default=".",
                        help="Where to write index.faiss and index_meta.json (default: .)")
    parser.add_argument("--model", default="ViT-B-32", choices=list(MODEL_REGISTRY),
                        help="CLIP model variant (default: ViT-B-32)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint .pt file (optional)")
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
 
    model, preprocess, dim = load_model(args.model, device, checkpoint)
    panels = load_dataset(panels_dir)
 
    # Save config — include checkpoint path so you know which index used fine-tuned weights
    config = {"model": args.model, "dim": dim, "checkpoint": args.checkpoint}
    with open(out_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[embed] Saved index config → {out_dir}/index_config.json")
 
    # -----------------------------------------------------------------------
    # Encode all panels in batches
    # -----------------------------------------------------------------------
    all_embeddings = []
    valid_panels   = []
    batch_tensors  = []
    batch_meta     = []
 
    print(f"[embed] Encoding {len(panels)} panels with {args.model} (batch={BATCH_SIZE})...")
 
    for entry in tqdm(panels, unit="panel"):
        tensor = load_image(panels_dir, entry, preprocess)
        if tensor is None:
            continue
 
        batch_tensors.append(tensor)
        batch_meta.append(entry)
 
        if len(batch_tensors) == BATCH_SIZE:
            all_embeddings.append(encode_batch(model, batch_tensors, device))
            valid_panels.extend(batch_meta)
            batch_tensors = []
            batch_meta    = []
 
    if batch_tensors:
        all_embeddings.append(encode_batch(model, batch_tensors, device))
        valid_panels.extend(batch_meta)
 
    if not all_embeddings:
        sys.exit("[embed] ERROR: No panels encoded. Check your bleach_panels/ directory.")
 
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[embed] Encoded {len(valid_panels)} panels → shape {embeddings_matrix.shape}.")
 
    index = build_faiss_index(embeddings_matrix, dim)
 
    faiss.write_index(index, str(out_dir / "index.faiss"))
    print(f"[embed] Saved FAISS index → {out_dir}/index.faiss")
 
    with open(out_dir / "index_meta.json", "w") as f:
        json.dump(valid_panels, f, indent=2)
    print(f"[embed] Saved metadata  → {out_dir}/index_meta.json")
 
    print(f"\n[embed] Done. Query with: python query.py --image <screenshot.jpg> --index-dir {out_dir}")
 
 
if __name__ == "__main__":
    main()