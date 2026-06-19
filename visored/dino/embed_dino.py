"""
embed_dino.py — Visored Panel Embedding Pipeline (DINOv2 variant)
====================================================================
Mirrors clip/embed.py exactly in structure and CLI, but uses DINOv2
instead of CLIP. Reads bleach_panels/dataset.json, encodes every panel
through DINOv2, saves a FAISS index + metadata.

Run:
    python embed_dino.py --out-dir index_dino
    python embed_dino.py --checkpoint dino_finetuned/best_checkpoint.pt --out-dir index_dino
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
from tqdm import tqdm

MODEL_NAME = "facebook/dinov2-base"
DINO_DIM   = 768
BATCH_SIZE = 32


def load_model(device: str, checkpoint: Path | None = None):
    print(f"[embed_dino] Loading {MODEL_NAME} on {device}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    if checkpoint is not None:
        if not checkpoint.exists():
            sys.exit(f"[embed_dino] ERROR: checkpoint not found at {checkpoint}")
        print(f"[embed_dino] Loading fine-tuned weights from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[embed_dino] Checkpoint: epoch {ckpt['epoch']}, "
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


def load_dataset(panels_dir: Path) -> list[dict]:
    dataset_path = panels_dir / "dataset.json"
    if not dataset_path.exists():
        sys.exit(f"[embed_dino] ERROR: {dataset_path} not found. Run panel_splicer.py first.")
    with open(dataset_path) as f:
        data = json.load(f)
    panels = data["panels"]
    print(f"[embed_dino] Found {len(panels)} panels across {data['total_chapters']} chapters.")
    return panels


def load_image(panels_dir: Path, entry: dict, transform) -> torch.Tensor | None:
    img_path = panels_dir / entry["folder"] / entry["file"]
    try:
        img = Image.open(img_path).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"[embed_dino] WARNING: skipping {img_path} — {e}")
        return None


def encode_batch(model, batch_tensors: list[torch.Tensor], device: str) -> np.ndarray:
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=batch)
        # CLS token = global image embedding, same choice as finetune_dino.py
        features = outputs.last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def build_faiss_index(embeddings: np.ndarray, dim: int) -> faiss.Index:
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[embed_dino] FAISS index built — {index.ntotal} vectors, dim={dim}.")
    return index


def main():
    parser = argparse.ArgumentParser(description="Embed Bleach manga panels with DINOv2.")
    parser.add_argument("--panels-dir", default="../bleach_panels")
    parser.add_argument("--out-dir",    default=".")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned DINOv2 checkpoint .pt file (optional)")
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

    model, transform = load_model(device, checkpoint)
    panels = load_dataset(panels_dir)

    config = {"model": MODEL_NAME, "dim": DINO_DIM, "checkpoint": args.checkpoint}
    with open(out_dir / "index_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[embed_dino] Saved index config → {out_dir}/index_config.json")

    all_embeddings = []
    valid_panels   = []
    batch_tensors  = []
    batch_meta     = []

    print(f"[embed_dino] Encoding {len(panels)} panels with {MODEL_NAME} (batch={BATCH_SIZE})...")

    for entry in tqdm(panels, unit="panel"):
        tensor = load_image(panels_dir, entry, transform)
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
        sys.exit("[embed_dino] ERROR: No panels encoded.")

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[embed_dino] Encoded {len(valid_panels)} panels → shape {embeddings_matrix.shape}.")

    index = build_faiss_index(embeddings_matrix, DINO_DIM)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    print(f"[embed_dino] Saved FAISS index → {out_dir}/index.faiss")

    with open(out_dir / "index_meta.json", "w") as f:
        json.dump(valid_panels, f, indent=2)
    print(f"[embed_dino] Saved metadata  → {out_dir}/index_meta.json")

    print(f"\n[embed_dino] Done. Query with: python query_dino.py --image <screenshot.jpg> --index-dir {out_dir}")


if __name__ == "__main__":
    main()