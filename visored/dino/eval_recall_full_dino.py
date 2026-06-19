"""
eval_recall_full_dino.py — Evaluate DINOv2 Recall@K against the FULL FAISS index
==================================================================================
Mirrors eval_recall_full.py exactly but uses DINOv2 instead of CLIP.
Searches against the full 5,123-panel index, not just the val subset,
to get real-world retrieval numbers.

Run:
    python eval_recall_full_dino.py --index-dir index_dino
    python eval_recall_full_dino.py --index-dir index_dino_zeroshot
"""

import argparse
import json
import random
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

MODEL_NAME      = "facebook/dinov2-base"
LABELS_FILE     = Path("../labels.json")
SCREENSHOTS_DIR = Path("../screenshots")
PANELS_DIR      = Path("../bleach_panels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",  required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint — auto-detected from index_config.json if omitted")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "index_meta.json") as f:
        meta = json.load(f)
    print(f"[eval_dino] Loaded index — {index.ntotal} panels")

    # Panel lookup: "folder/file" → integer index position in FAISS
    panel_lookup = {f"{p['folder']}/{p['file']}": i for i, p in enumerate(meta)}

    # Auto-detect checkpoint from index config if not explicitly passed
    checkpoint = args.checkpoint
    if checkpoint is None:
        config_path = index_dir / "index_config.json"
        if config_path.exists():
            with open(config_path) as f:
                checkpoint = json.load(f).get("checkpoint")

    # ── Panel-aware val split — same seed/logic as finetune_dino.py ─────────
    pairs = json.load(open(LABELS_FILE))
    valid_pairs = [
        p for p in pairs
        if (SCREENSHOTS_DIR / p["anime_screenshot"]).exists()
        and (PANELS_DIR / p["manga_panel"]).exists()
    ]

    unique_panels = list({p["manga_panel"] for p in valid_pairs})
    random.shuffle(unique_panels)
    split = int(len(unique_panels) * 0.8)
    val_panel_set = set(unique_panels[split:])
    val_pairs = [p for p in valid_pairs if p["manga_panel"] in val_panel_set]
    print(f"[eval_dino] Evaluating on {len(val_pairs)} val pairs "
          f"against FULL {index.ntotal}-panel index")

    # ── Load DINOv2 ──────────────────────────────────────────────────────────
    device = "cpu"
    print(f"[eval_dino] Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"[eval_dino] Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[eval_dino] Loaded checkpoint — epoch {ckpt['epoch']}, "
              f"reported val Recall@1: {ckpt['recall_at_1']:.2%}")
    else:
        print(f"[eval_dino] No checkpoint — evaluating zero-shot DINOv2")

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

    # ── Query each val anime screenshot against the full index ───────────────
    SEARCH_K = 50
    ranks    = []
    skipped  = 0

    with torch.no_grad():
        for p in tqdm(val_pairs, desc="Querying"):
            target_key = p["manga_panel"]
            target_idx = panel_lookup.get(target_key)
            if target_idx is None:
                skipped += 1
                continue

            img    = Image.open(SCREENSHOTS_DIR / p["anime_screenshot"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            outputs = model(pixel_values=tensor)
            feat   = outputs.last_hidden_state[:, 0, :]          # CLS token
            feat   = F.normalize(feat, dim=-1).cpu().numpy().astype(np.float32)

            scores, indices = index.search(feat, SEARCH_K)
            indices = indices[0]

            if target_idx in indices:
                rank = int(np.where(indices == target_idx)[0][0]) + 1
            else:
                rank = None
            ranks.append(rank)

    if skipped:
        print(f"[eval_dino] WARNING: {skipped} val pairs skipped — panel not in index")

    n = len(ranks)
    print()
    for k in [1, 5, 10, 20, 50]:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"Recall@{k:>2}: {hits:>3}/{n} = {hits/n:.2%}")

    not_found = sum(1 for r in ranks if r is None)
    print(f"\nNot in top {SEARCH_K}: {not_found}/{n} = {not_found/n:.2%}")

    # Median rank for found items — useful secondary metric
    found_ranks = [r for r in ranks if r is not None]
    if found_ranks:
        median_rank = sorted(found_ranks)[len(found_ranks) // 2]
        print(f"Median rank (when found): {median_rank}")


if __name__ == "__main__":
    main()