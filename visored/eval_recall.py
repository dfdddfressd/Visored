"""
eval_recall_full.py — Evaluate Recall@K against the FULL FAISS index
=======================================================================
The previous eval_recall.py only searched within the validation subset
(139 panels), which dramatically inflates Recall@K because finding the
right answer among 139 candidates is much easier than among the full
5,123-panel index. This script searches against the real index, which
matches what query.py and the labeler actually experience.
 
Run:
    python eval_recall_full.py --index-dir index_finetuned3
"""
 
import argparse
import json
import random
import sys
from pathlib import Path
 
import faiss
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
 
LABELS_FILE     = Path("labels.json")
SCREENSHOTS_DIR = Path("screenshots")
PANELS_DIR      = Path("bleach_panels")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",  required=True, help="Directory with index.faiss + index_meta.json")
    parser.add_argument("--checkpoint", default="clip_finetuned/best_checkpoint.pt")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
 
    random.seed(args.seed)
 
    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "index_meta.json") as f:
        meta = json.load(f)
    print(f"[eval] Loaded index — {index.ntotal} panels")
 
    # Build a lookup so we can find the index position of a given (folder, file)
    panel_lookup = {f"{p['folder']}/{p['file']}": i for i, p in enumerate(meta)}
 
    # ── Load val pairs using the SAME panel-aware split as finetune.py ──────
    pairs = json.load(open(LABELS_FILE))
    valid_pairs = [p for p in pairs
                   if (SCREENSHOTS_DIR / p['anime_screenshot']).exists()
                   and (PANELS_DIR / p['manga_panel']).exists()]
 
    unique_panels = list({p['manga_panel'] for p in valid_pairs})
    random.shuffle(unique_panels)
    split = int(len(unique_panels) * 0.8)
    val_panel_set = set(unique_panels[split:])
    val_pairs = [p for p in valid_pairs if p['manga_panel'] in val_panel_set]
    print(f"[eval] Evaluating on {len(val_pairs)} val pairs against FULL {index.ntotal}-panel index")
 
    # ── Load model + checkpoint ──────────────────────────────────────────────
    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"[eval] Loaded checkpoint — epoch {ckpt['epoch']}, "
          f"reported val Recall@1: {ckpt['recall_at_1']:.2%}")
 
    # ── Embed each val anime screenshot and search the FULL index ───────────
    ranks = []  # rank (1-indexed) of correct panel for each query, or None if not found in top-K searched
    SEARCH_K = 50  # search deep enough to compute Recall@10 reliably, and see how far off we are otherwise
 
    skipped = 0
    with torch.no_grad():
        for p in tqdm(val_pairs, desc="Querying"):
            target_key = p['manga_panel']  # e.g. "Chapter 7/p006_panel05.jpg"
            target_idx = panel_lookup.get(target_key)
            if target_idx is None:
                skipped += 1
                continue  # this panel isn't in the index for some reason
 
            img = Image.open(SCREENSHOTS_DIR / p['anime_screenshot']).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            feat = model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1).cpu().numpy().astype(np.float32)
 
            scores, indices = index.search(feat, SEARCH_K)
            indices = indices[0]
 
            if target_idx in indices:
                rank = int(np.where(indices == target_idx)[0][0]) + 1
            else:
                rank = None  # not found even in top 50
            ranks.append(rank)
 
    if skipped:
        print(f"[eval] WARNING: {skipped} val pairs skipped — panel not found in index_meta.json")
 
    n = len(ranks)
    for k in [1, 5, 10, 20, 50]:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"Recall@{k}: {hits}/{n} = {hits/n:.2%}")
 
    not_found = sum(1 for r in ranks if r is None)
    print(f"\nNot found in top {SEARCH_K} at all: {not_found}/{n} = {not_found/n:.2%}")
 
 
if __name__ == "__main__":
    main()
 















