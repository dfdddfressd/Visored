"""
eval_recall_full_dino.py — Evaluate DINOv2 Recall@K against the FULL FAISS index
==================================================================================
Mirrors eval_recall_full.py exactly but uses DINOv2 instead of CLIP.
Searches against the full 42,112-panel index, not just the val subset,
to get real-world retrieval numbers.

Run:
    python eval_recall_full_dino.py --index-dir index_dino_full
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

# Failure tier thresholds
NEAR_MISS_MAX   = 5    # rank 2-5
SOFT_FAIL_MAX   = 25   # rank 6-25
# rank > 25 or not found = hard failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir",  required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint — auto-detected from index_config.json if omitted")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--audit",      action="store_true",
                        help="After eval, dump hard failures as a clean audit list with absolute paths")
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

    # Per-pair results for failure analysis
    results  = []

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
            scores  = scores[0]
            indices = indices[0]

            # Rank of correct panel
            if target_idx in indices:
                rank = int(np.where(indices == target_idx)[0][0]) + 1
                correct_score = float(scores[rank - 1])
            else:
                rank = None
                correct_score = None

            # Top-1 retrieved panel
            top1_idx   = int(indices[0])
            top1_score = float(scores[0])
            top1_meta  = meta[top1_idx]
            top1_key   = f"{top1_meta['folder']}/{top1_meta['file']}"

            ranks.append(rank)
            results.append({
                "anime_screenshot": p["anime_screenshot"],
                "correct_panel":    target_key,
                "rank":             rank,
                "correct_score":    correct_score,
                "top1_panel":       top1_key,
                "top1_score":       top1_score,
                "is_correct":       rank == 1,
            })

    if skipped:
        print(f"[eval_dino] WARNING: {skipped} val pairs skipped — panel not in index")

    # ── Aggregate Recall@K ───────────────────────────────────────────────────
    n = len(ranks)
    print()
    for k in [1, 5, 10, 20, 50]:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"Recall@{k:>2}: {hits:>3}/{n} = {hits/n:.2%}")

    not_found = sum(1 for r in ranks if r is None)
    print(f"\nNot in top {SEARCH_K}: {not_found}/{n} = {not_found/n:.2%}")

    found_ranks = [r for r in ranks if r is not None]
    if found_ranks:
        median_rank = sorted(found_ranks)[len(found_ranks) // 2]
        print(f"Median rank (when found): {median_rank}")

    # ── Failure breakdown ────────────────────────────────────────────────────
    misses = [r for r in results if not r["is_correct"]]

    near_misses   = [r for r in misses if r["rank"] is not None and r["rank"] <= NEAR_MISS_MAX]
    soft_failures = [r for r in misses if r["rank"] is not None and NEAR_MISS_MAX < r["rank"] <= SOFT_FAIL_MAX]
    hard_failures = [r for r in misses if r["rank"] is None or r["rank"] > SOFT_FAIL_MAX]

    def print_failure_table(title, rows):
        if not rows:
            print(f"\n  (none)")
            return
        # Sort by rank ascending (None = worst, goes last)
        rows = sorted(rows, key=lambda r: r["rank"] if r["rank"] is not None else 9999)
        col_w = {
            "rank":    6,
            "screenshot": 32,
            "correct": 34,
            "top1":    34,
            "top1_sc": 10,
            "corr_sc": 10,
        }
        header = (
            f"{'Rank':<{col_w['rank']}} "
            f"{'Anime Screenshot':<{col_w['screenshot']}} "
            f"{'Correct Panel':<{col_w['correct']}} "
            f"{'Top-1 Retrieved':<{col_w['top1']}} "
            f"{'Top-1 Sc':>{col_w['top1_sc']}} "
            f"{'Corr Sc':>{col_w['corr_sc']}}"
        )
        sep = "-" * len(header)
        print(f"\n{sep}")
        print(title)
        print(sep)
        print(header)
        print(sep)
        for r in rows:
            rank_str    = str(r["rank"]) if r["rank"] is not None else "50+"
            screenshot  = r["anime_screenshot"][-col_w['screenshot']:]
            correct     = r["correct_panel"][-col_w['correct']:]
            top1        = r["top1_panel"][-col_w['top1']:]
            top1_sc     = f"{r['top1_score']:.4f}"
            corr_sc     = f"{r['correct_score']:.4f}" if r["correct_score"] is not None else "  —   "
            print(
                f"{rank_str:<{col_w['rank']}} "
                f"{screenshot:<{col_w['screenshot']}} "
                f"{correct:<{col_w['correct']}} "
                f"{top1:<{col_w['top1']}} "
                f"{top1_sc:>{col_w['top1_sc']}} "
                f"{corr_sc:>{col_w['corr_sc']}}"
            )
        print(sep)

    print(f"\n\n{'='*40}")
    print(f"FAILURE BREAKDOWN  ({len(misses)} total misses)")
    print(f"{'='*40}")

    print(f"\n▸ NEAR-MISSES  (rank 2–{NEAR_MISS_MAX})  — {len(near_misses)} pairs")
    print_failure_table("Near-misses: correct panel was close but edged out", near_misses)

    print(f"\n▸ SOFT FAILURES  (rank {NEAR_MISS_MAX+1}–{SOFT_FAIL_MAX})  — {len(soft_failures)} pairs")
    print_failure_table("Soft failures: model found it but ranked it poorly", soft_failures)

    print(f"\n▸ HARD FAILURES  (rank >{SOFT_FAIL_MAX} or not found)  — {len(hard_failures)} pairs")
    print_failure_table("Hard failures: complete misses", hard_failures)

    # ── Audit mode — clean dump of hard failures for manual review ───────────
    if args.audit and hard_failures:
        print(f"\n\n{'='*40}")
        print(f"AUDIT LIST  ({len(hard_failures)} hard failures)")
        print(f"Inspect each pair and decide: correct label, mislabel, or re-label needed.")
        print(f"{'='*40}\n")
        for i, r in enumerate(hard_failures, 1):
            screenshot_path = (SCREENSHOTS_DIR / r["anime_screenshot"]).resolve()
            panel_path      = (PANELS_DIR      / r["correct_panel"]).resolve()
            print(f"[{i:02d}] Screenshot : {screenshot_path}")
            print(f"      Panel      : {panel_path}")
            print(f"      Top-1 got  : {r['top1_panel']}  (score {r['top1_score']:.4f})")
            print()


if __name__ == "__main__":
    main()