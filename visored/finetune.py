"""
finetune.py — Fine-tune CLIP on anime/manga pairs for Visored
=============================================================
Reads labels.json, trains CLIP with InfoNCE contrastive loss,
saves fine-tuned weights to clip_finetuned/.

Run:
    python finetune.py
    python finetune.py --epochs 20 --batch-size 16 --lr 1e-5

Requirements:
    pip install open-clip-torch torch tqdm
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME   = "ViT-L-14"
PRETRAINED   = "openai"
LABELS_FILE  = Path("labels.json")
PANELS_DIR   = Path("bleach_panels")
SCREENSHOTS_DIR = Path("screenshots")
OUT_DIR      = Path("clip_finetuned")

# How many of the 24 transformer blocks to unfreeze for training.
# Unfreezing only the last N blocks preserves low-level visual knowledge
# while allowing high-level semantic representations to adapt.
# 4 is a good starting point for 130 pairs — more blocks = more capacity
# but also more risk of overfitting on a small dataset.
UNFREEZE_LAST_N_BLOCKS = 4


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnimeMangaDataset(Dataset):
    """
    Each item is a (anime_tensor, manga_tensor) positive pair.
    InfoNCE loss treats every other pair in the batch as a negative,
    so we don't need explicit negative examples — they come for free
    from the other items in each batch.
    """

    def __init__(self, pairs: list[dict], preprocess, manga_mode: bool = True):
        self.pairs      = pairs
        self.preprocess = preprocess
        self.manga_mode = manga_mode

    def __len__(self):
        return len(self.pairs)

    def _manga_preprocess(self, img: Image.Image) -> Image.Image:
        """
        Same preprocessing as query.py --manga-mode.
        Applied to anime screenshots to shift them toward manga embedding space.
        """
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return img.convert("RGB")

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        # ── Load anime screenshot ──────────────────────────────────────────
        # anime_screenshot is stored as a relative path e.g. "Chapter 1/frame.png"
        anime_path = SCREENSHOTS_DIR / pair["anime_screenshot"]
        anime_img  = Image.open(anime_path).convert("RGB")
        if self.manga_mode:
            anime_img = self._manga_preprocess(anime_img)
        anime_tensor = self.preprocess(anime_img)

        # ── Load manga panel ───────────────────────────────────────────────
        # manga_panel is stored as "Chapter 1/p020_panel03.jpg"
        manga_path   = PANELS_DIR / pair["manga_panel"]
        manga_img    = Image.open(manga_path).convert("RGB")
        manga_tensor = self.preprocess(manga_img)

        return anime_tensor, manga_tensor


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def infonce_loss(anime_emb: torch.Tensor, manga_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Symmetric InfoNCE (contrastive) loss.

    Given a batch of B anime embeddings and B manga embeddings:
    - For each anime[i], the positive is manga[i], negatives are manga[j≠i]
    - For each manga[i], the positive is anime[i], negatives are anime[j≠i]
    - Loss is the average of both directions (symmetric)

    The temperature parameter controls how sharply the model distinguishes
    between positives and negatives. Lower = sharper = harder training.
    0.07 is CLIP's original value.

    This works because every pair in the batch that ISN'T the matched pair
    is implicitly treated as a negative example. With batch size 16 you get
    16 positives and 16*15 = 240 negatives per batch — very efficient.
    """
    # L2 normalize so dot product = cosine similarity
    anime_emb = F.normalize(anime_emb, dim=-1)
    manga_emb = F.normalize(manga_emb, dim=-1)

    # Similarity matrix: (B, B) where [i,j] = similarity(anime_i, manga_j)
    logits = torch.matmul(anime_emb, manga_emb.T) / temperature

    # Labels: diagonal is the correct match (anime_i matches manga_i)
    labels = torch.arange(len(anime_emb), device=anime_emb.device)

    # Cross entropy in both directions, averaged
    loss_anime = F.cross_entropy(logits, labels)        # each anime → correct manga
    loss_manga = F.cross_entropy(logits.T, labels)      # each manga → correct anime
    return (loss_anime + loss_manga) / 2


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_model(model):
    """Freeze all parameters — starting point before selective unfreezing."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n: int):
    """
    Unfreeze the last N transformer blocks of the visual encoder.

    ViT-L/14 has 24 transformer blocks (model.visual.transformer.resblocks).
    We also unfreeze the final layer norm (ln_post) and projection head
    since those directly produce the embedding vector we're training on.

    Everything before block (24-n) stays frozen — those early layers encode
    low-level features (edges, textures) that are already style-invariant
    and don't need to change for anime→manga adaptation.
    """
    blocks = model.visual.transformer.resblocks
    total  = len(blocks)
    print(f"[finetune] ViT has {total} transformer blocks — unfreezing last {n}")

    for i, block in enumerate(blocks):
        if i >= total - n:
            for param in block.parameters():
                param.requires_grad = True

    # Always unfreeze the output projection and final norm
    for param in model.visual.ln_post.parameters():
        param.requires_grad = True
    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        model.visual.proj.requires_grad = True
        
    # Override — keep projection frozen even though it's after ln_post
    # The proj matrix is sensitive and destabilizes easily on small datasets
    if hasattr(model.visual, 'proj') and model.visual.proj is not None:
        model.visual.proj.requires_grad = False
        print("[finetune] Projection layer kept frozen for stability")


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[finetune] Trainable params: {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.1f}%)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device: str) -> tuple[float, float]:
    """
    Compute validation loss and Recall@1.

    Recall@1: for each anime embedding in the validation set, is the
    correct manga panel its nearest neighbor? This is the metric that
    directly measures what Visored does — so it's our north star.
    """
    model.eval()
    total_loss = 0.0
    all_anime  = []
    all_manga  = []

    for anime_batch, manga_batch in val_loader:
        anime_emb = model.encode_image(anime_batch.to(device))
        manga_emb = model.encode_image(manga_batch.to(device))
        loss = infonce_loss(anime_emb, manga_emb)
        total_loss += loss.item()
        all_anime.append(F.normalize(anime_emb, dim=-1).cpu())
        all_manga.append(F.normalize(manga_emb, dim=-1).cpu())

    # Recall@1 across the full validation set
    all_anime = torch.cat(all_anime)   # (N_val, dim)
    all_manga = torch.cat(all_manga)   # (N_val, dim)
    sim_matrix = torch.matmul(all_anime, all_manga.T)  # (N_val, N_val)
    top1_preds = sim_matrix.argmax(dim=1)              # each anime's best manga match
    correct    = (top1_preds == torch.arange(len(all_anime))).float().mean().item()

    model.train()
    return total_loss / len(val_loader), correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on anime/manga pairs.")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=16,
                        help="Keep ≤16 on CPU; batch size determines negative count")
    parser.add_argument("--lr",         type=float, default=1e-5,
                        help="Learning rate — keep small (1e-5 to 5e-6) to avoid forgetting")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--no-manga-mode", action="store_true",
                        help="Disable grayscale preprocessing on anime screenshots")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[finetune] Device: {device}")

    # ── Load pairs ──────────────────────────────────────────────────────────
    if not LABELS_FILE.exists():
        sys.exit(f"[finetune] {LABELS_FILE} not found — run the labeler first.")
    with open(LABELS_FILE) as f:
        all_pairs = json.load(f)
    print(f"[finetune] Loaded {len(all_pairs)} labeled pairs.")

    # Filter out pairs where either file is missing
    valid_pairs = []
    for p in all_pairs:
        a = SCREENSHOTS_DIR / p["anime_screenshot"]
        m = PANELS_DIR / p["manga_panel"]
        if a.exists() and m.exists():
            valid_pairs.append(p)
        else:
            print(f"[finetune] WARNING: skipping missing files — {p['anime_screenshot']}")
    print(f"[finetune] {len(valid_pairs)} pairs with files present.")

    if len(valid_pairs) < 8:
        sys.exit("[finetune] Need at least 8 valid pairs to train.")

    # 80/20 split — shuffle first for random split
    random.shuffle(valid_pairs)
    split      = int(len(valid_pairs) * 0.8)
    train_pairs = valid_pairs[:split]
    val_pairs   = valid_pairs[split:]
    print(f"[finetune] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[finetune] Loading {MODEL_NAME} ({PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.to(device)

    # Freeze everything, then selectively unfreeze last N blocks
    freeze_model(model)
    unfreeze_last_n_blocks(model, UNFREEZE_LAST_N_BLOCKS)

    # ── Datasets and loaders ─────────────────────────────────────────────────
    manga_mode = not args.no_manga_mode
    train_ds = AnimeMangaDataset(train_pairs, preprocess, manga_mode=manga_mode)
    val_ds   = AnimeMangaDataset(val_pairs,   preprocess, manga_mode=manga_mode)

    # drop_last=True on train so every batch is full (important for InfoNCE —
    # a partial last batch with 1-2 items has almost no negatives)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Optimizer ───────────────────────────────────────────────────────────
    # Only pass trainable parameters to the optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    # Cosine LR schedule — gradually reduces LR to near-zero by final epoch.
    # Prevents the model from making large updates late in training when it's
    # already close to a good solution.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0
    best_epoch  = 0

    print(f"\n[finetune] Starting training — {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}, manga_mode={manga_mode}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches    = 0

        for anime_batch, manga_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            anime_batch = anime_batch.to(device)
            manga_batch = manga_batch.to(device)

            # Forward pass — encode both modalities through the SAME visual encoder.
            # We use the image encoder for both since we're doing image→image matching,
            # not image→text. Both anime screenshots and manga panels are images.
            anime_emb = model.encode_image(anime_batch)
            manga_emb = model.encode_image(manga_batch)

            loss = infonce_loss(anime_emb, manga_emb)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — prevents rare large gradient updates from
            # destabilizing the partially-frozen model
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(batches, 1)

        # Validation
        val_loss, recall_at_1 = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} — "
              f"train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"Recall@1: {recall_at_1:.2%}")

        # Save best checkpoint by Recall@1
        if recall_at_1 >= best_recall:
            best_recall = recall_at_1
            best_epoch  = epoch
            ckpt_path   = OUT_DIR / "best_checkpoint.pt"
            torch.save({
                "epoch":       epoch,
                "model_name":  MODEL_NAME,
                "pretrained":  PRETRAINED,
                "recall_at_1": recall_at_1,
                "val_loss":    val_loss,
                "state_dict":  model.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (Recall@1: {recall_at_1:.2%})")

    print(f"\n[finetune] Done. Best Recall@1: {best_recall:.2%} at epoch {best_epoch}")
    print(f"[finetune] Checkpoint saved to {OUT_DIR}/best_checkpoint.pt")
    print(f"\nNext: python embed.py --model ViT-L-14 --checkpoint {OUT_DIR}/best_checkpoint.pt --out-dir index_finetuned")


if __name__ == "__main__":
    main()
