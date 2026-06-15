"""
finetune.py — Fine-tune CLIP on anime/manga pairs for Visored
=============================================================
Architecture: frozen ViT-L/14 backbone + two lightweight MLP projection
heads (one for anime, one for manga). InfoNCE contrastive loss aligns
the two projection spaces for matched pairs.

Why this over unfreezing transformer blocks:
- ViT-L/14 has 427M params; unfreezing 4 blocks = 51M trainable on ~320 pairs → overfitting
- Two small MLPs = ~2M trainable params → appropriate for dataset size
- Checkpoint is ~50MB instead of 1.6GB (only MLP weights saved, not full ViT)
- Frozen backbone preserves CLIP's general visual knowledge entirely

Run:
    python finetune.py --no-manga-mode --epochs 40 --lr 1e-4
    python finetune.py --no-manga-mode --epochs 40 --lr 1e-4 --checkpoint clip_finetuned/best_checkpoint.pt

Requirements:
    pip install open-clip-torch torch tqdm
"""

import argparse
import json
import random
import sys
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME      = "ViT-L-14"
PRETRAINED      = "openai"
LABELS_FILE     = Path("labels.json")
PANELS_DIR      = Path("bleach_panels")
SCREENSHOTS_DIR = Path("screenshots")
OUT_DIR         = Path("clip_finetuned")

CLIP_DIM    = 768   # ViT-L/14 output dimension
PROJ_DIM    = 256   # projection head output dimension
            # Smaller than CLIP_DIM — forces the head to learn a compact,
            # discriminative representation rather than just passing through


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Two-layer MLP that maps CLIP embeddings to a smaller shared space.

    Architecture: Linear → GELU → LayerNorm → Linear → L2-normalize

    - GELU is smoother than ReLU and standard in transformer-adjacent work
    - LayerNorm between layers stabilizes training (acts like batch norm
      but works on single samples, important for small batches)
    - Final L2 normalization puts outputs on the unit sphere so cosine
      similarity == dot product, same as the main CLIP embedding space

    One head for anime, one for manga. They start with the same random
    weights but diverge as InfoNCE pushes anime embeddings toward their
    matched manga embeddings and away from non-matches.
    """
    def __init__(self, in_dim: int = CLIP_DIM, out_dim: int = PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),      # hidden layer same size as input
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),     # project down to PROJ_DIM
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)       # unit sphere


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnimeMangaDataset(Dataset):
    """
    Each item is a (anime_tensor, manga_tensor) positive pair.
    InfoNCE loss treats every other pair in the batch as a negative.
    """

    def __init__(self, pairs: list[dict], preprocess):
        self.pairs      = pairs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        anime_path   = SCREENSHOTS_DIR / pair["anime_screenshot"]
        anime_img    = Image.open(anime_path).convert("RGB")
        anime_tensor = self.preprocess(anime_img)

        manga_path   = PANELS_DIR / pair["manga_panel"]
        manga_img    = Image.open(manga_path).convert("RGB")
        manga_tensor = self.preprocess(manga_img)

        return anime_tensor, manga_tensor


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def infonce_loss(
    anime_emb: torch.Tensor,
    manga_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss. Inputs are already L2-normalized by the
    projection heads, so dot product == cosine similarity.

    temperature=0.07 is CLIP's original value. With projection heads
    outputting to a smaller 256-dim space we could experiment with
    slightly higher temps (0.1) if training is unstable, but start here.
    """
    # Already normalized by ProjectionHead.forward() — no need to re-normalize
    logits = torch.matmul(anime_emb, manga_emb.T) / temperature
    labels = torch.arange(len(anime_emb), device=anime_emb.device)
    loss_a = F.cross_entropy(logits,   labels)
    loss_m = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_m) / 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    clip_model,
    anime_head: ProjectionHead,
    manga_head: ProjectionHead,
    val_loader,
    device: str,
) -> tuple[float, float]:
    """
    Compute val loss and Recall@1.

    Recall@1: for each anime embedding in val, is the correct manga panel
    its nearest neighbor in the projected 256-dim space?

    Note: validation searches only within the val set (81 panels), not the
    full 5,123-panel FAISS index. This is intentional — it measures whether
    the model can correctly rank matched pairs, which is what matters for
    training signal. Real-world retrieval over the full index is measured
    by running query.py after embedding.
    """
    clip_model.eval()
    anime_head.eval()
    manga_head.eval()

    total_loss = 0.0
    all_anime  = []
    all_manga  = []

    for anime_batch, manga_batch in val_loader:
        anime_feat = clip_model.encode_image(anime_batch.to(device))
        manga_feat = clip_model.encode_image(manga_batch.to(device))

        anime_emb  = anime_head(anime_feat)
        manga_emb  = manga_head(manga_feat)

        loss = infonce_loss(anime_emb, manga_emb)
        total_loss += loss.item()

        all_anime.append(anime_emb.cpu())
        all_manga.append(manga_emb.cpu())

    all_anime  = torch.cat(all_anime)
    all_manga  = torch.cat(all_manga)
    sim_matrix = torch.matmul(all_anime, all_manga.T)
    top1_preds = sim_matrix.argmax(dim=1)
    correct    = (top1_preds == torch.arange(len(all_anime))).float().mean().item()

    clip_model.train()
    anime_head.train()
    manga_head.train()

    return total_loss / len(val_loader), correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP projection heads on anime/manga pairs.")
    parser.add_argument("--epochs",     type=int,   default=40,
                        help="More epochs are fine — only 2M params, overfitting risk is low")
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Higher LR than before — MLPs train faster than transformer blocks")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Resume from a previous projection head checkpoint")
    parser.add_argument("--no-manga-mode", action="store_true",
                        help="Disable grayscale preprocessing (always use this for colored manga)")
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

    valid_pairs = []
    for p in all_pairs:
        a = SCREENSHOTS_DIR / p["anime_screenshot"]
        m = PANELS_DIR      / p["manga_panel"]
        if a.exists() and m.exists():
            valid_pairs.append(p)
        else:
            print(f"[finetune] WARNING: missing files — {p['anime_screenshot']}")
    print(f"[finetune] {len(valid_pairs)} pairs with files present.")

    if len(valid_pairs) < 8:
        sys.exit("[finetune] Need at least 8 valid pairs to train.")

    # Panel-aware split: group by manga panel so the same panel never
    # appears in both train and val (eliminates data leakage)
    unique_panels = list({p["manga_panel"] for p in valid_pairs})
    random.shuffle(unique_panels)
    split_idx     = int(len(unique_panels) * 0.8)
    train_panel_set = set(unique_panels[:split_idx])
    val_panel_set   = set(unique_panels[split_idx:])

    train_pairs = [p for p in valid_pairs if p["manga_panel"] in train_panel_set]
    val_pairs   = [p for p in valid_pairs if p["manga_panel"] in val_panel_set]
    print(f"[finetune] Train: {len(train_pairs)} pairs ({len(train_panel_set)} unique panels) | "
          f"Val: {len(val_pairs)} pairs ({len(val_panel_set)} unique panels)")

    # ── Load CLIP backbone (frozen) ──────────────────────────────────────────
    print(f"[finetune] Loading {MODEL_NAME} ({PRETRAINED}) — backbone will be fully frozen...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    clip_model.to(device)

    # Freeze entire backbone — no transformer blocks trained at all
    for param in clip_model.parameters():
        param.requires_grad = False
    clip_model.eval()

    frozen_params = sum(p.numel() for p in clip_model.parameters())
    print(f"[finetune] Backbone frozen: {frozen_params:,} params (0 trained)")

    # ── Projection heads ─────────────────────────────────────────────────────
    anime_head = ProjectionHead(CLIP_DIM, PROJ_DIM).to(device)
    manga_head = ProjectionHead(CLIP_DIM, PROJ_DIM).to(device)

    trainable = sum(p.numel() for p in anime_head.parameters()) + \
                sum(p.numel() for p in manga_head.parameters())
    print(f"[finetune] Projection heads: {trainable:,} trainable params "
          f"({100*trainable/(frozen_params+trainable):.2f}% of total)")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"[finetune] Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        anime_head.load_state_dict(ckpt["anime_head"])
        manga_head.load_state_dict(ckpt["manga_head"])
        print(f"[finetune] Resumed — epoch {ckpt['epoch']}, Recall@1: {ckpt['recall_at_1']:.2%}")

    # ── Datasets and loaders ─────────────────────────────────────────────────
    train_ds     = AnimeMangaDataset(train_pairs, preprocess)
    val_ds       = AnimeMangaDataset(val_pairs,   preprocess)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Only optimize projection head parameters — backbone is frozen
    optimizer = torch.optim.AdamW(
        list(anime_head.parameters()) + list(manga_head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0
    best_epoch  = 0

    print(f"\n[finetune] Starting training — {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        anime_head.train()
        manga_head.train()
        epoch_loss = 0.0
        batches    = 0

        for anime_batch, manga_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            anime_batch = anime_batch.to(device)
            manga_batch = manga_batch.to(device)

            # Backbone is frozen so we don't need gradients through it.
            # torch.no_grad() here saves memory and speeds up the forward pass.
            with torch.no_grad():
                anime_feat = clip_model.encode_image(anime_batch)
                manga_feat = clip_model.encode_image(manga_batch)

            # Only the projection heads are trained
            anime_emb = anime_head(anime_feat)
            manga_emb = manga_head(manga_feat)

            loss = infonce_loss(anime_emb, manga_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(anime_head.parameters()) + list(manga_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(batches, 1)

        val_loss, recall_at_1 = validate(clip_model, anime_head, manga_head, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} — "
              f"train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"Recall@1: {recall_at_1:.2%}")

        if recall_at_1 >= best_recall:
            best_recall = recall_at_1
            best_epoch  = epoch
            ckpt_path   = OUT_DIR / "best_checkpoint.pt"
            # Save only the projection head weights — tiny file (~50MB vs 1.6GB)
            torch.save({
                "epoch":       epoch,
                "recall_at_1": recall_at_1,
                "val_loss":    val_loss,
                "anime_head":  anime_head.state_dict(),
                "manga_head":  manga_head.state_dict(),
                "clip_dim":    CLIP_DIM,
                "proj_dim":    PROJ_DIM,
                "model_name":  MODEL_NAME,
                "pretrained":  PRETRAINED,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (Recall@1: {recall_at_1:.2%})")

    print(f"\n[finetune] Done. Best Recall@1: {best_recall:.2%} at epoch {best_epoch}")
    print(f"[finetune] Checkpoint saved to {OUT_DIR}/best_checkpoint.pt")
    print(f"\nNext: update embed.py and query.py to load projection heads, then re-embed.")


if __name__ == "__main__":
    main()