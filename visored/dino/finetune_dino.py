"""
finetune_dino.py — Fine-tune DINOv2 on anime/manga pairs for Visored
=======================================================================
Parallel experiment to clip/finetune.py. Same InfoNCE loss, same dataset,
same overall training loop structure — the ONLY thing that changes is the
backbone (DINOv2 ViT-B/14 instead of CLIP ViT-L/14). This isolates whether
the backbone itself is the bottleneck.

Why DINOv2:
- CLIP is trained with image-TEXT contrastive loss, so its embedding space
  is shaped by what's "describable in language" — it can underweight pure
  visual structure that doesn't map cleanly to words.
- DINOv2 is trained with self-distillation (no language supervision at all),
  optimizing purely for visual consistency across augmented views of the
  same image. For a same-content-different-style task like anime↔manga,
  this is a closer match to what we actually need.

Why ViT-B/14 (not ViT-L/14 or ViT-S/14):
- ViT-L/14 DINOv2 is ~300M params — too slow to fine-tune on CPU at any
  reasonable iteration speed.
- ViT-S/14 is fast but lower capacity; B/14 (86M params) is the balance.

Why only 2 blocks unfrozen (conservative start):
- DINOv2 ViT-B/14 has 12 transformer blocks (vs CLIP's 24), so 2 blocks
  here is proportionally similar risk to CLIP's "2 block" experiment,
  which performed reasonably (66.67%) without the overfitting CLIP showed
  at 4 blocks. Starting conservative — can unfreeze more later if needed.

Run:
    python finetune_dino.py --epochs 20 --lr 5e-6
    python finetune_dino.py --epochs 20 --lr 5e-6 --checkpoint dino_finetuned/best_checkpoint.pt

Requirements:
    pip install transformers torch tqdm pillow
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# facebook/dinov2-base = ViT-B/14, 86M params, 768-dim output
# (DINOv2's "base" naming maps to ViT-B, same convention as CLIP)
MODEL_NAME      = "facebook/dinov2-base"
LABELS_FILE     = Path("../labels.json")
PANELS_DIR      = Path("../bleach_panels")
SCREENSHOTS_DIR = Path("../screenshots")
OUT_DIR         = Path("dino_finetuned")

DINO_DIM = 768   # ViT-B/14 output dimension (same as CLIP ViT-L/14, convenient coincidence)

# Conservative — see module docstring for reasoning
UNFREEZE_LAST_N_BLOCKS = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnimeMangaDataset(Dataset):
    """
    Identical structure to clip/finetune.py's dataset class.
    No manga-mode preprocessing — we already established color should stay
    since both anime and manga here are colored.

    DINOv2 uses its own normalization stats (different from CLIP's), which
    is why we use AutoImageProcessor's transform instead of open_clip's
    preprocess function.
    """

    def __init__(self, pairs: list[dict], transform):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        anime_path = SCREENSHOTS_DIR / pair["anime_screenshot"]
        anime_img  = Image.open(anime_path).convert("RGB")
        anime_tensor = self.transform(anime_img)

        manga_path = PANELS_DIR / pair["manga_panel"]
        manga_img  = Image.open(manga_path).convert("RGB")
        manga_tensor = self.transform(manga_img)

        return anime_tensor, manga_tensor


# ---------------------------------------------------------------------------
# InfoNCE loss — identical to clip/finetune.py, unchanged on purpose
# ---------------------------------------------------------------------------

def infonce_loss(anime_emb: torch.Tensor, manga_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Same symmetric InfoNCE as the CLIP version. Kept identical so the ONLY
    variable that changes between the two experiments is the backbone.
    """
    anime_emb = F.normalize(anime_emb, dim=-1)
    manga_emb = F.normalize(manga_emb, dim=-1)

    logits = torch.matmul(anime_emb, manga_emb.T) / temperature
    labels = torch.arange(len(anime_emb), device=anime_emb.device)

    loss_anime = F.cross_entropy(logits,   labels)
    loss_manga = F.cross_entropy(logits.T, labels)
    return (loss_anime + loss_manga) / 2


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n: int):
    """
    DINOv2's HuggingFace implementation exposes transformer blocks at
    model.encoder.layer (list of Dinov2Layer), analogous to CLIP's
    model.visual.transformer.resblocks.

    DINOv2 has no separate "proj" matrix like CLIP — the final hidden
    state IS the embedding (after layer norm), so there's nothing
    equivalent to CLIP's proj-freezing concern here.
    """
    blocks = model.encoder.layer
    total  = len(blocks)
    print(f"[finetune_dino] DINOv2 has {total} transformer blocks — unfreezing last {n}")

    for i, block in enumerate(blocks):
        if i >= total - n:
            for param in block.parameters():
                param.requires_grad = True

    # Unfreeze final layernorm — analogous to CLIP's ln_post
    if hasattr(model, "layernorm"):
        for param in model.layernorm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[finetune_dino] Trainable params: {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.1f}%)")


def get_embedding(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Extract a single embedding vector per image from DINOv2's output.

    DINOv2's forward pass returns a sequence of patch tokens + 1 CLS token.
    The CLS token (index 0) is the standard choice for a global image
    embedding — it's trained to aggregate information from the whole image,
    analogous to CLIP's pooled output.
    """
    outputs = model(pixel_values=pixel_values)
    cls_embedding = outputs.last_hidden_state[:, 0, :]   # (batch, 768)
    return cls_embedding


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_anime  = []
    all_manga  = []

    for anime_batch, manga_batch in val_loader:
        anime_emb = get_embedding(model, anime_batch.to(device))
        manga_emb = get_embedding(model, manga_batch.to(device))
        loss = infonce_loss(anime_emb, manga_emb)
        total_loss += loss.item()
        all_anime.append(F.normalize(anime_emb, dim=-1).cpu())
        all_manga.append(F.normalize(manga_emb, dim=-1).cpu())

    all_anime  = torch.cat(all_anime)
    all_manga  = torch.cat(all_manga)
    sim_matrix = torch.matmul(all_anime, all_manga.T)
    top1_preds = sim_matrix.argmax(dim=1)
    correct    = (top1_preds == torch.arange(len(all_anime))).float().mean().item()

    model.train()
    return total_loss / len(val_loader), correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on anime/manga pairs.")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-6,
                        help="Same conservative LR as the CLIP baseline run for fair comparison")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--checkpoint", type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[finetune_dino] Device: {device}")

    # ── Load pairs ──────────────────────────────────────────────────────────
    if not LABELS_FILE.exists():
        sys.exit(f"[finetune_dino] {LABELS_FILE} not found.")
    with open(LABELS_FILE) as f:
        all_pairs = json.load(f)
    print(f"[finetune_dino] Loaded {len(all_pairs)} labeled pairs.")

    valid_pairs = []
    for p in all_pairs:
        a = SCREENSHOTS_DIR / p["anime_screenshot"]
        m = PANELS_DIR      / p["manga_panel"]
        if a.exists() and m.exists():
            valid_pairs.append(p)
        else:
            print(f"[finetune_dino] WARNING: missing files — {p['anime_screenshot']}")
    print(f"[finetune_dino] {len(valid_pairs)} pairs with files present.")

    if len(valid_pairs) < 8:
        sys.exit("[finetune_dino] Need at least 8 valid pairs to train.")

    # Panel-aware split — same fix we applied to the CLIP pipeline, so the
    # same manga panel never appears in both train and val (no leakage)
    unique_panels = list({p["manga_panel"] for p in valid_pairs})
    random.shuffle(unique_panels)
    split_idx = int(len(unique_panels) * 0.8)
    train_panel_set = set(unique_panels[:split_idx])
    val_panel_set   = set(unique_panels[split_idx:])

    train_pairs = [p for p in valid_pairs if p["manga_panel"] in train_panel_set]
    val_pairs   = [p for p in valid_pairs if p["manga_panel"] in val_panel_set]
    print(f"[finetune_dino] Train: {len(train_pairs)} pairs ({len(train_panel_set)} panels) | "
          f"Val: {len(val_pairs)} pairs ({len(val_panel_set)} panels)")

    # ── Load DINOv2 ─────────────────────────────────────────────────────────
    print(f"[finetune_dino] Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

    # Build a torchvision transform matching the processor's settings,
    # since we need a callable transform per-image for the Dataset class
    # (AutoImageProcessor expects batched PIL inputs, this is simpler for
    # a Dataset's __getitem__ which works one image at a time)
    image_mean = processor.image_mean
    image_std  = processor.image_std
    image_size = processor.crop_size["height"] if hasattr(processor, "crop_size") else 224

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            sys.exit(f"[finetune_dino] Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[finetune_dino] Resuming — epoch {ckpt['epoch']}, Recall@1: {ckpt['recall_at_1']:.2%}")

    freeze_model(model)
    unfreeze_last_n_blocks(model, UNFREEZE_LAST_N_BLOCKS)

    # ── Datasets and loaders ─────────────────────────────────────────────────
    train_ds = AnimeMangaDataset(train_pairs, transform)
    val_ds   = AnimeMangaDataset(val_pairs,   transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0
    best_epoch  = 0

    print(f"\n[finetune_dino] Starting training — {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches    = 0

        for anime_batch, manga_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            anime_batch = anime_batch.to(device)
            manga_batch = manga_batch.to(device)

            anime_emb = get_embedding(model, anime_batch)
            manga_emb = get_embedding(model, manga_batch)

            loss = infonce_loss(anime_emb, manga_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            batches    += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(batches, 1)

        val_loss, recall_at_1 = validate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{args.epochs} — "
              f"train_loss: {avg_train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"Recall@1: {recall_at_1:.2%}")

        if recall_at_1 >= best_recall:
            best_recall = recall_at_1
            best_epoch  = epoch
            ckpt_path   = OUT_DIR / "best_checkpoint.pt"
            torch.save({
                "epoch":       epoch,
                "model_name":  MODEL_NAME,
                "recall_at_1": recall_at_1,
                "val_loss":    val_loss,
                "state_dict":  model.state_dict(),
                "dino_dim":    DINO_DIM,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (Recall@1: {recall_at_1:.2%})")

    print(f"\n[finetune_dino] Done. Best Recall@1: {best_recall:.2%} at epoch {best_epoch}")
    print(f"[finetune_dino] Checkpoint saved to {OUT_DIR}/best_checkpoint.pt")
    print(f"\nNext: python embed_dino.py --checkpoint {OUT_DIR}/best_checkpoint.pt --out-dir index_dino")


if __name__ == "__main__":
    main()