"""
finetune_dino_hardneg.py — DINOv2 fine-tuning with hard negative mining
===========================================================================
Builds on finetune_dino.py (3-block unfrozen, confirmed stable across two
seeds at 69-81% full-index Recall@1). Adds hard negative mining: every few
epochs, the current model embeds the FULL 5,123-panel index and mines each
training anime screenshot's hardest confusable panels. These are injected
into training batches alongside the true match, forcing the model to learn
fine-grained disambiguation it never sees from random in-batch negatives.

Why this targets our actual failure mode:
Our remaining errors are page-level confusion (e.g. p019 vs p020 of the
same chapter) — panels that are visually similar and would almost never
land in the same random batch of 16 out of 5,123 candidates. Random
InfoNCE negatives give near-zero loss gradient for these cases since the
model already easily distinguishes random unrelated panels. Hard mining
fixes this by directly sourcing the negatives that matter.

Run:
    python finetune_dino_hardneg.py --epochs 20 --lr 5e-6 --mine-every 3

Requirements:
    pip install transformers torch torchvision faiss-cpu tqdm
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME      = "facebook/dinov2-base"
LABELS_FILE     = Path("../labels.json")
PANELS_DIR      = Path("../bleach_panels")
SCREENSHOTS_DIR = Path("../screenshots")
OUT_DIR         = Path("dino_finetuned_hardneg")

DINO_DIM = 768
UNFREEZE_LAST_N_BLOCKS = 3   # confirmed best config from prior experiments

HARD_NEG_K          = 3   # how many hard negatives to inject per anime screenshot
HARD_NEG_POOL_SIZE  = 15  # how many top candidates to mine from (excluding the true match)


# ---------------------------------------------------------------------------
# Dataset — now returns hard negatives alongside the positive pair
# ---------------------------------------------------------------------------

class AnimeMangaDataset(Dataset):
    """
    Each item returns (anime_tensor, manga_tensor, hard_neg_tensors).
    hard_neg_tensors is a list of HARD_NEG_K manga panel tensors that are
    NOT the correct match but were ranked highly by the current model.

    On the first pass (before any mining has happened), hard_negatives_map
    is empty and we fall back to returning an empty list — the training
    loop handles this by using pure random InfoNCE for the first few epochs
    until the first mining pass populates the map.
    """

    def __init__(self, pairs: list[dict], transform, hard_negatives_map: dict):
        self.pairs               = pairs
        self.transform            = transform
        self.hard_negatives_map   = hard_negatives_map   # anime_screenshot -> [manga_panel paths]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pair = self.pairs[idx]

        anime_path   = SCREENSHOTS_DIR / pair["anime_screenshot"]
        anime_img    = Image.open(anime_path).convert("RGB")
        anime_tensor = self.transform(anime_img)

        manga_path   = PANELS_DIR / pair["manga_panel"]
        manga_img    = Image.open(manga_path).convert("RGB")
        manga_tensor = self.transform(manga_img)

        # Load hard negatives if we have them for this screenshot yet
        hard_negs = self.hard_negatives_map.get(pair["anime_screenshot"], [])
        hard_neg_tensors = []
        for neg_path in hard_negs[:HARD_NEG_K]:
            try:
                neg_img = Image.open(PANELS_DIR / neg_path).convert("RGB")
                hard_neg_tensors.append(self.transform(neg_img))
            except Exception:
                continue   # skip if a mined path somehow doesn't load

        return anime_tensor, manga_tensor, hard_neg_tensors


def collate_with_hardneg(batch):
    """
    Custom collate function since hard_neg_tensors is a variable-length list
    per item (some screenshots may have fewer than HARD_NEG_K mined negatives,
    especially early on). Stacks anime/manga normally, keeps hard negs as a
    flat list we'll handle manually in the training loop.
    """
    anime_tensors = torch.stack([item[0] for item in batch])
    manga_tensors = torch.stack([item[1] for item in batch])
    # Flatten all hard negatives across the batch into one list
    all_hard_negs = []
    for item in batch:
        all_hard_negs.extend(item[2])
    hard_neg_tensors = torch.stack(all_hard_negs) if all_hard_negs else None
    return anime_tensors, manga_tensors, hard_neg_tensors


# ---------------------------------------------------------------------------
# InfoNCE loss — extended to optionally include hard negatives
# ---------------------------------------------------------------------------

def infonce_loss_with_hardneg(
    anime_emb: torch.Tensor,
    manga_emb: torch.Tensor,
    hard_neg_emb: torch.Tensor | None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard symmetric InfoNCE, but the manga side of the similarity matrix
    is extended with hard negative embeddings as extra columns. This means
    each anime embedding's softmax denominator includes not just the other
    15 random in-batch manga panels, but also however many hard negatives
    were mined for this batch — forcing genuinely harder discrimination.

    The manga->anime direction loss is unaffected by hard negatives (hard
    negs don't have a "correct anime match" of their own), so that
    direction still only uses the standard in-batch panels.
    """
    anime_emb = F.normalize(anime_emb, dim=-1)
    manga_emb = F.normalize(manga_emb, dim=-1)

    B = anime_emb.shape[0]
    labels = torch.arange(B, device=anime_emb.device)

    if hard_neg_emb is not None and hard_neg_emb.shape[0] > 0:
        hard_neg_emb = F.normalize(hard_neg_emb, dim=-1)
        # Extended manga pool: [true manga panels | hard negative panels]
        extended_manga = torch.cat([manga_emb, hard_neg_emb], dim=0)
        logits_a2m = torch.matmul(anime_emb, extended_manga.T) / temperature
        loss_anime = F.cross_entropy(logits_a2m, labels)

        # manga->anime direction: only use the true B×B block (hard negs
        # don't have a corresponding anime image to be "correct" for)
        logits_m2a = torch.matmul(manga_emb, anime_emb.T) / temperature
        loss_manga = F.cross_entropy(logits_m2a, labels)
    else:
        # No hard negatives available yet — fall back to standard InfoNCE
        logits = torch.matmul(anime_emb, manga_emb.T) / temperature
        loss_anime = F.cross_entropy(logits,   labels)
        loss_manga = F.cross_entropy(logits.T, labels)

    return (loss_anime + loss_manga) / 2


# ---------------------------------------------------------------------------
# Freeze / unfreeze — identical to finetune_dino.py
# ---------------------------------------------------------------------------

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_last_n_blocks(model, n: int):
    blocks = model.encoder.layer
    total  = len(blocks)
    print(f"[finetune_hardneg] DINOv2 has {total} transformer blocks — unfreezing last {n}")
    for i, block in enumerate(blocks):
        if i >= total - n:
            for param in block.parameters():
                param.requires_grad = True
    if hasattr(model, "layernorm"):
        for param in model.layernorm.parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[finetune_hardneg] Trainable params: {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.1f}%)")


def get_embedding(model, pixel_values: torch.Tensor) -> torch.Tensor:
    outputs = model(pixel_values=pixel_values)
    return outputs.last_hidden_state[:, 0, :]


# ---------------------------------------------------------------------------
# Hard negative mining — the core new logic
# ---------------------------------------------------------------------------

@torch.no_grad()
def mine_hard_negatives(
    model,
    transform,
    train_pairs: list[dict],
    all_panels_meta: list[dict],
    device: str,
) -> dict:
    """
    Embed the FULL manga panel index with the current model state, then for
    each training anime screenshot, find its top HARD_NEG_POOL_SIZE nearest
    panels via FAISS and exclude the true match. The remaining candidates
    are the hard negative pool for that screenshot — panels the CURRENT
    model confuses for the right answer, which is exactly the training
    signal we want.

    Returns: dict mapping anime_screenshot path -> list of hard negative
    manga_panel paths (folder/file format, ready for Dataset lookup)
    """
    model.eval()
    print(f"[mine] Embedding full {len(all_panels_meta)}-panel index with current model...")

    # Embed all manga panels in batches
    panel_embeddings = []
    batch_tensors = []
    BATCH = 32
    for entry in tqdm(all_panels_meta, desc="Mining: embedding panels", leave=False):
        img_path = PANELS_DIR / entry["folder"] / entry["file"]
        try:
            img = Image.open(img_path).convert("RGB")
            batch_tensors.append(transform(img))
        except Exception:
            batch_tensors.append(torch.zeros(3, 224, 224))   # placeholder, won't match anything meaningfully

        if len(batch_tensors) == BATCH:
            batch = torch.stack(batch_tensors).to(device)
            emb = get_embedding(model, batch)
            emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)
            panel_embeddings.append(emb)
            batch_tensors = []

    if batch_tensors:
        batch = torch.stack(batch_tensors).to(device)
        emb = get_embedding(model, batch)
        emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)
        panel_embeddings.append(emb)

    panel_matrix = np.vstack(panel_embeddings)
    panel_lookup = {f"{p['folder']}/{p['file']}": i for i, p in enumerate(all_panels_meta)}

    # Build a temporary FAISS index for mining
    mining_index = faiss.IndexFlatIP(panel_matrix.shape[1])
    mining_index.add(panel_matrix)

    # For each training anime screenshot, find its hardest negatives
    hard_negatives_map = {}
    print(f"[mine] Mining hard negatives for {len(train_pairs)} training screenshots...")
    for pair in tqdm(train_pairs, desc="Mining: querying", leave=False):
        anime_path = SCREENSHOTS_DIR / pair["anime_screenshot"]
        try:
            img = Image.open(anime_path).convert("RGB")
        except Exception:
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        emb = get_embedding(model, tensor)
        emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)

        scores, indices = mining_index.search(emb, HARD_NEG_POOL_SIZE + 1)
        true_idx = panel_lookup.get(pair["manga_panel"])

        candidates = []
        for idx in indices[0]:
            if idx == true_idx:
                continue   # exclude the true match — we only want negatives
            panel = all_panels_meta[idx]
            candidates.append(f"{panel['folder']}/{panel['file']}")
            if len(candidates) >= HARD_NEG_POOL_SIZE:
                break

        hard_negatives_map[pair["anime_screenshot"]] = candidates

    model.train()
    print(f"[mine] Done. Mined hard negatives for {len(hard_negatives_map)} screenshots.")
    return hard_negatives_map


# ---------------------------------------------------------------------------
# Validation — unchanged from finetune_dino.py, standard InfoNCE eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_anime  = []
    all_manga  = []

    for anime_batch, manga_batch, _ in val_loader:
        anime_emb = get_embedding(model, anime_batch.to(device))
        manga_emb = get_embedding(model, manga_batch.to(device))
        loss = infonce_loss_with_hardneg(anime_emb, manga_emb, None)
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
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 with hard negative mining.")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--mine-every", type=int,   default=3,
                        help="Re-mine hard negatives every N epochs (default: 3)")
    parser.add_argument("--checkpoint", type=str,   default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[finetune_hardneg] Device: {device}")

    # ── Load pairs ──────────────────────────────────────────────────────────
    if not LABELS_FILE.exists():
        sys.exit(f"[finetune_hardneg] {LABELS_FILE} not found.")
    with open(LABELS_FILE) as f:
        all_pairs = json.load(f)

    valid_pairs = []
    for p in all_pairs:
        a = SCREENSHOTS_DIR / p["anime_screenshot"]
        m = PANELS_DIR      / p["manga_panel"]
        if a.exists() and m.exists():
            valid_pairs.append(p)
    print(f"[finetune_hardneg] {len(valid_pairs)} valid pairs.")

    unique_panels = list({p["manga_panel"] for p in valid_pairs})
    random.shuffle(unique_panels)
    split_idx = int(len(unique_panels) * 0.8)
    train_panel_set = set(unique_panels[:split_idx])
    val_panel_set   = set(unique_panels[split_idx:])

    train_pairs = [p for p in valid_pairs if p["manga_panel"] in train_panel_set]
    val_pairs   = [p for p in valid_pairs if p["manga_panel"] in val_panel_set]
    print(f"[finetune_hardneg] Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # ── Load full panel metadata for mining (needs the WHOLE 5,123 index) ──
    dataset_path = PANELS_DIR / "dataset.json"
    if not dataset_path.exists():
        sys.exit(f"[finetune_hardneg] {dataset_path} not found — needed for hard neg mining pool.")
    with open(dataset_path) as f:
        all_panels_meta = json.load(f)["panels"]
    print(f"[finetune_hardneg] Full mining pool: {len(all_panels_meta)} panels.")

    # ── Load DINOv2 ─────────────────────────────────────────────────────────
    print(f"[finetune_hardneg] Loading {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

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
            sys.exit(f"[finetune_hardneg] Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[finetune_hardneg] Resumed from epoch {ckpt['epoch']}, "
              f"Recall@1: {ckpt['recall_at_1']:.2%}")

    freeze_model(model)
    unfreeze_last_n_blocks(model, UNFREEZE_LAST_N_BLOCKS)

    # ── Hard negatives map — starts empty, populated by first mining pass ──
    hard_negatives_map = {}

    train_ds = AnimeMangaDataset(train_pairs, transform, hard_negatives_map)
    val_ds   = AnimeMangaDataset(val_pairs,   transform, {})   # val never uses hard negs

    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_with_hardneg)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_recall = 0.0
    best_epoch  = 0

    print(f"\n[finetune_hardneg] Starting training — {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr}, mine_every={args.mine_every}\n")

    for epoch in range(1, args.epochs + 1):

        # ── Mine hard negatives every N epochs (and always before epoch 1) ──
        if (epoch - 1) % args.mine_every == 0:
            mined = mine_hard_negatives(model, transform, train_pairs, all_panels_meta, device)
            hard_negatives_map.clear()
            hard_negatives_map.update(mined)
            # Dataset holds a reference to this dict, so it sees the update automatically

        # Rebuild loader each epoch since hard_negatives_map content changes
        # (the dict itself is the same object, but DataLoader workers=0 means
        # no caching issue — this is mostly for clarity)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=0,
                                  collate_fn=collate_with_hardneg)

        model.train()
        epoch_loss = 0.0
        batches    = 0

        for anime_batch, manga_batch, hard_neg_batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False):
            anime_batch = anime_batch.to(device)
            manga_batch = manga_batch.to(device)

            anime_emb = get_embedding(model, anime_batch)
            manga_emb = get_embedding(model, manga_batch)

            hard_neg_emb = None
            if hard_neg_batch is not None:
                hard_neg_batch = hard_neg_batch.to(device)
                hard_neg_emb = get_embedding(model, hard_neg_batch)

            loss = infonce_loss_with_hardneg(anime_emb, manga_emb, hard_neg_emb)

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

    print(f"\n[finetune_hardneg] Done. Best Recall@1: {best_recall:.2%} at epoch {best_epoch}")
    print(f"[finetune_hardneg] Checkpoint saved to {OUT_DIR}/best_checkpoint.pt")


if __name__ == "__main__":
    main()