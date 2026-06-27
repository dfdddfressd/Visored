# Visored — Decisions & Improvements Log

## Project Overview
Cross-domain visual retrieval: given an anime screenshot, retrieve the corresponding manga panel from a 40,000+ panel corpus. Built on fine-tuned DINOv2 + FAISS vector search.

---

## Architecture Decisions

### Backbone: DINOv2 over CLIP
**Decision:** Switched from CLIP ViT-L/14 to DINOv2 ViT-B/14.  
**Why:** CLIP is trained with image-text contrastive loss — its embedding space is shaped by what's "describable in language." For anime→manga retrieval, the task is purely visual: same scene, different artistic style. DINOv2 is trained with self-distillation (DINO objective, no language supervision), optimizing purely for visual consistency across augmented views of the same image. This is a closer match to what cross-domain visual retrieval actually requires.  
**Evidence:** Empirically confirmed — DINOv2 zero-shot outperformed CLIP zero-shot on the retrieval task. DINOv2 finetuned achieved 74.65% Recall@1 on the full 42k-panel index (chapters 1-32 val set).  
**Note for paper:** CLIP baseline results should still be reported for completeness. CLIP finetuning code retained.

### Model Size: ViT-B/14
**Decision:** DINOv2 ViT-B/14 (86M params, 768-dim output).  
**Why:** ViT-L/14 (~300M params) is too slow for CPU fine-tuning. ViT-S/14 has lower capacity. ViT-B/14 is the practical balance between representational capacity and training speed on CPU hardware.

### Unfrozen Blocks: 3
**Decision:** Unfreeze last 3 of 12 transformer blocks + final layernorm.  
**Why:** Conservative start to avoid catastrophic forgetting of DINOv2's pretrained visual features. 3 blocks = 24.6% of trainable params. Proportionally similar to CLIP experiments.  
**Result:** Achieved good generalization without overfitting at correct LR.

### Loss Function: InfoNCE (symmetric)
**Decision:** Symmetric InfoNCE with temperature=0.07.  
**Why:** Standard choice for contrastive cross-modal learning. Symmetric means both anime→manga and manga→anime directions are trained simultaneously, which is appropriate since the embedding space should be shared.

---

## Training Decisions

### Learning Rate: 1e-6 (revised from 5e-6)
**Decision:** Dropped LR from 5e-6 to 1e-6 after observing overfitting.  
**Why:** At 5e-6, train loss collapsed to ~0.04 by epoch 7 while val loss stayed at ~0.97 — a clear overfitting signature. The model was memorizing training pairs rather than learning generalizable embeddings. At 1e-6, train/val loss gap is tighter and Recall@1 improves steadily across epochs.  
**Result:** Best checkpoint at epoch 11, val Recall@1 44.35%. Full-index Recall@1 22.67% on diverse val set (vs 12.89% zero-shot — ~75% improvement over baseline).

### Panel-Aware Train/Val Split
**Decision:** Split on unique manga panels, not pairs. 80/20 split by panel.  
**Why:** Naive random split would allow the same manga panel to appear in both train and val with different anime screenshots — data leakage. Panel-aware split ensures the model is evaluated on panels it has never seen during training.

### Hard Negative Mining: Abandoned
**Decision:** Implemented then abandoned hard negative mining.  
**Why:** Performed worse than standard InfoNCE. Likely cause: with an undertrained model, hard negatives are noisy — the model hasn't learned a good enough embedding space to distinguish "hard but wrong" from "actually correct." Hard negatives require a well-initialized model to be beneficial.

### Optimizer: AdamW
**Decision:** AdamW with weight_decay=0.01, gradient clipping max_norm=1.0.  
**Why:** Weight decay regularizes to prevent overfitting. Gradient clipping prevents loss spikes during early training. Standard best practice for ViT fine-tuning.

### Scheduler: CosineAnnealingLR
**Decision:** Cosine LR decay over T_max=epochs.  
**Why:** Smooth LR decay allows the model to settle into a good minimum without oscillating at the end of training.

---

## Dataset & Indexing Decisions

### Panel Splicing: Kumiko-style OpenCV Contour Detection
**Decision:** Custom panel splicer based on OpenCV contour detection (inspired by Kumiko by njean42).  
**Key fix — Dual threshold:** Original single-threshold approach only marked near-white pixels as background. Colored Bleach pages have black gutters between panels, which were being treated as foreground content — causing adjacent panels to merge. Fix: run two threshold passes (white-gutter mask + black-gutter mask) and combine. Any pixel that is very white OR very black is treated as a separator.  
**Key fix — Full-page spread detection:** Added coverage ratio check — if detected panels cover <50% of the page area, or the largest panel covers >65% of the page, treat the whole page as a single panel. Prevents full-page spreads (e.g. Yamamoto's Zanka no Tachi reveal) from being incorrectly split.

### Language Coverage
**Decision:** English for chapters 1-53 and 480-686, Latin American Spanish (es-la) for chapters 54-479.  
**Why:** MangaDex colored edition availability. English chapters available at both ends of the series; Spanish fills the middle gap. Visually identical — language only affects speech bubble text, which is not a primary retrieval signal.

### Panel Corpus Size
- v1: ~5,000 panels (50 chapters)
- v2: 42,112 panels (706 chapters) — full Bleach series
- Current (after re-splice fix): 39,916 panels  
**Note:** Panel count reduction from v2 to current is expected — full-page spread detection now correctly saves spreads as single panels rather than fragmenting them.

### FAISS Index: IndexFlatIP
**Decision:** Flat inner product index (exact cosine similarity search).  
**Why:** At 40k vectors of 768 dimensions, flat search is fast enough (~1-2 seconds per query on CPU). Approximate nearest neighbor indices (IVF, HNSW) trade accuracy for speed — not necessary at this scale. Exact search ensures no retrieval errors from approximation.

---

## Labeling Decisions

### Label Quality Standard
**Decision:** Context and dialogue match, not pure visual similarity.  
**Why:** The task is scene alignment, not visual similarity matching. A label where the anime frame and manga panel depict the same scene moment is correct even if visual similarity is imperfect (due to anime artistic liberties). Pure visual similarity matching produces mislabels that actively hurt training.  
**Key insight:** The labeler presents FAISS candidates — the labeler's job is to correct the model, not confirm its guesses.

### Skip Aggressively
**Decision:** Skip anime frames where no manga panel is a reasonable scene match.  
**Why:** A skipped frame produces no training signal. A mislabeled frame produces negative training signal. Skipping is always better than a weak or wrong label.

### Anime-Manga Divergence
**Decision:** Labels are valid when the anime takes artistic liberty with a manga moment, as long as the scene and context clearly correspond.  
**Why:** Bleach's anime frequently extends, reorders, or reinterprets manga scenes. The correct label is "the closest manga panel to what's happening in this anime frame," not "the visually identical panel."

---

## Evaluation

### Eval Script: eval_recall_full_dino.py
Evaluates against the full FAISS index (not just val subset) to get real-world retrieval numbers. Reports Recall@K for K=1,5,10,20,50. Also reports per-pair failure breakdown tiered as:
- Near-miss (rank 2-5): retrieval precision problem
- Soft failure (rank 6-25): model found it but ranked poorly
- Hard failure (rank >25 or not found): complete miss or training data gap

### Val Set Evolution
| Run | Recall@1 | Val pairs | Coverage |
|-----|----------|-----------|----------|
| CLIP zero-shot | ~30% | 137 | chapters 1-32 |
| DINOv2 zero-shot (chapters 1-32 val) | ~45% | 137 | chapters 1-32 |
| DINOv2 finetuned v1 | 74.65% | 142 | chapters 1-32 only |
| DINOv2 finetuned v2 (5e-6 LR) | 38.20% | 233 | multi-arc |
| DINOv2 zero-shot (multi-arc val) | 12.89% | 225 | multi-arc |
| DINOv2 finetuned v3 (1e-6 LR) | 22.67% | 225 | multi-arc |

**Important:** Val set comparisons across runs are not apples-to-apples. The multi-arc val set is significantly harder than the chapters 1-32 val set. Finetuned v3 at 22.67% on multi-arc val represents ~75% improvement over zero-shot on the same val set.

---

## Current Bottleneck
Labeled data volume and arc coverage. Current labeled episodes:
- Chapters 1-32: well covered (~750 pairs)
- Chapter 281 (Grimmjow/Hueco Mundo): ~81 pairs, 1 episode
- Chapter 365 (Vizards/Karakura): ~116 pairs, 1 episode
- Chapter 506 (Yamamoto/TYBW): ~104 pairs, 1 episode
- Chapter 566 (Rukia vs As Nodt/TYBW): ~132 pairs, 1 episode

**Target:** 3-5 episodes per arc, plus Soul Society arc (chapters 70-182) which has zero coverage. Goal: ~25-30 total episodes before next major retrain.

---

## Potential Paper Angle
Workshop paper at ECCV/CVPR/ICCV focused on:
- Novel task: cross-modal anime-to-manga panel retrieval
- Dataset contribution: 706-chapter labeled alignment dataset (first of its kind)
- Empirical findings: DINOv2 vs CLIP for cross-domain visual retrieval, effect of training data diversity
- Target venues: multimedia retrieval or comics/manga understanding workshops
