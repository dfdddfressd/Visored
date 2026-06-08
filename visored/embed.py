"""
embed.py — Visored Panel Embedding Pipeline
============================================
Reads bleach_panels/dataset.json, encodes every panel image through CLIP,
and saves a FAISS index + parallel metadata file for fast nearest-neighbor search.
 
Run:
    python embed.py
    python embed.py --panels-dir bleach_panels --out-dir .
"""
 
import argparse
import json
import os
import sys
from pathlib import Path
 
import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"          # OpenAI's original CLIP weights, best zero-shot baseline
EMBEDDING_DIM = 512            # ViT-B/32 output dimension — fixed by the model
BATCH_SIZE = 64                # How many panels to encode per forward pass
                               # Lower this (e.g. 16) if you hit memory errors
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def load_model(device: str):
    """
    Load the CLIP model and its image preprocessor.
 
    open_clip.create_model_and_transforms returns three things:
      - model:     the neural network
      - _:         a text tokenizer we don't need here
      - preprocess: a torchvision transform that resizes + normalizes images
                    to exactly what CLIP expects (224x224, ImageNet stats)
    """
    print(f"[embed] Loading {MODEL_NAME} ({PRETRAINED} weights) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval()           # Disable dropout — we're doing inference, not training
    model.to(device)
    return model, preprocess
 
 
def load_dataset(panels_dir: Path) -> list[dict]:
    """
    Read dataset.json and return the flat list of panel entries.
    Each entry has keys: chapter, page, panel, file, folder, bbox, etc.
    """
    dataset_path = panels_dir / "dataset.json"
    if not dataset_path.exists():
        sys.exit(f"[embed] ERROR: {dataset_path} not found. Run panel_splicer.py first.")
 
    with open(dataset_path) as f:
        data = json.load(f)
 
    panels = data["panels"]
    print(f"[embed] Found {len(panels)} panels across {data['total_chapters']} chapters.")
    return panels
 
 
def load_image(panels_dir: Path, entry: dict, preprocess) -> torch.Tensor | None:
    """
    Load a single panel image from disk and run CLIP's preprocessor on it.
 
    Returns a (1, 3, 224, 224) tensor, or None if the file is missing/corrupt.
    CLIP's preprocess handles: resize → center crop → normalize to [-1, 1].
    """
    img_path = panels_dir / entry["folder"] / entry["file"]
    try:
        img = Image.open(img_path).convert("RGB")
        return preprocess(img)           # → (3, 224, 224) tensor
    except Exception as e:
        print(f"[embed] WARNING: skipping {img_path} — {e}")
        return None
 
 
def encode_batch(model, batch_tensors: list[torch.Tensor], device: str) -> np.ndarray:
    """
    Run a batch of preprocessed image tensors through CLIP's image encoder.
 
    Steps:
      1. Stack list of (3,224,224) tensors → (B, 3, 224, 224) batch tensor
      2. Move to device (CPU or CUDA)
      3. Forward pass through CLIP's visual encoder → (B, 512) raw embeddings
      4. L2-normalize each vector so dot product == cosine similarity
         This is critical: FAISS IndexFlatIP does inner product search,
         which equals cosine similarity only when vectors are unit length.
 
    torch.no_grad() disables gradient tracking — saves memory, speeds up inference.
    """
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        features = model.encode_image(batch)           # (B, 512)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
 
    return features.cpu().numpy().astype(np.float32)   # FAISS expects float32
 
 
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat inner-product index from the embedding matrix.
 
    IndexFlatIP = exact (brute-force) search using inner product.
    Since we L2-normalized everything, inner product == cosine similarity.
    'Flat' means no compression or approximation — perfect recall, small dataset.
 
    For scale (100k+ panels) you'd switch to IndexIVFFlat or IndexHNSWFlat,
    which trade a tiny bit of recall for dramatically faster search.
    At 10-chapter scale, flat is fine and simpler.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)                              # Add all vectors at once
    print(f"[embed] FAISS index built — {index.ntotal} vectors, dim={EMBEDDING_DIM}.")
    return index
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(description="Embed Bleach manga panels with CLIP.")
    parser.add_argument(
        "--panels-dir", default="bleach_panels",
        help="Path to the panel_splicer output directory (default: bleach_panels)"
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Where to write index.faiss and index_meta.json (default: current dir)"
    )
    args = parser.parse_args()
 
    panels_dir = Path(args.panels_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Detect device — MPS is Apple Silicon GPU (M1/M2/M3), CUDA is NVIDIA
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
 
    model, preprocess = load_model(device)
    panels = load_dataset(panels_dir)
 
    # -----------------------------------------------------------------------
    # Encode all panels in batches
    # -----------------------------------------------------------------------
    all_embeddings = []   # Will be list of (batch_size, 512) arrays
    valid_panels   = []   # Parallel list — only panels we successfully loaded
 
    batch_tensors  = []   # Current batch accumulator
    batch_meta     = []   # Metadata for current batch
 
    print(f"[embed] Encoding {len(panels)} panels (batch size {BATCH_SIZE})...")
 
    for entry in tqdm(panels, unit="panel"):
        tensor = load_image(panels_dir, entry, preprocess)
        if tensor is None:
            continue                           # Skip corrupt/missing files
 
        batch_tensors.append(tensor)
        batch_meta.append(entry)
 
        if len(batch_tensors) == BATCH_SIZE:
            emb = encode_batch(model, batch_tensors, device)
            all_embeddings.append(emb)
            valid_panels.extend(batch_meta)
            batch_tensors = []
            batch_meta    = []
 
    # Flush any remaining images that didn't fill a full batch
    if batch_tensors:
        emb = encode_batch(model, batch_tensors, device)
        all_embeddings.append(emb)
        valid_panels.extend(batch_meta)
 
    if not all_embeddings:
        sys.exit("[embed] ERROR: No panels were successfully encoded. Check your bleach_panels/ directory.")
 
    # Stack all batch arrays into one big matrix: (total_panels, 512)
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"[embed] Encoded {len(valid_panels)} panels → matrix shape {embeddings_matrix.shape}.")
 
    # -----------------------------------------------------------------------
    # Build and save the FAISS index
    # -----------------------------------------------------------------------
    index = build_faiss_index(embeddings_matrix)
 
    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"[embed] Saved FAISS index → {index_path}")
 
    # -----------------------------------------------------------------------
    # Save metadata
    # -----------------------------------------------------------------------
    # index_meta.json is a list where index_meta[i] is the panel metadata
    # for the i-th vector in the FAISS index. This is how query.py maps
    # a FAISS result ID back to "Chapter 3, page 7, panel 2".
    meta_path = out_dir / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(valid_panels, f, indent=2)
    print(f"[embed] Saved metadata → {meta_path}")
 
    print("\n[embed] Done. Run query.py --image <screenshot.jpg> to search.")
 
 
if __name__ == "__main__":
    main()
 


