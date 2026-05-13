# DeepPanel: panel segmentation and splicing — agent specification

This document describes **exactly** what the DeepPanel **Python repository** implements today, and how a separate system must **derive panel crops (“splicing”)** from that output. It is written so a coding agent can reproduce the behavior without reading the rest of the codebase first.

---

## 1. Critical scope statement (read first)

| Question | Answer in *this* repo |
|----------|------------------------|
| Does this codebase crop individual comic panels from a page image? | **No.** There is no bounding-box extraction, no `crop()`, no contour tracing, and no geometry that splits a page into separate panel bitmaps. |
| What *does* it do? | It runs a **semantic segmentation** model on a **224×224** square (after padding and resize), producing **one of three class labels per pixel**. It can **save a grayscale visualization** of that label map. |
| Where might real “splicing” live? | The README points to **DeepPanelAndroid** and **DeepPaneliOS** for on-device use. Any rectangle extraction, reading order, or UI cropping is **out of scope for this repository** unless you add it yourself. |

**Implication for an agent:** To “recreate the process” end-to-end including **physical panel splicing**, you must implement **post-processing** (Section 8) that is **not present** in this tree. Sections 3–7 are **faithful** to the existing code.

---

## 2. Semantic model: what “a panel” means in the network

The network is trained for **3 classes** per pixel. Integer labels are fixed in code:

| Integer label | Constant in `utils.py` | Meaning (dataset / supervision) |
|---------------|------------------------|----------------------------------|
| `0` | `BACKGROUND_LABEL` | Page background (outside panels / gutters / margins as annotated). |
| `1` | `BORDER_LABEL` | Panel borders / separators / framing ink as annotated. |
| `2` | `CONTENT_LABEL` | Interior of a “reading unit” (the README states the model is trained so related panels may be **one** content region, mimicking human grouping). |

**Class index ↔ model output channel:** The final convolution outputs **3 channels** (`OUTPUT_CHANNELS = 3` in `DeepPanel.py`). With `SparseCategoricalCrossentropy(from_logits=True)`, channel index `k` corresponds to class label `k`. Therefore:

```text
predicted_label[y, x] = argmax_k logits[y, x, k]   # k ∈ {0, 1, 2}
```

Use **`argmax` on the last dimension (channel axis)**. Do not apply softmax before argmax for the discrete class index (argmax of logits equals argmax of softmax).

---

## 3. On-disk dataset layout (training / evaluation)

### 3.1 Directory structure

From the README and `utils.load_data_set()`:

```text
dataset/
  training/
    raw/                  # page images: *.jpg
    segmentation_mask/  # masks: *.png (paired by basename)
  test/
    raw/
    segmentation_mask/
```

### 3.2 Pairing rule (`utils.parse_image`)

1. Start from a path to a raw JPEG, e.g. `.../raw/foo.jpg`.
2. Replace the path substring `raw` with `segmentation_mask`.
3. Replace the extension `jpg` with `png`.

Example: `./dataset/test/raw/001.jpg` → `./dataset/test/segmentation_mask/001.png`.

### 3.3 Human-facing mask colors (README)

The README instructs authors to paint masks with **full RGB**:

- Blue → background  
- Red → border  
- Green → panel content  

### 3.4 How training actually reads masks (`utils.parse_image`)

Training does **not** branch on RGB tuples in the main loader. It uses:

```python
mask = tf.io.read_file(mask_path)
mask = tf.image.decode_png(mask, channels=1)  # single channel
```

Then it maps **specific single-channel byte values** to labels (this handles masks that, after decode to one channel, yield these gray levels—e.g. from palette or color-to-luminance):

| Raw byte in decoded 1-channel mask | Assigned label |
|------------------------------------|----------------|
| `255` | `BACKGROUND_LABEL` (0) |
| `29` | `BACKGROUND_LABEL` (0) |
| `76` | `BORDER_LABEL` (1) |
| `134` | `BORDER_LABEL` (1) |
| `149` | `CONTENT_LABEL` (2) |

**Important:** Any pixel value that never matches these `tf.where` conditions **keeps its original byte value** and is **not** coerced to a valid class. A coding agent reproducing training must either replicate this exactly or **normalize masks** so every pixel hits one of these buckets (recommended for robustness).

### 3.5 Auxiliary mask tooling (dataset QA)

- **`CheckSegmentationMasksQuality.py`**: Loads masks as **RGB** with Pillow, snaps “invalid” channel values to pure `(255,0,0)`, `(0,255,0)`, `(0,0,255)` by dominance rules, saves fixed files in place.
- **`WeightCalculator.py`**: Counts class fractions after **PIL** `ImageOps.expand` + **`resize((IMAGE_SIZE, IMAGE_SIZE), NEAREST)`** — used for analytics, **not** the same code path as `tf.image.resize_with_pad` in training. Do **not** assume WeightCalculator’s padding matches TensorFlow’s unless you verify numerically.

---

## 4. Spatial preprocessing (must match for train / test / inference)

### 4.1 Constants

- `IMAGE_SIZE = 224` (square side length in pixels).

### 4.2 Page and mask alignment

In both `load_image_train` and `load_image_test` (`utils.py`):

1. **Images (RGB):**  
   `input_image = tf.image.resize_with_pad(datapoint['image'], target_height=224, target_width=224)`
2. **Masks (single-channel labels):**  
   `input_mask = tf.image.resize_with_pad(datapoint['segmentation_mask'], target_height=224, target_width=224)`

**Semantics of `resize_with_pad`:** TensorFlow resizes the content to **fit entirely inside** 224×224 **preserving aspect ratio**, then pads (value `0` for images; for masks the pad is also `0`—ensure your label `0` is background so padded regions are background). The result is always **224×224**.

**Training-only** (`load_image_train`): With probability `0.5`, apply `tf.image.flip_left_right` to **both** image and mask identically.

### 4.3 Normalization

After resize (and optional flip):

```python
input_image = tf.cast(input_image, tf.float32) / 255.0
# mask stays integer type for sparse CE; not divided in code
```

So the model sees **RGB float in [0, 1]**.

### 4.4 Training-only: per-pixel loss weights

Still in `load_image_train`, for each sample the code computes the fraction of pixels in each class on the **224×224** mask, then builds a per-pixel weight tensor:

- `background_weight = 0.33 / fraction_background`
- `border_weight = 0.33 / fraction_border`
- `content_weight = 0.34 / fraction_content`

Each pixel’s weight is the weight of **its true class**. The tuple `(input_image, input_mask, weights)` is what training consumes (the `fit` path must use a weighted loss or sample weights—follow how `DeepPanel.py` wires the dataset; reproducing training requires matching that).

---

## 5. Model I/O contract (for any reimplementation)

### 5.1 Input

- Shape: `[batch, 224, 224, 3]`
- Dtype: `float32`
- Value range: `[0.0, 1.0]` per channel (post-normalization).

### 5.2 Output

- Shape: `[batch, 224, 224, 3]` (same spatial size as input after the built-in U-Net head; the project’s `compare_accuracy` loops assume **224×224** label maps).
- Dtype: `float32`
- Semantics: **logits** for 3 classes (no softmax in the loss).

### 5.3 Architecture summary (for loading checkpoints)

- Encoder: **`tf.keras.applications.MobileNetV2`** with `input_shape=[224,224,3]`, `include_top=False`, **frozen** (`down_stack.trainable = False`).
- Skip layer names (outputs of these layers are used):  
  `block_1_expand_relu`, `block_3_expand_relu`, `block_6_expand_relu`, `block_13_expand_relu`, `block_16_project`
- Decoder: **`tensorflow_examples.models.pix2pix.upsample`** blocks with channel sizes `[576, 192, 144, 96]` in order, each with kernel size `3`, with **concat skip connections** from the encoder in the U-Net pattern.
- Head: **`tf.keras.layers.Conv2DTranspose`** with `filters=3`, `kernel_size=3`, `strides=2`, `padding='same'`.

### 5.4 Saved artifacts (as implemented)

- After training: `model.save("./model/model.keras")` and `model.export("./model")` (SavedModel directory layout under `./model/`).
- `DeepPanelTest.py` loads: `keras.models.load_model("./model/model.keras", custom_objects={...})` with the metric functions from `metrics.py`.
- `DeepPanelMobile.py`: `tf.lite.TFLiteConverter.from_saved_model("./model/")`, `optimizations=[DEFAULT]`, `supported_types=[float16]`, output `./model/deepPanel.tflite`.

---

## 6. Inference pipeline (`DeepPanelTest.py`) — step by step

This is the **canonical** “what the Python repo does to a page” flow.

1. **Seed:** `tf.random.set_seed(11)` (matches training script seed).
2. **Load model** from `./model/model.keras` with `custom_objects` including: `border_acc`, `background_acc`, `content_acc`, `iou_coef`, `dice_coef` (required for deserialization).
3. **Build test dataset:**
   - `testing_num_files = count_files_in_folder("./dataset/test/raw")` where `count_files_in_folder` uses **`sorted`** `os.listdir` file list (only files, not subdirs).
   - `TESTING_BATCH_SIZE = testing_num_files` (entire test set in one batch).
   - `load_data_set()['test']` → `map(load_image_test)` → `batch(TESTING_BATCH_SIZE)`.
4. **Materialize batch:** The script uses a `for images, true_masks in test_dataset: pass` pattern, then `images = images.numpy()`, `true_masks = true_masks.numpy()`.
5. **Predict:** `predictions = model.predict(test_dataset)` — same batched dataset; shape `[N, 224, 224, 3]` logits.
6. **Per-image argmax:** For each `prediction` in `predictions` with shape `(224, 224, 3)`:
   - `predicted_mask = map_prediction_to_mask(prediction)`.
   - **`map_prediction_to_mask` implementation detail:** The outer loop variable is named `x` but it iterates **axis 0** of `prediction`; the inner loop variable is named `y` but it iterates **axis 1**. Effectively:
     - `predicted_mask[i, j] = np.argmax(prediction[i, j, :])`
     - So **`i` is the image row (vertical, top → bottom)** and **`j` is the column (horizontal, left → right)** — standard NumPy image indexing. The names `x,y` in the source are **misleading**; when reimplementing, use `i,j` or `row,col` to avoid swapping axes.
7. **Save visualization:** For each integer label map `predicted_result`:
   - `labeled_prediction_to_image(predicted_result)` → PIL `Image`:
     - `label_to_rgb(0)` → **0** (black) for background  
     - `label_to_rgb(2)` → **127** for content  
     - `label_to_rgb(1)` → **255** for border  
   - Saved as `./output/{index:03d}.jpg` with `index` from `0` upward in **prediction loop order** (not necessarily the same as lexicographic sort of filenames—see Section 7).

**Accuracy reporting (optional for splicing):** `compare_accuracy` compares `true_masks` to `labeled_predictions` with nested loops over `IMAGE_SIZE`, using the same `(x, y)` indexing convention.

---

## 7. File ordering footgun (reproducibility)

- **Test input order** comes from `tf.data.Dataset.list_files("./dataset/test/raw/*.jpg", shuffle=False)`. TensorFlow’s ordering is **not guaranteed** to match `sorted(os.listdir(...))`.
- **Output filenames** are `000.jpg`, `001.jpg`, … in **dataset iteration order**, while `generate_output_template()` pairs rows using `files_in_folder` which **sorts** names—those two orderings can **diverge**.

**Agent instruction:** If you need strict correspondence between a specific page file and an output mask, **do not** rely on index alone; either sort the file list the same way before building `tf.data`, or carry the **source path** through the pipeline and name outputs from that path.

---

## 8. Deriving panel “splices” (not in repo — required spec for full recreation)

The repository **stops** at a **224×224** label map. To obtain **rectangular crops** (or vector polygons) of panels on the **original** page, implement the following **deterministic** stages.

### Stage A — Label map in network space

You already have `L[y, x] ∈ {0, 1, 2}` for `y, x ∈ [0, 223]` from argmax logits.

### Stage B — (Optional) Upsample mask to full resolution before region extraction

Segmentation is at **224×224**. For cleaner crops on high-res scans, you may **nearest-neighbor** upsample `L` to the **padded intermediate** size then strip padding (Section 8.3). Alternatively, extract regions at 224×224 then map bounding boxes back—less precise on fine borders.

### Stage C — Region definition for “one panel”

Minimal approach consistent with the class definitions:

1. Take the binary mask **`M_content = (L == CONTENT_LABEL)`**.
2. Optionally dilate/erode if you need to close gaps (not in original project).
3. Run **connected components** on `M_content` (4- or 8-connectivity—**pick one and document it**; 8-connectivity is a common default).
4. Each connected component is **one logical reading region** (the training data may have merged adjacent micro-panels into one component on purpose).

**Borders:** The network also predicts `BORDER_LABEL`. A stricter “panel interior only” crop can use content; a “full panel including frame” crop might union content with adjacent border pixels per component—**that union rule is not defined in this repo**; choose explicitly.

### Stage D — Bounding box per component

For each component with pixel set `S` of `(row, col)` pairs matching **`predicted_mask[row, col]`**:

- `col_min = min(col for (row, col) in S)`, `col_max = max(col)`
- `row_min = min(row for (row, col) in S)`, `row_max = max(row)`
- Optional margin: expand the rectangle by `m` pixels, then clip rows/cols to `[0, 223]`.

These boxes live in **224×224 padded coordinates** (row = vertical axis, col = horizontal axis).

### Stage E — Map rectangle from 224×224 padded space to original image pixels

You must invert the same **`resize_with_pad`** used on the raw page.

Let original image width `W`, height `H`, target `T = 224`.

TensorFlow `resize_with_pad` logic (conceptually):

1. `scale = min(T / H, T / W)` (fit inside the square).
2. `new_h = round(H * scale)`, `new_w = round(W * scale)`.
3. Resize the image to `(new_h, new_w)` with the same interpolation as TF (bilinear for images; for **label maps** use **nearest** when resizing masks).
4. Pad to `T×T`:  
   `pad_top = floor((T - new_h) / 2)`, `pad_bottom = T - new_h - pad_top`,  
   `pad_left = floor((T - new_w) / 2)`, `pad_right = T - new_w - pad_left`.

**Inverse map** (a network coordinate `(row, col)` in the 224 tensor to original image `(Y, X)` in pixel space):

1. If `(row, col)` falls in a padded band (outside the resized content rectangle), that pixel **does not correspond** to the comic page—exclude or clamp.
2. Otherwise, subtract pads: `row' = row - pad_top`, `col' = col - pad_left`.
3. Map to original: `Y = row' / scale`, `X = col' / scale` (use consistent rounding: e.g. floor for min edge, ceil for max edge when converting a **box**).

Apply the inverse to the four corners of each bounding box from Stage D, then **clamp** to `[0, H-1]` (rows) and `[0, W-1]` (cols), then **`crop` the original RGB image** with PIL / TF / NumPy (height spans `Y`, width spans `X`).

### Stage F — Export

For each panel crop, save with a naming scheme tied to `(page_id, component_id)` and optionally store JSON sidecars with the **224-space** and **original-space** boxes for debugging.

---

## 9. Visualization-only output (do not confuse with splicing)

`labeled_prediction_to_image` produces a **single-channel** image:

- Background → `0`
- Content → `127`
- Border → `255`

Saved as **JPEG** under `./output/`. This is a **human-readable mask preview**, not individual panel files.

---

## 10. Checklist for a coding agent

- [ ] Load page RGB, apply **`tf.image.resize_with_pad` to 224×224**, divide by `255.0`, **NHWC**, `float32`.
- [ ] Run model, take **`argmax` over channel dimension** → `224×224` int labels `{0,1,2}`.
- [ ] If producing crops: implement **Section 8** inverse geometry; verify round-trip on synthetic rectangles.
- [ ] Do not assume **OpenCV contour** behavior from this repo—it is **not used**.
- [ ] Align mask authoring with **either** README RGB workflow **or** the exact `parse_image` grayscale remapping if using `decode_png(..., channels=1)`.
- [ ] Fix **file ordering** if pairing predictions to source files (Section 7).

---

## 11. File reference index

| File | Role |
|------|------|
| `utils.py` | `IMAGE_SIZE`, label constants, `parse_image`, `resize_with_pad`, train/test loaders, `map_prediction_to_mask`, `labeled_prediction_to_image`, accuracy loops |
| `DeepPanel.py` | Model definition, training, `model.save`, `model.export` |
| `DeepPanelTest.py` | Load `.keras`, predict, save `./output/*.jpg`, HTML report |
| `DeepPanelMobile.py` | TFLite export from SavedModel |
| `metrics.py` | Per-label metrics, `argmax` usage in metrics |
| `CheckSegmentationMasksQuality.py` | RGB mask color normalization helper |
| `WeightCalculator.py` | PIL-based class histogram on resized masks (analytics) |

---

*This specification was derived from the DeepPanel repository source at the time of writing. If you change `IMAGE_SIZE`, architecture, or preprocessing, update Sections 4–6 and 8 accordingly.*
