# MODEL.md

Paper implementation spec and reproduction targets for this repo, distilled from
`assets/tubevit.md` (["Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning"](https://arxiv.org/abs/2212.03229),
Piergiovanni, Kuo, Angelova — CVPR 2023).

This document is the authoritative reference for **what the paper specifies** and **how far the current code is from
reproducing it**. Claims cite the paper section / table so they can be checked against `assets/tubevit.md`.

---

## 1. Model overview

TubeViT converts a standard ViT into a joint image+video model by tokenizing the input with a small set of
sparsely-strided 3D convolutional "tubes" of different shapes. The resulting tokens are concatenated and fed —
unchanged — to a standard ViT encoder. No factorized attention, no video-specific transformer blocks.

Key ideas:

| Idea                                                                                                                                           | Where it shows up                       |
|------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| Multiple tube shapes (multi-view sampling)                                                                                                     | §3.4 — "Multi-Tube"; Table 14           |
| Single learnable 3D kernel shared across tubes via tri-linear interpolation                                                                    | §3.4 — "Interpolated Kernels"; Table 7e |
| Fixed 3D sin/cos position embedding in *physical video coordinates* (stride + offset + kernel center)                                          | §3.3; Eqs. 3–7; Table 7a                |
| Joint image+video training using the same tokenizer/backbone                                                                                   | §3.5; Table 5                           |
| Image-to-video scaling: reuse tubes from a small jointly-trained model to upgrade a larger image-only ViT with a frozen trunk + gated residual | §3.6; Eq. 8; Tables 6, 9                |

---

## 2. Input / output

### Input

- Shape: `(N, C, T, H, W)` — `CTHW` after `ToTensorVideo`.
- Channels: `C = 3` (RGB).
- Spatial: `H = W = 224` (standard ViT resolution).
- Temporal: `T = 64` for Kinetics-400 / 600 / 700, `T = 128` for Charades (30 s clips), `T = 32` for Something-Something
  V2. See Table 10.
- Sampling rate: 15 FPS (Kinetics), 6 FPS (Charades), 24 FPS (SSv2). Table 10.
- Normalization: ImageNet mean/std (`[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`) — matches convention; paper does
  not state explicitly.
- Augmentation (training): random spatial + temporal crops, `RandAugment` with `num_layers=2, magnitude=10`. SSv2 and
  Charades additionally use Mixup (0.3), Dropout (0.2–0.3), Label Smoothing (0.1–0.3). See §A, Table 10.

### Output

- Logits: `(N, num_classes)` — raw scores, apply softmax externally for probabilities.
- Readout: **attention pooling** over the full token sequence + FC head (Fig. 1 right block). The paper's Fig. 1 shows "
  Attention Pooling + FC"; the standard CLS-token readout is *not* used.
- Inference aggregation: multi-crop — `4 × 3` (4 temporal × 3 spatial) by default; ablation in Table 7f shows `1 × 1`
  already gets within ~2 points of `10 × 10`.

### Dataset `num_classes`

| Dataset                        | Classes                                                                                |
|--------------------------------|----------------------------------------------------------------------------------------|
| ImageNet-1k                    | 1000                                                                                   |
| Kinetics-400                   | 400                                                                                    |
| Kinetics-600                   | 600                                                                                    |
| Kinetics-700                   | 700                                                                                    |
| Something-Something V2         | 174                                                                                    |
| Charades                       | 157 (multi-label, sigmoid + mAP, not softmax)                                          |
| UCF101 *(this repo's default)* | 101 — **not reported in the paper**; included here because the codebase defaults to it |

---

## 3. Architecture specification

### 3.1 Sparse Tube Tokenizer

A ViT-style 3D patch embedding, but with `N_tubes` conv3d operations with *different* `(kernel, stride, offset)` triples
applied to the same CTHW input. Each tube's output is flattened `(N, hidden_dim, L_i)` then concatenated along the token
axis.

**Main-results configuration (paper §4.1, used for all headline K-400/600/700 numbers):**

| # | Kernel `(T, H, W)` | Stride `(T, H, W)` | Offset `(T, H, W)` |
|---|--------------------|--------------------|--------------------|
| 1 | `8 × 8 × 8`        | `(16, 32, 32)`     | `(0, 0, 0)`        |
| 2 | `16 × 4 × 4`       | `(6, 32, 32)`      | `(4, 8, 8)`        |
| 3 | `4 × 12 × 12`      | `(16, 32, 32)`     | `(0, 16, 16)`      |
| 4 | `1 × 16 × 16`      | `(32, 16, 16)`     | `(0, 0, 0)`        |

Paper states this yields **559 tokens** for a `32 × 224 × 224` clip — this is an illustrative example; headline Kinetics
results use `T = 64` (Table 10), producing ~1176 tokens.

**Interpolated Kernels (§3.4, Table 7e).** The paper version of the main model uses one learnable 3D kernel of shape
`(hidden_dim, 3, 8, 8, 8)` and produces the others via tri-linear interpolation. Table 7e shows this config reaches 83.8
on K600 vs 84.5 for independent per-tube kernels — a small drop for a large parameter saving.

**Space-to-Depth (§3.4, Table 7c).** Optional. Replace a tube's channel dim `d → d/S`, then concatenate `S` tokens along
the channel axis → same #tokens and final dim, larger effective kernel, no extra params. Table 12 shows the reference
per-tube S2D setting (tube 1: `2× temporal`, tube 2: `4× spatial`, tubes 3–4: none). Each tube's feature dim in Table 12
is `{512, 256, 576, 256}`, implying the raw conv output dim differs per tube and is concatenated to match `hidden_dim`
after S2D.

### 3.2 Positional embedding (§3.3, Eqs. 3–7)

Fixed 3D sin/cos embedding evaluated at the **physical voxel center** of each token, not a per-tube token index:

- For a token produced at tube index `(t, x, y)` by tube `i` with stride `s_i` and offset `o_i` and kernel `k_i`, the
  physical coordinate used is `(t * s_{i,t} + o_{i,t} + k_{i,t}/2, ...)`.
- `ω_j = 1 / (10000^j)` for `j = 0..d/6`, with a sin/cos pair encoded per axis → `6 * (d/6) = d`. (Paper Eqs. 3–7 split
  `d` into 3 axes × 2 functions; `d` must be divisible by 6.)
- A CLS-token-style zero row is prepended if the readout uses a CLS token (optional — the paper uses attention pooling).
- **Critical**: because coordinates are physical, tokens from different tubes that sample overlapping regions share a
  consistent encoding. Table 7a: this beats "Learned" (+5.3 K600), "Relative" (+7.0), and "Fixed cosine without
  stride/offset" (+6.8).

### 3.3 Encoder

Standard ViT encoder stack (Eqs. 1–2), no factorized attention. Backbone sizes (Table 11):

| Model         | Layers | Hidden |  MLP | Heads | Params |
|---------------|-------:|-------:|-----:|------:|-------:|
| ViT-Base (B)  |     12 |    768 | 3072 |    12 |   86 M |
| ViT-Large (L) |     24 |   1024 | 4096 |    16 |  307 M |
| ViT-Huge (H)  |     32 |   1280 | 5120 |    16 |  632 M |

Per-tube parameter cost is small (1–3 M total, a few percent of backbone). Table 12 shows per-tube feature dims that add
up to fit the backbone's hidden size through S2D.

### 3.4 Head

Attention pooling (Fig. 1) → linear classifier. For joint image+video training (§A, "Joint ImageNet and Kinetics
Training") a **separate FC head per dataset** is used, each computing cross-entropy on its own labels.

### 3.5 Image-to-Video scaling (§3.6, Eq. 8)

Procedure to build TubeViT-H without full training:

1. Jointly train a *small* model (ViT-Tiny / ViT-Base) on images + videos → this produces the tube weights.
2. Take a *large* image-pretrained ViT (ViT-H). Attach the tubes from step 1.
3. Freeze layers `0..s−1` (paper uses `s = 26` of 32 for ViT-H, Table 6).
4. Insert a gated residual at layer `s` with raw tube tokens `z_0`:
   `z_s = MLP(LN(y_s)) + y_s + tanh(α) * z_0`, with scalar `α` initialized to 0 so the ViT is unchanged at step 0.
5. Finetune only the non-frozen layers.

Table 6 (K600): Last-FC-only 85.6 → +last-4 86.3 → +last-8 86.8 → +last-8+gated 89.7 vs full-finetune 91.8. Training
time cut by ~43%.

---

## 4. Training recipe (Table 10)

| Hyperparameter  | K400 / K600 / K700           | Charades | SSv2    |
|-----------------|------------------------------|----------|---------|
| Optimizer       | Adam                         | Adam     | Adam    |
| Batch size      | 256                          | 64       | 256     |
| LR schedule     | cosine decay + linear warmup | same     | same    |
| Warmup steps    | 10,000                       | 10,000   | 10,000  |
| Base LR         | 5e-5 (B) / 1e-5 (L, H)       | 1e-3     | 2e-5    |
| Total steps     | 300,000                      | 300,000  | 300,000 |
| Weight decay    | 0.001 (B) / 1e-5 (L, H)      | same     | same    |
| RandAugment     | N=2, M=10                    | same     | same    |
| Mixup           | —                            | —        | 0.3     |
| Dropout         | —                            | 0.2      | 0.3     |
| Label smoothing | —                            | 0.1      | 0.3     |
| Frames `T`      | 64                           | 128      | 32      |
| FPS             | 15                           | 6        | 24      |

**Stability notes (§A).** ViT-L and ViT-H were prone to training collapse (loss flat, acc → 0); fix was to lower weight
decay **and** LR. Charades required Mixup/Dropout/Label-Smoothing added.

**Joint ImageNet + Kinetics.** Use one shared backbone with dataset-specific FC heads, compute loss per batch against
the relevant head, backprop. No gradient balancing reported.

---

## 5. Benchmark results (targets for reproduction)

All numbers are Top-1 / Top-5 accuracy unless noted. Pre-training column matches what the paper reports for its own
entries.

### 5.1 Kinetics-400 (Table 1)

| Model                          | Pre-train   | Top-1 | Top-5 | Crops | TFLOPs |
|--------------------------------|-------------|------:|------:|-------|-------:|
| TubeViT-B                      | ImageNet-1k |  88.6 |  97.6 | 4×3   |   0.87 |
| TubeViT-L                      | ImageNet-1k |  90.2 |  98.6 | 4×3   |   9.53 |
| TubeViT-H *(created via §3.6)* | ImageNet-1k |  90.9 |  98.9 | 4×3   |  17.64 |

Best prior non-TubeViT at comparable pre-training: MTV-B (82.4, ImageNet-21k). TubeViT-H beats MTV-H (89.9) trained on
WTS280p while using ~4× fewer FLOPs.

### 5.2 Kinetics-600 (Table 2)

| Model                 | Pre-train   | Top-1 | Top-5 |
|-----------------------|-------------|------:|------:|
| TubeViT-B             | ImageNet-1k |  90.9 |  97.3 |
| TubeViT-L             | ImageNet-1k |  91.5 |  98.7 |
| TubeViT-H *(created)* | ImageNet-1k |  91.8 |  98.9 |

### 5.3 Kinetics-700 (Table 3)

| Model     | Pre-train   | Top-1 | Top-5 |
|-----------|-------------|------:|------:|
| TubeViT-L | ImageNet-1k |  83.8 |  96.6 |

### 5.4 Something-Something V2 (Table 4)

| Model     | Pre-train          | Top-1 | Top-5 |
|-----------|--------------------|------:|------:|
| TubeViT-L | ImageNet-1k + K600 |  76.1 |  95.2 |

### 5.5 Charades (Table 13, multi-label mAP)

Charades requires the larger tube shapes listed in §B (`1×16×16`, `16×16×16`, `32×8×8`, `4×32×32`) — longer temporal
extent to cover 30 s clips.

| Model                  |  mAP |
|------------------------|-----:|
| MultiTube TubeViT-L    | 61.8 |
| Interpolated TubeViT-L | 66.2 |

### 5.6 ImageNet-1k sanity check (§4.1 narrative)

| Setting                                      | Top-1 |
|----------------------------------------------|------:|
| TubeViT-B, ImageNet only                     |  78.1 |
| TubeViT-B, jointly trained with Kinetics-600 |  81.4 |

### 5.7 Co-training ablation (Table 5, K600, ViT-L)

| Training schedule                        | K600 Top-1 |
|------------------------------------------|-----------:|
| Kinetics only                            |       85.6 |
| ImageNet → Kinetics (two-stage)          |       90.4 |
| ImageNet + Kinetics (joint)              |   **91.5** |
| 2D patches only, ImageNet + Kinetics     |       87.6 |
| Inflated 3D patches, ImageNet → Kinetics |       88.4 |

### 5.8 Tube-shape ablation (Table 14, K600, 50k steps, ViT-B)

Shapes `(a)..(h)` and strides `(i)..(v)` defined in §C.

| Config                      |     K600 |
|-----------------------------|---------:|
| `(a+iv)+(b+v)+(f+iv)`       |     87.9 |
| `(c+iv)+(e+v)+(g+iv)`       |     87.5 |
| `(a+iv)+(e+v)+(g+iv)`       |     87.8 |
| `(b+iv)+(e+v)+(g+iv)`       |     87.7 |
| `(a+iv)+(d+v)+(e+iv)+(h+v)` |     88.6 |
| `(a+iv)+(b+v)+(c+iv)+(h+v)` |     87.9 |
| `(a+iv)+(d+v)+(e+iv)+(f+v)` | **88.9** |

Note the name of the checkpoint filename this repo ships (`tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt`) refers to the last
row here. But see §6.

---

## 6. Gaps between this repo and the paper

These are the items a future reproduction effort must close. Line numbers refer to the current `dev` branch.

| #  | Gap                                  | Current                                                                                                | Paper                                                                                                                                       | Location                                          |
|----|--------------------------------------|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| 1  | Backbone pre-training dataset        | ImageNet-1k via `torchvision.ViT_B_16_Weights.DEFAULT` (inflated)                                      | ImageNet-1k joint with Kinetics                                                                                                             | `scripts/convert_vit_weight.py`                   |
| 2  | Co-training                          | single UCF101 loader                                                                                   | joint ImageNet + Kinetics with separate FC heads                                                                                            | `scripts/train.py`                                |
| 3  | Frames per clip                      | 32 (default CLI)                                                                                       | 64 (Kinetics) / 128 (Charades) / 32 (SSv2)                                                                                                  | `scripts/train.py:22`                             |
| 4  | ~~Readout~~                          | ~~CLS token prepended but unused~~ → **fixed**: CLS token removed; `SelfAttentionPooling` over tube tokens only | attention pooling; paper does not prepend a CLS token                                                                                 | `tubevit/model.py` ✓ **closed**                   |
| 5  | ~~Interpolated kernels~~             | ~~enabled unconditionally~~ → **fixed**: `interpolated_kernels=False` (independent, default) or `True` (shared + interpolated) selectable via CLI `--interpolated-kernels` | main results use independent per-tube kernels (Table 7e); interpolated is the paper ablation | `tubevit/model.py`, all scripts ✓ **closed**      |
| 6  | ~~Space-to-Depth~~                   | ~~not implemented~~ → **fixed**: `_apply_s2d` + per-tube `s2d_factors`; `SparseTubesTokenizer` uses reduced conv-out channels (`hidden_dim // (tf × sf²)`) then folds back; position embedding uses effective stride/offset | used in Table 12 reference config and §3.6 scaling | `tubevit/model.py` ✓ **closed**                   |
| 7  | Tube/stride config matching filename | hard-coded to the §4.1 main-results config (stride `(16,32,32)` for 8×8×8 tube, etc.)                  | the filename `(a+iv)+(d+v)+(e+iv)+(f+v)` implies Table 14's best ablation config, which uses strides `(iv)=(16,16,16)` and `(v)=(32,32,32)` | `tubevit/model.py:143-162`                        |
| 8  | ~~LR schedule~~                      | ~~`OneCycleLR` with `max_lr = lr`~~ → **fixed**: `LambdaLR` with cosine decay + linear warmup; total steps from `self.trainer.estimated_stepping_batches`; `warmup_steps` param added (paper default: 10 000) | cosine decay with 10k linear warmup                                                                                            | `tubevit/model.py` ✓ **closed**                   |
| 9  | Hyperparameters                      | Adam, lr=1e-4, wd=1e-3, 10 epochs                                                                      | Adam, lr=5e-5 (B), wd=1e-3, 300k steps                                                                                                      | `scripts/train.py:145-149`                        |
| 10 | Dataset                              | UCF101 (not in paper)                                                                                  | K-400/600/700, SSv2, Charades, ImageNet                                                                                                     | `tubevit/dataset.py`                              |
| 11 | Multi-crop inference                 | single crop (`predict_step` runs model once per clip)                                                  | 4×3 default, 10×10 for upper bound                                                                                                          | `tubevit/model.py:327-332`, `scripts/infernce.py` |
| 12 | ~~Position embedding dim assertion~~ | ~~`embed_dim % 4 == 0`~~ → **fixed**: `embed_dim % 6 == 0`                                             | paper Eq. 7 requires `embed_dim % 6 == 0` (3 axes × 2 sin/cos)                                                                             | `tubevit/positional_encoding.py:27` ✓ **closed** |
| 13 | Image-to-Video scaling (§3.6)        | not implemented                                                                                        | gated residual + partial freeze recipe for H-model creation                                                                                 | —                                                 |

**Minor — 559 vs 539 token mismatch (verified 2026-04-21 via author email):** the paper reports **559 tokens** for a
`32×224×224` clip under the §4.1 four-tube config; the author confirmed this number directly. This repo's
`_calc_conv_shape` (standard `Conv3d` floor) + `SparseTubesTokenizer.forward` produces **539** (540 with CLS). The
20-token gap is **not** a standard padding/ceil convention — `PyTorch VALID floor`, `TF VALID ceil`, `TF SAME`, and
`PyTorch ceil_mode=True` (with/without offset-as-slice) were all tried and none yields 559. Likely causes (
unresolved): (a) Space-to-Depth counting differs — Table 12 assigns S2D `2× temporal` to tube 0 and `4× spatial` to tube
1, and the author may be counting an S2D-expanded intermediate grid rather than the post-pack token count; (b) minor
tokenizer config drift from the §4.1 text (stride/kernel off-by-one, or an extra 2D-patch-with-temporal-stride branch in
the real TF code); (c) extra modality/register/pad tokens injected before the encoder. **The author's official reference
implementation is in TensorFlow, and was not open-sourced at paper publication time** — any bit-exact reproduction or
checkpoint-loading work should request the tokenizer snippet / tube output shapes from the authors directly rather than
reverse-engineering from paper text. 539 vs 559 does **not** invalidate training correctness in this PyTorch repo.

---

## 7. Reproduction checklist

Minimum bar to claim "matching the paper":

- [ ] Train TubeViT-B jointly on ImageNet-1k + Kinetics-400 with the Table 10 schedule.
- [ ] Evaluate with 4×3 crops on the K400 validation set; report Top-1 / Top-5.
- [ ] **Target:** Top-1 ≥ 88.6, Top-5 ≥ 97.6 (Table 1 TubeViT-B line).
- [ ] Implement the joint dataloader (alternating or concatenated batches) with per-dataset FC heads and verify the
  ImageNet-only vs joint gap reported in §5.6 (78.1 → 81.4).
- [ ] Optional: reproduce Table 7a position-embedding ablation — a sanity check that the physical-coordinate sincos
  embedding is wired correctly.

Stretch:

- [ ] Add K600 + K700 heads and reproduce §5.2 / §5.3.
- [ ] Implement §3.6 scaling and reproduce TubeViT-H 90.9 / 91.8 on K400 / K600.
- [ ] Charades mAP 66.2 with the longer-tube, interpolated-kernel variant from §B.

---

## 8. Where things live in the paper

Mapping for fast lookup in `assets/tubevit.md`:

- Tokenizer design → §3.2 (Sparse Video Tubes), §3.4 (tube construction variants)
- Positional embedding → §3.3, Eqs. 3–7
- Joint training → §3.5, §A ("Joint ImageNet and Kinetics Training")
- Image-to-Video scaling → §3.6, Eq. 8, Tables 6 & 9
- Main numeric results → Tables 1, 2, 3, 4, 13
- Ablations → Table 7 (position embed, number of tubes, S2D, eval tokens, interpolated kernel, multi-crop), Table 8 (
  factorized-attention failure mode), Table 14 (tube-shape ablation)
- Hyperparameters → Table 10
- Tube reference config → Table 12
