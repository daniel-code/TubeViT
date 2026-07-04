# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unofficial PyTorch implementation of TubeViT (["Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning"](https://arxiv.org/abs/2212.03229)). Built on `torch`, `torchvision`, `lightning` (PyTorch Lightning 2.x), `pytorchvideo`, `torchmetrics`, and `click` CLIs.

Python `>=3.12` (see `.python-version`, `pyproject.toml`). Dependencies are managed with `uv` (`uv.lock` committed); run `uv sync` to set up the env.

## Commands

```bash
# Install / sync env (authoritative)
uv sync

# Lint & format (pre-commit wires black, ruff, isort + hygiene hooks)
pre-commit run --all-files
ruff check .
black .
isort .

# Build the pre-trained TubeViT weight from torchvision ViT-B/16
# Writes tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt at repo root (this exact
# filename is what scripts/train.py hard-codes as its weight_path).
python scripts/convert_vit_weight.py

# Train on UCF101 (metadata is cached to ucf101-{train,val}-meta.pickle
# at repo root on first run; these ~24 MB pickles are committed).
python scripts/train.py -r <dataset-root> -a <annotation-path>
python scripts/train.py -r ... -a ... --fast-dev-run   # quick sanity run

# Evaluate a trained checkpoint (requires UCF101 classInd.txt via --label-path)
python scripts/evaluate.py -r <dataset-root> -a <annotation-path> \
    -m <checkpoint.ckpt> --label-path <classInd.txt>

# Single-video inference (note: filename is misspelled "infernce.py")
python scripts/infernce.py <video-path> -m <checkpoint.ckpt> --label-path <classInd.txt>

# TensorBoard
tensorboard --logdir logs        # training logger writes logs/TubeViT/...
```

There is no test suite.

## Architecture

All model code is in `tubevit/`; all user-facing entry points are `click` CLIs in `scripts/`.

### `tubevit/model.py` — the whole model

`TubeViT` composes four pieces, in order:

1. **`SparseTubesTokenizer`** — the core idea of the paper. **Default (`interpolated_kernels=False`, paper main results):** each tube has its **own independent 3D conv kernel** (`conv_proj_weights`, a `ParameterList` of 4 tensors with shapes matching each `kernel_sizes[i]`). **Ablation (`interpolated_kernels=True`, `--interpolated-kernels` CLI flag, paper Table 7e):** a **single** shared kernel (`conv_proj_weight`, singular, shape `(hidden_dim, 3, *kernel_sizes[0])`) is trilinearly resized to each tube's size at runtime. Each tube always has its own bias (`conv_proj_biases` ParameterList), its own `offsets[i]` (applied as a slice on CTHW input), and its own `strides[i]`. The four tube configs are hard-coded in `TubeViT.__init__` — editing them requires regenerating positional embeddings and is incompatible with existing checkpoints. Output of each tube is flattened and concatenated along the token dimension.

2. **Position embedding** — built once in `_generate_position_embedding` using 3D sincos encoding (`tubevit/positional_encoding.py`). Crucially, positions are computed in *physical input-video coordinates* (`stride*i + offset + kernel_size/2`), not per-tube token indices, so tokens from different tubes share a consistent 3D location space. Stored as a non-trainable `nn.Parameter` and added after a CLS token is prepended.

3. **Encoder** — a thin wrapper around a stack of `torchvision.models.vision_transformer.EncoderBlock`s. It intentionally omits position embedding (handled by `TubeViT` itself).

4. **Head** — `SelfAttentionPooling` over the entire sequence, then a linear head. Note: a CLS token is still prepended in `forward`, but the head pools over all tokens (including CLS) via attention rather than reading CLS directly. This differs from stock ViT.

`_calc_conv_shape` mirrors the Conv3d output-shape formula and is used only to size the positional embedding; keep it in sync with any tokenizer changes.

`TubeViTLightningModule` wraps `TubeViT` with Adam + `LambdaLR` (cosine decay + linear warmup; `warmup_steps` param, total steps from `trainer.estimated_stepping_batches`), label-smoothed cross-entropy, and logs `{train,val}_{loss,acc,f1}` each step. Passing `weight_path` loads a state dict with `strict=False` — this is how `train.py` picks up the inflated ViT weights.

### `tubevit/dataset.py`

Thin subclass `MyUCF101` of `torchvision.datasets.UCF101` that applies a `transform` over the clip and returns `(video, label)`, dropping the audio/info tuple. All scripts request `output_format="THWC"` from UCF101 and rely on `ToTensorVideo` in the transform to produce `CTHW`, which is what the tokenizer expects.

### `scripts/convert_vit_weight.py` — inflating ImageNet ViT-B/16 to TubeViT

Takes `torchvision.ViT_B_16_Weights.DEFAULT`, then for the patch-embedding conv: bilinearly resizes `conv_proj.weight` from `(768, 3, 16, 16)` to `(768, 3, 8, 8)`, unsqueezes a temporal dim, repeats it 8× along time, and divides by 8 — producing the `(768, 3, 8, 8, 8)` kernel that matches `kernel_sizes[0]`. The encoder's `pos_embedding` and `heads.head.*` are popped before `load_state_dict(strict=False)` because they don't match. The output filename pattern `tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt` encodes which tube configs were used and is what `train.py` expects.

### Metadata pickles

`ucf101-{train,val}-meta.pickle` are cached by `train.py`/`evaluate.py`/`visualise_dataset.py` at the repo root. They exist to avoid UCF101's slow first-time metadata scan; they are regenerated automatically if deleted. They are dataset-specific — invalidate them whenever `dataset-root`, `frames-per-clip`, or annotation splits change.

### Checkpoint format compatibility

The default `tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt` uses **independent-kernel format** (keys: `conv_proj_weights.{0..3}`). Running `train.py --interpolated-kernels` expects key `conv_proj_weight` (singular) — loading the wrong file silently random-inits the tokenizer (`strict=False` swallows the mismatch). Generate a separate compatible file first:

```bash
python scripts/convert_vit_weight.py --interpolated-kernels -o tubevit_b_interpolated.pt
```
