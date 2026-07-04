# TubeViT 實驗計畫 — B1 輕量版

驗證論文兩個 **架構 trend**（非絕對數字；論文 headline 是 Kinetics + ImageNet-1k，這裡用 UCF101 + Imagenette2-320 取近似）：

1. **Table 5** — joint image+video 訓練 > video-only
2. **Table 7e** — per-tube independent kernels > 共享 interpolated kernel

---

## 1. 前置條件

### 環境
- Python ≥ 3.12、[`uv`](https://docs.astral.sh/uv/)
- GPU：CUDA 強烈建議；Apple MPS 亦可（慢 3–5×）
- 磁碟：約 30 GB（資料 + logs + checkpoints）

### 安裝
```bash
uv sync
```

### 資料集佈局
令 `DATA_ROOT` 是資料根目錄，需要：
```
$DATA_ROOT/
├── UCF-101/                # torchvision.datasets.UCF101 格式
├── ucfTrainTestlist/
│   ├── trainlist01.txt
│   ├── testlist01.txt
│   └── classInd.txt
└── imagenette2-320/
    ├── train/              # ImageFolder 格式
    └── val/
```

### 預訓練權重
`scripts/train.py` hard-code 載入檔名 `tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt`。**兩種 kernel 格式不相容**，所以要準備兩份：

```bash
# (1) Run A/B 用 — independent-kernel 格式（keys: conv_proj_weights.{0,1,2,3}）
python scripts/convert_vit_weight.py \
  -o tubevit_b_independent.pt

# (2) Run C 用 — interpolated-kernel 格式（key: conv_proj_weight 單數，shape (768,3,8,8,8)）
python scripts/convert_vit_weight.py --interpolated-kernels \
  -o tubevit_b_interpolated.pt
```

每次 Run 之前需要把對應檔案 copy 到 hard-code 位置（詳見 §3 的每個 Run）：
```bash
cp tubevit_b_independent.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"
# 或
cp tubevit_b_interpolated.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"
```

> **為什麼**：Run A/B 用 `independent` 模式，checkpoint 必須含 `conv_proj_weights.0..3`；Run C 用 `interpolated` 模式，checkpoint 必須含單數 `conv_proj_weight`。若用錯檔 tokenizer 會 fallback 到 `.normal_()` 隨機初始化 → 跟有載入的 run 不公平比較。

### Metadata pickles
`ucf101-{train,val}-meta.pickle` 綁定在建立當下的 `dataset-root` / `frames-per-clip` / annotation split。**搬到新機器或改動這三者之一，先刪除讓它重建：**
```bash
rm -f ucf101-train-meta.pickle ucf101-val-meta.pickle
```

### 保留舊產出
每個 run 結束會覆寫 `models/tubevit_ucf101.ckpt`。若 repo 已有舊 ckpt，先備份：
```bash
mv models/tubevit_ucf101.ckpt models/_pre_exp_backup.ckpt 2>/dev/null || true
```

---

## 2. 共通超參數（全部 run 統一）

| 參數 | 值 | 備註 |
|---|---|---|
| `frames-per-clip` | 32 | MPS 可行；paper T=64（Kinetics），但僅 3 epoch trend 夠 |
| `step-between-clips` | **32** | **必填**。torchvision 預設 1 → UCF-101 產生 ~2M 重疊 clips（單 epoch 64k steps，單卡跑不完）；32 = 非重疊 clips，全資料集 ~71k clips、train split ~51k → ~1.6k steps/epoch @ b32 |
| `video-size` | 224×224 | default |
| `batch-size` | **CUDA: 32 / MPS: 8** | 視顯存調整 |
| `precision` | **CUDA: bf16-mixed / MPS: 32-true** | RTX 4080 16 GB：fp32 @ b32 會 OOM 或極緊；bf16-mixed 約 8–9 GB |
| `max-epochs` | 3 | 夠看相對趨勢 |
| `lr` | 5e-5 | paper Table 10 (ViT-B) |
| `weight-decay` | 0.001 | paper |
| `warmup-steps` | 200 | ~4% 訓練步數（paper 10 000 是 300 k 步 recipe，不適用） |
| `seed` | 42 | 單一 seed（B1 不量方差） |
| `num-workers` | 8 | 影片解碼是瓶頸（每 step 解 32×32=1024 幀）；4080 機器建議 ≥8 |

### RTX 4080 (16 GB) 專用注意事項

- 每個 run 命令加：`-s 32 -b 32 --precision bf16-mixed --num-workers 8`
- 若仍 OOM：`-b 16 --accumulate-grad-batches 2`（等效 batch 不變）
- `ucf101-*-meta.pickle` 內存的是相對路徑（`data/raw/UCF-101/...`）。若 4080 機器上 repo 佈局相同（從 repo root 執行、`-r data/raw/UCF-101`）可直接沿用；否則先刪除重建（首次掃描約 10–20 分鐘）
- `step-between-clips` 改變**不**需要刪 pickle（metadata 只綁 dataset root / 影片清單）

---

## 3. 三個 Run

> **注意**：下面範例以 RTX 4080（CUDA, 16 GB）為準：`-s 32 -b 32 --precision bf16-mixed`。MPS 改用 `-b 8 --precision 32-true`。

### Run A — baseline：video-only + independent kernels
對照組；同時是 Table 5 與 Table 7e 兩個對照的 baseline。
```bash
cp tubevit_b_independent.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"

python scripts/train.py \
  -r $DATA_ROOT/UCF-101 \
  -a $DATA_ROOT/ucfTrainTestlist \
  -f 32 -s 32 -b 32 --precision bf16-mixed \
  --max-epochs 3 \
  --lr 5e-5 --weight-decay 0.001 --warmup-steps 200 \
  --num-workers 8 \
  --seed 42

# 結束後：
mv models/tubevit_ucf101.ckpt models/run_a_baseline.ckpt
```
預期 TensorBoard 版本：`logs/TubeViT/version_{N}`，請記下 N。

### Run B — joint + independent kernels（測 Table 5）
新增 `--image-dataset-path`、`--image-num-classes 10`。Weight 檔同 Run A。
```bash
cp tubevit_b_independent.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"

python scripts/train.py \
  -r $DATA_ROOT/UCF-101 \
  -a $DATA_ROOT/ucfTrainTestlist \
  --image-dataset-path $DATA_ROOT/imagenette2-320 \
  --image-num-classes 10 \
  -f 32 -s 32 -b 32 --precision bf16-mixed \
  --max-epochs 3 \
  --lr 5e-5 --weight-decay 0.001 --warmup-steps 200 \
  --num-workers 8 \
  --seed 42

# 結束後：
mv models/tubevit_ucf101.ckpt models/run_b_joint.ckpt
```
**注意**：joint 用 `ConcatDataset(UCF, Imagenette)`，每 epoch 步數 ≈ (|UCF| + |Imagenette|) / batch — 單 epoch 時間約為 Run A 的 1.8–2.0×。

### Run C — video-only + interpolated kernels（測 Table 7e）
**關鍵**：要先把 weight 檔換成 interpolated 變體，否則 tokenizer 會隨機初始化。
```bash
cp tubevit_b_interpolated.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"

python scripts/train.py \
  -r $DATA_ROOT/UCF-101 \
  -a $DATA_ROOT/ucfTrainTestlist \
  -f 32 -s 32 -b 32 --precision bf16-mixed \
  --max-epochs 3 \
  --lr 5e-5 --weight-decay 0.001 --warmup-steps 200 \
  --interpolated-kernels \
  --num-workers 8 \
  --seed 42

# 結束後：
mv models/tubevit_ucf101.ckpt models/run_c_interpolated.ckpt
```

---

## 4. 監控：TensorBoard

```bash
tensorboard --logdir logs --bind_all --port 6006
# http://localhost:6006/
```

關心的 scalar（每個 run 一個 `version_*`）：
| Metric | 來源 |
|---|---|
| `train_loss`, `val_loss` | 所有 run |
| `train_acc`, `val_acc` | 所有 run |
| `train_f1`, `val_f1` | 所有 run |
| `train_img_loss`, `train_img_acc`, `train_img_f1` | 只有 Run B |
| `lr-*` | LearningRateMonitor callback |

比較方式：把 A / B / C 三條線疊在同一張 chart 上對視。

---

## 5. 評估（可選，不是 B1 的必需項）

若想要除 TB `val_acc` 之外的完整指標（top-5 / AUROC / F1 / confusion matrix）：
```bash
python scripts/evaluate.py \
  -r $DATA_ROOT/UCF-101 \
  -a $DATA_ROOT/ucfTrainTestlist \
  --label-path $DATA_ROOT/ucfTrainTestlist/classInd.txt \
  -m models/run_a_baseline.ckpt
```
對三個 ckpt 各跑一次。輸出 `output.png` 是 confusion matrix（每次會覆寫，記得改名）。

---

## 6. 驗證準則

### Table 5（Run B vs Run A）
| 結論 | 準則 |
|---|---|
| ✅ 通過 | `val_acc(B) − val_acc(A) ≥ 1.0 point`（epoch 3 末） |
| 🟡 部分通過 | 0 ≤ 差距 < 1.0 point（joint 至少不傷害） |
| ❌ 不通過 | B < A |

論文對應差距：K600 上 joint vs video-only ≈ +5.9 points（§5.7 Table 5）。Imagenette 只有 10 類、~9.5k 張，訊號會比 ImageNet-1k 弱很多；1 point 是相對寬鬆的 pass bar。

### Table 7e（Run C vs Run A）
| 結論 | 準則 |
|---|---|
| ✅ 通過 | `val_acc(A) − val_acc(C) ∈ [0.3, 2.0] points` |
| 🟡 部分通過 | C < A 但差距 < 0.3 |
| ❌ 不通過 | C ≥ A |

論文對應差距：K600 上 independent vs interpolated = 84.5 − 83.8 = 0.7 points（Table 7e）。

### 共通 sanity（所有 run 必須滿足）
- `train_loss` 單調下降（或至少 epoch 1 → 3 有下降）
- `val_acc` 在 epoch 3 末 > 30%（ImageNet-inflated ViT 預期 30–60%）
- 無 NaN / Inf；無 loss flat（若出現 → 降 lr）

---

## 7. 產出

寫到 repo 的：
- `logs/TubeViT/version_{N, N+1, N+2}/` — TB 事件檔
- `models/run_a_baseline.ckpt`、`run_b_joint.ckpt`、`run_c_interpolated.ckpt`
- `docs/EXP_RESULTS.md`（手動填寫）— 三個 run 的 final metric + 對 Table 5 / 7e 的結論

建議 `docs/EXP_RESULTS.md` 欄位：
```
| Run | TB version | final val_loss | final val_acc | final val_f1 | notes |
|-----|-----------|---------------|---------------|--------------|-------|
| A   |           |               |               |              |       |
| B   |           |               |               |              |       |
| C   |           |               |               |              |       |

Table 5 結論：...
Table 7e 結論：...
```

---

## 8. 時程估算

| 硬體 | 單 run (Run A/C) | Run B (joint) | 全程 |
|---|---|---|---|
| RTX 4080 (bf16, `-s 32`, ~1.6k steps/epoch) | ~25–45 min | ~45–80 min | **~1.5–3 h** |
| CUDA A100 / A10 | ~30–45 min | ~60–90 min | **~2.5–3 h** |
| CUDA T4 | ~60–90 min | ~120–180 min | **~5–6 h** |
| MPS (M1/M2/M3) | ~3–5 h | ~6–10 h | **~12–20 h** |

估算前提：`-s 32` 非重疊取樣。若把 `-s` 調小（更多 clips/epoch），時間線性放大；`-s 1`（torchvision 預設）是 ~40× 的量，**不要用**。實際速度以影片解碼吞吐為主，`--num-workers` 開足。

---

## 9. 已知限制

- 單一 seed — 無方差估計；若 A/B/C 差距在 1 point 內，本實驗無法斷言顯著性
- UCF101 ≠ Kinetics；Imagenette2-320 ≠ ImageNet-1k — 相對趨勢可比，絕對值不可比
- 3 epoch 是短訓練；若曲線還在陡峭上升 → 把 `--max-epochs` 拉到 5–8
- 所有 run 使用 single-crop inference（非 paper 的 4×3）
- Run A 和 Run B 起點相同（`tubevit_b_independent.pt`），Run C 起點是 `tubevit_b_interpolated.pt`。兩者 tube-0 kernel 完全一樣（都由 ViT-B patch-embed 同樣的 bilinear+temporal 展開產生）；Run A 的 tube-1..3 kernel 由 `convert_vit_weight.py` 預先三線性插值生成；Run C 在 forward 時才做同樣的三線性插值。因此 step 0 的 tokenizer 輸出實質等價，Run A/C 差異來自「獨立學 vs 共享學」的訓練動態，符合 Table 7e 的意圖。

---

## 10. Run 之前的檢查清單

- [ ] `uv sync` 跑過，`uv run python -c "import lightning, torchvision, click"` 沒報錯
- [ ] `$DATA_ROOT/UCF-101/`、`$DATA_ROOT/ucfTrainTestlist/`、`$DATA_ROOT/imagenette2-320/` 都存在
- [ ] `tubevit_b_independent.pt` 與 `tubevit_b_interpolated.pt` 都已產生於 repo root
- [ ] 舊 `models/tubevit_ucf101.ckpt` 已備份或可丟
- [ ] 若從別台機器搬過來，`ucf101-*-meta.pickle` 已刪除
- [ ] TensorBoard 已啟動，確認 http://localhost:6006 可達
- [ ] `nvidia-smi`（CUDA）或 `sudo powermetrics --samplers gpu_power`（MPS）確認 GPU 可見
