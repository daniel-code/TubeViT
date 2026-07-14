# TubeViT 實驗計畫 — B2 乾淨 ablation

驗證論文兩個 **架構 trend**（非絕對數字；論文 headline 是 Kinetics + ImageNet-1k，這裡用 UCF101 + Imagenette2-320 取近似）：

1. **Table 5** — joint image+video 訓練 > video-only
2. **Table 7e** — per-tube independent kernels > 共享 interpolated kernel

> **歷史**：`logs/TubeViT/version_0..7` 為 B1 階段的 smoke test / 髒 run，用途是驗證機器與 code 可正常運行，**不作為結論依據**。其中 version_6（video-only）/ version_7（joint）給出髒訊號 joint ≈ +5.9 pt @ epoch 3，但兩者 dropout、LR schedule、早停政策皆不同。B2 在單一變因下重跑全部實驗。

## 0. B1 smoke test 學到的事（B2 設計依據）

| 發現 | B2 對策 |
|---|---|
| val_loss 與 val_acc 後期背離：以 val_loss 選 best ckpt 比末期 val_acc 低 2.4 pt（v7） | ModelCheckpoint 改 monitor `val_acc` (max) + `save_last` |
| early stopping (patience 5 × val_check 0.2 ≈ 1 epoch) 砍掉還在上升的 val_acc（v6） | 關閉早停，固定訓練預算 |
| dropout=0 + label_smoothing=0 時 train_acc→100%、val_loss 回升（v7 明顯過擬合） | 統一 dropout 0.1 + label smoothing 0.1 |
| max_epochs 影響 cosine 總長：v6 (max=20 早停) 實際恆定 LR，v7 (max=10) 衰減到 0 | 全 arm 統一 max_epochs 8、無早停 |
| 吞吐實測：video-only ~1.19 h/epoch、joint ~1.27 h/epoch（`-s 16`, b8×acc32, bf16） | 時程估算依此（§8） |
| val_acc 於 epoch 6–7 進入平原（v7 最終 37.6%、Imagenette 73.4%） | 8 epochs 足夠 |

---

## 1. 前置條件

### 環境
- Python ≥ 3.12、[`uv`](https://docs.astral.sh/uv/)；`uv sync`
- GPU：RTX 4080 16 GB（bf16-mixed）；磁碟約 30 GB

### 資料集佈局
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

### 預訓練權重（兩種 kernel 格式不相容，各準備一份）
```bash
# Arm A/B 用 — independent-kernel 格式（keys: conv_proj_weights.{0..3}）
python scripts/convert_vit_weight.py -o tubevit_b_independent.pt

# Arm C 用 — interpolated-kernel 格式（key: conv_proj_weight 單數）
    python scripts/convert_vit_weight.py --interpolated-kernels -o tubevit_b_interpolated.pt
```
每個 arm 開跑前把對應檔 copy 到 hard-code 位置 `tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt`（`scripts/run_experiment.sh` 已自動處理）。用錯檔案時 `strict=False` 會**靜默隨機初始化 tokenizer**，比較即失效。

### train.py 必要變更（B2 開跑前完成；見 §0 依據）
1. `ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True, ...)`
2. 新增 `--label-smoothing` CLI（default 0.0），傳入 `TubeViTLightningModule(label_smoothing=...)`

變更後以 `--fast-dev-run` 各過一次 video-only 與 joint。

### Metadata pickles
`ucf101-{train,val}-meta.pickle` 綁 `dataset-root` / `frames-per-clip` / annotation split。機器與 `-f 32` 未變則沿用；否則先刪除重建（首掃 10–20 分鐘）。`-s` 改變**不需**重建。

### 保留舊產出
```bash
mv models/tubevit_ucf101.ckpt models/_pre_b2_backup.ckpt 2>/dev/null || true
```

---

## 2. 統一 recipe（三個 arm 完全相同）

| 參數 | 值 | 備註 |
|---|---|---|
| `-f` frames-per-clip | 32 | pickle 不需重建 |
| `-s` step-between-clips | 16 | ~382 video optimizer steps/epoch |
| `-b` / `--accumulate-grad-batches` | 8 / 32 | effective batch 256（16 GB 可容） |
| `--precision` | bf16-mixed | |
| `--max-epochs` | 8 | 統一 cosine 總長；無早停 |
| `--lr` / `--weight-decay` / `--warmup-steps` | 5e-5 / 0.001 / 200 | paper recipe（warmup ≈ 6% 步數） |
| `--dropout` / `--attention-dropout` | 0.1 / 0.1 | |
| `--label-smoothing` | 0.1 | 新 CLI |
| early stopping | 關閉（不傳 flag） | |
| `--num-workers` | 2 | 依訓練機器實測固定為 2 |
| `--seed` | 42（見 §5） | |

---

## 3. 三個 arm

唯一差異欄位加粗。建議順序 **A → C → B**（先收 Table 7e 的兩個 arm，讓 seed 加跑決策儘早出來）。

| Arm | 資料 | Tokenizer | 權重檔 | ckpt 保存名 |
|---|---|---|---|---|
| A（baseline） | video-only | independent | `tubevit_b_independent.pt` | `models/b2_a_video_only.ckpt` |
| B（Table 5） | **joint（+ Imagenette）** | independent | `tubevit_b_independent.pt` | `models/b2_b_joint.ckpt` |
| C（Table 7e） | video-only | **interpolated** | `tubevit_b_interpolated.pt` | `models/b2_c_interpolated.ckpt` |

一鍵執行（依序 A → C → B，含權重檔切換與 ckpt 改名）：
```bash
setsid bash scripts/run_experiment.sh >> logs/experiment.log 2>&1 &
```

或手動單跑（以 Arm B 為例）：
```bash
cp tubevit_b_independent.pt "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"
uv run python scripts/train.py \
  -r $DATA_ROOT/UCF-101 -a $DATA_ROOT/ucfTrainTestlist \
  -f 32 -s 16 -b 8 --accumulate-grad-batches 32 --precision bf16-mixed \
  --max-epochs 8 --lr 5e-5 --weight-decay 0.001 --warmup-steps 200 \
  --dropout 0.1 --attention-dropout 0.1 --label-smoothing 0.1 \
  --num-workers 2 --seed 42 \
  --image-dataset-path $DATA_ROOT/imagenette2-320 --image-num-classes 10
mv models/tubevit_ucf101.ckpt models/b2_b_joint.ckpt
```

每個 arm 開跑後記下 TensorBoard `logs/TubeViT/version_N`。

---

## 4. 混淆因子帳本

**已控制（arm 間完全相同）**：dropout、label smoothing、max_epochs / LR schedule、effective batch、資料取樣、augmentation、seed、checkpoint 與早停政策、硬體。

**Treatment 固有、不視為混淆（記錄即可）**：
- **B 每 epoch 多 ~37 個 image optimizer steps**（382→419，+9.7%），cosine 總長隨之拉長 → 同一 video step 上 B 的 LR 略高。這是 joint training 方法的一部分（paper 亦然）；比較基準為「相同 video epochs」。
- **C 的初始權重檔不同**，但 step-0 tokenizer 輸出與 A 實質等價：tube-0 kernel 相同；A 的 tube-1..3 為離線三線性插值、C 為 forward 時插值。A/C 差異純來自「獨立學 vs 共享學」的訓練動態。
- **joint loss = video + image 以 1:1 相加**（無權重超參）；`val_loss`/`val_acc` 只計 video 分支，兩種 arm 可直接比。

---

## 5. Seed 策略（分層，省算力）

- **第一輪**：A/B/C 各跑 seed 42（~29 h）。
- **A vs B（Table 5）**：B1 髒訊號 ≈ +6 pt，遠大於預期 seed noise（±0.5–1 pt）；第一輪差距 ≥ 3 pt 即可單 seed 下結論。
- **A vs C（Table 7e）**：論文差距僅 0.7 pt，與 seed noise 同量級。**若第一輪 |A−C| < 1.5 pt，對 A、C 加跑 seed 43、44**（+4 runs ≈ 38 h），以 3-seed mean ± range 判定；否則不加跑。

---

## 6. 監控與評估

### TensorBoard
```bash
tensorboard --logdir logs --bind_all --port 6006
```
關心的 scalar：`{train,val}_{loss,acc,f1}`（所有 arm）、`{train,val}_img_{loss,acc,f1}`（只有 B）、`lr-Adam`。把 A/B/C 疊同圖對視。

### 主指標
UCF101 `val_acc`，每個 arm 報三個數：
1. **best val_acc**（整段最大值 = best ckpt）
2. **final val_acc**（epoch 8 末）
3. epoch 3 / 5 / 8 對齊值（比收斂速度）

### 判定準則

**Table 5（B vs A，best val_acc）**
| 結論 | 準則 |
|---|---|
| ✅ 通過 | B − A ≥ 1.0 pt |
| 🟡 部分通過 | 0 ≤ 差距 < 1.0 pt |
| ❌ 不通過 | B < A |

**Table 7e（A vs C，best val_acc；多 seed 時用 mean）**
| 結論 | 準則 |
|---|---|
| ✅ 通過 | A − C ∈ [0.3, 2.0] pt 且方向在所有 seed 一致 |
| 🟡 部分通過 | C < A 但差距 < 0.3，或 seed 間方向不一致 |
| ❌ 不通過 | C ≥ A |

**Sanity（所有 arm）**：train_loss 下降、無 NaN/Inf；val_acc @ep8 ≥ 30%；B 的 `val_img_acc` ≥ 70%（v7 實測 73.4%）。

### 完整評估（可選）
對三顆 best ckpt 各跑 `scripts/evaluate.py`（top-5 / AUROC / F1 / confusion matrix；`output.png` 每次覆寫記得改名）：
```bash
python scripts/evaluate.py -r $DATA_ROOT/UCF-101 -a $DATA_ROOT/ucfTrainTestlist \
  --label-path $DATA_ROOT/ucfTrainTestlist/classInd.txt -m models/b2_a_video_only.ckpt
```

---

## 7. 產出

- `logs/TubeViT/version_{N..}` — TB 事件檔
- `models/b2_{a_video_only,b_joint,c_interpolated}.ckpt`
- `docs/EXP2_RESULTS.md`（手動填寫）：
```
| Arm | seed | TB ver | best val_acc | final val_acc | val_acc@ep3 | notes |
|-----|------|--------|--------------|---------------|-------------|-------|
| A   | 42   |        |              |               |             |       |
| B   | 42   |        |              |               |             |       |
| C   | 42   |        |              |               |             |       |

Table 5 結論：...
Table 7e 結論：...
```

---

## 8. 時程估算（RTX 4080，B1 實測外推）

| Arm | epochs | 估時 |
|---|---|---|
| A | 8 | ~9.5 h |
| C | 8 | ~9.5 h |
| B | 8 | ~10.2 h |
| **第一輪合計** | | **~29 h** |
| （條件觸發）A/C × seed 43,44 | 32 | +38 h |

---

## 9. 已知限制

- UCF101 ≠ Kinetics、Imagenette2-320 ≠ ImageNet-1k：只驗 trend，絕對值不可比 paper。
- 8 epoch 仍屬短訓練；single-crop val（非 paper 4×3）。
- Table 7e 即使 3 seeds，0.3–0.7 pt 級別差距仍在統計解析度邊緣，結論以「方向一致性」為主。
- B1 另一觀察：v7 的 Imagenette val 只到 73.4%（pretrained ViT-B 理應 >95%），暗示影像走 tube tokenizer 的路徑未完全承接預訓練特徵——非 B2 範圍，留待後續。

---

## 10. Run 之前的檢查清單

- [ ] §1 的 train.py 變更完成（checkpoint monitor `val_acc` + `--label-smoothing`）
- [ ] `--fast-dev-run` 過一次 video-only 與 joint
- [ ] `uv sync` 跑過；`tubevit_b_independent.pt`、`tubevit_b_interpolated.pt` 都在 repo root
- [ ] `$DATA_ROOT/UCF-101/`、`ucfTrainTestlist/`、`imagenette2-320/` 都存在
- [ ] 舊 `models/tubevit_ucf101.ckpt` 已備份或可丟
- [ ] 機器/資料路徑未變 → `ucf101-*-meta.pickle` 沿用；否則刪除重建
- [ ] TensorBoard 已啟動；`nvidia-smi` 確認 GPU 可見
