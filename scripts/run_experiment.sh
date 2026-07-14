#!/usr/bin/env bash
# EXP.md B2 experiment — Arm A, C, B sequentially (see EXP.md §3 for ordering).
# Launch: setsid bash scripts/run_experiment.sh >> logs/experiment.log 2>&1 &
# Paths are derived from this script's location; DATA_ROOT can be overridden:
#   DATA_ROOT=/some/where bash scripts/run_experiment.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/raw}"
LOG="$REPO_ROOT/logs/experiment.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

cd "$REPO_ROOT" || exit 1
mkdir -p logs models

run_train() {
    local run_name=$1; shift
    local ckpt_name=$1; shift
    local weight_src=$1; shift

    log "=== $run_name start ==="
    cp "$weight_src" "tubevit_b_(a+iv)+(d+v)+(e+iv)+(f+v).pt"

    PYTHONPATH=. uv run python scripts/train.py "$@" >> "$LOG" 2>&1
    local exit_code=$?

    if [[ -f models/tubevit_ucf101.ckpt ]]; then
        mv models/tubevit_ucf101.ckpt "models/$ckpt_name"
        log "=== $run_name done — checkpoint: models/$ckpt_name (exit=$exit_code) ==="
    else
        log "=== $run_name FAILED — no checkpoint saved (exit=$exit_code) ==="
    fi
}

# B2 unified recipe (EXP.md §2) — identical across all arms.
COMMON_ARGS=(
    -r "$DATA_ROOT/UCF-101"
    -a "$DATA_ROOT/ucfTrainTestlist"
    -f 32 -s 16 -b 8 --accumulate-grad-batches 32
    --precision bf16-mixed
    --max-epochs 8
    --lr 5e-5 --weight-decay 0.001 --warmup-steps 200
    --dropout 0.1 --attention-dropout 0.1 --label-smoothing 0.1
    --num-workers 2
    --seed 42
)

# Arm A — video-only + independent kernels (baseline)
run_train "Arm A" "b2_a_video_only.ckpt" "tubevit_b_independent.pt" "${COMMON_ARGS[@]}"

# Arm C — video-only + interpolated kernels (Table 7e)
run_train "Arm C" "b2_c_interpolated.ckpt" "tubevit_b_interpolated.pt" \
    "${COMMON_ARGS[@]}" \
    --interpolated-kernels

# Arm B — joint image+video + independent kernels (Table 5)
run_train "Arm B" "b2_b_joint.ckpt" "tubevit_b_independent.pt" \
    "${COMMON_ARGS[@]}" \
    --image-dataset-path "$DATA_ROOT/imagenette2-320" \
    --image-num-classes 10

log "=== ALL RUNS COMPLETE ==="
