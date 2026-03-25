#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
ROOT_PATH="${ROOT_PATH:-./dataset/slopemine}"
DATA_PATH="${DATA_PATH:-slopemine_aggregate.csv}"
FEATURES="${FEATURES:-MS}"
TARGET="${TARGET:-deformation}"
ENC_IN="${ENC_IN:-7}"
DEC_IN="${DEC_IN:-7}"
C_OUT="${C_OUT:-7}"
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-96}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-0}"
ITR="${ITR:-1}"
CHECKPOINTS="${CHECKPOINTS:-./checkpoints}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Example:
#   PRED_LENS="1" bash scripts/slopemine/run_slopemine_baselines.sh
#   PRED_LENS="1 3 6 12" bash scripts/slopemine/run_slopemine_baselines.sh
read -r -a PRED_LENS <<< "${PRED_LENS:-1}"

COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --data custom
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --features "$FEATURES"
  --target "$TARGET"
  --freq h
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --enc_in "$ENC_IN"
  --dec_in "$DEC_IN"
  --c_out "$C_OUT"
  --train_epochs "$TRAIN_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --num_workers "$NUM_WORKERS"
  --itr "$ITR"
  --checkpoints "$CHECKPOINTS"
  --des slopemine
)

run_model() {
  local model="$1"
  local pred_len="$2"
  shift 2

  mkdir -p "$MPLCONFIGDIR"
  # shellcheck disable=SC2086
  MPLCONFIGDIR="$MPLCONFIGDIR" "$PYTHON_BIN" -u run.py \
    "${COMMON_ARGS[@]}" \
    --model "$model" \
    --model_id "slopemine_${model}_sl${SEQ_LEN}_pl${pred_len}" \
    --pred_len "$pred_len" \
    "$@" \
    $EXTRA_ARGS
}

for pred_len in "${PRED_LENS[@]}"; do
  run_model DLinear "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 64 \
    --d_ff 128 \
    --moving_avg 25 \
    --dropout 0.1

  run_model PatchTST "$pred_len" \
    --e_layers 1 \
    --d_layers 1 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads 4 \
    --dropout 0.1 \
    --patch_len 16 \
    --stride 8

  run_model TimeFilter "$pred_len" \
    --e_layers 1 \
    --d_layers 1 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads 4 \
    --dropout 0.1 \
    --patch_len 24 \
    --alpha 0.1 \
    --top_p 0.5 \
    --pos 1

  run_model MICN "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --d_model 64 \
    --d_ff 128 \
    --n_heads 4 \
    --dropout 0.1

  run_model LSTM "$pred_len" \
    --e_layers 2 \
    --d_model 64 \
    --dropout 0.1

  run_model TCN "$pred_len" \
    --e_layers 3 \
    --d_model 64 \
    --d_conv 3 \
    --dropout 0.1

  run_model STGCN "$pred_len" \
    --e_layers 2 \
    --d_model 32 \
    --d_conv 3 \
    --dropout 0.1
done
