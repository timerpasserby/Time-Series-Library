#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SlopeMine 天气基线实验（仅天气，无爆破）
# 与 run_epochs30.sh 配合使用，用于 A/B 对比爆破特征的贡献
#
# 数据：slopemine_agg_weather.csv（11 特征：6内部 + 4天气 + deformation）
###############################################################################

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_EPOCHS=30
SEQ_LEN=24
LABEL_LEN=24
PRED_LENS="1 3 6 12"
ENC_IN=11
ROOT_PATH="./dataset/slopemine"
DATA_PATH="slopemine_agg_weather.csv"
CHECKPOINTS="./checkpoints"

if [ "${USE_GPU:-auto}" = "auto" ]; then
  if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    USE_GPU=1
  else
    USE_GPU=0
  fi
fi

[ "$USE_GPU" = "1" ] && GPU_ARGS="" || GPU_ARGS="--no_use_gpu"

LOG_DIR="./scripts/slopemine/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/weather_only_e30_${TIMESTAMP}.log"

COMMON_ARGS=(
  --task_name long_term_forecast --is_training 1 --data custom
  --root_path "$ROOT_PATH" --data_path "$DATA_PATH"
  --features MS --target deformation --freq h
  --seq_len "$SEQ_LEN" --label_len "$LABEL_LEN"
  --enc_in "$ENC_IN" --dec_in "$ENC_IN" --c_out "$ENC_IN"
  --train_epochs "$TRAIN_EPOCHS" --batch_size "$BATCH_SIZE"
  --learning_rate 0.0001 --num_workers "$NUM_WORKERS"
  --itr 1 --checkpoints "$CHECKPOINTS" --des slopemine_weather_e30
)

run_model() {
  local model="$1" pred_len="$2"
  shift 2
  local model_id="slopemine_${model}_sl${SEQ_LEN}_pl${pred_len}_e${TRAIN_EPOCHS}_weather"
  echo "[$(date '+%H:%M:%S')] $model pl=$pred_len ..." | tee -a "$LOG_FILE"
  "$PYTHON_BIN" -u run.py "${COMMON_ARGS[@]}" \
    --model "$model" --model_id "$model_id" --pred_len "$pred_len" \
    "$@" $GPU_ARGS >> "$LOG_FILE" 2>&1 \
    && echo "  ✅ $(grep -oP 'mse:\K[0-9.]+' "$LOG_FILE" | tail -1) / $(grep -oP 'mae:\K[0-9.]+' "$LOG_FILE" | tail -1)" | tee -a "$LOG_FILE" \
    || echo "  ❌ FAILED" | tee -a "$LOG_FILE"
}

echo "Weather-only baseline (enc_in=11, epochs=30)" | tee "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

for pred_len in $PRED_LENS; do
  echo ">>> pl=$pred_len" | tee -a "$LOG_FILE"
  run_model DLinear "$pred_len" --e_layers 2 --d_layers 1 --d_model 64 --d_ff 128 --moving_avg 25 --dropout 0.1
  run_model PatchTST "$pred_len" --e_layers 1 --d_layers 1 --d_model 64 --d_ff 128 --n_heads 4 --dropout 0.1 --patch_len 8 --stride 4
  run_model TimeFilter "$pred_len" --e_layers 1 --d_layers 1 --d_model 64 --d_ff 128 --n_heads 4 --dropout 0.1 --patch_len 24 --alpha 0.1 --top_p 0.5 --pos 1
  run_model LSTM "$pred_len"
  run_model TCN "$pred_len"
  run_model STGCN "$pred_len"
  echo "" | tee -a "$LOG_FILE"
done

echo "Done. Log: $LOG_FILE" | tee -a "$LOG_FILE"
