#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# SlopeMine 批量实验脚本
# 功能：在 slopemine_full.csv（天气+爆破，17特征）上，对所有基线模型
#       运行 pred_len=1/3/6/12，train_epochs=30
#
# 用法：
#   cd /path/to/Time-Series-Library
#   bash scripts/slopemine/run_epochs30.sh
#
# 可选环境变量：
#   PYTHON_BIN    - Python 路径（默认 python）
#   USE_GPU       - 是否使用 GPU，1=是 0=否（默认自动检测）
#   BATCH_SIZE    - 批大小（默认 32）
#   NUM_WORKERS   - DataLoader workers（默认 0）
###############################################################################

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_EPOCHS=30
SEQ_LEN=24
LABEL_LEN=24
PRED_LENS="1 3 6 12"
ENC_IN=18
ROOT_PATH="./dataset/slopemine"
DATA_PATH="slopemine_full.csv"
CHECKPOINTS="./checkpoints"
RESULTS_DIR="./results"

# GPU 检测
if [ "${USE_GPU:-auto}" = "auto" ]; then
  if "$PYTHON_BIN" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    USE_GPU=1
  else
    USE_GPU=0
  fi
fi

if [ "$USE_GPU" = "1" ]; then
  GPU_ARGS=""
  echo ">>> 使用 GPU"
else
  GPU_ARGS="--no_use_gpu"
  echo ">>> 使用 CPU"
fi

# 日志目录
LOG_DIR="./scripts/slopemine/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/epochs30_${TIMESTAMP}.log"

# 通用参数
COMMON_ARGS=(
  --task_name long_term_forecast
  --is_training 1
  --data custom
  --root_path "$ROOT_PATH"
  --data_path "$DATA_PATH"
  --features MS
  --target deformation
  --freq h
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --enc_in "$ENC_IN"
  --dec_in "$ENC_IN"
  --c_out "$ENC_IN"
  --train_epochs "$TRAIN_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate 0.0001
  --num_workers "$NUM_WORKERS"
  --itr 1
  --checkpoints "$CHECKPOINTS"
  --des slopemine_full_e30
)

run_model() {
  local model="$1"
  local pred_len="$2"
  shift 2

  local model_id="slopemine_${model}_sl${SEQ_LEN}_pl${pred_len}_e${TRAIN_EPOCHS}_full"
  local start_time=$SECONDS

  echo "================================================================" | tee -a "$LOG_FILE"
  echo "[$(date '+%H:%M:%S')] Running: $model | pred_len=$pred_len" | tee -a "$LOG_FILE"
  echo "================================================================" | tee -a "$LOG_FILE"

  # shellcheck disable=SC2086
  if "$PYTHON_BIN" -u run.py \
    "${COMMON_ARGS[@]}" \
    --model "$model" \
    --model_id "$model_id" \
    --pred_len "$pred_len" \
    "$@" \
    $GPU_ARGS \
    >> "$LOG_FILE" 2>&1; then

    local elapsed=$(( SECONDS - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    # 从日志提取最终指标
    local metrics
    metrics=$(grep -oP 'mse:\K[0-9.]+' "$LOG_FILE" | tail -1)
    local mae
    mae=$(grep -oP 'mae:\K[0-9.]+' "$LOG_FILE" | tail -1)

    echo "[$(date '+%H:%M:%S')] ✅ Done: $model pl=$pred_len | MSE=$metrics MAE=$mae | ${mins}m${secs}s" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%H:%M:%S')] ❌ FAILED: $model pl=$pred_len" | tee -a "$LOG_FILE"
  fi
  echo "" | tee -a "$LOG_FILE"
}

echo "============================================================" | tee "$LOG_FILE"
echo "SlopeMine Batch Experiment - $(date)" | tee -a "$LOG_FILE"
echo "Data: $DATA_PATH | Features: $ENC_IN | Epochs: $TRAIN_EPOCHS" | tee -a "$LOG_FILE"
echo "Pred lens: $PRED_LENS | GPU: $USE_GPU" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

for pred_len in $PRED_LENS; do
  echo ">>> ========== pred_len = $pred_len ========== <<<" | tee -a "$LOG_FILE"

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
    --patch_len 8 \
    --stride 4

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

  run_model LSTM "$pred_len"

  run_model TCN "$pred_len"

  run_model STGCN "$pred_len"
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 提取汇总结果
echo ""
echo ">>> 汇总结果（从日志提取）:"
echo "------------------------------------------------------------"
echo "Model | PredLen | MSE | MAE"
echo "------------------------------------------------------------"
grep "✅ Done:" "$LOG_FILE" | while IFS= read -r line; do
  model=$(echo "$line" | grep -oP 'Done: \K\w+')
  pl=$(echo "$line" | grep -oP 'pl=\K[0-9]+')
  mse=$(echo "$line" | grep -oP 'MSE=\K[0-9.]+')
  mae=$(echo "$line" | grep -oP 'MAE=\K[0-9.]+')
  printf "%-12s | pl=%-3s | %-10s | %-10s\n" "$model" "$pl" "$mse" "$mae"
done
echo "------------------------------------------------------------"
