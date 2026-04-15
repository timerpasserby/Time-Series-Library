#!/bin/bash

PYTHON_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/torch/bin/python"
DATA_PATH="./dataset/radar_sim/"
DATA_FILE="radar.csv"

# 1. TimesNet
echo "Running TimesNet..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_timesnet \
  --model TimesNet \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1004 \
  --dec_in 1004 \
  --c_out 1004 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0

# 2. PatchTST
echo "Running PatchTST..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_patchtst \
  --model PatchTST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 1004 \
  --dec_in 1004 \
  --c_out 1004 \
  --d_model 64 \
  --d_ff 128 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0

# 3. TimeFilter
echo "Running TimeFilter..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_timefilter \
  --model TimeFilter \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 1004 \
  --dec_in 1004 \
  --c_out 1004 \
  --d_model 64 \
  --d_ff 128 \
  --patch_len 16 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0
