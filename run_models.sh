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
  --num_workers 0 \
  --gpu_type mps

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
  --num_workers 0 \
  --gpu_type mps

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
  --num_workers 0 \
  --gpu_type mps

# 4. DLinear
echo "Running DLinear..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_dlinear \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 1004 \
  --dec_in 1004 \
  --c_out 1004 \
  --moving_avg 25 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0 \
  --gpu_type mps

# 5. Informer
echo "Running Informer..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_informer \
  --model Informer \
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
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0 \
  --gpu_type mps

# 6. Autoformer
echo "Running Autoformer..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_autoformer \
  --model Autoformer \
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
  --moving_avg 25 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0 \
  --gpu_type mps

# 7. TimeMixer
echo "Running TimeMixer..."
$PYTHON_PATH -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $DATA_PATH \
  --data_path $DATA_FILE \
  --model_id radar_timemixer \
  --model TimeMixer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 1004 \
  --dec_in 1004 \
  --c_out 1004 \
  --d_model 64 \
  --d_ff 128 \
  --down_sampling_layers 3 \
  --down_sampling_window 2 \
  --down_sampling_method avg \
  --channel_independence 1 \
  --decomp_method moving_avg \
  --moving_avg 25 \
  --use_norm 1 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --train_epochs 1 \
  --target node_0 \
  --num_workers 0 \
  --gpu_type mps
