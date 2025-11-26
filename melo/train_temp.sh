#!/bin/bash

CONFIG=$1
GPUS=${2:-1}  # default ke 1 kalau argumen kedua gak dikasih
PORT=${3:-10902}

if [ -z "$CONFIG" ]; then
  echo "Usage: bash train.sh <config_path> [num_gpus] [port]"
  exit 1
fi

MODEL_NAME=$(basename "$(dirname "$CONFIG")")

while true; do
  echo "Starting training for $MODEL_NAME using config $CONFIG on $GPUS GPU(s)..."
  
  /home/intern/miniconda3/envs/melo-tts/bin/torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py --c "$CONFIG" --model "$MODEL_NAME"

  echo "Training stopped or crashed. Cleaning up..."
  pgrep -f "$CONFIG" | xargs -r kill -9
  echo "Restarting in 30 seconds..."
  sleep 30
done
