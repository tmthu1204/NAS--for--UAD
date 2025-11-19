#!/usr/bin/env bash
set -e

DATASET=${1:-}
GPU=${2:-0}

if [ -z "$DATASET" ]; then
  echo "Usage: $0 <dataset_name|source_npz> [gpu]"
  echo "Examples:"
  echo "  $0 uci_har 0"
  echo "  $0 data/uci_har/source.npz 0"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU}
echo "Running pipeline. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m src.pipeline \
  --dataset_or_paths "${DATASET}" \
  --window 128 \
  --epochs_pretrain 20 \
  --search_candidates 5 \
  --batch_size 64
