#!/usr/bin/env bash
set -e

DATASET_OR_PATHS=${1:-}
MODE=${2:-uad_source}
GPU=${3:-0}

if [ -z "$DATASET_OR_PATHS" ]; then
  echo "Usage: $0 <dataset_or_paths> [mode] [gpu]"
  echo "Examples:"
  echo "  $0 data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/val_mixed.npz uad_source 0"
  echo "  $0 data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/target_pool_unlabeled.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz adaptnas_combined 0"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU}
echo "Running pipeline. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m src.pipeline \
  --dataset_or_paths "${DATASET_OR_PATHS}" \
  --mode "${MODE}" \
  --epochs_pretrain 20 \
  --search_candidates 5 \
  --batch_size 64
