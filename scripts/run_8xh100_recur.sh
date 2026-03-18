#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-recur18_u5_d672_$(date +%Y%m%d_%H%M%S)}"
export DATA_PATH="${DATA_PATH:-$ROOT/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/data/tokenizers/fineweb_1024_bpe.model}"

export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export NUM_LAYERS="${NUM_LAYERS:-18}"
export NUM_UNIQUE_BLOCKS="${NUM_UNIQUE_BLOCKS:-5}"
export MODEL_DIM="${MODEL_DIM:-672}"
export NUM_HEADS="${NUM_HEADS:-12}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-6}"
export MLP_MULT="${MLP_MULT:-2}"

export TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-524288}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-524288}"
export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

export INT8_AXIS_MODE="${INT8_AXIS_MODE:-auto}"
export INT8_RESIDUAL_RANK="${INT8_RESIDUAL_RANK:-1}"
export INT8_RESIDUAL_BUDGET_BYTES="${INT8_RESIDUAL_BUDGET_BYTES:-65536}"
export SDP_BACKEND="${SDP_BACKEND:-flash}"
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-1}"

cd "$ROOT"
exec torchrun --standalone --nproc_per_node=8 train_gpt.py
