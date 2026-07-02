#!/usr/bin/env bash
set -euo pipefail

model=${1:-openai/whisper-small}
batch_size=${2:-8}

python whisper/prepare_whisper_data.py
python whisper/decode_whisper.py \
  --manifest whisper/data/test.jsonl \
  --model "$model" \
  --out-dir exp/whisper_small/baseline_decode_test \
  --batch-size "$batch_size"
