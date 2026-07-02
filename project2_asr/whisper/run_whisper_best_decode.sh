#!/usr/bin/env bash
set -euo pipefail

model=${1:-exp/whisper_small/finetuned/best}
batch_size=${2:-4}

mkdir -p exp/whisper_small/log

python whisper/decode_whisper.py \
  --manifest whisper/data/dev.jsonl \
  --model "$model" \
  --out-dir exp/whisper_small/decode_dev_best_norm \
  --batch-size "$batch_size" \
  2>&1 | tee exp/whisper_small/log/decode_dev_best_norm.log

python whisper/decode_whisper.py \
  --manifest whisper/data/test.jsonl \
  --model "$model" \
  --out-dir exp/whisper_small/decode_test_best_norm \
  --batch-size "$batch_size" \
  2>&1 | tee exp/whisper_small/log/decode_test_best_norm.log
