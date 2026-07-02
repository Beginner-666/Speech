#!/usr/bin/env bash
set -euo pipefail

model=${1:-openai/whisper-small}
train_batch_size=${2:-2}
eval_batch_size=${3:-2}
decode_batch_size=${4:-4}

mkdir -p exp/whisper_small/log

python whisper/prepare_whisper_data.py 2>&1 | tee exp/whisper_small/log/prepare_whisper_data.log
python whisper/train_whisper.py \
  --model "$model" \
  --train-batch-size "$train_batch_size" \
  --eval-batch-size "$eval_batch_size" \
  --gradient-accumulation-steps 4 \
  --num-workers 0 \
  2>&1 | tee exp/whisper_small/log/train_whisper.log

python whisper/decode_whisper.py \
  --manifest whisper/data/dev.jsonl \
  --model exp/whisper_small/finetuned \
  --out-dir exp/whisper_small/decode_dev \
  --batch-size "$decode_batch_size" \
  2>&1 | tee exp/whisper_small/log/decode_dev.log

python whisper/decode_whisper.py \
  --manifest whisper/data/test.jsonl \
  --model exp/whisper_small/finetuned \
  --out-dir exp/whisper_small/decode_test \
  --batch-size "$decode_batch_size" \
  2>&1 | tee exp/whisper_small/log/decode_test.log
