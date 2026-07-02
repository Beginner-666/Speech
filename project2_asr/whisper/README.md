# Whisper Experiment

This directory adds a second ASR system for Project 2 using Hugging Face Whisper.
It reuses the existing Kaldi data split under `project2_asr/data/{train,dev,test}` so
the Kaldi GMM-HMM, Kaldi TDNN, and Whisper results are directly comparable.

## Dependencies

Install the Python dependencies if they are not already available:

```bash
pip install -r whisper/requirements.txt
```

## Data Conversion

```bash
python whisper/prepare_whisper_data.py
```

This writes:

```text
whisper/data/train.jsonl
whisper/data/dev.jsonl
whisper/data/test.jsonl
```

Each JSONL record contains `utt_id`, absolute `audio` path, and normalized Chinese
`text` without spaces.

## Baseline Evaluation

Run the pretrained model before fine-tuning:

```bash
bash whisper/run_whisper_baseline.sh openai/whisper-small 8
```

Output:

```text
exp/whisper_small/baseline_decode_test/hyp.txt
exp/whisper_small/baseline_decode_test/ref.txt
exp/whisper_small/baseline_decode_test/wer.txt
exp/whisper_small/baseline_decode_test/cer.txt
```

## Fine-tuning

```bash
bash whisper/run_whisper_finetune.sh openai/whisper-small
```

The default training output is:

```text
exp/whisper_small/finetuned
```

The final dev/test decoding outputs are:

```text
exp/whisper_small/decode_dev
exp/whisper_small/decode_test
```

Report both pretrained baseline and fine-tuned WER/CER. CER is the primary metric
for this assignment.
