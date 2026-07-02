#!/usr/bin/env python3
"""Decode JSONL manifests with a Whisper model and score WER/CER."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils

transformers_import_utils.is_flash_attn_2_available = lambda: False
transformers_utils.is_flash_attn_2_available = lambda: False

from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return "".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []

    hyp_path = args.out_dir / "hyp.txt"
    ref_path = args.out_dir / "ref.txt"
    with hyp_path.open("w", encoding="utf-8") as hyp_handle, ref_path.open(
        "w", encoding="utf-8"
    ) as ref_handle:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            audio_arrays = [librosa.load(item["audio"], sr=16000, mono=True)[0] for item in batch]
            inputs = processor.feature_extractor(
                audio_arrays,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.to(device=device, dtype=dtype)
            with torch.no_grad():
                pred_ids = model.generate(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    language=args.language,
                    task=args.task,
                )
            texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            for item, text in zip(batch, texts):
                hyp = normalize_text(text)
                ref = normalize_text(item["text"])
                hyp_handle.write(f"{item['utt_id']} {hyp}\n")
                ref_handle.write(f"{item['utt_id']} {ref}\n")
            print(f"decoded {min(start + args.batch_size, len(records))}/{len(records)}")

    score_script = Path(__file__).resolve().parent / "score_whisper.py"
    subprocess.run(
        [
            sys.executable,
            str(score_script),
            "--ref",
            str(ref_path),
            "--hyp",
            str(hyp_path),
            "--out-dir",
            str(args.out_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
