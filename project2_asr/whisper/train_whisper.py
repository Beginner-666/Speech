#!/usr/bin/env python3
"""Fine-tune Whisper on the Project 2 AISHELL subset with a minimal PyTorch loop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import librosa
import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils
from torch.utils.data import DataLoader, Dataset

transformers_import_utils.is_flash_attn_2_available = lambda: False
transformers_utils.is_flash_attn_2_available = lambda: False

from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402


def normalize_text(text: str) -> str:
    return "".join(text.split())


def load_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


class WhisperJsonlDataset(Dataset):
    def __init__(self, path: Path):
        self.records = load_manifest(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.records[index]


class WhisperCollator:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        audio_arrays = [librosa.load(item["audio"], sr=16000, mono=True)[0] for item in batch]
        input_features = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features
        texts = [normalize_text(item["text"]) for item in batch]
        labels_batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ref_item != hyp_item),
            )
        prev = cur
    return prev[-1]


def cer(predictions: list[str], references: list[str]) -> float:
    errors = 0
    total = 0
    for pred, ref in zip(predictions, references):
        pred_chars = list(normalize_text(pred))
        ref_chars = list(normalize_text(ref))
        errors += edit_distance(ref_chars, pred_chars)
        total += len(ref_chars)
    return 100.0 * errors / max(total, 1)


@torch.no_grad()
def evaluate(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    records: list[dict[str, str]],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
) -> float:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    limit = len(records) if max_batches <= 0 else min(len(records), max_batches * batch_size)
    for start in range(0, limit, batch_size):
        batch = records[start : start + batch_size]
        audio_arrays = [librosa.load(item["audio"], sr=16000, mono=True)[0] for item in batch]
        input_features = processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features.to(device=device)
        pred_ids = model.generate(input_features, max_new_tokens=128, language="zh", task="transcribe")
        predictions.extend(processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
        references.extend(item["text"] for item in batch)
    model.train()
    return cer(predictions, references)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=Path("whisper/data/train.jsonl"))
    parser.add_argument("--dev-jsonl", type=Path, default=Path("whisper/data/dev.jsonl"))
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/whisper_small/finetuned"))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-max-batches", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    dtype = torch.float16 if use_amp else torch.float32

    processor = WhisperProcessor.from_pretrained(args.model, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    train_dataset = WhisperJsonlDataset(args.train_jsonl)
    dev_records = load_manifest(args.dev_jsonl)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=WhisperCollator(processor),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_cer = math.inf
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        for batch in train_loader:
            input_features = batch["input_features"].to(device=device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(input_features=input_features, labels=labels).loss
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}/{args.max_steps} loss {loss.item() * args.gradient_accumulation_steps:.4f}")

            if global_step % args.eval_steps == 0:
                dev_cer = evaluate(
                    model,
                    processor,
                    dev_records,
                    args.eval_batch_size,
                    device,
                    dtype,
                    args.eval_max_batches,
                )
                print(f"step {global_step}: dev CER {dev_cer:.2f}")
                if dev_cer < best_cer:
                    best_cer = dev_cer
                    best_dir = args.output_dir / "best"
                    model.save_pretrained(best_dir)
                    processor.save_pretrained(best_dir)
                    (args.output_dir / "best_cer.txt").write_text(f"{best_cer:.2f}\n", encoding="utf-8")

            if global_step % args.save_steps == 0:
                ckpt_dir = args.output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)

            if global_step >= args.max_steps:
                break

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"finished training, best dev CER observed: {best_cer:.2f}")


if __name__ == "__main__":
    main()
