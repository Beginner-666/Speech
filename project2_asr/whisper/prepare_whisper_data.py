#!/usr/bin/env python3
"""Convert Kaldi data directories into JSONL manifests for Whisper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_kaldi_map(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            items[key] = value
    return items


def normalize_text(text: str) -> str:
    return "".join(text.split())


def resolve_wav_command(value: str, project_dir: Path) -> str:
    """Resolve simple Kaldi wav.scp paths; reject shell pipelines for Whisper."""
    if value.endswith("|"):
        raise ValueError(f"Whisper JSONL expects file paths, got wav.scp command: {value}")
    path = Path(value)
    if not path.is_absolute():
        path = project_dir / path
    return os.path.abspath(path)


def convert_split(project_dir: Path, split: str, output_dir: Path) -> int:
    data_dir = project_dir / "data" / split
    wavs = read_kaldi_map(data_dir / "wav.scp")
    texts = read_kaldi_map(data_dir / "text")

    missing = sorted(set(texts) - set(wavs))
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} utterances have text but no wav, e.g. {missing[:3]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for utt_id in sorted(texts):
            record = {
                "utt_id": utt_id,
                "audio": resolve_wav_command(wavs[utt_id], project_dir),
                "text": normalize_text(texts[utt_id]),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(texts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir or project_dir / "whisper" / "data"
    for split in args.splits:
        count = convert_split(project_dir, split, output_dir)
        print(f"{split}: wrote {count} records to {output_dir / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
