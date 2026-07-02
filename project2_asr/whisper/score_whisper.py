#!/usr/bin/env python3
"""Score Whisper hypotheses with WER and CER."""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

try:
    from opencc import OpenCC
except Exception:
    OpenCC = None


_OPENCC = OpenCC("t2s") if OpenCC is not None else None


def read_text_map(path: Path) -> dict[str, str]:
    items: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            items[utt_id] = text
    return items


def normalize_text(text: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    text = unicodedata.normalize("NFKC", text)
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    text = text.lower()
    chars: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        if char.isspace():
            continue
        chars.append(char)
    return "".join(chars)


def normalize_for_cer(text: str, enabled: bool = True) -> str:
    return normalize_text(text, enabled)


def normalize_for_wer(text: str, enabled: bool = True) -> str:
    text = normalize_text(text, enabled)
    text = re.sub(r"\s+", " ", text.strip())
    if " " in text:
        return text
    return " ".join(list(text))


def edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dist = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        dist[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(1, cols):
        dist[0][j] = j
        ops[0][j] = (j, 0, 0)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                choices = [(dist[i - 1][j - 1], ops[i - 1][j - 1])]
            else:
                s_ops = ops[i - 1][j - 1]
                choices = [(dist[i - 1][j - 1] + 1, (s_ops[0], s_ops[1], s_ops[2] + 1))]
            i_ops = ops[i][j - 1]
            d_ops = ops[i - 1][j]
            choices.append((dist[i][j - 1] + 1, (i_ops[0] + 1, i_ops[1], i_ops[2])))
            choices.append((dist[i - 1][j] + 1, (d_ops[0], d_ops[1] + 1, d_ops[2])))
            dist[i][j], ops[i][j] = min(choices, key=lambda x: x[0])
    return ops[-1][-1]


def score(
    refs: dict[str, str], hyps: dict[str, str], unit: str, normalize: bool
) -> tuple[float, int, int, int, int]:
    ins = dels = subs = total = 0
    for utt_id in sorted(refs):
        ref_text = refs[utt_id]
        hyp_text = hyps.get(utt_id, "")
        if unit == "char":
            ref_units = list(normalize_for_cer(ref_text, normalize))
            hyp_units = list(normalize_for_cer(hyp_text, normalize))
        else:
            ref_units = normalize_for_wer(ref_text, normalize).split()
            hyp_units = normalize_for_wer(hyp_text, normalize).split()
        i, d, s = edit_distance(ref_units, hyp_units)
        ins += i
        dels += d
        subs += s
        total += len(ref_units)
    rate = 100.0 * (ins + dels + subs) / max(total, 1)
    return rate, total, ins, dels, subs


def write_filtered_outputs(refs: dict[str, str], hyps: dict[str, str], out_dir: Path, normalize: bool) -> None:
    def spaced_chars(text: str) -> str:
        return " ".join(list(normalize_for_cer(text, normalize)))

    with (out_dir / "test_filt.txt").open("w", encoding="utf-8") as ref_f, \
        (out_dir / "test_filt.chars.txt").open("w", encoding="utf-8") as ref_c, \
        (out_dir / "hyp_filt.txt").open("w", encoding="utf-8") as hyp_f, \
        (out_dir / "hyp_filt.chars.txt").open("w", encoding="utf-8") as hyp_c:
        for utt_id in sorted(refs):
            ref_text = normalize_for_cer(refs[utt_id], normalize)
            hyp_text = normalize_for_cer(hyps.get(utt_id, ""), normalize)
            ref_f.write(f"{utt_id} {ref_text}\n")
            ref_c.write(f"{utt_id} {spaced_chars(refs[utt_id])}\n")
            hyp_f.write(f"{utt_id} {hyp_text}\n")
            hyp_c.write(f"{utt_id} {spaced_chars(hyps.get(utt_id, ''))}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--hyp", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    refs = read_text_map(args.ref)
    hyps = read_text_map(args.hyp)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalize = not args.no_normalize

    write_filtered_outputs(refs, hyps, args.out_dir, normalize)
    wer = score(refs, hyps, "word", normalize)
    cer = score(refs, hyps, "char", normalize)
    norm_note = "normalized" if normalize else "raw"
    (args.out_dir / "wer.txt").write_text(
        f"%WER {wer[0]:.2f} [ {wer[2] + wer[3] + wer[4]} / {wer[1]}, "
        f"{wer[2]} ins, {wer[3]} del, {wer[4]} sub ] ({norm_note})\n",
        encoding="utf-8",
    )
    (args.out_dir / "cer.txt").write_text(
        f"%CER {cer[0]:.2f} [ {cer[2] + cer[3] + cer[4]} / {cer[1]}, "
        f"{cer[2]} ins, {cer[3]} del, {cer[4]} sub ] ({norm_note})\n",
        encoding="utf-8",
    )
    print((args.out_dir / "wer.txt").read_text(encoding="utf-8").strip())
    print((args.out_dir / "cer.txt").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    main()
