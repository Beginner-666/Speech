#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: local/prepare_aishell10h_subset.sh [--copy|--symlink] <AISHELL1_ROOT>

Prepare the fixed Project 2 AISHELL-1 subset from a full AISHELL-1 download.

The split is defined only by data_aishell/transcript/aishell_transcript_v0.8.txt:
  train: lines     1-8000   (8,000 utterances)
  dev:   lines  8001-22326  (14,326 utterances)
  test:  lines 22327-29502  (7,176 utterances)

By default this script creates symlinks under data_aishell/wav/{train,dev,test}.
Use --copy if the final submission must contain physical wav files.

The script does not delete existing wav files. Existing destination files are
left in place, and missing utterance ids are written to data_aishell/missing_*.ids.
EOF
}

mode=symlink
case "${1:-}" in
  --copy)
    mode=copy
    shift
    ;;
  --symlink)
    mode=symlink
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
esac

if [ $# -ne 1 ]; then
  usage >&2
  exit 2
fi

src_root=$1
if [ ! -d "$src_root" ]; then
  echo "Source AISHELL-1 root does not exist: $src_root" >&2
  exit 1
fi
src_root=$(cd "$src_root" && pwd)

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
recipe_dir=$(cd "$script_dir/.." && pwd)
transcript=$recipe_dir/data_aishell/transcript/aishell_transcript_v0.8.txt

if [ ! -f "$transcript" ]; then
  echo "Transcript not found: $transcript" >&2
  exit 1
fi

line_count=$(wc -l < "$transcript")
if [ "$line_count" -ne 29502 ]; then
  echo "Expected 29502 transcript lines, got $line_count: $transcript" >&2
  exit 1
fi

mkdir -p "$recipe_dir/data_aishell/transcript" "$recipe_dir/data_aishell/wav"

awk 'NR <= 8000 { print }' "$transcript" > "$recipe_dir/data_aishell/transcript/train.txt"
awk 'NR > 8000 && NR <= 22326 { print }' "$transcript" > "$recipe_dir/data_aishell/transcript/dev.txt"
awk 'NR > 22326 { print }' "$transcript" > "$recipe_dir/data_aishell/transcript/test.txt"

for split in train dev test; do
  awk '{ print $1 }' "$recipe_dir/data_aishell/transcript/$split.txt" \
    > "$recipe_dir/data_aishell/transcript/$split.ids"
done

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/aishell10h.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT

index=$tmpdir/wav_index.tsv
find "$src_root" -type f -name 'BAC009S*.wav' | awk '
  {
    path = $0
    n = split(path, parts, "/")
    utt = parts[n]
    sub(/\.wav$/, "", utt)
    if (!(utt in seen)) {
      seen[utt] = 1
      print utt "\t" path
    }
  }
' > "$index"

if [ ! -s "$index" ]; then
  echo "No AISHELL wav files were found under: $src_root" >&2
  exit 1
fi

declare -A wav_by_utt
while IFS=$'\t' read -r utt wav_path; do
  wav_by_utt[$utt]=$wav_path
done < "$index"

prepare_split() {
  local split=$1
  local ids=$recipe_dir/data_aishell/transcript/$split.ids
  local missing=$recipe_dir/data_aishell/missing_$split.ids
  local linked=0
  local existing=0
  local absent=0

  : > "$missing"
  mkdir -p "$recipe_dir/data_aishell/wav/$split"

  while read -r utt; do
    [ -n "$utt" ] || continue
    local speaker
    speaker=$(printf '%s\n' "$utt" | sed -n 's/^BAC009\(S[0-9][0-9][0-9][0-9]\)W[0-9][0-9][0-9][0-9]$/\1/p')
    if [ -z "$speaker" ]; then
      echo "Bad utterance id format: $utt" >&2
      exit 1
    fi

    local dst_dir=$recipe_dir/data_aishell/wav/$split/$speaker
    local dst=$dst_dir/$utt.wav
    mkdir -p "$dst_dir"

    if [ -e "$dst" ] || [ -L "$dst" ]; then
      existing=$((existing + 1))
      continue
    fi

    local src=${wav_by_utt[$utt]:-}
    if [ -z "$src" ]; then
      printf '%s\n' "$utt" >> "$missing"
      absent=$((absent + 1))
      continue
    fi

    if [ "$mode" = copy ]; then
      cp "$src" "$dst"
    else
      ln -s "$src" "$dst"
    fi
    linked=$((linked + 1))
  done < "$ids"

  local expected
  expected=$(wc -l < "$ids")
  local present
  present=$(find "$recipe_dir/data_aishell/wav/$split" \( -type f -o -type l \) -name "BAC009S*.wav" | wc -l)

  printf '%-5s expected=%s present=%s existing=%s added=%s missing=%s\n' \
    "$split" "$expected" "$present" "$existing" "$linked" "$absent"
}

prepare_split train
prepare_split dev
prepare_split test

mkdir -p "$recipe_dir/resource_aishell"
for resource in lexicon.txt speaker.info; do
  found=$(find "$src_root" -type f -name "$resource" | head -n 1 || true)
  if [ -n "$found" ]; then
    cp "$found" "$recipe_dir/resource_aishell/$resource"
    echo "Copied resource_aishell/$resource"
  else
    echo "Resource not found under source root: $resource" >&2
  fi
done

echo "Subset preparation finished. Check data_aishell/missing_*.ids before running ./run.sh."
