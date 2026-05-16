#!/usr/bin/env python3
"""
Convert Praat TextGrid files to alignment JSON for alignment_to_lhotse.py.

Supports the Praat short text format with a single 'words' IntervalTier.
Silent intervals (empty text "") are dropped.

Input options:
  1. --tg-dir DIR  — directory of {utt_id}.TextGrid files
  2. --scp FILE    — wav2textgrid.scp: <wav_path> <textgrid_path>

Output: {output_dir}/{utt_id}.json:
  {"utt_id": "...", "words": [{"word": str, "start": float, "end": float}]}

This output feeds directly into alignment_to_lhotse.py, so both Qwen3-aligned
and externally-aligned TextGrid data use the same downstream pipeline.

Usage:
  # From a directory of TextGrid files
  python textgrid_to_alignment_json.py \
      --tg-dir textgrids/ \
      --output-dir alignments/

  # From wav2textgrid.scp
  python textgrid_to_alignment_json.py \
      --scp wav2textgrid.scp \
      --output-dir alignments/
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


def parse_textgrid_short(path: Path) -> List[Interval]:
    """Parse Praat short-format TextGrid into word intervals."""
    content = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    if len(lines) < 11 or lines[0] != 'File type = "ooTextFile"':
        raise ValueError(f"Not a valid Praat short-format TextGrid: {path}")

    # Find tier count and locate the 'words' tier
    try:
        tier_count = int(lines[5])
    except ValueError:
        raise ValueError(f"Cannot parse tier count from: {path}")

    name_idx = None
    for i in range(6, len(lines)):
        if lines[i] == '"words"':
            name_idx = i
            break
    if name_idx is None:
        raise ValueError(f"No 'words' tier found in: {path}")

    # After name comes xmin, xmax, interval_count
    if name_idx + 3 >= len(lines):
        raise ValueError(f"Unexpected end of TextGrid after 'words' tier: {path}")
    interval_count = int(lines[name_idx + 3])

    intervals = []
    base = name_idx + 4
    for idx in range(interval_count):
        xmin = float(lines[base + idx * 3])
        xmax = float(lines[base + idx * 3 + 1])
        text = lines[base + idx * 3 + 2].strip('"')
        if text:  # drop silent intervals
            intervals.append(Interval(xmin=xmin, xmax=xmax, text=text))
    return intervals


def textgrid_to_alignment_json(tg_path: Path, utt_id: str, output_dir: Path) -> bool:
    try:
        intervals = parse_textgrid_short(tg_path)
    except Exception as e:
        print(f"[SKIP] {utt_id}: {e}")
        return False

    if not intervals:
        print(f"[SKIP] {utt_id}: no non-empty intervals")
        return False

    words = [{"word": iv.text, "start": iv.xmin, "end": iv.xmax} for iv in intervals]
    data = {"utt_id": utt_id, "words": words}
    (output_dir / f"{utt_id}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  → {utt_id}: {len(words)} words")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Convert Praat TextGrid → alignment JSON")
    p.add_argument("--tg-dir", type=Path,
                   help="Directory of {utt_id}.TextGrid files")
    p.add_argument("--scp", type=Path,
                   help="wav2textgrid.scp: <wav_path> <textgrid_path>")
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    if args.scp:
        with open(args.scp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                wav_path = Path(parts[0])
                tg_path = Path(parts[1])
                items.append((wav_path.stem, tg_path))
    elif args.tg_dir:
        for tg_path in sorted(args.tg_dir.glob("*.TextGrid")):
            items.append((tg_path.stem, tg_path))
    else:
        print("[ERROR] Either --tg-dir or --scp is required")
        return 1

    ok = 0
    for utt_id, tg_path in items:
        if textgrid_to_alignment_json(tg_path, utt_id, args.output_dir):
            ok += 1

    print(f"\nDone. Converted {ok}/{len(items)} TextGrid(s) → {args.output_dir}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
