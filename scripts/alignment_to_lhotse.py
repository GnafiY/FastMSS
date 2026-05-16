#!/usr/bin/env python3
"""
Convert alignment JSON to Lhotse MonoCut/CutSet for FastMSS.

Reads per-utterance alignment JSONs from run_alignment.py output and creates
Lhotse CutSets with word-level alignment in SupervisionSegment.alignment["word"].

Alignment JSON format:
  {"utt_id": "...", "words": [{"word": str, "start": float, "end": float}]}

Speaker assignment (priority):
  1. --spk-scp mapping file  (recommended for multi-speaker corpora)
  2. word-level "speaker" field in alignment JSON
  3. --speaker default

Output:  {output}/{prefix}_{split}.jsonl.gz

Usage:
  # Full pipeline: wav2spk for speaker labels
  python alignment_to_lhotse.py \
      --align-dir alignments/ \
      --wav-dir wavs/ \
      --spk-scp wav2spk.scp \
      --output lhotse_cutsets/ \
      --prefix my_corpus \
      --splits train,dev,test --train-ratio 0.8
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from lhotse import CutSet, MonoCut, Recording
from lhotse.supervision import AlignmentItem, SupervisionSegment


def load_alignment(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def words_to_alignment_items(words: List[Dict]) -> List[AlignmentItem]:
    items = []
    for w in words:
        try:
            start = float(w["start"])
            end = float(w["end"])
            dur = round(end - start, 8)
            if dur > 0:
                items.append(AlignmentItem(symbol=w["word"], start=start, duration=dur, score=None))
        except (KeyError, ValueError, TypeError):
            continue
    return items


def resolve_speaker(words: List[Dict], default_speaker: str) -> str:
    """Determine speaker from word-level speaker fields, else fallback."""
    speaker_dur = defaultdict(float)
    for w in words:
        if "speaker" in w:
            dur = float(w.get("end", 0)) - float(w.get("start", 0))
            if dur > 0:
                speaker_dur[w["speaker"]] += dur
    if speaker_dur:
        return max(speaker_dur, key=speaker_dur.get)
    return default_speaker


def load_spk_scp(path: Path) -> Dict[str, str]:
    """Parse wav2spk.scp: <wav_path> <speaker_id>."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                utt_id = Path(parts[0]).stem
                mapping[utt_id] = parts[1]
    return mapping


def split_indices(n: int, ratios: Dict[str, float], seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    split = {}
    start, cum = 0, 0.0
    names = list(ratios.keys())
    for i, name in enumerate(names):
        if i == len(names) - 1:
            end = n
        else:
            cum += ratios[name]
            end = int(n * cum)
        split[name] = sorted(indices[start:end])
        start = end
    return split


def main():
    p = argparse.ArgumentParser(description="Convert alignment JSON → Lhotse CutSet")
    p.add_argument("--align-dir", required=True, type=Path,
                   help="Directory of {utt_id}.json alignment files")
    p.add_argument("--wav-dir", required=True, type=Path,
                   help="Directory of {utt_id}.wav audio files")
    p.add_argument("--spk-scp", type=Path, default=None,
                   help="wav2spk.scp: <wav_path> <speaker_id> (per-line)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--prefix", default="cutset")
    p.add_argument("--splits", default="all",
                   help="'all' or comma-separated names (default: all)")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--speaker", default="spk0",
                   help="Fallback speaker when no spk-scp or word-level speaker")
    args = p.parse_args()

    # --- load speaker mapping ---
    spk_map = {}
    if args.spk_scp and args.spk_scp.exists():
        spk_map = load_spk_scp(args.spk_scp)
        print(f"Loaded {len(spk_map)} speaker mappings from {args.spk_scp}")

    # --- discover alignments ---
    align_files = sorted(args.align_dir.glob("*.json"))
    align_files = [f for f in align_files if f.name != "alignment_meta.json"]
    if not align_files:
        print(f"[ERROR] No alignment JSONs in {args.align_dir}")
        return 1
    print(f"Found {len(align_files)} alignment JSON(s) in {args.align_dir}")

    # --- convert ---
    cuts = []
    skipped = {"no_wav": 0, "bad_json": 0, "empty_aln": 0, "audio_err": 0}

    for af in align_files:
        utt_id = af.stem
        wav_path = args.wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            skipped["no_wav"] += 1
            continue

        data = load_alignment(af)
        if data is None:
            skipped["bad_json"] += 1
            continue

        words = data.get("words", [])
        if not words:
            skipped["empty_aln"] += 1
            continue

        aln_items = words_to_alignment_items(words)
        if not aln_items:
            skipped["empty_aln"] += 1
            continue

        # speaker: spk-scp > word-level > fallback
        if utt_id in spk_map:
            speaker = spk_map[utt_id]
        else:
            speaker = resolve_speaker(words, args.speaker)

        text = " ".join(w.get("word", "") for w in words)

        try:
            rec = Recording.from_file(wav_path, recording_id=utt_id)
        except Exception:
            skipped["audio_err"] += 1
            continue

        sup = SupervisionSegment(
            id=f"{utt_id}-sup000",
            recording_id=utt_id,
            start=0.0,
            duration=rec.duration,
            channel=0,
            text=text,
            speaker=speaker,
            language="auto",
            alignment={"word": aln_items},
        )
        cut = MonoCut(
            id=utt_id, start=0.0, duration=rec.duration,
            channel=0, recording=rec, supervisions=[sup],
        )
        cuts.append(cut)

    print(f"Converted: {len(cuts)} cuts  "
          f"(skipped: no_wav={skipped['no_wav']} bad_json={skipped['bad_json']} "
          f"empty_aln={skipped['empty_aln']} audio_err={skipped['audio_err']})")
    if not cuts:
        return 1

    # --- split ---
    args.output.mkdir(parents=True, exist_ok=True)
    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.splits == "all" or split_names == ["all"]:
        CutSet.from_cuts(cuts).to_file(args.output / f"{args.prefix}_all.jsonl.gz")
        print(f"Saved {len(cuts)} cuts → {args.output / f'{args.prefix}_all.jsonl.gz'}")
    else:
        n = len(split_names)
        if n == 2:
            ratios = {split_names[0]: args.train_ratio}
            ratios[split_names[1]] = round(1.0 - args.train_ratio, 8)
        elif n == 3:
            ratios = {split_names[0]: args.train_ratio, split_names[1]: args.dev_ratio}
            ratios[split_names[2]] = round(1.0 - args.train_ratio - args.dev_ratio, 8)
        else:
            eq = round(1.0 / n, 8)
            ratios = {name: eq for name in split_names}
        s = sum(ratios.values())
        if s > 0:
            ratios = {k: v / s for k, v in ratios.items()}

        assignments = split_indices(len(cuts), ratios, args.seed)
        for name in split_names:
            split_cuts = [cuts[i] for i in assignments[name]]
            if split_cuts:
                out = args.output / f"{args.prefix}_{name}.jsonl.gz"
                CutSet.from_cuts(split_cuts).to_file(out)
                print(f"  {name}: {len(split_cuts)} cuts → {out}")

    # --- stats ---
    dur = [c.duration for c in cuts]
    print(f"Duration: total={sum(dur)/3600:.1f}h mean={sum(dur)/len(dur):.0f}s "
          f"min={min(dur):.0f}s max={max(dur):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
