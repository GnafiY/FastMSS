#!/usr/bin/env python3
"""
Batch Qwen3 forced alignment for FastMSS source data.

Reads wav2transcript.scp and runs Qwen3-ForcedAligner on each utterance.
Supports HuggingFace model ID (auto-download) or local model path.

Input:  wav2transcript.scp — one line per utterance:
          /path/to/utt.wav hello world this is the transcript
        (split: wav_path, transcript = line.strip().split(' ', maxsplit=1))

Output: {output_dir}/{utt_id}.json — standard alignment JSON:
          {"utt_id": "...", "words": [{"word": str, "start": float, "end": float}]}

Requirements: qwen-asr, transformers, torch (qwen3-asr conda env)

Usage:
  python run_alignment.py \
      --scp wav2transcript.scp \
      --output-dir alignments/ \
      --model Qwen/Qwen3-ForcedAligner-0.6B \
      --batch-size 8
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

MAX_DURATION = 300.0


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


class Qwen3Aligner:
    """Lightweight Qwen3-ForcedAligner wrapper (lazy-load, singleton)."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self, model_path: str, device: str = "cuda:0"):
        if self._loaded:
            return
        self.model_path = model_path
        self.device = device
        self._model = None
        self._loaded = True

    def _load(self):
        if self._model is not None:
            return
        import torch
        from qwen_asr import Qwen3ForcedAligner
        print(f"[Aligner] Loading: {self.model_path}  device={self.device}")
        mp = self.model_path
        if not os.path.exists(mp):
            print(f"[Aligner] Local path not found, downloading from HuggingFace: {mp}")
        self._model = Qwen3ForcedAligner.from_pretrained(
            mp, dtype=torch.bfloat16, device_map=self.device,
        )

    def align(self, audio_path: str, text: str, language: str = "auto") -> List[WordTimestamp]:
        """Single-utterance alignment (delegates to batch)."""
        results = self.align_batch([audio_path], [text], [language])
        return results[0] if results else []

    def align_batch(self, audio_paths: List[str], texts: List[str],
                    languages: Optional[List[str]] = None) -> List[List[WordTimestamp]]:
        self._load()
        if languages is None:
            languages = ["auto"] * len(texts)
        resolved_langs = []
        for i, lang in enumerate(languages):
            if lang == "auto":
                lang = "Chinese" if re.search(r"[\u4e00-\u9fff]", texts[i]) else "English"
            resolved_langs.append(lang)
        raw = self._model.align(audio=audio_paths, text=texts, language=resolved_langs)
        all_words = []
        for segment_group in raw:
            words = []
            if segment_group:
                for item in segment_group:
                    words.append(WordTimestamp(
                        word=item.text, start=item.start_time, end=item.end_time,
                    ))
            all_words.append(words)
        return all_words


def _get_audio_duration(wav_path: str) -> float:
    try:
        import soundfile as sf
        return sf.info(wav_path).duration
    except Exception:
        return 0.0


def parse_scp(scp_path: Path) -> List[dict]:
    entries = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ", maxsplit=1)
            if len(parts) != 2:
                print(f"[WARN] Skipping: {line[:80]}")
                continue
            wav_path = Path(parts[0])
            if not wav_path.exists():
                print(f"[WARN] WAV not found: {wav_path}")
                continue
            entries.append({"utt_id": wav_path.stem, "wav_path": str(wav_path),
                            "text": parts[1].strip()})
    return entries


def main():
    p = argparse.ArgumentParser(description="Batch Qwen3 forced alignment")
    p.add_argument("--scp", required=True, type=Path, help="wav2transcript.scp")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B",
                   help="HF model ID or local path (default: HF auto-download)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--language", default="auto", help="auto / Chinese / English")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of utterances per batch alignment call (default: 1)")
    args = p.parse_args()

    entries = parse_scp(args.scp)
    if not entries:
        print("[ERROR] No valid entries in scp")
        return 1
    print(f"Loaded {len(entries)} entries from {args.scp}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligner = Qwen3Aligner(model_path=args.model, device=args.device)

    # Pre-filter: check duration
    valid = []
    for e in entries:
        dur = _get_audio_duration(e["wav_path"])
        if dur > MAX_DURATION:
            print(f"[SKIP] {e['utt_id']}: duration={dur:.0f}s > {MAX_DURATION}s")
            continue
        valid.append(e)
    print(f"Filtered: {len(valid)}/{len(entries)} entries within {MAX_DURATION}s")

    bs = max(1, args.batch_size)
    ok = fail = 0

    for batch_start in range(0, len(valid), bs):
        batch = valid[batch_start:batch_start + bs]
        audio_paths = [e["wav_path"] for e in batch]
        texts = [e["text"] for e in batch]
        languages = [args.language] * len(batch)
        ids = [e["utt_id"] for e in batch]

        print(f"  batch [{batch_start+1}-{batch_start+len(batch)}/{len(valid)}] "
              f"{','.join(ids[:3])}{'...' if len(ids) > 3 else ''}", flush=True)

        try:
            batch_words = aligner.align_batch(audio_paths, texts, languages)
        except Exception as exc:
            print(f"  FAIL (batch): {exc}")
            fail += len(batch)
            continue

        for utt_id, words in zip(ids, batch_words):
            if not words:
                print(f"    {utt_id}: no words")
                fail += 1
                continue

            json_out = {
                "utt_id": utt_id,
                "words": [{"word": w.word, "start": w.start, "end": w.end} for w in words],
            }
            (args.output_dir / f"{utt_id}.json").write_text(
                json.dumps(json_out, ensure_ascii=False, indent=2))

            print(f"    {utt_id}: OK ({len(words)} words)")
            ok += 1

    print(f"\nDone. OK={ok} FAIL={fail} TOTAL={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
