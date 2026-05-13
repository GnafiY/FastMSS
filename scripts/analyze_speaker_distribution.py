import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to nemo_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.output_dir is None:
        args.output_dir = Path(args.manifest).parent

    output_dir = Path(args.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    count_by_spk = defaultdict(int)
    dur_by_spk = defaultdict(float)

    with open(manifest_path) as f:
        for line in f:
            s = json.loads(line)
            n_spk = s["num_speakers"]
            count_by_spk[n_spk] += 1
            dur_by_spk[n_spk] += s["duration"] / 60.0

    spk_range = sorted(count_by_spk.keys())
    counts = [count_by_spk[s] for s in spk_range]
    durs = [dur_by_spk[s] for s in spk_range]
    max_dur = max(durs)
    if max_dur > 60:
        durs = [d / 60.0 for d in durs]
        dur_unit = "hr"
    else:
        dur_unit = "min"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4.5))

    bar_width = (max(spk_range) - min(spk_range) + 1.0) / len(spk_range)

    bars1 = ax1.bar(spk_range, counts, width=bar_width, edgecolor="black", alpha=0.75)
    ax1.set_xlabel("#Speakers")
    ax1.set_ylabel("#Sessions")
    ax1.set_title("Session Count")
    ax1.set_xticks(spk_range)
    ax1.set_xlim(min(spk_range) - 0.6, max(spk_range) + 0.6)
    for bar, v in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 str(v), ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(spk_range, durs, width=bar_width, edgecolor="black", alpha=0.75, color="orange")
    ax2.set_xlabel("#Speakers")
    ax2.set_ylabel(f"Total Duration / {dur_unit}")
    ax2.set_title("Total Duration")
    ax2.set_xticks(spk_range)
    ax2.set_xlim(min(spk_range) - 0.6, max(spk_range) + 0.6)
    from matplotlib.ticker import MaxNLocator
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    for bar, v in zip(bars2, durs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Session Statistics by Number of Speakers", fontsize=14)
    plt.tight_layout()
    out_png = output_dir / "stats_by_speakers.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"Total sessions: {sum(counts)}")
    for s, c, d in zip(spk_range, counts, durs):
        print(f"  {s} speaker(s): {c:4d} sessions  total duration {d:8.1f} {dur_unit}")
    print(f"\nPlot saved to {out_png}")


if __name__ == "__main__":
    main()
