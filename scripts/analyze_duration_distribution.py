import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to nemo_manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if args.output_dir is None:
        args.output_dir = Path(args.manifest).parent

    output_dir = Path(args.output_dir) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    durations = []
    with open(manifest_path) as f:
        for line in f:
            s = json.loads(line)
            durations.append(s["duration"])

    durations = np.array(durations)
    total_hours = durations.sum() / 3600.0
    mean_dur = durations.mean()
    median_dur = np.median(durations)
    min_dur = durations.min()
    max_dur = durations.max()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durations, bins=args.bins, edgecolor="black", alpha=0.75, color="steelblue")
    ax.axvline(mean_dur, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_dur:.1f}s")
    ax.axvline(median_dur, color="green", linestyle="--", linewidth=2, label=f"Median: {median_dur:.1f}s")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("#Sessions")
    ax.set_title("Audio Duration Distribution")
    ax.legend()
    plt.tight_layout()
    out_png = output_dir / "duration_histogram.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    thresholds = [60, 90, 120, 150, 180]
    lines = []
    lines.append(f"Total sessions: {len(durations)}")
    lines.append(f"Total duration: {total_hours:.2f} hours")
    lines.append(f"Mean duration:  {mean_dur:.2f} s")
    lines.append(f"Median duration:{median_dur:.2f} s")
    lines.append(f"Min duration:   {min_dur:.2f} s")
    lines.append(f"Max duration:   {max_dur:.2f} s")
    lines.append("")
    lines.append("Cumulative Probability Distribution (CPD)")
    for t in thresholds:
        prob = (durations < t).mean()
        line = f"  P(duration < {t}s) = {prob:.4f} ({prob * 100:.2f}%)"
        lines.append(line)
    lines.append("")
    lines.append(f"Plot saved to {out_png}")

    out_txt = output_dir / "duration_statistics.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    for line in lines:
        print(line)
    print(f"\nStatistics saved to {out_txt}")


if __name__ == "__main__":
    main()
