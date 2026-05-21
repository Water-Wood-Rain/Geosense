#!/usr/bin/env python3
"""
Filter samples whose id starts with "vsi_" from a JSON training file,
and report the distribution of conversation turns in a target file.

Sub-commands:
  filter   Filter by ID prefix and save (default when no sub-command given)
  stat     Show turn-count distribution of an existing JSON file

Usage:
    # filter
    python scripts/preprocess/filter_vsi_samples.py filter
    python scripts/preprocess/filter_vsi_samples.py filter -i <input> -o <output> --prefix vsi_

    # stat
    python scripts/preprocess/filter_vsi_samples.py stat -i data/train/vsi_tfft_recon.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def cmd_filter(args):
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_vsi.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [s for s in data if str(s.get("id", "")).startswith(args.prefix)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Input : {input_path}  ({len(data)} samples)")
    print(f"Output: {output_path}  ({len(filtered)} kept, {len(data) - len(filtered)} removed)")
    cmd_stat_data(filtered, label=str(output_path))


def cmd_stat(args):
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cmd_stat_data(data, label=str(input_path))


def cmd_stat_data(data, label="data"):
    """Print turn-count distribution. A 'turn' = one human+gpt pair."""
    turn_counts = []
    for s in data:
        convs = s.get("conversations", [])
        # count complete human/gpt pairs
        pairs = sum(
            1 for i in range(0, len(convs) - 1, 1)
            if convs[i].get("from") == "human" and convs[i + 1].get("from") == "gpt"
        )
        # simpler: number of messages / 2 rounded down
        pairs = len(convs) // 2
        turn_counts.append(pairs)

    dist = Counter(turn_counts)
    max_turns = max(dist)
    total = len(data)

    print(f"\n=== Turn distribution: {label} ({total} samples) ===")
    print(f"{'Turns':>6}  {'Count':>8}  {'Ratio':>7}")
    print("-" * 28)
    for turns in sorted(dist):
        count = dist[turns]
        print(f"{turns:>6}  {count:>8}  {count/total:>6.1%}")
    print("-" * 28)
    print(f"{'max':>6}  {max_turns:>8}")
    print(f"{'total':>6}  {total:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd")

    # --- filter ---
    p_filter = sub.add_parser("filter", help="Filter by ID prefix and save")
    p_filter.add_argument("-i", "--input", default="data/train/spatial_general_shuffle_v3.json")
    p_filter.add_argument("-o", "--output", default=None)
    p_filter.add_argument("--prefix", default="vsi_")

    # --- stat ---
    p_stat = sub.add_parser("stat", help="Show turn-count distribution")
    p_stat.add_argument("-i", "--input", default="data/train/vsi_tfft_recon.json")

    args = parser.parse_args()

    if args.cmd == "stat":
        cmd_stat(args)
    else:
        # default: filter (also prints stat of output)
        if args.cmd != "filter":
            # called with no sub-command: parse as filter with remaining argv
            args = p_filter.parse_args([] if args.cmd is None else [])
        cmd_filter(args)


if __name__ == "__main__":
    main()
