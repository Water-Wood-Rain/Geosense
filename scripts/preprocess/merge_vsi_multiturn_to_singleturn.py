#!/usr/bin/env python3
"""
Convert 2-turn VSI-style training samples back into single-turn samples.

For samples with conversations like:
  human1 -> gpt1(with <vggt>) -> human2(with <image><vggt>) -> gpt2(final answer)
this script rewrites them into:
  human1 -> gpt1 + ". My answer is " + gpt2

More precisely, it inserts the second-turn assistant answer after the first
assistant message's <vggt> tag, e.g.:
  "...<vggt>" -> "...<vggt>. My answer is 1"

Single-turn samples are kept unchanged.

Usage:
  python scripts/preprocess/merge_vsi_multiturn_to_singleturn.py
  python scripts/preprocess/merge_vsi_multiturn_to_singleturn.py \
      -i data/train/vsi_tfft_recon.json \
      -o data/train/vsi_tfft_recon_singleturn.json
"""

import argparse
import copy
import json
from pathlib import Path

VGGT_TAG = "<vggt>"
DEFAULT_SUFFIX_TEMPLATE = ". My answer is {answer}"


def merge_sample(sample: dict, suffix_template: str = DEFAULT_SUFFIX_TEMPLATE):
    convs = sample.get("conversations", [])

    # Only rewrite exact 2-turn structure: [human, gpt, human, gpt]
    if len(convs) != 4:
        return sample, False

    if convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
        return sample, False
    if convs[2].get("from") != "human" or convs[3].get("from") != "gpt":
        return sample, False

    first_gpt = convs[1].get("value", "")
    second_gpt = convs[3].get("value", "")

    if VGGT_TAG not in first_gpt:
        return sample, False

    merged = copy.deepcopy(sample)
    suffix = suffix_template.format(answer=second_gpt)
    merged_first_gpt = first_gpt.replace(VGGT_TAG, VGGT_TAG + suffix, 1)

    merged["conversations"] = [
        copy.deepcopy(convs[0]),
        {"from": "gpt", "value": merged_first_gpt},
    ]
    return merged, True


def turn_count(sample: dict) -> int:
    return len(sample.get("conversations", [])) // 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        default="data/train/vsi_tfft_recon.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file (default: <input_stem>_singleturn.json)",
    )
    parser.add_argument(
        "--suffix-template",
        default=DEFAULT_SUFFIX_TEMPLATE,
        help='Text inserted after <vggt>; use {answer} as placeholder. Default: ". My answer is {answer}"',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_singleturn.json")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    converted_count = 0
    for sample in data:
        new_sample, changed = merge_sample(sample, suffix_template=args.suffix_template)
        converted.append(new_sample)
        converted_count += int(changed)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    before_dist = {}
    after_dist = {}
    for s in data:
        before_dist[turn_count(s)] = before_dist.get(turn_count(s), 0) + 1
    for s in converted:
        after_dist[turn_count(s)] = after_dist.get(turn_count(s), 0) + 1

    print(f"Input : {input_path} ({len(data)} samples)")
    print(f"Output: {output_path} ({len(converted)} samples)")
    print(f"Converted 2-turn -> 1-turn samples: {converted_count}")
    print(f"Turn distribution before: {dict(sorted(before_dist.items()))}")
    print(f"Turn distribution after : {dict(sorted(after_dist.items()))}")


if __name__ == "__main__":
    main()
