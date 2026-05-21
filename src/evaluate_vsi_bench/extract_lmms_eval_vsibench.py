import argparse
import json
from pathlib import Path


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}

NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}

NA_METRIC_KEY = "MRA:.5:.95:.05"
VGGT_TAG = "<vggt>"


def extract_entry(record: dict) -> dict | None:
    score = record.get("vsibench_score") or record.get("doc") or {}
    raw_pred = ""

    filtered_resps = record.get("filtered_resps") or []
    if filtered_resps:
        raw_pred = filtered_resps[0]
    else:
        resps = record.get("resps") or []
        if resps and isinstance(resps[0], list) and resps[0]:
            raw_pred = resps[0][0]

    if not isinstance(raw_pred, str) or VGGT_TAG not in raw_pred:
        return None

    clean_pred = raw_pred.replace(VGGT_TAG, " ").strip()
    question_type = score.get("question_type")

    if question_type in MCA_QUESTION_TYPES:
        reward_score = float(score.get("accuracy", 0.0))
    elif question_type in NA_QUESTION_TYPES:
        reward_score = float(score.get(NA_METRIC_KEY, 0.0))
    else:
        reward_score = 0.0

    return {
        "id": score.get("id", "unknown"),
        "question_type": question_type,
        "is_triggered": True,
        "raw_prediction": raw_pred,
        "clean_prediction": clean_pred,
        "ground_truth": score.get("ground_truth"),
        "reward_score": reward_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract VGGT-triggered VSI-Bench records from lmms-eval samples jsonl."
    )
    parser.add_argument("input_jsonl", help="Path to lmms-eval samples_vsibench.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSONL path. Defaults to <input_dir>/vsibench_extract.jsonl",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output) if args.output else input_path.with_name("vsibench_extract.jsonl")

    kept = 0
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            entry = extract_entry(record)
            if entry is None:
                continue
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Read {total} records from {input_path}")
    print(f"Wrote {kept} VGGT-triggered records to {output_path}")


if __name__ == "__main__":
    main()
