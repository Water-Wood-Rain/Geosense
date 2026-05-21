import argparse
import json
import re
from pathlib import Path


ANSWER_PREFIXES = [
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "My answer is",
    "Final answer is",
    "Answer:",
]

VGGT_TAG = "<vggt>"
LETTER_PATTERN = re.compile(r"[ABCDEF]")


def extract_characters_regex(text: str) -> str:
    text = text.strip()
    for prefix in ANSWER_PREFIXES:
        text = text.replace(prefix, "")

    if len(text.split()) > 10 and not LETTER_PATTERN.search(text):
        return ""

    match = LETTER_PATTERN.search(text)
    if match is None:
        return ""
    return match[0]


def get_raw_prediction(record: dict) -> str:
    filtered_resps = record.get("filtered_resps") or []
    if filtered_resps and isinstance(filtered_resps[0], str):
        return filtered_resps[0]

    resps = record.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0] and isinstance(resps[0][0], str):
        return resps[0][0]

    score = record.get("cvbench_score") or record.get("doc") or {}
    pred = score.get("pred_answer")
    return pred if isinstance(pred, str) else ""


def clean_prediction(raw_pred: str) -> str:
    return raw_pred.replace(VGGT_TAG, " ").strip()


def score_record(record: dict) -> dict:
    doc = dict(record.get("cvbench_score") or record.get("doc") or {})
    raw_pred = get_raw_prediction(record)
    clean_pred = clean_prediction(raw_pred)
    extracted = extract_characters_regex(clean_pred)
    target = str(doc.get("answer", ""))
    target_letter = target[1] if len(target) >= 2 else ""
    result = 1 if extracted == target_letter else 0
    is_triggered = VGGT_TAG in raw_pred

    doc["pred_answer"] = extracted
    doc["result"] = result
    doc["raw_prediction"] = raw_pred
    doc["clean_prediction"] = clean_pred
    doc["is_vggt_triggered"] = is_triggered

    return {
        "idx": doc.get("idx"),
        "source": doc.get("source"),
        "task": doc.get("task"),
        "answer": target,
        "target_letter": target_letter,
        "raw_prediction": raw_pred,
        "clean_prediction": clean_pred,
        "extracted_prediction": extracted,
        "is_triggered": is_triggered,
        "result": result,
        "scored_doc": doc,
    }


def mean_or_zero(values):
    return sum(values) / len(values) if values else 0.0


def aggregate_results(scored_records: list[dict]) -> dict:
    ade = [r["result"] for r in scored_records if r["source"] == "ADE20K"]
    coco = [r["result"] for r in scored_records if r["source"] == "COCO"]
    omni = [r["result"] for r in scored_records if r["source"] == "Omni3D"]

    accuracy_2d_ade = mean_or_zero(ade)
    accuracy_2d_coco = mean_or_zero(coco)
    accuracy_3d_omni = mean_or_zero(omni)

    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    by_task = {}
    for task in ["Count", "Relation", "Distance", "Depth"]:
        by_task[task] = mean_or_zero([r["result"] for r in scored_records if r["task"] == task])

    triggered = [r for r in scored_records if r["is_triggered"]]

    return {
        "metrics": {
            "accuracy_2d": accuracy_2d,
            "accuracy_3d": accuracy_3d,
            "combined_accuracy": combined_accuracy,
            **by_task,
        },
        "total_samples": len(scored_records),
        "triggered_samples": len(triggered),
        "trigger_rate": len(triggered) / len(scored_records) if scored_records else 0.0,
        "triggered_accuracy": mean_or_zero([r["result"] for r in triggered]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute CVBench accuracy from lmms-eval samples_cvbench.jsonl."
    )
    parser.add_argument("input_jsonl", help="Path to samples_cvbench.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="Output jsonl path. Defaults to <input_dir>/cvbench_recomputed.jsonl",
    )
    parser.add_argument(
        "-s",
        "--summary",
        help="Summary json path. Defaults to <input_dir>/cvbench_recomputed_summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output) if args.output else input_path.with_name("cvbench_recomputed.jsonl")
    summary_path = Path(args.summary) if args.summary else input_path.with_name("cvbench_recomputed_summary.json")

    scored_records = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scored = score_record(record)
            scored_records.append(scored)
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")

    summary = aggregate_results(scored_records)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Read {len(scored_records)} records from {input_path}")
    print(f"Wrote recomputed samples to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Combined accuracy: {summary['metrics']['combined_accuracy'] * 100:.4f}")
    print(f"Trigger rate: {summary['trigger_rate'] * 100:.4f}")
    print(f"Triggered accuracy: {summary['triggered_accuracy'] * 100:.4f}")


if __name__ == "__main__":
    main()
