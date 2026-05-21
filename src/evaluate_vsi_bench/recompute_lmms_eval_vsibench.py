import argparse
import json
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np


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

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}

VGGT_TAG = "<vggt>"
NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
ANSWER_PATTERNS = [
    r"my\s+answer\s+is\s*[:：]?\s*([-+]?\d*\.?\d+|[A-Za-z])",
    r"answer\s*(?:is|:|：)\s*([-+]?\d*\.?\d+|[A-Za-z])",
    r"final\s+answer\s*(?:is|:|：)\s*([-+]?\d*\.?\d+|[A-Za-z])",
]


def fuzzy_matching(pred):
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred):
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def get_raw_prediction(record):
    filtered_resps = record.get("filtered_resps") or []
    if filtered_resps and isinstance(filtered_resps[0], str):
        return filtered_resps[0]

    resps = record.get("resps") or []
    if resps and isinstance(resps[0], list) and resps[0] and isinstance(resps[0][0], str):
        return resps[0][0]

    doc = record.get("vsibench_score") or record.get("doc") or {}
    prediction = doc.get("prediction")
    return prediction if isinstance(prediction, str) else ""


def clean_prediction(raw_pred):
    return raw_pred.replace(VGGT_TAG, " ").strip()


def extract_choice_answer(text, options=None):
    valid_letters = []
    if isinstance(options, list) and options:
        valid_letters = [chr(ord("A") + i) for i in range(len(options))]
    else:
        valid_letters = [chr(ord("A") + i) for i in range(8)]

    upper_text = text.upper()

    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, upper_text, flags=re.IGNORECASE)
        for match in reversed(matches):
            match = match.upper()
            if match in valid_letters:
                return match

    for match in reversed(re.findall(r"\b([A-Z])\b", upper_text)):
        if match in valid_letters:
            return match

    return fuzzy_matching(text).upper() or None


def extract_numeric_answer(text):
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in reversed(matches):
            if NUMBER_PATTERN.fullmatch(match):
                return match

    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        return numbers[-1]

    return fuzzy_matching(text) or None


def extract_effective_answer(question_type, clean_pred, options=None):
    if question_type in MCA_QUESTION_TYPES:
        return extract_choice_answer(clean_pred, options)
    if question_type in NA_QUESTION_TYPES:
        return extract_numeric_answer(clean_pred)
    return fuzzy_matching(clean_pred)


def score_doc(doc, raw_pred):
    doc = dict(doc)
    clean_pred = clean_prediction(raw_pred)
    extracted_pred = extract_effective_answer(doc.get("question_type"), clean_pred, doc.get("options"))

    doc["prediction"] = extracted_pred
    doc["raw_prediction"] = raw_pred
    doc["clean_prediction"] = clean_pred
    doc["is_vggt_triggered"] = VGGT_TAG in raw_pred

    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(extracted_pred or "", str(doc["ground_truth"]))
        reward_score = doc.get("accuracy", 0.0)
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(extracted_pred), to_float(doc["ground_truth"]))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
        reward_score = doc.get("MRA:.5:.95:.05", 0.0)
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {
        "id": doc.get("id", "unknown"),
        "question_type": doc.get("question_type"),
        "ground_truth": doc.get("ground_truth"),
        "raw_prediction": raw_pred,
        "clean_prediction": clean_pred,
        "extracted_prediction": extracted_pred,
        "is_vggt_triggered": doc.get("is_vggt_triggered", False),
        "reward_score": reward_score,
        "scored_doc": doc,
    }


def aggregate_results(scored_docs):
    grouped = defaultdict(list)
    for item in scored_docs:
        grouped[item["question_type"]].append(item["scored_doc"])

    output = {}
    for question_type, docs in grouped.items():
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = sum(doc.get(metric, 0.0) for doc in docs) / len(docs)
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = sum(doc.get(metric, 0.0) for doc in docs) / len(docs)
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    output["object_rel_direction_accuracy"] = sum([
        output.pop("object_rel_direction_easy_accuracy", 0),
        output.pop("object_rel_direction_medium_accuracy", 0),
        output.pop("object_rel_direction_hard_accuracy", 0),
    ]) / 3.0

    if len(output) > 0:
        output["overall"] = sum(output.values()) / len(output)
    else:
        output["overall"] = 0.0

    triggered = [item for item in scored_docs if item["is_vggt_triggered"]]
    triggered_avg = sum(item["reward_score"] for item in triggered) / len(triggered) if triggered else 0.0

    return {
        "metrics": output,
        "total_samples": len(scored_docs),
        "triggered_samples": len(triggered),
        "triggered_average_reward": triggered_avg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recompute VSI-Bench scores from lmms-eval samples_vsibench.jsonl by re-extracting answers."
    )
    parser.add_argument("input_jsonl", help="Path to samples_vsibench.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        help="Output jsonl path. Defaults to <input_dir>/vsibench_recomputed.jsonl",
    )
    parser.add_argument(
        "-s",
        "--summary",
        help="Summary json path. Defaults to <input_dir>/vsibench_recomputed_summary.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output) if args.output else input_path.with_name("vsibench_recomputed.jsonl")
    summary_path = Path(args.summary) if args.summary else input_path.with_name("vsibench_recomputed_summary.json")

    scored_docs = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc = record.get("vsibench_score") or record.get("doc") or {}
            raw_pred = get_raw_prediction(record)
            scored = score_doc(doc, raw_pred)
            scored_docs.append(scored)
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")

    summary = aggregate_results(scored_docs)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Read {len(scored_docs)} records from {input_path}")
    print(f"Wrote recomputed samples to {output_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Overall: {summary['metrics']['overall'] * 100:.4f}")
    print(f"Triggered average reward: {summary['triggered_average_reward'] * 100:.4f}")


if __name__ == "__main__":
    main()
