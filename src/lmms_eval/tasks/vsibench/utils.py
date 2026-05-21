
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]
if os.path.isdir(dataset_path):
    cache_dir = dataset_path
else:
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)

def vsibench_doc_to_visual(doc):
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    import json
    
    # doc['prediction'] = results[0]
    VGGT_TAG = "<vggt>"
    raw_pred = results[0]
    
    is_vggt = VGGT_TAG in raw_pred
    clean_pred = raw_pred.replace(VGGT_TAG, " ").strip()
    
    doc['prediction'] = clean_pred
    doc['is_vggt_triggered'] = is_vggt
    
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")
    
    if is_vggt:
        # 提取当前样本获得的“分数” (Reward)
        current_score = 0.0
        if doc['question_type'] in MCA_QUESTION_TYPES:
            # MCA 类题目关注 accuracy (0 或 1)
            current_score = doc.get("accuracy", 0.0)
        elif doc['question_type'] in NA_QUESTION_TYPES:
            # NA 类题目关注 MRA (0.0 - 1.0)
            # key 对应 METRICS_FOR_NA 里的定义
            na_key = "MRA:.5:.95:.05"
            current_score = doc.get(na_key, 0.0)
            
        # 实时写入日志文件
        log_entry = {
            "id": doc.get("id", "unknown"),
            "question_type": doc.get("question_type"),
            "is_triggered": True,
            "raw_prediction": raw_pred,     # 包含 <vggt> 的原始输出
            "clean_prediction": clean_pred, # 用于判分的清洗后输出
            "ground_truth": doc.get("ground_truth"),
            "reward_score": current_score   # 这就是你需要的 Reward 值
        }
        
        with open("vggt_vsibench_debug.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    # --- 新增: 统计并打印 VGGT 专属准确率 ---
    if 'is_vggt_triggered' in results.columns:
        # 筛选出触发了 vggt 的样本
        vggt_df = results[results['is_vggt_triggered'] == True]
        
        if len(vggt_df) > 0:
            # 收集所有触发样本的分数
            # 注意：不同题目类型的分数存放字段不同 (accuracy vs MRA)
            vggt_scores = []
            
            for idx, row in vggt_df.iterrows():
                q_type = row['question_type']
                if q_type in MCA_QUESTION_TYPES:
                    vggt_scores.append(row.get("accuracy", 0.0))
                elif q_type in NA_QUESTION_TYPES:
                    vggt_scores.append(row.get("MRA:.5:.95:.05", 0.0))
            
            # 计算平均分
            if vggt_scores:
                avg_score = sum(vggt_scores) / len(vggt_scores)
                print("="*60)
                print(f"[VGGT Tracker] Total Triggered Samples: {len(vggt_df)}")
                print(f"[VGGT Tracker] Average Score (Reward):   {avg_score:.4f} ({avg_score:.2%})")
                print(f"[VGGT Tracker] Detailed log saved to:    vggt_vsibench_debug.jsonl")
                print("="*60)
    # ----------------------------------------------------
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    # output['object_rel_direction_accuracy'] = sum([
    #     output.pop('object_rel_direction_easy_accuracy'),
    #     output.pop('object_rel_direction_medium_accuracy'),
    #     output.pop('object_rel_direction_hard_accuracy'),
    # ]) / 3.
    
    # output['overall'] = sum([_ for _ in output.values()]) / len(output)
    # eval_logger.info(f"Evaluation results: {output}")
    
    
    try:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy', 0),
            output.pop('object_rel_direction_medium_accuracy', 0),
            output.pop('object_rel_direction_hard_accuracy', 0),
        ]) / 3.
    except Exception as e:
        eval_logger.warning(f"Error aggregating object_rel_direction: {e}")

    # 计算 overall
    if len(output) > 0:
        output['overall'] = sum([_ for _ in output.values()]) / len(output)
    else:
        output['overall'] = 0.0

    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
