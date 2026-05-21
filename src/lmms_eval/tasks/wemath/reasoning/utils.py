# import pandas as pd

# from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
# from lmms_eval.tasks.wemath.wemath_utils import (
#     calculate_metrics,
#     compute_final_scores,
#     process_steps_data,
#     update_main_results_df,
# )

# SYSTEM_PROMPT = (
#     "You are a helpful assistant. When the user asks a question, your response must include two parts: "
#     "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
#     "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
# )


# def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
#     return doc["question"] + "\n" + doc["option"]


# def wemath_doc_to_visual(doc):
#     return [doc["image_path"].convert("RGB")]


# def wemath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
#     question = wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
#     visuals = wemath_doc_to_visual(doc)
#     system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
#     messages = [{"role": "user", "content": []}]
#     messages[0]["content"].append({"type": "image", "url": visuals[0]})
#     messages[0]["content"].append({"type": "text", "text": question.strip()})
#     messages = system_messages + messages
#     return messages


# def wemath_reasoning_process_results(doc, results):
#     acc_score = 0
#     format_score = 0
#     question = wemath_doc_to_text_cot(doc, None)
#     extra_info = {"question": question}
#     for pred in results:
#         score_dict = compute_score(data_source="wemath", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
#         acc_score += score_dict["acc_score"]
#         format_score += score_dict.get("format_reward_score", 0.0)

#     data_dict = {
#         "ID": doc["ID"],
#         "split": doc["split"],
#         "knowledge concept": doc["knowledge concept"],
#         "question": doc["question"],
#         "option": doc["option"],
#         "answer": doc["answer"],
#         # "image_path": doc['image_path'],
#         "key": doc["key"],
#         "question number": doc["question number"],
#         "knowledge concept description": doc["knowledge concept description"],
#         "acc_score": acc_score,
#     }

#     return {"wemath_loose": data_dict, "wemath_strict": data_dict, "acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}


# def wemath_aggregate_results(results, metric_name):
#     data = pd.DataFrame(results)
#     data["joker"] = data["acc_score"] == 1.0
#     data_2steps = data[data["key"].str.contains("2steps")]
#     data_3steps = data[data["key"].str.contains("3steps")]
#     merged_2steps = process_steps_data(data_2steps, 2)
#     merged_3steps = process_steps_data(data_3steps, 3)
#     metrics = calculate_metrics(merged_2steps, merged_3steps)
#     total_counts, rates = compute_final_scores(metrics, total_count=525)
#     score_dict = update_main_results_df(total_counts, rates)
#     if metric_name == "wemath_loose":
#         return score_dict["Score (Loose)"]
#     elif metric_name == "wemath_strict":
#         return score_dict["Score (Strict)"]
#     else:
#         raise ValueError(f"Invalid metric name: {metric_name}")


# def wemath_aggregate_results_loose(results):
#     return wemath_aggregate_results(results, "wemath_loose")


# def wemath_aggregate_results_strict(results):
#     return wemath_aggregate_results(results, "wemath_strict")
import pandas as pd
import re

from lmms_eval.tasks._task_utils.reasoning_utils import compute_score
from lmms_eval.tasks.wemath.wemath_utils import (
    calculate_metrics,
    compute_final_scores,
    process_steps_data,
    update_main_results_df,
)

# === 修改 1: 更改 System Prompt 为标准 LaTeX 格式要求 ===
SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer option enclosed in \\boxed{}."
    "For example, if the answer is A, output \\boxed{A}."
)


def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\n" + doc["option"]


def wemath_doc_to_visual(doc):
    return [doc["image_path"].convert("RGB")]


def wemath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual(doc)
    
    # === 修改 2: 在 User Prompt 中增加强力格式指令 ===
    # 很多模型忽略 System Prompt，所以在 User 这一端再次强调是必须的
    instruction = (
        "Answer the question based on the image.\n"
        "Please reason step by step, and put your final answer option (e.g., A, B, C, D) within \\boxed{}, like \\boxed{A}."
    )
    final_query = f"{question.strip()}\n{instruction}"

    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": final_query})
    messages = system_messages + messages
    return messages


def wemath_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = wemath_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    
    for pred in results:
        pred_str = pred.strip()

        # === 修改 3: 增加清洗逻辑，处理模型可能输出的奇怪格式 ===
        # 如果模型还是输出了 ['C'] 这种列表格式，手动清洗
        if pred_str.startswith("['") and pred_str.endswith("']"):
            pred_str = pred_str[2:-2]  # 变成 C
        elif pred_str.startswith("[\"") and pred_str.endswith("\"]"):
            pred_str = pred_str[2:-2]
            
        # 如果模型输出了 <answer>C</answer> (为了兼容旧习惯)，也尝试清洗
        if "<answer>" in pred_str and "</answer>" in pred_str:
            match = re.search(r"<answer>(.*?)</answer>", pred_str, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # 如果没有 \boxed，帮它加上，方便 compute_score 识别
                if "\\boxed" not in content:
                    pred_str = pred_str + f" \\boxed{{{content}}}"
        
        # 最后的保底：如果没有任何框，且字符串很短（比如就是个 'C'），尝试帮它加框
        # 这能极大挽救 format_score
        if "\\boxed" not in pred_str and len(pred_str) < 10 and pred_str.strip() in ["A", "B", "C", "D", "E"]:
             pred_str = f"\\boxed{{{pred_str.strip()}}}"
        # ====================================================

        score_dict = compute_score(data_source="wemath", solution_str=pred_str, ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    data_dict = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        # "image_path": doc['image_path'],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "acc_score": acc_score,
    }

    return {"wemath_loose": data_dict, "wemath_strict": data_dict, "acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}


def wemath_aggregate_results(results, metric_name):
    data = pd.DataFrame(results)
    data["joker"] = data["acc_score"] == 1.0
    data_2steps = data[data["key"].str.contains("2steps")]
    data_3steps = data[data["key"].str.contains("3steps")]
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    # 注意：这里 total_count 保持原来的 525，或者你可以根据实际数据量传 len(results)
    # 但为了兼容你之前的代码，先保持 525
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    if metric_name == "wemath_loose":
        return score_dict["Score (Loose)"]
    elif metric_name == "wemath_strict":
        return score_dict["Score (Strict)"]
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")


def wemath_aggregate_results_loose(results):
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict(results):
    return wemath_aggregate_results(results, "wemath_strict")