# from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

# SYSTEM_PROMPT = (
#     "You are a helpful assistant. When the user asks a question, your response must include two parts: "
#     "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
#     "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
# )


# def logicvista_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
#     return doc["question"]


# def logicvista_doc_to_visual(doc):
#     return [doc["image"].convert("RGB")]


# def logicvista_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
#     question = logicvista_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
#     visuals = logicvista_doc_to_visual(doc)
#     system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
#     messages = [{"role": "user", "content": []}]
#     messages[0]["content"].append({"type": "image", "url": visuals[0]})
#     messages[0]["content"].append({"type": "text", "text": question.strip()})
#     messages = system_messages + messages
#     return messages


# def logicvista_reasoning_process_results(doc, results):
#     acc_score = 0
#     format_score = 0
#     question = logicvista_doc_to_text_cot(doc, None)
#     extra_info = {"question": question}
#     for pred in results:
#         score_dict = compute_score(data_source="logicvista", solution_str=pred.strip(), ground_truth=doc["answer"], extra_info=extra_info)
#         acc_score += score_dict["acc_score"]
#         format_score += score_dict.get("format_reward_score", 0.0)

#     return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}

import re
from lmms_eval.tasks._task_utils.reasoning_utils import compute_score

# === 修改 1: 将 System Prompt 改为要求 \boxed{} 格式 ===
SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer option enclosed in \\boxed{}."
    "For example, if the answer is Option A, output \\boxed{A}."
)


def logicvista_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def logicvista_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def logicvista_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = logicvista_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = logicvista_doc_to_visual(doc)
    
    # === 修改 2: 在 User 输入中强制追加格式指令 ===
    # 很多模型忽略 System Prompt，必须在 User 层面再次强调
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


def logicvista_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    question = logicvista_doc_to_text_cot(doc, None)
    extra_info = {"question": question}
    
    for pred in results:
        pred_str = pred.strip()

        # === 修改 3: 增加清洗逻辑 (Critical Fix) ===
        
        # 1. 清洗列表符号 ['A'] -> A
        if pred_str.startswith("['") and pred_str.endswith("']"):
            pred_str = pred_str[2:-2]
        elif pred_str.startswith("[\"") and pred_str.endswith("\"]"):
            pred_str = pred_str[2:-2]
            
        # 2. 兼容 XML 标签 <answer>A</answer> (如果模型还是输出了旧格式)
        if "<answer>" in pred_str and "</answer>" in pred_str:
            match = re.search(r"<answer>(.*?)</answer>", pred_str, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # 提取后如果没框，手动加框
                if "\\boxed" not in content:
                    pred_str = pred_str + f" \\boxed{{{content}}}"
        
        # 3. 最后的保底：如果字符串很短且是单个字母 (A-E)，手动加框
        # compute_score 对 \boxed{A} 的识别率远高于光秃秃的 A
        if "\\boxed" not in pred_str and len(pred_str) < 10:
             clean_content = pred_str.strip()
             # LogicVista 的答案通常是 A, B, C, D 或 1, 2, 3, 4
             if clean_content in ["A", "B", "C", "D", "E", "1", "2", "3", "4"]:
                 pred_str = f"\\boxed{{{clean_content}}}"
        
        # ============================================

        score_dict = compute_score(data_source="logicvista", solution_str=pred_str, ground_truth=doc["answer"], extra_info=extra_info)
        acc_score += score_dict["acc_score"]
        format_score += score_dict.get("format_reward_score", 0.0)

    return {"acc_score": acc_score / len(results) if results else 0.0, "format_score": format_score / len(results) if results else 0.0}