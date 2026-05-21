import os
import re
import math

# ==========================================
# 1. 路径配置 (请修改这里!)
# ==========================================
# 你的视频文件夹绝对路径，确保里面是 simulation_0300.mp4 这种文件
VIDEO_ROOT = "data/evaluation/quantiphy/validation_videos" 


# ==========================================
# 2. Input 处理函数
# ==========================================

def quantiphy_doc_to_visual(doc):
    """
    将 video_id 转换为视频的绝对路径
    """
    video_id = doc['video_id']
    video_path = os.path.join(VIDEO_ROOT, f"{video_id}.mp4")
    
    # 可选：检查文件是否存在，防止报错
    if not os.path.exists(video_path):
        print(f"Warning: Video not found at {video_path}")
        
    return [video_path] # lmms-eval 通常期望返回一个列表

def quantiphy_doc_to_text(doc):
    """
    构建 Prompt：必须包含 Prior (物理前提)
    """
    question = doc['question']
    prior = doc['prior']
    
    # 你的数据集中 prior 可能为 None 或字符串
    # 如果有 prior 信息，必须拼接到问题前面
    if prior and str(prior).lower() != 'none' and str(prior).strip() != "":
        prompt = f"Context: {prior}\nQuestion: {question}\nAnswer the question with a single number."
    else:
        prompt = f"Question: {question}\nAnswer the question with a single number."
        
    return prompt


# ==========================================
# 3. 核心评分逻辑 (已修正为官方算法)
# ==========================================

def extract_number(text):
    """
    提取最后一个数字。
    """
    if not text:
        return None
    # 匹配浮点数或整数
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        try:
            val = float(numbers[-1])
            return val
        except ValueError:
            return None
    return None

def quantiphy_process_results(doc, results):
    """
    官方 MRA 计算逻辑复刻：
    C = [0.1, 0.2, ..., 0.95]
    Score = sum( error < (1-theta) ) / 10
    """
    pred_text = results[0]
    pred_val = extract_number(pred_text)
    
    # 注意：evaluator.py 中从 dataset 读取的已经是 float
    try:
        gt_val = float(doc['answer'])
    except (ValueError, TypeError):
        # 如果 GT 本身有问题，为了不报错跳过，返回 0
        return {"quantiphy_mra": 0.0}

    # 官方逻辑：如果解析不出数字，或者解析结果为 0，或者 GT 为 0，都视为无效/无法计算，得 0 分
    # 原码引用: zero_values = (parsed_numeric == 0).sum() -> 视为 invalid
    if pred_val is None or pred_val == 0 or gt_val == 0:
        return {"quantiphy_mra": 0.0}

    # 计算相对误差
    # 原码: abs(row['parsed_value'] - row['ground_truth_posterior']) / row['ground_truth_posterior']
    relative_error = abs(pred_val - gt_val) / abs(gt_val)

    # 官方定义的阈值列表 C
    # theta=0.95 意味着 1-0.95=0.05 (5% 误差容忍度)
    # theta=0.10 意味着 1-0.10=0.90 (90% 误差容忍度)
    C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    passed_count = 0
    for theta in C:
        # 原码: < (1 - theta)
        allowable_error = 1.0 - theta
        if relative_error < allowable_error:
            passed_count += 1
            
    # 最终分数是 0 到 1 之间的浮点数
    final_score = passed_count / 10.0

    return {
        "quantiphy_mra": final_score
    }

def quantiphy_aggregate_results(results):
    """
    计算所有样本的平均 MRA
    """
    # 过滤掉 None (如果有的话)，计算平均值
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return 0.0
    return sum(valid_results) / len(valid_results)