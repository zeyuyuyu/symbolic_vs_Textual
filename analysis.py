# analysis.py

import numpy as np

def calculate_attention_entropy(attn_matrix):
    """
    计算注意力矩阵的平均熵。

    参数:
        attn_matrix (numpy.ndarray): 注意力矩阵，形状为 (seq_len, seq_len)。

    返回:
        float: 平均熵值。
    """
    # 对每一行进行归一化，确保为概率分布
    attn_probs = attn_matrix / (attn_matrix.sum(axis=-1, keepdims=True) + 1e-10)
    
    # 计算熵
    entropy = -np.sum(attn_probs * np.log(attn_probs + 1e-10), axis=-1)  # Shape: (seq_len,)
    
    # 返回平均熵
    return entropy.mean()

def calculate_attention_on_keywords(attn_matrix, keywords_indices):
    """
    计算注意力矩阵中关键token的注意力占比。

    参数:
        attn_matrix (numpy.ndarray): 注意力矩阵，形状为 (seq_len, seq_len)。
        keywords_indices (list): 关键token的索引列表。

    返回:
        float: 关键token的注意力占比。
    """
    if not keywords_indices:
        return 0.0
    
    # 计算所有query对关键key的注意力总和
    key_attention = attn_matrix[:, keywords_indices].sum()
    
    # 计算总注意力
    total_attention = attn_matrix.sum()
    
    # 返回占比
    return key_attention / (total_attention + 1e-10)

def get_keyword_indices(tokens, input_type):
    """
    根据输入类型和token内容，返回关键token的索引。

    参数:
        tokens (list): token列表。
        input_type (str): 'symbolic' 或 'verbal'。

    返回:
        list: 关键token的索引列表。
    """
    keywords = []
    if input_type == "symbolic":
        # 定义符号化输入中的关键符号
        symbolic_keywords = {"x", "y", "a", "b", "m", "n", "=", "+", "-", "*", "/", "求"}
        for idx, token in enumerate(tokens):
            # 去除特殊字符后检查是否在关键符号集合中
            token_clean = token.strip("=+-*/")
            if token_clean in symbolic_keywords or token in symbolic_keywords:
                keywords.append(idx)
    elif input_type == "verbal":
        # 定义文字描述输入中的关键词
        verbal_keywords = {"求", "加", "减", "乘", "除", "等于"}
        for idx, token in enumerate(tokens):
            if token in verbal_keywords:
                keywords.append(idx)
    return keywords

def analyze_attention(attn_matrix, tokens, input_type):
    """
    分析注意力矩阵，计算熵和关键token的注意力占比。

    参数:
        attn_matrix (numpy.ndarray): 注意力矩阵，形状为 (seq_len, seq_len)。
        tokens (list): token列表。
        input_type (str): 'symbolic' 或 'verbal'。

    返回:
        dict: 包含熵和关键token占比的字典。
    """
    entropy = calculate_attention_entropy(attn_matrix)
    keywords_indices = get_keyword_indices(tokens, input_type)
    keyword_ratio = calculate_attention_on_keywords(attn_matrix, keywords_indices)
    return {
        "attention_entropy": entropy,
        "keyword_attention_ratio": keyword_ratio
    }
