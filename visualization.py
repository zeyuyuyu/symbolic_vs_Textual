# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attn_matrix, tokens, title="Attention Heatmap"):
    """
    绘制注意力矩阵的热力图。

    参数:
        attn_matrix (numpy.ndarray): 注意力矩阵，形状为 (seq_len, seq_len)。
        tokens (list): token列表。
        title (str): 图表标题。
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_multiple_attention_heads(attn_matrices, tokens, layer, heads, problem_id, input_type):
    """
    绘制多个注意力头的热力图。

    参数:
        attn_matrices (torch.Tensor): 注意力矩阵，形状为 (num_layers, num_heads, seq_len, seq_len)。
        tokens (list): token列表。
        layer (int): 要绘制的层数（从0开始）。
        heads (list): 要绘制的头索引列表。
        problem_id (int): 问题ID，用于标题。
        input_type (str): 输入类型，'symbolic'或'verbal'，用于标题。
    """
    num_heads = len(heads)
    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
    
    for i, head in enumerate(heads):
        attn_matrix = attn_matrices[layer, head].numpy()
        sns.heatmap(attn_matrix, ax=axes[i], cmap="viridis")
        axes[i].set_title(f"Layer {layer+1}, Head {head}")
        axes[i].set_xlabel("Key Tokens")
        axes[i].set_ylabel("Query Tokens")
        axes[i].tick_params(axis='x', rotation=90)
        axes[i].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f"Problem {problem_id} - {input_type.capitalize()} Input\nAttention Heads Comparison")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_average_attention_heatmap(attn_matrices, tokens, layer, problem_id, input_type):
    """
    绘制指定层所有头的平均注意力热力图。

    参数:
        attn_matrices (torch.Tensor): 注意力矩阵，形状为 (num_layers, num_heads, seq_len, seq_len)。
        tokens (list): token列表。
        layer (int): 要绘制的层数（从0开始）。
        problem_id (int): 问题ID，用于标题。
        input_type (str): 输入类型，'symbolic'或'verbal'，用于标题。
    """
    # 获取该层所有头的注意力矩阵
    all_heads_attn = attn_matrices[layer].numpy()  # Shape: (num_heads, seq_len, seq_len)
    
    # 计算平均注意力
    avg_attn = all_heads_attn.mean(axis=0)  # Shape: (seq_len, seq_len)
    
    # 绘制热力图
    title = f"Problem {problem_id} - {input_type.capitalize()} Input\nLayer {layer+1} Average Attention"
    plot_attention_heatmap(avg_attn, tokens, title)
