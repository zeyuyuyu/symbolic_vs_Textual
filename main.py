# main.py

import pickle
from data import problems
from model_loader import load_model
from attention_extractor import get_attention
from visualization import plot_attention_heatmap, plot_multiple_attention_heads, plot_average_attention_heatmap
from analysis import analyze_attention
import os

def main():
    # 1. 加载模型和分词器
    tokenizer, model, device = load_model(model_name="t5-base")
    print(f"使用设备: {device}")
    
    # 2. 定义要分析的层和头
    # 可以选择多个层，例如第6层到第12层
    target_layers = [5, 6, 7, 8, 9, 10, 11]  # 索引从0开始
    heads_to_plot = [0, 1, 2, 3]  # 可以根据需要调整
    
    # 3. 创建存储目录
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # 4. 处理每个问题
    for problem in problems:
        pid = problem["id"]
        print(f"处理问题 ID: {pid}")
        
        for input_type in ["symbolic", "verbal"]:
            input_text = problem[input_type]
            attentions, tokens = get_attention(model, tokenizer, input_text, device)
            
            for layer in target_layers:
                # 选择特定层的注意力矩阵
                layer_attn = attentions[layer]  # Shape: (num_heads, seq_len, seq_len)
                
                # 4.1 可视化单个注意力头
                head = heads_to_plot[0]
                attn_matrix = layer_attn[head].numpy()
                title = f"Problem {pid} - {input_type.capitalize()} Input\nLayer {layer+1}, Head {head}"
                plot_attention_heatmap(attn_matrix, tokens, title)
                
                # 4.2 可视化多个注意力头
                plot_multiple_attention_heads(attn_matrices=attentions, tokens=tokens, layer=layer, heads=heads_to_plot, problem_id=pid, input_type=input_type)
                
                # 4.3 可视化平均注意力
                plot_average_attention_heatmap(attn_matrices=attentions, tokens=tokens, layer=layer, problem_id=pid, input_type=input_type)
                
                # 4.4 分析注意力熵和关键token占比
                # 计算平均注意力矩阵
                avg_attn = layer_attn.mean(dim=0).numpy()  # Shape: (seq_len, seq_len)
                analysis_result = analyze_attention(avg_attn, tokens, input_type)
                
                # 输出分析结果
                print(f"Problem {pid} - {input_type.capitalize()} Input - Layer {layer+1}:")
                print(f"  Attention Entropy: {analysis_result['attention_entropy']:.4f}")
                print(f"  Keyword Attention Ratio: {analysis_result['keyword_attention_ratio']:.4f}\n")
                
                # 4.5 保存分析结果
                result = {
                    "tokens": tokens,
                    "attention_entropy": analysis_result["attention_entropy"],
                    "keyword_attention_ratio": analysis_result["keyword_attention_ratio"]
                }
                with open(f"results/problem_{pid}_{input_type}_layer_{layer+1}_analysis.pkl", "wb") as f:
                    pickle.dump(result, f)
    
    print("实验完成。分析结果保存在 'results' 目录下。")

if __name__ == "__main__":
    main()

