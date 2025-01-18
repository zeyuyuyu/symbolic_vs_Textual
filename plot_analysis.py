# plot_analysis.py

import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_pkl(file_path):
    """
    加载.pkl文件并返回其内容。
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def collect_data(results_dir="results"):
    """
    从指定目录加载所有.pkl文件，并整理成DataFrame。
    """
    records = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".pkl"):
            file_path = os.path.join(results_dir, filename)
            data = load_pkl(file_path)
            
            # 从文件名提取信息
            parts = filename.rstrip(".pkl").split("_")
            # 假设文件名格式为: problem_<id>_<input_type>_layer_<layer_number>_analysis.pkl
            # 例如: problem_1_symbolic_layer_6_analysis.pkl
            if len(parts) >= 5:
                pid = parts[1]
                input_type = parts[2]
                layer = parts[4]
            else:
                pid = "Unknown"
                input_type = "Unknown"
                layer = "Unknown"
            
            record = {
                "Problem_ID": pid,
                "Input_Type": input_type,
                "Layer": layer,
                "Attention_Entropy": data["attention_entropy"],
                "Keyword_Attention_Ratio": data["keyword_attention_ratio"]
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    return df

def plot_bar(df, metric, title, ylabel, save_path):
    """
    绘制指定指标的条形图，标注数值，并保存为图片文件。
    
    参数:
        df (DataFrame): 包含分析数据的DataFrame。
        metric (str): 要绘制的指标列名（例如 'Attention_Entropy'）。
        title (str): 图表标题。
        ylabel (str): y轴标签。
        save_path (str): 图片保存路径。
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Problem_ID", y=metric, hue="Input_Type", data=df, palette="viridis")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Problem ID")
    plt.legend(title="Input Type")
    plt.tight_layout()
    
    # 标注数值
    for p in ax.patches:
        height = p.get_height()
        if pd.notna(height):
            ax.annotate(f'{height:.4f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    
    plt.savefig(save_path)
    plt.close()  # 关闭图形以释放内存

def main():
    results_dir = "results"
    plots_dir = "plots"
    
    # 创建保存图片的目录（如果不存在）
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    df = collect_data(results_dir)
    
    # 检查是否有数据
    if df.empty:
        print("没有找到有效的.pkl文件或数据为空。")
        return
    
    # 获取所有独特的层
    layers = df["Layer"].unique()
    
    for layer in layers:
        layer_df = df[df["Layer"] == layer]
        
        # 绘制 Attention Entropy 条形图
        entropy_title = f"Attention Entropy by Problem and Input Type (Layer {layer})"
        entropy_ylabel = "Attention Entropy"
        entropy_save_path = os.path.join(plots_dir, f"attention_entropy_layer_{layer}.png")
        plot_bar(
            df=layer_df,
            metric="Attention_Entropy",
            title=entropy_title,
            ylabel=entropy_ylabel,
            save_path=entropy_save_path
        )
        print(f"Attention Entropy 图表已保存至 {entropy_save_path}")
        
        # 绘制 Keyword Attention Ratio 条形图
        ratio_title = f"Keyword Attention Ratio by Problem and Input Type (Layer {layer})"
        ratio_ylabel = "Keyword Attention Ratio"
        ratio_save_path = os.path.join(plots_dir, f"keyword_attention_ratio_layer_{layer}.png")
        plot_bar(
            df=layer_df,
            metric="Keyword_Attention_Ratio",
            title=ratio_title,
            ylabel=ratio_ylabel,
            save_path=ratio_save_path
        )
        print(f"Keyword Attention Ratio 图表已保存至 {ratio_save_path}")
    
    # 保存整个DataFrame为CSV
    csv_save_path = "attention_analysis_summary.csv"
    df.to_csv(csv_save_path, index=False)
    print(f"分析摘要已保存为 '{csv_save_path}'。")

if __name__ == "__main__":
    main()
