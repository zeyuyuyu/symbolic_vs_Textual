# model_loader.py

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(model_name="t5-base", device=None):
    """
    加载T5模型和分词器。

    参数:
        model_name (str): 预训练模型名称，默认为't5-base'。
        device (str): 计算设备，'cuda'或'cpu'。如果为None，则自动检测。

    返回:
        tokenizer: T5分词器。
        model: T5模型，设置为评估模式。
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
    
    if device:
        model.to(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    
    model.eval()
    return tokenizer, model, device
