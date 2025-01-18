# attention_extractor.py

import torch

def get_attention(model, tokenizer, input_text, device):
    """
    对输入文本进行编码，运行模型，并提取注意力权重。

    参数:
        model: 已加载的T5模型。
        tokenizer: T5分词器。
        input_text (str): 输入文本。
        device (str): 计算设备。

    返回:
        attentions: 注意力权重，形状为 (num_layers, num_heads, seq_len, seq_len)。
        tokens: 分词后的token列表。
    """
    # 编码输入
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # 前向传播，获取注意力权重
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)
    
    attentions = outputs.attentions  # Tuple of length num_layers
    attentions = torch.stack(attentions)  # 形状: (num_layers, batch_size, num_heads, seq_len, seq_len)
    attentions = attentions.squeeze(1)  # 移除batch维度: (num_layers, num_heads, seq_len, seq_len)
    
    # 解码token
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return attentions.cpu(), tokens
