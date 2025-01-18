from transformers import T5EncoderModel, T5Tokenizer

# 选择模型名称
model_name = "t5-base"

# 加载分词器和编码器模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# 获取编码器的配置
encoder_config = model.config

# 打印编码器层数
print(f"模型 '{model_name}' 的编码器层数为: {encoder_config.num_layers} 层")
