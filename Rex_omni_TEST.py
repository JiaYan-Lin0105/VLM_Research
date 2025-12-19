import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests

# 开启详细错误追踪，防止报错信息被掩盖
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. 加载处理器和模型
model_path = "./models/Rex_omni"
# 使用 use_fast=False 避开你之前看到的 Fast Processor 警告，增加稳定性
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
    # 修正 torch_dtype 警告，使用 dtype
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto" if device == "cuda" else None,
)

# --- 关键修正：强制对齐 Embedding 层 ---
# 如果微调时新增了 Token，这里必须手动调整模型大小，否则就会报错 IndexError
if len(processor.tokenizer) > model.config.vocab_size:
    print(f"检测到词表扩充: {model.config.vocab_size} -> {len(processor.tokenizer)}")
    model.resize_token_embeddings(len(processor.tokenizer))

# 2. 准备图像 (确保转换为 RGB)
url = "./images/China_MotorBike_001239.jpg"
raw_image = Image.open(url).convert("RGB")

# 3. 构造消息
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": raw_image},
            {"type": "text", "text": "How many road defect in the image?"}
        ]
    },
]

# 4. 生成 Inputs
# 确保使用 apply_chat_template 转换
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[prompt],
    images=[raw_image],
    padding=True,
    return_tensors="pt",
).to(device)

# --- 防御性 ID 检查 ---
max_id = inputs["input_ids"].max().item()
vocab_limit = model.get_input_embeddings().weight.shape[0]
print(f"检查：最大 Token ID = {max_id}, 模型词表限制 = {vocab_limit}")

if max_id >= vocab_limit:
    raise ValueError(f"致命错误：Token ID {max_id} 越界！请检查是否缺少模型权重文件中的特殊标记定义。")

# 5. 生成输出
with torch.inference_mode():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False,
        use_cache=True
    )

# 6. 解码
# 仅截取生成的回复部分
input_len = inputs["input_ids"].shape[1]
generated_ids = outputs[0][input_len:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print("\n" + "="*30)
print("模型回答:", response)
print("="*30)