import os
import torch
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize
import json
# 1. 處理模型路徑 (針對 Windows 的絕對路徑優化)
# 確保路徑格式正確，避免 Transformers 誤認為是 Repo ID
model_dir = "../models/Rex_omni"
abs_model_path = os.path.abspath(model_dir).replace("\\", "/")

print(f"正在初始化模型自: {abs_model_path}")

# 2. 初始化模型
# 加入 attn_implementation="sdpa" 避免 FlashAttention2 報錯
# 加入 torch_dtype 節省顯存
model = RexOmniWrapper(
    model_path=abs_model_path,
    backend="transformers",
    attn_implementation="sdpa",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 3. 讀取圖片 (建議增加 .convert("RGB") 確保格式正確)
image_path = "../images/China_MotorBike_001461.jpg" # 請確認路徑正確
if not os.path.exists(image_path):
    print(f"錯誤：找不到圖片 {image_path}")
    
    # 這裡可以用你之前的 person_dog.jpg 測試
else:
    image = Image.open(image_path).convert("RGB")

# 4. 瑕疵檢測推理
# 類別使用具體的描述詞，有助於 VLM 識別
defect_categories = ["scratch"]

results = model.inference(
    images=image,
    task="detection",
    categories=defect_categories,
    # 額外參數：確保輸出完整且不亂跳
    max_new_tokens=1024,
    do_sample=False 
)

result = results[0]

# 5. 列印檢測結果 (方便偵錯)
# print(results)
output_data = results[0]
with open("scratch_detection_results.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)
# 6. 可視化並儲存
vis = RexOmniVisualize(
    image=image,
    predictions=result["extracted_predictions"],
    font_size=20,
    draw_width=5,
    show_labels=True,
)
output_name = "./scratch_visualize.jpg"
vis.save(output_name)
print(f"\n視覺化結果已儲存至: {output_name}")