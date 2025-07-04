from PIL import Image
import cv2
import os
from openCV.openCV import detect_and_crop, get_raw_stroke_mask_only
from ocr.ocr import paddleOCR
from llm.ai import run
import json

# 1. 自動抓劃撥單，裁下整張區塊
cropped_img, w, h = detect_and_crop('./openCV/open_template.png', './images/t.jpg', './crops/crops.png')
if cropped_img is None:
    print("❌ 偵測失敗，無法裁切")
    exit()

image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

# 3. 定義小區塊座標
w_ratio = w/210
h_ratio = h/110

regions = {
    "account": (w_ratio * 5.8, h_ratio * 5, w_ratio * 65, h_ratio * 20),
    "amount": (w_ratio * 75, h_ratio * 10, w_ratio * 142.5, h_ratio * 20),
    "Beneficiary's Account": (w_ratio * 77, h_ratio * 20, w_ratio * 142.5, h_ratio * 30),
    "Applicant":(w_ratio * 65, h_ratio * 35, w_ratio * 108, h_ratio * 45),
    "Address":(w_ratio * 65, h_ratio * 53, w_ratio * 108, h_ratio * 72),
    "Tel":(w_ratio * 65, h_ratio * 72, w_ratio * 108, h_ratio * 80),
    "Remark":(w_ratio * 0, h_ratio* 25 ,w_ratio * 54, h_ratio * 110)
}


results = {}

temp_dir = "./cropped_regions"  # 自訂資料夾路徑
os.makedirs(temp_dir, exist_ok=True)  # 若沒資料夾就建立
filtered_dir = "./filtered"
os.makedirs(filtered_dir, exist_ok=True)

for label, box in regions.items():
    region_img = image.crop(box)

    # 放大解析度兩倍
    w, h = region_img.size
    region_img_2x = region_img.resize((w * 2, h * 2), Image.LANCZOS)

    temp_path = os.path.join(temp_dir, f"{label}.png")
    region_img_2x.save(temp_path)

    # 去除虛線
    if label == 'account' or label == 'amount':
        get_raw_stroke_mask_only(temp_path, temp_path)

        result_text = run(temp_path)
        results[label] = result_text

    else:
        results[label] = paddleOCR(temp_path)



cleaned_results = {k: v.replace(" ", "").replace("\n", "") for k, v in results.items()}
json_output = json.dumps(cleaned_results, ensure_ascii=False, indent=2)
print(json_output)