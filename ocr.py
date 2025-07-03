from PIL import Image
import cv2
import os
import tempfile
from openCV.openCV import detect_and_crop
from ai import run

# 1. 自動抓劃撥單，裁下整張區塊
cropped_img, w, h = detect_and_crop('./openCV/open_template.png', './images/test.png')
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

with tempfile.TemporaryDirectory(prefix="cropped_regions_") as temp_dir:
    for label, box in regions.items():
        region_img = image.crop(box)

        temp_path = os.path.join(temp_dir, f"{label}.png")
        region_img.save(temp_path)

        result_text = run(temp_path)
        results[label] = result_text

    for k, v in results.items():
        v_no_space = v.replace(" ", "")
        print(f"{k}: {v_no_space}")