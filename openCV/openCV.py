import cv2
import numpy as np
import os

def detect_and_crop(template_path, target_path, save_box_path=None):
    # 讀圖（彩色＋灰階）
    img_color = cv2.imread(target_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # === 預處理：提升對比、降雜訊 ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    template = clahe.apply(template)
    img_gray = clahe.apply(img_gray)

    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # === 建立 ORB 特徵偵測器 ===
    orb = cv2.ORB_create(nfeatures=8000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        print("❌ 特徵描述子計算失敗")
        return None, 0, 0

    # === 比對特徵點 ===
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:100]

    if len(good_matches) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # === 建立單應矩陣 ===
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)  # RANSAC 閾值放寬

        if M is None:
            print("❌ 單應矩陣建立失敗")
            return None, 0, 0

        h, w = template.shape
        template_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst_corners = cv2.perspectiveTransform(template_corners, M)

        # === 繪製邊框 ===
        img_box = img_color.copy()
        cv2.polylines(img_box, [np.int32(dst_corners)], True, (0, 255, 0), 3)

        if save_box_path is not None:
            os.makedirs(os.path.dirname(save_box_path), exist_ok=True)
            success = cv2.imwrite(save_box_path, img_box)
            print(f"📦 偵測框已儲存：{save_box_path}，成功？{success}")

        # === 透視變換與裁切 ===
        dst_pts_rect = dst_corners.reshape(-1, 2)
        src_pts_rect = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
        M_correct = cv2.getPerspectiveTransform(dst_pts_rect, src_pts_rect)
        warped = cv2.warpPerspective(img_color, M_correct, (w, h))

        return warped, w, h

    else:
        print(f"❌ 匹配點不足（目前 {len(good_matches)}）")
        return None, 0, 0


def get_raw_stroke_mask_only(img_path, save_path=None, scale_factor=2):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"❌ 圖片無法讀取：{img_path}")

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 黑筆範圍
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 90, 90])

    # 藍筆範圍
    lower_blue = np.array([95, 120, 50])
    upper_blue = np.array([130, 255, 255])

    # 產生遮罩
    mask_black = cv2.inRange(img_hsv, lower_black, upper_black)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(mask_black, mask_blue)

    # 放大遮罩解析度
    if scale_factor != 1:
        h, w = mask.shape
        mask = cv2.resize(mask, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)

    if save_path:
        cv2.imwrite(save_path, mask)
        print(f"✅ 遮罩已儲存：{save_path}")

    return mask
