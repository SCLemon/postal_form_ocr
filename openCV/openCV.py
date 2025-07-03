import cv2
import numpy as np

def detect_and_crop(template_path, target_path, save_box_path=None):
    # 讀取範本圖與目標圖（灰階）
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(target_path)  # 彩色圖用來裁切與畫框

    # 影像對比增強（直方圖均衡化）
    template = cv2.equalizeHist(template)
    img = cv2.equalizeHist(img)

    # 不調整範本大小，保留原始尺寸

    # 建立 ORB 特徵點偵測器，nfeatures 調大
    orb = cv2.ORB_create(nfeatures=3000)

    # 偵測與計算特徵點與描述子
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    if des1 is None or des2 is None:
        print("❌ 特徵描述子計算失敗")
        return None, 0, 0

    # 建立暴力匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 取前 100 個好匹配點
    good_matches = matches[:100]

    if len(good_matches) >= 8:  # 放寬條件，至少 8 個匹配點
        # 取得匹配點位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 計算單應矩陣，RANSAC 閾值設為 5.0
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            print("❌ 單應矩陣計算失敗")
            return None, 0, 0

        # 範本圖的四個角點（以範本原始大小）
        h, w = template.shape
        template_corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # 映射到目標圖找出邊框角點
        dst_corners = cv2.perspectiveTransform(template_corners, M)

        # 畫出偵測到的邊框（可選擇存檔）
        img_box = img_color.copy()
        pts = np.int32(dst_corners)
        cv2.polylines(img_box, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        if save_box_path is not None:
            cv2.imwrite(save_box_path, img_box)

        # 建立透視變換矩陣（將目標圖中指定區域拉成矩形）
        dst_pts_rect = np.float32(dst_corners).reshape(-1, 2)
        src_pts_rect = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

        M_correct = cv2.getPerspectiveTransform(dst_pts_rect, src_pts_rect)

        # 執行裁切與拉直
        warped = cv2.warpPerspective(img_color, M_correct, (w, h))

        return warped, w, h

    else:
        print(f"❌ 匹配點太少 ({len(good_matches)})，無法建立單應矩陣")
        return None, 0, 0
