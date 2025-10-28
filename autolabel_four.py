import base64
import datetime
import glob
import json
import os

import cv2
from tqdm import tqdm

from ultralytics import YOLO

# =========================
# 설정
# =========================
model_path = "experiments/colony_2class_85_small/weights/best.pt"
img_dir = "C:/workspace/datasets/colony_177"
output_root = "C:/workspace/autolabel_output"

yolo_dir = os.path.join(output_root, "labels_yolo")
json_dir = os.path.join(output_root, "labels_json")
vis_dir = os.path.join(output_root, "visualize")
os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# 클래스 이름/색상
class_names = {0: "COLONY", 1: "USELESS"}
class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # COLONY=초록, USELESS=빨강

# Inference 하이퍼파라미터(촘촘한 소형 객체용)
infer_conf = 0.5
infer_iou = 0.5
infer_max_det = 2000  # 모델 stage에서의 1차 제한

# 병합(NMS) 하이퍼파라미터
merge_nms_iou = 0.5  # 4분할 결과 병합용 NMS IoU
merge_max_det = 5000  # 최종 결과 상한


# =========================
# 유틸: IoU, NMS (클래스별)
# =========================
def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(dets, iou_thresh=0.5, max_keep=5000):
    """
    dets: list of (cls, conf, x1,y1,x2,y2)
    클래스별로 나눠 NMS → keep 반환.
    """
    kept = []
    dets = sorted(dets, key=lambda x: x[1], reverse=True)  # conf desc
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[0], []).append(d)

    for cls_id, items in by_cls.items():
        items_sorted = items  # 이미 conf desc
        keep_cls = []
        suppressed = [False] * len(items_sorted)
        for i, di in enumerate(items_sorted):
            if suppressed[i]:
                continue
            keep_cls.append(di)
            if len(keep_cls) >= max_keep:
                break
            for j in range(i + 1, len(items_sorted)):
                if suppressed[j]:
                    continue
                dj = items_sorted[j]
                if iou_xyxy(di[2:], dj[2:]) >= iou_thresh:
                    suppressed[j] = True
        kept.extend(keep_cls)

    # 최종 상한
    kept = sorted(kept, key=lambda x: x[1], reverse=True)[:max_keep]
    return kept


# =========================
# 4분할 Inference
# =========================
def infer_quadrants(model, img_bgr):
    """
    이미지를 4분할하여 각 부분을 추론하고, 원본 좌표계로 변환된 det 리스트를 반환.

    return: list of (cls, conf, x1,y1,x2,y2).
    """
    H, W = img_bgr.shape[:2]
    w_mid = W // 2
    h_mid = H // 2

    # (x1,y1,x2,y2) in original image coords
    quads = [
        (0, 0, w_mid, h_mid),  # TL
        (w_mid, 0, W, h_mid),  # TR
        (0, h_mid, w_mid, H),  # BL
        (w_mid, h_mid, W, H),  # BR
    ]

    merged = []
    for qx1, qy1, qx2, qy2 in quads:
        crop = img_bgr[qy1:qy2, qx1:qx2]
        # YOLO는 np.ndarray(BGR) 바로 받아도 됨
        res = model(crop, conf=infer_conf, iou=infer_iou, max_det=infer_max_det, verbose=False)[0]

        # crop 좌표 → 원본 좌표로 보정
        for b in res.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0])  # crop 좌표
            # offset 추가
            X1 = x1 + qx1
            Y1 = y1 + qy1
            X2 = x2 + qx1
            Y2 = y2 + qy1

            # 경계 클램프
            X1 = max(0.0, min(float(W), X1))
            Y1 = max(0.0, min(float(H), Y1))
            X2 = max(0.0, min(float(W), X2))
            Y2 = max(0.0, min(float(H), Y2))

            if X2 > X1 and Y2 > Y1:
                merged.append((cls, conf, X1, Y1, X2, Y2))

    return merged


# =========================
# 메인
# =========================
model = YOLO(model_path)
img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

for img_path in tqdm(img_list, desc="Auto Labeling (4-split)"):
    if img_path != "C:/workspace/datasets/colony_177\\20251004_2025-10-02_Images_YPD.png":
        continue
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read: {img_path}")
        continue
    H, W = img.shape[:2]

    # 4분할 추론 → 병합 NMS
    dets_4 = infer_quadrants(model, img)
    dets_final = nms_per_class(dets_4, iou_thresh=merge_nms_iou, max_keep=merge_max_det)

    # YOLO txt / Dreamer JSON / 시각화 준비
    yolo_lines = []
    shapes_json = []

    for cls, conf, x1, y1, x2, y2 in dets_final:
        # ① YOLO txt (상대좌표)
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # ② Dreamer JSON (절대좌표)
        shapes_json.append(
            {
                "label": class_names.get(cls, "UNKNOWN"),
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

        # ③ 시각화
        color = class_colors.get(cls, (255, 255, 255))
        label_text = f"{class_names.get(cls, 'UNKNOWN')} {conf:.4f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img, label_text, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
        )

    # 저장: YOLO txt
    with open(os.path.join(yolo_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    # 저장: Dreamer JSON (imageData 포함)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(img_path, "rb") as f:
        image_data_b64 = base64.b64encode(f.read()).decode("utf-8")

    json_data = {
        "version": "1.4.1",
        "user": "Offline User",
        "time": now,
        "label_dict": {"COLONY": 0, "USELESS": 1},
        "flags": {},
        "shapes": shapes_json,
        # 필요 시 프로젝트 상대 경로로 바꾸세요.
        "imagePath": f"DETECTION\\ATI\\colony_4\\{os.path.basename(img_path)}",
        "imageData": image_data_b64,
        "multichannel": {},
        "imageHeight": H,
        "imageWidth": W,
    }
    with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # 저장: 시각화 PNG
    cv2.imwrite(os.path.join(vis_dir, f"{name}_pred.png"), img)

print("\n✅ 4분할 자동 라벨링 완료!")
print(f"YOLO txt   : {yolo_dir}")
print(f"JSON       : {json_dir}")
print(f"Visualize  : {vis_dir}")
