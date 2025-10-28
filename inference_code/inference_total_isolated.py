import glob
import os

import cv2
from tqdm import tqdm

from ultralytics import YOLO

# =============================
# Stage 설정 (사용자 지정)
# =============================
STAGE_CONFIGS = [
    {"rows": 1, "cols": 1, "conf": 0.5, "iou": 0.5, "max_det": 3000},
    {"rows": 2, "cols": 2, "conf": 0.5, "iou": 0.4, "max_det": 3000},
    {"rows": 4, "cols": 4, "conf": 0.5, "iou": 0.3, "max_det": 3000},
]

MERGE_NMS_IOU = 0.5
MERGE_MAX_KEEP = 5000

ISOLATED_IOU_THR = 0.05  # 겹침 판단용 IoU (작게)
ISOLATED_IOA_THR = 0.05  # 겹침 판단용 IoA (작은 박스 기준 비율)

# 클래스/색
CLASS_COLONY_ID = 0
BOX_COLOR = (255, 0, 0)  # BGR: green


# =============================
# 유틸: IoU/IoA 계산
# =============================
def _intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter


def iou_xyxy(a, b):
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ioa_smaller(a, b):
    """작은 박스 기준 IoA = inter / min(area_a, area_b)."""
    inter = _intersection(a, b)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0


# =============================
# 정사각형 변환 (긴 변 기준, expand) - COLONY만
# =============================
def make_square_box(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    x1_new = cx - side / 2.0
    y1_new = cy - side / 2.0
    x2_new = cx + side / 2.0
    y2_new = cy + side / 2.0

    # clamp
    x1_new = max(0.0, x1_new)
    y1_new = max(0.0, y1_new)
    x2_new = min(float(img_w), x2_new)
    y2_new = min(float(img_h), y2_new)

    return x1_new, y1_new, x2_new, y2_new


# =============================
# Global NMS (COLONY 우선 → conf 내림차순)
# dets: [cls, x1, y1, x2, y2, conf]
# =============================
def global_nms(dets, iou_thresh=0.5, max_keep=5000):
    dets = sorted(dets, key=lambda x: (x[0] != CLASS_COLONY_ID, -x[5]))
    kept = []
    suppressed = [False] * len(dets)

    for i, di in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(di)
        if len(kept) >= max_keep:
            break
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            dj = dets[j]
            if iou_xyxy(di[1:5], dj[1:5]) >= iou_thresh:
                suppressed[j] = True
    return kept


# =============================
# Isolated 필터 (IoU + IoA 동시 사용)
#  - 두 박스가 "겹친다"로 간주: (IoU >= thr_iou) or (IoA >= thr_ioa)
#  - isolated로 남으려면: 모든 j에 대해 IoU<thr_iou AND IoA<thr_ioa
# =============================
def filter_isolated_boxes(preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR):
    isolated = []
    for i, bi in enumerate(preds):
        box_i = bi[1:5]
        iso = True
        for j, bj in enumerate(preds):
            if i == j:
                continue
            box_j = bj[1:5]
            iouv = iou_xyxy(box_i, box_j)
            ioav = ioa_smaller(box_i, box_j)
            if (iouv >= thr_iou) or (ioav >= thr_ioa):
                iso = False
                break
        if iso:
            isolated.append(bi)
    return isolated


# =============================
# Grid-based Inference
# =============================
def infer_grid(model, img, rows, cols, conf, iou, max_det=3000):
    H, W = img.shape[:2]
    preds = []
    xs = [int(W * i / cols) for i in range(cols + 1)]
    ys = [int(H * j / rows) for j in range(rows + 1)]

    for r in range(rows):
        for c in range(cols):
            x1g, x2g = xs[c], xs[c + 1]
            y1g, y2g = ys[r], ys[r + 1]
            crop = img[y1g:y2g, x1g:x2g]
            if crop.size == 0:
                continue

            result = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
            for box in result.boxes:
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                # crop → global
                X1, Y1 = x1 + x1g, y1 + y1g
                X2, Y2 = x2 + x1g, y2 + y1g
                preds.append([cls, X1, Y1, X2, Y2, conf_score])
    return preds


# =============================
# Multi-Stage Inference
# 1) 수집 → 2) COLONY만 정사각형 보정 → 3) Global NMS
# 4) Isolated 필터(IoU+IoA) → 5) COLONY만 유지
# =============================
def infer_multiscale_colony_isolated(model, img):
    H, W = img.shape[:2]
    merged = []
    for cfg in STAGE_CONFIGS:
        merged += infer_grid(model, img, cfg["rows"], cfg["cols"], cfg["conf"], cfg["iou"], cfg["max_det"])

    # COLONY만 정사각형 보정
    squared = []
    for cls, x1, y1, x2, y2, conf in merged:
        if cls == CLASS_COLONY_ID:
            x1s, y1s, x2s, y2s = make_square_box(x1, y1, x2, y2, W, H)
            squared.append([cls, x1s, y1s, x2s, y2s, conf])
        else:
            squared.append([cls, x1, y1, x2, y2, conf])

    # Global NMS
    merged_nms = global_nms(squared, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)

    # Isolated 필터 (IoU + IoA)
    isolated = filter_isolated_boxes(merged_nms, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR)

    # COLONY만 유지
    isolated_colony = [p for p in isolated if p[0] == CLASS_COLONY_ID]
    return isolated_colony


# =============================
# 시각화 및 저장 (라벨: 'isolated' 고정)
# =============================
def save_isolated_result(img_path, preds, save_dir="results_multiscale_isolated"):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        return
    vis = img.copy()
    for cls, x1, y1, x2, y2, conf in preds:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
        cv2.putText(vis, "isolated", (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1)
    name = os.path.splitext(os.path.basename(img_path))[0]
    out = os.path.join(save_dir, f"{name}_isolated.png")
    cv2.imwrite(out, vis)


# =============================
# 전체 이미지 처리
# =============================
def process_all_images(model_path, dataset_root):
    model = YOLO(model_path)
    img_dirs = [
        os.path.join(dataset_root, "train"),
        os.path.join(dataset_root, "val"),
        os.path.join(dataset_root, "test"),
    ]
    images = []
    for d in img_dirs:
        images += glob.glob(os.path.join(d, "*.png"))
        images += glob.glob(os.path.join(d, "*.jpg"))
    images = sorted(images)

    print(f"🔍 Total Images Found: {len(images)}")
    for img_path in tqdm(images, desc="Infer & Save Isolated COLONY"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        preds_isolated = infer_multiscale_colony_isolated(model, img)
        save_isolated_result(img_path, preds_isolated, save_dir="results_multiscale_isolated")
    print("✅ 완료! 결과는 results_multiscale_isolated 폴더에 저장되었습니다.")


# =============================
# 실행부
# =============================
if __name__ == "__main__":
    MODEL_PATH = "experiments/colony_2class_85_small/weights/best.pt"
    DATASET_ROOT = "C:/workspace/datasets/colony_2class/images"
    process_all_images(MODEL_PATH, DATASET_ROOT)
