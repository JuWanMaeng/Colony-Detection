import glob
import os

import cv2
from tqdm import tqdm

from ultralytics import YOLO

# ================================
# 설정값
# ================================
MODEL_PATH = "experiments/colony_2class_small_noval/weights/best.pt"
IMG_DIR = "C:/workspace/datasets/colony_2class_noval/images/test"
LABEL_DIR = "C:/workspace/datasets/colony_2class_noval/labels/test"
OUTPUT_DIR = "visual_results"

IOU_THR_MATCH = 0.5  # TP 매칭 기준 IoU
MERGE_NMS_IOU = 0.1  # 멀티스케일 병합용 NMS IoU
MERGE_MAX_KEEP = 5000
OVERLAP_RATIO = 0.2  # 그리드 크롭 overlap

# 멀티스케일 단계
STAGE_CONFIGS = [
    {"rows": 1, "cols": 1, "conf": 0.5, "iou": 0.5, "max_det": 3000},
    {"rows": 2, "cols": 2, "conf": 0.5, "iou": 0.4, "max_det": 3000},
    {"rows": 4, "cols": 4, "conf": 0.5, "iou": 0.3, "max_det": 3000},
]

# 클래스/표기
CLASS_NAMES = {0: "COLONY", 1: "USELESS"}

# 상태별 색상
COLOR_TP = (0, 255, 0)  # Green
COLOR_FP = (0, 0, 255)  # Red
COLOR_FN = (0, 255, 255)  # Yellow

# 클래스별 선 두께(간이 점선 효과 대용)
LINE_SOLID = 2  # COLONY (class 0)
LINE_DASH = 1  # USELESS (class 1)


# ================================
# 기하 유틸
# ================================
def _intersection_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a, b):
    inter = _intersection_xyxy(a, b)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def make_square_box(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    x1n = max(0.0, cx - side * 0.5)
    y1n = max(0.0, cy - side * 0.5)
    x2n = min(float(img_w), cx + side * 0.5)
    y2n = min(float(img_h), cy + side * 0.5)
    return x1n, y1n, x2n, y2n


# ================================
# 병합 NMS (conf 내림차순)
# dets: [cls, x1, y1, x2, y2, conf]
# ================================
def global_nms(dets, iou_thresh=0.5, max_keep=5000):
    dets = sorted(dets, key=lambda x: -x[5])
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


# ================================
# 겹침 그리드 추론
# ================================
def infer_grid_with_overlap(model, img, rows, cols, conf, iou, max_det=3000, overlap_ratio=0.2):
    H, W = img.shape[:2]
    preds = []
    cell_w = W / cols
    cell_h = H / rows

    for r in range(rows):
        for c in range(cols):
            x1 = int(cell_w * c)
            y1 = int(cell_h * r)
            x2 = int(cell_w * (c + 1))
            y2 = int(cell_h * (r + 1))

            x1_ov = int(max(0, x1 - cell_w * overlap_ratio))
            y1_ov = int(max(0, y1 - cell_h * overlap_ratio))
            x2_ov = int(min(W, x2 + cell_w * overlap_ratio))
            y2_ov = int(min(H, y2 + cell_h * overlap_ratio))

            crop = img[y1_ov:y2_ov, x1_ov:x2_ov]
            if crop.size == 0:
                continue

            res = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
            for b in res.boxes:
                cls = int(b.cls[0])
                confv = float(b.conf[0])
                bx1, by1, bx2, by2 = map(float, b.xyxy[0])
                # crop -> 원본 좌표
                X1 = bx1 + x1_ov
                Y1 = by1 + y1_ov
                X2 = bx2 + x1_ov
                Y2 = by2 + y1_ov
                preds.append([cls, X1, Y1, X2, Y2, confv])
    return preds


# ================================
# 멀티스케일 추론 (정사각형 보정 + 병합 NMS)
# return: [cls, x1, y1, x2, y2, conf]
# ================================
def infer_multiscale(model, img, overlap_ratio=0.2):
    H, W = img.shape[:2]
    merged = []
    for cfg in STAGE_CONFIGS:
        merged += infer_grid_with_overlap(
            model,
            img,
            rows=cfg["rows"],
            cols=cfg["cols"],
            conf=cfg["conf"],
            iou=cfg["iou"],
            max_det=cfg["max_det"],
            overlap_ratio=overlap_ratio,
        )

    # COLONY만 정사각형 보정
    sq = []
    for cls, x1, y1, x2, y2, conf in merged:
        if cls == 0:  # CLASS_COLONY_ID
            x1s, y1s, x2s, y2s = make_square_box(x1, y1, x2, y2, W, H)
            sq.append([cls, x1s, y1s, x2s, y2s, conf])
        else:
            sq.append([cls, x1, y1, x2, y2, conf])

    return global_nms(sq, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)


# ================================
# 라벨 로드 (YOLO txt -> xyxy)
# ================================
def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([int(cls), x1, y1, x2, y2])
    return boxes


# ================================
# 박스 그리기
# ================================
def draw_box(vis, box, color, cls, text):
    x1, y1, x2, y2 = map(int, box)
    thickness = LINE_SOLID if cls == 0 else LINE_DASH
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(vis, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ================================
# TP / FP / FN 시각화
# ================================
def visualize_tp_fp_fn(img_path, preds, gt_boxes):
    img = cv2.imread(img_path)
    if img is None:
        return
    vis = img.copy()

    matched_pred = set()
    matched_gt = set()

    # 1) TP: 같은 클래스에서 IoU 최대 매칭, 임계치 이상만
    for i, pred in enumerate(preds):
        cls_p, px1, py1, px2, py2, conf = pred
        best_iou, best_gt = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            cls_g, gx1, gy1, gx2, gy2 = gt
            if cls_p != cls_g:
                continue
            iou = iou_xyxy([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
            if iou > best_iou:
                best_iou, best_gt = iou, j

        if best_iou >= IOU_THR_MATCH and best_gt not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(best_gt)
            draw_box(vis, [px1, py1, px2, py2], COLOR_TP, cls_p, f"{CLASS_NAMES.get(cls_p, '?')} TP")

    # 2) FP: 매칭 실패한 예측
    for i, pred in enumerate(preds):
        if i in matched_pred:
            continue
        cls_p, px1, py1, px2, py2, _conf = pred
        draw_box(vis, [px1, py1, px2, py2], COLOR_FP, cls_p, f"{CLASS_NAMES.get(cls_p, '?')} FP")

    # 3) FN: 매칭 실패한 GT
    for j, gt in enumerate(gt_boxes):
        if j in matched_gt:
            continue
        cls_g, gx1, gy1, gx2, gy2 = gt
        draw_box(vis, [gx1, gy1, gx2, gy2], COLOR_FN, cls_g, f"{CLASS_NAMES.get(cls_g, '?')} FN")

    # 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_perf.png"), vis)


# ================================
# 메인 루프
# ================================
def main():
    model = YOLO(MODEL_PATH)
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")) + glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    print(f"🔍 Total Images Found: {len(images)}")

    for img_path in tqdm(images, desc="Processing"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        # 멀티스케일 추론
        preds = infer_multiscale(model, img, overlap_ratio=OVERLAP_RATIO)  # [cls,x1,y1,x2,y2,conf]

        # GT 로드
        label_path = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        gt_boxes = load_yolo_labels(label_path, W, H)  # [cls,x1,y1,x2,y2]

        # TP/FP/FN 시각화 저장
        visualize_tp_fp_fn(img_path, preds, gt_boxes)

    print(f"✅ 완료! 결과는 '{OUTPUT_DIR}' 폴더에 저장되었습니다.")


# ================================
# 실행부
# ================================
if __name__ == "__main__":
    main()
