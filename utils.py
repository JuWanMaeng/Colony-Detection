import os

import cv2


# ==============================================
# 🚩 1. IoU & 평가 함수
# ==============================================
def iou(box1, box2):
    """box: [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def load_gt_boxes(img_path, labels_dir):
    """
    GT YOLO txt 파일을 읽어서 GT bounding box 리스트로 반환
    return: [[cls, x1, y1, x2, y2], ...]  (절대좌표 기준).
    """
    img = cv2.imread(img_path)
    if img is None:
        return []

    H, W = img.shape[:2]
    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, f"{name}.txt")

    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls, xc, yc, bw, bh = map(float, line.strip().split())
                x1 = (xc - bw / 2) * W
                y1 = (yc - bh / 2) * H
                x2 = (xc + bw / 2) * W
                y2 = (yc + bh / 2) * H
                gt_boxes.append([int(cls), x1, y1, x2, y2])

    return gt_boxes


def evaluate_f1(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    입력:
        gt_boxes:     [[cls_id, x1, y1, x2, y2], ...]
        pred_boxes:   [[cls_id, x1, y1, x2, y2, conf], ...].

    출력:
        f1, precision, recall, TP, FP, FN
    """
    matched_gt = [False] * len(gt_boxes)
    pred_flags = [False] * len(pred_boxes)

    for i, p in enumerate(pred_boxes):
        p_cls, px1, py1, px2, py2, _ = p
        best_iou, best_j = 0, -1
        for j, g in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            g_cls, gx1, gy1, gx2, gy2 = g
            if p_cls != g_cls:
                continue
            iou_val = iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2])
            if iou_val > best_iou:
                best_iou = iou_val
                best_j = j
        if best_iou >= iou_threshold:
            matched_gt[best_j] = True
            pred_flags[i] = True

    TP = sum(pred_flags)
    FP = len(pred_flags) - TP
    FN = matched_gt.count(False)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return f1, precision, recall, TP, FP, FN


# =========================
# 유틸: IoU, NMS(클래스별)
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
    dets: list[(cls, conf, x1, y1, x2, y2)]
    클래스별 NMS → keep 반환.
    """
    kept = []
    dets = sorted(dets, key=lambda x: x[1], reverse=True)  # conf desc
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[0], []).append(d)

    for cls_id, items in by_cls.items():
        items_sorted = items
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

    kept = sorted(kept, key=lambda x: x[1], reverse=True)[:max_keep]
    return kept


# ==============================
# 시각화 함수 추가
# ==============================
def draw_boxes_on_image(img, boxes, class_names, save_path, title=None):
    """
    boxes: [[cls, x1, y1, x2, y2, conf(optional)], ...]
    save_path: 저장할 경로
    title: optional, 이미지 상단에 텍스트 표시.
    """
    vis_img = img.copy()

    for box in boxes:
        if len(box) == 6:
            cls, x1, y1, x2, y2, conf = box
            label_text = f"{class_names.get(cls, 'UNK')} {conf:.2f}"
        else:
            cls, x1, y1, x2, y2 = box
            label_text = f"{class_names.get(cls, 'GT')}"
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # COLONY: green, USELESS: red
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis_img, label_text, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if title:
        cv2.putText(vis_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imwrite(save_path, vis_img)


# ==============================
# 결과 저장 함수
# ==============================
def save_visualizations(img_path, img, gt_boxes, full_preds, stage3_preds, class_names):
    name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join("results", name)
    os.makedirs(save_dir, exist_ok=True)
    if full_preds is None:
        full_preds = []
    if stage3_preds is None:
        stage3_preds = []

    # GT 시각화
    draw_boxes_on_image(img, gt_boxes, class_names, save_path=os.path.join(save_dir, "gt.png"), title="Ground Truth")

    # Full Inference 시각화
    draw_boxes_on_image(
        img, full_preds, class_names, save_path=os.path.join(save_dir, "full_pred.png"), title="Full Image Prediction"
    )

    # Stage3 시각화
    draw_boxes_on_image(
        img, stage3_preds, class_names, save_path=os.path.join(save_dir, "stage3_pred.png"), title="3-Stage Prediction"
    )
