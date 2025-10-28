import glob
import os

import cv2
from tqdm import tqdm

from ultralytics import YOLO

# =============================
# Stage 설정 (사용자 지정)
# =============================
STAGE_CONFIGS = [
    {"rows": 1, "cols": 1, "conf": 0.5, "iou": 0.5, "max_det": 3000},  # Full (멀티스케일에서 자동 skip)
    {"rows": 2, "cols": 2, "conf": 0.5, "iou": 0.4, "max_det": 3000},  # 2x2
    {"rows": 4, "cols": 4, "conf": 0.5, "iou": 0.3, "max_det": 3000},  # 4x4,
]

MERGE_NMS_IOU = 0.1
MERGE_MAX_KEEP = 5000

# Isolated 판정 파라미터 (요청값 유지)
ISOLATED_IOU_THR = 0.2  # 겹침 판단용 IoU(작게)
ISOLATED_IOA_THR = 0.2  # 겹침 판단용 IoA(작은 박스 기준 비율)

# 클래스/색
CLASS_NAMES = {0: "COLONY", 1: "USELESS"}
CLASS_COLONY_ID = 0
COLOR_COLONY = (0, 255, 0)  # green
COLOR_USELESS = (0, 0, 255)  # red
COLOR_ISOLATED = (255, 0, 0)  # blue-ish (요청 텍스트는 green이었지만 시각 구분 위해 파랑)


# =============================
# 유틸: IoU/IoA 계산
# =============================
def _intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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

    x1_new = max(0.0, cx - side / 2.0)
    y1_new = max(0.0, cy - side / 2.0)
    x2_new = min(float(img_w), cx + side / 2.0)
    y2_new = min(float(img_h), cy + side / 2.0)
    return x1_new, y1_new, x2_new, y2_new


# =============================
# Global NMS (COLONY 우선 → conf 내림차순)
# dets: [cls, x1, y1, x2, y2, conf]
# =============================
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


# =============================
# Isolated 필터 (IoU + IoA 동시 사용)
#  - 겹친다: (IoU >= thr_iou) or (IoA >= thr_ioa)
#  - isolated: 모든 j에 대해 IoU<thr_iou AND IoA<thr_ioa
# =============================
def filter_isolated_boxes(preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR):
    isolated = []
    for i, bi in enumerate(preds):
        box_i = bi[1:5]
        independent = True
        for j, bj in enumerate(preds):
            if i == j:
                continue
            box_j = bj[1:5]
            if (iou_xyxy(box_i, box_j) >= thr_iou) or (ioa_smaller(box_i, box_j) >= thr_ioa):
                independent = False
                break
        if independent:
            isolated.append(bi)
    return isolated


# =============================
# YOLO: Full Image Inference
# =============================
def infer_full_image(model, img, conf=0.5, iou=0.5, max_det=3000):
    preds = []
    res = model.predict(img, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    for b in res.boxes:
        cls = int(b.cls[0])
        confv = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        preds.append([cls, x1, y1, x2, y2, confv])
    return preds


# =============================
# Grid-based Inference with Overlap
# =============================
def infer_grid_with_overlap(model, img, rows, cols, conf, iou, max_det=3000, overlap_ratio=0.2):
    """overlap_ratio: 각 grid가 겹쳐지는 비율 (0.2면 20% 겹침)."""
    H, W = img.shape[:2]
    preds = []

    cell_w = W / cols
    cell_h = H / rows

    for r in range(rows):
        for c in range(cols):
            # 기본 cell 경계
            x1 = int(cell_w * c)
            y1 = int(cell_h * r)
            x2 = int(cell_w * (c + 1))
            y2 = int(cell_h * (r + 1))

            # overlap 확장
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

                # crop 좌표 → 원본 좌표
                X1 = bx1 + x1_ov
                Y1 = by1 + y1_ov
                X2 = bx2 + x1_ov
                Y2 = by2 + y1_ov

                preds.append([cls, X1, Y1, X2, Y2, confv])
    return preds


# =============================
# Multi-Scale Inference (정사각형 보정 + Global NMS)
# =============================
def infer_multiscale(model, img, overlap_ratio=0.2):
    H, W = img.shape[:2]
    merged = []

    for cfg in STAGE_CONFIGS:
        r, c = cfg["rows"], cfg["cols"]

        merged += infer_grid_with_overlap(
            model,
            img,
            rows=r,
            cols=c,
            conf=cfg["conf"],
            iou=cfg["iou"],
            max_det=cfg["max_det"],
            overlap_ratio=overlap_ratio,
        )

    # COLONY만 정사각형 보정
    sq = []
    for cls, x1, y1, x2, y2, conf in merged:
        if cls == CLASS_COLONY_ID:
            x1s, y1s, x2s, y2s = make_square_box(x1, y1, x2, y2, W, H)
            sq.append([cls, x1s, y1s, x2s, y2s, conf])
        else:
            sq.append([cls, x1, y1, x2, y2, conf])

    # Global NMS
    return global_nms(sq, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)


# =============================
# Multi-Scale → Isolated(COLONY)
# =============================
def infer_multiscale_isolated_colony(model, img, overlap_ratio=0.2):
    ms_preds = infer_multiscale(model, img, overlap_ratio=overlap_ratio)
    iso = filter_isolated_boxes(ms_preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR)
    iso_colony = [p for p in iso if p[0] == CLASS_COLONY_ID]
    return iso_colony


# =============================
# 시각화 & 저장
# =============================
def draw_and_save(img_path, preds, mode, out_dir="total_results", cut_ratio=False):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(img_path)
    if img is None:
        return

    vis = img.copy()
    H, W = vis.shape[:2]

    if cut_ratio:
        x_min_cut = W * cut_ratio
        x_max_cut = W * (1 - cut_ratio)
        y_min_cut = H * cut_ratio
        y_max_cut = H * (1 - cut_ratio)

    for cls, x1, y1, x2, y2, conf in preds:
        if cut_ratio:
            if (x1 < x_min_cut) or (y1 < y_min_cut) or (x2 > x_max_cut) or (y2 > y_max_cut):
                continue  # 배양기 끝부분 오탐은 무시

        if mode == "isolated":
            color = COLOR_ISOLATED
            label = None  # isolated 표시는 박스만
        else:
            color = COLOR_COLONY if cls == CLASS_COLONY_ID else COLOR_USELESS
            label = CLASS_NAMES.get(cls, "UNK")

        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if label is not None:  # 안전하게 라벨 있을 때만 텍스트 출력
            cv2.putText(vis, label, (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    name = os.path.splitext(os.path.basename(img_path))[0]
    suffix = {"full": "_a_full.png", "ms": "_b_ms.png", "isolated": "_isolated.png"}[mode]
    cv2.imwrite(os.path.join(out_dir, name + suffix), vis)


# =============================
# 전체 이미지 처리 (train/val/test 모두)
# =============================
def process_all_images(model_path, dataset_root, overlap_ratio=0.2, cut_ratio=False):
    model = YOLO(model_path)
    img_dirs = [
        # os.path.join(dataset_root, "train"),
        # os.path.join(dataset_root, "val"),
        os.path.join(dataset_root, "test"),
    ]
    images = []
    for d in img_dirs:
        images += glob.glob(os.path.join(d, "*.png"))
        images += glob.glob(os.path.join(d, "*.jpg"))
    images = sorted(images)
    images = ["yeast_0930_1002_2025-10-01_Images_A3_100ul_48h.png"]

    print(f"🔍 Total Images Found: {len(images)}")
    for img_path in tqdm(images, desc="Saving Full/MS/Isolated"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 1) Full Prediction (원본 전체)
        full_preds = infer_full_image(model, img, conf=0.5, iou=0.5, max_det=3000)
        draw_and_save(img_path, full_preds, mode="full", out_dir="total_results", cut_ratio=cut_ratio)

        # 2) Multi-Scale Prediction (Overlap 적용, 1x1은 내부에서 skip)
        ms_preds = infer_multiscale(model, img, overlap_ratio=overlap_ratio)
        draw_and_save(img_path, ms_preds, mode="ms", out_dir="total_results", cut_ratio=cut_ratio)

        # 3) Isolated (COLONY only)
        iso_preds = infer_multiscale_isolated_colony(model, img, overlap_ratio=overlap_ratio)
        draw_and_save(img_path, iso_preds, mode="isolated", out_dir="total_results", cut_ratio=cut_ratio)

    print("✅ 완료! 결과는 total_results 폴더에 저장되었습니다.")


# =============================
# 실행부
# =============================
if __name__ == "__main__":
    MODEL_PATH = "experiments/colony_2class_small_noval/weights/best.pt"
    DATASET_ROOT = "C:/workspace/datasets/colony_2class_noval/images"
    cut_ratio = 0.05
    process_all_images(MODEL_PATH, DATASET_ROOT, overlap_ratio=0.2, cut_ratio=cut_ratio)
