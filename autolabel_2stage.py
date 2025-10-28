import base64
import datetime
import glob
import json
import os

import cv2
from tqdm import tqdm

from ultralytics import YOLO

# =========================
# 설정 (요청 그대로)
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

# Inference 하이퍼파라미터
CONF = 0.5
IOU = 0.5
MAXDET_CROP = 2000

# 병합(NMS) 하이퍼파라미터 (최종)
MERGE_NMS_IOU = 0.5
MERGE_MAX_KEEP = 5000

# 16분할(4x4, no overlap)
GRID_ROWS = 4
GRID_COLS = 4


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
    dets: list of (cls, conf, x1,y1,x2,y2)
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


# =========================
# Inference helpers
# =========================
def infer_full(model, img_bgr):
    """원본 이미지 전체 inference → dets(list of (cls, conf, x1,y1,x2,y2))."""
    H, W = img_bgr.shape[:2]
    res = model(img_bgr, conf=CONF, iou=IOU, max_det=MAXDET_CROP, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        # clamp
        x1 = max(0.0, min(float(W), x1))
        y1 = max(0.0, min(float(H), y1))
        x2 = max(0.0, min(float(W), x2))
        y2 = max(0.0, min(float(H), y2))
        if x2 > x1 and y2 > y1:
            dets.append((cls, conf, x1, y1, x2, y2))
    return dets


def infer_grid_4x4(model, img_bgr):
    """
    4x4 그리드(16분할) inference → 원본 좌표로 복원된 det 리스트 반환
    return: list of (cls, conf, x1,y1,x2,y2).
    """
    H, W = img_bgr.shape[:2]
    dets_all = []
    xs = [round(W * i / GRID_COLS) for i in range(GRID_COLS + 1)]
    ys = [round(H * j / GRID_ROWS) for j in range(GRID_ROWS + 1)]

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1g, x2g = xs[c], xs[c + 1]
            y1g, y2g = ys[r], ys[r + 1]
            if x2g <= x1g or y2g <= y1g:
                continue

            crop = img_bgr[y1g:y2g, x1g:x2g]
            res = model(crop, conf=CONF, iou=IOU, max_det=MAXDET_CROP, verbose=False)[0]

            for b in res.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(float, b.xyxy[0])  # crop 좌표
                X1 = x1 + x1g
                Y1 = y1 + y1g
                X2 = x2 + x1g
                Y2 = y2 + y1g
                # clamp
                X1 = max(0.0, min(float(W), X1))
                Y1 = max(0.0, min(float(H), Y1))
                X2 = max(0.0, min(float(W), X2))
                Y2 = max(0.0, min(float(H), Y2))
                if X2 > X1 and Y2 > Y1:
                    dets_all.append((cls, conf, X1, Y1, X2, Y2))

    return dets_all


def filter_split_by_full(full_dets, split_dets, iou_thr=0.5):
    """
    FULL 박스 우선: split 박스가 동일 클래스 full 박스와 IoU >= iou_thr면 버림.

    return: kept_split (list of dets).
    """
    kept = []
    for sc, sconf, sx1, sy1, sx2, sy2 in split_dets:
        sbox = [sx1, sy1, sx2, sy2]
        drop = False
        for fc, fconf, fx1, fy1, fx2, fy2 in full_dets:
            if fc != sc:
                continue  # 동일 클래스만 비교
            fbox = [fx1, fy1, fx2, fy2]
            if iou_xyxy(sbox, fbox) >= iou_thr:
                drop = True
                break
        if not drop:
            kept.append((sc, sconf, sx1, sy1, sx2, sy2))
    return kept


# =========================
# 메인
# =========================
model = YOLO(model_path)
img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

for img_path in tqdm(img_list, desc="Auto Labeling (Full + 16-split, Full-priority)"):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read: {img_path}")
        continue
    H, W = img.shape[:2]

    # --------------------
    # 1) Full inference
    # --------------------
    dets_full = infer_full(model, img)

    # Full-only 시각화 저장 (요청: 분할 전 결과도 저장)
    img_full_vis = img.copy()
    for cls, conf, x1, y1, x2, y2 in dets_full:
        color = class_colors.get(cls, (255, 255, 255))
        cv2.rectangle(img_full_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img_full_vis,
            f"{class_names.get(cls, 'UNK')} {conf:.2f}",
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(os.path.join(vis_dir, f"{name}_full.png"), img_full_vis)

    # --------------------
    # 2) 16-split inference
    # --------------------
    dets_split = infer_grid_4x4(model, img)

    # --------------------
    # 3) Full 우선 병합
    #    - split에서 full과 IoU>=0.5(동일 클래스)면 제거
    #    - 남은 split만 full에 추가
    #    - 최종 클래스별 NMS로 정리
    # --------------------
    dets_split_kept = filter_split_by_full(dets_full, dets_split, iou_thr=IOU)
    dets_merged = dets_full + dets_split_kept
    dets_final = nms_per_class(dets_merged, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)

    # --------------------
    # 4) 저장: YOLO txt / Dreamer JSON / 최종 시각화
    # --------------------
    # YOLO txt
    yolo_lines = []
    # JSON
    shapes_json = []
    # 최종 시각화
    img_final_vis = img.copy()

    for cls, conf, x1, y1, x2, y2 in dets_final:
        # YOLO txt (상대좌표)
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # JSON (절대좌표)
        shapes_json.append(
            {
                "label": class_names.get(cls, "UNKNOWN"),
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

        # 최종 시각화
        color = class_colors.get(cls, (255, 255, 255))
        cv2.rectangle(img_final_vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            img_final_vis,
            f"{class_names.get(cls, 'UNK')} {conf:.2f}",
            (int(x1), max(15, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # txt 저장
    with open(os.path.join(yolo_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    # JSON 저장 (imageData 포함)
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
        "imagePath": f"DETECTION\\ATI\\colony_4\\{os.path.basename(img_path)}",  # 필요 시 수정
        "imageData": image_data_b64,
        "multichannel": {},
        "imageHeight": H,
        "imageWidth": W,
    }
    with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # 최종 시각화 저장
    cv2.imwrite(os.path.join(vis_dir, f"{name}_pred.png"), img_final_vis)

print("\n✅ Full + 16-split(Full-priority) 자동 라벨링 완료!")
print(f"YOLO txt   : {yolo_dir}")
print(f"JSON       : {json_dir}")
print(f"Visualize  : {vis_dir} (full-only: *_full.png, final: *_pred.png)")
