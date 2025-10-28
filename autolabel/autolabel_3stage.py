from ultralytics import YOLO
import cv2, os, json, glob, datetime, base64
import numpy as np
from tqdm import tqdm

# =========================
# 설정
# =========================
model_path  = "experiments/colony_2class_85_small/weights/best.pt"
img_dir     = "C:/workspace/datasets/colony_177"
output_root = "C:/workspace/autolabel_output"

yolo_dir = os.path.join(output_root, "labels_yolo")
json_dir = os.path.join(output_root, "labels_json")
vis_dir  = os.path.join(output_root, "visualize")
os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(vis_dir,  exist_ok=True)

# 클래스 이름/색상
class_names  = {0: "COLONY", 1: "USELESS"}
class_colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # COLONY=초록, USELESS=빨강

# 공통 추론 파라미터 (요청대로 통일)
CONF = 0.5
IOU  = 0.5
MAXDET_CROP = 2000

# 최종 병합 NMS
MERGE_NMS_IOU  = 0.5
MERGE_MAX_KEEP = 5000

# 그리드 설정
GRID_2x2 = (2, 2)  # 4분할
GRID_4x4 = (4, 4)  # 16분할

# =========================
# 유틸: IoU, NMS(클래스별)
# =========================
def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0: return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def nms_per_class(dets, iou_thresh=0.5, max_keep=5000):
    """
    dets: list[(cls, conf, x1, y1, x2, y2)]
    클래스별 NMS → keep 반환
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
            for j in range(i+1, len(items_sorted)):
                if suppressed[j]:
                    continue
                dj = items_sorted[j]
                if iou_xyxy(di[2:], dj[2:]) >= iou_thresh:
                    suppressed[j] = True
        kept.extend(keep_cls)

    kept = sorted(kept, key=lambda x: x[1], reverse=True)[:max_keep]
    return kept


def infer_full(model, img_bgr):
    """원본 이미지 전체 inference"""
    H, W = img_bgr.shape[:2]
    res = model(img_bgr, conf=CONF, iou=IOU, max_det=MAXDET_CROP, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = int(b.cls[0]); conf = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        # clamp
        x1 = max(0.0, min(float(W), x1)); y1 = max(0.0, min(float(H), y1))
        x2 = max(0.0, min(float(W), x2)); y2 = max(0.0, min(float(H), y2))
        if x2 > x1 and y2 > y1:
            dets.append((cls, conf, x1, y1, x2, y2))
    return dets

def infer_grid(model, img_bgr, rows, cols):
    """
    rows x cols 그리드로 나눠 inference → 원본 좌표로 복원
    return: list[(cls, conf, x1, y1, x2, y2)]
    """
    H, W = img_bgr.shape[:2]
    dets_all = []
    xs = [int(round(W * i / cols)) for i in range(cols + 1)]
    ys = [int(round(H * j / rows)) for j in range(rows + 1)]

    for r in range(rows):
        for c in range(cols):
            x1g, x2g = xs[c], xs[c+1]
            y1g, y2g = ys[r], ys[r+1]
            if x2g <= x1g or y2g <= y1g:
                continue
            crop = img_bgr[y1g:y2g, x1g:x2g]
            res = model(crop, conf=CONF, iou=IOU, max_det=MAXDET_CROP, verbose=False)[0]
            for b in res.boxes:
                cls  = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = map(float, b.xyxy[0])  # crop 좌표
                X1 = x1 + x1g; Y1 = y1 + y1g
                X2 = x2 + x1g; Y2 = y2 + y1g
                # clamp
                X1 = max(0.0, min(float(W), X1)); Y1 = max(0.0, min(float(H), Y1))
                X2 = max(0.0, min(float(W), X2)); Y2 = max(0.0, min(float(H), Y2))
                if X2 > X1 and Y2 > Y1:
                    dets_all.append((cls, conf, X1, Y1, X2, Y2))
    return dets_all


def filter_by_higher_priority(kept_higher, candidate_lower, iou_thr=0.5):
    """
    kept_higher: 상위 우선순위에서 이미 채택된 박스들 (삭제 안 함)
    candidate_lower: 하위 단계의 후보 박스들
    규칙: 같은 클래스에서 IoU≥iou_thr이면 candidate 제거 -> “상위 박스와 겹치면 하위 박스를 버리는” 기능을 수행합니다.
    """
    out = []
    for sc, sconf, sx1, sy1, sx2, sy2 in candidate_lower:
        sbox = [sx1, sy1, sx2, sy2]
        drop = False
        for hc, hconf, hx1, hy1, hx2, hy2 in kept_higher:
            if sc != hc:
                continue
            if iou_xyxy(sbox, [hx1, hy1, hx2, hy2]) >= iou_thr:
                drop = True
                break
        if not drop:
            out.append((sc, sconf, sx1, sy1, sx2, sy2))
    return out


def save_outputs(name, img_bgr, dets_final, yolo_dir, json_dir, vis_dir):
    H, W = img_bgr.shape[:2]

    # YOLO txt
    yolo_lines = []
    for cls, conf, x1, y1, x2, y2 in dets_final:
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(os.path.join(yolo_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    # Dreamer JSON (imageData 포함)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(img_dir, f"{name}.png"), "rb") if os.path.exists(os.path.join(img_dir, f"{name}.png")) \
         else open(os.path.join(img_dir, f"{name}.jpg"), "rb") as f:
        image_data_b64 = base64.b64encode(f.read()).decode("utf-8")

    shapes_json = [{
        "label": class_names.get(cls, "UNKNOWN"),
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {}
    } for cls, conf, x1, y1, x2, y2 in dets_final]

    json_data = {
        "version": "1.4.1",
        "user": "Offline User",
        "time": now,
        "label_dict": {"COLONY": 0, "USELESS": 1},
        "flags": {},
        "shapes": shapes_json,
        "imagePath": f"DETECTION\\ATI\\colony_4\\{name}.png",  # 필요 시 수정
        "imageData": image_data_b64,
        "multichannel": {},
        "imageHeight": H,
        "imageWidth":  W
    }
    with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # 시각화
    vis = img_bgr.copy()
    for cls, conf, x1, y1, x2, y2 in dets_final:
        color = class_colors.get(cls, (255, 255, 255))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"{class_names.get(cls,'UNK')} {conf:.2f}",
                    (int(x1), max(15, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(vis_dir, f"{name}_pred.png"), vis)


# =========================
# 메인
# =========================
model = YOLO(model_path)
img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

for img_path in tqdm(img_list, desc="Auto Labeling (Full → 4-split → 16-split, priority merge)"):
    name = os.path.splitext(os.path.basename(img_path))[0]
    img  = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read: {img_path}")
        continue

    # 1) Full
    det_full = infer_full(model, img)

    # Full-only 시각화도 저장(비교용)
    vis_full = img.copy()
    for cls, conf, x1, y1, x2, y2 in det_full:
        color = class_colors.get(cls, (255, 255, 255))
        cv2.rectangle(vis_full, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis_full, f"{class_names.get(cls,'UNK')} {conf:.2f}",
                    (int(x1), max(15, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(vis_dir, f"{name}_full.png"), vis_full)

    # 2) 4-split
    r2, c2 = GRID_2x2
    det_4  = infer_grid(model, img, r2, c2)
    det_4_kept = filter_by_higher_priority(det_full, det_4, iou_thr=IOU)

    # 3) 16-split
    r4, c4 = GRID_4x4
    det_16 = infer_grid(model, img, r4, c4)
    det_16_kept = filter_by_higher_priority(det_full + det_4_kept, det_16, iou_thr=IOU)

    # 4) 최종 병합 + NMS
    det_merged = det_full + det_4_kept + det_16_kept
    det_final  = nms_per_class(det_merged, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)

    # 5) 저장
    save_outputs(name, img, det_final, yolo_dir, json_dir, vis_dir)

print("\n✅ Full → 4-split → 16-split (우선순위 병합) 자동 라벨링 완료!")
print(f"YOLO txt   : {yolo_dir}")
print(f"JSON       : {json_dir}")
print(f"Visualize  : {vis_dir} (full-only: *_full.png, final: *_pred.png)")
