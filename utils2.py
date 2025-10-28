import os
import cv2
import numpy as np

# =============================
# IoU / IoA 함수
# =============================
def _intersection(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou_xyxy(a, b):
    inter = _intersection(a, b)
    if inter <= 0: return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def ioa_smaller(a, b):
    inter = _intersection(a, b)
    if inter <= 0: return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0

# =============================
# 정사각형 보정
# =============================
def make_square_box(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x1n = max(0, cx - side / 2)
    y1n = max(0, cy - side / 2)
    x2n = min(img_w, cx + side / 2)
    y2n = min(img_h, cy + side / 2)
    return x1n, y1n, x2n, y2n

# =============================
# NMS
# =============================
def global_nms(dets, iou_thresh=0.5, max_keep=5000):
    dets = sorted(dets, key=lambda x: (x[0] != CLASS_COLONY_ID, -x[5]))
    kept = []
    suppressed = [False] * len(dets)
    for i, di in enumerate(dets):
        if suppressed[i]: continue
        kept.append(di)
        if len(kept) >= max_keep: break
        for j in range(i+1, len(dets)):
            if suppressed[j]: continue
            if iou_xyxy(di[1:5], dets[j][1:5]) >= iou_thresh:
                suppressed[j] = True
    return kept

# =============================
# Isolated 필터
# =============================
def filter_isolated_boxes(preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR):
    iso = []
    for i, bi in enumerate(preds):
        independent = True
        for j, bj in enumerate(preds):
            if i == j: continue
            if (iou_xyxy(bi[1:5], bj[1:5]) >= thr_iou) or (ioa_smaller(bi[1:5], bj[1:5]) >= thr_ioa):
                independent = False
                break
        if independent:
            iso.append(bi)
    return iso

# =============================
# YOLO inference
# =============================
def infer_full_image(model, img, conf=0.5, iou=0.5, max_det=3000):
    res = model.predict(img, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    preds = []
    for b in res.boxes:
        cls = int(b.cls[0]); confv = float(b.conf[0])
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        preds.append([cls, x1, y1, x2, y2, confv])
    return preds

def infer_grid_with_overlap(model, img, rows, cols, conf, iou, max_det=3000, overlap_ratio=0.2):
    H, W = img.shape[:2]
    preds = []
    cw, ch = W / cols, H / rows

    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(cw * c), int(ch * r)
            x2, y2 = int(cw * (c + 1)), int(ch * (r + 1))
            x1o = int(max(0, x1 - cw * overlap_ratio))
            y1o = int(max(0, y1 - ch * overlap_ratio))
            x2o = int(min(W, x2 + cw * overlap_ratio))
            y2o = int(min(H, y2 + ch * overlap_ratio))

            crop = img[y1o:y2o, x1o:x2o]
            if crop.size == 0: continue
            res = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
            for b in res.boxes:
                cls = int(b.cls[0]); confv = float(b.conf[0])
                bx1, by1, bx2, by2 = map(float, b.xyxy[0])
                preds.append([cls, bx1+x1o, by1+y1o, bx2+x1o, by2+y1o, confv])
    return preds

def infer_multiscale(model, img, overlap_ratio=0.2):
    H, W = img.shape[:2]
    merged = []
    for cfg in STAGE_CONFIGS:
        preds = infer_grid_with_overlap(model, img, cfg["rows"], cfg["cols"],
                                        cfg["conf"], cfg["iou"], cfg["max_det"], overlap_ratio)
        merged += preds

    sq = []
    for cls, x1, y1, x2, y2, conf in merged:
        if cls == CLASS_COLONY_ID:
            x1s, y1s, x2s, y2s = make_square_box(x1, y1, x2, y2, W, H)
            sq.append([cls, x1s, y1s, x2s, y2s, conf])
        else:
            sq.append([cls, x1, y1, x2, y2, conf])
    return global_nms(sq, MERGE_NMS_IOU, MERGE_MAX_KEEP)

def infer_multiscale_isolated_colony(model, img, overlap_ratio=0.2):
    ms = infer_multiscale(model, img, overlap_ratio)
    iso = filter_isolated_boxes(ms, ISOLATED_IOU_THR, ISOLATED_IOA_THR)
    return [b for b in iso if b[0] == CLASS_COLONY_ID]

# =============================
# GT 로드 (YOLO → xyxy)
# =============================
def yolo_to_xyxy(cx, cy, w, h, W, H):
    cx, cy, w, h = cx*W, cy*H, w*W, h*H
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    return [x1, y1, x2, y2]

def load_gt_boxes(img_path, labels_dir, img_shape):
    H, W = img_shape[:2]
    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, name + ".txt")
    if not os.path.exists(label_path):
        return []
    gts = []
    with open(label_path, "r") as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.strip().split()[:5])
            gts.append([int(cls), *yolo_to_xyxy(cx, cy, w, h, W, H)])
    return gts

# =============================
# TP / FP / FN 계산
# =============================
def match_and_count(preds, gts, iou_thr=0.5):
    matched = set()
    TP = FP = 0
    for p in preds:
        p_cls = p[0]; p_box = p[1:5]
        best_iou, best_idx = 0, -1
        for i, g in enumerate(gts):
            if i in matched: continue
            if g[0] != p_cls: continue
            iouv = iou_xyxy(p_box, g[1:5])
            if iouv > best_iou:
                best_iou, best_idx = iouv, i
        if best_iou >= iou_thr:
            TP += 1
            matched.add(best_idx)
        else:
            FP += 1
    FN = len(gts) - len(matched)
    return TP, FP, FN

# =============================
# 평가 함수
# =============================
def evaluate_method(model, img_paths, labels_dir, method, iou_thr=0.5, overlap_ratio=0.2):
    TP = FP = FN = 0
    for img_path in tqdm(img_paths, desc=f"{method.upper()} 평가"):
        img = cv2.imread(img_path)
        if img is None: continue
        gts = load_gt_boxes(img_path, labels_dir, img.shape)

        if method == "full":
            preds = infer_full_image(model, img)
        elif method == "ms":
            preds = infer_multiscale(model, img, overlap_ratio)
        elif method == "isolated":
            preds = infer_multiscale_isolated_colony(model, img, overlap_ratio)
        else:
            raise ValueError("Unknown method")

        t, f, n = match_and_count(preds, gts, iou_thr)
        TP += t; FP += f; FN += n

    P = TP / (TP + FP + 1e-6)
    R = TP / (TP + FN + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6)

    print(f"\n📊 {method.upper()} 결과")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision={P:.4f}, Recall={R:.4f}, F1={F1:.4f}\n")
    return F1

def infer_full_image_with_conf(model, img, conf, iou=0.5, max_det=3000):
    return infer_full_image(model, img, conf=conf, iou=iou, max_det=max_det)

def infer_multiscale_with_conf(model, img, conf, overlap_ratio=0.2):
    # STAGE_CONFIGS의 conf 값을 동적으로 변경
    original_confs = [cfg["conf"] for cfg in STAGE_CONFIGS]
    for cfg in STAGE_CONFIGS:
        cfg["conf"] = conf
    preds = infer_multiscale(model, img, overlap_ratio)
    # 변경 후 원복(안정성 위해)
    for cfg, orig in zip(STAGE_CONFIGS, original_confs):
        cfg["conf"] = orig
    return preds

def sweep_conf_thresholds(model, img_paths, labels_dir, method, conf_list, iou_thr=0.5, overlap_ratio=0.2):
    """
    여러 confidence 값에 대해 F1 score를 계산하고 결과를 출력
    """
    best_conf = None
    best_f1 = -1
    result_table = []

    for conf in conf_list:
        print(f"\n🚀 Confidence={conf:.2f} 평가 중...")
        
        # 각 방식에 맞는 inference 적용
        def _infer(img):
            if method == "full":
                return infer_full_image_with_conf(model, img, conf, iou=iou_thr)
            elif method == "ms":
                return infer_multiscale_with_conf(model, img, conf, overlap_ratio)
            else:
                raise ValueError("Unknown method")
        
        TP = FP = FN = 0
        for img_path in tqdm(img_paths, desc=f"{method.upper()} conf={conf:.2f}"):
            img = cv2.imread(img_path)
            if img is None: continue
            gts = load_gt_boxes(img_path, labels_dir, img.shape)
            preds = _infer(img)
            t, f, n = match_and_count(preds, gts, iou_thr)
            TP += t; FP += f; FN += n

        P = TP / (TP + FP + 1e-6)
        R = TP / (TP + FN + 1e-6)
        F1 = 2 * P * R / (P + R + 1e-6)

        print(f"Conf={conf:.2f} → Precision={P:.4f}, Recall={R:.4f}, F1={F1:.4f}")
        result_table.append((conf, P, R, F1))

        if F1 > best_f1:
            best_f1 = F1
            best_conf = conf

    print("\n🔥 최종 결과 요약 (Conf Sweep)")
    for conf, P, R, F1 in result_table:
        print(f"conf={conf:.2f} | Precision={P:.4f} | Recall={R:.4f} | F1={F1:.4f}")
    print(f"\n✅ Best Conf for {method.upper()} = {best_conf:.2f} (F1={best_f1:.4f})")

    return best_conf, best_f1

