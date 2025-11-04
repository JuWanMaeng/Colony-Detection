import os
import glob
import cv2
from tqdm import tqdm
import numpy as np
import onnxruntime as ort

# =============================
# Stage 설정 (원본과 동일)
# =============================
STAGE_CONFIGS = [
    {"rows": 1, "cols": 1, "conf": 0.5, "iou": 0.5, "max_det": 3000},  # Full (1x1)
    {"rows": 2, "cols": 2, "conf": 0.5, "iou": 0.4, "max_det": 3000},  # 2x2
    {"rows": 4, "cols": 4, "conf": 0.5, "iou": 0.3, "max_det": 3000},  # 4x4,
]

MERGE_NMS_IOU   = 0.1
MERGE_MAX_KEEP  = 5000

# Isolated 판정 파라미터 (원본과 동일)
ISOLATED_IOU_THR = 0.05
ISOLATED_IOA_THR = 0.05

# 클래스/색 (원본과 동일)
CLASS_NAMES = {0: "COLONY", 1: "USELESS"}
NUM_CLASSES = len(CLASS_NAMES)
CLASS_COLONY_ID = 0
COLOR_COLONY   = (0, 255, 0)   # green
COLOR_USELESS  = (0, 0, 255)   # red
COLOR_ISOLATED = (255, 0, 0)   # blue

# ONNX 모델 입력 크기
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# =============================
# 유틸: IoU/IoA 계산 (원본과 동일)
# =============================
def _intersection(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou_xyxy(a, b):
    inter = _intersection(a, b)
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def ioa_smaller(a, b):
    inter = _intersection(a, b)
    if inter <= 0: return 0.0
    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    denom = min(area_a, area_b)
    return inter / denom if denom > 0 else 0.0

# =============================
# 정사각형 변환 (원본과 동일)
# =============================
def make_square_box(x1, y1, x2, y2, img_w, img_h):
    w = x2 - x1; h = y2 - y1
    side = max(w, h)
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    x1_new = max(0.0, cx - side / 2.0)
    y1_new = max(0.0, cy - side / 2.0)
    x2_new = min(float(img_w), cx + side / 2.0)
    y2_new = min(float(img_h), cy + side / 2.0)
    return x1_new, y1_new, x2_new, y2_new

# =============================
# Global NMS (원본과 동일)
# =============================
def global_nms(dets, iou_thresh=0.5, max_keep=5000):
    dets = sorted(dets, key=lambda x: -x[5]) # conf 내림차순
    kept = []
    suppressed = [False] * len(dets)
    for i, di in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(di)
        if len(kept) >= max_keep:
            break
        for j in range(i+1, len(dets)):
            if suppressed[j]: continue
            dj = dets[j]
            if iou_xyxy(di[1:5], dj[1:5]) >= iou_thresh:
                suppressed[j] = True
    return kept

# =============================
# Isolated 필터 (원본과 동일)
# =============================
def filter_isolated_boxes(preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR):
    isolated = []
    for i, bi in enumerate(preds):
        box_i = bi[1:5]
        independent = True
        for j, bj in enumerate(preds):
            if i == j: continue
            box_j = bj[1:5]
            if (iou_xyxy(box_i, box_j) >= thr_iou) or (ioa_smaller(box_i, box_j) >= thr_ioa):
                independent = False
                break
        if independent:
            isolated.append(bi)
    return isolated

# =============================
# ONNX: Letterbox (C# 포팅용 최종본)
# =============================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto: 
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top) # (left, top) 정수 패딩 반환

# =============================
# ONNX: Full Prediction Pipeline (model.predict() 대체 함수)
# =============================
def run_onnx_prediction(session, img_original, conf_thresh, iou_thresh, max_det):
    """
    ONNX 세션을 사용하여 단일 이미지(또는 타일)에 대해
    전처리, 추론, 후처리(NMS)를 모두 수행합니다.
    C#에서 이 함수 내부의 로직을 구현해야 합니다.
    """
    
    # --- 1. 전처리 (Letterbox) ---
    if img_original is None or img_original.size == 0:
        return []
    
    img_original_h, img_original_w = img_original.shape[:2]
    img_resized, ratio, pad = letterbox(img_original, new_shape=(INPUT_WIDTH, INPUT_HEIGHT), auto=False)
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = img_chw.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # --- 2. 추론 (Inference) ---
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    detections = outputs[0]

    # --- 3. 후처리 (NMS) ---
    boxes_data = np.transpose(detections[0]) 
    boxes_xywh = boxes_data[:, :4] # (cx, cy, w, h)
    scores_data = boxes_data[:, 4:] # (scores...)

    # (cx, cy, w, h) -> (x1, y1, x2, y2)
    cx = boxes_xywh[:, 0]; cy = boxes_xywh[:, 1]
    w = boxes_xywh[:, 2]; h = boxes_xywh[:, 3]
    x1 = cx - w / 2; y1 = cy - h / 2
    x2 = cx + w / 2; y2 = cy + h / 2
    boxes_xyxy = np.column_stack([x1, y1, x2, y2])

    nms_boxes = []
    nms_scores = []
    nms_class_ids = []

    # Class-Specific NMS
    for cls_id in range(NUM_CLASSES):
        class_scores = scores_data[:, cls_id]
        mask = class_scores > conf_thresh
        
        boxes_xyxy_this_class = boxes_xyxy[mask]
        scores_this_class = class_scores[mask]
        
        if len(scores_this_class) == 0:
            continue
            
        # NMS를 위해 (x, y, w, h) 형식으로 변환 (C# NMSBoxes용)
        x = boxes_xyxy_this_class[:, 0]
        y = boxes_xyxy_this_class[:, 1]
        w_box = boxes_xyxy_this_class[:, 2] - x
        h_box = boxes_xyxy_this_class[:, 3] - y
        boxes_xywh_this_class = np.column_stack([x, y, w_box, h_box])

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh_this_class.tolist(),
            scores_this_class.tolist(), 
            conf_thresh, 
            iou_thresh
        )

        if len(indices) > 0:
            final_indices = indices.flatten()
            for i in final_indices:
                nms_boxes.append(boxes_xyxy_this_class[i])
                nms_scores.append(scores_this_class[i])
                nms_class_ids.append(cls_id)

    if len(nms_boxes) == 0:
        return []

    # MaxDet 적용
    indices_sorted = np.argsort(nms_scores)[::-1]
    indices_limited = indices_sorted[:max_det]

    # --- 4. 좌표 복원 (C#에서 동일하게 구현 필요) ---
    final_preds_list = []
    
    ratio_w, ratio_h = ratio
    pad_w, pad_h = pad # (left, top) 정수 패딩

    for i in indices_limited:
        box = nms_boxes[i] # (x1, y1, x2, y2)
        cls_id = nms_class_ids[i]
        score = nms_scores[i]
        
        # (정확) 패딩(pad_w, pad_h) 제거
        x1_unpad = box[0] - pad_w
        y1_unpad = box[1] - pad_h
        x2_unpad = box[2] - pad_w
        y2_unpad = box[3] - pad_h
        
        # 원본 타일 이미지 크기로 스케일링
        x1_orig = x1_unpad / ratio_w
        y1_orig = y1_unpad / ratio_h
        x2_orig = x2_unpad / ratio_w
        y2_orig = y2_unpad / ratio_h
        
        # 클리핑 (C#에서도 필요)
        x1_orig = max(0.0, x1_orig)
        y1_orig = max(0.0, y1_orig)
        x2_orig = min(float(img_original_w), x2_orig)
        y2_orig = min(float(img_original_h), y2_orig)
        
        # [cls, x1, y1, x2, y2, conf] 형식으로 반환
        final_preds_list.append([cls_id, x1_orig, y1_orig, x2_orig, y2_orig, score])
        
    return final_preds_list

# =============================
# ONNX: Grid-based Inference
# =============================
def infer_grid_with_overlap_onnx(session, img, rows, cols, conf, iou, max_det, overlap_ratio=0.2):
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

            # (변경) PyTorch model.predict() -> run_onnx_prediction()
            # tile_preds는 (x1, y1)이 (0,0) 기준인 타일 좌표계
            tile_preds = run_onnx_prediction(
                session, crop, 
                conf_thresh=conf, iou_thresh=iou, max_det=max_det
            )

            for p in tile_preds:
                cls, bx1, by1, bx2, by2, confv = p
                
                # crop 좌표 -> 원본 (Full) 좌표
                X1 = bx1 + x1_ov
                Y1 = by1 + y1_ov
                X2 = bx2 + x1_ov
                Y2 = by2 + y1_ov

                preds.append([cls, X1, Y1, X2, Y2, confv])
    return preds

# =============================
# ONNX: Multi-Scale Inference
# =============================
def infer_multiscale_onnx(session, img, overlap_ratio=0.2):
    H, W = img.shape[:2]
    merged = []

    for cfg in STAGE_CONFIGS:
        r, c = cfg["rows"], cfg["cols"]
        
        # (변경) ONNX 기반 그리드 추론 호출
        merged += infer_grid_with_overlap_onnx(
            session, img,
            rows=r, cols=c,
            conf=cfg["conf"], iou=cfg["iou"], max_det=cfg["max_det"],
            overlap_ratio=overlap_ratio
        )

    # COLONY만 정사각형 보정 (원본과 동일)
    sq = []
    for cls, x1, y1, x2, y2, conf in merged:
        if cls == CLASS_COLONY_ID:
            x1s, y1s, x2s, y2s = make_square_box(x1, y1, x2, y2, W, H)
            sq.append([cls, x1s, y1s, x2s, y2s, conf])
        else:
            sq.append([cls, x1, y1, x2, y2, conf])

    # Global NMS (원본과 동일)
    return global_nms(sq, iou_thresh=MERGE_NMS_IOU, max_keep=MERGE_MAX_KEEP)

# =============================
# ONNX: Multi-Scale Isolated
# =============================
def infer_multiscale_isolated_colony_onnx(session, img, overlap_ratio=0.2):
    # (변경) ONNX 멀티스케일 호출
    ms_preds = infer_multiscale_onnx(session, img, overlap_ratio=overlap_ratio)
    
    iso = filter_isolated_boxes(ms_preds, thr_iou=ISOLATED_IOU_THR, thr_ioa=ISOLATED_IOA_THR)
    iso_colony = [p for p in iso if p[0] == CLASS_COLONY_ID]
    return iso_colony

# =============================
# 시각화 & 저장 (원본과 동일)
# =============================
def draw_and_save(img_path, preds, mode, out_dir="total_results_onnx", cut_ratio=False):
    img = cv2.imread(img_path)
    if img is None:
        return
    vis = img.copy()
    H, W = vis.shape[:2]

    name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(out_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    input_path = os.path.join(save_dir, "input.png")
    if not os.path.exists(input_path):
        cv2.imwrite(input_path, img)

    if cut_ratio:
        x_min_cut = W * cut_ratio
        x_max_cut = W * (1 - cut_ratio)
        y_min_cut = H * cut_ratio
        y_max_cut = H * (1 - cut_ratio)

    for cls, x1, y1, x2, y2, conf in preds:
        if cut_ratio:
            if (x1 < x_min_cut) or (y1 < y_min_cut) or (x2 > x_max_cut) or (y2 > y_max_cut):
                continue

        if mode == "isolated":
            color = COLOR_ISOLATED
            label = None
        else:
            color = COLOR_COLONY if cls == CLASS_COLONY_ID else COLOR_USELESS
            label = f"{CLASS_NAMES.get(cls, 'UNK')} {conf:.2f}" # 신뢰도 표시

        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if label is not None:
            cv2.putText(vis, label, (int(x1), max(15, int(y1)-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    suffix_map = {"full": "single_inference.png", "ms": "multi_scale_inference.png", "isolated": "isolated.png"}
    suffix = suffix_map.get(mode, "result.png")
    cv2.imwrite(os.path.join(save_dir, suffix), vis)

# =============================
# ONNX: 전체 이미지 처리 (메인 루프)
# =============================
def process_all_images_onnx(onnx_model_path, dataset_root, overlap_ratio=0.2, cut_ratio = False):
    
    # (변경) YOLO 모델 로드 -> ONNX 세션 로드
    print(f"모델 로드 중: {onnx_model_path}")
    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        print(f"ONNX 런타임 세션 생성 성공 (Provider: {session.get_providers()[0]})")
    except Exception as e:
        print(f"GPU(CUDA) 세션 생성 실패 (CPU로 전환): {e}")
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_model_path, providers=providers)
        print(f"ONNX 런타임 세션 생성 성공 (Provider: {session.get_providers()[0]})")

    
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

    print(f"Total Images Found: {len(images)}")
    for img_path in tqdm(images, desc="Running ONNX Full/MS/Isolated"):
        img = cv2.imread(img_path)
        if img is None: continue

        # (변경) 1) Full Prediction (ONNX)
        full_cfg = STAGE_CONFIGS[0] # 1x1 config
        full_preds = run_onnx_prediction(
            session, img, 
            conf_thresh=full_cfg["conf"], 
            iou_thresh=full_cfg["iou"], 
            max_det=full_cfg["max_det"]
        )
        draw_and_save(img_path, full_preds, mode="full", out_dir="total_results_onnx", cut_ratio=cut_ratio)

        # (변경) 2) Multi-Scale Prediction (ONNX)
        ms_preds = infer_multiscale_onnx(session, img, overlap_ratio=overlap_ratio)
        draw_and_save(img_path, ms_preds, mode="ms", out_dir="total_results_onnx", cut_ratio=cut_ratio)

        # (변경) 3) Isolated (ONNX)
        # (ms_preds를 재사용하거나, 격리 필터만 따로 실행)
        # 원본 스크립트는 infer_multiscale_isolated_colony 함수를 사용
        iso_preds = infer_multiscale_isolated_colony_onnx(session, img, overlap_ratio=overlap_ratio)
        draw_and_save(img_path, iso_preds, mode="isolated", out_dir="total_results_onnx", cut_ratio=cut_ratio)

    print("완료! 결과는 total_results_onnx 폴더에 저장되었습니다.")

# =============================
# 실행부
# =============================
if __name__ == "__main__":
    # (변경) .pt 대신 .onnx 모델 경로
    ONNX_MODEL_PATH = "onnx_weight\colony_model_opset12.onnx" # 또는 "colony_model_opset17_nosimplify.onnx"
    
    DATASET_ROOT = "C:/workspace/datasets/colony_2class_noval/images"
    cut_ratio = 0.05
    
    # (변경) ONNX 버전의 메인 함수 호출
    process_all_images_onnx(
        ONNX_MODEL_PATH, 
        DATASET_ROOT, 
        overlap_ratio=0.2, 
        cut_ratio = cut_ratio
    )