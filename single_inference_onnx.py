import cv2
import numpy as np
import onnxruntime as ort
import os

# =============================
# 1. 설정
# =============================

ONNX_MODEL_PATH = "onnx_weight\colony_model_opset12.onnx" 
IMAGE_PATH = "C:/workspace/datasets/colony_2class_noval/images/test/20251004_2025-10-03_Images_EBY100_YPD_d4_5_36h.png"
OUTPUT_DIR = "onnx_inference_cv2_nms_FIXED" # 최종 결과 폴더
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

CLASS_NAMES = {0: "COLONY", 1: "USELESS"} 
NUM_CLASSES = len(CLASS_NAMES)
COLOR_COLONY   = (0, 255, 0)
COLOR_USELESS  = (0, 0, 255)

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MAX_DET = 3000

# =============================
# 2. Letterbox 전처리 함수 (버그 수정된 최종본)
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
    
    dw /= 2 # center=True
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    # (핵심) 반올림된 *실제* 패딩 값 계산
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # (버그 수정) (dw, dh) 대신 (left, top) 반환
    return im, ratio, (left, top)

# =============================
# 3. ONNX 런타임 세션 준비
# =============================
print(f"모델 로드 중: {ONNX_MODEL_PATH}")
try:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    print(f"ONNX 런타임 세션 생성 성공 (Provider: {session.get_providers()[0]})")
except Exception as e:
    print(f"GPU(CUDA) 세션 생성 실패 (CPU로 전환): {e}")
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    print(f"ONNX 런타임 세션 생성 성공 (Provider: {session.get_providers()[0]})")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# =============================
# 4. 이미지 전처리 (auto=True)
# =============================
print(f"이미지 로드 및 전처리 (Letterbox, auto=True): {IMAGE_PATH}")
img_original = cv2.imread(IMAGE_PATH)
if img_original is None:
    print(f"오류: 이미지를 찾을 수 없습니다. (경로 확인): {IMAGE_PATH}")
    exit()

img_original_h, img_original_w = img_original.shape[:2]
img_resized, ratio, pad = letterbox(img_original, new_shape=(INPUT_WIDTH, INPUT_HEIGHT), auto=True)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_chw = np.transpose(img_rgb, (2, 0, 1))
input_tensor = img_chw.astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, axis=0)

# =============================
# 5. 추론 (Inference)
# =============================
print("추론 실행...")
outputs = session.run([output_name], {input_name: input_tensor})
detections = outputs[0]
print(f"   -> 추론 완료. 출력 Shape: {detections.shape}")

# =============================
# 6. 후처리 (cv2.dnn.NMSBoxes 형식 수정)
# =============================
print("후처리 (cv2.dnn.NMSBoxes + MaxDet) 실행...")

boxes_data = np.transpose(detections[0]) 
boxes_xywh = boxes_data[:, :4]
scores_data = boxes_data[:, 4:]

# (cx, cy, w, h) -> (x1, y1, x2, y2)
cx = boxes_xywh[:, 0]; cy = boxes_xywh[:, 1]
w = boxes_xywh[:, 2]; h = boxes_xywh[:, 3]
x1 = cx - w / 2; y1 = cy - h / 2
x2 = cx + w / 2; y2 = cy + h / 2
boxes_xyxy = np.column_stack([x1, y1, x2, y2])

nms_boxes = []
nms_scores = []
nms_class_ids = []

for cls_id in range(NUM_CLASSES):
    class_scores = scores_data[:, cls_id]
    mask = class_scores > CONF_THRESHOLD
    
    # (x1, y1, x2, y2) 형식의 박스들
    boxes_xyxy_this_class = boxes_xyxy[mask]
    scores_this_class = class_scores[mask]
    
    if len(scores_this_class) == 0:
        continue
        
    # (핵심) NMS를 위해 (x, y, w, h) 형식으로 변환
    x = boxes_xyxy_this_class[:, 0]
    y = boxes_xyxy_this_class[:, 1]
    w = boxes_xyxy_this_class[:, 2] - x
    h = boxes_xyxy_this_class[:, 3] - y
    boxes_xywh_this_class = np.column_stack([x, y, w, h])

    # NMS 실행 (C#의 NMSBoxes와 동일)
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh_this_class.tolist(), # (x, y, w, h) 리스트 전달 매우 중요
        scores_this_class.tolist(), 
        CONF_THRESHOLD, 
        IOU_THRESHOLD
    )

    if len(indices) > 0:
        final_indices = indices.flatten()
        for i in final_indices:
            # 저장할 때는 원본 (x1, y1, x2, y2) 박스를 저장
            nms_boxes.append(boxes_xyxy_this_class[i])
            nms_scores.append(scores_this_class[i])
            nms_class_ids.append(cls_id)

if len(nms_boxes) == 0:
    print("   -> 감지된 객체가 없습니다.")
else:
    print(f"   -> NMS 완료. {len(nms_boxes)}개 감지 (정렬 전).")
    
    # MaxDet 적용
    indices_sorted = np.argsort(nms_scores)[::-1]
    indices_limited = indices_sorted[:MAX_DET]

    final_boxes = [nms_boxes[i] for i in indices_limited]
    final_scores = [nms_scores[i] for i in indices_limited]
    final_class_ids = [nms_class_ids[i] for i in indices_limited]
    
    print(f"   -> MaxDet 적용. 최종 객체 {len(final_boxes)}개.")

# =============================
# 7. 시각화 및 저장 (버그 수정된 좌표계 사용)
# =============================
vis = img_original.copy()
ratio_w, ratio_h = ratio
pad_w, pad_h = pad # (left, top) 정수 값이 들어옴

if 'final_boxes' in locals() and len(final_boxes) > 0:
    for i in range(len(final_boxes)):
        box = final_boxes[i] # (x1, y1, x2, y2) 형식
        cls_id = final_class_ids[i]
        score = final_scores[i]
        
        # (정확) 반올림된 패딩(pad_w, pad_h)을 뺌
        x1_unpad = box[0] - pad_w
        y1_unpad = box[1] - pad_h
        x2_unpad = box[2] - pad_w
        y2_unpad = box[3] - pad_h
        
        # 원본 이미지 크기로 스케일링
        x1_orig = int(x1_unpad / ratio_w)
        y1_orig = int(y1_unpad / ratio_h)
        x2_orig = int(x2_unpad / ratio_w)
        y2_orig = int(y2_unpad / ratio_h)
        
        x1_orig = max(0, x1_orig)
        y1_orig = max(0, y1_orig)
        x2_orig = min(img_original_w, x2_orig)
        y2_orig = min(img_original_h, y2_orig)
        
        color = COLOR_COLONY if cls_id == 0 else COLOR_USELESS
        label = CLASS_NAMES.get(cls_id, "UNK")
        text = f"{label} {score:.2f}"

        cv2.rectangle(vis, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
        cv2.putText(vis, text, (x1_orig, max(15, y1_orig - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# 8. 결과 저장
output_filename = os.path.basename(IMAGE_PATH)
save_path = os.path.join(OUTPUT_DIR, output_filename)
cv2.imwrite(save_path, vis)

print(f"완료! 결과가 {save_path} 에 저장되었습니다.")