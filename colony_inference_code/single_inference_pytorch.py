import os
import glob
import cv2
from ultralytics import YOLO
from tqdm import tqdm
# import numpy as np # 이 코드에서는 필수는 아님

# =============================
# 설정 (사용자 지정)
# =============================
# 클래스/색 (시각화에 필요)
CLASS_NAMES = {0: "COLONY", 1: "USELESS"}
CLASS_COLONY_ID = 0
COLOR_COLONY   = (0, 255, 0)   # green
COLOR_USELESS  = (0, 0, 255)   # red

# 추론 시 사용할 기본값
DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.5
DEFAULT_MAX_DET = 3000

# =============================
# 시각화 & 저장 (간소화)
# =============================
def draw_and_save_simple(img_path, preds, out_dir="simple_results"):
    """
    추론 결과를 이미지에 그리고 지정된 폴더에 저장합니다.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return

    vis = img.copy()

    # 1. 저장 폴더 생성
    os.makedirs(out_dir, exist_ok=True)
    
    # 원본 파일명과 동일하게 저장
    name = os.path.basename(img_path) 
    save_path = os.path.join(out_dir, name)

    # 2. Detection 박스 시각화
    # preds: [cls, x1, y1, x2, y2, conf] 리스트
    for cls, x1, y1, x2, y2, conf in preds:
        color = COLOR_COLONY if cls == CLASS_COLONY_ID else COLOR_USELESS
        label = CLASS_NAMES.get(cls, "UNK")

        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 라벨 텍스트 (클래스명 + 신뢰도)
        text = f"{label} {conf:.2f}"
        cv2.putText(vis, text, (int(x1), max(15, int(y1)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. 결과 이미지 저장
    cv2.imwrite(save_path, vis)

# =============================
# 전체 이미지 처리 (간소화된 Single-Inference)
# =============================
def process_images_simple(model_path, dataset_root, out_dir="simple_results"):
    
    # 1. 모델 로드
    model = YOLO(model_path)
    
    # 2. 이미지 경로 탐색 (원본 코드와 동일하게 test 폴더만)
    img_dir = os.path.join(dataset_root, "test")
    
    images = []
    images += glob.glob(os.path.join(img_dir, "*.png"))
    images += glob.glob(os.path.join(img_dir, "*.jpg"))
    images = sorted(images)

    print(f"🔍 Total Images Found in '{img_dir}': {len(images)}")
    
    # 3. 각 이미지별 추론 및 저장
    for img_path in tqdm(images, desc="Running simple inference"):
        img = cv2.imread(img_path)
        if img is None: 
            continue

        # 4. (핵심) 단일 추론 실행
        # model.predict()는 리스트 형태의 결과를 반환하므로 [0]으로 첫 번째 결과에 접근
        res = model.predict(img, 
                            conf=DEFAULT_CONF, 
                            iou=DEFAULT_IOU, 
                            max_det=DEFAULT_MAX_DET, 
                            verbose=False)[0]

        # 5. 결과 포맷팅 [cls, x1, y1, x2, y2, conf]
        preds_list = []
        for b in res.boxes:
            cls = int(b.cls[0])
            confv = float(b.conf[0])
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            preds_list.append([cls, x1, y1, x2, y2, confv])

        # 6. 시각화 및 저장
        draw_and_save_simple(img_path, preds_list, out_dir=out_dir)

    print(f"✅ 완료! 결과는 {out_dir} 폴더에 저장되었습니다.")

# =============================
# 실행부
# =============================
if __name__ == "__main__":
    MODEL_PATH = "C:/workspace/ultralytics/experiments/colony_2class_small_noval/weights/best.pt"
    DATASET_ROOT = "C:/workspace/datasets/colony_2class_noval/images"
    OUTPUT_DIR = "simple_results" # 결과 저장 폴더

    process_images_simple(MODEL_PATH, DATASET_ROOT, out_dir=OUTPUT_DIR)