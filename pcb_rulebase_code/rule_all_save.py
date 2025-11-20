import cv2
import numpy as np
import os
import glob

def find_all_defects(image, params): 
    """
    이미지에서 룰베이스로 "모든" 결함을 찾고 바운딩 박스 "리스트"를 반환합니다.
    (튜닝된 최종 룰셋 적용)
    """
    # 1. 이미지의 높이, 너비 저장
    (img_height, img_width) = image.shape[:2]
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 (cv2.inRange 사용) - params에서 값 가져오기
    lower_bound = params["MIN_BRIGHTNESS"]
    upper_bound = params["MAX_BRIGHTNESS"]
    binary_mask = cv2.inRange(gray, lower_bound, upper_bound)
    
    # 4. 윤곽선(Contours) 찾기
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 모서리 제외를 위한 "중앙 안전 영역" 경계 계산 - params에서 값 가져오기
    margin_ratio = params["EDGE_MARGIN_RATIO"]
    x_min_safe = img_width * margin_ratio
    x_max_safe = img_width * (1 - margin_ratio) 
    y_min_safe = img_height * margin_ratio
    y_max_safe = img_height * (1 - margin_ratio)
    
    bboxes = [] # 6. 모든 결함의 bbox를 저장할 리스트
    
    # 7. 모든 윤곽선 검사
    for c in contours:
        
        # 7-1. 바운딩 박스 계산
        (x, y, w, h) = cv2.boundingRect(c)
        
        # 7-2. 크기(면적) 필터링 - params에서 값 가져오기
        area = cv2.contourArea(c)
        if area < params["MIN_AREA"] or area > params["MAX_AREA"]:
            continue 
            
        # 7-3. (종횡비 룰 삭제됨)
        
        # 7-4. 모서리 영역 필터링 (최종 복합 룰)
        # 기준 좌표 정의 (★ 사용자가 마지막에 수정한 로직 ★)
        top_left_x = x
        top_left_y = y
        top_right_x = x + w
        bottom_y = y + h # (top_right_y -> bottom_y 로 명확화)

        # 룰 적용
        if (top_left_x < x_min_safe) or \
           (top_right_x > x_max_safe) or \
           (top_left_y < y_min_safe) or \
           (bottom_y > y_max_safe): # 객체의 바닥(y+h)이 경계 침범 시
            continue 

        # 7-5. 모든 필터를 통과한 경우에만 바운딩 박스 추가
        bboxes.append((x, y, w, h))

    # 8. 모든 바운딩 박스 리스트와 이미지 크기 반환
    return bboxes, (img_height, img_width)

# --- 메인 코드 실행 (YOLO .txt 저장 모드) ---

# ----------------------------------------------------
# 📌 (필수) 하이퍼파라미터 (튜닝 완료된 값)
# ----------------------------------------------------
PARAMS = {
    # 1. 밝기 범위 (0 ~ 255)
    "MIN_BRIGHTNESS": 75,
    "MAX_BRIGHTNESS": 180,
    
    # 2. 결함 크기 (픽셀 면적)
    "MIN_AREA": 2,
    "MAX_AREA": 500,
    
    # 3. 모서리 제외 비율 (0.0 ~ 0.5)
    "EDGE_MARGIN_RATIO": 0.01
}

# --- 작업 설정 ---
INPUT_DIR = r"C:\data\product2" 
IMAGE_EXTENSIONS = ["*.bmp", "*.jpg", "*.png", "*.jpeg"]

# ★ (필수) .txt 라벨을 저장할 폴더
OUTPUT_DIR = r"C:\data\product2_labels" # 예시 (없으면 자동 생성)

# ★ (필수) YOLO 클래스 ID
CLASS_ID = 0
# ----------------------------------------------------


# 라벨 저장 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모든 이미지 파일 경로 찾기
image_files = []
for ext in IMAGE_EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

# 하이퍼파라미터 값 출력
print(f"--- 총 {len(image_files)}개의 이미지를 처리하여 YOLO .txt 파일로 저장합니다. ---")
print(f"    [저장 폴더] {OUTPUT_DIR}")
print(f"    [적용된 룰]")
print(f"    - Brightness: {PARAMS['MIN_BRIGHTNESS']}~{PARAMS['MAX_BRIGHTNESS']}")
print(f"    - Area: {PARAMS['MIN_AREA']}~{PARAMS['MAX_AREA']}")
print(f"    - Edge Margin: {PARAMS['EDGE_MARGIN_RATIO'] * 100:.0f}% (최종 복합 룰 기준)")
print("--------------------------------------------------\n")

for img_path in image_files:
    # 1. 이미지 읽기
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"⚠️ {img_path} 파일을 읽을 수 없습니다.")
        continue
        
    # 2. 룰베이스로 "모든" 결함 검출
    bboxes, (img_height, img_width) = find_all_defects(
        original_image, 
        PARAMS 
    )
    
    base_filename = os.path.basename(img_path)
    # .txt 파일명 생성 (예: image.bmp -> image.txt)
    txt_filename = os.path.splitext(base_filename)[0] + ".txt"
    output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    
    yolo_lines = [] # .txt 파일에 쓸 모든 라인을 저장할 리스트
    
    if bboxes:
        # 3. 검출된 모든 bbox에 대해 YOLO 포맷으로 변환
        for (x, y, w, h) in bboxes:
            # (픽셀 -> 상대값, 좌상단 -> 중심점)
            x_center_rel = (x + w / 2) / img_width
            y_center_rel = (y + h / 2) / img_height
            width_rel = w / img_width
            height_rel = h / img_height
            
            # 4. YOLO 포맷 문자열 생성
            yolo_string = f"{CLASS_ID} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}\n"
            yolo_lines.append(yolo_string)
        
        # 5. TXT 파일로 "한 번에" 저장
        with open(output_txt_path, 'w') as f:
            f.writelines(yolo_lines)
        
        print(f"✅ [ {base_filename} ] -> {len(bboxes)}개 결함 저장 완료 ({txt_filename})")

    else:
        # 6. 결함이 없는 경우 (YOLO 학습을 위해 빈 파일을 생성)
        with open(output_txt_path, 'w') as f:
            pass # 빈 파일 생성
            
        print(f"❌ [ {base_filename} ] : 검출된 결함 없음 (빈 .txt 파일 생성)")

print("\n--- 모든 작업이 완료되었습니다. ---")