import cv2
import numpy as np
import os
import glob

def find_all_defects(image, thresh_value, min_area=10):
    """
    이미지에서 룰베이스로 "모든" 결함을 찾고 바운딩 박스 "리스트"를 반환합니다.
    "중앙" 룰을 제거하고 "최소 면적" 룰을 추가합니다.
    """
    # 1. 이미지의 높이, 너비 저장 (YOLO 변환 시 필요)
    (img_height, img_width) = image.shape[:2]
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 (Thresholding)
    _, binary_mask = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    
    # (선택 사항) 노이즈 제거가 필요하면 이 부분의 주석을 해제하세요.
    # kernel = np.ones((3, 3), np.uint8) # 커널 크기를 3x3 정도로 작게
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    # binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # 4. 윤곽선(Contours) 찾기
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = [] # 5. 모든 결함의 bbox를 저장할 리스트
    
    # 6. ★ 룰 수정 ★: "중앙" 대신 "모든 윤곽선"을 검사
    for c in contours:
        # 6-1. ★ 룰 추가 ★: 최소 면적 필터링 (노이즈 제거)
        # 윤곽선의 면적이 min_area보다 작으면 무시
        if cv2.contourArea(c) < min_area:
            continue
            
        # 6-2. 바운딩 박스 계산 및 리스트에 추가
        (x, y, w, h) = cv2.boundingRect(c)
        bboxes.append((x, y, w, h))

    # 7. 모든 바운딩 박스 리스트 반환
    return bboxes, (img_height, img_width)

# --- 메인 코드 실행 ---

# ----------------------------------------------------
# 📌 (필수) 여기를 수정하세요
# ----------------------------------------------------
# 1. 이전에 찾은 최적의 임계값
YOUR_THRESH_VALUE = 180  # 예: 180 (직접 찾은 값으로 변경)

# 2. (★추가★) 최소 결함 크기 (픽셀 단위 면적)
#    - 너무 작은 노이즈가 잡히지 않도록 조절 (예: 5, 10, 20)
MIN_DEFECT_AREA = 10 

# 3. 원본 이미지가 있는 폴더 경로
INPUT_DIR = r"C:\data\product2" 

# 4. 라벨(.txt) 파일을 저장할 폴더 경로
OUTPUT_DIR = r"C:\data\product2_labels" 

# 5. 결함의 클래스 ID
CLASS_ID = 0

# 6. 찾을 이미지 확장자
IMAGE_EXTENSIONS = ["*.bmp", "*.jpg", "*.png", "*.jpeg"]
# ----------------------------------------------------

# 라벨 저장 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모든 이미지 파일 경로 찾기
image_files = []
for ext in IMAGE_EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

print(f"총 {len(image_files)}개의 이미지를 처리합니다.")

for img_path in image_files:
    # 1. 이미지 읽기
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"⚠️ {img_path} 파일을 읽을 수 없습니다.")
        continue
        
    # 2. 룰베이스로 "모든" 결함 검출
    # bboxes는 [(x1,y1,w1,h1), (x2,y2,w2,h2), ...] 형태의 리스트
    bboxes, (img_height, img_width) = find_all_defects(original_image, YOUR_THRESH_VALUE, MIN_DEFECT_AREA)
    
    base_filename = os.path.basename(img_path)

    # 3. ★ 로직 수정 ★: 결함이 "하나라도" 검출된 경우
    if bboxes: # bboxes 리스트가 비어있지 않다면
        
        yolo_lines = [] # .txt 파일에 쓸 모든 라인을 저장할 리스트
        
        # 4. ★ 로직 추가 ★: 모든 bbox에 대해 반복
        for (x, y, w, h) in bboxes:
            
            # 5. YOLO 형식으로 변환
            x_center_rel = (x + w / 2) / img_width
            y_center_rel = (y + h / 2) / img_height
            width_rel = w / img_width
            height_rel = h / img_height
            
            # 6. YOLO 포맷 문자열 생성
            yolo_string = f"{CLASS_ID} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}\n"
            yolo_lines.append(yolo_string)
        
        # 7. TXT 파일로 "한 번에" 저장 (모든 라인을 쓴다)
        txt_filename = os.path.splitext(base_filename)[0] + ".txt"
        output_txt_path = os.path.join(OUTPUT_DIR, txt_filename)
        
        with open(output_txt_path, 'w') as f:
            f.writelines(yolo_lines)
            
        print(f"✅ [검출 성공] {base_filename} -> {len(bboxes)}개 결함 저장")

    else:
        # 결함을 찾지 못한 경우 (또는 모두 노이즈로 필터링 된 경우)
        print(f"❌ [검출 실패] {base_filename} 에서 결함을 찾지 못했습니다.")

print("\n--- 모든 작업이 완료되었습니다. ---")