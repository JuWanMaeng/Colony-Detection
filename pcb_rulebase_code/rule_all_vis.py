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
        top_left_x = x
        top_left_y = y
        top_right_x = x + w
        bottom_y = y + h # (top_right_y -> bottom_y 로 명확화)

        # 룰 적용
        if (top_left_x < x_min_safe) or \
           (top_right_x > x_max_safe) or \
           (top_left_y < y_min_safe) or \
           (bottom_y > y_max_safe): 
            continue 

        # 7-5. 모든 필터를 통과한 경우에만 바운딩 박스 추가
        bboxes.append((x, y, w, h))

    # 8. ★ 수정 ★: 모든 바운딩 박스 리스트와 시각화용 마스크 반환
    return bboxes, (img_height, img_width), binary_mask

# --- 메인 코드 실행 (시각화 모드) ---

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
# ----------------------------------------------------


# 모든 이미지 파일 경로 찾기
image_files = []
for ext in IMAGE_EXTENSIONS:
    image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

# 하이퍼파라미터 값 출력
print(f"--- 총 {len(image_files)}개의 이미지 시각화를 시작합니다. ---")
print(f"    [현재 설정]")
print(f"    - Brightness: {PARAMS['MIN_BRIGHTNESS']}~{PARAMS['MAX_BRIGHTNESS']}")
print(f"    - Area: {PARAMS['MIN_AREA']}~{PARAMS['MAX_AREA']}")
print(f"    - Edge Margin: {PARAMS['EDGE_MARGIN_RATIO'] * 100:.0f}% (최종 복합 룰 기준)")
print("    [조작법]")
print("    - 아무 키 (Space, Enter 등): 다음 이미지")
print("    - q 또는 Esc: 프로그램 종료")
print("--------------------------------------------------\n")

for img_path in image_files:
    # 1. 이미지 읽기
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"⚠️ {img_path} 파일을 읽을 수 없습니다.")
        continue
        
    # 2. ★ 수정 ★: 룰베이스로 "모든" 결함 검출 (binary_mask도 반환받음)
    bboxes, (img_height, img_width), binary_mask = find_all_defects(
        original_image, 
        PARAMS 
    )
    
    base_filename = os.path.basename(img_path)
    
    # 3. ★ 추가 ★: 시각화용 이미지(복사본)에 결과 그리기
    viz_image = original_image.copy()
    
    if bboxes:
        # 모든 검출된 bbox에 대해 초록색 사각형 그리기
        for (x, y, w, h) in bboxes:
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        print(f"✅ [ {base_filename} ] : {len(bboxes)}개 결함 검출")
    else:
        # 결함을 찾지 못한 경우
        print(f"❌ [ {base_filename} ] : 검출된 결함 없음")

    # 4. ★ 추가 ★: 이미지 보여주기
    cv2.imshow(f"Result: {base_filename}", viz_image)
    cv2.imshow("Binary Mask (Check Parameters)", binary_mask)

    # 5. ★ 추가 ★: 키 입력 대기
    key = cv2.waitKey(0)

    # 6. 'q' 또는 'Esc' (ASCII 27) 키를 누르면 루프 종료
    if key == ord('q') or key == 27:
        print("\n--- 시각화를 종료합니다. ---")
        break

# 7. ★ 추가 ★: 모든 작업 완료 후 창 닫기
cv2.destroyAllWindows()