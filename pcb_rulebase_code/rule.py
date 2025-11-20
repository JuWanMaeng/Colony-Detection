import cv2
import numpy as np

def find_central_defect(image):
    """
    이미지에서 룰베이스로 중앙 결함을 찾고 바운딩 박스를 반환합니다.
    """
    # 1. 이미지의 중앙 좌표 구하기
    (img_height, img_width) = image.shape[:2]
    center_x = img_width // 2
    center_y = img_height // 2
    
    # 2. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 (Thresholding) - 룰 적용
    # "결함은 흰색(밝고), 배경은 검정/회색(어둡다)"
    # 여기서는 200보다 밝은 값을 모두 255 (흰색)으로 만듭니다.
    # 이 임계값(200)은 실제 이미지에 맞게 조절해야 할 수 있습니다.
    thresh_value = 100
    _, binary_mask = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    
    # 4. 윤곽선(Contours) 찾기
    # binary_mask에서 흰색 영역의 외곽선을 모두 찾습니다.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    central_contour = None
    
    # 5. "결함은 항상 중앙에 있다" - 룰 적용
    for c in contours:
        # 찾은 윤곽선(c) 내부에 이미지의 중앙점(center_x, center_y)이 포함되는지 확인
        if cv2.pointPolygonTest(c, (center_x, center_y), False) >= 0:
            central_contour = c
            break # 중앙 결함을 찾았으므로 반복 중단

    # 6. 중앙 결함의 바운딩 박스(x, y, w, h) 반환
    if central_contour is not None:
        (x, y, w, h) = cv2.boundingRect(central_contour)
        return (x, y, w, h), binary_mask
    else:
        # 중앙에서 결함을 찾지 못한 경우
        return None, binary_mask

# --- 메인 코드 실행 ---

# 1. 테스트용 더미 이미지 생성
# (실제 사용 시에는 이 부분을 original_image = cv2.imread("파일경로.jpg") 로 바꾸세요)
original_image = cv2.imread(r"C:\data\product2\0858_Pad.bmp")

# 2. 룰베이스로 결함 검출
bbox, mask = find_central_defect(original_image)

# 3. 결과 시각화
if bbox:
    (x, y, w, h) = bbox
    print(f"✅ 중앙 결함 검출 성공!")
    print(f"   바운딩 박스 (x, y, w, h): {x}, {y}, {w}, {h}")
    
    # 원본 이미지에 초록색 사각형으로 검출 결과 그리기
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("❌ 중앙에서 결함을 찾지 못했습니다.")

# 4. 이미지 보여주기
cv2.imshow("Original Image with BBox", original_image)
cv2.imshow("Binary Mask (Rule Applied)", mask) # 룰이 어떻게 적용됐는지 마스크 확인

print("\n(아무 키나 누르면 창이 닫힙니다...)")
cv2.waitKey(0)
cv2.destroyAllWindows()