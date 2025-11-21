import os
import json
import base64
from glob import glob
from datetime import datetime

# --- 설정 변수 ---
# 1. YOLO .txt 파일이 있는 디렉토리 (입력)
YOLO_LABELS_DIR = r"C:\Users\jwmaeng\AppData\Local\AdvancedTechnologyInc\ATIDreamer100\data\DETECTION\ATI\PCB_product1\label\Offline User"

# 2. 변환된 .json 파일을 저장할 디렉토리 (출력)
# \label\ 대신 \json\ 폴더에 저장되도록 자동 설정
JSON_OUTPUT_DIR = YOLO_LABELS_DIR.replace("\\label\\", "\\json\\") 

# 3. 이미지 파일(*.bmp)이 실제로 저장된 절대 경로 (JSON imagePath 계산에 사용)
IMAGE_BASE_DIR = r"C:\Users\jwmaeng\AppData\Local\AdvancedTechnologyInc\ATIDreamer100\data\DETECTION\ATI\PCB_product1"

# 4. JSON imagePath에 들어갈 상대 경로 (Windows 경로 구분자를 /로 바꿔서 저장)
IMAGE_PATH_RELATIVE_PREFIX = r"DETECTION\ATI\PCB_product1"

# 5. 이미지 크기 (YOLO 정규화 좌표 변환에 사용)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320

# 6. YOLO 클래스 ID (0, 1, 2)를 JSON 레이블로 매핑 (***수정된 부분***)
CLASS_MAPPING = {
    0: "bridge",
    1: "point",
    2: "black"
}

# --- 필수 함수: Base64 인코딩 ---

def get_base64_data(image_full_path):
    """
    이미지 파일을 바이너리로 읽어 Base64 인코딩 문자열을 반환합니다.
    """
    try:
        # 'rb' (read binary) 모드로 파일 열기
        with open(image_full_path, "rb") as f:
            encoded_bytes = base64.b64encode(f.read())
            # 디코딩하여 JSON에 넣을 수 있는 문자열 형태로 반환
            return encoded_bytes.decode('utf-8')
    except Exception as e:
        # FileNotFoundError를 포함한 모든 예외를 처리하고 None 반환
        print(f"⚠️ 경고: 이미지 파일 {image_full_path}을 읽거나 Base64 인코딩 중 오류 발생: {e}. imageData를 None으로 설정합니다.")
        return None

# --- 좌표 변환 함수 ---

def yolo_to_labelme_points(center_x_norm, center_y_norm, width_norm, height_norm, img_w, img_h):
    """
    YOLO 정규화된 바운딩 박스 좌표를 LabelMe 픽셀 좌표로 변환합니다.
    """
    center_x = center_x_norm * img_w
    center_y = center_y_norm * img_h
    width = width_norm * img_w
    height = height_norm * img_h

    # 좌측 상단 (x_min, y_min) 및 우측 하단 (x_max, y_max) 픽셀 좌표 계산
    x_min = round(center_x - width / 2.0, 6)
    y_min = round(center_y - height / 2.0, 6)
    x_max = round(center_x + width / 2.0, 6)
    y_max = round(center_y + height / 2.0, 6)

    return [
        [x_min, y_min],
        [x_max, y_max]
    ]

# --- JSON 생성 함수 ---

def create_labelme_json(txt_path, image_base_dir, image_path_relative_prefix, image_w, image_h, class_map):
    """
    하나의 YOLO .txt 파일을 읽고 모든 필드를 채운 LabelMe JSON 딕셔너리를 생성합니다.
    """
    
    # 1. 파일 이름 및 경로 설정
    file_name = os.path.basename(txt_path)
    base_name = os.path.splitext(file_name)[0]
    
    # JSON 저장 경로 계산
    json_path = os.path.join(JSON_OUTPUT_DIR, base_name + ".json")

    # 이미지 파일 이름 및 경로 계산 (예: 0000_Pad_sample.txt -> 0000_Pad.bmp)
    image_file_name = base_name.replace("_sample", "") + ".bmp" 
    image_full_path = os.path.join(image_base_dir, image_file_name)
    
    # JSON에 들어갈 상대 경로
    image_path_relative = os.path.join(image_path_relative_prefix, image_file_name) 

    # 1-1. 원본 이미지 파일 존재 여부 확인 후 없으면 즉시 건너뛰기
    if not os.path.exists(image_full_path):
        print(f"⚠️ 경고: 대응되는 이미지 파일 {image_full_path}을 찾을 수 없습니다. JSON 생성을 건너뜁니다.")
        return None, None
    
    # 2. 이미지 데이터 (imageData) Base64 인코딩
    image_data_encoded = get_base64_data(image_full_path)
    
    # 3. YOLO 내용 파싱 및 shapes 리스트 생성
    shapes_list = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {txt_path}, 오류: {e}")
        return None, None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:])
        except ValueError:
            continue
            
        # CLASS_MAPPING을 사용하여 ID를 레이블 이름으로 변환
        label_name = class_map.get(class_id)
        if label_name is None:
            # 매핑에 없는 클래스 ID는 건너뜁니다.
            print(f"   ℹ️ 정보: 알 수 없는 클래스 ID {class_id}를 건너뜁니다.")
            continue
            
        labelme_points = yolo_to_labelme_points(center_x, center_y, width, height, image_w, image_h)

        shape_data = {
            "label": label_name,
            "points": labelme_points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes_list.append(shape_data)
        
    # 4. 최종 LabelMe JSON 구조 생성
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_data = {
        "version": "1.4.1",
        "user": "Offline User",
        "time": now,
        # class_map을 뒤집어서 {라벨명: ID} 형태로 저장
        "label_dict": {
            v: k for k, v in class_map.items() 
        },
        "flags": {},
        "shapes": shapes_list,
        # imagePath는 슬래시(/)로 통일하여 저장
        "imagePath": image_path_relative.replace("\\", "/"), 
        "imageData": image_data_encoded,
        "multichannel": {},
        "imageHeight": image_h,
        "imageWidth": image_w
    }
    
    return json_data, json_path


# --- 메인 실행 ---
if __name__ == "__main__":
    
    # JSON 출력 디렉토리 생성
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    
    # YOLO .txt 파일 목록 가져오기 (라벨 파일 기준으로만 작업)
    yolo_files = glob(os.path.join(YOLO_LABELS_DIR, "*.txt"))
    
    print(f"총 {len(yolo_files)}개의 YOLO 라벨 파일을 찾았습니다.")

    for txt_file in yolo_files:
        print(f"변환 시작: {os.path.basename(txt_file)}...")
        
        json_data, json_path = create_labelme_json(
            txt_file, 
            IMAGE_BASE_DIR, 
            IMAGE_PATH_RELATIVE_PREFIX, 
            IMAGE_WIDTH, 
            IMAGE_HEIGHT, 
            CLASS_MAPPING
        )

        if json_data:
            try:
                # JSON 파일 저장 (UTF-8 인코딩, 보기 좋게 4칸 들여쓰기)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)
                print(f"✅ 성공: {os.path.basename(json_path)}")
            except Exception as e:
                print(f"❌ JSON 파일 저장 중 오류 발생: {json_path}, 오류: {e}")
        else:
             # create_labelme_json 함수에서 None을 반환한 경우 (대응 이미지 없음)
             print(f"❌ 실패: {os.path.basename(txt_file)}의 변환을 건너뛰었습니다.")


    print("\n🎉 모든 파일 변환 완료.")