import os
import json
import cv2

# 경로 설정
json_dir = "C:/Users/jwmaeng/AppData/Local/AdvancedTechnologyInc/ATIDreamer100/data/DETECTION/ATI/colony_177/json/Offline User"
label_dir = "C:/Users/jwmaeng/AppData/Local/AdvancedTechnologyInc/ATIDreamer100/data/DETECTION/ATI/colony_177/label/Offline User"  # YOLO txt 파일 경로
image_dir = "C:/workspace/datasets/colony_177"  # 원본 이미지 폴더 경로

# 클래스 이름 매핑 (JSON 구조에 맞게)
class_names = {0: "COLONY", 1: "USELESS"}

# 전체 파일 반복
for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    name = os.path.splitext(json_file)[0]
    json_path = os.path.join(json_dir, json_file)
    label_path = os.path.join(label_dir, name + ".txt")

    # txt 라벨이 없으면 스킵
    if not os.path.exists(label_path):
        print(f"[경고] {label_path} 없음 → 건너뜀")
        continue

    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 크기 읽기 (JSON에서 읽거나 직접 이미지 열기)
    img_path_png = os.path.join(image_dir, name + ".png")
    img_path_jpg = os.path.join(image_dir, name + ".jpg")
    if os.path.exists(img_path_png):
        img = cv2.imread(img_path_png)
    elif os.path.exists(img_path_jpg):
        img = cv2.imread(img_path_jpg)
    else:
        print(f"[경고] {name} 이미지 파일 없음 → 건너뜀")
        continue

    h, w = img.shape[:2]

    # shapes 새로 생성
    new_shapes = []
    with open(label_path, "r") as lf:
        for line in lf.readlines():
            cls, xc, yc, bw, bh = map(float, line.strip().split())
            cls = int(cls)
            # YOLO 상대좌표 → 절대좌표 변환
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            # JSON 구조에 맞게 append
            new_shapes.append({
                "label": class_names.get(cls, "UNKNOWN"),
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

    # JSON의 shapes 갱신
    data["shapes"] = new_shapes
    data["imageHeight"] = h
    data["imageWidth"] = w

    # JSON 다시 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[완료] {json_file} 업데이트됨 (shape={len(new_shapes)}개)")

print("\n✅ 모든 JSON 파일을 label txt 기준으로 복구 완료!")
