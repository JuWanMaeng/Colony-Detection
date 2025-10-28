from ultralytics import YOLO
import cv2, os, json, glob, datetime, base64
from tqdm import tqdm

# ---------------------------------
# 설정
# ---------------------------------
model_path = "experiments/colony_2class_85_small/weights/best.pt"
img_dir = "C:/workspace/datasets/colony_177"
output_root = "C:/workspace/autolabel_output"

yolo_dir = os.path.join(output_root, "labels_yolo")
json_dir = os.path.join(output_root, "labels_json")
vis_dir = os.path.join(output_root, "visualize")
os.makedirs(yolo_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# 클래스 이름 및 색상 지정
class_names = {
    0: "COLONY",
    1: "USELESS"
}
class_colors = {
    0: (0, 255, 0),    # 초록색 (COLONY)
    1: (0, 0, 255)     # 빨강색 (USELESS)
}

# ---------------------------------
# 모델 로드
# ---------------------------------
model = YOLO(model_path)

# ---------------------------------
# 이미지 리스트
# ---------------------------------
img_list = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png"))

# ---------------------------------
# 이미지별 처리
# ---------------------------------
for img_path in tqdm(img_list, desc="Auto Labeling"):
    if img_path != 'C:/workspace/datasets/colony_177\\20251004_2025-10-02_Images_YPD.png':
        continue
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 모델 추론
    results = model(img_path, conf=0.5, iou=0.5, verbose=False,max_det=2000)[0]

    yolo_lines = []
    shapes_json = []

    # ---------------------------
    # 박스 저장 (YOLO + JSON + 시각화)
    # ---------------------------
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])

        # ① YOLO txt용 (상대좌표)
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # ② Dreamer JSON용 (절대좌표)
        shapes_json.append({
            "label": class_names.get(cls, "UNKNOWN"),
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })

        # ③ 시각화 이미지
        color = class_colors.get(cls, (255, 255, 255))
        label_text = f"{class_names.get(cls, 'UNKNOWN')} {conf:.4f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label_text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ---------------------------
    # ① YOLO txt 저장
    # ---------------------------
    with open(os.path.join(yolo_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))

    # ---------------------------
    # ② Dreamer JSON 생성 (imageData 포함)
    # ---------------------------
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 이미지 base64 인코딩
    with open(img_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    json_data = {
        "version": "1.4.1",
        "user": "Offline User",
        "time": now,
        "label_dict": {"COLONY": 0, "USELESS": 1},
        "flags": {},
        "shapes": shapes_json,
        "imagePath": f"DETECTION\\ATI\\colony_4\\{os.path.basename(img_path)}",
        "imageData": image_data,  # ✅ base64 인코딩 이미지 포함
        "multichannel": {},
        "imageHeight": h,
        "imageWidth": w
    }

    with open(os.path.join(json_dir, f"{name}.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    # ---------------------------
    # ③ 시각화 이미지 저장
    # ---------------------------
    cv2.imwrite(os.path.join(vis_dir, f"{name}_pred.png"), img)

print("\n✅ Dreamer 호환 자동 라벨링 완료!")
print(f"YOLO txt 저장 경로: {yolo_dir}")
print(f"JSON 저장 경로: {json_dir}")
print(f"시각화 이미지: {vis_dir}")
