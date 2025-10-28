import os
import shutil

# 경로 설정
test_folder = (
    r"C:\Users\jwmaeng\AppData\Local\AdvancedTechnologyInc\ATIDreamer100\data\DETECTION\ATI\colony_testdataset"
)
src_json_folder = r"C:\Users\jwmaeng\AppData\Local\AdvancedTechnologyInc\ATIDreamer100\data\DETECTION\ATI\colony_177\json\Offline User"
src_label_folder = r"C:\Users\jwmaeng\AppData\Local\AdvancedTechnologyInc\ATIDreamer100\data\DETECTION\ATI\colony_177\label\Offline User"

dst_json_folder = os.path.join(test_folder, "json", "Offline User")
dst_label_folder = os.path.join(test_folder, "label", "Offline User")

# 목적지 폴더 생성
os.makedirs(dst_json_folder, exist_ok=True)
os.makedirs(dst_label_folder, exist_ok=True)

# test 폴더 안 PNG 파일 리스트 추출
png_files = [f for f in os.listdir(test_folder) if f.lower().endswith(".png")]

print(f"총 {len(png_files)}개의 PNG 파일을 기준으로 JSON과 LABEL을 복사합니다.")

copy_count_json = 0
copy_count_label = 0

for png in png_files:
    base_name = os.path.splitext(png)[0]  # 파일 이름(확장자 제거)

    # JSON 파일 복사
    json_file = base_name + ".json"
    src_json_path = os.path.join(src_json_folder, json_file)
    dst_json_path = os.path.join(dst_json_folder, json_file)
    if os.path.exists(src_json_path):
        shutil.copy2(src_json_path, dst_json_path)
        copy_count_json += 1
    else:
        print(f"[경고] JSON 파일 없음: {json_file}")

    # LABEL 파일 복사 (일반적으로 .txt 가정)
    label_file = base_name + ".txt"
    src_label_path = os.path.join(src_label_folder, label_file)
    dst_label_path = os.path.join(dst_label_folder, label_file)
    if os.path.exists(src_label_path):
        shutil.copy2(src_label_path, dst_label_path)
        copy_count_label += 1
    else:
        print(f"[경고] LABEL 파일 없음: {label_file}")

print("\n복사 완료!")
print(f"JSON 파일 복사: {copy_count_json}개")
print(f"LABEL 파일 복사: {copy_count_label}개")
