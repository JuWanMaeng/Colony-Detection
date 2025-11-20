import os
from ultralytics import YOLO

# =============================
# 1. 설정 (사용자 지정)
# =============================

# 💡 원본 .pt 모델 경로 (이전 코드에서 가져옴)
MODEL_PATH = "experiments/colony_2class_small_noval/weights/best.pt"

# 💡 저장할 .onnx 파일 이름
EXPORT_NAME = "colony_model_opset12.onnx"

# 💡 Opset 버전 (onnx==1.10.0은 최대 13까지 지원하나, 12가 안정적)
OPSET_VERSION = 12 

# 💡 Export 실행 장치 ('cpu' 또는 0, 1 등 GPU ID)
DEVICE = 'cpu'

# =============================
# 2. ONNX Export 실행 함수
# =============================
def export_to_onnx(pt_path, export_name, opset, device):
    """
    YOLOv8 모델(.pt)을 ONNX 파일로 변환합니다.
    """
    if not os.path.exists(pt_path):
        print(f"❌ 오류: 모델 파일을 찾을 수 없습니다: {pt_path}")
        return

    print(f"🚀 모델 로드 중: {pt_path}")
    try:
        # 1. YOLO 모델 로드
        model = YOLO(pt_path)
        
        # 2. ONNX로 Export
        print(f"🔥 ONNX Export 시작...")
        print(f"   Format: onnx")
        print(f"   Opset: {opset} (onnx==1.10.0 호환)")
        print(f"   Simplify: True")
        print(f"   Device: {device}")

        model.export(
            format="onnx",
            opset=opset,
            simplify=True,  # onnx-simplifier 필요
            device=device
        )
        
        # 3. 파일명 변경
        # (기본적으로 'best.onnx'처럼 .pt와 동일한 이름으로 생성됨)
        default_onnx_name = pt_path.replace(".pt", ".onnx")
        
        if os.path.exists(export_name):
             os.remove(export_name) # 기존 파일이 있다면 삭제

        os.rename(default_onnx_name, export_name)
        
        print(f"\n✅ ONNX Export 성공!")
        print(f"   -> 저장된 파일: {os.path.abspath(export_name)}")

    except Exception as e:
        print(f"\n❌ ONNX Export 실패:")
        print(f"   -> 오류 메시지: {e}")
        print("   -> (참고) 'ultralytics' 버전과 'onnx==1.10.0' 간의 호환성을 확인하세요.")

# =============================
# 3. 실행
# =============================
if __name__ == "__main__":
    export_to_onnx(MODEL_PATH, EXPORT_NAME, opset=OPSET_VERSION, device=DEVICE)