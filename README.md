# Obejct Detection 프로젝트

이 저장소는 Ultralytics YOLO 프레임워크를 활용한 객체 탐지 및 추론 프로젝트를 포함하고 있습니다. 
현재 구성 요소는 **Colony 프로젝트**와 **PCB 프로젝트**입니다.

## 1. Colony 프로젝트

Colony 프로젝트는 군집(Colony) 데이터의 탐지 및 분석에 중점을 둡니다.

### 주요 구성 요소
- **설정 (Configuration)**: 
  - `colony.yaml`: Colony 데이터셋/모델을 위한 설정 파일입니다.
- **추론 코드 (Inference Code)**: `colony_inference_code/`
  - 다양한 백엔드(PyTorch, ONNX)를 사용하여 추론을 실행하는 스크립트가 포함되어 있습니다.
  - 주요 스크립트:
    - `inference.py`: PyTorch 기반의 일반 추론 스크립트로 Full, Multi-Scale, Isolated 모드를 지원합니다.
    - `single_inference_onnx.py`: ONNX Runtime을 사용하여 단일 이미지에 대한 추론을 수행합니다.
    - `single_inference_pytorch.py`: PyTorch를 사용한 간소화된 단일 이미지 추론 스크립트입니다.
    - `single_inference_onnx_ms.py`: ONNX Runtime 기반으로 그리드 중복(Overlap) 및 NMS를 포함한 멀티스케일 추론을 수행합니다.
    - `inference_total_overlap_boxes.py`: Ground Truth와 비교하여 TP/FP/FN을 시각화하는 추론 스크립트입니다.
    - `inference_total_overlap_metric.py`: 모델 성능(Precision, Recall, F1, FPS)을 평가하고 파라미터 스윕(Sweep)을 수행하는 스크립트입니다.
    - `inference_total_overlap_saveimgs.py`: Full, Multi-Scale, Isolated 모드의 추론 결과를 이미지로 저장합니다.
    - `inference_total_overlap_savimgs_savetxts.py`: 추론 결과 이미지를 저장하고 YOLO 포맷의 텍스트 라벨도 함께 저장합니다.


---

## 2. PCB 프로젝트

PCB(인쇄 회로 기판) 프로젝트는 PCB 제품의 결함이나 부품을 탐지하기 위해 설계되었습니다. 
제품별로 특화된 추론 로직과 규칙 기반(Rule-based) 후처리를 포함합니다.

### 주요 구성 요소
- **설정 (Configuration)**:
  - `pcb_product1.yaml`: PCB Product 1을 위한 설정 파일입니다.
  - `pcb_product2.yaml`: PCB Product 2를 위한 설정 파일입니다.
- **추론 코드 (Inference Code)**: `pcb_inference_code/`
  - `pcb_inference_product1.py`: Product 1에 최적화된 추론 스크립트입니다.
  - `pcb_inference_product2.py`: Product 2에 최적화된 추론 스크립트입니다.
- **규칙 기반 처리 (Rule-based Processing)**: `pcb_rulebase_code/`
  - 탐지 결과의 규칙 기반 필터링 또는 후처리를 위한 로직이 포함되어 있습니다.
  - `rule.py`, `rule_all.py`: 핵심 규칙 구현 파일입니다.
  - `rule_all_vis.py`: 규칙 기반 처리 결과를 시각화하는 스크립트입니다.



