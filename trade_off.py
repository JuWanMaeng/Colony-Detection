import os, glob, csv, time, math
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# torch가 있을 경우 CUDA 동기화로 정확한 시간 측정
try:
    import torch
    HAS_TORCH = True
except:
    HAS_TORCH = False

# === 너의 유틸을 그대로 사용 ===
from utils2 import (
    global_nms,
    load_gt_boxes,
    evaluate_f1,
    # 시각화는 벤치마크 시간에 영향 주므로 사용하지 않음
)

# ------------------------------
# 필수: 네 프로젝트의 두 추론 함수 그대로 사용
# ------------------------------
def infer_grid(model, img, rows, cols, conf, iou, max_det=3000):
    H, W = img.shape[:2]
    preds = []
    xs = [int(W * i / cols) for i in range(cols + 1)]
    ys = [int(H * j / rows) for j in range(rows + 1)]
    for r in range(rows):
        for c in range(cols):
            x1g, x2g = xs[c], xs[c+1]
            y1g, y2g = ys[r], ys[r+1]
            crop = img[y1g:y2g, x1g:x2g]
            if crop.size == 0:
                continue
            result = model.predict(crop, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
            for box in result.boxes:
                cls = int(box.cls[0]); conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                X1, Y1 = x1 + x1g, y1 + y1g
                X2, Y2 = x2 + x1g, y2 + y1g
                preds.append([cls, X1, Y1, X2, Y2, conf_score])
    return preds

def infer_full_image(model, img, cfg):
    return infer_grid(model, img, cfg["rows"], cfg["cols"], cfg["conf"], cfg["iou"], max_det=cfg.get("max_det", 3000))

def infer_multiscale(model, img, stage_configs, merge_nms_iou=0.5, max_keep=5000):
    merged = []
    for cfg in stage_configs:
        merged += infer_grid(model, img, cfg["rows"], cfg["cols"], cfg["conf"], cfg["iou"], max_det=cfg.get("max_det", 3000))
    final_preds = global_nms(merged, iou_thresh=merge_nms_iou, max_keep=max_keep)
    return final_preds

# ------------------------------
# 측정 보조: CUDA 동기화 + 타이머
# ------------------------------
def _sync():
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()

def _now():
    _sync()
    return time.perf_counter()

# ------------------------------
# 벤치마크 실행
# ------------------------------
def benchmark_tradeoff(
    model_path,
    img_dir,
    labels_dir,
    save_csv="results/tradeoff_benchmark.csv",
    eval_iou_threshold=0.5,           # 평가 기준 IoU (고정)
    warmup_images=3,                   # 워밍업 이미지 개수
    repeats=1,                         # 이미지당 반복 횟수(평균 낼 때 사용). 속도 측정 정밀도를 올리고 싶으면 2~3
    scenarios=None
):
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    # 시나리오(=설정) 목록: Full / Multi-scale 여러 조합
    # name, type, settings 로 구성
    if scenarios is None:
        scenarios = [
            # Full (baseline)
            {
                "name": "Full_1x1_conf0.5_iou0.5",
                "type": "full",
                "full_cfg": {"rows": 1, "cols": 1, "conf": 0.5, "iou": 0.5, "max_det": 3000},
            },
            # Multi-Scale 예시 (1x1, 2x2, 4x4)
            {
                "name": "MS_1-2-4_conf[0.5,0.35,0.25]_iou0.5_merge0.45",
                "type": "ms",
                "stages": [
                    {"rows": 1, "cols": 1, "conf": 0.5,  "iou": 0.5, "max_det": 3000},
                    {"rows": 2, "cols": 2, "conf": 0.5, "iou": 0.5, "max_det": 3000},
                    {"rows": 4, "cols": 4, "conf": 0.55, "iou": 0.5, "max_det": 3000},
                ],
                "merge_nms_iou": 0.1,
                "max_keep": 5000,
            },
            # 더 가벼운 멀티스케일(속도↑)
            {
                "name": "MS_1-2_conf[0.5,0.35]_iou0.5_merge0.45",
                "type": "ms",
                "stages": [
                    {"rows": 1, "cols": 1, "conf": 0.5,  "iou": 0.5, "max_det": 3000},
                    {"rows": 2, "cols": 2, "conf":0.5, "iou": 0.5, "max_det": 3000},
                ],
                "merge_nms_iou": 0.1,
                "max_keep": 5000,
            },
        ]

    # 데이터셋 로드
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_list:
        print(f"[!] 이미지가 없습니다: {img_dir}")
        return

    # 모델 로드
    model = YOLO(model_path)

    # 결과 CSV 준비
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "type", "images",
            "latency_ms_avg", "latency_ms_p50", "latency_ms_p90", "latency_ms_p95",
            "throughput_fps",
            "mean_F1", "mean_precision", "mean_recall"
        ])

        for sc in scenarios:
            latencies = []
            f1s, precisions, recalls = [], [], []

            # ---- Warm-up (모델/런타임 캐시 예열) ----
            for img_path in img_list[:min(warmup_images, len(img_list))]:
                img = cv2.imread(img_path)
                if img is None: continue
                if sc["type"] == "full":
                    _ = infer_full_image(model, img, sc["full_cfg"])
                else:
                    _ = infer_multiscale(model, img, sc["stages"], sc["merge_nms_iou"], sc["max_keep"])

            # ---- 본 측정 ----
            for img_path in tqdm(img_list, desc=f"Scenario: {sc['name']}"):
                img = cv2.imread(img_path)
                if img is None: continue
                gt_boxes = load_gt_boxes(img_path, labels_dir)

                # 반복 측정
                elapsed_sum = 0.0
                for _ in range(repeats):
                    t0 = _now()
                    if sc["type"] == "full":
                        preds = infer_full_image(model, img, sc["full_cfg"])
                    else:
                        preds = infer_multiscale(model, img, sc["stages"], sc["merge_nms_iou"], sc["max_keep"])
                    t1 = _now()
                    elapsed_sum += (t1 - t0)

                # 이미지당 평균 latency
                avg_elapsed = elapsed_sum / repeats
                latencies.append(avg_elapsed * 1000.0)  # ms

                # 평가 (IoU=0.5 고정)
                f1, p, r, _, _, _ = evaluate_f1(gt_boxes, preds, iou_threshold=eval_iou_threshold)
                f1s.append(f1); precisions.append(p); recalls.append(r)

            # 시나리오 집계
            lat_ms = np.array(latencies)
            latency_ms_avg = float(lat_ms.mean())
            latency_ms_p50 = float(np.percentile(lat_ms, 50))
            latency_ms_p90 = float(np.percentile(lat_ms, 90))
            latency_ms_p95 = float(np.percentile(lat_ms, 95))
            throughput_fps  = 1000.0 / latency_ms_avg if latency_ms_avg > 0 else 0.0

            mean_F1 = float(np.mean(f1s))
            mean_P  = float(np.mean(precisions))
            mean_R  = float(np.mean(recalls))

            writer.writerow([
                sc["name"], sc["type"], len(img_list),
                round(latency_ms_avg, 2), round(latency_ms_p50, 2), round(latency_ms_p90, 2), round(latency_ms_p95, 2),
                round(throughput_fps, 2),
                round(mean_F1, 4), round(mean_P, 4), round(mean_R, 4)
            ])

            print(f"\n[Summary] {sc['name']}")
            print(f"  Latency avg/p50/p90/p95 (ms): {latency_ms_avg:.2f} / {latency_ms_p50:.2f} / {latency_ms_p90:.2f} / {latency_ms_p95:.2f}")
            print(f"  Throughput (FPS): {throughput_fps:.2f}")
            print(f"  Mean F1 / P / R: {mean_F1:.4f} / {mean_P:.4f} / {mean_R:.4f}\n")

    print(f"\n✅ 결과 CSV 저장: {save_csv}")

    # ------------------------------
    # (선택) F1 vs Latency 산점도 그리기
    # ------------------------------
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # df = pd.read_csv(save_csv)
    # plt.figure(figsize=(7,5))
    # plt.scatter(df["latency_ms_avg"], df["mean_F1"])
    # for i,row in df.iterrows():
    #     plt.text(row["latency_ms_avg"], row["mean_F1"], row["scenario"], fontsize=8)
    # plt.xlabel("Latency (ms)")
    # plt.ylabel("Mean F1")
    # plt.title("Speed-Accuracy Trade-off")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # === 경로만 네 환경에 맞게 수정 ===
    MODEL_PATH = "experiments/colony_2class_85_small/weights/best.pt"
    IMG_DIR    = "C:/workspace/datasets/colony_2class/images/test"
    LABELS_DIR = "C:/workspace/datasets/colony_2class/images/test"

    benchmark_tradeoff(
        model_path=MODEL_PATH,
        img_dir=IMG_DIR,
        labels_dir=LABELS_DIR,
        save_csv="results/tradeoff_benchmark.csv",
        eval_iou_threshold=0.5,  # 평가 기준 IoU 고정
        warmup_images=3,
        repeats=1,
        scenarios=None  # 기본 시나리오 3개 사용 (위에서 정의)
    )
