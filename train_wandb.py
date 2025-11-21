import wandb
from ultralytics import YOLO
import os
import glob
import cv2

# -------------------------------
# 1) Training metrics logging
# -------------------------------
def log_train_metrics(trainer):
    log_data = {
        "epoch": trainer.epoch,
        "train/box_loss": trainer.loss_items[0].item(),
        "train/cls_loss": trainer.loss_items[1].item(),
        "train/dfl_loss": trainer.loss_items[2].item(),
    }
    if hasattr(trainer, "metrics") and trainer.metrics:
        for k, v in trainer.metrics.items():
            log_data[f"val/{k}"] = v
    wandb.log(log_data, step=trainer.epoch)

# -------------------------------
# 2) IoU & Metrics
# -------------------------------
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    xa, ya, xb, yb = box2
    inter_x1, inter_y1 = max(x1, xa), max(y1, ya)
    inter_x2, inter_y2 = min(x2, xb), min(y2, yb)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xb - xa) * (yb - ya)
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def compute_metrics(pred_boxes, gt_boxes, iou_thresh=0.5):
    TP, FP, FN = 0, 0, 0
    used = [False] * len(gt_boxes)

    for pb in pred_boxes:
        matched = False
        for i, gb in enumerate(gt_boxes):
            if not used[i] and iou(pb, gb) >= iou_thresh:
                TP += 1
                used[i] = True
                matched = True
                break
        if not matched:
            FP += 1

    FN = used.count(False)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-6) if (precision + recall) > 0 else 0
    return TP, FP, FN, precision, recall, f1

# -------------------------------
# 3) 학습 끝난 후 inference & wandb 업로드 (metrics만)
# -------------------------------
def run_inference_and_log(best_model_path, test_images):
    model = YOLO(best_model_path)

    total_TP, total_FP, total_FN = 0, 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    valid_count = 0

    for img_path in test_images:
        results = model(img_path, conf=0.25)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # GT 박스
        gt_boxes = []
        label_path = img_path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    gt_boxes.append([x1, y1, x2, y2])

        # Pred 박스
        pred_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pred_boxes.append([x1, y1, x2, y2])

        # Metrics 계산
        TP, FP, FN, precision, recall, f1 = compute_metrics(pred_boxes, gt_boxes, iou_thresh=0.5)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        valid_count += 1

    # ✅ 전체 평균 계산
    avg_precision = total_precision / valid_count if valid_count > 0 else 0
    avg_recall = total_recall / valid_count if valid_count > 0 else 0
    avg_f1 = total_f1 / valid_count if valid_count > 0 else 0

    print("\n📊 Test Dataset Summary")
    print(f"Images evaluated: {valid_count}")
    print(f"Avg Precision={avg_precision:.4f}, Avg Recall={avg_recall:.4f}, Avg F1={avg_f1:.4f}")
    print(f"Total TP={total_TP}, FP={total_FP}, FN={total_FN}")

    # ✅ wandb에 한 번만 로깅
    wandb.log({
        "test/mean/F1": avg_f1
    }, commit=True)


# -------------------------------
# 4) Main training function
# -------------------------------
def main():
    exp_name = 'PCB_product1_small'
    wandb.init(project="PCB", name=f"{exp_name}")
    
    model = YOLO("yolo12s.pt")

    # 콜백 연결
    model.add_callback("on_train_epoch_end", log_train_metrics)

    # 학습 실행
    results = model.train(
        data="C:/data/product1_yolo/data.yaml",
        epochs=500,
        imgsz=640,
        cfg=f"cfgs/train_colony.yaml",
        project="experiments",
        name=f"{exp_name}",
        plots=True,
    )

    # 학습 완료 후 best.pt로 inference
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    val_img_dir = f"C:/data/product1_yolo/test/images"
    test_images = glob.glob(os.path.join(val_img_dir, "*.jpg")) + \
                  glob.glob(os.path.join(val_img_dir, "*.png"))

    run_inference_and_log(best_model_path, test_images)

if __name__ == "__main__":
    main()
