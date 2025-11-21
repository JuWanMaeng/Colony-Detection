import os
import cv2
import glob
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] unnormalized
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)  # bottom right y
    return y

def compute_iou(box1, box2):
    # box1: [x1, y1, x2, y2]
    # box2: [x1, y1, x2, y2]
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def main():
    # Paths
    model_path = r'experiments\PCB_product1_small\weights\best.pt'
    output_dir = r'experiments\PCB_product1_small\results'
    images_dir = r'C:\data\product1_yolo\test\images'
    labels_dir = r'C:\data\product1_yolo\test\labels'
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get images
    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(images_dir, '*.png')) + \
                  glob.glob(os.path.join(images_dir, '*.bmp'))
    
    print(f"Found {len(image_files)} images.")
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    conf_threshold = 0.5
    iou_threshold = 0.5 # Standard IoU threshold for TP
    
    # Class names
    class_names = {0: 'bridge', 1: 'point', 2: 'black'}

    # Per-class metrics
    class_metrics = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in class_names.keys()}

    for img_path in image_files:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
        
        # Load image
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Load GT labels
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    # x_center, y_center, width, height (normalized)
                    box = [float(x) for x in parts[1:]]
                    # Convert to xyxy
                    xyxy = xywhn2xyxy(np.array(box), w=w, h=h)
                    gt_boxes.append((xyxy, cls))
        
        # Run inference
        results = model.predict(img, conf=conf_threshold, verbose=False)
        pred_boxes = results[0].boxes.xyxy.cpu().numpy()
        pred_cls = results[0].boxes.cls.cpu().numpy()
        
        # --- Global Matching (Class Agnostic) ---
        matched_gt = [False] * len(gt_boxes)
        current_tp = 0
        current_fp = 0
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, (gt_box, gt_cls) in enumerate(gt_boxes):
                if not matched_gt[i]:
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
            
            if best_iou >= iou_threshold:
                current_tp += 1
                matched_gt[best_gt_idx] = True
            else:
                current_fp += 1
        
        current_fn = len(gt_boxes) - sum(matched_gt)
        
        tp_total += current_tp
        fp_total += current_fp
        fn_total += current_fn

        # --- Per-Class Matching ---
        for cls_id in class_names.keys():
            # Filter GT and Pred for this class
            cls_gt_boxes = [box for box, c in gt_boxes if c == cls_id]
            cls_pred_boxes = [box for i, box in enumerate(pred_boxes) if int(pred_cls[i]) == cls_id]
            
            cls_matched_gt = [False] * len(cls_gt_boxes)
            cls_tp = 0
            cls_fp = 0
            
            for p_box in cls_pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for i, g_box in enumerate(cls_gt_boxes):
                    if not cls_matched_gt[i]:
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    cls_tp += 1
                    cls_matched_gt[best_gt_idx] = True
                else:
                    cls_fp += 1
            
            cls_fn = len(cls_gt_boxes) - sum(cls_matched_gt)
            
            class_metrics[cls_id]['tp'] += cls_tp
            class_metrics[cls_id]['fp'] += cls_fp
            class_metrics[cls_id]['fn'] += cls_fn
        
        # Visualization
        # Draw GT in Green (With Label)
        for box, cls in gt_boxes:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, 'GT', (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # Draw Pred in Red (or Blue) with Label
        for i, box in enumerate(pred_boxes):
            cls_idx = int(pred_cls[i])
            label_text = class_names.get(cls_idx, f'Class {cls_idx}')
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            cv2.putText(img, label_text, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
        # Save result
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img)
        
    # Calculate Global F1
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"Total Images: {len(image_files)}")
    print(f"Global TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")
    print(f"Global Precision: {precision:.4f}")
    print(f"Global Recall: {recall:.4f}")
    print(f"Global F1 Score (conf={conf_threshold}): {f1:.4f}")
    print("-" * 30)
    
    # Calculate Per-Class F1
    print("Per-Class Metrics:")
    for cls_id, name in class_names.items():
        m = class_metrics[cls_id]
        p = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        r = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        print(f"  {name}: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, P={p:.4f}, R={r:.4f}, F1={f:.4f}")
    
    print("-" * 30)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
