import glob
import time

import cv2
import torch

from ultralytics import YOLO

# =============================
# 1️⃣ 모델 로드
# =============================
MODEL_PATH = "yolo12s.pt"  # 또는 best.pt
model = YOLO(MODEL_PATH)
device = model.device
print(f"✅ Model loaded on device: {device}")

# =============================
# 2️⃣ 테스트 이미지 로드
# =============================
IMG_DIR = "C:/workspace/datasets/colony_2class/images/test"
img_paths = sorted(glob.glob(f"{IMG_DIR}/*.png"))[:16]
print(f"🔍 Loaded {len(img_paths)} images")

# =============================
# 3️⃣ Batch 텐서 준비
# =============================
imgs = []
for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.resize(img, (640, 640))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    imgs.append(tensor)

batch_tensor = torch.stack(imgs).to(device)  # (B, 3, 640, 640)
print(f"📦 Batch tensor shape: {batch_tensor.shape}")

# ==========================================================
# 4️⃣ SINGLE INFERENCE (한 장씩 순차 실행)
# ==========================================================
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    for img_tensor in imgs:
        input_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 640, 640)
        _ = model.model(input_tensor)

torch.cuda.synchronize()
elapsed_single = time.time() - start
fps_single = len(imgs) / elapsed_single

print("\n🧩 Single Inference")
print(f"⏱️ Elapsed: {elapsed_single * 1000:.2f} ms | FPS: {fps_single:.2f}")

# ==========================================================
# 5️⃣ BATCH INFERENCE (16장 한 번에 실행)
# ==========================================================
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    _ = model.model(batch_tensor)

torch.cuda.synchronize()
elapsed_batch = time.time() - start
fps_batch = len(imgs) / elapsed_batch

print("\n🧩 Batch Inference")
print(f"⏱️ Elapsed: {elapsed_batch * 1000:.2f} ms | FPS: {fps_batch:.2f}")

# ==========================================================
# 6️⃣ 비교 요약
# ==========================================================
speedup = elapsed_single / elapsed_batch
print("\n🚀 성능 비교 요약")
print(f"Single: {elapsed_single * 1000:.2f} ms ({fps_single:.2f} FPS)")
print(f"Batch : {elapsed_batch * 1000:.2f} ms ({fps_batch:.2f} FPS)")
print(f"⚡ Speed-up (Batch vs Single): {speedup:.2f}x")
