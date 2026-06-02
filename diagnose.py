# diagnose.py - 一键诊断活体模型 + 身份库 是否正常
# 用法: python diagnose.py
# 把全部输出复制发回即可定位问题。
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
import cv2

print("=" * 60)
print("环境信息")
print("=" * 60)
import sys
print("Python:", sys.version.split()[0])
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow 导入失败:", e)
print("OpenCV:", cv2.__version__)
print("FaceDetectorYN:", hasattr(cv2, "FaceDetectorYN"),
      "| FaceRecognizerSF:", hasattr(cv2, "FaceRecognizerSF"))

# ---------------- 文件检查 ----------------
print("\n" + "=" * 60)
print("文件检查")
print("=" * 60)
for f in ["face_liveness_weights.h5", "identities.npz",
          "models/face_detection_yunet_2023mar.onnx",
          "models/face_recognition_sface_2021dec.onnx"]:
    if os.path.exists(f):
        print(f"  [OK] {f}  ({os.path.getsize(f)/1e6:.1f} MB)")
    else:
        print(f"  [缺失] {f}")
# 权重文件太小 -> 多半是 Git LFS 指针没还原
if os.path.exists("face_liveness_weights.h5") and os.path.getsize("face_liveness_weights.h5") < 1e6:
    print("  [!!] face_liveness_weights.h5 太小，可能是 LFS 指针文件，不是真权重！")

# ---------------- 活体模型在固定图片上的输出 ----------------
print("\n" + "=" * 60)
print("活体模型固定图片自检 (参考值: fake≈0.89, real≈0.11)")
print("=" * 60)
from recognize import build_liveness_model, LIVENESS_WEIGHTS, IMG_SIZE
m = build_liveness_model()
m.load_weights(LIVENESS_WEIGHTS)
print("[*] 权重加载完成")

ref_imgs = sorted(glob.glob("fengyizhuo/fengyizhuo/*.jpg"))[:3]
for p in ref_imgs:
    img = cv2.imread(p)
    if img is None:
        continue
    roi = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    out = m.predict(np.expand_dims(roi, 0), verbose=0)[0]
    print(f"  {os.path.basename(p):42s} fake={out[0]:.3f} real={out[1]:.3f}")
print(">> 若这里的 fake/real 和参考值相近 -> 模型加载正常(问题在摄像头)")
print(">> 若数字明显不同(如 real 全是 0 或 nan) -> 模型/版本加载有问题")

# ---------------- 身份库自检 ----------------
print("\n" + "=" * 60)
print("身份库自检 (用本人照片应能匹配到 fengyizhuo, 相似度>0.4)")
print("=" * 60)
from face_id import FaceID
fid = FaceID()
print("身份库包含:", list(fid.db.keys()))
for p in ref_imgs:
    img = cv2.imread(p)
    feat = fid.embedding_from_image(img)
    if feat is None:
        print(f"  {os.path.basename(p):42s} 未检测到人脸")
        continue
    name, score = fid.identify(feat)
    print(f"  {os.path.basename(p):42s} -> {name}  (相似度={score:.3f})")

# ---------------- 摄像头实拍 ----------------
print("\n" + "=" * 60)
print("摄像头实拍 (真人请正对镜头，采 10 帧)")
print("=" * 60)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  [!] 摄像头打开失败")
else:
    got = 0
    tries = 0
    while got < 10 and tries < 60:
        tries += 1
        ret, frame = cap.read()
        if not ret:
            continue
        faces = fid.detect(frame)
        if faces is None or len(faces) == 0:
            continue
        face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        x, y, fw, fh = face[:4].astype(int)
        x, y = max(0, x), max(0, y)
        crop = frame[y:y+fh, x:x+fw]
        if crop.size == 0:
            continue
        roi = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype("float32")/255.0
        out = m.predict(np.expand_dims(roi, 0), verbose=0)[0]
        feat = fid.embedding(frame, face)
        name, score = fid.identify(feat)
        got += 1
        print(f"  帧{got}: 活体 fake={out[0]:.3f} real={out[1]:.3f} | 身份={name} 相似度={score:.3f}")
    if got == 0:
        print("  [!] 10 帧内没检测到人脸，请靠近镜头/改善光线")
cap.release()
print("\n完成。请把以上全部输出复制发回。")
