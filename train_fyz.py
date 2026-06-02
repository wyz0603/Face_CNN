# train_fyz.py - 用 fyz/ 数据集(real/fake, 整帧)微调活体模型
# 关键：用 YuNet 把每张图裁成人脸(+margin)，与 recognize.py 推理时的输入完全一致。
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import shutil
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from recognize import build_liveness_model, IMG_SIZE, LIVENESS_WEIGHTS, FACE_MARGIN
from face_id import FaceID

# real -> 1, fake -> 0（与原 train.py 一致：spoof/fake=0, real=1）
DATA = {1: "fyz/real", 0: "fyz/fake"}
VIDEO_STRIDE = 5
EPOCHS = 40
BATCH = 16

_fid = FaceID(db_path=None)


def crop_face(frame):
    """YuNet 取最大人脸 + margin，裁剪并 resize 到 224。无脸返回 None。"""
    faces = _fid.detect(frame)
    if faces is None or len(faces) == 0:
        return None
    face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    h, w = frame.shape[:2]
    x, y, fw, fh = face[:4].astype(int)
    x, y = max(0, x), max(0, y)
    mx, my = int(fw * FACE_MARGIN), int(fh * FACE_MARGIN)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0


def load(directory, label):
    X = []
    img_paths = [p for p in glob.glob(os.path.join(directory, "*"))
                 if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    vid_paths = [p for p in glob.glob(os.path.join(directory, "*"))
                 if p.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    miss = 0
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        c = crop_face(img)
        if c is not None:
            X.append(c)
        else:
            miss += 1
    for p in vid_paths:
        cap = cv2.VideoCapture(p)
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % VIDEO_STRIDE == 0:
                c = crop_face(frame)
                if c is not None:
                    X.append(c)
            i += 1
        cap.release()
    print(f"[INFO] {directory}: 裁出人脸 {len(X)} 张 (图片漏检 {miss})")
    return X, [label] * len(X)


def main():
    X, y = [], []
    for label, d in DATA.items():
        Xi, yi = load(d, label)
        X += Xi
        y += yi
    X = np.array(X, dtype="float32")
    y = np.array(y)
    print(f"[INFO] 合计 {len(X)} 张: real={int(np.sum(y==1))}, fake={int(np.sum(y==0))}")

    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    print("[INFO] 预计算 VGG16 特征中(含水平翻转增强)...")
    f1 = base.predict(X, batch_size=BATCH, verbose=1)
    f2 = base.predict(X[:, :, ::-1, :], batch_size=BATCH, verbose=1)
    feats = np.concatenate([f1, f2], 0)
    labels = np.concatenate([y, y], 0)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(feats))
    feats, labels = feats[perm], labels[perm]
    n_val = max(1, int(0.2 * len(feats)))
    Xv, yv = feats[:n_val], labels[:n_val]
    Xt, yt = feats[n_val:], labels[n_val:]

    cw = {0: len(labels) / (2 * np.sum(labels == 0)),
          1: len(labels) / (2 * np.sum(labels == 1))}

    inp = Input(shape=feats.shape[1:])
    x = Flatten()(inp)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(2, activation="softmax")(x)
    head = Model(inp, out)
    head.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    head.fit(Xt, to_categorical(yt, 2), validation_data=(Xv, to_categorical(yv, 2)),
             epochs=EPOCHS, batch_size=BATCH, class_weight=cw, verbose=2)

    # 验证集详细指标
    pv = head.predict(Xv, verbose=0).argmax(1)
    acc = float(np.mean(pv == yv))
    real_recall = float(np.mean(pv[yv == 1] == 1)) if np.any(yv == 1) else 0
    fake_recall = float(np.mean(pv[yv == 0] == 0)) if np.any(yv == 0) else 0
    print(f"[INFO] 验证 accuracy={acc:.3f}  真人召回={real_recall:.3f}  假脸召回={fake_recall:.3f}")

    full = build_liveness_model()
    dense_full = [l for l in full.layers if isinstance(l, Dense)]
    dense_head = [l for l in head.layers if isinstance(l, Dense)]
    for df, dh in zip(dense_full, dense_head):
        df.set_weights(dh.get_weights())

    if os.path.exists(LIVENESS_WEIGHTS):
        shutil.copy(LIVENESS_WEIGHTS, LIVENESS_WEIGHTS + ".orig.bak")
        print(f"[INFO] 原权重已备份为 {LIVENESS_WEIGHTS}.orig.bak")
    full.save_weights(LIVENESS_WEIGHTS)
    print(f"[INFO] 新权重已保存到 {LIVENESS_WEIGHTS}")


if __name__ == "__main__":
    main()
