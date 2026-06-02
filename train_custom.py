# train_custom.py - 用自采样本(custom_data)微调活体模型，适配你的摄像头
# 用法: python train_custom.py
# 训练完会备份旧权重为 face_liveness_weights.h5.bak，并写出新的 face_liveness_weights.h5
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
from tensorflow.keras.layers import Dense as _Dense

from recognize import build_liveness_model, IMG_SIZE, LIVENESS_WEIGHTS

REAL_DIR = "custom_data/real"
SPOOF_DIR = "custom_data/spoof"
EPOCHS = 40
BATCH = 16


def load_images(directory, label):
    X, y = [], []
    for p in glob.glob(os.path.join(directory, "*")):
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        X.append(img)
        y.append(label)
    return X, y


def main():
    # spoof=0, real=1（与原 train.py 一致）
    Xs, ys = load_images(SPOOF_DIR, 0)
    Xr, yr = load_images(REAL_DIR, 1)
    print(f"[INFO] 真人(real)样本: {len(Xr)} 张, 翻拍(spoof)样本: {len(Xs)} 张")
    if len(Xr) < 30 or len(Xs) < 30:
        raise RuntimeError("样本太少，请先用 collect_samples.py 采集 real 和 spoof 各 300+ 张。")

    X = np.array(Xs + Xr, dtype="float32")
    y = np.array(ys + yr)

    # ---- 用冻结的 VGG16 base 预计算特征（含水平翻转增强），CPU 也很快 ----
    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    print("[INFO] 预计算 VGG16 特征中...")
    feat = base.predict(X, batch_size=BATCH, verbose=1)
    feat_flip = base.predict(X[:, :, ::-1, :], batch_size=BATCH, verbose=1)  # 水平翻转增强
    feats = np.concatenate([feat, feat_flip], axis=0)
    labels = np.concatenate([y, y], axis=0)

    # 打乱并切分
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(feats))
    feats, labels = feats[perm], labels[perm]
    n_val = max(1, int(0.2 * len(feats)))
    Xv, yv = feats[:n_val], labels[:n_val]
    Xt, yt = feats[n_val:], labels[n_val:]
    yt_cat, yv_cat = to_categorical(yt, 2), to_categorical(yv, 2)

    # 类别权重
    cw = {0: len(labels) / (2 * np.sum(labels == 0)),
          1: len(labels) / (2 * np.sum(labels == 1))}

    # ---- 训练分类头 ----
    inp = Input(shape=feats.shape[1:])
    x = Flatten()(inp)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(2, activation="softmax")(x)
    head = Model(inp, out)
    head.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    head.fit(Xt, yt_cat, validation_data=(Xv, yv_cat),
             epochs=EPOCHS, batch_size=BATCH, class_weight=cw, verbose=2)

    val = head.evaluate(Xv, yv_cat, verbose=0)
    print(f"[INFO] 验证集 accuracy = {val[1]:.3f}")

    # ---- 组装成 recognize.py 用的完整模型并保存权重 ----
    full = build_liveness_model()  # VGG16(imagenet) + 同结构 head
    dense_full = [l for l in full.layers if isinstance(l, _Dense)]
    dense_head = [l for l in head.layers if isinstance(l, _Dense)]
    for df, dh in zip(dense_full, dense_head):
        df.set_weights(dh.get_weights())

    if os.path.exists(LIVENESS_WEIGHTS):
        shutil.copy(LIVENESS_WEIGHTS, LIVENESS_WEIGHTS + ".bak")
        print(f"[INFO] 旧权重已备份为 {LIVENESS_WEIGHTS}.bak")
    full.save_weights(LIVENESS_WEIGHTS)
    print(f"[INFO] 新权重已保存到 {LIVENESS_WEIGHTS}")
    print("[INFO] 现在重新运行 python recognize.py 即可（真人应判 Real）。")


if __name__ == "__main__":
    main()
