# train_fyz.py - 用 fyz/(real,fake) + custom_data/(real,spoof) 微调活体模型
# 关键：用 YuNet 把每张图裁成人脸(+margin)，与 recognize.py 推理输入一致；
#       并对样本做增强(亮度/模糊/JPEG压缩/翻转)，提升对不同翻拍方式的泛化。
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

# real -> 1, fake/spoof -> 0
REAL_DIRS = ["fyz/real", "custom_data/real"]
FAKE_DIRS = ["fyz/fake", "custom_data/spoof"]
VIDEO_STRIDE = 5
EPOCHS = 40
BATCH = 16

_fid = FaceID(db_path=None)


def crop_face(frame):
    """YuNet 取最大人脸 + margin，裁剪并 resize 到 224(uint8 BGR)。无脸返回 None。"""
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
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def augment(img):
    """对一张 uint8 BGR 人脸生成多种变体，提升泛化。"""
    out = [img, img[:, ::-1, :]]  # 原图 + 水平翻转
    out.append(np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8))   # 变暗
    out.append(np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8))   # 变亮
    out.append(cv2.GaussianBlur(img, (5, 5), 0))                                  # 模糊
    enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 25])[1]            # 低质 JPEG
    out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    return out


def load(directory, label):
    crops = []
    if not os.path.isdir(directory):
        return crops, []
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
            crops.append(c)
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
                    crops.append(c)
            i += 1
        cap.release()
    print(f"[INFO] {directory}: 裁出人脸 {len(crops)} 张 (图片漏检 {miss})")
    return crops, [label] * len(crops)


def main():
    raw, y_raw = [], []
    for label, dirs in [(1, REAL_DIRS), (0, FAKE_DIRS)]:
        for d in dirs:
            c, yy = load(d, label)
            raw += c
            y_raw += yy
    n_real = sum(1 for v in y_raw if v == 1)
    n_fake = sum(1 for v in y_raw if v == 0)
    print(f"[INFO] 原始人脸: real={n_real}, fake={n_fake}")
    if n_real < 30 or n_fake < 30:
        raise RuntimeError("样本太少。请确保 fyz/real、fyz/fake(或 custom_data) 各有足够人脸。")

    # 增强
    X, y = [], []
    for img, label in zip(raw, y_raw):
        for v in augment(img):
            X.append(v.astype("float32") / 255.0)
            y.append(label)
    X = np.array(X, dtype="float32")
    y = np.array(y)
    print(f"[INFO] 增强后样本: {len(X)} 张 (real={int(np.sum(y==1))}, fake={int(np.sum(y==0))})")

    base = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    print("[INFO] 预计算 VGG16 特征中...")
    feats = base.predict(X, batch_size=BATCH, verbose=1)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(feats))
    feats, y = feats[perm], y[perm]
    n_val = max(1, int(0.2 * len(feats)))
    Xv, yv = feats[:n_val], y[:n_val]
    Xt, yt = feats[n_val:], y[n_val:]

    cw = {0: len(y) / (2 * np.sum(y == 0)), 1: len(y) / (2 * np.sum(y == 1))}

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

    pv = head.predict(Xv, verbose=0).argmax(1)
    print(f"[INFO] 验证 accuracy={np.mean(pv==yv):.3f}  "
          f"真人召回={np.mean(pv[yv==1]==1):.3f}  假脸召回={np.mean(pv[yv==0]==0):.3f}")

    full = build_liveness_model()
    dense_full = [l for l in full.layers if isinstance(l, Dense)]
    dense_head = [l for l in head.layers if isinstance(l, Dense)]
    for df, dh in zip(dense_full, dense_head):
        df.set_weights(dh.get_weights())

    # 先存到一个新文件（即使目标文件被占用/只读也不会丢失训练成果）
    import stat
    tmp = LIVENESS_WEIGHTS + ".new.h5"
    full.save_weights(tmp)
    print(f"[INFO] 新权重已保存到 {tmp}")

    # 再尝试覆盖正式权重文件
    try:
        if os.path.exists(LIVENESS_WEIGHTS):
            os.chmod(LIVENESS_WEIGHTS, stat.S_IWRITE)        # 清除只读
            shutil.copy(LIVENESS_WEIGHTS, LIVENESS_WEIGHTS + ".bak")
            print(f"[INFO] 旧权重已备份为 {LIVENESS_WEIGHTS}.bak")
        os.replace(tmp, LIVENESS_WEIGHTS)
        print(f"[INFO] 已更新 {LIVENESS_WEIGHTS}，现在运行 python recognize.py")
    except Exception as e:
        print(f"[!] 无法覆盖 {LIVENESS_WEIGHTS}: {e}")
        print(f"[!] 训练成果已安全保存在 {tmp}。")
        print(f"[!] 请关闭占用该文件的程序(如另一个 recognize.py 窗口)，再手动改名：")
        print(f"      del {LIVENESS_WEIGHTS}  &&  ren {tmp} {LIVENESS_WEIGHTS}")


if __name__ == "__main__":
    main()
