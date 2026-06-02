# 导入必要的包
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imutils import paths
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

# --- 关键修复：确保能正确导入你写的模型 ---
from model import VGG16Liveness

# ---------------- 配置参数 ----------------
DATASET_PATHS = [
    "LCC_FASD/LCC_FASD_training",
    "custom_data",
]
IMG_WIDTH = 224  # VGG16标准输入尺寸
IMG_HEIGHT = 224
BATCH_SIZE = 16  # 减小batch size
EPOCHS = 20
LEARNING_RATE = 1e-4
BEST_MODEL_PATH = "face_liveness_best.keras"

# ---------------- 1. 加载数据 ----------------
print("[INFO] 正在加载图像数据...")
imagePaths = []
for dataset_path in DATASET_PATHS:
    if os.path.isdir(dataset_path):
        found = list(paths.list_images(dataset_path))
        imagePaths.extend(found)
        print(f"[INFO] 数据源 {dataset_path}: {len(found)} 张图片")
    else:
        print(f"[INFO] 数据源 {dataset_path} 不存在，已跳过")

data = []
labels = []

# 循环处理每一张图片
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]  # 获取倒数第二个文件夹名

    image = cv2.imread(imagePath)
    if image is None:
        print(f"[警告] 无法读取图片: {imagePath}")
        continue

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # spo -spoof -> 0, real -> 1
    if label == "spoof":
        labels.append(0)
        data.append(image)
    elif label == "real":
        labels.append(1)
        data.append(image)
    else:
        print(f"[警告] 发现未知文件夹名: {label}，已跳过。")
        continue

# 归一化
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)

if len(data) == 0:
    raise RuntimeError("[错误] 没有加载到任何训练图片，请检查数据集路径和 real/spoof 目录。")

print(f"[INFO] 数据加载完成: {len(data)} 张图片")
print(f"[INFO] 标签分布: 假脸 {np.sum(labels==0)} 张, 真脸 {np.sum(labels==1)} 张")

# 数据不平衡处理
from collections import Counter
class_counts = Counter(labels)
print(f"[INFO] 类别分布: {dict(class_counts)}")

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels
)
class_weight = {0: float(class_weights[0]), 1: float(class_weights[1])}
print(f"[INFO] 类别权重: {class_weight}")

# ---------------- 2. 数据预处理 ----------------
# 划分训练集和测试集
(trainX, testX, trainYRaw, testYRaw) = train_test_split(
    data,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

trainY = to_categorical(trainYRaw, num_classes=2)
testY = to_categorical(testYRaw, num_classes=2)

# 数据增强
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True,
                         fill_mode="nearest")

# ---------------- 3. 初始化并编译模型 ----------------
print("[INFO] 正在编译模型...")
model = VGG16Liveness.build(width=IMG_WIDTH, height=IMG_HEIGHT, depth=3, classes=2)

def lr_schedule(epoch):
    return LEARNING_RATE * (0.9 ** epoch)

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
opt = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# ---------------- 4. 训练网络 ----------------
print("[INFO] 开始训练网络...")
H = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
              validation_data=(testX, testY),
              steps_per_epoch=max(1, len(trainX) // BATCH_SIZE),
              epochs=EPOCHS,
              verbose=1,
              callbacks=[lr_scheduler, checkpoint, early_stop],
              class_weight=class_weight)

# ---------------- 5. 评估模型 ----------------
print("[INFO] 正在评估模型...")
predictions = model.predict(testX, batch_size=BATCH_SIZE)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["spoof", "real"]))

# ---------------- 6. 保存模型 ----------------
print("[INFO] 正在保存模型...")
model.save("face_liveness.keras")
model.save_weights("face_liveness_weights.h5")

# 绘制训练历史
plt.style.use("ggplot")
plt.figure()
epochs_ran = np.arange(0, len(H.history["loss"]))
plt.plot(epochs_ran, H.history["loss"], label="train_loss")
plt.plot(epochs_ran, H.history["val_loss"], label="val_loss")
plt.plot(epochs_ran, H.history["accuracy"], label="train_acc")
plt.plot(epochs_ran, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("training_plot.png")
plt.show()
