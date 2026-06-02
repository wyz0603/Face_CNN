# test_video.py - 动态构建模型测试视频
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array


# ================= 配置参数 =================
VIDEO_PATH = "77e2ae96a1130f451e34e1c24d058d83.mp4"
FACE_DETECTOR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

FRAME_HISTORY = 10
CONF_THRESHOLD = 0.6
IMG_SIZE = 224
# ============================================


def build_model():
    """动态构建VGG16模型"""
    from tensorflow.keras.applications import VGG16

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    return model


class LivenessDetector:
    def __init__(self, face_cascade_path, frame_history=10, conf_threshold=0.6):
        print("[*] 构建模型...")
        self.model = build_model()
        self.model.load_weights('face_liveness_weights.h5')
        print("[*] 模型权重加载成功")

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.frame_history = frame_history
        self.conf_threshold = conf_threshold
        self.frame_buffer = deque(maxlen=frame_history)

    def preprocess_face(self, face_roi):
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = img_to_array(face_roi) / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        return face_roi

    def predict_single(self, face_roi):
        face_roi = self.preprocess_face(face_roi)
        (fake_prob, real_prob) = self.model.predict(face_roi, verbose=0)[0]
        return fake_prob, real_prob

    def predict_with_voting(self, face_roi):
        fake_prob, real_prob = self.predict_single(face_roi)
        self.frame_buffer.append((fake_prob, real_prob))

        if len(self.frame_buffer) < 3:
            return fake_prob, real_prob

        avg_fake = np.mean([f for f, r in self.frame_buffer])
        avg_real = np.mean([r for f, r in self.frame_buffer])

        return avg_fake, avg_real

    def detect_and_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            fake_prob, real_prob = self.predict_with_voting(face_roi)

            if fake_prob >= CONF_THRESHOLD:
                label = "Fake"
            elif real_prob >= CONF_THRESHOLD:
                label = "Real"
            else:
                label = "Waiting"

            confidence = max(real_prob, fake_prob) * 100
            results.append({'label': label, 'confidence': confidence})

            color = (0, 255, 0) if label == "Real" else (0, 0, 255) if label == "Fake" else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return results


def main():
    detector = LivenessDetector(FACE_DETECTOR_PATH, FRAME_HISTORY, CONF_THRESHOLD)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[!] 无法打开视频: {VIDEO_PATH}")
        return

    print(f"[*] 处理视频: {VIDEO_PATH}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    real_count = 0
    fake_count = 0
    waiting_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_and_predict(frame)

        for r in results:
            if r['label'] == 'Real':
                real_count += 1
            elif r['label'] == 'Fake':
                fake_count += 1
            else:
                waiting_count += 1

        cv2.imshow("Video Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("视频检测结果统计")
    print("="*50)
    print(f"Real: {real_count}")
    print(f"Fake: {fake_count}")
    print(f"Waiting: {waiting_count}")
    print("="*50)


if __name__ == "__main__":
    main()