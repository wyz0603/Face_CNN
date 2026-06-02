# test.py - 严格假脸检测模式
import cv2
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array


# ================= 配置参数 =================
FACE_DETECTOR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

FRAME_HISTORY = 10
CONF_THRESHOLD = 0.5  # 阈值
IMG_SIZE = 224

# 严格模式参数
REQUIRED_CONSECUTIVE_REAL = 20  # 需要连续20帧高置信度真脸
REAL_THRESHOLD = 0.95  # 真脸阈值需要95%以上

# 主动活体挑战参数
ENABLE_ACTIVE_CHALLENGE = True
CHALLENGE_STABLE_FRAMES = 5
LEFT_BOUNDARY = 0.42
RIGHT_BOUNDARY = 0.58
CENTER_MIN = 0.45
CENTER_MAX = 0.55
# ============================================


def build_model():
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
    def __init__(self, face_cascade_path, frame_history=10, conf_threshold=0.5):
        print("[*] 构建模型...")
        self.model = build_model()
        self.model.load_weights('face_liveness_weights.h5')
        print("[*] 模型权重加载成功")

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.frame_history = frame_history
        self.conf_threshold = conf_threshold
        self.frame_buffer = deque(maxlen=frame_history)
        self.consecutive_real = 0  # 连续真脸计数
        self.is_verified_real = False  # 是否已验证为真脸
        self.challenge_steps = []
        self.challenge_index = 0
        self.challenge_hits = 0
        self.challenge_completed = not ENABLE_ACTIVE_CHALLENGE
        self.reset_challenge()

    def reset_challenge(self):
        if not ENABLE_ACTIVE_CHALLENGE:
            self.challenge_steps = []
            self.challenge_index = 0
            self.challenge_hits = 0
            self.challenge_completed = True
            return

        directions = ["LEFT", "RIGHT"]
        random.shuffle(directions)
        self.challenge_steps = ["CENTER"] + directions + ["CENTER"]
        self.challenge_index = 0
        self.challenge_hits = 0
        self.challenge_completed = False

    def reset_state(self):
        self.frame_buffer.clear()
        self.consecutive_real = 0
        self.is_verified_real = False
        self.reset_challenge()

    def current_challenge_text(self):
        if self.challenge_completed:
            return "Challenge: passed"

        text_map = {
            "LEFT": "Challenge: move left",
            "RIGHT": "Challenge: move right",
            "CENTER": "Challenge: keep center",
        }
        target = self.challenge_steps[self.challenge_index]
        return f"{text_map[target]} ({self.challenge_index + 1}/{len(self.challenge_steps)})"

    def _face_position(self, bbox, frame_width):
        x, _, w, _ = bbox
        center_ratio = (x + w / 2) / frame_width

        if CENTER_MIN <= center_ratio <= CENTER_MAX:
            return "CENTER"
        if center_ratio < LEFT_BOUNDARY:
            return "LEFT"
        if center_ratio > RIGHT_BOUNDARY:
            return "RIGHT"
        return "MIDDLE"

    def update_challenge(self, bbox, frame_shape, model_real_ok):
        if self.challenge_completed or not ENABLE_ACTIVE_CHALLENGE:
            return

        if not model_real_ok:
            self.challenge_hits = 0
            return

        frame_width = frame_shape[1]
        target = self.challenge_steps[self.challenge_index]
        position = self._face_position(bbox, frame_width)

        if position == target:
            self.challenge_hits += 1
        else:
            self.challenge_hits = max(0, self.challenge_hits - 1)

        if self.challenge_hits >= CHALLENGE_STABLE_FRAMES:
            self.challenge_index += 1
            self.challenge_hits = 0

            if self.challenge_index >= len(self.challenge_steps):
                self.challenge_completed = True

    def preprocess_face(self, face_roi):
        face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face_roi = img_to_array(face_roi) / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        return face_roi

    def predict_single(self, face_roi):
        (fake_prob, real_prob) = self.model.predict(face_roi, verbose=0)[0]
        return fake_prob, real_prob

    def predict_with_voting(self, face_roi_prepared, original_face=None):
        fake_prob, real_prob = self.predict_single(face_roi_prepared)
        model_real_ok = real_prob > REAL_THRESHOLD

        # 严格模式：只有连续20帧 > 95% 置信度才认为是真脸
        if model_real_ok:
            self.consecutive_real += 1
        else:
            self.consecutive_real = max(0, self.consecutive_real - 1)

        # 未通过完整验证前，始终偏向假脸，避免照片/视频凭单帧分数直接放行。
        if not self.is_verified_real:
            # 强制提高假脸概率
            fake_prob = max(fake_prob, 1 - real_prob + 0.3)

        self.frame_buffer.append((fake_prob, real_prob))

        if len(self.frame_buffer) < 3:
            return fake_prob, real_prob, model_real_ok

        avg_fake = np.mean([f for f, r in self.frame_buffer])
        avg_real = np.mean([r for f, r in self.frame_buffer])

        return avg_fake, avg_real, model_real_ok

    def detect_and_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        results = []

        for (x, y, w, h) in faces:
            face_roi_original = frame[y:y + h, x:x + w].copy()
            face_roi_prepared = self.preprocess_face(face_roi_original)
            fake_prob, real_prob, model_real_ok = self.predict_with_voting(
                face_roi_prepared,
                original_face=face_roi_original
            )
            self.update_challenge((x, y, w, h), frame.shape, model_real_ok)
            self.is_verified_real = (
                self.consecutive_real >= REQUIRED_CONSECUTIVE_REAL
                and self.challenge_completed
            )

            # 判断结果
            if self.is_verified_real:
                label = "Real"
                color = (0, 255, 0)
            elif real_prob >= REAL_THRESHOLD and not self.challenge_completed:
                label = "Verify"
                color = (255, 255, 0)
            elif fake_prob >= self.conf_threshold:
                label = "Fake"
                color = (0, 0, 255)
            else:
                label = "Waiting"
                color = (255, 255, 0)

            confidence = max(real_prob, fake_prob) * 100
            results.append({
                'bbox': (x, y, w, h),
                'label': label,
                'confidence': confidence,
                'color': color,
                'real_prob': real_prob,
                'fake_prob': fake_prob,
                'challenge': self.current_challenge_text()
            })

        return results


def draw_results(frame, results):
    for r in results:
        x, y, w, h = r['bbox']
        label = r['label']
        confidence = r['confidence']
        color = r['color']

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        text = f"{label}: {confidence:.1f}%"
        cv2.putText(frame, text, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        bar_width = w
        real_bar = int(bar_width * r['real_prob'])
        cv2.rectangle(frame, (x, y + h + 5), (x + real_bar, y + h + 15), (0, 255, 0), -1)
        cv2.rectangle(frame, (x + real_bar, y + h + 5), (x + w, y + h + 15), (0, 0, 255), -1)

        cv2.putText(frame, r['challenge'], (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def main():
    detector = LivenessDetector(
        FACE_DETECTOR_PATH,
        frame_history=FRAME_HISTORY,
        conf_threshold=CONF_THRESHOLD
    )

    print("[*] 启动摄像头... 按 'q' 退出, 按 'r' 重置")
    print(f"[*] 严格模式：需要连续{REQUIRED_CONSECUTIVE_REAL}帧 >{REAL_THRESHOLD*100:.0f}% 置信度才判定为真脸")
    if ENABLE_ACTIVE_CHALLENGE:
        print("[*] 主动活体模式：需要按屏幕提示完成左右移动挑战")

    vs = cv2.VideoCapture(0)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = detector.detect_and_predict(frame)
        draw_results(frame, results)

        status = f"Verified: {detector.is_verified_real}, Buffer: {len(detector.frame_buffer)}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_state()
            print("[*] 状态已重置，已生成新的活体挑战")

    print("[*] 程序结束")
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
