# recognize.py - 活体检测 + 身份识别
# 流程：YuNet 检测人脸 -> VGG16 活体模型判断真人/假脸 -> SFace 识别是否为 fengyizhuo
#   真人(活体)         -> 绿框，True
#   照片/手机播放视频   -> 红框，False (Fake)
#   若真人且匹配身份库   -> 在标签上额外标注其名字 (fengyizhuo)，其他人不标名字
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from collections import deque

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16

from face_id import FaceID

# ================= 配置参数 =================
LIVENESS_WEIGHTS = "face_liveness_weights.h5"
IMG_SIZE = 224

# 调试：在画面上叠加原始 real_prob / 身份相似度，用于校准阈值
DEBUG = True

# 活体判定
REAL_THRESHOLD = 0.3          # 平滑后 real_prob 超过该值才算真人（按你的摄像头实际分数再调）
FRAME_HISTORY = 8             # 活体概率平滑窗口
REQUIRED_CONSECUTIVE = 3      # 需要连续多少帧判为真人才确认（防抖、抗单帧攻击）
FACE_MARGIN = 0.2             # 活体输入人脸裁剪外扩比例（贴合训练数据，过紧会误判 Fake）

# 身份识别
FACE_SCORE_THRESHOLD = 0.6    # YuNet 人脸检测置信度
COSINE_THRESHOLD = 0.40       # SFace 余弦相似度阈值
ID_SMOOTH = 7                 # 身份投票窗口

# 简单跨帧人脸跟踪：中心点距离小于该比例(相对画面宽)视为同一张脸
TRACK_DIST_RATIO = 0.15
TRACK_TTL = 15                # 多少帧未更新则丢弃该轨迹
# ============================================


def build_liveness_model():
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
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


class Track:
    """一张被持续跟踪的人脸的状态。"""
    _next_id = 0

    def __init__(self, cx, cy):
        self.id = Track._next_id
        Track._next_id += 1
        self.cx, self.cy = cx, cy
        self.live_buffer = deque(maxlen=FRAME_HISTORY)   # (fake_prob, real_prob)
        self.id_votes = deque(maxlen=ID_SMOOTH)          # name or None
        self.consecutive_real = 0
        self.ttl = TRACK_TTL

    def update_pos(self, cx, cy):
        self.cx, self.cy = cx, cy
        self.ttl = TRACK_TTL

    def push_liveness(self, fake_prob, real_prob):
        self.live_buffer.append((fake_prob, real_prob))
        if real_prob >= REAL_THRESHOLD:
            self.consecutive_real += 1
        else:
            self.consecutive_real = max(0, self.consecutive_real - 1)

    def smoothed(self):
        avg_fake = float(np.mean([f for f, r in self.live_buffer]))
        avg_real = float(np.mean([r for f, r in self.live_buffer]))
        return avg_fake, avg_real

    def is_real(self):
        if len(self.live_buffer) == 0:
            return False
        _, avg_real = self.smoothed()
        return avg_real >= REAL_THRESHOLD and self.consecutive_real >= REQUIRED_CONSECUTIVE

    def push_id(self, name):
        self.id_votes.append(name)

    def voted_name(self):
        """返回出现次数最多且过半的身份名，否则 None。"""
        if not self.id_votes:
            return None
        names = [n for n in self.id_votes if n is not None]
        if not names:
            return None
        best = max(set(names), key=names.count)
        if names.count(best) >= max(2, len(self.id_votes) // 2):
            return best
        return None


class Recognizer:
    def __init__(self):
        print("[*] 加载身份识别模型 (YuNet + SFace)...")
        self.face_id = FaceID(cosine_threshold=COSINE_THRESHOLD,
                              score_threshold=FACE_SCORE_THRESHOLD)
        if not self.face_id.db:
            print("[!] 警告：身份库为空，请先运行 python enroll.py")

        print("[*] 构建活体检测模型 (VGG16)...")
        self.live_model = build_liveness_model()
        self.live_model.load_weights(LIVENESS_WEIGHTS)
        print("[*] 活体模型权重加载成功")

        self.tracks = []

    # ---------- 跟踪 ----------
    def _match_track(self, cx, cy, frame_w):
        best, best_d = None, TRACK_DIST_RATIO * frame_w
        for t in self.tracks:
            d = np.hypot(cx - t.cx, cy - t.cy)
            if d < best_d:
                best, best_d = t, d
        return best

    def _age_tracks(self):
        for t in self.tracks:
            t.ttl -= 1
        self.tracks = [t for t in self.tracks if t.ttl > 0]

    # ---------- 活体预测 ----------
    def _liveness(self, face_crop):
        roi = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        fake_prob, real_prob = self.live_model.predict(roi, verbose=0)[0]
        return float(fake_prob), float(real_prob)

    # ---------- 主处理 ----------
    def process(self, frame):
        h, w = frame.shape[:2]
        faces = self.face_id.detect(frame)
        results = []

        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                x, y = max(0, x), max(0, y)
                fw, fh = min(fw, w - x), min(fh, h - y)
                if fw <= 0 or fh <= 0:
                    continue
                cx, cy = x + fw / 2, y + fh / 2

                track = self._match_track(cx, cy, w)
                if track is None:
                    track = Track(cx, cy)
                    self.tracks.append(track)
                else:
                    track.update_pos(cx, cy)

                # 活体：裁剪人脸时外扩一点 margin，更贴合训练数据（过紧易误判 Fake）
                mx, my = int(fw * FACE_MARGIN), int(fh * FACE_MARGIN)
                x0, y0 = max(0, x - mx), max(0, y - my)
                x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
                crop = frame[y0:y1, x0:x1]
                fake_prob, real_prob = self._liveness(crop)
                track.push_liveness(fake_prob, real_prob)
                real = track.is_real()

                # 身份：对每张检测到的人脸都识别（与活体解耦，便于验证 / 调试）
                feat = self.face_id.embedding(frame, face)
                matched, id_score = self.face_id.identify(feat)
                track.push_id(matched)
                name = track.voted_name()

                avg_fake, avg_real = track.smoothed()
                if real:
                    # 真人：若匹配身份库则标名字，否则仅 Real
                    label = name if name else "Real"
                    color = (0, 255, 0)
                    conf = avg_real
                else:
                    label = "Fake"
                    color = (0, 0, 255)
                    conf = avg_fake

                results.append({
                    'bbox': (x, y, fw, fh),
                    'label': label,
                    'name': name,
                    'real': real,
                    'conf': conf,
                    'color': color,
                    'real_prob': avg_real,
                    'id_score': id_score,
                })

        self._age_tracks()
        return results


def draw(frame, results):
    for r in results:
        x, y, w, h = r['bbox']
        color = r['color']
        text = f"{r['label']} {r['conf']*100:.0f}%"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 28), (x + max(w, 120), y), color, -1)
        cv2.putText(frame, text, (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if DEBUG:
            # 在框下方显示原始分数，便于校准阈值：
            #   real_prob=活体真人概率(>=REAL_THRESHOLD 才算真人)
            #   id=与 fengyizhuo 的相似度(>=COSINE_THRESHOLD 才匹配)
            dbg = f"real_prob={r['real_prob']:.2f} (thr {REAL_THRESHOLD})  id={r['id_score']:.2f} (thr {COSINE_THRESHOLD})"
            cv2.putText(frame, dbg, (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    rec = Recognizer()
    print("[*] 启动摄像头... 按 'q' 退出")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = rec.process(frame)
        draw(frame, results)
        cv2.imshow("Liveness + Face ID", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
