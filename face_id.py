# face_id.py - 人脸身份识别 (YuNet 检测 + SFace 特征)
# 用于在活体检测通过后，判断这张脸是否为 "fengyizhuo"。
import os
import cv2
import numpy as np

# 模型文件（OpenCV Zoo）
DETECTOR_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")

# SFace 余弦相似度阈值：>= 视为同一个人。官方推荐 0.363，
# 这里略微调高以减少误识别（把别人认成 fengyizhuo）。
COSINE_THRESHOLD = 0.40

# 已登记身份库文件
DB_PATH = "identities.npz"


class FaceID:
    def __init__(self, detector_path=DETECTOR_PATH, recognizer_path=RECOGNIZER_PATH,
                 db_path=DB_PATH, cosine_threshold=COSINE_THRESHOLD,
                 score_threshold=0.6):
        if not os.path.exists(detector_path):
            raise FileNotFoundError(f"找不到人脸检测模型: {detector_path}")
        if not os.path.exists(recognizer_path):
            raise FileNotFoundError(f"找不到人脸特征模型: {recognizer_path}")

        # score_threshold 默认 0.6（YuNet 官方默认 0.9 对高分辨率自拍偏严）
        self.detector = cv2.FaceDetectorYN.create(
            detector_path, "", (320, 320), score_threshold)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        self.cosine_threshold = cosine_threshold

        # 身份库: {name: 平均特征向量(归一化)}
        self.db = {}
        if db_path and os.path.exists(db_path):
            self.load_db(db_path)

    # ---------------- 检测 ----------------
    def detect(self, image):
        """返回 YuNet 的人脸检测结果 (N, 15) 数组，没有则返回 None。"""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        return faces

    # ---------------- 特征 ----------------
    def embedding(self, image, face_row):
        """对单张人脸（YuNet 的一行检测结果）做对齐并提取 128 维特征。"""
        aligned = self.recognizer.alignCrop(image, face_row)
        feat = self.recognizer.feature(aligned)  # (1, 128)
        feat = feat.flatten().astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def embedding_from_image(self, image):
        """从整张图里取最大的一张脸，返回其特征。找不到脸返回 None。"""
        faces = self.detect(image)
        if faces is None or len(faces) == 0:
            return None
        # 取面积最大的脸
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return self.embedding(image, faces[0])

    # ---------------- 身份匹配 ----------------
    def identify(self, feat):
        """把特征和身份库比对，返回 (name, score)。无匹配返回 (None, best_score)。"""
        best_name, best_score = None, -1.0
        for name, ref in self.db.items():
            score = float(np.dot(feat, ref))  # 两者均已归一化 -> 余弦相似度
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= self.cosine_threshold:
            return best_name, best_score
        return None, best_score

    # ---------------- 身份库读写 ----------------
    def add_identity(self, name, embeddings):
        """用一组特征的平均值登记一个身份。"""
        embeddings = np.array(embeddings, dtype=np.float32)
        mean = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        self.db[name] = mean

    def save_db(self, db_path=DB_PATH):
        if not self.db:
            raise RuntimeError("身份库为空，无法保存。")
        np.savez(db_path, **self.db)
        print(f"[*] 身份库已保存到 {db_path}，包含: {list(self.db.keys())}")

    def load_db(self, db_path=DB_PATH):
        data = np.load(db_path)
        self.db = {name: data[name] for name in data.files}
        print(f"[*] 已加载身份库: {list(self.db.keys())}")
