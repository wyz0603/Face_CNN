# enroll.py - 登记身份
# 扫描 fengyizhuo/ 文件夹里的照片和视频，提取人脸特征，
# 生成身份库 identities.npz，供实时识别时把这个人标注为 "fengyizhuo"。
import os
import cv2
import numpy as np
from face_id import FaceID

# 每个身份的素材目录: {名字: 目录}
IDENTITY_DIRS = {
    "fengyizhuo": "fengyizhuo/fengyizhuo",
}

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
VIDEO_FRAME_STRIDE = 10   # 视频每隔多少帧采一次
VIDEO_MAX_FRAMES = 30     # 每个视频最多采多少张有效人脸


def collect_embeddings(face_id, directory):
    embeddings = []
    if not os.path.isdir(directory):
        print(f"[警告] 目录不存在: {directory}")
        return embeddings

    files = sorted(os.listdir(directory))
    for fname in files:
        path = os.path.join(directory, fname)
        ext = os.path.splitext(fname)[1].lower()

        if ext in IMG_EXT:
            image = cv2.imread(path)
            if image is None:
                continue
            feat = face_id.embedding_from_image(image)
            if feat is not None:
                embeddings.append(feat)
            else:
                print(f"  [跳过] 未检测到人脸: {fname}")

        elif ext in VIDEO_EXT:
            cap = cv2.VideoCapture(path)
            idx, got = 0, 0
            while got < VIDEO_MAX_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % VIDEO_FRAME_STRIDE == 0:
                    feat = face_id.embedding_from_image(frame)
                    if feat is not None:
                        embeddings.append(feat)
                        got += 1
                idx += 1
            cap.release()
            print(f"  [视频] {fname}: 采集 {got} 张人脸")

    return embeddings


def main():
    face_id = FaceID(db_path=None)  # 登记阶段不加载旧库

    for name, directory in IDENTITY_DIRS.items():
        print(f"[*] 正在登记身份: {name}  <- {directory}")
        embeddings = collect_embeddings(face_id, directory)
        print(f"[*] {name}: 共提取 {len(embeddings)} 个人脸特征")
        if len(embeddings) == 0:
            print(f"[错误] {name} 没有有效人脸，跳过。")
            continue
        face_id.add_identity(name, embeddings)

    face_id.save_db()


if __name__ == "__main__":
    main()
