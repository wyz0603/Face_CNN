# collect_samples.py - 用摄像头采集活体训练样本
# 真人样本:  python collect_samples.py --label real  --max-samples 400
# 翻拍样本:  python collect_samples.py --label spoof --max-samples 400
#   (翻拍 = 对着摄像头展示 打印照片 / 手机里的照片 / 手机播放的视频 等)
# 样本保存到 custom_data/real 和 custom_data/spoof，供 train_custom.py 训练。
import os
import argparse
import time
import cv2

# 与 recognize.py 保持一致的裁剪参数，保证训练分布与推理一致
IMG_SIZE = 224
FACE_MARGIN = 0.2
DETECTOR_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
FACE_SCORE_THRESHOLD = 0.6


def crop_face(frame, face):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=["real", "spoof"])
    ap.add_argument("--max-samples", type=int, default=400)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="每隔多少帧存一张，避免重复")
    args = ap.parse_args()

    out_dir = os.path.join("custom_data", args.label)
    os.makedirs(out_dir, exist_ok=True)

    detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), FACE_SCORE_THRESHOLD)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[*] 采集 [{args.label}] 样本，目标 {args.max_samples} 张，保存到 {out_dir}")
    if args.label == "real":
        print("[*] 请本人正对摄像头，缓慢转头/移动/变换表情与距离，光线尽量多样。")
    else:
        print("[*] 请对着摄像头展示 照片/手机照片/手机播放的视频，多换几张、多换角度与远近。")
    print("[*] 按 q 结束。")

    count = 0
    idx = 0
    while count < args.max_samples:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:
            face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            x, y, fw, fh = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            if idx % args.every == 0:
                face_img = crop_face(frame, face)
                if face_img is not None:
                    fname = os.path.join(out_dir, f"{args.label}_{int(time.time()*1000)}_{count}.jpg")
                    cv2.imwrite(fname, face_img)
                    count += 1

        cv2.putText(frame, f"{args.label}: {count}/{args.max_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("collect", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[*] 完成，共保存 {count} 张到 {out_dir}")


if __name__ == "__main__":
    main()
