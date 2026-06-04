# test_recognize_video.py - 用视频文件离线测试 recognize.py 的完整流程
# 用法: python test_recognize_video.py <视频路径> [--show]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import cv2
from collections import Counter
from recognize import Recognizer, draw


def main():
    if len(sys.argv) < 2:
        print("用法: python test_recognize_video.py <视频路径> [--show]")
        return
    video = sys.argv[1]
    show = "--show" in sys.argv

    rec = Recognizer()
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[!] 无法打开视频: {video}")
        return

    print(f"[*] 处理视频: {video}")
    stats = Counter()
    name_stats = Counter()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = rec.process(frame)
        for r in results:
            stats["Real" if r['real'] else "Fake"] += 1
            if r['name']:
                name_stats[r['name']] += 1
        if show:
            frame = draw(frame, results)
            cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    print("\n==== 统计 ====")
    print(f"总帧数: {frame_idx}")
    print(f"活体判定: {dict(stats)}")
    print(f"识别到的身份(帧数): {dict(name_stats)}")


if __name__ == "__main__":
    main()
