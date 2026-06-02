# utils.py
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model # 注意：这里我们需要加载训练好的模型

MODEL_PATH = "face_liveness.keras"
INPUT_SIZE = (224, 224)
BOX = (400, 100, 900, 550)

# 全局加载模型
print("[INFO] 加载模型...")
model = load_model(MODEL_PATH)

def predictperson():
    # 打开摄像头
    video_capture = cv2.VideoCapture(0)
    
    # 加载OpenCV自带的人脸检测分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        # 按 'b' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
            
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("[警告] 摄像头画面读取失败")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 绘制提示框 (蓝色)
        box_x1, box_y1, box_x2, box_y2 = BOX
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)
        cv2.putText(frame, "Please keep your head inside the blue box", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        faces_inside_box = []
        
        # 遍历检测到的人脸
        for (x, y, w, h) in faces:
            # 判断人脸是否在蓝色框内
            if x > box_x1 and y > box_y1 and (x + w) < box_x2 and (y + h) < box_y2:
                faces_inside_box.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # 画绿色框
                
        # 根据检测到的人数进行判断
        if len(faces_inside_box) > 1:
            cv2.putText(frame, "Multiple Faces detected!", (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif len(faces_inside_box) == 1:
            x, y, w, h = faces_inside_box[0]
            face_roi = frame[y:y + h, x:x + w]
            image = cv2.resize(face_roi, INPUT_SIZE)
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) # 增加batch维度
            
            # 进行预测
            fake_prob, real_prob = model.predict(image, verbose=0)[0]
            
            # 训练时: fake=0, real=1 -> 模型输出 [fake_prob, real_prob]
            # fake_prob > real_prob 为假脸
            label = "fake" if fake_prob > real_prob else "real"
            confidence = max(fake_prob, real_prob)
            color = (0, 0, 255) if label == "fake" else (0, 255, 0)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Please come closer to the camera", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predictperson()
