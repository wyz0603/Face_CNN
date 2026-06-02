# Face Liveness Detection - 人脸活体检测 (VGG16)

## ★ 活体检测 + 身份识别 (recognize.py)

在原有"真人/假脸"活体检测的基础上，新增了身份识别：
- **真人(活体)** -> 绿框，True
- **照片 / 手机播放的视频** -> 红框，Fake，False
- 当判定为真人、且与身份库匹配时，额外把名字标注出来（如 `fengyizhuo`）；
  其他人只显示 `Real`，不标注名字。

技术方案：
- 活体检测：TensorFlow VGG16 模型 (face_liveness_weights.h5)，强烈抑制照片/翻拍。
- 人脸检测+身份识别：OpenCV 内置 YuNet (检测) + SFace (128 维特征)，
  余弦相似度与身份库 identities.npz 比对，无需 dlib / face_recognition。
- 跨帧跟踪 + 活体/身份时间投票，抗单帧抖动与单帧攻击。

### 使用步骤

1) 安装依赖：`pip install -r requirements.txt`
   （需联网下载 models/ 下的两个 onnx 模型，已随仓库提供）

2) 登记身份（扫描 fengyizhuo/ 文件夹生成 identities.npz）：
   ```bash
   python enroll.py
   ```
   想新增其他人：在 enroll.py 的 IDENTITY_DIRS 里加一行 `"名字": "目录"` 再运行。

3) 摄像头实时识别：
   ```bash
   python recognize.py        # 按 q 退出
   ```

4) 用视频文件离线测试整套流程：
   ```bash
   python test_recognize_video.py 某视频.mp4 [--show]
   ```

可调参数（recognize.py 顶部）：REAL_THRESHOLD（活体阈值，越高越严格地拒绝翻拍）、
COSINE_THRESHOLD（身份阈值，越高越不容易把别人认成 fengyizhuo）。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| model.py | VGG16预训练模型 |
| train.py | 训练模型 |
| collect_samples.py | 采集本机摄像头 real/spoof 再训练样本 |
| test.py | 摄像头实时检测 |
| real_time_test.py | 简化版摄像头实时检测 |
| test_video.py | 测试视频文件 |
| face_liveness.keras | 训练好的模型 |
| face_liveness_weights.h5 | test.py/test_video.py 使用的模型权重 |
| requirements.txt | 依赖包 |

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 摄像头实时检测

```bash
python test.py
```

- 按 `q` 退出
- 按 `r` 重置缓冲区

### 3. 测试视频

```bash
python test_video.py
```

### 4. 简化版摄像头检测

```bash
python real_time_test.py
```

### 5. 采集本机失败样本

采集真人样本：

```bash
python collect_samples.py --label real --max-samples 300
```

采集照片、手机视频、屏幕翻拍等攻击样本：

```bash
python collect_samples.py --label spoof --max-samples 500
```

样本会保存到 `custom_data/real` 和 `custom_data/spoof`，`train.py` 会自动合并这些样本。

### 6. 重新训练（如需）

```bash
python train.py
```

## 识别结果

- **Real（绿色）**: 真人（活体）
- **Fake（红色）**: 照片/视频/假脸（非活体）
