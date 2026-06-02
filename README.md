# 人脸活体检测 + 身份识别 (Face Liveness + Face ID)

一套基于摄像头的人脸**活体检测**系统，并在其上叠加**指定人物身份识别**。

| 场景 | 结果 | 标注 |
|------|------|------|
| 真人（活体）出现在镜头前 | `True` | 绿框 `Real` |
| 用照片、手机播放的视频/图片冒充 | `False` | 红框 `Fake` |
| 真人且匹配身份库中的人（如 `fengyizhuo`） | `True` | 绿框 + 名字 `fengyizhuo` |
| 真人但不在身份库中（其他人） | `True` | 绿框 `Real`（不标名字） |

---

## 1. 工作原理

系统对每一帧做三件事：

1. **人脸检测** —— OpenCV 内置 **YuNet**（`cv2.FaceDetectorYN`）定位画面中所有人脸。
2. **活体检测** —— 把人脸送入 **TensorFlow VGG16** 模型（`face_liveness_weights.h5`），
   输出 `real / fake` 概率。该模型在 LCC-FASD 数据集上训练，能强烈抑制照片、屏幕翻拍、手机回放等攻击。
3. **身份识别** —— 仅当判定为真人时，用 OpenCV 内置 **SFace**（`cv2.FaceRecognizerSF`）
   提取 128 维人脸特征，与身份库 `identities.npz` 做余弦相似度比对，命中则标注其名字。

此外还有两层稳健性处理：

- **跨帧跟踪**：按人脸中心位置在帧间关联同一张脸，各自维护独立状态。
- **时间投票**：活体结果做滑动平均 + 连续帧确认（抗单帧抖动 / 单帧攻击）；
  身份结果做多帧多数投票（抗偶发误识别）。

> **为什么身份识别不需要训练？**
> SFace 是预训练好的人脸特征模型，识别靠“比对相似度”而非“训练分类器”。
> 登记一个人 = 提取其几张照片的特征取平均存盘，几秒完成，无需 GPU、无需重训。
> 新增/更换人物只要换照片重跑 `enroll.py` 即可。

---

## 2. 环境与依赖

- **Python 3.8**
- **TensorFlow 2.13.1**（用于加载活体模型权重；注意 TF 2.15 不支持 Python 3.8）
- **opencv-python ≥ 4.10**（需自带 `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`，本项目用 4.13 验证）

安装：

```bash
pip install -r requirements.txt
```

依赖 `models/` 下两个 ONNX 模型（已随项目提供）：

| 模型 | 文件 |
|------|------|
| YuNet 人脸检测 | `models/face_detection_yunet_2023mar.onnx` |
| SFace 人脸特征 | `models/face_recognition_sface_2021dec.onnx` |

如缺失，可从 [OpenCV Zoo](https://github.com/opencv/opencv_zoo) 重新下载到 `models/`。

> ⚠️ **活体模型权重未随仓库提供**：`face_liveness_weights.h5`（约 106MB）超过 GitHub 100MB
> 上限，未纳入版本库。请单独获取该权重文件并放到项目根目录后，再运行 `recognize.py`。
> 同理 `LCC_FASD/` 训练数据集（约 4.7GB）也未上传，需要训练时请自行下载到该目录。

---

## 3. 快速开始

### ① 登记身份（生成身份库）

扫描 `fengyizhuo/` 文件夹里的照片和视频，提取人脸特征，生成 `identities.npz`：

```bash
python enroll.py
```

输出示例：
```
[*] 正在登记身份: fengyizhuo  <- fengyizhuo/fengyizhuo
[*] fengyizhuo: 共提取 88 个人脸特征
[*] 身份库已保存到 identities.npz，包含: ['fengyizhuo']
```

### ② 摄像头实时识别（主程序）

```bash
python recognize.py
```

- 真人 → 绿框 `Real`；若是 fengyizhuo → 绿框标注 `fengyizhuo`
- 照片 / 手机回放视频 → 红框 `Fake`
- 按 `q` 退出

### ③ 用视频文件离线测试

无摄像头时，可用视频文件跑通整套流程：

```bash
python test_recognize_video.py 某段视频.mp4          # 仅打印统计
python test_recognize_video.py 某段视频.mp4 --show    # 同时显示画面
```

---

## 4. 新增/更换识别对象

1. 在项目下新建一个文件夹，例如 `zhangsan/`，放入此人的多张正脸照片（也可放短视频）。
2. 编辑 `enroll.py`，在 `IDENTITY_DIRS` 中加一行：

   ```python
   IDENTITY_DIRS = {
       "fengyizhuo": "fengyizhuo/fengyizhuo",
       "zhangsan":   "zhangsan",
   }
   ```
3. 重新运行 `python enroll.py`，即可把新身份并入 `identities.npz`。**全程无需训练。**

> 建议每人提供 20+ 张不同角度/表情的清晰正脸照，识别更稳。

---

## 5. 参数调优

集中在 `recognize.py` 顶部：

| 参数 | 含义 | 调整建议 |
|------|------|----------|
| `REAL_THRESHOLD` | 活体阈值，real_prob 超过才算真人 | 真人常被误判 Fake → 调低（假脸概率多在 0.03 附近，余量大）；想更严格拒绝翻拍 → 调高 |
| `REQUIRED_CONSECUTIVE` | 连续多少帧判真才确认 | 越大越稳、越抗攻击，但确认越慢 |
| `FRAME_HISTORY` | 活体概率平滑窗口 | 越大越平滑 |
| `COSINE_THRESHOLD` | 身份相似度阈值 | 把别人误认成已登记者 → 调高（如 0.45–0.5）；本人偶尔认不出 → 调低 |
| `FACE_SCORE_THRESHOLD` | YuNet 人脸检测置信度 | 漏检人脸 → 调低；误检背景 → 调高 |

> 同名参数也可在 `face_id.py` 顶部找到（`COSINE_THRESHOLD`、检测阈值默认值）。

---

## 6. 文件说明

### 本系统核心（活体 + 身份）

| 文件 | 说明 |
|------|------|
| `recognize.py` | **主程序**：摄像头实时「活体检测 + 身份识别」 |
| `face_id.py` | 身份识别模块（YuNet 检测 + SFace 特征 + 余弦匹配 + 身份库读写） |
| `enroll.py` | 登记身份，扫描素材文件夹生成 `identities.npz` |
| `test_recognize_video.py` | 用视频文件离线测试整套流程 |
| `identities.npz` | 已登记的身份库（当前含 `fengyizhuo`） |
| `models/*.onnx` | YuNet + SFace 预训练模型 |
| `face_liveness_weights.h5` | 活体检测 VGG16 模型权重 |

### 原项目（仅活体检测，可选）

| 文件 | 说明 |
|------|------|
| `model.py` | VGG16 活体模型结构定义 |
| `train.py` | 训练 / 微调活体模型（仅在想提升活体精度时用，与身份识别无关） |
| `test.py` | 旧版摄像头活体检测（含主动活体挑战：左右转头） |
| `test_video.py` | 旧版视频文件活体检测 |
| `utils.py` | 旧版工具函数 |
| `LCC_FASD/` | 活体训练/评测数据集 |

---

## 7. 实测验证

在本机数据集与视频上的等价验证结果：

- **身份区分**：fengyizhuo 本人照片相似度 **0.69–0.89**；其他人（LCC 数据集）≈ **−0.08**。区分度大、误识别风险低。
- **活体抑制翻拍**：LCC spoof（打印/翻拍）样本 real_prob 均值 **0.027**，几乎全判 Fake；真人样本均值 **0.6**。
- **整体流程**：摄像头管线在视频输入上跑通无报错；回放视频被正确判为 Fake（符合“播放视频=false”）。

> 注：摄像头实路径需在带摄像头/显示器的环境下运行 `recognize.py` 实测。
> 若真人偶尔被判 Fake，按第 5 节调低 `REAL_THRESHOLD` 即可。

---

## 8. 常见问题

**Q：识别 fengyizhuo 需要重新训练吗？**
不需要。身份识别基于预训练 SFace 的特征比对，`enroll.py` 只是“登记”不是“训练”。

**Q：为什么播放 fengyizhuo 本人的视频，结果是 Fake 而不标名字？**
因为这是“手机/屏幕回放”攻击，活体检测正确判为 Fake；身份识别只在判定真人时才进行，故不标名字。这正是预期行为。

**Q：提示找不到 `cv2.FaceDetectorYN`？**
你的 opencv-python 版本过低或不含该 API，请升级到 4.10+。

**Q：活体模型加载报权重不匹配？**
请确认用 TensorFlow 2.13.x（`.h5` 权重按层加载，跨小版本通常兼容；TF 2.15 不支持 Python 3.8）。
