# model.py - 使用VGG16预训练模型

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import backend as K


class VGG16Liveness:
    @staticmethod
    def build(width, height, depth, classes):
        # VGG16输入要求224x224，但也可以用其他尺寸
        # 使用预训练权重，加载ImageNet权重
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(height, width, depth)
        )

        # 冻结基础模型层（可选微调）
        for layer in base_model.layers:
            layer.trainable = False

        # 构建完整模型
        model = Sequential()
        model.add(base_model)

        # 添加自定义分类头
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(classes, activation='softmax'))

        return model

    @staticmethod
    def build_finetune(width, height, depth, classes):
        """微调版本 - 解冻部分VGG16层"""
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(height, width, depth)
        )

        # 解冻最后几层
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        for layer in base_model.layers[-4:]:
            layer.trainable = True

        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model