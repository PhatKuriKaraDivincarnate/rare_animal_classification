# training/model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


def build_resnet_model(input_shape=(128, 128, 3), num_classes=5):
    # Sử dụng ResNet50 với include_top=False để loại bỏ các lớp phân loại mặc định
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # Freeze base_model trong giai đoạn đầu (fine-tuning có thể thực hiện sau)
    base_model.trainable = False

    # Thêm các lớp mới phù hợp với bài toán của bạn
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model
