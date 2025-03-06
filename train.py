# training/train.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_resnet_model

# Cấu hình chung
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# Đường dẫn đến thư mục dataset (đảm bảo đường dẫn đúng)
data_dir = os.path.join(os.path.pardir, "dataset")

# Sử dụng hàm tiền xử lý của ResNet50 (chuyển đổi ảnh sao cho phù hợp với ResNet)
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    validation_split=0.2  # 20% dữ liệu để validation
)

# Generator cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Generator cho tập validation
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Xây dựng và compile mô hình ResNet
model = build_resnet_model(input_shape=(128, 128, 3), num_classes=5)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Lưu mô hình vào thư mục gốc dự án
model.save(os.path.join(os.path.pardir, "animal_classifier.h5"))
print("Training complete. Model saved as animal_classifier.h5")
