import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load mô hình đã được huấn luyện (đảm bảo file animal_classifier.h5 nằm ở thư mục gốc)
model_path = os.path.join(os.path.pardir, "animal_classifier.h5")
model = tf.keras.models.load_model(model_path)

# Mapping từ class index sang tên động vật bằng tiếng Việt
class_indices = {0: 'voi', 1: 'báo đốm', 2: 'sư tử', 3: 'gấu trúc', 4: 'hổ'}


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_relative_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Tạo thư mục uploads nếu chưa có
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            # Tạo đường dẫn ảnh tương đối
            img_relative_path = f"uploads/{filename}"

            # Tiền xử lý và dự đoán
            img = prepare_image(file_path)
            prediction = model.predict(img)[0]  # Lấy mảng xác suất dự đoán
            sorted_probs = np.sort(prediction)[::-1]  # Sắp xếp xác suất theo thứ tự giảm dần
            max_prob = sorted_probs[0]  # Xác suất cao nhất
            second_max_prob = sorted_probs[1]  # Xác suất cao thứ hai
            confidence = max_prob * 100  # chuyển sang phần trăm

            threshold = 0.8  # Ngưỡng tin cậy 80%
            delta = max_prob - second_max_prob  # Chênh lệch giữa 2 xác suất cao nhất

            if max_prob < threshold or delta < 0.2:  # Nếu mô hình không đủ chắc chắn
                result = f"Ảnh không thuộc bộ dữ liệu huấn luyện (độ chắc chắn: {confidence:.2f}%)"
            else:
                predicted_class = np.argmax(prediction)
                predicted_label = class_indices.get(predicted_class, 'Không xác định')
                result = f"{predicted_label} (độ chắc chắn: {confidence:.2f}%)"

    return render_template('index.html', result=result, img_relative_path=img_relative_path)


if __name__ == '__main__':
    app.run(debug=True)
