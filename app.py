import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Cấu hình trang ---
st.set_page_config(page_title="Nhận dạng hoa quả", page_icon="🍎", layout="centered")

# --- Tiêu đề ---
st.title("🍇 Ứng dụng nhận dạng hoa quả bằng CNN")
st.write("Tải lên ảnh hoa quả để mô hình dự đoán loại của nó.")

# --- Tải mô hình ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("fruit_cnn_best.h5")
    return model

model = load_model()

# --- Danh sách nhãn ---
# ⚠️ Bạn cần thay đổi danh sách này cho đúng với tập huấn luyện của bạn
class_names = ['apple_braeburn_1', 'apple_crimson_snow_1', 'apple_granny_smith_1', 'apple_granny_smith_1', 'apple_red_2', 'apple_red_yellow_1','cabbage_white_1','carrot_1','cucumber_1','cucumber_3','eggplant_violet_1','pear_1','pear_3','zucchini_1','zucchini_dark_1']

# --- Upload ảnh ---
uploaded_file = st.file_uploader("Tải ảnh hoa quả lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Ảnh đã tải lên', use_column_width=True)

    # Tiền xử lý ảnh
    img = img.resize((100, 100))  # kích thước theo input_shape của mô hình
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Dự đoán
    preds = model.predict(img_array)
    score = np.max(preds)
    label = class_names[np.argmax(preds)]

    # Hiển thị kết quả
    st.subheader(f"Kết quả dự đoán: **{label.upper()}** 🍏")
    st.write(f"Độ tin cậy: `{score:.2f}`")

    # Biểu đồ thanh xác suất
    st.bar_chart(preds[0])
