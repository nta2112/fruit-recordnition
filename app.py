import streamlit as st
import numpy as np
from PIL import Image
import json
import os

# Try to load model from either tensorflow.keras or keras depending on environment
try:
    from tensorflow.keras.models import load_model
except Exception:
    try:
        from keras.models import load_model
    except Exception:
        load_model = None


@st.cache_resource
def load_artifacts(model_path="model.h5", labels_path="labels.json"):
    """Load Keras model and labels file. Return (model, labels) or (None, None) with error msg."""
    if load_model is None:
        return None, None, "Keras/tensorflow not available in the environment"

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path} - save your model as Keras .h5 in the project folder."

    if not os.path.exists(labels_path):
        return None, None, f"Labels file not found: {labels_path} - save class names to labels.json"

    try:
        model = load_model(model_path)
    except Exception as e:
        return None, None, f"Failed loading model: {e}"

    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception as e:
        return None, None, f"Failed loading labels.json: {e}"

    return model, labels, None


def preprocess_image(img: Image.Image, target_size=(100, 100)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    # model expects shape (1, h, w, 3)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, labels, img_arr, top_k=5):
    preds = model.predict(img_arr)
    preds = np.squeeze(preds)
    # get top k indices
    top_idx = preds.argsort()[-top_k:][::-1]
    return [(labels[int(i)], float(preds[int(i)])) for i in top_idx]


def main():
    st.title("Fruit Recognition Demo (Streamlit)")
    st.write("Upload an image of a fruit and the model will predict the class.")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path", value="model.h5")
    labels_path = st.sidebar.text_input("Labels path", value="labels.json")
    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=10, value=5)

    model, labels, err = load_artifacts(model_path, labels_path)
    if err:
        st.warning(err)
        st.info("From your notebook: use `model.save('model.h5')` and `json.dump(list(target_labels), open('labels.json','w'))` to create artifacts.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        img_arr = preprocess_image(image, target_size=(100, 100))

        if model is None:
            st.error("Model is not loaded — can't run prediction.")
        else:
            with st.spinner("Predicting..."):
                try:
                    results = predict(model, labels, img_arr, top_k=top_k)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            st.subheader("Predictions")
            for label, prob in results:
                st.write(f"{label}: {prob:.4f}")


if __name__ == "__main__":
    main()
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
class_names = ['apple', 'banana', 'orange', 'mango', 'pineapple', 'grape']

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
