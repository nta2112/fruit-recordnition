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
def load_artifacts(model_path="fruit_cnn_best.h5", labels_path="labels.json"):
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
    return map_predictions(preds, labels, top_k=top_k)


def map_predictions(preds, labels, top_k=5):
    """Map a prediction vector (1D or 2D squeezed) to (label, prob) tuples safely."""
    preds = np.squeeze(preds)
    # get top k indices
    top_idx = preds.argsort()[-top_k:][::-1]

    # determine number of classes from prediction vector
    try:
        n_classes = preds.shape[-1]
    except Exception:
        n_classes = len(preds)

    results = []
    for i in top_idx:
        idx = int(i)
        prob = float(preds[idx])
        # resolve label name safely
        label_name = None
        if isinstance(labels, (list, tuple)):
            if 0 <= idx < len(labels):
                label_name = labels[idx]
        elif isinstance(labels, dict):
            # try numeric key or string key
            label_name = labels.get(str(idx), labels.get(idx, None))

        if label_name is None:
            label_name = f"class_{idx}"

        results.append((label_name, prob))

    return results


def main():
    st.title("Fruit Recognition Demo (Streamlit)")
    st.write("Upload an image of a fruit and the model will predict the class.")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path", value="fruit_cnn_best.h5")
    labels_path = st.sidebar.text_input("Labels path", value="labels.json")
    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=10, value=5)

    model, labels, err = load_artifacts(model_path, labels_path)
    if err:
        st.warning(err)
        st.info("From your notebook: use `model.save('fruit_cnn_best.h5')` and `json.dump(list(target_labels), open('labels.json','w'))` to create artifacts.")

    # Environment / GPU info (sidebar) — helps explain CUDA/TensorFlow messages
    with st.sidebar.expander("Environment info", expanded=False):
        try:
            import tensorflow as _tf
            gpus = _tf.config.list_physical_devices('GPU')
            st.write(f"TensorFlow: {_tf.__version__}")
            if gpus:
                st.write(f"GPUs found: {len(gpus)}")
                for gpu in gpus:
                    st.write(f" - {gpu}")
            else:
                st.write("GPUs found: 0 — TensorFlow will use CPU. If you expected GPU, install CUDA/cuDNN and matching TF build.")
        except Exception as e:
            st.write("TensorFlow not available or failed to import:", e)

    debug = st.sidebar.checkbox("Show debug info (pred shapes)", value=False)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Streamlit sizing: prefer width='stretch' (new API). Fall back if not supported.
        try:
            st.image(image, caption="Uploaded image", width='stretch')
        except Exception:
            st.image(image, caption="Uploaded image", use_container_width=True)

        img_arr = preprocess_image(image, target_size=(100, 100))

        if model is None:
            st.error("Model is not loaded — can't run prediction.")
        else:
            with st.spinner("Predicting..."):
                try:
                    # run model once and optionally show debug info
                    preds = model.predict(img_arr)

                    if debug:
                        try:
                            st.write("preds shape:", np.shape(preds))
                            st.write("argmax:", int(np.argmax(preds)))
                            st.write("labels length:", len(labels) if labels is not None else "no labels")
                            # show probability distribution as bar chart when possible
                            try:
                                probs = preds[0] if getattr(preds, 'ndim', 1) > 1 else preds
                                st.bar_chart(probs)
                            except Exception:
                                pass
                        except Exception:
                            # ignore debug printing errors
                            pass

                    results = map_predictions(preds, labels, top_k=top_k)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            st.subheader("Predictions")
            # If labels length doesn't match model output, fall back to generic names
            for label, prob in results:
                try:
                    st.write(f"{label}: {prob:.4f}")
                except Exception:
                    # fallback: label might be invalid / index out of range
                    st.write(f"class (index unknown): {prob:.4f}")


if __name__ == "__main__":
    main()
