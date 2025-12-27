import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os

# ===============================
# ECA Layer (WAJIB untuk load model)
# ===============================
class ECALayer(tf.keras.layers.Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channel = input_shape[-1]
        t = int(abs((tf.math.log(tf.cast(channel, tf.float32)) / tf.math.log(2.0)) + self.b) / self.gamma)
        k = t if t % 2 else t + 1

        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=k,
            padding="same",
            use_bias=False
        )

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv(x)
        x = tf.squeeze(x, axis=-1)
        x = tf.nn.sigmoid(x)
        return inputs * tf.expand_dims(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "b": self.b})
        return config


# ===============================
# Konfigurasi
# ===============================
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

st.set_page_config(
    page_title="Deteksi Penyakit Mata",
    layout="centered"
)

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model_deteksi_mata_v2.h5",
        custom_objects={"ECALayer": ECALayer},
        compile=False
    )
    return model

model = load_model()

# VALIDASI MODEL
if model is None:
    st.error("❌ Model gagal dimuat")
    st.stop()

# Ambil input size dari model (AMAN)
_, H, W, C = model.input_shape
IMG_SIZE = (W, H)

# ===============================
# UI Upload
# ===============================
st.markdown("## 🧿 Unggah Gambar Citra Fundus Mata")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Unggah Gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    detect_button = st.button("🔍 Deteksi Sekarang", use_container_width=True)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

# ===============================
# Proses Deteksi
# ===============================
if uploaded_file and detect_button:
    with st.spinner("Memproses gambar..."):
        image_resized = image.resize(IMG_SIZE)

        img_array = np.array(image_resized, dtype=np.float32)

        # Normalisasi sesuai MobileNetV2 (AMAN)
        img_array = img_array / 255.0

        img_batch = np.expand_dims(img_array, a_
