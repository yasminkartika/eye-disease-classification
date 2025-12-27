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
        super(ECALayer, self).__init__(**kwargs)
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
            padding='same',
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
        config.update({
            "gamma": self.gamma,
            "b": self.b
        })
        return config


# ===============================
# Konfigurasi
# ===============================
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

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
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Gambar yang Diperiksa", width=300)

    with col2:
        st.markdown("### 📊 Hasil Deteksi")
        st.markdown(f"""
        **<span style='font-size:26px'>{predicted_class}</span>**  
        <span style='font-size:32px; color:red; font-weight:bold'>
        {confidence:.2f}%
        </span>
        """, unsafe_allow_html=True)

    # ===============================
    # Simpan Riwayat
    # ===============================
    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("💾 Simpan Hasil"):
            if not os.path.exists("riwayat_deteksi"):
                os.makedirs("riwayat_deteksi")

            file_path = os.path.join(
                "riwayat_deteksi", "riwayat_deteksi.txt"
            )

            with open(file_path, "a") as f:
                f.write(
                    f"{datetime.datetime.now()} | "
                    f"{predicted_class} | "
                    f"{confidence:.2f}%\n"
                )

            st.success("Hasil berhasil disimpan!")

    with colB:
        if st.button("🔁 Deteksi Ulang"):
            st.experimental_rerun()
