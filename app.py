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
# KONFIGURASI
# ===============================
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

st.set_page_config(
    page_title="EyeCare - Deteksi Penyakit Mata",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}

.header-box {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    padding: 30px;
    border-radius: 15px;
    color: white;
}

.result-card {
    background: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

.upload-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
}

.badge {
    padding: 8px 18px;
    border-radius: 20px;
    color: white;
    font-weight: bold;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL (TIDAK DIUBAH)
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

if model is None:
    st.error("❌ Model gagal dimuat")
    st.stop()

# AMBIL INPUT SIZE DARI MODEL (TETAP)
_, H, W, C = model.input_shape
IMG_SIZE = (W, H)

# ===============================
# HEADER
# ===============================
st.markdown("""
<div class="header-box">
    <h1>👁️ EyeCare AI Diagnostic System</h1>
    <p>Sistem Deteksi Penyakit Mata Berbasis Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ===============================
# MAIN LAYOUT
# ===============================
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Citra Fundus")

    uploaded_file = st.file_uploader(
        "Unggah gambar fundus",
        type=["jpg","jpeg","png"]
    )

    detect_button = st.button("🔍 Analisis Sekarang", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview Citra", use_container_width=True)

# ===============================
# DETEKSI (TIDAK DIUBAH LOGIKANYA)
# ===============================
if uploaded_file and detect_button:
    with st.spinner("Sedang menganalisis citra..."):
        image_resized = image.resize(IMG_SIZE)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch, verbose=0)

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("📊 Hasil Analisis")

    color_dict = {
        "Normal": "#2ecc71",
        "Cataract": "#f39c12",
        "Diabetic Retinopathy": "#e74c3c",
        "Glaucoma": "#8e44ad"
    }

    badge_color = color_dict.get(predicted_class, "#34495e")

    st.markdown(
        f"""
        <div class="badge" style="background:{badge_color};">
        {predicted_class}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"### Tingkat Keyakinan: {confidence:.2f}%")
    st.progress(int(confidence))

    st.markdown("---")

    disease_info = {
        "Normal": "Tidak ditemukan indikasi kelainan signifikan pada retina.",
        "Cataract": "Katarak adalah kondisi kekeruhan pada lensa mata yang menyebabkan penglihatan kabur.",
        "Diabetic Retinopathy": "Komplikasi diabetes yang merusak pembuluh darah retina.",
        "Glaucoma": "Kerusakan saraf optik akibat peningkatan tekanan bola mata."
    }

    st.markdown("### 📝 Informasi Medis")
    st.write(disease_info.get(predicted_class,""))

    st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # SIMPAN
    # ===============================
    colA, colB = st.columns(2)

    with colA:
        if st.button("💾 Simpan Hasil Rekam Medis"):
            os.makedirs("riwayat_deteksi", exist_ok=True)
            with open("riwayat_deteksi/riwayat_deteksi.txt", "a") as f:
                f.write(
                    f"{datetime.datetime.now()} | "
                    f"{predicted_class} | "
                    f"{confidence:.2f}%\n"
                )
            st.success("Data berhasil disimpan")

    with colB:
        if st.button("🔁 Analisis Ulang"):
            st.rerun()
