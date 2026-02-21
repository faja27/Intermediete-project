import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Konfigurasi Halaman & Judul
st.set_page_config(page_title="ASL Recognition Final", page_icon="🤟", layout="centered")

st.title("🤟 Deteksi Bahasa Isyarat (ASL)")
st.markdown("""
Aplikasi ini menggunakan model **MobileNetV2** yang telah di-*fine-tune* untuk mengenali 29 kategori bahasa isyarat ASL.
---
""")

# 2. Load Model Hasil Training Final
@st.cache_resource
def load_asl_model():
    # Pastikan file model .keras berada di folder yang sama dengan file .py ini
    model_path = '1. asl_mobilenet_final.keras' 
    return tf.keras.models.load_model(model_path)

try:
    model = load_asl_model()
    st.sidebar.success("✅ Model berhasil dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model. Pastikan file '{model_path}' tersedia.")
    st.stop()

# 3. Daftar Label (Urutan Alfabetis Keras)
# Ini adalah urutan standar yang dihasilkan oleh image_dataset_from_directory
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# 4. Fitur Upload Gambar
uploaded_file = st.file_uploader("Unggah foto tangan atau ambil gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Memproses Gambar
    image = Image.open(uploaded_file).convert('RGB')
    
    # Layout untuk menampilkan input dan hasil
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Gambar Input', use_container_width=True)

    with st.spinner('🔄 Model sedang menganalisis...'):
        # 5. Preprocessing (Sesuai file.py)
        # Resize ke 224x224
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized).astype('float32')
        img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch (1, 224, 224, 3)

        # Catatan: Kita tidak membagi 255 secara manual karena di dalam model 
        # sudah ada layer preprocess_input(inputs) dari MobileNetV2.

        # 6. Prediksi
        predictions = model.predict(img_array)
        probabilities = predictions[0] # Output sudah Softmax
        class_idx = np.argmax(probabilities)
        confidence = probabilities[class_idx] * 100

    with col2:
        st.markdown("### 📊 Hasil Analisis")
        st.metric(label="Prediksi Huruf", value=labels[class_idx])
        st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2f}%")
        
        if confidence < 70:
            st.warning("⚠️ Keyakinan rendah. Pastikan pencahayaan cukup dan tangan terlihat jelas.")
        else:
            st.success("✅ Prediksi Stabil")

    # 7. Visualisasi Probabilitas Top 5
    st.write("---")
    st.write("#### 🔍 5 Kemungkinan Terbesar:")
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    
    for i in top_5_indices:
        label_name = labels[i]
        prob_val = probabilities[i]
        st.write(f"**{label_name}**")
        st.progress(float(prob_val))

# Footer
st.markdown("---")
st.caption("Dibuat dengan Streamlit & TensorFlow MobileNetV2")