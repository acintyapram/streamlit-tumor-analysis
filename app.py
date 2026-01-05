import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit.components.v1 as components

# --- 1. SUPER CSS HIDER (Membasmi Kotak Merah & Error Visual) ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Auto-recovery script
components.html(
    """
    <script>
    window.addEventListener('unhandledrejection', (event) => {
        if (event.reason?.message?.includes('Failed to fetch')) { window.location.reload(); }
    });
    </script>
    """, height=0,
)

# --- 2. SIDEBAR PARAMETER ---
st.sidebar.header("⚙️ Kontrol Parameter")
metode = st.sidebar.selectbox("Metode Segmentasi:", ["Manual", "Otsu"])

t_val = 160
if metode == "Manual":
    t_val = st.sidebar.number_input("Threshold (T Segment):", 0, 255, 160, step=5)

operasi_morf = st.sidebar.selectbox("Operasi Morfologi:", ["None", "Opening", "Closing", "Erosi", "Dilasi"])
kernel_size = st.sidebar.number_input("Size Filter (Kernel):", 1, 15, 3, step=2)

# --- 3. TAMPILAN UTAMA ---
st.title("🧠 Dashboard Analisis Tumor")

tab1, tab2 = st.tabs(["🚀 Analisis Citra", "ℹ️ Informasi Metode Lengkap"])

with tab1:
    uploaded_file = st.file_uploader("Upload Foto Medis", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file).convert('L'))
        
        # Isolasi Area Dasar
        mask_init = cv2.inRange(img, 100, 255) 
        img_target_tumor = cv2.bitwise_and(img, img, mask=mask_init)

        # Logika Segmentasi
        if metode == 'Manual':
            _, binary_res = cv2.threshold(img_target_tumor, t_val, 255, cv2.THRESH_BINARY)
        else:
            pixels_tumor = img_target_tumor[img_target_tumor > 0]
            if len(pixels_tumor) > 0:
                val_otsu, _ = cv2.threshold(pixels_tumor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, binary_res = cv2.threshold(img_target_tumor, val_otsu, 255, cv2.THRESH_BINARY)
            else:
                binary_res = np.zeros_like(img_target_tumor)

        # Logika Morfologi
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if operasi_morf == 'Opening': final = cv2.morphologyEx(binary_res, cv2.MORPH_OPEN, kernel)
        elif operasi_morf == 'Closing': final = cv2.morphologyEx(binary_res, cv2.MORPH_CLOSE, kernel)
        elif operasi_morf == 'Erosi': final = cv2.erode(binary_res, kernel)
        elif operasi_morf == 'Dilasi': final = cv2.dilate(binary_res, kernel)
        else: final = binary_res

        # VISUALISASI GAMBAR
        st.subheader("🖼️ Hasil Analisis Citra")
        fig, ax = plt.subplots(1, 3, figsize=(22, 7))
        ax[0].imshow(img, cmap='gray'); ax[0].set_title("1. Original"); ax[0].axis('off')
        ax[1].imshow(img_target_tumor, cmap='inferno'); ax[1].set_title("2. Isolasi Area (Inferno)"); ax[1].axis('off')
        ax[2].imshow(final, cmap='gray'); ax[2].set_title(f"3. Hasil Biner ({metode})"); ax[2].axis('off')
        st.pyplot(fig)

        # --- SEKSI HISTOGRAM (KETERANGAN DI BAWAH) ---
        st.markdown("---")
        st.subheader("📊 Histogram Citra")
        
        fig_h, ax_h = plt.subplots(1, 2, figsize=(15, 5))
        ax_h[0].hist(img.ravel(), 256, [0, 256], color='gray', alpha=0.7); ax_h[0].set_title("Histogram Seluruh Citra")
        
        pixels_iso = img_target_tumor[img_target_tumor > 0].ravel()
        if len(pixels_iso) > 0:
            ax_h[1].hist(pixels_iso, 256, [1, 256], color='orange', alpha=0.8); ax_h[1].set_title("Histogram Distribusi Massa Tumor")
        st.pyplot(fig_h)

        # KETERANGAN DI BAWAH HISTOGRAM
        st.markdown("""
        **💡 Interpretasi Hasil Histogram:**
        1. **Sumbu X (Intensitas Piksel)**: Mewakili tingkat kecerahan dari 0 (Hitam pekat) hingga 255 (Putih terang).
        2. **Sumbu Y (Jumlah Piksel)**: Mewakili seberapa banyak piksel yang memiliki nilai kecerahan tersebut.
        3. **Analisis Kurva**: 
           * **Histogram Asli** menunjukkan sebaran global dari background dan jaringan otak. 
           * **Histogram Tumor (Jingga)** berfokus pada area yang diisolasi. Jika grafik menumpuk di area angka tinggi (kanan), hal ini secara ilmiah membuktikan bahwa massa tumor memiliki karakteristik **hyperintens** (lebih terang) dibandingkan jaringan sekitarnya pada citra MRI ini.
        """)
        
        st.success(f"Estimasi Luas Massa Tumor: {np.sum(final == 255)} Piksel")

with tab2:
    st.header("📘 Informasi Metode Lengkap")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🔹 Metode Segmentasi")
        st.write("**Manual Thresholding**: Pemisahan objek menggunakan nilai ambang batas statis (User-defined).")
        st.write("**Otsu's Method**: Algoritma otomatis yang membagi piksel menjadi dua kelas (foreground & background) dengan mencari nilai threshold yang meminimalkan varians di dalam kelas.")
    with col_b:
        st.subheader("🔸 Operasi Morfologi")
        st.write("**Erosi/Opening**: Digunakan untuk membuang noise kecil (bintik putih) agar fokus hanya pada massa utama.")
        st.write("**Dilasi/Closing**: Digunakan untuk menutup lubang kecil (hole filling) agar area tumor terdeteksi secara utuh tanpa celah.")
